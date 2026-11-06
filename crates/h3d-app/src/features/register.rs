//! Fitur Penyejajaran (8.11, 8.4.5) untuk tampilan 2D mode "Sejajarkan":
//! viewer pembanding irisan `i` terhadap tetangganya ke arah acuan
//! (campur / papan catur / selisih; [`viewer`]), pilihan alat di bilah atas
//! kanvas ([`tool_options`]: mode banding, terapkan landmark atau
//! geser/putar, tandai rusak), dan isi laci "Penyejajaran" ([`align_drawer`]:
//! "Sejajarkan semua" yang bisa dilanjutkan dan daftar QC dengan irisan
//! bertanda di atas). Pemilihan alat (geser pandangan/landmark/geser/putar)
//! ada di rel alat kiri.

use egui::Ui;
use h3d_core::{SlideId, SlideStatus, SlideTransform};
use h3d_engine::{QcRow, Session};
use h3d_i18n::t;
use h3d_render::{CompareMode, CompareParams, H3dPaintCallback, Scene, Scene2D};

use crate::geom::{
    manual_adjusted_transform, navigate, neighbor_toward_reference, LandmarkCollector, LandmarkSide,
};
use crate::gpu::Gpu;
use crate::state::{AppState, Drawer, SliceMode, View};
use crate::ui_util::{icon_selectable, icon_text_button, icon_text_button_enabled};
use crate::viewer::{
    hide_background_checkbox, paint_background, slide_layer, tissue_mask_if_enabled, Viewer2D,
};

/// Mode tampilan pembanding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CompareKind {
    /// Campur tembus pandang.
    #[default]
    Blend,
    /// Papan catur.
    Checkerboard,
    /// Selisih warna.
    Difference,
}

/// Alat aktif penyejajaran.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RegTool {
    /// Geser/zoom pandangan.
    #[default]
    View,
    /// Klik pasangan landmark.
    Landmark,
    /// Geser irisan.
    Translate,
    /// Putar irisan.
    Rotate,
}

impl RegTool {
    /// Semua alat, urutan di rel alat.
    pub const ALL: [RegTool; 4] = [
        RegTool::View,
        RegTool::Landmark,
        RegTool::Translate,
        RegTool::Rotate,
    ];

    /// Ikon alat.
    pub fn icon(self) -> h3d_ui::icons::MaterialIcon {
        use h3d_ui::icons;
        match self {
            RegTool::View => icons::PAN,
            RegTool::Landmark => icons::LANDMARK,
            RegTool::Translate => icons::MOVE,
            RegTool::Rotate => icons::ROTATE,
        }
    }

    /// Kunci terjemahan nama alat.
    pub fn i18n_key(self) -> &'static str {
        match self {
            RegTool::View => "tool-pan",
            RegTool::Landmark => "register-landmarks",
            RegTool::Translate => "register-translate",
            RegTool::Rotate => "register-rotate",
        }
    }
}

/// Tombol jungkit balik kiri-kanan / atas-bawah untuk koreksi manual
/// (irisan yang terpasang terbalik di kaca). Dipakai mode 2D dan kanvas 3D.
pub fn flip_buttons(ui: &mut Ui, flip: &mut [bool; 2]) {
    if crate::ui_util::icon_selectable(ui, flip[0], h3d_ui::icons::FLIP_H, &t!("register-flip-h"))
        .clicked()
    {
        flip[0] = !flip[0];
    }
    if crate::ui_util::icon_selectable(ui, flip[1], h3d_ui::icons::FLIP_V, &t!("register-flip-v"))
        .clicked()
    {
        flip[1] = !flip[1];
    }
}

/// Keadaan mode Registrasi.
pub struct RegisterState {
    /// Viewer pembanding.
    pub viewer: Viewer2D,
    /// Mode banding.
    pub compare: CompareKind,
    /// Opasitas lapisan bergerak pada mode campur.
    pub alpha: f32,
    /// Ukuran kotak papan catur (point).
    pub checker: f32,
    /// Alat aktif.
    pub tool: RegTool,
    /// Pasangan landmark yang dikumpulkan.
    pub landmarks: LandmarkCollector,
    /// Geseran manual yang belum diterapkan (µm, W).
    pub shift: [f64; 2],
    /// Putaran manual yang belum diterapkan (radian).
    pub angle: f64,
    /// Cermin manual yang belum diterapkan: `[kiri-kanan, atas-bawah]`.
    pub flip: [bool; 2],
    /// Irisan yang koreksinya sedang disusun.
    pub editing: Option<SlideId>,
    /// Tabel QC (di-cache).
    pub qc: Vec<QcRow>,
    /// Tabel QC perlu dimuat ulang.
    pub qc_dirty: bool,
}

impl Default for RegisterState {
    fn default() -> Self {
        RegisterState {
            viewer: Viewer2D::default(),
            compare: CompareKind::Blend,
            alpha: 0.5,
            checker: 48.0,
            tool: RegTool::View,
            landmarks: LandmarkCollector::default(),
            shift: [0.0, 0.0],
            angle: 0.0,
            flip: [false, false],
            editing: None,
            qc: Vec::new(),
            qc_dirty: true,
        }
    }
}

impl RegisterState {
    /// Buang koreksi yang belum diterapkan.
    pub fn reset_edit(&mut self) {
        self.landmarks.clear();
        self.shift = [0.0, 0.0];
        self.angle = 0.0;
        self.flip = [false, false];
    }

    /// Ada geser/putar/balik manual yang belum diterapkan.
    pub fn has_manual(&self) -> bool {
        self.shift != [0.0, 0.0] || self.angle != 0.0 || self.flip != [false, false]
    }

    /// Irisan gagal QC menurut tabel yang di-cache, urutan tumpukan.
    pub fn flagged_ids(&self) -> Vec<SlideId> {
        self.qc
            .iter()
            .filter(|r| r.flagged)
            .map(|r| r.slide)
            .collect()
    }
}

/// Buka `slide` di mode Sejajarkan 2D dengan alat landmark dan laci
/// Penyejajaran terbuka: titik masuk tombol "Perbaiki" dan navigasi
/// antar irisan gagal QC.
pub fn start_fix(app: &mut AppState, slide: SlideId) {
    app.open_slice_2d(slide);
    app.slice_mode = SliceMode::Align;
    app.register.tool = RegTool::Landmark;
    app.register.compare = CompareKind::Blend;
    app.drawer = Some(Drawer::Align);
}

/// Irisan gagal QC berikutnya setelah `current` (melingkar), tidak termasuk
/// `current` sendiri; `None` bila tidak ada yang lain.
pub fn next_flagged(
    flagged: &[SlideId],
    current: Option<SlideId>,
    forward: bool,
) -> Option<SlideId> {
    let others: Vec<SlideId> = flagged
        .iter()
        .copied()
        .filter(|f| Some(*f) != current)
        .collect();
    if others.is_empty() {
        return None;
    }
    let pos = current.and_then(|c| flagged.iter().position(|f| *f == c));
    let n = flagged.len();
    let start = match (pos, forward) {
        (Some(p), true) => (p + 1) % n,
        (Some(p), false) => (p + n - 1) % n,
        (None, _) => 0,
    };
    (0..n)
        .map(|i| {
            if forward {
                (start + i) % n
            } else {
                (start + n - i) % n
            }
        })
        .map(|i| flagged[i])
        .find(|f| Some(*f) != current)
}

/// Alasan gagal QC satu baris, satu kalimat per ambang yang dilanggar.
/// Ambang yang dipakai adalah bawaan `QcThresholds` (sama dengan
/// "Sejajarkan semua" dari aplikasi).
fn failure_reasons(row: &QcRow) -> Vec<String> {
    let thr = h3d_engine::QcThresholds::default();
    let v = thr.violations(
        row.ncc.unwrap_or(f32::NAN),
        row.mask_dice.unwrap_or(f32::NAN),
        row.area_ratio.unwrap_or(f32::NAN),
    );
    let mut out = Vec::new();
    if v.ncc {
        out.push(t!(
            "register-fix-ncc",
            v = fmt_opt(row.ncc),
            min = format!("{:.2}", thr.ncc_min)
        ));
    }
    if v.dice {
        out.push(t!(
            "register-fix-dice",
            v = fmt_opt(row.mask_dice),
            min = format!("{:.2}", thr.dice_min)
        ));
    }
    if v.area_ratio {
        out.push(t!(
            "register-fix-area",
            v = fmt_opt(row.area_ratio),
            lo = format!("{:.2}", thr.area_ratio.0),
            hi = format!("{:.2}", thr.area_ratio.1)
        ));
    }
    out
}

/// Kartu panduan perbaikan QC di atas laci Penyejajaran. Tanpa irisan gagal:
/// tidak digambar. Di luar mode Sejajarkan: spanduk + tombol "Perbaiki".
/// Di mode Sejajarkan pada irisan gagal: langkah-langkah, alasan gagal,
/// navigasi antar irisan gagal, dan jalan pintas Terima / Tandai rusak.
fn fix_card(app: &mut AppState, sess: &mut Session, ui: &mut Ui) {
    let flagged = app.register.flagged_ids();
    if flagged.is_empty() {
        return;
    }
    let n = flagged.len();
    let pos = app
        .current
        .and_then(|c| flagged.iter().position(|f| *f == c));
    let in_fix = app.view == View::Slice2D && app.slice_mode == SliceMode::Align && pos.is_some();

    let mut go: Option<SlideId> = None;
    egui::Frame::new()
        .fill(h3d_ui::theme::WARNING.gamma_multiply(0.18))
        .stroke(egui::Stroke::new(1.0, h3d_ui::theme::WARNING))
        .corner_radius(6)
        .inner_margin(egui::Margin::symmetric(8, 6))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    h3d_ui::icons::WARNING
                        .rich_text()
                        .color(h3d_ui::theme::WARNING),
                );
                if in_fix {
                    let k = pos.map_or(0, |p| p + 1);
                    crate::ui_util::strong(
                        ui,
                        t!(
                            "register-fix-progress",
                            k = k.to_string(),
                            n = n.to_string()
                        ),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .add_enabled(n > 1, egui::Button::new(h3d_ui::icons::NEXT.rich_text()))
                            .on_hover_text(t!("register-fix-next"))
                            .clicked()
                        {
                            go = next_flagged(&flagged, app.current, true);
                        }
                        if ui
                            .add_enabled(n > 1, egui::Button::new(h3d_ui::icons::PREV.rich_text()))
                            .on_hover_text(t!("register-fix-prev"))
                            .clicked()
                        {
                            go = next_flagged(&flagged, app.current, false);
                        }
                    });
                } else {
                    ui.label(t!("register-flagged-count", n = n.to_string()));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if icon_text_button(ui, h3d_ui::icons::TUNE, &t!("register-fix"))
                            .on_hover_text(t!("register-fix-hint"))
                            .clicked()
                        {
                            go = Some(pos.map_or(flagged[0], |p| flagged[p]));
                        }
                    });
                }
            });
            if !in_fix {
                return;
            }
            let Some(slide) = app.current else {
                return;
            };

            // Mengapa gagal.
            if let Some(row) = app.register.qc.iter().find(|r| r.slide == slide) {
                let reasons = failure_reasons(row);
                if !reasons.is_empty() {
                    ui.add_space(4.0);
                    ui.weak(t!("register-fix-why"));
                    for r in reasons {
                        ui.label(r);
                    }
                }
            }

            // Langkah.
            ui.add_space(6.0);
            ui.label(t!("register-fix-step1"));
            ui.add_space(2.0);
            ui.label(t!("register-fix-step2"));
            ui.horizontal_wrapped(|ui| {
                for tool in [RegTool::Landmark, RegTool::Translate, RegTool::Rotate] {
                    if icon_selectable(
                        ui,
                        app.register.tool == tool,
                        tool.icon(),
                        &t!(tool.i18n_key()),
                    )
                    .clicked()
                    {
                        set_tool(app, tool);
                    }
                }
            });
            let tool_hint = match app.register.tool {
                RegTool::Translate => Some(t!("register-fix-tool-translate-hint")),
                RegTool::Rotate => Some(t!("register-fix-tool-rotate-hint")),
                RegTool::Landmark => Some(t!("register-fix-tool-landmark-hint")),
                RegTool::View => None,
            };
            if let Some(h) = tool_hint {
                ui.small(h);
            }
            ui.add_space(2.0);
            ui.label(t!("register-fix-step3"));

            // Jalan pintas tanpa koreksi.
            ui.add_space(6.0);
            ui.weak(t!("register-fix-or"));
            ui.horizontal_wrapped(|ui| {
                if icon_text_button(ui, h3d_ui::icons::CHECK, &t!("register-accept"))
                    .on_hover_text(t!("register-accept-hint"))
                    .clicked()
                {
                    match sess.accept_registration(slide) {
                        Ok(()) => {
                            app.toasts.info(t!("register-accepted"));
                            app.register.qc_dirty = true;
                            go = next_flagged(&flagged, app.current, true);
                        }
                        Err(e) => app.toasts.engine_error(&e),
                    }
                }
                if icon_text_button(ui, h3d_ui::icons::DAMAGED, &t!("register-mark-damaged"))
                    .on_hover_text(t!("register-mark-damaged-hint"))
                    .clicked()
                {
                    match sess.set_status(slide, SlideStatus::Damaged) {
                        Ok(()) => {
                            app.register.qc_dirty = true;
                            go = next_flagged(&flagged, app.current, true);
                        }
                        Err(e) => app.toasts.engine_error(&e),
                    }
                }
            });
        });
    if let Some(id) = go {
        start_fix(app, id);
    }
    ui.add_space(4.0);
}

/// Urutkan baris QC: irisan gagal QC dulu, lalu yang belum terdaftar, lalu
/// sisanya; di dalam kelompok menurut urutan tumpukan (8.4.5).
pub fn sort_qc(rows: &mut [QcRow]) {
    let rank = |r: &QcRow| -> u8 {
        if r.flagged {
            0
        } else if !r.registered && r.status == SlideStatus::Ok {
            1
        } else {
            2
        }
    };
    rows.sort_by_key(|r| (rank(r), r.order));
}

/// Tetangga irisan ke arah acuan menurut keadaan proyek.
pub fn neighbor_of(sess: &Session, slide: SlideId) -> Option<SlideId> {
    let reference = sess.project().effective_reference()?;
    let usable: Vec<SlideId> = sess
        .project()
        .usable_slides_ordered()
        .iter()
        .map(|s| s.id)
        .collect();
    neighbor_toward_reference(&usable, reference, slide)
}

/// Segarkan tabel QC bila kotor dan buang koreksi yang belum diterapkan bila
/// irisan aktif berganti. Dipanggil sebelum viewer, pilihan alat, dan laci.
pub fn sync(app: &mut AppState, sess: &Session) {
    app.fix_current(sess);
    if app.register.qc_dirty {
        let mut rows = sess.qc_table();
        sort_qc(&mut rows);
        app.register.qc = rows;
        app.register.qc_dirty = false;
    }
    if app.register.editing != app.current {
        app.register.reset_edit();
        app.register.editing = app.current;
    }
}

/// Setel alat aktif (dari rel alat).
pub fn set_tool(app: &mut AppState, tool: RegTool) {
    app.register.tool = tool;
}

fn fmt_opt(v: Option<f32>) -> String {
    v.map(|x| format!("{x:.2}")).unwrap_or_else(|| "-".into())
}

/// Isi laci "Penyejajaran": Sejajarkan semua, ringkasan, dan daftar QC.
pub fn align_drawer(app: &mut AppState, sess: &mut Session, ui: &mut Ui) {
    sync(app, sess);
    fix_card(app, sess, ui);
    let needs = sess.needs_registration().len();
    ui.horizontal(|ui| {
        if icon_text_button_enabled(
            ui,
            app.job.is_none(),
            h3d_ui::icons::PLAY,
            &t!("register-all"),
        )
        .on_hover_text(t!("stack-align-hint"))
        .clicked()
        {
            app.start_register_all();
        }
        if needs > 0 {
            ui.weak(t!("register-pending-count", n = needs.to_string()));
        }
    });
    ui.small(t!("register-resumable-hint"));
    ui.separator();
    h3d_ui::widgets::section_title(ui, &t!("register-qc-list"));
    let mut select = None;
    egui::ScrollArea::both()
        .id_salt("qc-scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            egui::Grid::new("qc-grid")
                .num_columns(5)
                .striped(true)
                .show(ui, |ui| {
                    crate::ui_util::strong(ui, t!("slides-file-name"));
                    crate::ui_util::strong(ui, t!("register-ncc"));
                    crate::ui_util::strong(ui, t!("register-dice"));
                    crate::ui_util::strong(ui, t!("register-area-ratio"));
                    crate::ui_util::strong(ui, "");
                    ui.end_row();
                    for row in &app.register.qc {
                        let name = format!("{}. {}", row.order + 1, row.file);
                        let selected = app.current == Some(row.slide);
                        let text = if row.flagged {
                            egui::RichText::new(name).color(h3d_ui::theme::WARNING)
                        } else {
                            egui::RichText::new(name)
                        };
                        if ui.selectable_label(selected, text).clicked() {
                            select = Some(row.slide);
                        }
                        ui.label(fmt_opt(row.ncc));
                        ui.label(fmt_opt(row.mask_dice));
                        ui.label(fmt_opt(row.area_ratio));
                        ui.horizontal(|ui| {
                            if row.is_reference {
                                ui.label(h3d_ui::icons::REFERENCE.rich_text())
                                    .on_hover_text(t!("slides-is-reference"));
                            }
                            if row.flagged {
                                ui.label(
                                    h3d_ui::icons::FLAG
                                        .rich_text()
                                        .color(h3d_ui::theme::WARNING),
                                )
                                .on_hover_text(t!("register-flagged"));
                            }
                            if row.manual {
                                ui.label(h3d_ui::icons::TUNE.rich_text())
                                    .on_hover_text(t!("register-manual"));
                            }
                            match row.status {
                                SlideStatus::Ok => {
                                    if !row.registered {
                                        ui.weak(t!("register-not-registered"));
                                    }
                                }
                                SlideStatus::Damaged => {
                                    ui.label(
                                        h3d_ui::icons::DAMAGED
                                            .rich_text()
                                            .color(h3d_ui::theme::ERROR),
                                    )
                                    .on_hover_text(t!("slides-status-damaged"));
                                }
                                SlideStatus::Excluded => {
                                    ui.label(h3d_ui::icons::EXCLUDE.rich_text())
                                        .on_hover_text(t!("slides-status-excluded"));
                                }
                            }
                        });
                        ui.end_row();
                    }
                });
        });
    if let Some(id) = select {
        app.current = Some(id);
    }
}

/// Pilihan alat penyejajaran di bilah atas kanvas 2D: mode banding dan
/// parameternya, lalu aksi alat aktif (terapkan landmark / geser-putar) dan
/// tandai rusak.
pub fn tool_options(app: &mut AppState, sess: &mut Session, ui: &mut Ui) {
    sync(app, sess);
    ui.horizontal_wrapped(|ui| {
        ui.label(t!("register-compare-mode"));
        let st = &mut app.register;
        for (kind, key) in [
            (CompareKind::Blend, "register-compare-blend"),
            (CompareKind::Checkerboard, "register-compare-checkerboard"),
            (CompareKind::Difference, "register-compare-difference"),
        ] {
            if ui.selectable_label(st.compare == kind, t!(key)).clicked() {
                st.compare = kind;
            }
        }
        match st.compare {
            CompareKind::Blend => {
                ui.add(
                    egui::Slider::new(&mut st.alpha, 0.0..=1.0).text(t!("register-blend-alpha")),
                );
            }
            CompareKind::Checkerboard => {
                ui.add(
                    egui::Slider::new(&mut st.checker, 8.0..=256.0)
                        .text(t!("register-checker-size")),
                );
            }
            CompareKind::Difference => {}
        }
        ui.separator();
        hide_background_checkbox(app, ui);
    });
    ui.horizontal_wrapped(|ui| {
        let st = &mut app.register;
        let Some(slide) = app.current else {
            return;
        };
        let neighbor = neighbor_of(sess, slide);
        match st.tool {
            RegTool::Landmark => {
                let side = match st.landmarks.next_side() {
                    LandmarkSide::Fixed => t!("register-landmark-click-fixed"),
                    LandmarkSide::Moving => t!("register-landmark-click-moving"),
                };
                ui.label(side);
                ui.label(t!(
                    "register-landmark-count",
                    n = st.landmarks.pairs.len().to_string()
                ));
                if icon_text_button_enabled(
                    ui,
                    st.landmarks.can_apply() && neighbor.is_some(),
                    h3d_ui::icons::CHECK,
                    &t!("register-landmark-apply"),
                )
                .clicked()
                {
                    match sess.apply_landmarks(slide, &st.landmarks.pairs) {
                        Ok(()) => {
                            st.reset_edit();
                            st.qc_dirty = true;
                            app.toasts.info(t!("register-correction-applied"));
                        }
                        Err(e) => app.toasts.engine_error(&e),
                    }
                }
                if ui.button(t!("action-undo")).clicked() {
                    st.landmarks.pop();
                }
                if ui.button(t!("register-landmark-clear")).clicked() {
                    st.landmarks.clear();
                }
            }
            RegTool::Translate | RegTool::Rotate => {
                ui.label(format!(
                    "{}: {:.0}, {:.0} {}   {}: {:.2} deg",
                    t!("register-translate"),
                    st.shift[0],
                    st.shift[1],
                    t!("unit-um"),
                    t!("register-rotate"),
                    st.angle.to_degrees()
                ));
                flip_buttons(ui, &mut st.flip);
                if icon_text_button_enabled(
                    ui,
                    st.has_manual() && neighbor.is_some(),
                    h3d_ui::icons::CHECK,
                    &t!("action-apply"),
                )
                .clicked()
                {
                    if let Ok(entry) = sess.slide(slide) {
                        let t_old = entry.transform.affine();
                        let size = entry.size_um();
                        let pivot = t_old.apply([size[0] / 2.0, size[1] / 2.0]);
                        // `t_new` memetakan U irisan → W (arah `T_i`, 5.3).
                        let t_new =
                            manual_adjusted_transform(&t_old, st.shift, st.angle, st.flip, pivot);
                        match sess.apply_manual_transform(slide, t_new) {
                            Ok(()) => {
                                st.reset_edit();
                                st.qc_dirty = true;
                                app.toasts.info(t!("register-correction-applied"));
                            }
                            Err(e) => app.toasts.engine_error(&e),
                        }
                    }
                }
                if ui.button(t!("register-reset-transform")).clicked() {
                    st.shift = [0.0, 0.0];
                    st.angle = 0.0;
                    st.flip = [false, false];
                }
            }
            RegTool::View => {
                if neighbor.is_none() {
                    ui.weak(t!("register-no-neighbor"));
                }
            }
        }
        ui.separator();
        let status = sess.slide(slide).map(|s| s.status).ok();
        let (label, target) = if status == Some(SlideStatus::Damaged) {
            (t!("slides-unmark-damaged"), SlideStatus::Ok)
        } else {
            (t!("register-mark-damaged"), SlideStatus::Damaged)
        };
        if icon_text_button(ui, h3d_ui::icons::DAMAGED, &label).clicked() {
            if let Err(e) = sess.set_status(slide, target) {
                app.toasts.engine_error(&e);
            }
            app.register.qc_dirty = true;
        }
    });
    if app.register.tool == RegTool::Landmark {
        ui.small(t!("register-landmark-hint"));
    }
}

/// Viewer pembanding: irisan aktif (bergerak) di atas tetangganya (tetap)
/// dengan mode banding, landmark, dan pegangan geser/putar, mengisi sisa `ui`.
pub fn viewer(app: &mut AppState, sess: &mut Session, ui: &mut Ui, gpu: Option<&Gpu>) {
    sync(app, sess);
    let Some(slide) = app.current else {
        ui.centered_and_justified(|ui| ui.weak(t!("slides-empty")));
        return;
    };
    let Ok(entry) = sess.slide(slide).cloned() else {
        return;
    };
    let neighbor = neighbor_of(sess, slide);
    let t_old = entry.transform.affine();
    let size = entry.size_um();
    let pivot = t_old.apply([size[0] / 2.0, size[1] / 2.0]);
    let (bmin, bmax) = t_old.transform_rect(size[0], size[1]);

    let primary_pans =
        app.register.tool == RegTool::View || !app.touch.may_draw(app.settings.finger_draw);
    let frame = app
        .register
        .viewer
        .begin(ui, u64::from(slide.0), Some((bmin, bmax)), primary_pans);
    if let Some(dir) = frame.nav {
        let order = AppState::ordered_ids(sess);
        app.current = navigate(&order, app.current, dir);
    }
    let rect = frame.rect;
    let camera = app.register.viewer.camera;
    paint_background(ui, rect, gpu.is_some());

    // Masukan alat (setelah kamera diperbarui).
    let st = &mut app.register;
    let t_fixed = neighbor
        .and_then(|n| sess.slide(n).ok())
        .map(|n| n.transform.affine());
    let t_moving = manual_adjusted_transform(&t_old, st.shift, st.angle, st.flip, pivot);
    if !primary_pans && !frame.multi_touch {
        match st.tool {
            RegTool::Landmark => {
                if let (true, Some(pos), Some(tf)) = (
                    frame.response.clicked(),
                    frame.response.interact_pointer_pos(),
                    t_fixed,
                ) {
                    let w = camera.screen_to_world_um(pos, rect);
                    st.landmarks.push_click(
                        w,
                        &SlideTransform::Affine(tf),
                        &SlideTransform::Affine(t_moving),
                    );
                }
            }
            RegTool::Translate => {
                if frame.response.dragged_by(egui::PointerButton::Primary) {
                    let d = frame.response.drag_delta();
                    st.shift[0] += f64::from(d.x) * camera.um_per_px;
                    st.shift[1] += f64::from(d.y) * camera.um_per_px;
                }
            }
            RegTool::Rotate => {
                if frame.response.dragged_by(egui::PointerButton::Primary) {
                    if let Some(pos) = frame.response.interact_pointer_pos() {
                        let pivot_now = [pivot[0] + st.shift[0], pivot[1] + st.shift[1]];
                        let c = camera.world_um_to_screen(pivot_now, rect);
                        let prev = pos - frame.response.drag_delta();
                        let a1 = (pos.y - c.y).atan2(pos.x - c.x);
                        let a0 = (prev.y - c.y).atan2(prev.x - c.x);
                        let mut d = f64::from(a1 - a0);
                        if d > std::f64::consts::PI {
                            d -= std::f64::consts::TAU;
                        } else if d < -std::f64::consts::PI {
                            d += std::f64::consts::TAU;
                        }
                        st.angle += d;
                    }
                }
            }
            RegTool::View => {}
        }
    }
    let t_moving = manual_adjusted_transform(&t_old, st.shift, st.angle, st.flip, pivot);

    // Adegan. Masker jaringan menyembunyikan latar kaca di luar jaringan
    // (sama dengan "Sembunyikan latar" di 3D).
    let mut moving = slide_layer(&mut app.tiles, sess, gpu, slide, &t_moving, &frame.view);
    moving.mask = tissue_mask_if_enabled(app, sess, gpu, ui.ctx(), slide);
    let scene = match (neighbor, t_fixed) {
        (Some(n), Some(tf)) => {
            let mut fixed = slide_layer(&mut app.tiles, sess, gpu, n, &tf, &frame.view);
            fixed.mask = tissue_mask_if_enabled(app, sess, gpu, ui.ctx(), n);
            let st = &app.register;
            let mode = if st.tool == RegTool::Landmark {
                // Saat mengeklik landmark hanya sisi yang ditunggu yang tampil.
                CompareMode::Blend {
                    alpha: match st.landmarks.next_side() {
                        LandmarkSide::Fixed => 0.0,
                        LandmarkSide::Moving => 1.0,
                    },
                }
            } else {
                match st.compare {
                    CompareKind::Blend => CompareMode::Blend { alpha: st.alpha },
                    CompareKind::Checkerboard => CompareMode::Checkerboard {
                        size_px: st.checker,
                    },
                    CompareKind::Difference => CompareMode::Difference,
                }
            };
            Scene2D {
                view: camera.view(rect),
                base: None,
                label_layers: Vec::new(),
                compare: Some(CompareParams {
                    fixed,
                    moving,
                    mode,
                }),
            }
        }
        _ => Scene2D {
            view: camera.view(rect),
            base: Some(moving),
            label_layers: Vec::new(),
            compare: None,
        },
    };
    if gpu.is_some() {
        ui.painter_at(rect)
            .add(H3dPaintCallback::new(Scene::TwoD(scene)).into_egui(rect));
    }

    // Hiasan: landmark dan pegangan.
    let painter = ui.painter_at(rect);
    let st = &app.register;
    let magenta = egui::Color32::from_rgb(235, 60, 200);
    let green = egui::Color32::from_rgb(60, 220, 110);
    let font = egui::FontId::proportional(12.0);
    if let Some(tf) = t_fixed {
        for (i, (um, uf)) in st.landmarks.pairs.iter().enumerate() {
            let pf = camera.world_um_to_screen(tf.apply(*uf), rect);
            let pm = camera.world_um_to_screen(t_moving.apply(*um), rect);
            painter.line_segment([pf, pm], egui::Stroke::new(1.0, egui::Color32::WHITE));
            painter.circle_stroke(pf, 6.0, egui::Stroke::new(2.0, magenta));
            painter.circle_stroke(pm, 4.0, egui::Stroke::new(2.0, green));
            painter.text(
                pf + egui::vec2(8.0, -8.0),
                egui::Align2::LEFT_BOTTOM,
                (i + 1).to_string(),
                font.clone(),
                egui::Color32::WHITE,
            );
        }
        if let Some(uf) = st.landmarks.pending_fixed {
            let pf = camera.world_um_to_screen(tf.apply(uf), rect);
            painter.circle_stroke(pf, 6.0, egui::Stroke::new(2.0, magenta));
        }
    }
    if matches!(st.tool, RegTool::Translate | RegTool::Rotate) {
        let pivot_now = [pivot[0] + st.shift[0], pivot[1] + st.shift[1]];
        let c = camera.world_um_to_screen(pivot_now, rect);
        let accent = h3d_ui::theme::ACCENT;
        painter.circle_filled(c, 13.0, egui::Color32::from_black_alpha(160));
        let icon = if st.tool == RegTool::Translate {
            h3d_ui::icons::MOVE
        } else {
            h3d_ui::icons::ROTATE
        };
        painter.text(
            c,
            egui::Align2::CENTER_CENTER,
            icon.codepoint,
            egui::FontId::new(18.0, icon.font_family()),
            accent,
        );
        if st.tool == RegTool::Rotate {
            let r = (rect.width().min(rect.height()) * 0.3).max(40.0);
            painter.circle_stroke(c, r, egui::Stroke::new(1.5, accent.gamma_multiply(0.7)));
            let (s, co) = (st.angle as f32).sin_cos();
            painter.circle_filled(c + egui::vec2(co, s) * r, 6.0, accent);
        }
    }
    // Keterangan lapisan.
    let caption = match neighbor {
        Some(n) => format!(
            "{}  |  {}: {}",
            crate::ui_util::slide_title(sess, slide),
            t!("register-neighbor"),
            crate::ui_util::slide_title(sess, n)
        ),
        None => format!(
            "{}  |  {}",
            crate::ui_util::slide_title(sess, slide),
            t!("slides-is-reference")
        ),
    };
    let pos = rect.left_bottom() + egui::vec2(8.0, -8.0);
    let galley = painter.layout_no_wrap(caption, font, egui::Color32::WHITE);
    let bg = egui::Rect::from_min_size(
        pos - egui::vec2(4.0, galley.size().y + 4.0),
        galley.size() + egui::vec2(8.0, 8.0),
    );
    painter.rect_filled(bg, 4.0, egui::Color32::from_black_alpha(170));
    painter.galley(
        pos - egui::vec2(0.0, galley.size().y),
        galley,
        egui::Color32::WHITE,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: u32, order: u32, flagged: bool, registered: bool) -> QcRow {
        QcRow {
            slide: SlideId(id),
            order,
            file: format!("s{id}.png"),
            status: SlideStatus::Ok,
            is_reference: false,
            registered,
            fixed: None,
            method: None,
            ncc: None,
            mask_dice: None,
            area_ratio: None,
            flagged,
            manual: false,
        }
    }

    #[test]
    fn next_flagged_cycles_and_skips_current() {
        let f = [SlideId(1), SlideId(4), SlideId(9)];
        assert_eq!(next_flagged(&f, Some(SlideId(1)), true), Some(SlideId(4)));
        assert_eq!(next_flagged(&f, Some(SlideId(9)), true), Some(SlideId(1)));
        assert_eq!(next_flagged(&f, Some(SlideId(1)), false), Some(SlideId(9)));
        // Irisan aktif bukan yang gagal → mulai dari yang pertama.
        assert_eq!(next_flagged(&f, Some(SlideId(7)), true), Some(SlideId(1)));
        assert_eq!(next_flagged(&f, None, false), Some(SlideId(1)));
        // Satu-satunya irisan gagal adalah yang aktif → tidak ada tujuan.
        assert_eq!(next_flagged(&f[..1], Some(SlideId(1)), true), None);
        assert_eq!(next_flagged(&[], None, true), None);
    }

    #[test]
    fn failure_reasons_name_each_violated_threshold() {
        let mut r = row(1, 0, true, true);
        r.ncc = Some(0.14);
        r.mask_dice = Some(0.73);
        r.area_ratio = Some(1.0);
        assert_eq!(failure_reasons(&r).len(), 2);
        r.ncc = Some(0.9);
        r.mask_dice = Some(0.95);
        r.area_ratio = Some(1.3);
        assert_eq!(failure_reasons(&r).len(), 1);
        r.area_ratio = None;
        assert_eq!(
            failure_reasons(&r).len(),
            1,
            "metrik hilang dihitung pelanggaran"
        );
    }

    #[test]
    fn flagged_slides_come_first_in_qc_list() {
        let mut rows = vec![
            row(0, 0, false, true),
            row(1, 1, true, true),
            row(2, 2, false, false),
            row(3, 3, true, true),
            row(4, 4, false, true),
        ];
        sort_qc(&mut rows);
        let ids: Vec<u32> = rows.iter().map(|r| r.slide.0).collect();
        assert_eq!(ids, [1, 3, 2, 0, 4]);
    }

    #[test]
    fn edit_state_resets() {
        let mut st = RegisterState::default();
        assert!(st.qc_dirty && !st.has_manual());
        st.shift = [5.0, 0.0];
        st.angle = 0.1;
        assert!(st.has_manual());
        st.reset_edit();
        assert!(!st.has_manual());
    }
}
