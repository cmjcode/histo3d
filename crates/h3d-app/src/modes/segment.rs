//! Mode `Segment` (8.11, 8.5.2, 8.12.10): viewer slide + overlay label,
//! panel label (jenis berikon, grup), kuas/penghapus/laso/poligon, isi
//! lubang, salin ke irisan berikut, usulan klasik/AI/interpolasi dengan
//! Terima/Tolak, overlay inti positif/negatif untuk kalibrasi `t_D`, dan
//! "Tandai sudah ditinjau".
//!
//! Posisi penunjuk selalu dikonversi layar → W → U sebelum masuk ke
//! `LabelStore` (8.5.2). Satu goresan = satu langkah undo, lalu label
//! irisan langsung disimpan (9.4). Usulan tidak pernah menulis label tanpa
//! "Terima" (aturan keras 8).

use std::hash::{Hash, Hasher};
use std::path::PathBuf;

use egui::Ui;
use h3d_core::{
    AttrDef, AttrKind, LabelDef, LabelId, LabelKind, SlideId, SlideTransform, Stain, VesselKind,
};
use h3d_engine::{AttrParams, ClassicParams, EngineError, PaintMode, Proposal, Session};
use h3d_i18n::t;
use h3d_render::{Camera2D, H3dPaintCallback, LabelOverlayDraw, Scene, Scene2D};
use h3d_segment::StainChannel;

use crate::geom::{brush_radius_um, navigate, screen_to_slide_um};
use crate::gpu::Gpu;
use crate::panels::sample::combo;
use crate::state::{AppState, JobOutput};
use crate::tiles::{label_tiles_touched, palette_from_labels};
use crate::ui_util::{icon_label, icon_selectable, icon_text_button, icon_text_button_enabled};
use crate::viewer::{label_layer, nav_bar, paint_background, slide_layer, Viewer2D};

/// Alat aktif mode Segmentasi.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Tool {
    /// Geser/zoom pandangan.
    Pan,
    /// Kuas.
    #[default]
    Brush,
    /// Penghapus.
    Eraser,
    /// Laso.
    Lasso,
    /// Poligon titik demi titik.
    Polygon,
}

/// Inti terdeteksi pada wilayah yang tampil, untuk kalibrasi `t_D`.
#[derive(Clone, Debug, PartialEq)]
pub struct NucleiOverlay {
    /// Irisan.
    pub slide: SlideId,
    /// `(posisi di U µm, rata-rata OD kromogen)` per inti.
    pub points: Vec<([f64; 2], f32)>,
    /// Ukuran piksel kerja (µm).
    pub mpp: f64,
}

impl NucleiOverlay {
    /// `(positif, negatif)` untuk ambang `t_d` (inti positif bila `D > t_d`).
    pub fn counts(&self, t_d: f32) -> (usize, usize) {
        let pos = self.points.iter().filter(|(_, d)| *d > t_d).count();
        (pos, self.points.len() - pos)
    }
}

struct ProposalTex {
    key: u64,
    items: Vec<(egui::TextureHandle, [f64; 2], [f64; 2])>,
}

/// Tekstur overlay usulan yang di-cache (opak bagi modul lain).
pub struct ProposalTexCache(ProposalTex);

/// Keadaan mode Segmentasi.
pub struct SegmentState {
    /// Viewer.
    pub viewer: Viewer2D,
    /// Alat aktif.
    pub tool: Tool,
    /// Label aktif.
    pub active_label: Option<LabelId>,
    /// Jari-jari dasar kuas (µm).
    pub brush_radius_um: f64,
    /// Opasitas overlay label.
    pub opacity: f32,
    /// Hanya garis tepi label.
    pub outline: bool,
    /// Goresan berjalan: titik U (µm) dan jari-jari (µm) tiap titik.
    pub stroke: Vec<([f64; 2], f64)>,
    /// Titik laso/poligon berjalan (U, µm).
    pub polygon: Vec<[f64; 2]>,
    /// Usulan klasik yang menunggu keputusan.
    pub classic: Option<(SlideId, Proposal)>,
    /// Parameter segmentasi klasik.
    pub classic_params: ClassicParams,
    /// Ambang klasik otomatis (Otsu).
    pub classic_otsu: bool,
    /// Folder model AI di `models/` bundel.
    pub models: Vec<PathBuf>,
    /// Daftar model sudah dipindai.
    pub models_scanned: bool,
    /// Model terpilih.
    pub model_index: usize,
    /// Overlay inti untuk kalibrasi.
    pub nuclei: Option<NucleiOverlay>,
    /// Ambang OD kromogen `t_D`.
    pub t_d: f32,
    /// Cache tekstur usulan.
    pub proposal_tex: Option<ProposalTexCache>,
    /// Nama label baru.
    pub new_label_name: String,
    /// Jenis label baru.
    pub new_label_kind: LabelKind,
    /// Penyangga ganti nama label.
    pub name_edit: Option<(LabelId, String)>,
    /// Kotak pandang terakhir di ruang U irisan aktif (wilayah kalibrasi).
    pub last_view_u: Option<([f64; 2], [f64; 2])>,
}

impl Default for SegmentState {
    fn default() -> Self {
        SegmentState {
            viewer: Viewer2D::default(),
            tool: Tool::Brush,
            active_label: None,
            brush_radius_um: 80.0,
            opacity: 0.5,
            outline: false,
            stroke: Vec::new(),
            polygon: Vec::new(),
            classic: None,
            classic_params: ClassicParams::default(),
            classic_otsu: true,
            models: Vec::new(),
            models_scanned: false,
            model_index: 0,
            nuclei: None,
            t_d: h3d_imgproc::ihc::IhcThresholds::default().t_d,
            proposal_tex: None,
            new_label_name: String::new(),
            new_label_kind: LabelKind::Tumor,
            name_edit: None,
            last_view_u: None,
        }
    }
}

impl SegmentState {
    /// Simpan usulan klasik baru (menggantikan yang lama).
    pub fn set_classic(&mut self, slide: SlideId, proposal: Proposal) {
        self.classic = Some((slide, proposal));
        self.proposal_tex = None;
    }

    /// Buang goresan/poligon yang sedang berjalan.
    pub fn cancel_drawing(&mut self) {
        self.stroke.clear();
        self.polygon.clear();
    }
}

/// Ikon jenis label (8.12.10: pemilih `kind` berikon).
pub fn kind_icon(kind: LabelKind) -> h3d_ui::icons::MaterialIcon {
    use h3d_ui::icons;
    match kind {
        LabelKind::Tumor => icons::KIND_TUMOR,
        LabelKind::Necrosis => icons::KIND_NECROSIS,
        LabelKind::Stroma => icons::KIND_STROMA,
        LabelKind::Normal => icons::KIND_NORMAL,
        LabelKind::Vessel(_) => icons::KIND_VESSEL,
        LabelKind::Nerve => icons::KIND_NERVE,
        LabelKind::Capsule => icons::KIND_CAPSULE,
        LabelKind::Margin => icons::KIND_MARGIN,
        LabelKind::Surface => icons::KIND_SURFACE,
        LabelKind::Reference => icons::MEASURE,
        LabelKind::Other => icons::KIND_OTHER,
    }
}

/// Kunci terjemahan jenis label; pembuluh dibedakan menurut jenisnya.
pub fn kind_key(kind: LabelKind) -> &'static str {
    match kind {
        LabelKind::Vessel(VesselKind::Blood) => "label-kind-vessel-blood",
        LabelKind::Vessel(VesselKind::Lymph) => "label-kind-vessel-lymph",
        LabelKind::Vessel(VesselKind::Unknown) => "label-kind-vessel-unknown",
        other => other.i18n_key(),
    }
}

/// Combo jenis label berikon; `true` bila berubah.
pub fn kind_combo(ui: &mut Ui, id: &str, kind: &mut LabelKind) -> bool {
    let mut changed = false;
    egui::ComboBox::from_id_salt(id)
        .selected_text(icon_label(kind_icon(*kind), &t!(kind_key(*kind))))
        .show_ui(ui, |ui| {
            for k in LabelKind::ALL {
                if ui
                    .selectable_label(*kind == k, icon_label(kind_icon(k), &t!(kind_key(k))))
                    .clicked()
                    && *kind != k
                {
                    *kind = k;
                    changed = true;
                }
            }
        });
    changed
}

/// Goreskan `points` (U µm, jari-jari per titik) ke label irisan sebagai
/// **satu** langkah undo. Jari-jari tiap ruas = rata-rata kedua ujungnya,
/// sehingga tekanan Pencil mengubah lebar sepanjang goresan.
pub fn commit_stroke(
    sess: &mut Session,
    slide: SlideId,
    label: LabelId,
    points: &[([f64; 2], f64)],
    erase: bool,
) -> Result<bool, EngineError> {
    if points.is_empty() {
        return Ok(false);
    }
    let pts = points.to_vec();
    // Penghapus menghapus label apa pun (label latar pada mode `Erase`).
    let (target, mode, touched) = if erase {
        (LabelId::BACKGROUND, PaintMode::Erase, None)
    } else {
        (label, PaintMode::Replace, Some(label))
    };
    let name = if erase { "penghapus" } else { "kuas" };
    sess.edit_labels(slide, touched, name, move |store| {
        if pts.len() == 1 {
            store.paint_disc(pts[0].0, pts[0].1, target, mode);
        }
        for w in pts.windows(2) {
            let r = (w[0].1 + w[1].1) / 2.0;
            store.paint_stroke(&[w[0].0, w[1].0], r, target, mode);
        }
    })
}

/// Deteksi inti pada wilayah U `[lo, hi]` irisan dan ukur OD kromogennya
/// (untuk overlay kalibrasi `t_D`). Level dipilih supaya wilayah paling
/// besar 2048 piksel per sisi.
pub fn detect_nuclei_region(
    sess: &Session,
    slide: SlideId,
    lo: [f64; 2],
    hi: [f64; 2],
) -> Result<NucleiOverlay, EngineError> {
    use h3d_imgproc::color::{deconvolve, StainMatrix};
    use h3d_imgproc::nuclei::{detect_nuclei, measure_chromogen, NucleiParams};

    let entry = sess.slide(slide)?.clone();
    let size = entry.size_um();
    let lo = [lo[0].max(0.0), lo[1].max(0.0)];
    let hi = [hi[0].min(size[0]), hi[1].min(size[1])];
    if !(hi[0] > lo[0] && hi[1] > lo[1]) {
        return Err(EngineError::invalid("wilayah kalibrasi di luar irisan"));
    }
    let src = sess.open_source(slide)?;
    let info = src.info().clone();
    let mut level = info.levels.len().saturating_sub(1) as u32;
    for l in 0..info.levels.len() as u32 {
        if let Some(s) = crate::tiles::level_scale(&info, l) {
            let w = (hi[0] - lo[0]) / entry.mpp[0] / s[0];
            let h = (hi[1] - lo[1]) / entry.mpp[1] / s[1];
            if w.max(h) <= 2048.0 {
                level = l;
                break;
            }
        }
    }
    let s = crate::tiles::level_scale(&info, level)
        .ok_or_else(|| EngineError::invalid("slide tanpa level"))?;
    let px = [entry.mpp[0] * s[0], entry.mpp[1] * s[1]];
    let x0 = (lo[0] / px[0]).floor();
    let y0 = (lo[1] / px[1]).floor();
    let w = (((hi[0] - lo[0]) / px[0]).ceil().clamp(1.0, 2048.0)) as u32;
    let h = (((hi[1] - lo[1]) / px[1]).ceil().clamp(1.0, 2048.0)) as u32;
    let img = src.read_region(level, x0 as i64, y0 as i64, w, h)?;
    let matrix = match entry.ihc.as_ref().and_then(|i| i.od_vectors) {
        Some(v) => StainMatrix::new(v)?,
        None => StainMatrix::h_dab(),
    };
    let [hch, dch, _] = deconvolve(img.as_raw(), w as usize, h as usize, &matrix)?;
    let mpp = (px[0] + px[1]) / 2.0;
    let mut nuclei = detect_nuclei(&hch, mpp, None, &NucleiParams::default());
    measure_chromogen(&mut nuclei, &dch, mpp);
    let points = nuclei
        .iter()
        .map(|n| {
            (
                [(x0 + n.x_um / mpp) * px[0], (y0 + n.y_um / mpp) * px[1]],
                n.d_mean,
            )
        })
        .collect();
    Ok(NucleiOverlay { slide, points, mpp })
}

/// Folder model di `models/` bundel (subfolder berisi `model.json`).
pub fn scan_models(sess: &Session) -> Vec<PathBuf> {
    let dir = sess.bundle().models_dir();
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .map(|e| e.path())
                .filter(|p| p.join("model.json").is_file())
                .collect()
        })
        .unwrap_or_default();
    out.sort();
    out
}

fn palette_hash(labels: &[LabelDef]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for l in labels {
        (l.id.0, l.color, l.visible).hash(&mut h);
    }
    // Tidak pernah 0 agar berbeda dari keadaan "belum dikirim".
    h.finish() | 1
}

/// Urung label irisan aktif.
pub fn undo(app: &mut AppState, sess: &mut Session, redo: bool) {
    let Some(slide) = app.current else {
        return;
    };
    let res = if redo {
        sess.redo(slide)
    } else {
        sess.undo(slide)
    };
    match res {
        Ok(Some(_)) => {
            app.labels_changed(slide);
            if let Err(e) = sess.save_labels(slide) {
                app.toasts.engine_error(&e);
            }
        }
        Ok(None) => {}
        Err(e) => app.toasts.engine_error(&e),
    }
}

/// Gambar mode Segmentasi.
pub fn ui(app: &mut AppState, sess: &mut Session, ui: &mut Ui, gpu: Option<&Gpu>) {
    app.fix_current(sess);
    let labels = sess.project().labels.clone();
    if !app
        .segment
        .active_label
        .is_some_and(|a| labels.iter().any(|l| l.id == a))
    {
        app.segment.active_label = labels.first().map(|l| l.id);
    }
    let key = palette_hash(&labels);
    if key != app.palette_key {
        if let Some(g) = gpu {
            let palette = palette_from_labels(&labels);
            g.with(|r, _, queue| r.set_palette(queue, &palette));
            app.palette_key = key;
        }
    }

    egui::Panel::right("segment-right")
        .default_size(330.0)
        .show(ui, |ui| {
            egui::ScrollArea::vertical()
                .id_salt("segment-right-scroll")
                .show(ui, |ui| {
                    label_panel(app, sess, ui);
                    ui.separator();
                    proposals_panel(app, sess, ui);
                    ui.separator();
                    calibration_panel(app, sess, ui);
                });
        });

    egui::CentralPanel::default().show(ui, |ui| {
        toolbar(app, sess, ui);
        ui.separator();
        viewer(app, sess, ui, gpu);
    });
}

fn toolbar(app: &mut AppState, sess: &mut Session, ui: &mut Ui) {
    let order = AppState::ordered_ids(sess);
    let index = app
        .current
        .and_then(|c| order.iter().position(|o| *o == c))
        .unwrap_or(0);
    ui.horizontal_wrapped(|ui| {
        if let Some(dir) = nav_bar(ui, index, order.len()) {
            app.current = navigate(&order, app.current, dir);
            app.segment.cancel_drawing();
        }
        ui.separator();
        for (tool, icon, key) in [
            (Tool::Pan, h3d_ui::icons::PAN, "tool-pan"),
            (Tool::Brush, h3d_ui::icons::BRUSH, "segment-tool-brush"),
            (Tool::Eraser, h3d_ui::icons::ERASER, "segment-tool-eraser"),
            (Tool::Lasso, h3d_ui::icons::LASSO, "segment-tool-lasso"),
            (
                Tool::Polygon,
                h3d_ui::icons::POLYGON,
                "segment-tool-polygon",
            ),
        ] {
            if icon_selectable(ui, app.segment.tool == tool, icon, &t!(key)).clicked() {
                app.segment.tool = tool;
                app.segment.cancel_drawing();
            }
        }
        ui.separator();
        ui.add(
            egui::Slider::new(&mut app.segment.brush_radius_um, 4.0..=2000.0)
                .logarithmic(true)
                .suffix(format!(" {}", t!("unit-um")))
                .text(t!("segment-brush-size")),
        );
    });
    ui.horizontal_wrapped(|ui| {
        let Some(slide) = app.current else {
            return;
        };
        let active = app.segment.active_label;
        if icon_text_button_enabled(
            ui,
            active.is_some(),
            h3d_ui::icons::FILL,
            &t!("segment-fill-holes"),
        )
        .clicked()
        {
            if let Some(label) = active {
                match sess.fill_label_holes(slide, label) {
                    Ok(_) => after_edit(app, sess, slide),
                    Err(e) => app.toasts.engine_error(&e),
                }
            }
        }
        if icon_text_button(ui, h3d_ui::icons::COPY, &t!("segment-copy-to-next")).clicked() {
            match sess.copy_labels_to_next(slide) {
                Ok(dst) => {
                    after_edit(app, sess, dst);
                    app.current = Some(dst);
                }
                Err(e) => app.toasts.engine_error(&e),
            }
        }
        if icon_text_button_enabled(
            ui,
            active.is_some(),
            h3d_ui::icons::REVIEWED,
            &t!("segment-mark-reviewed"),
        )
        .clicked()
        {
            if let Some(label) = active {
                let by = app.settings.user_initials.trim().to_owned();
                if by.is_empty() {
                    app.toasts.error(t!("settings-initials-required"));
                    app.show_settings = true;
                } else {
                    match sess.mark_reviewed(slide, label, &by) {
                        Ok(()) => app.report.needs_refresh = true,
                        Err(e) => app.toasts.engine_error(&e),
                    }
                }
            }
        }
        // Status provenance label aktif pada irisan ini.
        if let Some(rec) = active.and_then(|l| sess.project().annotation(slide, l)) {
            let source = match rec.source.short_name() {
                "manual" => t!("segment-source-manual"),
                "classic" => t!("segment-source-classic"),
                "ai" => t!("segment-source-ai"),
                "interpolated" => t!("segment-source-interpolated"),
                "imported" => t!("segment-source-imported"),
                other => other.to_string(),
            };
            let (text, color) = if rec.review.is_reviewed() {
                (t!("segment-review-reviewed"), h3d_ui::theme::SUCCESS)
            } else {
                (t!("segment-review-draft"), h3d_ui::theme::WARNING)
            };
            ui.label(format!("{source} /"));
            ui.colored_label(color, text);
        }
        ui.separator();
        ui.add(
            egui::Slider::new(&mut app.segment.opacity, 0.0..=1.0)
                .text(t!("segment-overlay-opacity")),
        );
        ui.toggle_value(
            &mut app.segment.outline,
            icon_label(h3d_ui::icons::OUTLINE, &t!("segment-outline")),
        );
        if crate::platform::is_mobile() || app.settings.finger_draw {
            let mut on = app.settings.finger_draw;
            if ui
                .toggle_value(
                    &mut on,
                    icon_label(h3d_ui::icons::TOUCH, &t!("segment-finger-draw")),
                )
                .changed()
            {
                app.settings.finger_draw = on;
                app.persist_settings();
            }
        }
    });
}

/// Setelah suntingan label non-goresan: segarkan tile, simpan (9.4).
fn after_edit(app: &mut AppState, sess: &mut Session, slide: SlideId) {
    app.labels_changed(slide);
    if let Err(e) = sess.save_labels(slide) {
        app.toasts.engine_error(&e);
    }
}

fn label_panel(app: &mut AppState, sess: &mut Session, ui: &mut Ui) {
    ui.heading(t!("label-panel"));
    let labels = sess.project().labels.clone();
    let mut update: Option<LabelDef> = None;
    let mut remove: Option<LabelId> = None;
    for def in &labels {
        let active = app.segment.active_label == Some(def.id);
        let mut d = def.clone();
        let mut changed = false;
        ui.horizontal(|ui| {
            if ui.radio(active, "").clicked() {
                app.segment.active_label = Some(def.id);
            }
            changed |= ui
                .color_edit_button_srgba_unmultiplied(&mut d.color)
                .changed();
            ui.label(kind_icon(def.kind).rich_text())
                .on_hover_text(t!(kind_key(def.kind)));
            // Nama: disunting di penyangga, diterapkan saat fokus lepas.
            let editing = app
                .segment
                .name_edit
                .as_ref()
                .is_some_and(|(id, _)| *id == def.id);
            let mut buf = if editing {
                app.segment
                    .name_edit
                    .as_ref()
                    .map(|(_, s)| s.clone())
                    .unwrap_or_default()
            } else {
                def.name.clone()
            };
            let resp = ui.add(egui::TextEdit::singleline(&mut buf).desired_width(110.0));
            if resp.changed() {
                app.segment.name_edit = Some((def.id, buf.clone()));
            }
            if resp.lost_focus() && editing {
                app.segment.name_edit = None;
                if buf.trim() != def.name {
                    d.name = buf.trim().to_owned();
                    changed = true;
                }
            }
            let vis_icon = if d.visible {
                h3d_ui::icons::VISIBILITY_ON
            } else {
                h3d_ui::icons::VISIBILITY_OFF
            };
            if h3d_ui::widgets::icon_button(ui, vis_icon, &t!("label-visible")).clicked() {
                d.visible = !d.visible;
                changed = true;
            }
            changed |= h3d_ui::widgets::toggle_icon_button(
                ui,
                &mut d.reconstruct,
                h3d_ui::icons::CUBE,
                &t!("label-reconstruct"),
            )
            .clicked();
        });
        if active {
            ui.indent(("label-detail", def.id.0), |ui| {
                ui.horizontal(|ui| {
                    ui.label(t!("label-kind"));
                    changed |= kind_combo(ui, "label-kind-combo", &mut d.kind);
                });
                ui.horizontal(|ui| {
                    ui.label(t!("label-group"));
                    changed |= crate::ui_util::opt_text(ui, &mut d.group, 120.0);
                    if h3d_ui::widgets::icon_button(ui, h3d_ui::icons::DELETE, &t!("action-delete"))
                        .clicked()
                    {
                        remove = Some(def.id);
                    }
                });
            });
        }
        if changed {
            update = Some(d);
        }
    }
    if let Some(d) = update {
        if let Err(e) = sess.update_label(d) {
            app.toasts.engine_error(&e);
        }
        app.recon.needs_reload = true;
    }
    if let Some(id) = remove {
        if let Err(e) = sess.remove_label(id) {
            app.toasts.engine_error(&e);
        }
    }
    ui.separator();
    ui.horizontal(|ui| {
        ui.add(
            egui::TextEdit::singleline(&mut app.segment.new_label_name)
                .hint_text(t!("label-name"))
                .desired_width(100.0),
        );
        kind_combo(ui, "new-label-kind", &mut app.segment.new_label_kind);
        if h3d_ui::widgets::icon_button(ui, h3d_ui::icons::ADD, &t!("action-add")).clicked() {
            let n = sess.project().labels.len() as f32;
            let c = h3d_core::Colormap::Viridis.sample((n * 0.37).fract());
            match sess.add_label(
                &app.segment.new_label_name,
                [c[0], c[1], c[2], 255],
                app.segment.new_label_kind,
            ) {
                Ok(id) => {
                    app.segment.active_label = Some(id);
                    app.segment.new_label_name.clear();
                }
                Err(e) => app.toasts.engine_error(&e),
            }
        }
    });
}

fn proposals_panel(app: &mut AppState, sess: &mut Session, ui: &mut Ui) {
    ui.heading(t!("segment-proposals"));
    ui.small(t!("segment-proposal-hint"));
    let Some(slide) = app.current else {
        return;
    };
    let active = app.segment.active_label;

    // --- Keputusan atas usulan yang menunggu (lapisan terpisah di viewer) ---
    let n_pending = sess.pending_proposals(slide).len();
    let has_classic = app
        .segment
        .classic
        .as_ref()
        .is_some_and(|(s, _)| *s == slide);
    if n_pending > 0 || has_classic {
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.label(crate::ui_util::icon_label_colored(
                h3d_ui::icons::LAYERS,
                &t!(
                    "segment-proposals-waiting",
                    n = (n_pending + usize::from(has_classic)).to_string()
                ),
                h3d_ui::theme::WARNING,
            ));
            for p in sess.pending_proposals(slide) {
                let name = sess
                    .project()
                    .label(p.proposal.label)
                    .map(|l| l.name.clone())
                    .unwrap_or_default();
                let conf = p
                    .confidence
                    .map(|c| format!(", {:.2}", c))
                    .unwrap_or_default();
                ui.small(format!(
                    "{name}: {:.3} {}{conf}",
                    p.proposal.area_um2() / 1.0e6,
                    t!("unit-mm2")
                ));
            }
            ui.horizontal(|ui| {
                if icon_text_button(ui, h3d_ui::icons::CHECK, &t!("segment-proposal-accept"))
                    .clicked()
                {
                    let mut ok = true;
                    if let Some((s, p)) = app.segment.classic.take() {
                        if s == slide {
                            if let Err(e) =
                                sess.accept_proposal(slide, &p, PaintMode::OnlyBackground)
                            {
                                app.toasts.engine_error(&e);
                                ok = false;
                            }
                        } else {
                            app.segment.classic = Some((s, p));
                        }
                    }
                    if ok && n_pending > 0 {
                        if let Err(e) = sess.accept_pending(slide, PaintMode::OnlyBackground) {
                            app.toasts.engine_error(&e);
                        }
                    }
                    app.segment.proposal_tex = None;
                    after_edit(app, sess, slide);
                }
                if icon_text_button(ui, h3d_ui::icons::CLOSE, &t!("segment-proposal-reject"))
                    .clicked()
                {
                    if has_classic {
                        app.segment.classic = None;
                    }
                    sess.reject_pending(slide);
                    app.segment.proposal_tex = None;
                }
            });
        });
    }
    let others = sess
        .pending_slides()
        .into_iter()
        .filter(|s| *s != slide)
        .count();
    if others > 0 {
        ui.weak(t!("segment-proposals-other-slides", n = others.to_string()));
    }

    // --- Klasik ---
    egui::CollapsingHeader::new(icon_label(
        h3d_ui::icons::CLASSIC,
        &t!("segment-proposal-classic"),
    ))
    .default_open(true)
    .show(ui, |ui| {
        let st = &mut app.segment;
        ui.horizontal(|ui| {
            ui.label(t!("segment-classic-channel"));
            combo(
                ui,
                "classic-channel",
                &mut st.classic_params.channel,
                &[
                    (StainChannel::Hematoxylin, t!("segment-channel-hematoxylin")),
                    (StainChannel::Eosin, t!("segment-channel-eosin")),
                    (StainChannel::Dab, t!("segment-channel-dab")),
                ],
            );
        });
        ui.checkbox(&mut st.classic_otsu, t!("segment-classic-otsu"));
        if !st.classic_otsu {
            let mut thr = st.classic_params.threshold.unwrap_or(0.5);
            ui.add(egui::Slider::new(&mut thr, 0.0..=1.0).text(t!("segment-threshold")));
            st.classic_params.threshold = Some(thr);
        } else {
            st.classic_params.threshold = None;
        }
        ui.checkbox(&mut st.classic_params.invert, t!("segment-classic-invert"));
        ui.checkbox(&mut st.classic_params.fill_holes, t!("segment-fill-holes"));
        let params = st.classic_params.clone();
        if icon_text_button_enabled(ui, active.is_some(), h3d_ui::icons::PLAY, &t!("action-run"))
            .clicked()
        {
            if let Some(label) = active {
                app.start_job(t!("segment-proposal-classic"), move |s, _p| {
                    s.segment_classic(slide, label, &params)
                        .map(|p| JobOutput::ClassicProposal(slide, Box::new(p)))
                });
            }
        }
    });

    // --- AI ---
    egui::CollapsingHeader::new(icon_label(h3d_ui::icons::AI, &t!("segment-proposal-ai")))
        .default_open(false)
        .show(ui, |ui| {
            if !app.segment.models_scanned {
                app.segment.models = scan_models(sess);
                app.segment.models_scanned = true;
            }
            let st = &mut app.segment;
            if st.models.is_empty() {
                ui.weak(t!("segment-ai-no-models"));
            } else {
                st.model_index = st.model_index.min(st.models.len() - 1);
                let name = |p: &PathBuf| {
                    p.file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default()
                };
                egui::ComboBox::from_id_salt("ai-model")
                    .selected_text(name(&st.models[st.model_index]))
                    .show_ui(ui, |ui| {
                        for (i, m) in st.models.iter().enumerate() {
                            ui.selectable_value(&mut st.model_index, i, name(m));
                        }
                    });
            }
            let model = st.models.get(st.model_index).cloned();
            ui.horizontal(|ui| {
                if h3d_ui::widgets::icon_button(ui, h3d_ui::icons::REFRESH, &t!("action-refresh"))
                    .clicked()
                {
                    app.segment.models_scanned = false;
                }
                if icon_text_button_enabled(
                    ui,
                    model.is_some(),
                    h3d_ui::icons::PLAY,
                    &t!("action-run"),
                )
                .clicked()
                {
                    if let Some(model) = model {
                        app.start_job(t!("segment-proposal-ai"), move |s, p| {
                            s.segment_ai(&model, &[slide], p)?;
                            Ok(JobOutput::PendingProposals)
                        });
                    }
                }
            });
        });

    // --- Interpolasi ---
    egui::CollapsingHeader::new(icon_label(
        h3d_ui::icons::INTERPOLATE,
        &t!("segment-proposal-interpolate"),
    ))
    .default_open(false)
    .show(ui, |ui| {
        ui.small(t!("segment-interpolate-hint"));
        if icon_text_button_enabled(ui, active.is_some(), h3d_ui::icons::PLAY, &t!("action-run"))
            .clicked()
        {
            if let Some(label) = active {
                app.start_job(t!("segment-proposal-interpolate"), move |s, p| {
                    s.interpolate_labels(label, p)?;
                    Ok(JobOutput::PendingProposals)
                });
            }
        }
    });
}

fn calibration_panel(app: &mut AppState, sess: &mut Session, ui: &mut Ui) {
    egui::CollapsingHeader::new(icon_label(
        h3d_ui::icons::CALIBRATE,
        &t!("segment-calibrate-threshold"),
    ))
    .default_open(false)
    .show(ui, |ui| {
        let Some(slide) = app.current else {
            return;
        };
        ui.small(t!("segment-calibrate-hint"));
        let stain = sess.slide(slide).map(|s| s.stain).ok();
        if stain != Some(Stain::Ihc) {
            ui.weak(t!("segment-calibrate-not-ihc"));
        }
        if icon_text_button(ui, h3d_ui::icons::PLAY, &t!("segment-calibrate-detect")).clicked() {
            if let Some((lo, hi)) = app.segment.last_view_u {
                app.start_job(t!("segment-calibrate-threshold"), move |s, _p| {
                    detect_nuclei_region(s, slide, lo, hi).map(|n| JobOutput::Nuclei(Box::new(n)))
                });
            }
        }
        ui.add(
            egui::Slider::new(&mut app.segment.t_d, 0.01..=1.0).text(t!("segment-threshold-td")),
        );
        if let Some(n) = app.segment.nuclei.as_ref().filter(|n| n.slide == slide) {
            let (pos, neg) = n.counts(app.segment.t_d);
            ui.horizontal(|ui| {
                ui.colored_label(
                    POSITIVE_COLOR,
                    format!("{}: {pos}", t!("segment-positive-nuclei")),
                );
                ui.colored_label(
                    NEGATIVE_COLOR,
                    format!("{}: {neg}", t!("segment-negative-nuclei")),
                );
            });
            if pos + neg > 0 {
                ui.label(format!(
                    "{:.1} {}",
                    100.0 * pos as f64 / (pos + neg) as f64,
                    t!("unit-percent")
                ));
            }
            if ui.button(t!("segment-calibrate-clear")).clicked() {
                app.segment.nuclei = None;
            }
        }
        // Hitung atribut skor IHC irisan ini dengan ambang terkalibrasi.
        let scoring = sess
            .slide(slide)
            .ok()
            .and_then(|s| s.ihc.as_ref().map(|i| i.scoring));
        if icon_text_button_enabled(
            ui,
            scoring.is_some(),
            h3d_ui::icons::ANALYZE,
            &t!("attr-compute-ihc"),
        )
        .clicked()
        {
            if let Some(scoring) = scoring {
                let t_d = app.segment.t_d;
                app.start_job(t!("attr-compute-ihc"), move |s, p| {
                    let mut params = AttrParams::default();
                    params.thresholds.t_d = t_d;
                    let def = AttrDef {
                        kind: AttrKind::IhcScore { slide, scoring },
                        ..Default::default()
                    };
                    s.compute_attr_with(def, &[slide], &params, p)
                        .map(JobOutput::AttrComputed)
                });
            }
        }
    });
}

const POSITIVE_COLOR: egui::Color32 = egui::Color32::from_rgb(240, 90, 60);
const NEGATIVE_COLOR: egui::Color32 = egui::Color32::from_rgb(70, 150, 250);

fn proposal_key(slide: SlideId, props: &[&Proposal]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    slide.hash(&mut h);
    for p in props {
        (
            p.label.0,
            p.mask.dim(),
            p.origin_um[0].to_bits(),
            p.mpp.to_bits(),
        )
            .hash(&mut h);
        p.area_um2().to_bits().hash(&mut h);
    }
    h.finish()
}

/// Gambar masker usulan sebagai citra bergaris diagonal berwarna label,
/// paling besar 1024 piksel per sisi.
pub fn proposal_image(p: &Proposal, color: [u8; 4]) -> egui::ColorImage {
    let (h, w) = p.mask.dim();
    let step = (h.max(w)).div_ceil(1024).max(1);
    let (ow, oh) = (w.div_ceil(step).max(1), h.div_ceil(step).max(1));
    let mut img = egui::ColorImage::filled([ow, oh], egui::Color32::TRANSPARENT);
    for y in 0..oh {
        for x in 0..ow {
            let (sx, sy) = (
                (x * step).min(w.saturating_sub(1)),
                (y * step).min(h.saturating_sub(1)),
            );
            if h > 0 && w > 0 && p.mask[[sy, sx]] != 0 {
                let alpha = if ((x + y) / 4) % 2 == 0 { 190 } else { 70 };
                img.pixels[y * ow + x] =
                    egui::Color32::from_rgba_unmultiplied(color[0], color[1], color[2], alpha);
            }
        }
    }
    img
}

fn viewer(app: &mut AppState, sess: &mut Session, ui: &mut Ui, gpu: Option<&Gpu>) {
    let Some(slide) = app.current else {
        ui.centered_and_justified(|ui| ui.weak(t!("slides-empty")));
        return;
    };
    let Ok(entry) = sess.slide(slide).cloned() else {
        return;
    };
    let t_affine = entry.transform.affine();
    let transform = entry.transform.clone();
    let size = entry.size_um();
    let (bmin, bmax) = t_affine.transform_rect(size[0], size[1]);

    let can_draw = app.segment.tool != Tool::Pan && app.touch.may_draw(app.settings.finger_draw);
    let frame = app
        .segment
        .viewer
        .begin(ui, u64::from(slide.0), Some((bmin, bmax)), !can_draw);
    if let Some(dir) = frame.nav {
        let order = AppState::ordered_ids(sess);
        app.current = navigate(&order, app.current, dir);
        app.segment.cancel_drawing();
    }
    let rect = frame.rect;
    let camera = app.segment.viewer.camera;
    app.segment.last_view_u = crate::tiles::view_bbox_in_slide(&t_affine, &frame.view);
    paint_background(ui, rect, gpu.is_some());

    if frame.multi_touch {
        // Gestur dua jari membatalkan goresan yang sedang berjalan.
        app.segment.cancel_drawing();
    } else if can_draw {
        handle_drawing(
            app,
            sess,
            slide,
            &transform,
            &camera,
            &frame.response,
            rect,
            ui,
        );
    }

    // Adegan: gambar + overlay label.
    let base = slide_layer(&mut app.tiles, sess, gpu, slide, &t_affine, &frame.view);
    let mut label_layers = Vec::new();
    match sess.labels(slide) {
        Ok(_) => {}
        Err(e) => log::warn!("label irisan {slide} gagal dimuat: {e}"),
    }
    if let Some(store) = sess.labels_loaded(slide) {
        let tiles = label_layer(&mut app.tiles, gpu, slide, store, &t_affine, &frame.view);
        if !tiles.is_empty() {
            label_layers.push(LabelOverlayDraw {
                affine: t_affine,
                tiles,
                opacity: app.segment.opacity,
                outline: app.segment.outline,
            });
        }
    }
    if gpu.is_some() {
        let scene = Scene2D {
            view: camera.view(rect),
            base: Some(base),
            label_layers,
            compare: None,
        };
        ui.painter_at(rect)
            .add(H3dPaintCallback::new(Scene::TwoD(scene)).into_egui(rect));
    }

    let painter = ui.painter_at(rect);
    let to_screen =
        |u: [f64; 2]| -> egui::Pos2 { camera.world_um_to_screen(transform.apply(u), rect) };

    // Lapisan usulan (terpisah dari label; menunggu Terima/Tolak).
    {
        let mut props: Vec<&Proposal> = sess
            .pending_proposals(slide)
            .iter()
            .map(|p| &p.proposal)
            .collect();
        if let Some((s, p)) = &app.segment.classic {
            if *s == slide {
                props.push(p);
            }
        }
        if props.is_empty() {
            app.segment.proposal_tex = None;
        } else {
            let key = proposal_key(slide, &props);
            let stale = app
                .segment
                .proposal_tex
                .as_ref()
                .is_none_or(|c| c.0.key != key);
            if stale {
                let items = props
                    .iter()
                    .enumerate()
                    .map(|(i, p)| {
                        let color = sess
                            .project()
                            .label(p.label)
                            .map_or([255, 200, 0, 255], |l| l.color);
                        let (h, w) = p.mask.dim();
                        let tex = ui.ctx().load_texture(
                            format!("proposal-{i}"),
                            proposal_image(p, color),
                            egui::TextureOptions::NEAREST,
                        );
                        (tex, p.origin_um, [w as f64 * p.mpp, h as f64 * p.mpp])
                    })
                    .collect();
                app.segment.proposal_tex = Some(ProposalTexCache(ProposalTex { key, items }));
            }
            if let Some(cache) = &app.segment.proposal_tex {
                for (tex, origin, size) in &cache.0.items {
                    let corners = [
                        [origin[0], origin[1]],
                        [origin[0] + size[0], origin[1]],
                        [origin[0] + size[0], origin[1] + size[1]],
                        [origin[0], origin[1] + size[1]],
                    ];
                    let uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
                    let mut mesh = egui::Mesh::with_texture(tex.id());
                    for (c, uv) in corners.iter().zip(uvs) {
                        mesh.vertices.push(egui::epaint::Vertex {
                            pos: to_screen(*c),
                            uv: egui::pos2(uv.0, uv.1),
                            color: egui::Color32::WHITE,
                        });
                    }
                    mesh.indices.extend([0, 1, 2, 0, 2, 3]);
                    painter.add(egui::Shape::mesh(mesh));
                    let outline: Vec<egui::Pos2> = corners.iter().map(|c| to_screen(*c)).collect();
                    painter.add(egui::Shape::closed_line(
                        outline,
                        egui::Stroke::new(1.0, h3d_ui::theme::WARNING),
                    ));
                }
            }
        }
    }

    // Overlay inti kalibrasi.
    if let Some(n) = app.segment.nuclei.as_ref().filter(|n| n.slide == slide) {
        let t_d = app.segment.t_d;
        for (u, d) in n.points.iter().take(20_000) {
            let p = to_screen(*u);
            if rect.contains(p) {
                let color = if *d > t_d {
                    POSITIVE_COLOR
                } else {
                    NEGATIVE_COLOR
                };
                painter.circle_stroke(p, 4.0, egui::Stroke::new(1.5, color));
            }
        }
    }

    // Pratinjau goresan/poligon berjalan.
    let color = app
        .segment
        .active_label
        .and_then(|l| sess.project().label(l))
        .map_or(egui::Color32::YELLOW, |l| {
            egui::Color32::from_rgb(l.color[0], l.color[1], l.color[2])
        });
    let st = &app.segment;
    for w in st.stroke.windows(2) {
        let r = ((w[0].1 + w[1].1) / camera.um_per_px) as f32;
        painter.line_segment(
            [to_screen(w[0].0), to_screen(w[1].0)],
            egui::Stroke::new(r.max(1.0), color.gamma_multiply(0.6)),
        );
    }
    if let [only] = st.stroke.as_slice() {
        painter.circle_filled(
            to_screen(only.0),
            (only.1 / camera.um_per_px) as f32,
            color.gamma_multiply(0.6),
        );
    }
    if !st.polygon.is_empty() {
        let pts: Vec<egui::Pos2> = st.polygon.iter().map(|u| to_screen(*u)).collect();
        painter.add(egui::Shape::line(
            pts.clone(),
            egui::Stroke::new(2.0, color),
        ));
        if st.tool == Tool::Polygon {
            for p in &pts {
                painter.circle_filled(*p, 3.0, color);
            }
        }
    }
    // Kursor kuas.
    if matches!(st.tool, Tool::Brush | Tool::Eraser) {
        if let Some(pos) = frame.response.hover_pos() {
            let r = (brush_radius_um(st.brush_radius_um, app.touch.pressure()) / camera.um_per_px)
                as f32;
            painter.circle_stroke(pos, r, egui::Stroke::new(1.0, egui::Color32::WHITE));
        }
    }
    if st.tool == Tool::Polygon && !st.polygon.is_empty() {
        let pos = rect.left_top() + egui::vec2(8.0, 8.0);
        painter.text(
            pos,
            egui::Align2::LEFT_TOP,
            t!("segment-polygon-hint"),
            egui::FontId::proportional(12.0),
            egui::Color32::WHITE,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_drawing(
    app: &mut AppState,
    sess: &mut Session,
    slide: SlideId,
    transform: &SlideTransform,
    camera: &Camera2D,
    response: &egui::Response,
    rect: egui::Rect,
    ui: &Ui,
) {
    let Some(label) = app.segment.active_label else {
        return;
    };
    let tool = app.segment.tool;
    let pressure = app.touch.pressure();
    // Layar → W → U (8.5.2).
    let to_u = |pos: egui::Pos2| screen_to_slide_um(camera, rect, pos, transform);
    let primary = egui::PointerButton::Primary;
    match tool {
        Tool::Brush | Tool::Eraser => {
            let radius = brush_radius_um(app.segment.brush_radius_um, pressure);
            let pos = response.interact_pointer_pos();
            if response.drag_started_by(primary) {
                app.segment.stroke.clear();
            }
            if response.dragged_by(primary) {
                if let Some(u) = pos.and_then(to_u) {
                    let far = app.segment.stroke.last().is_none_or(|(p, _)| {
                        ((p[0] - u[0]).powi(2) + (p[1] - u[1]).powi(2)).sqrt()
                            >= camera.um_per_px * 1.5
                    });
                    if far {
                        app.segment.stroke.push((u, radius));
                    }
                }
            }
            let mut finish = response.drag_stopped_by(primary);
            if response.clicked_by(primary) {
                if let Some(u) = pos.and_then(to_u) {
                    app.segment.stroke = vec![(u, radius)];
                    finish = true;
                }
            }
            if finish && !app.segment.stroke.is_empty() {
                let pts = std::mem::take(&mut app.segment.stroke);
                let erase = tool == Tool::Eraser;
                match commit_stroke(sess, slide, label, &pts, erase) {
                    Ok(_) => {
                        // Hanya tile yang tersentuh yang diunggah ulang.
                        let max_r = pts.iter().map(|p| p.1).fold(0.0, f64::max);
                        let only: Vec<[f64; 2]> = pts.iter().map(|p| p.0).collect();
                        let range = sess
                            .labels_loaded(slide)
                            .and_then(|store| label_tiles_touched(store, &only, max_r));
                        match range {
                            Some(r) => app.tiles.invalidate_label_tiles(slide, r),
                            None => app.tiles.invalidate_labels(slide),
                        }
                        app.recon.needs_reload = true;
                        app.report.needs_refresh = true;
                        // Simpan otomatis per goresan (9.4).
                        if let Err(e) = sess.save_labels(slide) {
                            app.toasts.engine_error(&e);
                        }
                    }
                    Err(e) => app.toasts.engine_error(&e),
                }
            }
        }
        Tool::Lasso => {
            if response.drag_started_by(primary) {
                app.segment.polygon.clear();
            }
            if response.dragged_by(primary) {
                if let Some(u) = response.interact_pointer_pos().and_then(to_u) {
                    app.segment.polygon.push(u);
                }
            }
            if response.drag_stopped_by(primary) {
                fill_polygon(app, sess, slide, label);
            }
        }
        Tool::Polygon => {
            let (enter, esc) = ui.input(|i| {
                (
                    i.key_pressed(egui::Key::Enter),
                    i.key_pressed(egui::Key::Escape),
                )
            });
            if esc {
                app.segment.polygon.clear();
            } else if response.double_clicked_by(primary) || enter {
                fill_polygon(app, sess, slide, label);
            } else if response.clicked_by(primary) {
                if let Some(u) = response.interact_pointer_pos().and_then(to_u) {
                    app.segment.polygon.push(u);
                }
            }
        }
        Tool::Pan => {}
    }
}

fn fill_polygon(app: &mut AppState, sess: &mut Session, slide: SlideId, label: LabelId) {
    let pts = std::mem::take(&mut app.segment.polygon);
    if pts.len() < 3 {
        return;
    }
    match sess.fill_polygon(slide, label, &pts, PaintMode::Replace) {
        Ok(_) => after_edit(app, sess, slide),
        Err(e) => app.toasts.engine_error(&e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_label_kind_has_icon_and_translation() {
        for k in LabelKind::ALL {
            assert!(!kind_icon(k).codepoint.is_empty());
            for lang in [h3d_i18n::Language::Id, h3d_i18n::Language::En] {
                assert!(h3d_i18n::has_key(lang, kind_key(k)), "{}", kind_key(k));
            }
        }
    }

    #[test]
    fn nuclei_counts_follow_threshold() {
        let n = NucleiOverlay {
            slide: SlideId(0),
            points: vec![([0.0, 0.0], 0.05), ([1.0, 0.0], 0.2), ([2.0, 0.0], 0.6)],
            mpp: 1.0,
        };
        assert_eq!(n.counts(0.15), (2, 1));
        assert_eq!(n.counts(0.5), (1, 2));
        assert_eq!(n.counts(0.9), (0, 3));
    }
}
