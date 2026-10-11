//! Mode `Register` (8.11, 8.4.5): viewer pembanding irisan `i` terhadap
//! tetangganya ke arah acuan (campur / papan catur / selisih), daftar QC
//! dengan irisan bertanda di atas, "Sejajarkan semua" (bisa dilanjutkan),
//! alat landmark, pegangan geser/putar di layar, dan tandai rusak.

use egui::Ui;
use h3d_core::{SlideId, SlideStatus, SlideTransform};
use h3d_engine::{QcRow, RegOptions, Session};
use h3d_i18n::t;
use h3d_render::{CompareMode, CompareParams, H3dPaintCallback, Scene, Scene2D};

use crate::geom::{
    manual_adjusted_transform, navigate, neighbor_toward_reference, LandmarkCollector, LandmarkSide,
};
use crate::gpu::Gpu;
use crate::state::{AppState, JobOutput};
use crate::ui_util::{icon_selectable, icon_text_button, icon_text_button_enabled};
use crate::viewer::{nav_bar, paint_background, slide_layer, Viewer2D};

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

/// Alat aktif mode Registrasi.
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
    }

    /// Ada geser/putar manual yang belum diterapkan.
    pub fn has_manual(&self) -> bool {
        self.shift != [0.0, 0.0] || self.angle != 0.0
    }
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

/// Gambar mode Registrasi.
pub fn ui(app: &mut AppState, sess: &mut Session, ui: &mut Ui, gpu: Option<&Gpu>) {
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

    egui::Panel::left("register-left")
        .default_size(330.0)
        .show(ui, |ui| qc_panel(app, sess, ui));

    egui::CentralPanel::default().show(ui, |ui| {
        toolbar(app, sess, ui);
        ui.separator();
        viewer(app, sess, ui, gpu);
    });
}

fn fmt_opt(v: Option<f32>) -> String {
    v.map(|x| format!("{x:.2}")).unwrap_or_else(|| "-".into())
}

fn qc_panel(app: &mut AppState, sess: &mut Session, ui: &mut Ui) {
    ui.heading(t!("register-qc-list"));
    let flagged = app.register.qc.iter().filter(|r| r.flagged).count();
    if flagged > 0 {
        h3d_ui::widgets::warning_banner(ui, &t!("register-flagged-count", n = flagged.to_string()));
    }
    let needs = sess.needs_registration().len();
    ui.horizontal(|ui| {
        if icon_text_button(ui, h3d_ui::icons::PLAY, &t!("register-all")).clicked() {
            app.start_job(t!("register-all"), |s, p| {
                s.register_all(&RegOptions::default(), p)
                    .map(JobOutput::Registered)
            });
        }
        if needs > 0 {
            ui.weak(t!("register-pending-count", n = needs.to_string()));
        }
    });
    ui.small(t!("register-resumable-hint"));
    ui.separator();
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

fn toolbar(app: &mut AppState, sess: &mut Session, ui: &mut Ui) {
    let order = AppState::ordered_ids(sess);
    let index = app
        .current
        .and_then(|c| order.iter().position(|o| *o == c))
        .unwrap_or(0);
    ui.horizontal_wrapped(|ui| {
        if let Some(dir) = nav_bar(ui, index, order.len()) {
            app.current = navigate(&order, app.current, dir);
        }
        ui.separator();
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
    });
    ui.horizontal_wrapped(|ui| {
        let st = &mut app.register;
        for (tool, icon, key) in [
            (RegTool::View, h3d_ui::icons::PAN, "tool-pan"),
            (
                RegTool::Landmark,
                h3d_ui::icons::LANDMARK,
                "register-landmarks",
            ),
            (
                RegTool::Translate,
                h3d_ui::icons::MOVE,
                "register-translate",
            ),
            (RegTool::Rotate, h3d_ui::icons::ROTATE, "register-rotate"),
        ] {
            if icon_selectable(ui, st.tool == tool, icon, &t!(key)).clicked() {
                st.tool = tool;
            }
        }
        ui.separator();
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
                        let t_new = manual_adjusted_transform(&t_old, st.shift, st.angle, pivot);
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

fn viewer(app: &mut AppState, sess: &mut Session, ui: &mut Ui, gpu: Option<&Gpu>) {
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
    let t_moving = manual_adjusted_transform(&t_old, st.shift, st.angle, pivot);
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
    let t_moving = manual_adjusted_transform(&t_old, st.shift, st.angle, pivot);

    // Adegan.
    let moving = slide_layer(&mut app.tiles, sess, gpu, slide, &t_moving, &frame.view);
    let scene = match (neighbor, t_fixed) {
        (Some(n), Some(tf)) => {
            let fixed = slide_layer(&mut app.tiles, sess, gpu, n, &tf, &frame.view);
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
