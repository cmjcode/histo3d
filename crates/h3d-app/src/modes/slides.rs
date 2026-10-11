//! Mode `Slides` (8.11): kisi thumbnail dalam urutan tumpukan, impor,
//! seret untuk mengurutkan, mpp/Z/status/acuan per irisan, usulan urutan
//! dan orientasi dengan pratinjau Terima/Tolak (8.4.7), serta panel sampel,
//! klinis, orientasi, pewarnaan, dan pra-analitik (8.12.10).

use std::collections::BTreeSet;

use egui::Ui;
use h3d_core::{Affine2, SlideId, SlideStatus};
use h3d_engine::{OrderSuggestion, Session};
use h3d_i18n::t;

use crate::geom::{flip_suggestion, order_changes, OrderChange};
use crate::panels::browser::{BrowserPurpose, BrowserState};
use crate::panels::sample::{sample_ui, ClinicalForm, OrientationForm};
use crate::panels::slide_meta::{slide_meta_ui, SlideMetaAction, SlideMetaState};
use crate::platform;
use crate::state::{AppState, ImportRequest, JobOutput};
use crate::ui_util::{icon_text_button, icon_text_button_enabled};

/// Pratinjau usulan urutan: usulan + irisan yang dicentang untuk diterima.
#[derive(Clone, Debug)]
pub struct OrderPreview {
    /// Usulan (belum diterapkan).
    pub suggestion: OrderSuggestion,
    /// Irisan yang dicentang.
    pub accept: BTreeSet<SlideId>,
}

impl OrderPreview {
    /// Pratinjau baru; semua irisan dicentang kecuali yang ditandai ragu.
    pub fn new(suggestion: OrderSuggestion) -> Self {
        let accept = suggestion
            .order
            .iter()
            .filter(|id| !suggestion.uncertain.contains(id))
            .copied()
            .collect();
        OrderPreview { suggestion, accept }
    }

    /// Baris pratinjau: hanya irisan yang posisinya berubah.
    pub fn changes(&self, current: &[SlideId]) -> Vec<OrderChange> {
        order_changes(current, &self.suggestion)
    }

    /// "Balik urutan".
    pub fn flip(&mut self) {
        flip_suggestion(&mut self.suggestion);
    }
}

/// Keputusan pengguna atas pratinjau.
#[derive(Clone, Debug, PartialEq)]
pub enum OrderDecision {
    /// Belum ada.
    None,
    /// Tolak seluruh usulan.
    Reject,
    /// Terima irisan-irisan ini.
    Accept(Vec<SlideId>),
}

/// Keadaan mode Irisan.
#[derive(Default)]
pub struct SlidesState {
    /// Irisan terpilih.
    pub selected: Option<SlideId>,
    /// Pratinjau usulan urutan yang menunggu keputusan.
    pub order_preview: Option<OrderPreview>,
    /// Form klinis.
    pub clinical: ClinicalForm,
    /// Form orientasi.
    pub orientation: OrientationForm,
    /// Penyangga panel irisan.
    pub meta: SlideMetaState,
}

/// Pindahkan elemen `from` ke posisi `to` (indeks setelah pemindahan).
pub fn move_in_order(order: &mut Vec<SlideId>, from: usize, to: usize) -> bool {
    if from >= order.len() || to >= order.len() || from == to {
        return false;
    }
    let id = order.remove(from);
    order.insert(to, id);
    true
}

/// Mulai impor: dialog berkas di desktop, peramban Documents di iPad.
pub fn begin_import(app: &mut AppState) {
    if platform::is_mobile() {
        app.browser = Some(BrowserState::new(BrowserPurpose::ImportSlides));
    } else {
        let paths = platform::pick_slides();
        app.start_import(
            ImportRequest {
                paths,
                embed: false,
            },
            None,
        );
    }
}

/// Gambar mode Irisan.
pub fn ui(app: &mut AppState, sess: &mut Session, ui: &mut Ui) {
    egui::Panel::left("slides-left")
        .default_size(300.0)
        .show(ui, |ui| {
            egui::ScrollArea::vertical()
                .id_salt("slides-left-scroll")
                .show(ui, |ui| {
                    egui::CollapsingHeader::new(t!("slides-sample-panel"))
                        .default_open(true)
                        .show(ui, |ui| sample_ui(ui, sess, &mut app.toasts));
                    egui::CollapsingHeader::new(t!("clinical-panel"))
                        .default_open(false)
                        .show(ui, |ui| app.slides.clinical.ui(ui, sess, &mut app.toasts));
                    egui::CollapsingHeader::new(t!("anat-panel"))
                        .default_open(true)
                        .show(ui, |ui| app.slides.orientation.ui(ui, sess));
                });
        });

    egui::Panel::right("slides-right")
        .default_size(300.0)
        .show(ui, |ui| {
            egui::ScrollArea::vertical()
                .id_salt("slides-right-scroll")
                .show(ui, |ui| {
                    ui.heading(t!("slides-slice-panel"));
                    match app.slides.selected.filter(|id| sess.slide(*id).is_ok()) {
                        Some(id) => {
                            match slide_meta_ui(ui, sess, &mut app.toasts, &mut app.slides.meta, id)
                            {
                                SlideMetaAction::None => {}
                                SlideMetaAction::MppChanged => {
                                    app.tiles.forget_slide(id);
                                    app.invalidate_derived();
                                }
                                SlideMetaAction::Removed => {
                                    app.tiles.forget_slide(id);
                                    app.thumbs.forget(id);
                                    app.slides.selected = None;
                                    app.fix_current(sess);
                                    app.invalidate_derived();
                                }
                            }
                        }
                        None => {
                            ui.weak(t!("slides-select-hint"));
                        }
                    }
                });
        });

    egui::CentralPanel::default().show(ui, |ui| {
        toolbar(app, sess, ui);
        ui.separator();
        grid(app, sess, ui);
    });

    order_preview_dialog(app, sess, ui.ctx());
}

fn toolbar(app: &mut AppState, sess: &mut Session, ui: &mut Ui) {
    let order = AppState::ordered_ids(sess);
    ui.horizontal_wrapped(|ui| {
        if icon_text_button(ui, h3d_ui::icons::IMPORT, &t!("slides-import"))
            .on_hover_text(t!("slides-import-hint"))
            .clicked()
        {
            begin_import(app);
        }
        if icon_text_button_enabled(
            ui,
            order.len() >= 3,
            h3d_ui::icons::SWAP_VERT,
            &t!("slides-suggest-order"),
        )
        .clicked()
        {
            app.start_job(t!("slides-suggest-order"), |s, p| {
                s.suggest_order(p).map(JobOutput::OrderSuggested)
            });
        }
        ui.separator();
        let sel = app
            .slides
            .selected
            .and_then(|id| order.iter().position(|o| *o == id));
        let mut move_to = None;
        if ui
            .add_enabled(
                sel.is_some_and(|i| i > 0),
                egui::Button::new(h3d_ui::icons::ARROW_UP.rich_text()),
            )
            .on_hover_text(t!("slides-move-earlier"))
            .clicked()
        {
            move_to = sel.map(|i| (i, i - 1));
        }
        if ui
            .add_enabled(
                sel.is_some_and(|i| i + 1 < order.len()),
                egui::Button::new(h3d_ui::icons::ARROW_DOWN.rich_text()),
            )
            .on_hover_text(t!("slides-move-later"))
            .clicked()
        {
            move_to = sel.map(|i| (i, i + 1));
        }
        if let Some((from, to)) = move_to {
            reorder(app, sess, from, to);
        }
        ui.weak(t!("slides-drag-to-reorder"));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(t!("common-slices-count", n = order.len().to_string()));
        });
    });
    if !sess.project().z_is_monotonic() {
        h3d_ui::widgets::warning_banner(ui, &t!("slides-z-not-monotonic"));
    }
    let missing = sess.missing_slides();
    if !missing.is_empty() {
        h3d_ui::widgets::warning_banner(
            ui,
            &t!("slides-missing-files", n = missing.len().to_string()),
        );
    }
}

fn reorder(app: &mut AppState, sess: &mut Session, from: usize, to: usize) {
    let mut order = AppState::ordered_ids(sess);
    if move_in_order(&mut order, from, to) {
        if let Err(e) = sess.reorder(&order) {
            app.toasts.engine_error(&e);
        }
        app.invalidate_derived();
    }
}

/// Ukuran kartu thumbnail (point).
const CARD: egui::Vec2 = egui::vec2(150.0, 178.0);

fn grid(app: &mut AppState, sess: &mut Session, ui: &mut Ui) {
    let order = AppState::ordered_ids(sess);
    if order.is_empty() {
        ui.add_space(24.0);
        ui.vertical_centered(|ui| {
            ui.label(t!("slides-empty"));
            ui.weak(t!("slides-import-hint"));
        });
        return;
    }
    let reference = sess.project().effective_reference();
    let mut drop: Option<(usize, usize)> = None;
    egui::ScrollArea::vertical()
        .id_salt("slides-grid")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing = egui::vec2(8.0, 8.0);
                for (index, id) in order.iter().enumerate() {
                    let Ok(entry) = sess.slide(*id).cloned() else {
                        continue;
                    };
                    let (rect, resp) = ui.allocate_exact_size(CARD, egui::Sense::click_and_drag());
                    if resp.clicked() {
                        app.slides.selected = Some(*id);
                    }
                    if resp.drag_started() {
                        app.slides.selected = Some(*id);
                        resp.dnd_set_drag_payload(index);
                    }
                    if let Some(from) = resp.dnd_release_payload::<usize>() {
                        drop = Some((*from, index));
                    }
                    if !ui.is_rect_visible(rect) {
                        continue;
                    }
                    let selected = app.slides.selected == Some(*id);
                    let hovering_drop = resp.dnd_hover_payload::<usize>().is_some();
                    let painter = ui.painter_at(rect);
                    let visuals = ui.visuals();
                    let stroke = if hovering_drop {
                        egui::Stroke::new(2.0, h3d_ui::theme::WARNING)
                    } else if selected {
                        egui::Stroke::new(2.0, h3d_ui::theme::ACCENT)
                    } else {
                        visuals.widgets.noninteractive.bg_stroke
                    };
                    painter.rect(
                        rect,
                        6.0,
                        visuals.faint_bg_color,
                        stroke,
                        egui::StrokeKind::Inside,
                    );
                    let img_rect = egui::Rect::from_min_size(
                        rect.min + egui::vec2(6.0, 6.0),
                        egui::vec2(CARD.x - 12.0, CARD.x - 12.0),
                    );
                    match app.thumbs.get(ui.ctx(), sess, *id) {
                        Some((tex, size)) => {
                            let tint = if entry.status == SlideStatus::Ok {
                                egui::Color32::WHITE
                            } else {
                                egui::Color32::from_gray(110)
                            };
                            paint_oriented_thumb(
                                &painter,
                                img_rect,
                                tex,
                                size,
                                entry.orient_hint.as_ref(),
                                tint,
                            );
                        }
                        None => {
                            painter.text(
                                img_rect.center(),
                                egui::Align2::CENTER_CENTER,
                                h3d_ui::icons::IMAGE.codepoint,
                                egui::FontId::new(28.0, h3d_ui::icons::IMAGE.font_family()),
                                visuals.weak_text_color(),
                            );
                        }
                    }
                    // Lencana status dan acuan.
                    let mut badge = img_rect.right_top() + egui::vec2(-4.0, 4.0);
                    let mut put = |icon: h3d_ui::icons::MaterialIcon, color: egui::Color32| {
                        painter.circle_filled(
                            badge + egui::vec2(-9.0, 9.0),
                            11.0,
                            egui::Color32::from_black_alpha(170),
                        );
                        painter.text(
                            badge + egui::vec2(-9.0, 9.0),
                            egui::Align2::CENTER_CENTER,
                            icon.codepoint,
                            egui::FontId::new(15.0, icon.font_family()),
                            color,
                        );
                        badge.x -= 24.0;
                    };
                    if reference == Some(*id) {
                        put(h3d_ui::icons::REFERENCE, h3d_ui::theme::ACCENT);
                    }
                    match entry.status {
                        SlideStatus::Ok => {}
                        SlideStatus::Excluded => {
                            put(h3d_ui::icons::EXCLUDE, h3d_ui::theme::WARNING)
                        }
                        SlideStatus::Damaged => put(h3d_ui::icons::DAMAGED, h3d_ui::theme::ERROR),
                    }
                    if entry.reg.as_ref().is_some_and(|r| r.flagged) {
                        put(h3d_ui::icons::FLAG, h3d_ui::theme::WARNING);
                    }
                    let name = entry.source.file_name();
                    let short: String = if name.chars().count() > 18 {
                        let head: String = name.chars().take(16).collect();
                        format!("{head}..")
                    } else {
                        name
                    };
                    painter.text(
                        egui::pos2(rect.center().x, img_rect.bottom() + 5.0),
                        egui::Align2::CENTER_TOP,
                        format!("{}. {}", entry.order + 1, short),
                        egui::FontId::proportional(12.0),
                        visuals.text_color(),
                    );
                    painter.text(
                        egui::pos2(rect.center().x, img_rect.bottom() + 20.0),
                        egui::Align2::CENTER_TOP,
                        format!("z {:.0} {}", entry.z_um, t!("unit-um")),
                        egui::FontId::proportional(10.5),
                        visuals.weak_text_color(),
                    );
                }
            });
        });
    if let Some((from, to)) = drop {
        reorder(app, sess, from, to);
    }
}

/// Gambar thumbnail di dalam `rect`, diputar/dicermin menurut `orient_hint`
/// (8.4.7: thumbnail searah). Hanya bagian putar + cermin yang dipakai;
/// petunjuk orientasi bukan `T_i`.
pub fn paint_oriented_thumb(
    painter: &egui::Painter,
    rect: egui::Rect,
    texture: egui::TextureId,
    size: [u32; 2],
    orient_hint: Option<&Affine2>,
    tint: egui::Color32,
) {
    let (w, h) = (size[0].max(1) as f32, size[1].max(1) as f32);
    let (angle, mirrored) = match orient_hint {
        Some(a) => {
            let p = a.decompose();
            (p.rotation as f32, p.mirrored)
        }
        None => (0.0, false),
    };
    let (sin, cos) = angle.sin_cos();
    let corners = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)];
    let pts: Vec<egui::Vec2> = corners
        .iter()
        .map(|(cx, cy)| {
            let x = cx * w * if mirrored { -1.0 } else { 1.0 };
            let y = cy * h;
            egui::vec2(cos * x - sin * y, sin * x + cos * y)
        })
        .collect();
    let ext_x = pts.iter().map(|p| p.x.abs()).fold(1.0f32, f32::max) * 2.0;
    let ext_y = pts.iter().map(|p| p.y.abs()).fold(1.0f32, f32::max) * 2.0;
    let scale = (rect.width() / ext_x).min(rect.height() / ext_y);
    let uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
    let mut mesh = egui::Mesh::with_texture(texture);
    for (p, uv) in pts.iter().zip(uvs) {
        mesh.vertices.push(egui::epaint::Vertex {
            pos: rect.center() + *p * scale,
            uv: egui::pos2(uv.0, uv.1),
            color: tint,
        });
    }
    mesh.indices.extend([0, 1, 2, 0, 2, 3]);
    painter.add(egui::Shape::mesh(mesh));
}

/// Gambar isi pratinjau usulan; dipisah dari jendela agar bisa diuji.
pub fn order_preview_ui(ui: &mut Ui, preview: &mut OrderPreview, sess: &Session) -> OrderDecision {
    let current = AppState::ordered_ids(sess);
    let changes = preview.changes(&current);
    let mut decision = OrderDecision::None;
    ui.label(t!("slides-suggest-explain"));
    ui.horizontal(|ui| {
        if icon_text_button(ui, h3d_ui::icons::SWAP_VERT, &t!("slides-flip-order")).clicked() {
            preview.flip();
        }
        ui.weak(t!("slides-flip-hint"));
    });
    ui.separator();
    if changes.is_empty() {
        ui.label(t!("slides-suggest-no-change"));
    } else {
        egui::ScrollArea::vertical()
            .id_salt("order-preview-scroll")
            .max_height(320.0)
            .show(ui, |ui| {
                egui::Grid::new("order-preview-grid")
                    .num_columns(6)
                    .striped(true)
                    .show(ui, |ui| {
                        ui.label("");
                        crate::ui_util::strong(ui, t!("slides-file-name"));
                        crate::ui_util::strong(ui, t!("slides-suggest-from-to"));
                        crate::ui_util::strong(ui, t!("slides-orientation"));
                        crate::ui_util::strong(ui, t!("slides-suggest-confidence"));
                        ui.label("");
                        ui.end_row();
                        for c in &changes {
                            let mut on = preview.accept.contains(&c.slide);
                            if ui.checkbox(&mut on, "").changed() {
                                if on {
                                    preview.accept.insert(c.slide);
                                } else {
                                    preview.accept.remove(&c.slide);
                                }
                            }
                            let name = sess
                                .slide(c.slide)
                                .map(|s| s.source.file_name())
                                .unwrap_or_default();
                            if c.uncertain {
                                ui.colored_label(h3d_ui::theme::WARNING, name);
                            } else {
                                ui.label(name);
                            }
                            ui.label(format!("{} -> {}", c.from + 1, c.to + 1));
                            let mut orient = format!("{:.0} deg", c.rotation_deg);
                            if c.mirrored {
                                orient = format!("{orient}, {}", t!("slides-suggest-mirrored"));
                            }
                            ui.label(orient);
                            ui.label(c.confidence.map(|v| format!("{v:.2}")).unwrap_or_default());
                            if c.uncertain {
                                ui.label(crate::ui_util::icon_label_colored(
                                    h3d_ui::icons::WARNING,
                                    &t!("slides-uncertain-slice"),
                                    h3d_ui::theme::WARNING,
                                ))
                                .on_hover_text(t!("slides-uncertain-hint"));
                            } else {
                                ui.label("");
                            }
                            ui.end_row();
                        }
                    });
            });
    }
    ui.separator();
    ui.horizontal(|ui| {
        let moved: Vec<SlideId> = changes.iter().map(|c| c.slide).collect();
        if icon_text_button_enabled(
            ui,
            !moved.is_empty(),
            h3d_ui::icons::CHECK,
            &t!("slides-accept-all"),
        )
        .clicked()
        {
            decision = OrderDecision::Accept(preview.suggestion.order.clone());
        }
        let picked: Vec<SlideId> = moved
            .iter()
            .filter(|id| preview.accept.contains(id))
            .copied()
            .collect();
        if icon_text_button_enabled(
            ui,
            !picked.is_empty(),
            h3d_ui::icons::CHECK,
            &t!("slides-accept-selected"),
        )
        .clicked()
        {
            decision = OrderDecision::Accept(picked);
        }
        if icon_text_button(ui, h3d_ui::icons::CLOSE, &t!("action-reject")).clicked() {
            decision = OrderDecision::Reject;
        }
    });
    decision
}

fn order_preview_dialog(app: &mut AppState, sess: &mut Session, ctx: &egui::Context) {
    let Some(mut preview) = app.slides.order_preview.take() else {
        return;
    };
    let mut decision = OrderDecision::None;
    egui::Window::new(t!("slides-suggest-preview"))
        .id(egui::Id::new("h3d-order-preview"))
        .collapsible(false)
        .default_width(560.0)
        .show(ctx, |ui| {
            decision = order_preview_ui(ui, &mut preview, sess);
        });
    match decision {
        OrderDecision::None => app.slides.order_preview = Some(preview),
        OrderDecision::Reject => {}
        OrderDecision::Accept(ids) => {
            // Satu-satunya jalur usulan → urutan proyek (aturan keras 8).
            match sess.apply_order_suggestion(&preview.suggestion, &ids) {
                Ok(()) => {
                    if !sess.project().z_is_monotonic() {
                        app.toasts.error(t!("slides-z-not-monotonic"));
                    }
                    app.toasts.info(t!("slides-order-applied"));
                }
                Err(e) => app.toasts.engine_error(&e),
            }
            app.invalidate_derived();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(v: &[u32]) -> Vec<SlideId> {
        v.iter().map(|i| SlideId(*i)).collect()
    }

    #[test]
    fn move_in_order_is_a_permutation() {
        let mut o = ids(&[0, 1, 2, 3]);
        assert!(move_in_order(&mut o, 0, 2));
        assert_eq!(o, ids(&[1, 2, 0, 3]));
        assert!(move_in_order(&mut o, 3, 0));
        assert_eq!(o, ids(&[3, 1, 2, 0]));
        assert!(!move_in_order(&mut o, 1, 1));
        assert!(!move_in_order(&mut o, 9, 0));
        assert_eq!(o.len(), 4);
    }

    #[test]
    fn preview_preselects_confident_slides_only() {
        let s = OrderSuggestion {
            order: ids(&[2, 0, 1]),
            orient: Vec::new(),
            confidence: Vec::new(),
            uncertain: ids(&[1]),
        };
        let mut p = OrderPreview::new(s);
        assert!(p.accept.contains(&SlideId(2)) && p.accept.contains(&SlideId(0)));
        assert!(!p.accept.contains(&SlideId(1)));
        let current = ids(&[0, 1, 2]);
        assert_eq!(p.changes(&current).len(), 3);
        p.flip();
        // 1, 0, 2 → hanya irisan 0 dan 1 yang pindah.
        assert_eq!(p.changes(&current).len(), 2);
    }
}
