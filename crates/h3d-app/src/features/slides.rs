//! Fitur Irisan: isi laci "Lapisan" (daftar irisan dalam urutan tumpukan
//! dengan thumbnail, lencana status/QC, mata untuk menyembunyikan di 3D,
//! seret untuk mengurutkan, klik ganda membuka 2D; daftar label ringkas),
//! impor, dan pratinjau usulan urutan/orientasi dengan Terima/Tolak (8.4.7).
//! Panel sampel, klinis, dan orientasi (8.12.10) dipakai laci "Sampel".

use std::collections::BTreeSet;

use egui::Ui;
use h3d_core::{Affine2, SlideId, SlideStatus};
use h3d_engine::{OrderSuggestion, Session};
use h3d_i18n::t;

use crate::geom::{flip_suggestion, order_changes, OrderChange};
use crate::panels::browser::{BrowserPurpose, BrowserState};
use crate::panels::sample::{ClinicalForm, OrientationForm};
use crate::panels::slide_meta::SlideMetaState;
use crate::platform;
use crate::state::{AppState, ImportRequest};
use crate::ui_util::{icon_text_button, icon_text_button_enabled};

/// Pratinjau usulan urutan: usulan + irisan yang dicentang untuk diterima.
#[derive(Clone, Debug)]
pub struct OrderPreview {
    /// Usulan (belum diterapkan).
    pub suggestion: OrderSuggestion,
    /// Irisan yang dicentang.
    pub accept: BTreeSet<SlideId>,
    /// Tabel rincian perubahan dibuka.
    pub details_open: bool,
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
        OrderPreview {
            suggestion,
            accept,
            details_open: false,
        }
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

/// Kejadian dari laci Lapisan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayersEvent {
    /// Tidak ada.
    None,
    /// Irisan diminta dibuka di 2D.
    OpenSlice(SlideId),
}

/// Keadaan fitur Irisan.
#[derive(Default)]
pub struct SlidesState {
    /// Pratinjau usulan urutan yang menunggu keputusan.
    pub order_preview: Option<OrderPreview>,
    /// Form klinis.
    pub clinical: ClinicalForm,
    /// Form orientasi.
    pub orientation: OrientationForm,
    /// Penyangga panel irisan.
    pub meta: SlideMetaState,
    /// Irisan yang berkasnya hilang, diperiksa ulang tiap
    /// [`MISSING_RECHECK_SECS`] (bukan `stat` tiap frame).
    pub missing: Option<(std::time::Instant, Vec<SlideId>)>,
}

/// Selang pemeriksaan ulang berkas slide yang hilang (detik).
pub const MISSING_RECHECK_SECS: u64 = 2;

impl SlidesState {
    /// Irisan yang berkasnya hilang (hasil di-cache beberapa detik).
    pub fn missing_slides(&mut self, sess: &Session) -> Vec<SlideId> {
        let fresh = self
            .missing
            .as_ref()
            .filter(|(t, _)| t.elapsed() < std::time::Duration::from_secs(MISSING_RECHECK_SECS));
        if let Some((_, v)) = fresh {
            return v.clone();
        }
        let v = sess.missing_slides();
        self.missing = Some((std::time::Instant::now(), v.clone()));
        v
    }
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
                embed_mode: h3d_engine::EmbedMode::Copy,
            },
            None,
        );
    }
}

/// Terapkan pemindahan `from → to` ke urutan proyek.
pub fn reorder(app: &mut AppState, sess: &mut Session, from: usize, to: usize) {
    let mut order = AppState::ordered_ids(sess);
    if move_in_order(&mut order, from, to) {
        if let Err(e) = sess.reorder(&order) {
            app.toasts.engine_error(&e);
        }
        app.invalidate_derived();
        app.recon.fitted = false;
    }
}

/// Sisi thumbnail di baris laci (point).
const ROW_THUMB: f32 = 44.0;

/// Isi laci "Lapisan".
pub fn layers_drawer(app: &mut AppState, sess: &mut Session, ui: &mut Ui) -> LayersEvent {
    let mut event = LayersEvent::None;
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
            order.len() >= crate::state::AUTO_STACK_MIN_SLIDES && app.job.is_none(),
            h3d_ui::icons::AUTO_STACK,
            &t!("stack-auto"),
        )
        .on_hover_text(t!("stack-auto-hint"))
        .clicked()
        {
            app.start_auto_stack(false);
        }
    });
    if !sess.project().z_is_monotonic() {
        h3d_ui::widgets::warning_banner(ui, &t!("slides-z-not-monotonic"));
    }
    let missing = app.slides.missing_slides(sess);
    if !missing.is_empty() {
        h3d_ui::widgets::warning_banner(
            ui,
            &t!("slides-missing-files", n = missing.len().to_string()),
        );
    }

    ui.horizontal(|ui| {
        h3d_ui::widgets::section_title(ui, &t!("layers-slices"));
        h3d_ui::widgets::count_badge(ui, order.len(), h3d_ui::theme::ACCENT);
    });
    if order.is_empty() {
        ui.weak(t!("slides-empty"));
        ui.small(t!("slides-import-hint"));
    } else {
        ui.small(t!("layers-reorder-hint"));
    }

    let reference = sess.project().effective_reference();
    let pending: BTreeSet<SlideId> = sess.pending_slides().into_iter().collect();
    let mut drop: Option<(usize, usize)> = None;
    let mut select: Option<SlideId> = None;
    let mut toggle_hidden: Option<SlideId> = None;
    egui::ScrollArea::vertical()
        .id_salt("layers-slices")
        .max_height(ui.available_height() * 0.62)
        .auto_shrink([false, true])
        .show(ui, |ui| {
            for (index, id) in order.iter().enumerate() {
                let Ok(entry) = sess.slide(*id).cloned() else {
                    continue;
                };
                let selected = app.current == Some(*id);
                let hidden = app.recon.hidden_slices.contains(id);
                let row_h = ROW_THUMB + 12.0;
                let (rect, resp) = ui.allocate_exact_size(
                    egui::vec2(ui.available_width(), row_h),
                    egui::Sense::click_and_drag(),
                );
                if resp.double_clicked() {
                    event = LayersEvent::OpenSlice(*id);
                } else if resp.clicked() {
                    select = Some(*id);
                }
                if resp.drag_started() {
                    select = Some(*id);
                    resp.dnd_set_drag_payload(index);
                }
                if let Some(from) = resp.dnd_release_payload::<usize>() {
                    drop = Some((*from, index));
                }
                if !ui.is_rect_visible(rect) {
                    continue;
                }
                let hovering_drop = resp.dnd_hover_payload::<usize>().is_some();
                let visuals = ui.visuals().clone();
                let stroke = if hovering_drop {
                    egui::Stroke::new(2.0, h3d_ui::theme::WARNING)
                } else if selected {
                    egui::Stroke::new(1.5, h3d_ui::theme::ACCENT)
                } else {
                    egui::Stroke::NONE
                };
                let fill = if selected {
                    h3d_ui::theme::ACCENT.gamma_multiply(0.18)
                } else if resp.hovered() {
                    visuals.widgets.hovered.bg_fill
                } else {
                    h3d_ui::theme::card_frame(&visuals).fill
                };
                let painter = ui.painter_at(rect);
                painter.rect(
                    rect.shrink(1.0),
                    6.0,
                    fill,
                    stroke,
                    egui::StrokeKind::Inside,
                );

                // Thumbnail searah `orient_hint`.
                let img_rect = egui::Rect::from_min_size(
                    rect.min + egui::vec2(6.0, 6.0),
                    egui::vec2(ROW_THUMB, ROW_THUMB),
                );
                let tint = if entry.status == SlideStatus::Ok && !hidden {
                    egui::Color32::WHITE
                } else {
                    egui::Color32::from_gray(110)
                };
                match app.thumbs.get(ui.ctx(), sess, &mut app.thumb_worker, *id) {
                    Some((tex, size)) => paint_oriented_thumb(
                        &painter,
                        img_rect,
                        tex,
                        size,
                        entry.orient_hint.as_ref(),
                        tint,
                    ),
                    None => {
                        painter.text(
                            img_rect.center(),
                            egui::Align2::CENTER_CENTER,
                            h3d_ui::icons::IMAGE.codepoint,
                            egui::FontId::new(22.0, h3d_ui::icons::IMAGE.font_family()),
                            visuals.weak_text_color(),
                        );
                    }
                }

                // Teks: nomor + nama, lalu z dan lencana.
                let text_x = img_rect.right() + 8.0;
                let name = entry.source.file_name();
                let max_chars = 26;
                let short: String = if name.chars().count() > max_chars {
                    let head: String = name.chars().take(max_chars - 2).collect();
                    format!("{head}..")
                } else {
                    name
                };
                let text_color = if hidden {
                    visuals.weak_text_color()
                } else {
                    visuals.text_color()
                };
                painter.text(
                    egui::pos2(text_x, rect.top() + 10.0),
                    egui::Align2::LEFT_TOP,
                    format!("{}. {}", entry.order + 1, short),
                    egui::FontId::proportional(12.5),
                    text_color,
                );
                painter.text(
                    egui::pos2(text_x, rect.top() + 28.0),
                    egui::Align2::LEFT_TOP,
                    t!("context-slice-z", z = format!("{:.0}", entry.z_um)),
                    egui::FontId::proportional(10.5),
                    visuals.weak_text_color(),
                );
                let mut badges: Vec<(h3d_ui::icons::MaterialIcon, egui::Color32, String)> =
                    Vec::new();
                if reference == Some(*id) {
                    badges.push((
                        h3d_ui::icons::REFERENCE,
                        h3d_ui::theme::ACCENT,
                        t!("slides-is-reference"),
                    ));
                }
                match entry.status {
                    SlideStatus::Ok => {}
                    SlideStatus::Excluded => badges.push((
                        h3d_ui::icons::EXCLUDE,
                        h3d_ui::theme::WARNING,
                        t!("slides-status-excluded"),
                    )),
                    SlideStatus::Damaged => badges.push((
                        h3d_ui::icons::DAMAGED,
                        h3d_ui::theme::ERROR,
                        t!("slides-status-damaged"),
                    )),
                }
                if entry.reg.as_ref().is_some_and(|r| r.flagged) {
                    badges.push((
                        h3d_ui::icons::FLAG,
                        h3d_ui::theme::WARNING,
                        t!("layers-qc-flagged"),
                    ));
                } else if entry.status == SlideStatus::Ok
                    && entry.reg.is_none()
                    && reference != Some(*id)
                {
                    badges.push((
                        h3d_ui::icons::COMPARE,
                        visuals.weak_text_color(),
                        t!("layers-not-registered"),
                    ));
                }
                if pending.contains(id) {
                    badges.push((
                        h3d_ui::icons::LAYERS,
                        h3d_ui::theme::WARNING,
                        t!("layers-pending"),
                    ));
                }
                for (k, (icon, color, tip)) in badges.into_iter().enumerate() {
                    let c = egui::pos2(text_x + 7.0 + 18.0 * k as f32, rect.top() + 48.0);
                    painter.text(
                        c,
                        egui::Align2::CENTER_CENTER,
                        icon.codepoint,
                        egui::FontId::new(13.0, icon.font_family()),
                        color,
                    );
                    let r = egui::Rect::from_center_size(c, egui::vec2(16.0, 16.0));
                    ui.interact(r, ui.id().with(("badge", id.0, k)), egui::Sense::hover())
                        .on_hover_text(tip);
                }

                // Mata: sembunyikan dari tumpukan 3D (tampilan saja).
                let eye_rect = egui::Rect::from_center_size(
                    egui::pos2(rect.right() - 18.0, rect.center().y),
                    egui::vec2(26.0, 26.0),
                );
                let eye = ui.interact(eye_rect, ui.id().with(("eye", id.0)), egui::Sense::click());
                let icon = if hidden {
                    h3d_ui::icons::VISIBILITY_OFF
                } else {
                    h3d_ui::icons::VISIBILITY_ON
                };
                painter.text(
                    eye_rect.center(),
                    egui::Align2::CENTER_CENTER,
                    icon.codepoint,
                    egui::FontId::new(16.0, icon.font_family()),
                    if eye.hovered() {
                        visuals.text_color()
                    } else {
                        visuals.weak_text_color()
                    },
                );
                if eye
                    .on_hover_text(if hidden {
                        t!("stack-slice-shown")
                    } else {
                        t!("stack-slice-hidden")
                    })
                    .clicked()
                {
                    toggle_hidden = Some(*id);
                }
            }
        });
    if let Some(id) = select {
        app.current = Some(id);
    }
    if let Some(id) = toggle_hidden {
        if !app.recon.hidden_slices.remove(&id) {
            app.recon.hidden_slices.insert(id);
        }
    }
    if let Some((from, to)) = drop {
        reorder(app, sess, from, to);
    }

    // Label ringkas: warna, nama, tampil, dibuat mesh. Penyuntingan lengkap di
    // laci "Label dan usulan".
    ui.separator();
    ui.horizontal(|ui| {
        h3d_ui::widgets::section_title(ui, &t!("layers-labels"));
        h3d_ui::widgets::count_badge(ui, sess.project().labels.len(), h3d_ui::theme::ACCENT);
    });
    let labels = sess.project().labels.clone();
    let mut update = None;
    for def in &labels {
        let mut d = def.clone();
        let mut changed = false;
        ui.horizontal(|ui| {
            let active = app.segment.active_label == Some(def.id);
            if ui
                .radio(active, "")
                .on_hover_text(t!("label-panel"))
                .clicked()
            {
                app.segment.active_label = Some(def.id);
            }
            h3d_ui::widgets::color_swatch(ui, def.color);
            ui.label(crate::features::segment::kind_icon(def.kind).rich_text())
                .on_hover_text(t!(crate::features::segment::kind_key(def.kind)));
            ui.label(&def.name);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                changed |= h3d_ui::widgets::toggle_icon_button(
                    ui,
                    &mut d.reconstruct,
                    h3d_ui::icons::CUBE,
                    &t!("label-reconstruct"),
                )
                .clicked();
                let vis_icon = if d.visible {
                    h3d_ui::icons::VISIBILITY_ON
                } else {
                    h3d_ui::icons::VISIBILITY_OFF
                };
                if h3d_ui::widgets::icon_button(ui, vis_icon, &t!("label-visible")).clicked() {
                    d.visible = !d.visible;
                    changed = true;
                }
            });
        });
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
    event
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

/// Isi kartu usulan susunan (judul, ringkasan, Balik, rincian yang bisa
/// dibuka, Terima semua / Terima dicentang / Tolak). Dipisah dari kartu agar
/// bisa diuji tanpa jendela.
pub fn order_preview_ui(ui: &mut Ui, preview: &mut OrderPreview, sess: &Session) -> OrderDecision {
    let current = AppState::ordered_ids(sess);
    let changes = preview.changes(&current);
    let uncertain = changes.iter().filter(|c| c.uncertain).count();
    let mut decision = OrderDecision::None;
    ui.horizontal(|ui| {
        ui.label(
            h3d_ui::icons::AUTO_STACK
                .rich_text()
                .size(h3d_ui::theme::ICON_SIZE_DEFAULT)
                .color(h3d_ui::theme::ACCENT),
        );
        crate::ui_util::strong(ui, t!("stack-proposal-title"));
        ui.weak(t!(
            "stack-proposal-summary",
            n = changes.len().to_string(),
            uncertain = uncertain.to_string()
        ));
    });
    ui.small(t!("stack-proposal-preview-hint"));
    ui.small(t!("slides-suggest-explain"));
    ui.horizontal_wrapped(|ui| {
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
            !picked.is_empty() && picked.len() != moved.len(),
            h3d_ui::icons::CHECK,
            &t!("slides-accept-selected"),
        )
        .clicked()
        {
            decision = OrderDecision::Accept(picked);
        }
        if icon_text_button(ui, h3d_ui::icons::SWAP_VERT, &t!("slides-flip-order"))
            .on_hover_text(t!("slides-flip-hint"))
            .clicked()
        {
            preview.flip();
        }
        if icon_text_button(ui, h3d_ui::icons::CLOSE, &t!("action-reject")).clicked() {
            decision = OrderDecision::Reject;
        }
        ui.toggle_value(&mut preview.details_open, t!("stack-proposal-details"));
    });
    if !preview.details_open {
        return decision;
    }
    ui.separator();
    if changes.is_empty() {
        ui.label(t!("slides-suggest-no-change"));
        return decision;
    }
    egui::ScrollArea::vertical()
        .id_salt("order-preview-scroll")
        .max_height(220.0)
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
    decision
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
        assert!(!p.details_open);
    }
}
