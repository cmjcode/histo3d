//! Viewer 2D bersama (mode Registrasi dan Segmentasi): kamera, masukan
//! geser/zoom (tetikus, sentuh), pelacak Pencil/jari, navigasi irisan, dan
//! penyusunan lapisan tile lewat [`crate::tiles::TileStreamer`].
//!
//! Masukan (8.11): desktop — roda = zoom, seret tombol tengah/kanan =
//! geser; iPad — Pencil menggambar, jari menggeser/zoom (cubit), geser dua
//! jari mendatar = pindah irisan.

use egui::{Rect, Response, Sense, Ui};
use h3d_core::{Affine2, SlideId};
use h3d_engine::{LabelStore, Session};
use h3d_render::{Camera2D, TileLayerDraw};

use crate::geom::{NavDir, SwipeDetector};
use crate::gpu::{Gpu, GpuSink};
use crate::tiles::{TileSink, TileStreamer, ViewW};

/// Jenis penunjuk yang sedang aktif.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointerKind {
    /// Tetikus/trackpad.
    Mouse,
    /// Jari.
    Finger,
    /// Apple Pencil (sentuhan bertekanan).
    Pencil,
}

/// Pelacak sentuhan: membedakan Pencil dari jari lewat tekanan
/// (`egui::Event::Touch { force, .. }`; jari melaporkan `None`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TouchTracker {
    /// Ada sentuhan (jari atau Pencil) yang sedang menempel.
    pub touch_active: bool,
    /// Sentuhan yang menempel adalah Pencil.
    pub pencil_active: bool,
    /// Tekanan terakhir Pencil (0..1).
    pub force: Option<f32>,
}

impl TouchTracker {
    /// Amati event frame ini. `any_touches` dari `InputState::any_touches`.
    pub fn observe(&mut self, events: &[egui::Event], any_touches: bool) {
        for ev in events {
            if let egui::Event::Touch { phase, force, .. } = ev {
                match phase {
                    egui::TouchPhase::Start | egui::TouchPhase::Move => {
                        self.touch_active = true;
                        if let Some(f) = force.filter(|f| *f > 0.0) {
                            self.pencil_active = true;
                            self.force = Some(f);
                        }
                    }
                    egui::TouchPhase::End | egui::TouchPhase::Cancel => {}
                }
            }
        }
        if !any_touches {
            self.touch_active = false;
            self.pencil_active = false;
            self.force = None;
        }
    }

    /// Jenis penunjuk saat ini.
    pub fn kind(&self) -> PointerKind {
        if self.pencil_active {
            PointerKind::Pencil
        } else if self.touch_active {
            PointerKind::Finger
        } else {
            PointerKind::Mouse
        }
    }

    /// Penunjuk saat ini boleh menggambar: tetikus dan Pencil selalu; jari
    /// hanya bila mode "jari menggambar" menyala.
    pub fn may_draw(&self, finger_draw: bool) -> bool {
        match self.kind() {
            PointerKind::Mouse | PointerKind::Pencil => true,
            PointerKind::Finger => finger_draw,
        }
    }

    /// Tekanan untuk kuas: hanya Pencil yang melaporkan tekanan.
    pub fn pressure(&self) -> Option<f32> {
        if self.pencil_active {
            self.force
        } else {
            None
        }
    }
}

/// Hasil satu frame viewer.
pub struct ViewerFrame {
    /// Persegi viewport (point egui).
    pub rect: Rect,
    /// Respons interaksi viewport (klik + seret).
    pub response: Response,
    /// Permintaan pindah irisan (panah, geser dua jari).
    pub nav: Option<NavDir>,
    /// Kotak pandang W untuk aliran tile.
    pub view: ViewW,
    /// Gestur dua jari sedang berlangsung (alat gambar harus diam).
    pub multi_touch: bool,
}

/// Viewer 2D: kamera + keadaan gestur.
#[derive(Clone, Debug, Default)]
pub struct Viewer2D {
    /// Kamera (µm, ruang W).
    pub camera: Camera2D,
    fitted_key: Option<u64>,
    swipe: SwipeDetector,
    gesture_start: Option<Camera2D>,
}

impl Viewer2D {
    /// Paksa "pas ke layar" pada frame berikutnya.
    pub fn request_fit(&mut self) {
        self.fitted_key = None;
    }

    /// Alokasikan viewport di sisa `ui`, pasang kamera ke `fit` (kotak W)
    /// saat `fit_key` berubah, lalu olah geser/zoom/navigasi.
    /// `primary_pans`: seret tombol utama (atau satu jari) menggeser.
    pub fn begin(
        &mut self,
        ui: &mut Ui,
        fit_key: u64,
        fit: Option<([f64; 2], [f64; 2])>,
        primary_pans: bool,
    ) -> ViewerFrame {
        let size = ui.available_size().max(egui::vec2(64.0, 64.0));
        let (rect, response) = ui.allocate_exact_size(size, Sense::click_and_drag());
        if self.fitted_key != Some(fit_key) {
            if let Some((min, max)) = fit {
                if max[0] > min[0] && max[1] > min[1] {
                    self.camera.fit_rect_um(min, max, rect.shrink(12.0));
                    self.fitted_key = Some(fit_key);
                }
            }
        }

        let mut nav = None;
        let ctx = ui.ctx().clone();
        let hovered = response.contains_pointer();
        let multi = ctx.input(|i| i.multi_touch());

        // Dua jari: cubit = zoom, geser = pan; geser mendatar tanpa cubit =
        // pindah irisan (kamera dikembalikan ke awal gestur).
        match multi {
            Some(mt) if mt.num_touches >= 2 => {
                if self.gesture_start.is_none() {
                    self.gesture_start = Some(self.camera);
                }
                self.camera.pan(mt.translation_delta);
                if (mt.zoom_delta - 1.0).abs() > 1e-4 {
                    self.camera
                        .zoom_at(f64::from(mt.zoom_delta), mt.center_pos, rect);
                }
                self.swipe
                    .update(Some((mt.translation_delta, mt.zoom_delta)));
            }
            _ => {
                if let Some(dir) = self.swipe.update(None) {
                    if let Some(start) = self.gesture_start {
                        self.camera = start;
                    }
                    nav = Some(dir);
                }
                self.gesture_start = None;
                if hovered {
                    let (scroll, zoom, pos) = ctx.input(|i| {
                        (
                            i.smooth_scroll_delta.y,
                            i.zoom_delta(),
                            i.pointer.hover_pos(),
                        )
                    });
                    let pos = pos.unwrap_or(rect.center());
                    if (zoom - 1.0).abs() > 1e-4 {
                        self.camera.zoom_at(f64::from(zoom), pos, rect);
                    } else if scroll.abs() > 0.0 {
                        self.camera
                            .zoom_at(f64::from((scroll * 0.004).exp()), pos, rect);
                    }
                }
                let pan = response.dragged_by(egui::PointerButton::Middle)
                    || response.dragged_by(egui::PointerButton::Secondary)
                    || (primary_pans && response.dragged_by(egui::PointerButton::Primary));
                if pan {
                    self.camera.pan(response.drag_delta());
                }
            }
        }

        // Tombol panah (bila tidak sedang mengetik).
        if !ctx.egui_wants_keyboard_input() {
            ctx.input(|i| {
                if i.key_pressed(egui::Key::ArrowLeft) {
                    nav = Some(NavDir::Prev);
                } else if i.key_pressed(egui::Key::ArrowRight) {
                    nav = Some(NavDir::Next);
                }
            });
        }

        ViewerFrame {
            rect,
            response,
            nav,
            view: ViewW::from_camera(&self.camera, rect, ctx.pixels_per_point()),
            multi_touch: self.swipe.is_active(),
        }
    }
}

/// Susun lapisan tile gambar irisan (`t` = `T_i`, U → W). Dengan GPU, tile
/// diunggah dalam kuota frame; tanpa GPU hanya permintaan dekode yang jalan.
pub fn slide_layer(
    tiles: &mut TileStreamer,
    session: &Session,
    gpu: Option<&Gpu>,
    slide: SlideId,
    t: &Affine2,
    view: &ViewW,
) -> TileLayerDraw {
    tiles.ensure_source(session, slide);
    let draws = match gpu {
        Some(g) => g
            .with(|renderer, device, queue| {
                let mut sink = GpuSink {
                    renderer,
                    device,
                    queue,
                };
                tiles.layer_tiles(Some(&mut sink as &mut dyn TileSink), slide, t, view)
            })
            .unwrap_or_default(),
        None => tiles.layer_tiles(None, slide, t, view),
    };
    TileLayerDraw {
        affine: *t,
        tiles: draws,
        opacity: 1.0,
    }
}

/// Susun tile overlay label irisan dari `store` (`t` = `T_i`, U → W).
pub fn label_layer(
    tiles: &mut TileStreamer,
    gpu: Option<&Gpu>,
    slide: SlideId,
    store: &LabelStore,
    t: &Affine2,
    view: &ViewW,
) -> Vec<h3d_render::TileDraw> {
    match gpu {
        Some(g) => g
            .with(|renderer, device, queue| {
                let mut sink = GpuSink {
                    renderer,
                    device,
                    queue,
                };
                tiles.label_tiles(Some(&mut sink as &mut dyn TileSink), slide, store, t, view)
            })
            .unwrap_or_default(),
        None => tiles.label_tiles(None, slide, store, t, view),
    }
}

/// Latar viewport dan, tanpa GPU, teks pengganti (tes headless).
pub fn paint_background(ui: &Ui, rect: Rect, has_gpu: bool) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, ui.visuals().extreme_bg_color);
    if !has_gpu {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            h3d_i18n::t!("viewer-no-gpu"),
            egui::FontId::proportional(14.0),
            ui.visuals().weak_text_color(),
        );
    }
}

/// Bilah navigasi irisan: sebelumnya / "i dari n" / berikutnya.
pub fn nav_bar(ui: &mut Ui, index: usize, count: usize) -> Option<NavDir> {
    let mut nav = None;
    ui.horizontal(|ui| {
        if ui
            .add_enabled(
                index > 0,
                egui::Button::new(h3d_ui::icons::PREV.rich_text()),
            )
            .on_hover_text(h3d_i18n::t!("action-prev-slice"))
            .clicked()
        {
            nav = Some(NavDir::Prev);
        }
        ui.label(h3d_i18n::t!(
            "nav-slice-of",
            index = (index + 1).to_string(),
            count = count.to_string()
        ));
        if ui
            .add_enabled(
                index + 1 < count,
                egui::Button::new(h3d_ui::icons::NEXT.rich_text()),
            )
            .on_hover_text(h3d_i18n::t!("action-next-slice"))
            .clicked()
        {
            nav = Some(NavDir::Next);
        }
    });
    nav
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(phase: egui::TouchPhase, force: Option<f32>) -> egui::Event {
        egui::Event::Touch {
            device_id: egui::TouchDeviceId(0),
            id: egui::TouchId(1),
            phase,
            pos: egui::pos2(10.0, 10.0),
            force,
        }
    }

    #[test]
    fn pencil_is_told_apart_from_finger_by_force() {
        let mut t = TouchTracker::default();
        assert_eq!(t.kind(), PointerKind::Mouse);
        assert!(t.may_draw(false));

        // Jari: tanpa tekanan → menggeser, kecuali mode jari-menggambar.
        t.observe(&[touch(egui::TouchPhase::Start, None)], true);
        assert_eq!(t.kind(), PointerKind::Finger);
        assert!(!t.may_draw(false));
        assert!(t.may_draw(true));
        assert_eq!(t.pressure(), None);
        t.observe(&[touch(egui::TouchPhase::End, None)], false);
        assert_eq!(t.kind(), PointerKind::Mouse);

        // Pencil: bertekanan → selalu menggambar, tekanan terbaca.
        t.observe(&[touch(egui::TouchPhase::Start, Some(0.4))], true);
        assert_eq!(t.kind(), PointerKind::Pencil);
        assert!(t.may_draw(false));
        t.observe(&[touch(egui::TouchPhase::Move, Some(0.8))], true);
        assert_eq!(t.pressure(), Some(0.8));
        let r = crate::geom::brush_radius_um(100.0, t.pressure());
        assert!((r - 86.0).abs() < 1e-3);
        t.observe(&[], false);
        assert_eq!(t.pressure(), None);
    }
}
