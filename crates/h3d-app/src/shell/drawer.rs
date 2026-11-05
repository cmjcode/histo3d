//! Laci kanan mengambang (pola `ducad-ui/src/items_drawer.rs`): satu laci
//! terbuka pada satu waktu, lebar tetap, menggulir.
//!
//! Pemetaan fitur lama:
//! - Lapisan ← kisi thumbnail, urutkan, tandai status, label (mode Irisan).
//! - Sampel ← panel Sampel, Klinis, Orientasi anatomis, dan panel per irisan
//!   (mpp, Z, status, acuan, pewarnaan, pra-analitik, IHC) dari mode Irisan.
//! - Label dan usulan ← panel label, usulan klasik/AI/interpolasi, kalibrasi
//!   IHC (mode Segmentasi).
//! - Penyejajaran ← daftar QC dan Sejajarkan semua (mode Registrasi).
//! - Tumor 3D ← parameter, visibilitas/opasitas, bidang potong, irisan di 3D,
//!   lapisan skalar, titik terpilih (mode Rekonstruksi).
//! - Deteksi AI ← model ML di bundel, cakupan, jalankan, tinjau usulan.

use egui::{Rect, Ui};
use h3d_core::SlideId;
use h3d_engine::Session;
use h3d_i18n::t;

use crate::features::{detect, reconstruct, register, segment, slides};
use crate::panels::sample::sample_ui;
use crate::panels::slide_meta::{slide_meta_ui, SlideMetaAction};
use crate::state::{AppState, Drawer};

/// Gambar laci yang terbuka di sisi kanan `area`. Mengembalikan lebar yang
/// dipakai (0 bila tertutup) agar panel lain tidak tumpang tindih.
pub fn show(app: &mut AppState, sess: Option<&mut Session>, ui: &mut Ui, area: Rect) -> f32 {
    let Some(drawer) = app.drawer else {
        return 0.0;
    };
    let margin = h3d_ui::theme::FLOAT_MARGIN;
    let width = h3d_ui::theme::DRAWER_WIDTH.min(area.width() * 0.6);
    let rect = Rect::from_min_max(
        egui::pos2(area.right() - margin - width, area.top() + margin),
        egui::pos2(area.right() - margin, area.bottom() - margin),
    );
    let mut close = false;
    let mut open_slice = None;
    egui::Area::new(egui::Id::new("h3d-drawer"))
        .order(egui::Order::Middle)
        .fixed_pos(rect.min)
        .show(ui.ctx(), |ui| {
            ui.set_width(width);
            ui.set_max_height(rect.height());
            h3d_ui::theme::panel_frame(ui.visuals()).show(ui, |ui| {
                ui.set_width(width - 20.0);
                ui.set_min_height(rect.height() - 12.0);
                close = h3d_ui::widgets::drawer_header(ui, drawer.icon(), &t!(drawer.i18n_key()));
                egui::ScrollArea::vertical()
                    .id_salt(("drawer", drawer as u8))
                    .max_height(rect.height() - 52.0)
                    .auto_shrink([false, false])
                    .show(ui, |ui| match sess {
                        None => {
                            ui.weak(t!("common-working"));
                        }
                        Some(sess) => open_slice = body(app, sess, ui, drawer),
                    });
            });
        });
    if close {
        app.drawer = None;
    }
    if let Some(id) = open_slice {
        app.open_slice_2d(id);
    }
    width + margin
}

fn body(app: &mut AppState, sess: &mut Session, ui: &mut Ui, drawer: Drawer) -> Option<SlideId> {
    match drawer {
        Drawer::Layers => match slides::layers_drawer(app, sess, ui) {
            slides::LayersEvent::OpenSlice(id) => Some(id),
            slides::LayersEvent::None => None,
        },
        Drawer::Sample => {
            sample_drawer(app, sess, ui);
            None
        }
        Drawer::Labels => {
            app.fix_current(sess);
            segment::labels_drawer(app, sess, ui);
            None
        }
        Drawer::Align => {
            register::align_drawer(app, sess, ui);
            None
        }
        Drawer::Tumor3D => reconstruct::tumor_drawer(app, sess, ui),
        Drawer::Detect => detect::detect_drawer(app, sess, ui),
    }
}

fn sample_drawer(app: &mut AppState, sess: &mut Session, ui: &mut Ui) {
    egui::CollapsingHeader::new(t!("slides-sample-panel"))
        .default_open(true)
        .show(ui, |ui| sample_ui(ui, sess, &mut app.toasts));
    egui::CollapsingHeader::new(t!("anat-panel"))
        .default_open(true)
        .show(ui, |ui| app.slides.orientation.ui(ui, sess));
    egui::CollapsingHeader::new(t!("clinical-panel"))
        .default_open(false)
        .show(ui, |ui| app.slides.clinical.ui(ui, sess, &mut app.toasts));
    ui.separator();
    h3d_ui::widgets::section_title(ui, &t!("drawer-slice"));
    match app.current.filter(|id| sess.slide(*id).is_ok()) {
        Some(id) => match slide_meta_ui(ui, sess, &mut app.toasts, &mut app.slides.meta, id) {
            SlideMetaAction::None => {}
            SlideMetaAction::MppChanged => {
                app.tiles.forget_slide(id);
                app.recon.uploaded_slices.remove(&id);
                app.invalidate_derived();
            }
            SlideMetaAction::Removed => {
                app.tiles.forget_slide(id);
                app.thumbs.forget(id);
                app.thumb_worker.forget(id);
                app.recon.uploaded_slices.remove(&id);
                app.current = None;
                app.fix_current(sess);
                app.invalidate_derived();
                app.recon.fitted = false;
            }
        },
        None => {
            ui.weak(t!("stack-selected-none"));
        }
    }
}
