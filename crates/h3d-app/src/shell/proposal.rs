//! Kartu usulan susunan (urutan + orientasi) di atas kanvas 3D: ringkasan,
//! Terima semua / Terima dicentang / Balik urutan / Tolak, dan rincian yang
//! bisa dibuka. Pratinjau tembus pandang digambar kanvas selama kartu
//! terbuka; proyek baru berubah lewat `AppState::decide_order` (aturan 8).

use egui::{Rect, Ui};
use h3d_engine::Session;

use crate::features::slides::{order_preview_ui, OrderDecision};
use crate::state::{AppState, View};

/// Gambar kartu bila ada usulan menunggu.
pub fn show(app: &mut AppState, sess: &mut Session, ui: &mut Ui, work: Rect) {
    let Some(mut preview) = app.slides.order_preview.take() else {
        return;
    };
    // Usulan hanya bermakna di tumpukan 3D.
    if app.view != View::Stack3D {
        app.set_view(View::Stack3D);
    }
    let mut decision = OrderDecision::None;
    let width = 520.0_f32.min(work.width() - 2.0 * h3d_ui::theme::TOOL_BUTTON_SIDE - 80.0);
    egui::Area::new(egui::Id::new("h3d-order-proposal"))
        .order(egui::Order::Middle)
        .pivot(egui::Align2::CENTER_TOP)
        .fixed_pos(egui::pos2(
            work.center().x,
            work.top() + h3d_ui::theme::FLOAT_MARGIN,
        ))
        .show(ui.ctx(), |ui| {
            ui.set_width(width);
            h3d_ui::theme::panel_frame(ui.visuals()).show(ui, |ui| {
                ui.set_width(width - 20.0);
                decision = order_preview_ui(ui, &mut preview, sess);
            });
        });
    app.slides.order_preview = Some(preview);
    if decision != OrderDecision::None {
        app.decide_order(sess, decision);
    }
}
