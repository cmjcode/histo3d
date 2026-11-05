//! Lembar laporan: panel di bawah kanvas yang bisa diubah tingginya, dengan
//! kepala (judul, perbesar/kecilkan, tutup) dan isi dari
//! `features::report::sheet`. Kalimat "Hanya untuk riset" ada di dalam isi.

use egui::Ui;
use h3d_engine::Session;
use h3d_i18n::t;

use crate::features::report;
use crate::state::AppState;

/// Gambar lembar laporan di bagian bawah `ui`.
pub fn show(app: &mut AppState, sess: Option<&mut Session>, ui: &mut Ui) {
    let full = ui.available_height();
    let default_h = (full * 0.45).clamp(220.0, 520.0);
    let mut close = false;
    let mut toggle_full = false;
    let frame = h3d_ui::theme::panel_frame(ui.visuals())
        .corner_radius(egui::CornerRadius {
            nw: 10,
            ne: 10,
            sw: 0,
            se: 0,
        })
        .shadow(egui::Shadow::NONE);
    let mut panel = egui::Panel::bottom("h3d-report-sheet")
        .frame(frame)
        .resizable(true)
        .default_size(default_h)
        .min_size(160.0);
    if app.report.expanded {
        panel = panel.exact_size(full - 60.0);
    }
    panel.show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(
                h3d_ui::icons::REPORT
                    .rich_text()
                    .size(h3d_ui::theme::ICON_SIZE_DEFAULT)
                    .color(h3d_ui::theme::ACCENT),
            );
            ui.label(
                egui::RichText::new(t!("report-title"))
                    .color(h3d_ui::widgets::strong_color(ui))
                    .size(14.0),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                close = h3d_ui::widgets::icon_button(ui, h3d_ui::icons::CLOSE, &t!("report-close"))
                    .clicked();
                let icon = if app.report.expanded {
                    h3d_ui::icons::EXIT_FULLSCREEN
                } else {
                    h3d_ui::icons::FULLSCREEN
                };
                toggle_full = h3d_ui::widgets::icon_button(ui, icon, "").clicked();
            });
        });
        ui.separator();
        match sess {
            Some(sess) => {
                if let Some(id) = report::sheet(app, sess, ui) {
                    app.open_slice_2d(id);
                }
            }
            None => {
                ui.weak(t!("common-working"));
            }
        }
    });
    if toggle_full {
        app.report.expanded = !app.report.expanded;
    }
    if close {
        app.report_open = false;
    }
}
