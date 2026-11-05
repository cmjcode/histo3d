//! Bilah konteks (kapsul) di bawah kanvas, pola `ducad-ui/src/context_bar.rs`:
//! aksi untuk irisan terpilih.
//!
//! - Tumpukan 3D: "n. nama · z", lencana QC/status, Buka 2D, Jadikan acuan,
//!   Tandai rusak / batalkan, Sembunyikan di 3D.
//! - Irisan 2D: Kembali ke 3D, irisan sebelumnya / "i dari n" / berikutnya.

use egui::{Rect, Ui};
use h3d_core::SlideStatus;
use h3d_engine::Session;
use h3d_i18n::t;

use crate::geom::{navigate, NavDir};
use crate::state::{AppState, View};
use crate::ui_util::{icon_label_colored, icon_text_button, slide_title};

/// Gambar kapsul konteks di tengah bawah `work`.
pub fn show(app: &mut AppState, sess: Option<&mut Session>, ui: &mut Ui, work: Rect) {
    let Some(sess) = sess else {
        return;
    };
    let Some(slide) = app.current.filter(|id| sess.slide(*id).is_ok()) else {
        if app.view == View::Stack3D && !app.recon.slices.is_empty() {
            hint_pill(ui, work, &t!("stack-selected-none"));
        }
        return;
    };
    let entry = match sess.slide(slide) {
        Ok(e) => e.clone(),
        Err(_) => return,
    };
    let order = AppState::ordered_ids(sess);
    let index = order.iter().position(|o| *o == slide).unwrap_or(0);
    let bottom = work.bottom() - h3d_ui::theme::FLOAT_MARGIN - 8.0;
    let mut nav: Option<NavDir> = None;
    let mut open_2d = false;
    let mut back_3d = false;
    let mut set_ref = false;
    let mut toggle_damaged = false;
    let mut toggle_hidden = false;

    egui::Area::new(egui::Id::new("h3d-context-bar"))
        .order(egui::Order::Middle)
        .pivot(egui::Align2::CENTER_BOTTOM)
        .fixed_pos(egui::pos2(work.center().x, bottom))
        .show(ui.ctx(), |ui| {
            h3d_ui::theme::pill_frame(ui.visuals()).show(ui, |ui| {
                ui.spacing_mut().interact_size.y = h3d_ui::theme::MIN_TOUCH_TARGET;
                ui.horizontal(|ui| match app.view {
                    View::Stack3D => {
                        crate::ui_util::strong(ui, slide_title(sess, slide));
                        ui.weak(t!("context-slice-z", z = format!("{:.0}", entry.z_um)));
                        if let Some(r) = &entry.reg {
                            if r.flagged {
                                ui.label(icon_label_colored(
                                    h3d_ui::icons::FLAG,
                                    &t!("layers-qc-flagged"),
                                    h3d_ui::theme::WARNING,
                                ));
                            }
                        } else if sess.project().effective_reference() != Some(slide)
                            && entry.status == SlideStatus::Ok
                        {
                            ui.weak(t!("layers-not-registered"));
                        }
                        ui.separator();
                        if icon_text_button(ui, h3d_ui::icons::SLICE_2D, &t!("view-open-2d"))
                            .clicked()
                        {
                            open_2d = true;
                        }
                        let is_ref = sess.project().effective_reference() == Some(slide);
                        if ui
                            .add_enabled(
                                !is_ref && entry.status == SlideStatus::Ok,
                                egui::Button::new(h3d_ui::icons::REFERENCE.rich_text()),
                            )
                            .on_hover_text(t!("slides-set-reference"))
                            .clicked()
                        {
                            set_ref = true;
                        }
                        let dmg = if entry.status == SlideStatus::Damaged {
                            t!("slides-unmark-damaged")
                        } else {
                            t!("slides-mark-damaged")
                        };
                        if ui
                            .add(
                                egui::Button::new(h3d_ui::icons::DAMAGED.rich_text())
                                    .selected(entry.status == SlideStatus::Damaged),
                            )
                            .on_hover_text(dmg)
                            .clicked()
                        {
                            toggle_damaged = true;
                        }
                        let hidden = app.recon.hidden_slices.contains(&slide);
                        let eye = if hidden {
                            h3d_ui::icons::VISIBILITY_OFF
                        } else {
                            h3d_ui::icons::VISIBILITY_ON
                        };
                        if ui
                            .add(egui::Button::new(eye.rich_text()))
                            .on_hover_text(if hidden {
                                t!("stack-slice-shown")
                            } else {
                                t!("stack-slice-hidden")
                            })
                            .clicked()
                        {
                            toggle_hidden = true;
                        }
                    }
                    View::Slice2D => {
                        if icon_text_button(ui, h3d_ui::icons::BACK, &t!("view-back-to-3d"))
                            .clicked()
                        {
                            back_3d = true;
                        }
                        ui.separator();
                        if ui
                            .add_enabled(
                                index > 0,
                                egui::Button::new(h3d_ui::icons::PREV.rich_text()),
                            )
                            .on_hover_text(t!("action-prev-slice"))
                            .clicked()
                        {
                            nav = Some(NavDir::Prev);
                        }
                        crate::ui_util::strong(ui, slide_title(sess, slide));
                        ui.weak(t!(
                            "nav-slice-of",
                            index = (index + 1).to_string(),
                            count = order.len().to_string()
                        ));
                        if ui
                            .add_enabled(
                                index + 1 < order.len(),
                                egui::Button::new(h3d_ui::icons::NEXT.rich_text()),
                            )
                            .on_hover_text(t!("action-next-slice"))
                            .clicked()
                        {
                            nav = Some(NavDir::Next);
                        }
                    }
                });
            });
        });

    if let Some(dir) = nav {
        app.current = navigate(&order, app.current, dir);
        app.segment.cancel_drawing();
    }
    if open_2d {
        app.open_slice_2d(slide);
    }
    if back_3d {
        app.set_view(View::Stack3D);
    }
    if set_ref {
        if let Err(e) = sess.set_reference(Some(slide)) {
            app.toasts.engine_error(&e);
        }
        app.invalidate_derived();
    }
    if toggle_damaged {
        let target = if entry.status == SlideStatus::Damaged {
            SlideStatus::Ok
        } else {
            SlideStatus::Damaged
        };
        if let Err(e) = sess.set_status(slide, target) {
            app.toasts.engine_error(&e);
        }
        app.invalidate_derived();
        app.recon.fitted = false;
    }
    if toggle_hidden && !app.recon.hidden_slices.remove(&slide) {
        app.recon.hidden_slices.insert(slide);
    }
}

fn hint_pill(ui: &mut Ui, work: Rect, text: &str) {
    egui::Area::new(egui::Id::new("h3d-context-hint"))
        .order(egui::Order::Middle)
        .pivot(egui::Align2::CENTER_BOTTOM)
        .fixed_pos(egui::pos2(
            work.center().x,
            work.bottom() - h3d_ui::theme::FLOAT_MARGIN - 8.0,
        ))
        .interactable(false)
        .show(ui.ctx(), |ui| {
            h3d_ui::theme::pill_frame(ui.visuals()).show(ui, |ui| {
                ui.weak(text);
            });
        });
}
