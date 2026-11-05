//! Bilah atas mengambang (pola `ducad-ui/src/top_bar.rs`): menu burger
//! (proyek, impor, ekspor, laporan, pengaturan, tentang), nama proyek,
//! pemilih tampilan 3D/2D di tengah, dan di kanan tombol laci, laporan,
//! urung/ulang (2D), tema, bahasa.
//!
//! Pemetaan fitur lama: menu Berkas (Baru/Buka/Simpan) dan Impor dari mode
//! Irisan; Ekspor (7 jenis) dari mode Laporan; urung/ulang dari Segmentasi.

use egui::{Rect, Ui};
use h3d_engine::Session;
use h3d_i18n::{t, Language};

use crate::features::{report::ExportChoice, segment, slides};
use crate::platform;
use crate::state::{AppState, Drawer, JobOutput, View};
use crate::ui_util::icon_label;

/// Tinggi bilah atas termasuk margin bawahnya (point).
pub const TOP_BAR_HEIGHT: f32 = 56.0;

/// Gambar bilah atas di tepi atas `canvas`. Mengembalikan tinggi yang
/// dipakai (untuk menempatkan panel lain di bawahnya).
pub fn show(app: &mut AppState, mut sess: Option<&mut Session>, ui: &mut Ui, canvas: Rect) -> f32 {
    let margin = h3d_ui::theme::FLOAT_MARGIN;
    let bar_rect = Rect::from_min_size(
        canvas.min + egui::vec2(margin, margin),
        egui::vec2(canvas.width() - 2.0 * margin, TOP_BAR_HEIGHT - 2.0 * margin),
    );
    let busy = app.job.is_some();
    egui::Area::new(egui::Id::new("h3d-top-bar"))
        .order(egui::Order::Middle)
        .fixed_pos(bar_rect.min)
        .show(ui.ctx(), |ui| {
            ui.set_width(bar_rect.width());
            h3d_ui::theme::panel_frame(ui.visuals()).show(ui, |ui| {
                ui.set_width(bar_rect.width() - 20.0);
                ui.set_height(h3d_ui::theme::TOOL_BUTTON_SIDE);
                ui.spacing_mut().interact_size.y = h3d_ui::theme::MIN_TOUCH_TARGET;
                ui.horizontal_centered(|ui| {
                    // Kiri: menu burger + judul.
                    menu(app, sess.as_deref_mut(), ui, busy);
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new(if app.has_project() {
                            app.project_name.clone()
                        } else {
                            t!("app-name")
                        })
                        .color(h3d_ui::widgets::strong_color(ui))
                        .size(14.0),
                    );
                    if app.has_project() && sess.as_ref().is_some_and(|s| s.is_dirty()) {
                        ui.weak("*");
                    }

                    // Tengah: pemilih tampilan.
                    if app.has_project() {
                        let center = ui.max_rect().center().x;
                        let w = 2.0 * h3d_ui::theme::TOOL_BUTTON_SIDE + 8.0;
                        let x = center - w / 2.0;
                        let spacer = (x - ui.cursor().left()).max(8.0);
                        ui.add_space(spacer);
                        ui.add_enabled_ui(!busy, |ui| view_switch(app, ui));
                    }

                    // Kanan: laci, laporan, urung/ulang, tema, bahasa.
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        right_controls(app, sess, ui, busy);
                    });
                });
            });
        });
    TOP_BAR_HEIGHT
}

fn view_switch(app: &mut AppState, ui: &mut Ui) {
    let mut target = None;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        if h3d_ui::widgets::tool_button(
            ui,
            h3d_ui::icons::STACK_3D,
            app.view == View::Stack3D,
            &t!("view-stack-3d"),
            Some(&t!("view-switch-hint")),
        )
        .clicked()
        {
            target = Some(View::Stack3D);
        }
        if h3d_ui::widgets::tool_button(
            ui,
            h3d_ui::icons::SLICE_2D,
            app.view == View::Slice2D,
            &t!("view-slice-2d"),
            Some(&t!("view-switch-hint")),
        )
        .clicked()
        {
            target = Some(View::Slice2D);
        }
    });
    if let Some(v) = target {
        if v == View::Slice2D {
            match app.current {
                Some(id) => app.open_slice_2d(id),
                None => app.toasts.info(t!("stack-selected-none")),
            }
        } else {
            app.set_view(v);
        }
    }
}

fn right_controls(app: &mut AppState, sess: Option<&mut Session>, ui: &mut Ui, busy: bool) {
    // Bahasa dan tema (selalu ada).
    let lang = app.settings.language();
    let lang_name = match lang {
        Language::Id => t!("lang-id"),
        Language::En => t!("lang-en"),
    };
    if ui
        .button(icon_label(h3d_ui::icons::LANGUAGE, &lang_name))
        .on_hover_text(t!("lang-current"))
        .clicked()
    {
        app.settings.set_language(match lang {
            Language::Id => Language::En,
            Language::En => Language::Id,
        });
        app.persist_settings();
    }
    let theme = app.settings.theme();
    let theme_icon = match theme {
        h3d_ui::ThemeMode::Dark => h3d_ui::icons::LIGHT_MODE,
        h3d_ui::ThemeMode::Light => h3d_ui::icons::DARK_MODE,
    };
    if h3d_ui::widgets::tool_button(ui, theme_icon, false, &t!(theme.toggled().i18n_key()), None)
        .clicked()
    {
        app.settings.set_theme(theme.toggled());
        app.apply_theme(ui.ctx());
        app.persist_settings();
    }
    if !app.has_project() {
        return;
    }
    ui.separator();

    // Laci dan laporan (dari kanan ke kiri, jadi urutan ALL dibalik).
    ui.add_enabled_ui(!busy, |ui| {
        if h3d_ui::widgets::tool_button(
            ui,
            h3d_ui::icons::REPORT,
            app.report_open,
            &t!("report-open"),
            None,
        )
        .clicked()
        {
            app.report_open = !app.report_open;
            if app.report_open {
                app.report.needs_refresh = true;
            }
        }
        ui.separator();
        for d in Drawer::ALL.iter().rev() {
            if h3d_ui::widgets::tool_button(
                ui,
                d.icon(),
                app.drawer == Some(*d),
                &t!(d.i18n_key()),
                None,
            )
            .clicked()
            {
                app.toggle_drawer(*d);
            }
        }
    });

    // Urung/ulang: suntingan label irisan aktif (tampilan 2D).
    if app.view == View::Slice2D {
        ui.separator();
        let (can_undo, can_redo) = match (&sess, app.current) {
            (Some(s), Some(c)) => (s.can_undo(c), s.can_redo(c)),
            _ => (false, false),
        };
        let shortcut = ui.ctx().input(|i| {
            let cmd = i.modifiers.command;
            let z = i.key_pressed(egui::Key::Z);
            (
                cmd && z && !i.modifiers.shift,
                cmd && z && i.modifiers.shift,
            )
        });
        let redo_clicked = ui
            .add_enabled_ui(can_redo, |ui| {
                h3d_ui::widgets::tool_button(
                    ui,
                    h3d_ui::icons::REDO,
                    false,
                    &t!("action-redo"),
                    None,
                )
            })
            .inner
            .clicked();
        let undo_clicked = ui
            .add_enabled_ui(can_undo, |ui| {
                h3d_ui::widgets::tool_button(
                    ui,
                    h3d_ui::icons::UNDO,
                    false,
                    &t!("action-undo"),
                    None,
                )
            })
            .inner
            .clicked();
        if let Some(s) = sess {
            if can_undo && (undo_clicked || shortcut.0) {
                segment::undo(app, s, false);
            } else if can_redo && (redo_clicked || shortcut.1) {
                segment::undo(app, s, true);
            }
        }
    }
}

fn menu(app: &mut AppState, sess: Option<&mut Session>, ui: &mut Ui, busy: bool) {
    let btn = h3d_ui::widgets::tool_button(ui, h3d_ui::icons::MENU, false, &t!("menu-main"), None);
    egui::Popup::menu(&btn)
        .gap(6.0)
        .info(egui::UiStackInfo::new(egui::UiKind::Menu).with_tag_value(
            egui::containers::menu::MenuConfig::MENU_CONFIG_TAG,
            egui::containers::menu::MenuConfig::new(),
        ))
        .show(|ui| {
            ui.set_min_width(230.0);
            ui.add_enabled_ui(!busy, |ui| {
                ui.label(egui::RichText::new(t!("menu-project")).weak().small());
                if ui
                    .button(icon_label(h3d_ui::icons::NEW, &t!("action-new")))
                    .clicked()
                {
                    app.request_new_project();
                    ui.close();
                }
                if ui
                    .button(icon_label(h3d_ui::icons::OPEN_FOLDER, &t!("action-open")))
                    .on_hover_text(t!("welcome-open-slide-folder"))
                    .clicked()
                {
                    app.request_open_project();
                    ui.close();
                }
                if app.has_project() {
                    let can_save = sess.as_ref().is_some_and(|s| s.is_dirty());
                    if ui
                        .add_enabled(
                            can_save,
                            egui::Button::new(icon_label(h3d_ui::icons::SAVE, &t!("action-save"))),
                        )
                        .clicked()
                    {
                        app.save_now();
                        ui.close();
                    }
                    ui.separator();
                    if ui
                        .button(icon_label(h3d_ui::icons::IMPORT, &t!("slides-import")))
                        .on_hover_text(t!("slides-import-hint"))
                        .clicked()
                    {
                        slides::begin_import(app);
                        ui.close();
                    }
                    ui.menu_button(
                        icon_label(h3d_ui::icons::EXPORT, &t!("menu-export")),
                        |ui| {
                            export_menu(app, ui);
                        },
                    );
                    if ui
                        .button(icon_label(h3d_ui::icons::REPORT, &t!("report-open")))
                        .clicked()
                    {
                        app.report_open = true;
                        app.report.needs_refresh = true;
                        ui.close();
                    }
                }
            });
            ui.separator();
            if ui
                .button(icon_label(h3d_ui::icons::SETTINGS, &t!("action-settings")))
                .clicked()
            {
                app.show_settings = true;
                ui.close();
            }
            if ui
                .button(icon_label(h3d_ui::icons::INFO, &t!("about-title")))
                .clicked()
            {
                app.show_about = true;
                ui.close();
            }
            if crate::update::updater_path().is_some()
                && ui
                    .add_enabled(
                        !app.update.is_busy(),
                        egui::Button::new(icon_label(h3d_ui::icons::UPDATE, &t!("update-check"))),
                    )
                    .clicked()
            {
                app.show_about = true;
                app.update.start_check();
                ui.close();
            }
        });
}

/// Submenu ekspor: tujuh jenis berkas + kotak "sertakan data klinis"
/// (mati secara bawaan; aturan keras 17).
fn export_menu(app: &mut AppState, ui: &mut Ui) {
    ui.set_min_width(240.0);
    ui.checkbox(&mut app.report.with_clinical, t!("clinical-export-with"));
    ui.small(t!("export-clinical-hint"));
    ui.separator();
    for choice in ExportChoice::ALL {
        if ui
            .button(icon_label(h3d_ui::icons::EXPORT, &t!(choice.i18n_key())))
            .clicked()
        {
            ui.close();
            let out = if platform::is_mobile() {
                app.session
                    .as_ref()
                    .and_then(|sh| crate::state::try_lock_session(sh).map(|s| s.bundle().out_dir()))
            } else {
                platform::pick_export_dir()
            };
            if let Some(out) = out {
                let with_clinical = app.report.with_clinical;
                let scalar = app.recon.scalar.clone();
                app.start_job(t!(choice.i18n_key()), move |s, _p| {
                    std::fs::create_dir_all(&out)
                        .map_err(|e| h3d_engine::EngineError::invalid(e.to_string()))?;
                    s.export_files(choice.kind(with_clinical, scalar), &out)
                        .map(JobOutput::Exported)
                });
            }
        }
    }
}
