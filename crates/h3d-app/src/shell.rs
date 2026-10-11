//! Cangkang aplikasi: satu frame UI (bilah atas, mode aktif, lapisan job,
//! dialog Tentang/Pengaturan, peramban Documents, isian mpp, pemberitahuan).
//! Tidak membutuhkan jendela: `AppState::ui` bisa dipanggil dari
//! `egui::Context::run_ui` di tes, dengan `gpu = None`.

use egui::Ui;
use h3d_engine::Session;
use h3d_i18n::{t, Language};
use h3d_ui::AppMode;

use crate::gpu::Gpu;
use crate::modes::{reconstruct, register, report, segment, slides};
use crate::panels::browser::{BrowserAction, BrowserPurpose, BrowserState};
use crate::panels::mpp_prompt::MppPromptAction;
use crate::platform;
use crate::state::{try_lock_session, AppState, ImportRequest};
use crate::ui_util::{icon_label, icon_text_button};

/// Selang simpan otomatis saat ada perubahan (detik).
pub const AUTOSAVE_SECS: f64 = 1.0;

impl AppState {
    /// Gambar satu frame. `gpu = None` → tidak ada paint callback (tes).
    pub fn ui(&mut self, ui: &mut Ui, gpu: Option<&Gpu>) {
        let ctx = ui.ctx().clone();
        self.frame_count += 1;
        let now = ctx.input(|i| i.time);
        self.toasts.begin_frame(now);
        if platform::take_memory_warning() {
            self.on_memory_warning(gpu);
        }
        ctx.input(|i| self.touch.observe(&i.events, i.any_touches()));
        self.thumbs.begin_frame();
        self.tiles.begin_frame();
        if let Some(g) = gpu {
            g.with(|r, _, _| r.begin_frame());
        }
        let job_running = self.poll_job();

        // Sesi dikunci tanpa memblokir; selama job berjalan kunci dipegang
        // thread job dan mode tidak digambar.
        let shared = self.session.clone();
        let mut guard = if job_running {
            None
        } else {
            shared.as_ref().and_then(try_lock_session)
        };

        self.top_bar(ui, guard.as_deref_mut());

        if self.job.is_some() {
            self.job_overlay(ui);
        } else if let Some(sess) = guard.as_deref_mut() {
            match self.mode {
                AppMode::Slides => slides::ui(self, sess, ui),
                AppMode::Register => register::ui(self, sess, ui, gpu),
                AppMode::Segment => segment::ui(self, sess, ui, gpu),
                AppMode::Reconstruct => reconstruct::ui(self, sess, ui, gpu),
                AppMode::Report => report::ui(self, sess, ui),
            }
            self.autosave(sess, &ctx, now);
        } else if self.session.is_some() {
            // Kunci belum bisa diambil (job baru saja selesai): coba lagi.
            egui::CentralPanel::default().show(ui, |ui| {
                ui.centered_and_justified(|ui| ui.label(t!("common-working")));
            });
            ctx.request_repaint();
        } else {
            self.welcome(ui);
        }
        drop(guard);

        self.dialogs(&ctx);
        self.toasts.show(&ctx);

        if self.job.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        } else if self.tiles.needs_repaint() {
            ctx.request_repaint();
        }
    }

    fn autosave(&mut self, sess: &mut Session, ctx: &egui::Context, now: f64) {
        let busy = ctx.input(|i| i.pointer.any_down()) || ctx.egui_wants_keyboard_input();
        if busy || !sess.is_dirty() || now - self.last_autosave < AUTOSAVE_SECS {
            return;
        }
        self.last_autosave = now;
        if let Err(e) = sess.save() {
            self.toasts.engine_error(&e);
        }
    }

    fn top_bar(&mut self, ui: &mut Ui, mut sess: Option<&mut Session>) {
        let busy = self.job.is_some();
        egui::Panel::top("h3d-top-bar").show(ui, |ui| {
            ui.horizontal(|ui| {
                crate::ui_util::strong(ui, t!("app-name"));
                if self.has_project() {
                    ui.label(&self.project_name);
                }
                ui.separator();
                ui.add_enabled_ui(!busy, |ui| {
                    if h3d_ui::widgets::icon_button(ui, h3d_ui::icons::NEW, &t!("action-new"))
                        .clicked()
                    {
                        self.request_new_project();
                    }
                    if h3d_ui::widgets::icon_button(
                        ui,
                        h3d_ui::icons::OPEN_FOLDER,
                        &t!("action-open"),
                    )
                    .clicked()
                    {
                        self.request_open_project();
                    }
                    let can_save = sess.as_ref().is_some_and(|s| s.is_dirty());
                    if ui
                        .add_enabled(
                            can_save,
                            egui::Button::new(h3d_ui::icons::SAVE.rich_text().size(18.0))
                                .frame(false),
                        )
                        .on_hover_text(t!("action-save"))
                        .clicked()
                    {
                        if let Some(s) = sess.as_deref_mut() {
                            if let Err(e) = s.save() {
                                self.toasts.engine_error(&e);
                            }
                        }
                    }
                });
                ui.separator();
                if self.has_project() {
                    ui.add_enabled_ui(!busy, |ui| {
                        let mut mode = self.mode;
                        if h3d_ui::widgets::mode_bar(ui, &mut mode) {
                            self.set_mode(mode);
                        }
                    });
                    ui.separator();
                    // Urung/ulang: suntingan label irisan aktif.
                    let (can_undo, can_redo) = match (&sess, self.current) {
                        (Some(s), Some(c)) if self.mode == AppMode::Segment => {
                            (s.can_undo(c), s.can_redo(c))
                        }
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
                    let undo_clicked = ui
                        .add_enabled(
                            can_undo,
                            egui::Button::new(h3d_ui::icons::UNDO.rich_text().size(18.0))
                                .frame(false),
                        )
                        .on_hover_text(t!("action-undo"))
                        .clicked();
                    let redo_clicked = ui
                        .add_enabled(
                            can_redo,
                            egui::Button::new(h3d_ui::icons::REDO.rich_text().size(18.0))
                                .frame(false),
                        )
                        .on_hover_text(t!("action-redo"))
                        .clicked();
                    if let Some(s) = sess {
                        if can_undo && (undo_clicked || shortcut.0) {
                            segment::undo(self, s, false);
                        } else if can_redo && (redo_clicked || shortcut.1) {
                            segment::undo(self, s, true);
                        }
                    }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if h3d_ui::widgets::icon_button(ui, h3d_ui::icons::INFO, &t!("about-title"))
                        .clicked()
                    {
                        self.show_about = true;
                    }
                    if h3d_ui::widgets::icon_button(
                        ui,
                        h3d_ui::icons::SETTINGS,
                        &t!("action-settings"),
                    )
                    .clicked()
                    {
                        self.show_settings = true;
                    }
                    let theme = self.settings.theme();
                    let theme_icon = match theme {
                        h3d_ui::ThemeMode::Dark => h3d_ui::icons::LIGHT_MODE,
                        h3d_ui::ThemeMode::Light => h3d_ui::icons::DARK_MODE,
                    };
                    if h3d_ui::widgets::icon_button(ui, theme_icon, &t!(theme.toggled().i18n_key()))
                        .clicked()
                    {
                        self.settings.set_theme(theme.toggled());
                        self.apply_theme(ui.ctx());
                        self.persist_settings();
                    }
                    let lang = self.settings.language();
                    let lang_name = match lang {
                        Language::Id => t!("lang-id"),
                        Language::En => t!("lang-en"),
                    };
                    if ui
                        .button(icon_label(h3d_ui::icons::LANGUAGE, &lang_name))
                        .on_hover_text(t!("lang-current"))
                        .clicked()
                    {
                        self.settings.set_language(match lang {
                            Language::Id => Language::En,
                            Language::En => Language::Id,
                        });
                        self.persist_settings();
                    }
                });
            });
        });
    }

    /// Terapkan tema dan ukuran target sentuh ke context egui.
    pub fn apply_theme(&self, ctx: &egui::Context) {
        if platform::is_mobile() {
            h3d_ui::theme::apply_with_touch(
                ctx,
                self.settings.theme(),
                h3d_ui::theme::TOUCH_TARGET_IPAD,
            );
        } else {
            h3d_ui::theme::apply(ctx, self.settings.theme());
        }
    }

    fn request_new_project(&mut self) {
        if platform::is_mobile() {
            self.browser = Some(BrowserState::new(BrowserPurpose::Projects));
        } else if let Some(path) = platform::pick_new_project() {
            self.create_project(&path);
        }
    }

    fn request_open_project(&mut self) {
        if platform::is_mobile() {
            self.browser = Some(BrowserState::new(BrowserPurpose::Projects));
        } else if let Some(path) = platform::pick_open_project() {
            self.open_project(&path);
        }
    }

    fn welcome(&mut self, ui: &mut Ui) {
        egui::CentralPanel::default().show(ui, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(ui.available_height() * 0.22);
                ui.heading(t!("app-name"));
                ui.label(t!("app-tagline"));
                ui.add_space(16.0);
                if icon_text_button(ui, h3d_ui::icons::NEW, &t!("action-new")).clicked() {
                    self.request_new_project();
                }
                if icon_text_button(ui, h3d_ui::icons::OPEN_FOLDER, &t!("action-open")).clicked() {
                    self.request_open_project();
                }
                ui.add_space(24.0);
                ui.horizontal(|ui| {
                    // Dipusatkan secara kasar; pemberitahuan wajib tetap tampil.
                    ui.add_space((ui.available_width() - 330.0).max(0.0) / 2.0);
                    h3d_ui::widgets::research_notice(ui);
                });
            });
        });
    }

    fn job_overlay(&mut self, ui: &mut Ui) {
        let Some(job) = &self.job else {
            return;
        };
        let fraction = job.handle.progress.fraction().clamp(0.0, 1.0);
        let stage = job.handle.progress.stage();
        let title = job.title.clone();
        let mut cancel = false;
        egui::CentralPanel::default().show(ui, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(ui.available_height() * 0.3);
                ui.heading(title);
                ui.add_space(8.0);
                ui.label(if stage.is_empty() {
                    t!("common-working")
                } else {
                    stage
                });
                ui.add(
                    egui::ProgressBar::new(fraction)
                        .desired_width(360.0)
                        .show_percentage(),
                );
                ui.add_space(8.0);
                if icon_text_button(ui, h3d_ui::icons::CLOSE, &t!("action-cancel")).clicked() {
                    cancel = true;
                }
                ui.small(t!("job-resumable-hint"));
            });
        });
        if cancel {
            if let Some(job) = &self.job {
                job.handle.cancel();
            }
        }
    }

    fn dialogs(&mut self, ctx: &egui::Context) {
        // Tentang: versi + kalimat wajib "hanya untuk riset" (aturan 16).
        let mut open = self.show_about;
        egui::Window::new(t!("about-title"))
            .id(egui::Id::new("h3d-about"))
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .show(ctx, about_ui);
        self.show_about = open;

        let mut open = self.show_settings;
        let mut changed = false;
        egui::Window::new(t!("action-settings"))
            .id(egui::Id::new("h3d-settings"))
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(icon_label(
                        h3d_ui::icons::PERSON,
                        &t!("settings-user-initials"),
                    ));
                    changed |= ui
                        .add(
                            egui::TextEdit::singleline(&mut self.settings.user_initials)
                                .desired_width(80.0),
                        )
                        .changed();
                });
                ui.small(t!("settings-user-initials-hint"));
                changed |= ui
                    .checkbox(
                        &mut self.settings.finger_draw,
                        icon_label(h3d_ui::icons::TOUCH, &t!("segment-finger-draw")),
                    )
                    .changed();
                ui.small(t!("settings-finger-draw-hint"));
            });
        self.show_settings = open;
        if changed {
            self.persist_settings();
            if let Some(shared) = self.session.clone() {
                if let Some(mut s) = try_lock_session(&shared) {
                    let initials = self.settings.user_initials.trim().to_owned();
                    if s.project().user_initials != initials {
                        s.set_user_initials(&initials);
                    }
                }
            }
        }

        // Peramban Documents.
        if let Some(mut browser) = self.browser.take() {
            match browser.ui(ctx) {
                BrowserAction::None => self.browser = Some(browser),
                BrowserAction::Close => {}
                BrowserAction::Open(path) => {
                    if !self.open_project(&path) {
                        self.browser = Some(browser);
                    }
                }
                BrowserAction::Create(path) => {
                    if !self.create_project(&path) {
                        self.browser = Some(browser);
                    }
                }
                BrowserAction::Import(paths) => {
                    // Dari Documents slide selalu disalin ke bundel (9.3).
                    self.start_import(ImportRequest { paths, embed: true }, None);
                }
            }
        }

        // Isian mpp.
        if let Some(mut prompt) = self.mpp_prompt.take() {
            match prompt.ui(ctx) {
                MppPromptAction::None => self.mpp_prompt = Some(prompt),
                MppPromptAction::Cancel => {}
                MppPromptAction::Apply(mpp) => self.start_import(prompt.request, Some(mpp)),
            }
        }
    }
}

/// Isi dialog Tentang.
pub fn about_ui(ui: &mut Ui) {
    ui.heading(t!("app-name"));
    ui.label(t!("app-tagline"));
    ui.label(t!("about-version", version = h3d_engine::APP_VERSION));
    ui.label(t!("about-license"));
    ui.separator();
    h3d_ui::widgets::research_notice(ui);
}
