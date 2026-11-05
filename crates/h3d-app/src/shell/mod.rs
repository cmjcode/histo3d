//! Cangkang aplikasi ala DUCAD: satu kanvas penuh (tumpukan 3D atau irisan
//! 2D) dengan panel-panel mengambang di atasnya — bilah atas (menu burger,
//! nama proyek, pemilih tampilan, tombol laci), rel alat kiri, laci kanan,
//! bilah konteks di bawah, kartu usulan susunan, kartu progres job, lembar
//! laporan, dialog Tentang/Pengaturan, peramban Documents, isian mpp, dan
//! pemberitahuan. Tidak membutuhkan jendela: `AppState::ui` bisa dipanggil
//! dari `egui::Context::run_ui` di tes, dengan `gpu = None`.
//!
//! Semua fitur lama (lima mode) tetap tersedia dari layar ini; lihat tabel
//! pemetaan di dokumentasi tiap submodul.

mod context_bar;
mod drawer;
mod proposal;
mod rail;
mod report_sheet;
mod top_bar;

use egui::Ui;
use h3d_engine::Session;
use h3d_i18n::t;

use crate::features::{reconstruct, register, segment};
use crate::gpu::Gpu;
use crate::panels::browser::{BrowserAction, BrowserPurpose, BrowserState};
use crate::panels::mpp_prompt::MppPromptAction;
use crate::platform;
use crate::state::{try_lock_session, AppState, ImportRequest, SliceMode, View};
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
        self.thumb_worker.poll();
        self.tiles.begin_frame();
        if let Some(g) = gpu {
            g.with(|r, _, _| r.begin_frame());
        }
        let job_running = self.poll_job();
        if self.update.poll() {
            ctx.request_repaint_after(std::time::Duration::from_millis(200));
        }

        // Sesi dikunci tanpa memblokir; selama job berjalan kunci dipegang
        // thread job dan kanvas menggambar dari cache.
        let shared = self.session.clone();
        let mut guard = if job_running {
            None
        } else {
            shared.as_ref().and_then(try_lock_session)
        };

        if !self.has_project() {
            self.welcome(ui);
        } else {
            // Lembar laporan mengambil bagian bawah; kanvas sisanya.
            if self.report_open {
                report_sheet::show(self, guard.as_deref_mut(), ui);
            }
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ui, |ui| {
                    let canvas = ui.max_rect();
                    self.canvas(ui, guard.as_deref_mut(), gpu, canvas);
                    self.overlays(ui, guard.as_deref_mut(), canvas);
                });
            if let Some(sess) = guard.as_deref_mut() {
                self.autosave(sess, &ctx, now);
            }
        }
        drop(guard);

        self.dialogs(&ctx);
        self.toasts.show(&ctx);

        if self.job.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        } else if self.tiles.needs_repaint() || self.thumb_worker.has_pending() {
            ctx.request_repaint();
        }
    }

    /// Kanvas penuh: tumpukan 3D atau irisan 2D.
    fn canvas(
        &mut self,
        ui: &mut Ui,
        sess: Option<&mut Session>,
        gpu: Option<&Gpu>,
        rect: egui::Rect,
    ) {
        match self.view {
            View::Stack3D => {
                let event = reconstruct::canvas(self, sess.as_deref(), ui, gpu, rect);
                match event {
                    reconstruct::CanvasEvent::None => {}
                    reconstruct::CanvasEvent::Selected(id) => self.current = Some(id),
                    reconstruct::CanvasEvent::OpenSlice(id) => self.open_slice_2d(id),
                }
            }
            View::Slice2D => {
                let mut child = ui.new_child(egui::UiBuilder::new().max_rect(rect));
                match sess {
                    Some(sess) => match self.slice_mode {
                        SliceMode::Annotate => segment::viewer(self, sess, &mut child, gpu),
                        SliceMode::Align => register::viewer(self, sess, &mut child, gpu),
                    },
                    None => {
                        crate::viewer::paint_background(&child, rect, gpu.is_some());
                        child.painter_at(rect).text(
                            rect.center(),
                            egui::Align2::CENTER_CENTER,
                            t!("common-working"),
                            egui::FontId::proportional(14.0),
                            ui.visuals().weak_text_color(),
                        );
                    }
                }
            }
        }
    }

    /// Panel-panel mengambang di atas kanvas, urutan dari belakang ke depan.
    fn overlays(&mut self, ui: &mut Ui, mut sess: Option<&mut Session>, canvas: egui::Rect) {
        let top_h = top_bar::show(self, sess.as_deref_mut(), ui, canvas);
        let below_top =
            egui::Rect::from_min_max(egui::pos2(canvas.left(), canvas.top() + top_h), canvas.max);
        let drawer_w = drawer::show(self, sess.as_deref_mut(), ui, below_top);
        let work = egui::Rect::from_min_max(
            below_top.min,
            egui::pos2(below_top.right() - drawer_w, below_top.bottom()),
        );
        rail::show(self, sess.as_deref_mut(), ui, work);
        if self.view == View::Stack3D {
            reconstruct::viewcube(self, ui, work);
        }
        if let Some(sess) = sess.as_deref_mut() {
            proposal::show(self, sess, ui, work);
            if self.view == View::Slice2D {
                self.tool_options_bar(sess, ui, work);
            }
        }
        context_bar::show(self, sess, ui, work);
        if self.job.is_some() {
            self.job_card(ui, work);
        }
    }

    /// Pilihan alat 2D (ukuran kuas, mode banding, dst.) di bawah bilah atas.
    fn tool_options_bar(&mut self, sess: &mut Session, ui: &mut Ui, work: egui::Rect) {
        let pos = egui::pos2(
            work.left() + h3d_ui::theme::TOOL_BUTTON_SIDE + 3.0 * h3d_ui::theme::FLOAT_MARGIN,
            work.top() + h3d_ui::theme::FLOAT_MARGIN,
        );
        egui::Area::new(egui::Id::new("h3d-tool-options"))
            .order(egui::Order::Middle)
            .fixed_pos(pos)
            .show(ui.ctx(), |ui| {
                ui.set_max_width(work.width() - 2.0 * h3d_ui::theme::TOOL_BUTTON_SIDE - 60.0);
                h3d_ui::theme::pill_frame(ui.visuals()).show(ui, |ui| match self.slice_mode {
                    SliceMode::Annotate => segment::tool_options(self, sess, ui),
                    SliceMode::Align => register::tool_options(self, sess, ui),
                });
            });
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

    pub(crate) fn request_new_project(&mut self) {
        if platform::is_mobile() {
            self.browser = Some(BrowserState::new(BrowserPurpose::Projects));
        } else if let Some(path) = platform::pick_new_project() {
            self.create_project(&path);
        }
    }

    pub(crate) fn request_open_project(&mut self) {
        if platform::is_mobile() {
            self.browser = Some(BrowserState::new(BrowserPurpose::Projects));
        } else if let Some(path) = platform::pick_open_project() {
            self.open_project(&path);
        }
    }

    /// Layar sambutan tanpa proyek.
    fn welcome(&mut self, ui: &mut Ui) {
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(h3d_ui::theme::canvas_fill(ui.visuals())))
            .show(ui, |ui| {
                let rect = ui.max_rect();
                let card = egui::Rect::from_center_size(
                    rect.center() - egui::vec2(0.0, rect.height() * 0.06),
                    egui::vec2(420.0_f32.min(rect.width() - 32.0), 300.0),
                );
                let mut child = ui.new_child(
                    egui::UiBuilder::new()
                        .max_rect(card)
                        .layout(egui::Layout::top_down(egui::Align::Center)),
                );
                h3d_ui::theme::panel_frame(child.visuals()).show(&mut child, |ui| {
                    ui.set_width(card.width() - 20.0);
                    ui.add_space(14.0);
                    ui.label(
                        h3d_ui::icons::STACK_3D
                            .rich_text()
                            .size(40.0)
                            .color(h3d_ui::theme::ACCENT),
                    );
                    ui.heading(t!("app-name"));
                    ui.label(t!("app-tagline"));
                    ui.add_space(6.0);
                    ui.label(egui::RichText::new(t!("welcome-hint")).weak().size(12.0));
                    ui.add_space(14.0);
                    ui.vertical_centered_justified(|ui| {
                        if icon_text_button(ui, h3d_ui::icons::NEW, &t!("action-new")).clicked() {
                            self.request_new_project();
                        }
                        if icon_text_button(ui, h3d_ui::icons::OPEN_FOLDER, &t!("action-open"))
                            .on_hover_text(t!("welcome-open-slide-folder"))
                            .clicked()
                        {
                            self.request_open_project();
                        }
                    });
                    ui.add_space(12.0);
                    h3d_ui::widgets::research_notice(ui);
                    ui.add_space(8.0);
                });
                // Bilah atas tetap ada untuk bahasa, tema, Tentang, Pengaturan.
                top_bar::show(self, None, ui, rect);
            });
    }

    /// Kartu progres job di bawah kanvas (kanvas tetap tampil).
    fn job_card(&mut self, ui: &mut Ui, work: egui::Rect) {
        let Some(job) = &self.job else {
            return;
        };
        let fraction = job.handle.progress.fraction().clamp(0.0, 1.0);
        let stage = job.handle.progress.stage();
        let title = job.title.clone();
        let mut cancel = false;
        let pos = egui::pos2(work.center().x - 190.0, work.bottom() - 110.0);
        egui::Area::new(egui::Id::new("h3d-job-card"))
            .order(egui::Order::Foreground)
            .fixed_pos(pos)
            .show(ui.ctx(), |ui| {
                h3d_ui::theme::panel_frame(ui.visuals()).show(ui, |ui| {
                    ui.set_width(380.0);
                    ui.horizontal(|ui| {
                        ui.add(egui::Spinner::new().size(16.0));
                        crate::ui_util::strong(ui, title);
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if h3d_ui::widgets::icon_button(
                                ui,
                                h3d_ui::icons::CLOSE,
                                &t!("action-cancel"),
                            )
                            .clicked()
                            {
                                cancel = true;
                            }
                        });
                    });
                    ui.small(if stage.is_empty() {
                        t!("common-working")
                    } else {
                        stage
                    });
                    ui.add(egui::ProgressBar::new(fraction).show_percentage());
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
            .show(ctx, |ui| {
                about_ui(ui);
                self.update_section(ui);
            });
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
                ui.separator();
                changed |= ui
                    .checkbox(
                        &mut self.settings.auto_stack,
                        icon_label(h3d_ui::icons::AUTO_STACK, &t!("settings-auto-stack")),
                    )
                    .changed();
                ui.small(t!("settings-auto-stack-hint"));
                ui.separator();
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
                BrowserAction::Import(paths, embed_mode) => {
                    // Dari Documents slide selalu dimasukkan ke bundel (9.3):
                    // dipindahkan (tanpa salinan ganda) atau disalin.
                    self.start_import(
                        ImportRequest {
                            paths,
                            embed: true,
                            embed_mode,
                        },
                        None,
                    );
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
