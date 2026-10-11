//! Integrasi eframe: renderer wgpu (tanpa glow), `SceneRenderer` di
//! `callback_resources`, simpan otomatis saat aplikasi masuk latar belakang
//! (`App::save`, 9.4), dan mode tangkapan layar untuk verifikasi.

use std::path::PathBuf;

use crate::args::CliArgs;
use crate::gpu::Gpu;
use crate::platform;
use crate::settings::Settings;
use crate::state::AppState;

/// Frame minimum sebelum tangkapan layar diambil.
const SHOT_MIN_FRAMES: u64 = 20;
/// Frame maksimum menunggu tile/job sebelum tetap mengambil tangkapan layar.
const SHOT_MAX_FRAMES: u64 = 900;

/// Opsi eframe: renderer wgpu dengan depth buffer 32-bit (format depth
/// pipeline `SceneRenderer` harus sama; lihat [`crate::gpu::DEPTH_FORMAT`]).
pub fn native_options() -> eframe::NativeOptions {
    let mut viewport = egui::ViewportBuilder::default().with_title("HISTO3D");
    if platform::is_mobile() {
        viewport = viewport.with_maximized(true);
    } else {
        viewport = viewport.with_inner_size([1440.0, 900.0]);
    }
    eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        depth_buffer: 32,
        viewport,
        // `persistence` hanya dipakai agar eframe memanggil `App::save`.
        persist_window: false,
        ..Default::default()
    }
}

/// Jalankan aplikasi sampai jendela ditutup.
pub fn run(args: CliArgs) -> anyhow::Result<()> {
    // Kelas delegate scene harus terdaftar sebelum UIKit mencarinya.
    #[cfg(target_os = "ios")]
    crate::apple_ios::register_scene_delegate();
    eframe::run_native(
        "HISTO3D",
        native_options(),
        Box::new(move |cc| Ok(Box::new(H3dApp::new(cc, args)))),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
}

struct Screenshot {
    path: PathBuf,
    requested: bool,
    done: bool,
}

/// Aplikasi eframe.
pub struct H3dApp {
    /// Keadaan aplikasi (bebas jendela).
    pub state: AppState,
    gpu: Option<Gpu>,
    shot: Option<Screenshot>,
}

impl H3dApp {
    /// Bangun aplikasi dari konteks eframe dan argumen baris perintah.
    pub fn new(cc: &eframe::CreationContext<'_>, args: CliArgs) -> Self {
        #[cfg(target_os = "ios")]
        {
            crate::apple_ios::attach_window_to_scene(cc);
            crate::apple_ios::install_memory_warning_observer();
        }
        let budgets = platform::app_budgets();
        let gpu = cc
            .wgpu_render_state
            .as_ref()
            .map(|rs| Gpu::install(rs, &budgets));
        if gpu.is_none() {
            log::error!("backend wgpu tidak tersedia; viewport tidak akan digambar");
        }

        let settings_path = Settings::default_path();
        let mut settings = Settings::load(&settings_path);
        if let Some(lang) = args.lang {
            settings.set_language(lang);
        }
        if let Some(theme) = args.theme {
            settings.set_theme(theme);
        }
        let mut state = AppState::new(settings, budgets);
        // Mode tangkapan layar tidak menimpa pengaturan pengguna.
        if args.screenshot.is_none() {
            state.settings_path = Some(settings_path);
        }
        state.apply_theme(&cc.egui_ctx);
        if let Some(path) = &args.project {
            state.open_project(path);
        }
        if let Some(mode) = args.mode {
            state.set_mode(mode);
        }
        if let (Some(n), Some(shared)) = (args.slide, state.session.clone()) {
            if let Some(s) = crate::state::try_lock_session(&shared) {
                if let Some(e) = s.slides().get(n - 1) {
                    state.current = Some(e.id);
                    state.slides.selected = Some(e.id);
                }
            }
        }
        if args.analyze {
            state.recon.want_first_scalar = true;
            state.start_job(h3d_i18n::t!("recon-analyze"), |s, p| {
                s.analyze_spatial(p)
                    .map(|_| crate::state::JobOutput::Analyzed)
            });
        }
        H3dApp {
            state,
            gpu,
            shot: args.screenshot.map(|path| Screenshot {
                path,
                requested: false,
                done: false,
            }),
        }
    }

    fn handle_screenshot(&mut self, ctx: &egui::Context) {
        let Some(shot) = &mut self.shot else {
            return;
        };
        if shot.done {
            return;
        }
        ctx.request_repaint();
        let image = ctx.input(|i| {
            i.events.iter().find_map(|e| match e {
                egui::Event::Screenshot { image, .. } => Some(image.clone()),
                _ => None,
            })
        });
        if let Some(image) = image {
            let [w, h] = image.size;
            let bytes: Vec<u8> = image.pixels.iter().flat_map(|c| c.to_array()).collect();
            let saved = image::RgbaImage::from_raw(w as u32, h as u32, bytes)
                .ok_or_else(|| "ukuran tangkapan layar tidak cocok".to_string())
                .and_then(|img| img.save(&shot.path).map_err(|e| e.to_string()));
            match saved {
                Ok(()) => log::info!("tangkapan layar: {}", shot.path.display()),
                Err(e) => eprintln!("histo3d: tangkapan layar gagal: {e}"),
            }
            shot.done = true;
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }
        let frames = self.state.frame_count;
        let settled = self.state.job.is_none() && !self.state.tiles.needs_repaint();
        if !shot.requested && ((frames >= SHOT_MIN_FRAMES && settled) || frames >= SHOT_MAX_FRAMES)
        {
            shot.requested = true;
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
        }
    }
}

impl eframe::App for H3dApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.state.ui(ui, self.gpu.as_ref());
        self.handle_screenshot(&ctx);
    }

    /// Dipanggil eframe saat OS menidurkan aplikasi (iPadOS) dan tiap
    /// `auto_save_interval`: simpan proyek dan label yang berubah (9.4).
    fn save(&mut self, _storage: &mut dyn eframe::Storage) {
        self.state.save_now();
    }

    fn auto_save_interval(&self) -> std::time::Duration {
        std::time::Duration::from_secs(if platform::is_mobile() { 20 } else { 30 })
    }

    fn persist_egui_memory(&self) -> bool {
        false
    }

    fn on_exit(&mut self) {
        self.state.save_now();
    }
}
