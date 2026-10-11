//! Keadaan aplikasi: proyek terbuka (`SharedSession`), job latar, mode,
//! dan keadaan tiap mode. Tidak bergantung pada jendela maupun GPU sehingga
//! bisa dibangun dan digambar di tes headless.
//!
//! **Model thread** (lihat `h3d_engine::job`): thread job mengunci sesi
//! selama job berjalan, jadi thread UI hanya memakai `try_lock()`; selama
//! job berjalan layar menampilkan lapisan job (tahap, progres, Batal).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};

use h3d_core::{AttrId, Budgets, Progress, SlideId};
use h3d_engine::{
    EngineError, ImportOptions, JobHandle, OrderSuggestion, Proposal, RegSummary, Session,
    SharedSession,
};
use h3d_i18n::t;
use h3d_ui::AppMode;

use crate::gpu::Gpu;
use crate::modes::{reconstruct, register, report, segment, slides};
use crate::panels::browser::BrowserState;
use crate::panels::mpp_prompt::MppPrompt;
use crate::settings::Settings;
use crate::tiles::TileStreamer;
use crate::ui_util::{ThumbCache, Toasts};
use crate::viewer::TouchTracker;

/// Hasil job latar.
pub enum JobOutput {
    /// Irisan terimpor.
    Imported(Vec<SlideId>),
    /// Usulan urutan dan orientasi (belum diterapkan; 8.4.7).
    OrderSuggested(OrderSuggestion),
    /// Registrasi selesai.
    Registered(RegSummary),
    /// Rekonstruksi selesai; jumlah mesh.
    Reconstructed(usize),
    /// Analisis spasial selesai.
    Analyzed,
    /// Usulan segmentasi klasik untuk satu irisan (belum diterima).
    ClassicProposal(SlideId, Box<Proposal>),
    /// Usulan tertunda (AI/interpolasi) sudah masuk antrean sesi.
    PendingProposals,
    /// Atribut dihitung.
    AttrComputed(AttrId),
    /// Berkas ekspor ditulis.
    Exported(Vec<PathBuf>),
    /// Inti terdeteksi untuk kalibrasi ambang IHC.
    Nuclei(Box<segment::NucleiOverlay>),
}

/// Job yang sedang berjalan.
pub struct RunningJob {
    /// Judul (sudah diterjemahkan).
    pub title: String,
    /// Pegangan job.
    pub handle: JobHandle<JobOutput>,
}

/// Permintaan impor yang bisa diulang setelah pengguna mengisi mpp.
#[derive(Clone, Debug, PartialEq)]
pub struct ImportRequest {
    /// Berkas slide.
    pub paths: Vec<PathBuf>,
    /// Salin ke dalam bundel (wajib di iPad, 9.3).
    pub embed: bool,
}

/// Keadaan aplikasi.
pub struct AppState {
    /// Pengaturan pengguna.
    pub settings: Settings,
    /// Tempat pengaturan disimpan; `None` = tidak disimpan (tes).
    pub settings_path: Option<PathBuf>,
    /// Anggaran memori/thread (dari `Budgets`, aturan 15).
    pub budgets: Budgets,
    /// Mode aktif.
    pub mode: AppMode,
    /// Proyek terbuka.
    pub session: Option<SharedSession>,
    /// Nama proyek untuk bilah atas.
    pub project_name: String,
    /// Job latar yang berjalan.
    pub job: Option<RunningJob>,
    /// Pemberitahuan.
    pub toasts: Toasts,
    /// Aliran tile viewer 2D.
    pub tiles: TileStreamer,
    /// Thumbnail irisan.
    pub thumbs: ThumbCache,
    /// Irisan aktif (mode Registrasi dan Segmentasi).
    pub current: Option<SlideId>,
    /// Pelacak Pencil/jari.
    pub touch: TouchTracker,
    /// Keadaan mode Irisan.
    pub slides: slides::SlidesState,
    /// Keadaan mode Registrasi.
    pub register: register::RegisterState,
    /// Keadaan mode Segmentasi.
    pub segment: segment::SegmentState,
    /// Keadaan mode Rekonstruksi.
    pub recon: reconstruct::ReconState,
    /// Keadaan mode Laporan.
    pub report: report::ReportState,
    /// Dialog Tentang terbuka.
    pub show_about: bool,
    /// Dialog Pengaturan terbuka.
    pub show_settings: bool,
    /// Peramban Documents (iPad; juga bisa dibuka di desktop).
    pub browser: Option<BrowserState>,
    /// Dialog isian mpp (impor melaporkan mpp hilang).
    pub mpp_prompt: Option<MppPrompt>,
    /// Impor terakhir, untuk diulang dengan mpp isian.
    pub pending_import: Option<ImportRequest>,
    /// Hash tabel label yang paletnya sudah dikirim ke GPU.
    pub palette_key: u64,
    /// Waktu simpan otomatis terakhir (detik egui).
    pub last_autosave: f64,
    /// Jumlah frame yang sudah digambar.
    pub frame_count: u64,
}

/// Kunci sesi tanpa memblokir. Mutex teracuni (job panik) tetap dipakai:
/// keadaan proyek di disk selalu ditulis atomik.
pub fn try_lock_session(shared: &SharedSession) -> Option<MutexGuard<'_, Session>> {
    match shared.try_lock() {
        Ok(g) => Some(g),
        Err(TryLockError::Poisoned(p)) => Some(p.into_inner()),
        Err(TryLockError::WouldBlock) => None,
    }
}

impl AppState {
    /// Keadaan awal tanpa proyek.
    pub fn new(settings: Settings, budgets: Budgets) -> Self {
        h3d_i18n::set_language(settings.language());
        AppState {
            tiles: TileStreamer::new(&budgets),
            settings,
            settings_path: None,
            budgets,
            mode: AppMode::Slides,
            session: None,
            project_name: String::new(),
            job: None,
            toasts: Toasts::default(),
            thumbs: ThumbCache::default(),
            current: None,
            touch: TouchTracker::default(),
            slides: Default::default(),
            register: Default::default(),
            segment: Default::default(),
            recon: Default::default(),
            report: Default::default(),
            show_about: false,
            show_settings: false,
            browser: None,
            mpp_prompt: None,
            pending_import: None,
            palette_key: 0,
            last_autosave: 0.0,
            frame_count: 0,
        }
    }

    /// Ada proyek terbuka.
    pub fn has_project(&self) -> bool {
        self.session.is_some()
    }

    fn install_session(&mut self, mut session: Session) {
        session.set_budgets(self.budgets);
        let initials = self.settings.user_initials.trim().to_owned();
        if !initials.is_empty() && session.project().user_initials != initials {
            session.set_user_initials(&initials);
        }
        self.project_name = session.project().sample.name.clone();
        if self.project_name.is_empty() {
            self.project_name = session
                .path()
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
        }
        self.current = session
            .project()
            .effective_reference()
            .or_else(|| session.slides().first().map(|s| s.id));
        self.tiles.reset();
        self.thumbs.clear();
        self.slides = Default::default();
        self.register = Default::default();
        self.segment = Default::default();
        self.recon = Default::default();
        self.report = Default::default();
        self.mpp_prompt = None;
        self.pending_import = None;
        self.palette_key = 0;
        self.session = Some(Arc::new(Mutex::new(session)));
    }

    /// Buka proyek yang sudah ada. Galat ditampilkan sebagai pemberitahuan.
    pub fn open_project(&mut self, path: &Path) -> bool {
        if self.job.is_some() {
            self.toasts.error(t!("job-busy"));
            return false;
        }
        self.save_now();
        match Session::open(path) {
            Ok(s) => {
                self.install_session(s);
                true
            }
            Err(e) => {
                self.toasts.engine_error(&e);
                false
            }
        }
    }

    /// Buat proyek baru di `path` (folder `.h3d`).
    pub fn create_project(&mut self, path: &Path) -> bool {
        if self.job.is_some() {
            self.toasts.error(t!("job-busy"));
            return false;
        }
        self.save_now();
        match Session::create(path) {
            Ok(s) => {
                self.install_session(s);
                self.mode = AppMode::Slides;
                true
            }
            Err(e) => {
                self.toasts.engine_error(&e);
                false
            }
        }
    }

    /// Simpan proyek dan label yang berubah sekarang juga (tombol Simpan,
    /// `eframe::App::save` saat aplikasi masuk latar belakang; 9.4). Bila
    /// job sedang memegang sesi, penyimpanan dilewati: job menyimpan
    /// hasilnya sendiri per irisan.
    pub fn save_now(&mut self) {
        let Some(shared) = self.session.clone() else {
            return;
        };
        let Some(mut s) = try_lock_session(&shared) else {
            return;
        };
        if s.is_dirty() {
            if let Err(e) = s.save() {
                self.toasts.engine_error(&e);
            }
        }
    }

    /// Simpan pengaturan (bila ada tempatnya) dan terapkan bahasa.
    pub fn persist_settings(&mut self) {
        h3d_i18n::set_language(self.settings.language());
        if let Some(path) = &self.settings_path {
            if let Err(e) = self.settings.save(path) {
                log::warn!("pengaturan tidak tersimpan: {e}");
            }
        }
    }

    /// Jalankan operasi sesi di thread latar. Ditolak bila job lain masih
    /// berjalan atau tidak ada proyek.
    pub fn start_job<F>(&mut self, title: String, f: F) -> bool
    where
        F: FnOnce(&mut Session, &dyn Progress) -> Result<JobOutput, EngineError> + Send + 'static,
    {
        if self.job.is_some() {
            self.toasts.error(t!("job-busy"));
            return false;
        }
        let Some(shared) = &self.session else {
            return false;
        };
        let handle = h3d_engine::spawn_session_job(shared, "ui", f);
        self.job = Some(RunningJob { title, handle });
        true
    }

    /// Mulai impor slide. Bila metadata tidak memuat mpp, job gagal dengan
    /// `MppMissing` dan dialog isian mpp dibuka (mpp tidak pernah ditebak).
    pub fn start_import(&mut self, request: ImportRequest, mpp: Option<[f64; 2]>) {
        if request.paths.is_empty() {
            return;
        }
        let paths = request.paths.clone();
        let embed = request.embed;
        self.pending_import = Some(request);
        self.start_job(t!("job-import"), move |s, p| {
            let opts = ImportOptions {
                mpp,
                embed,
                spacing_um: None,
            };
            let ids = s.import_slides_with(&paths, &opts, p)?;
            s.save()?;
            Ok(JobOutput::Imported(ids))
        });
    }

    /// Periksa job; bila selesai, terapkan hasilnya ke keadaan UI.
    /// Mengembalikan `true` selama job masih berjalan.
    pub fn poll_job(&mut self) -> bool {
        let Some(job) = &self.job else {
            return false;
        };
        let result = match job.handle.result.try_recv() {
            Ok(r) => r,
            Err(std::sync::mpsc::TryRecvError::Empty) => return true,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                Err(EngineError::invalid(t!("job-lost")))
            }
        };
        self.job = None;
        match result {
            Ok(out) => self.apply_job_output(out),
            Err(EngineError::MppMissing { file }) if self.pending_import.is_some() => {
                if let Some(request) = self.pending_import.take() {
                    self.mpp_prompt = Some(MppPrompt::new(request, file));
                }
            }
            Err(e) => {
                self.pending_import = None;
                self.toasts.engine_error(&e);
            }
        }
        // Job apa pun bisa mengubah proyek: segarkan turunan yang di-cache.
        self.invalidate_derived();
        false
    }

    /// Tunggu job sampai selesai (tes dan mode tangkapan layar).
    pub fn wait_job(&mut self) {
        while self.poll_job() {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    fn apply_job_output(&mut self, out: JobOutput) {
        match out {
            JobOutput::Imported(ids) => {
                self.pending_import = None;
                self.toasts
                    .info(t!("slides-imported", n = ids.len().to_string()));
                if let Some(first) = ids.first() {
                    self.slides.selected = Some(*first);
                    self.current.get_or_insert(*first);
                }
            }
            JobOutput::OrderSuggested(s) => {
                self.slides.order_preview = Some(slides::OrderPreview::new(s));
            }
            JobOutput::Registered(summary) => {
                self.toasts.info(t!(
                    "register-done",
                    registered = summary.registered.len().to_string(),
                    skipped = summary.skipped.len().to_string(),
                    flagged = summary.flagged.len().to_string()
                ));
            }
            JobOutput::Reconstructed(n) => {
                self.toasts.info(t!("recon-done", n = n.to_string()));
            }
            JobOutput::Analyzed => self.toasts.info(t!("recon-analyze-done")),
            JobOutput::ClassicProposal(slide, proposal) => {
                self.segment.set_classic(slide, *proposal);
            }
            JobOutput::PendingProposals => self.toasts.info(t!("segment-proposals-ready")),
            JobOutput::AttrComputed(_) => self.toasts.info(t!("attr-done")),
            JobOutput::Exported(paths) => {
                let dir = paths
                    .first()
                    .and_then(|p| p.parent())
                    .map(|p| p.display().to_string())
                    .unwrap_or_default();
                self.report.last_export = Some(dir.clone());
                self.toasts
                    .info(t!("export-done", n = paths.len().to_string(), dir = dir));
            }
            JobOutput::Nuclei(n) => self.segment.nuclei = Some(*n),
        }
    }

    /// Tandai semua turunan proyek (mesh, laporan, tile label, tabel QC)
    /// perlu dimuat ulang.
    pub fn invalidate_derived(&mut self) {
        self.recon.needs_reload = true;
        self.report.needs_refresh = true;
        self.register.qc_dirty = true;
        self.segment.proposal_tex = None;
        if let Some(cur) = self.current {
            self.tiles.invalidate_labels(cur);
        }
    }

    /// Setelah label irisan berubah lewat jalur selain goresan (undo, redo,
    /// usulan diterima, salin, isi lubang).
    pub fn labels_changed(&mut self, slide: SlideId) {
        self.tiles.invalidate_labels(slide);
        self.recon.needs_reload = true;
        self.report.needs_refresh = true;
    }

    /// Kait peringatan memori (9.4): kosongkan cache tile CPU dan GPU serta
    /// thumbnail. Dipanggil saat `platform::take_memory_warning()` benar;
    /// juga bisa dipanggil langsung.
    pub fn on_memory_warning(&mut self, gpu: Option<&Gpu>) {
        log::warn!("peringatan memori: cache tile dikosongkan");
        self.tiles.clear_cpu_cache();
        self.thumbs.clear();
        self.segment.proposal_tex = None;
        if let Some(g) = gpu {
            g.clear_tile_caches();
        }
        self.toasts.info(t!("memory-trimmed"));
    }

    /// Pindah mode; turunan yang dibutuhkan mode tujuan disegarkan.
    pub fn set_mode(&mut self, mode: AppMode) {
        self.mode = mode;
        match mode {
            AppMode::Reconstruct => self.recon.needs_reload = true,
            AppMode::Report => self.report.needs_refresh = true,
            AppMode::Register => self.register.qc_dirty = true,
            _ => {}
        }
    }

    /// Irisan terurut (id) dari sesi.
    pub fn ordered_ids(session: &Session) -> Vec<SlideId> {
        session.slides().iter().map(|s| s.id).collect()
    }

    /// Pastikan `current` menunjuk irisan yang masih ada.
    pub fn fix_current(&mut self, session: &Session) {
        let ok = self.current.is_some_and(|c| session.slide(c).is_ok());
        if !ok {
            self.current = session
                .project()
                .effective_reference()
                .or_else(|| session.slides().first().map(|s| s.id));
        }
    }
}
