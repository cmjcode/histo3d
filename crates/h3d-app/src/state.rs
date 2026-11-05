//! Keadaan aplikasi: proyek terbuka (`SharedSession`), job latar, tampilan
//! aktif (tumpukan 3D atau irisan 2D), laci kanan, dan keadaan tiap fitur.
//! Tidak bergantung pada jendela maupun GPU sehingga bisa dibangun dan
//! digambar di tes headless.
//!
//! **Model thread** (lihat `h3d_engine::job`): thread job mengunci sesi
//! selama job berjalan, jadi thread UI hanya memakai `try_lock()`; selama
//! job berjalan kanvas tetap menggambar adegan terakhir dan kartu progres
//! tampil di bawah.
//!
//! **Susun otomatis** (alur impor → tumpukan 3D): setelah impor selesai,
//! `suggest_order` dijalankan dan hasilnya tampil sebagai pratinjau tembus
//! pandang di kanvas 3D. Urutan proyek baru berubah setelah pengguna menekan
//! Terima (aturan keras 8); setelah itu `register_all` dijalankan otomatis.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};

use h3d_core::{AttrId, Budgets, Progress, SlideId};
use h3d_engine::{
    EngineError, ImportOptions, JobHandle, OrderSuggestion, Proposal, RegOptions, RegSummary,
    Session, SharedSession,
};
use h3d_i18n::t;

use crate::features::{reconstruct, register, report, segment, slides};
use crate::gpu::Gpu;
use crate::panels::browser::BrowserState;
use crate::panels::mpp_prompt::MppPrompt;
use crate::settings::Settings;
use crate::tiles::TileStreamer;
use crate::ui_util::{ThumbCache, Toasts};
use crate::viewer::TouchTracker;

/// Jumlah irisan `Ok` minimum agar usulan urutan berarti.
pub const AUTO_STACK_MIN_SLIDES: usize = 3;

/// Tampilan utama kanvas.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum View {
    /// Tumpukan irisan (dan mesh tumor) di ruang 3D.
    #[default]
    Stack3D,
    /// Satu irisan di 2D untuk menandai tumor atau menyejajarkan.
    Slice2D,
}

/// Cara kerja di tampilan 2D.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum SliceMode {
    /// Menandai label (kuas, laso, poligon, usulan).
    #[default]
    Annotate,
    /// Menyejajarkan terhadap irisan tetangga (banding, landmark, geser/putar).
    Align,
}

impl SliceMode {
    /// Kunci terjemahan nama mode.
    pub fn i18n_key(self) -> &'static str {
        match self {
            SliceMode::Annotate => "slice-mode-annotate",
            SliceMode::Align => "slice-mode-align",
        }
    }
}

/// Laci kanan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Drawer {
    /// Daftar irisan (urutan, status, QC) dan label.
    Layers,
    /// Sampel, klinis, orientasi anatomis, rincian irisan terpilih.
    Sample,
    /// Parameter rekonstruksi, visibilitas label, bidang potong, lapisan skalar.
    Tumor3D,
    /// Label, usulan klasik/AI/interpolasi, kalibrasi IHC.
    Labels,
    /// Daftar QC dan pengaturan banding.
    Align,
}

impl Drawer {
    /// Semua laci, urutan tombol di bilah atas.
    pub const ALL: [Drawer; 5] = [
        Drawer::Layers,
        Drawer::Sample,
        Drawer::Labels,
        Drawer::Align,
        Drawer::Tumor3D,
    ];

    /// Kunci terjemahan judul laci.
    pub fn i18n_key(self) -> &'static str {
        match self {
            Drawer::Layers => "drawer-layers",
            Drawer::Sample => "drawer-sample",
            Drawer::Tumor3D => "drawer-tumor3d",
            Drawer::Labels => "drawer-labels",
            Drawer::Align => "drawer-align",
        }
    }

    /// Ikon laci.
    pub fn icon(self) -> h3d_ui::icons::MaterialIcon {
        use h3d_ui::icons;
        match self {
            Drawer::Layers => icons::LAYERS,
            Drawer::Sample => icons::SAMPLE,
            Drawer::Tumor3D => icons::CUBE,
            Drawer::Labels => icons::DRAW,
            Drawer::Align => icons::COMPARE,
        }
    }
}

/// Hasil job latar.
pub enum JobOutput {
    /// Irisan terimpor beserta galat berkas yang dilewati pada impor banyak berkas.
    Imported {
        /// Id irisan yang berhasil diimpor.
        ids: Vec<SlideId>,
        /// Galat dari berkas yang dilewati (mis. unduhan belum selesai).
        skipped: Vec<EngineError>,
    },
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
    /// Masukkan ke dalam bundel (wajib di iPad, 9.3).
    pub embed: bool,
    /// Cara memasukkan bila `embed`: salin atau pindahkan.
    pub embed_mode: h3d_engine::EmbedMode,
}

/// Keadaan aplikasi.
pub struct AppState {
    /// Pengaturan pengguna.
    pub settings: Settings,
    /// Tempat pengaturan disimpan; `None` = tidak disimpan (tes).
    pub settings_path: Option<PathBuf>,
    /// Anggaran memori/thread (dari `Budgets`, aturan 15).
    pub budgets: Budgets,
    /// Tampilan kanvas.
    pub view: View,
    /// Cara kerja di tampilan 2D.
    pub slice_mode: SliceMode,
    /// Laci kanan yang terbuka.
    pub drawer: Option<Drawer>,
    /// Lembar laporan di bawah kanvas terbuka.
    pub report_open: bool,
    /// Alur susun otomatis sedang berjalan: setelah usulan diterima,
    /// registrasi dijalankan tanpa klik tambahan.
    pub auto_stack_active: bool,
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
    /// Irisan terpilih (di tumpukan 3D, laci, dan tampilan 2D).
    pub current: Option<SlideId>,
    /// Pelacak Pencil/jari.
    pub touch: TouchTracker,
    /// Keadaan fitur Irisan (daftar, usulan urutan, sampel, klinis).
    pub slides: slides::SlidesState,
    /// Keadaan fitur Penyejajaran.
    pub register: register::RegisterState,
    /// Keadaan fitur Penandaan.
    pub segment: segment::SegmentState,
    /// Keadaan kanvas 3D dan rekonstruksi.
    pub recon: reconstruct::ReconState,
    /// Keadaan laporan.
    pub report: report::ReportState,
    /// Dialog Tentang terbuka.
    pub show_about: bool,
    /// Dialog Pengaturan terbuka.
    pub show_settings: bool,
    /// Pemeriksa pembaruan (dialog Tentang).
    pub update: crate::update::UpdateState,
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
            view: View::Stack3D,
            slice_mode: SliceMode::Annotate,
            drawer: None,
            report_open: false,
            auto_stack_active: false,
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
            update: Default::default(),
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
        self.view = View::Stack3D;
        self.drawer = None;
        self.report_open = false;
        self.auto_stack_active = false;
        self.mpp_prompt = None;
        self.pending_import = None;
        self.palette_key = 0;
        self.session = Some(Arc::new(Mutex::new(session)));
    }

    /// Buka proyek yang sudah ada, atau inisialisasi proyek baru bila `path`
    /// adalah folder berisi berkas slide. Galat ditampilkan sebagai pemberitahuan.
    pub fn open_project(&mut self, path: &Path) -> bool {
        if self.job.is_some() {
            self.toasts.error(t!("job-busy"));
            return false;
        }
        self.save_now();
        if !crate::platform::is_bundle(path) && path.is_dir() {
            let listing = crate::platform::list_documents(path);
            if !listing.slides.is_empty() {
                match Session::create(path) {
                    Ok(s) => {
                        self.install_session(s);
                        let paths = listing.slides.into_iter().map(|e| e.path).collect();
                        self.start_import(
                            ImportRequest {
                                paths,
                                embed: false,
                                embed_mode: h3d_engine::EmbedMode::Copy,
                            },
                            None,
                        );
                        return true;
                    }
                    Err(e) => {
                        self.toasts.engine_error(&e);
                        return false;
                    }
                }
            }
        }
        match Session::open(path) {
            Ok(s) => {
                let unimported: Vec<PathBuf> = crate::platform::list_documents(path)
                    .slides
                    .into_iter()
                    .map(|e| e.path)
                    .filter(|p| {
                        let rel = s.bundle().make_relative(p);
                        !s.slides().iter().any(|entry| {
                            matches!(
                                &entry.source,
                                h3d_core::SlideSourceRef::External { path: existing, .. }
                                    if *existing == rel
                            )
                        })
                    })
                    .collect();
                self.install_session(s);
                if !unimported.is_empty() {
                    self.start_import(
                        ImportRequest {
                            paths: unimported,
                            embed: false,
                            embed_mode: h3d_engine::EmbedMode::Copy,
                        },
                        None,
                    );
                }
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
        let embed_mode = request.embed_mode;
        self.pending_import = Some(request);
        self.start_job(t!("job-import"), move |s, p| {
            let opts = ImportOptions {
                mpp,
                embed,
                embed_mode,
                spacing_um: None,
            };
            let mut expanded: Vec<PathBuf> = Vec::with_capacity(paths.len());
            for path in &paths {
                if path.is_dir() && !crate::platform::is_bundle(path) {
                    let slides = crate::platform::list_documents(path).slides;
                    if !slides.is_empty() {
                        expanded.extend(slides.into_iter().map(|e| e.path));
                        continue;
                    }
                }
                expanded.push(path.clone());
            }
            if !embed && expanded.len() > 1 {
                let not_dup: Vec<PathBuf> = expanded
                    .iter()
                    .filter(|path| {
                        let rel = s.bundle().make_relative(path);
                        !s.slides().iter().any(|entry| {
                            matches!(
                                &entry.source,
                                h3d_core::SlideSourceRef::External { path: existing, .. }
                                    if *existing == rel
                            )
                        })
                    })
                    .cloned()
                    .collect();
                if !not_dup.is_empty() {
                    expanded = not_dup;
                }
            }
            match s.import_slides_with(&expanded, &opts, p) {
                Ok(ids) => {
                    s.save()?;
                    Ok(JobOutput::Imported {
                        ids,
                        skipped: Vec::new(),
                    })
                }
                Err(EngineError::Slide(first_err)) if expanded.len() > 1 => {
                    let mut valid = Vec::new();
                    let mut skipped = Vec::new();
                    for path in &expanded {
                        match h3d_slide::open_slide(path) {
                            Ok(_) => valid.push(path.clone()),
                            Err(e) => skipped.push(EngineError::Slide(e)),
                        }
                    }
                    if valid.is_empty() {
                        return Err(EngineError::Slide(first_err));
                    }
                    let ids = s.import_slides_with(&valid, &opts, p)?;
                    s.save()?;
                    Ok(JobOutput::Imported { ids, skipped })
                }
                Err(e) => Err(e),
            }
        });
    }

    /// Mulai "Susun otomatis": usulan urutan dan orientasi dari gambar
    /// resolusi rendah. Hasilnya menunggu Terima di kartu usulan; bila
    /// `then_register`, registrasi menyusul otomatis setelah diterima.
    pub fn start_auto_stack(&mut self, then_register: bool) {
        if self.start_job(t!("stack-auto"), |s, p| {
            s.suggest_order(p).map(JobOutput::OrderSuggested)
        }) {
            self.auto_stack_active = then_register;
        }
    }

    /// Mulai "Sejajarkan semua" (registrasi seluruh tumpukan, bisa dilanjutkan).
    pub fn start_register_all(&mut self) {
        self.start_job(t!("register-all"), |s, p| {
            s.register_all(&RegOptions::default(), p)
                .map(JobOutput::Registered)
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
                self.auto_stack_active = false;
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

    /// Jumlah irisan berstatus `Ok` di proyek (0 bila sesi terkunci).
    fn usable_slide_count(&self) -> usize {
        self.session
            .as_ref()
            .and_then(|s| try_lock_session(s).map(|s| s.project().usable_slides_ordered().len()))
            .unwrap_or(0)
    }

    fn apply_job_output(&mut self, out: JobOutput) {
        match out {
            JobOutput::Imported { ids, skipped } => {
                self.pending_import = None;
                self.toasts
                    .info(t!("slides-imported", n = ids.len().to_string()));
                for err in &skipped {
                    self.toasts.engine_error(err);
                }
                if let Some(first) = ids.first() {
                    self.current.get_or_insert(*first);
                }
                self.view = View::Stack3D;
                self.recon.fitted = false;
                // Susun otomatis: usulan urutan (≥ 3 irisan) lalu registrasi.
                if self.settings.auto_stack && !ids.is_empty() {
                    let n = self.usable_slide_count();
                    if n >= AUTO_STACK_MIN_SLIDES {
                        self.start_auto_stack(true);
                    } else if n >= 2 {
                        self.start_register_all();
                    }
                }
            }
            JobOutput::OrderSuggested(s) => {
                let preview = slides::OrderPreview::new(s);
                let unchanged = self
                    .session
                    .as_ref()
                    .and_then(|sh| {
                        try_lock_session(sh)
                            .map(|s| preview.changes(&AppState::ordered_ids(&s)).is_empty())
                    })
                    .unwrap_or(false);
                if unchanged {
                    self.toasts.info(t!("stack-proposal-no-change"));
                    if self.auto_stack_active {
                        self.auto_stack_active = false;
                        self.start_register_all();
                    }
                } else {
                    self.slides.order_preview = Some(preview);
                }
            }
            JobOutput::Registered(summary) => {
                self.auto_stack_active = false;
                self.toasts.info(t!(
                    "register-done",
                    registered = summary.registered.len().to_string(),
                    skipped = summary.skipped.len().to_string(),
                    flagged = summary.flagged.len().to_string()
                ));
                self.recon.fitted = false;
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

    /// Keputusan pengguna atas pratinjau usulan urutan: satu-satunya jalur
    /// usulan → urutan proyek (aturan keras 8). Setelah Terima dalam alur
    /// susun otomatis, registrasi dijalankan.
    pub fn decide_order(&mut self, sess: &mut Session, decision: slides::OrderDecision) {
        match decision {
            slides::OrderDecision::None => {}
            slides::OrderDecision::Reject => {
                self.slides.order_preview = None;
                self.auto_stack_active = false;
            }
            slides::OrderDecision::Accept(ids) => {
                let Some(preview) = self.slides.order_preview.take() else {
                    return;
                };
                match sess.apply_order_suggestion(&preview.suggestion, &ids) {
                    Ok(()) => {
                        if !sess.project().z_is_monotonic() {
                            self.toasts.error(t!("slides-z-not-monotonic"));
                        }
                        self.toasts.info(t!("slides-order-applied"));
                    }
                    Err(e) => self.toasts.engine_error(&e),
                }
                self.invalidate_derived();
                self.recon.fitted = false;
                if self.auto_stack_active {
                    self.auto_stack_active = false;
                    self.start_register_all();
                }
            }
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
        self.tiles.clear_cpu_cache();
        let released = self.tiles.release_sources();
        log::warn!("peringatan memori: cache tile dikosongkan, {released} sumber slide dilepas");
        self.thumbs.clear();
        self.segment.proposal_tex = None;
        if let Some(g) = gpu {
            g.clear_tile_caches();
        }
        self.toasts.info(t!("memory-trimmed"));
    }

    /// Pindah tampilan; turunan yang dibutuhkan tampilan tujuan disegarkan.
    pub fn set_view(&mut self, view: View) {
        if self.view != view {
            self.segment.cancel_drawing();
            self.register.reset_edit();
        }
        self.view = view;
        match view {
            View::Stack3D => self.recon.needs_reload = true,
            View::Slice2D => self.register.qc_dirty = true,
        }
    }

    /// Buka irisan `id` di tampilan 2D (klik ganda di tumpukan, laci, atau
    /// tombol "Buka 2D").
    pub fn open_slice_2d(&mut self, id: SlideId) {
        self.current = Some(id);
        self.set_view(View::Slice2D);
    }

    /// Buka/tutup laci; membuka laci lain menutup yang sedang terbuka.
    pub fn toggle_drawer(&mut self, drawer: Drawer) {
        self.drawer = if self.drawer == Some(drawer) {
            None
        } else {
            Some(drawer)
        };
        match drawer {
            Drawer::Align => self.register.qc_dirty = true,
            Drawer::Tumor3D => self.recon.needs_reload = true,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drawer_and_mode_keys_exist() {
        for lang in h3d_i18n::Language::all() {
            for d in Drawer::ALL {
                assert!(h3d_i18n::has_key(*lang, d.i18n_key()), "{d:?}");
            }
            for m in [SliceMode::Annotate, SliceMode::Align] {
                assert!(h3d_i18n::has_key(*lang, m.i18n_key()), "{m:?}");
            }
        }
    }

    #[test]
    fn toggle_drawer_opens_closes_and_switches() {
        let mut app = AppState::new(Settings::default(), Budgets::IPAD);
        assert_eq!(app.drawer, None);
        app.toggle_drawer(Drawer::Layers);
        assert_eq!(app.drawer, Some(Drawer::Layers));
        app.toggle_drawer(Drawer::Sample);
        assert_eq!(app.drawer, Some(Drawer::Sample));
        app.toggle_drawer(Drawer::Sample);
        assert_eq!(app.drawer, None);
    }

    #[test]
    fn open_slice_switches_to_2d_and_back_keeps_selection() {
        let mut app = AppState::new(Settings::default(), Budgets::IPAD);
        app.open_slice_2d(SlideId(3));
        assert_eq!(app.view, View::Slice2D);
        assert_eq!(app.current, Some(SlideId(3)));
        app.set_view(View::Stack3D);
        assert_eq!(app.current, Some(SlideId(3)));
        assert!(app.recon.needs_reload);
    }
}
