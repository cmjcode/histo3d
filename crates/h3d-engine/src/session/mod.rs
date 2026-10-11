//! `Session`: proyek terbuka + bundel + cache + undo (8.8).

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::{Arc, Mutex};

use h3d_core::{
    Budgets, ClinicalMeta, IhcPanel, PreanalyticMeta, Project, ReconConfig, SlideEntry, SlideId,
    SlideStatus, SpatialConfig, SpecimenOrientation, Stain, UndoStack,
};
use h3d_io::ProjectBundle;
use h3d_segment::LabelStore;
use h3d_volume::{GridPlan, LabelMesh, SpatialAnalysis, VoxelMask};

use crate::clock::{Clock, SystemClock};
use crate::error::EngineError;

mod attrs;
mod export;
mod figures;
mod images;
mod import;
mod labels;
mod recon;
mod register;
mod spatial;

pub use attrs::AttrParams;
pub use export::ExportKind;
pub use figures::{MarginFigure, FIGURE_MARKER_RADIUS, FIGURE_MAX_SIDE};
pub use images::REG_IMAGE_MPP;
pub use import::ImportOptions;
pub use labels::PendingProposal;
pub use register::{QcRow, RegSummary, RegisterRun};

/// Versi aplikasi yang dicetak di laporan.
pub const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Label satu irisan di memori beserta status kotor dan riwayat undo-nya.
pub(crate) struct LabelSlot {
    pub(crate) store: LabelStore,
    pub(crate) dirty: bool,
    pub(crate) undo: UndoStack<LabelStore>,
}

/// Hasil rekonstruksi terakhir (juga tersimpan di `cache/vox/`).
#[derive(Clone)]
pub(crate) struct ReconState {
    pub(crate) key: u64,
    pub(crate) plan: GridPlan,
    /// Masker voxel semua label yang ada, terurut id.
    pub(crate) masks: Vec<(h3d_core::LabelId, VoxelMask)>,
    /// Mesh label ber-`reconstruct`, terurut id.
    pub(crate) meshes: Vec<LabelMesh>,
}

/// Proyek terbuka. `Send`: lihat modul [`crate::job`] untuk cara GUI
/// menjalankan operasi panjang di thread latar.
pub struct Session {
    pub(crate) bundle: ProjectBundle,
    pub(crate) project: Project,
    pub(crate) budgets: Budgets,
    pub(crate) clock: Arc<dyn Clock>,
    pub(crate) labels: BTreeMap<SlideId, LabelSlot>,
    pub(crate) pending: BTreeMap<SlideId, Vec<PendingProposal>>,
    pub(crate) recon: Option<ReconState>,
    pub(crate) analysis: Option<(u64, SpatialAnalysis)>,
    pub(crate) attr_notes: Vec<String>,
    pub(crate) project_dirty: bool,
    /// Irisan `External` yang berkasnya tidak cocok dengan sidik jari saat
    /// impor (diisi saat proyek dibuka dan tiap kali pembacaan mendapati
    /// ketidakcocokan). `Mutex` karena pembacaan piksel berjalan lewat `&self`.
    pub(crate) changed: Mutex<BTreeSet<SlideId>>,
}

// Jaminan waktu-kompilasi bahwa sesi boleh dipindah ke thread job.
const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<Session>();
};

impl Session {
    fn with(bundle: ProjectBundle, project: Project) -> Self {
        Session {
            bundle,
            project,
            budgets: Budgets::for_platform(),
            clock: Arc::new(SystemClock),
            labels: BTreeMap::new(),
            pending: BTreeMap::new(),
            recon: None,
            analysis: None,
            attr_notes: Vec::new(),
            project_dirty: false,
            changed: Mutex::new(BTreeSet::new()),
        }
    }

    /// Buat proyek baru di folder `path` (biasanya berakhiran `.h3d`). Nama
    /// sampel diambil dari nama folder. Galat bila proyek sudah ada.
    pub fn create(path: &Path) -> Result<Self, EngineError> {
        let bundle = ProjectBundle::create(path)?;
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let project = Project::new(&name);
        bundle.save_project(&project)?;
        Ok(Self::with(bundle, project))
    }

    /// Buka proyek yang sudah ada. Berkas rusak atau versi terlalu baru →
    /// galat. `clinical.json` dimuat bila ada.
    ///
    /// Sidik jari tiap berkas slide `External` dibandingkan dengan yang
    /// tersimpan saat impor. Ketidakcocokan **tidak** menggagalkan pembukaan
    /// proyek: irisannya masuk [`Self::changed_slides`] dan pikselnya ditolak
    /// dibaca ([`EngineError::SlideChanged`]) sampai pengguna menautkan ulang
    /// atau menerima berkas itu. Berkas yang hilang ada di
    /// [`Self::missing_slides`].
    pub fn open(path: &Path) -> Result<Self, EngineError> {
        let bundle = ProjectBundle::open(path)?;
        let project = bundle.load_project()?;
        if let Some(o) = &project.sample.orientation {
            o.validate()?;
        }
        let s = Self::with(bundle, project);
        s.recheck_slides();
        Ok(s)
    }

    /// Simpan `project.json` dan semua label yang berubah (atomik).
    /// Data klinis tidak ditulis di sini; lihat [`Self::set_clinical`].
    pub fn save(&mut self) -> Result<(), EngineError> {
        let dirty: Vec<SlideId> = self
            .labels
            .iter()
            .filter(|(_, s)| s.dirty)
            .map(|(id, _)| *id)
            .collect();
        for id in dirty {
            self.save_labels(id)?;
        }
        self.bundle.save_project(&self.project)?;
        self.project_dirty = false;
        Ok(())
    }

    /// Simpan hanya `project.json`.
    pub(crate) fn save_project(&mut self) -> Result<(), EngineError> {
        self.bundle.save_project(&self.project)?;
        self.project_dirty = false;
        Ok(())
    }

    // ---- akses ----

    /// Manifes proyek (hanya-baca).
    pub fn project(&self) -> &Project {
        &self.project
    }

    /// Bundel proyek.
    pub fn bundle(&self) -> &ProjectBundle {
        &self.bundle
    }

    /// Folder proyek.
    pub fn path(&self) -> &Path {
        &self.bundle.root
    }

    /// Benar bila ada perubahan manifes atau label yang belum disimpan.
    pub fn is_dirty(&self) -> bool {
        self.project_dirty || self.labels.values().any(|s| s.dirty)
    }

    /// Irisan terurut menurut posisi tumpukan.
    pub fn slides(&self) -> Vec<&SlideEntry> {
        self.project.slides_ordered()
    }

    /// Satu irisan.
    pub fn slide(&self, id: SlideId) -> Result<&SlideEntry, EngineError> {
        self.project.slide(id).ok_or(EngineError::SlideNotFound(id))
    }

    pub(crate) fn slide_mut(&mut self, id: SlideId) -> Result<&mut SlideEntry, EngineError> {
        self.project_dirty = true;
        self.project
            .slide_mut(id)
            .ok_or(EngineError::SlideNotFound(id))
    }

    /// Anggaran memori/thread yang dipakai rekonstruksi.
    pub fn budgets(&self) -> Budgets {
        self.budgets
    }

    /// Ganti anggaran (dipilih `h3d-app` saat mulai).
    pub fn set_budgets(&mut self, budgets: Budgets) {
        self.budgets = budgets;
    }

    /// Ganti sumber waktu provenance (tes, jalan emas CLI).
    pub fn set_clock(&mut self, clock: Arc<dyn Clock>) {
        self.clock = clock;
    }

    pub(crate) fn now(&self) -> String {
        self.clock.now_iso8601()
    }

    /// Catatan dari penghitungan atribut terakhir (mis. "TPS/CPS tidak tersedia").
    pub fn attr_notes(&self) -> &[String] {
        &self.attr_notes
    }

    // ---- penyetel metadata ----

    /// Isi mpp irisan (µm/px level 0). Registrasi irisan itu dan irisan yang
    /// didaftarkan kepadanya dibuang karena skala fisiknya berubah. Ditolak
    /// bila irisan sudah punya label: label tersimpan dalam µm, sehingga
    /// mengganti mpp akan menggesernya diam-diam dari jaringan.
    pub fn set_mpp(&mut self, slide: SlideId, mpp: [f64; 2]) -> Result<(), EngineError> {
        if !mpp.iter().all(|m| m.is_finite() && *m > 0.0) {
            return Err(EngineError::invalid(format!(
                "mpp tidak sah: {mpp:?} (harus positif)"
            )));
        }
        if self.slide(slide)?.mpp == mpp {
            return Ok(());
        }
        let has_labels = match self.labels.get(&slide) {
            Some(slot) => !slot.store.is_empty(),
            None => self
                .bundle
                .load_labels(slide)?
                .is_some_and(|st| !st.is_empty()),
        };
        if has_labels {
            return Err(EngineError::invalid(format!(
                "irisan {slide} sudah punya label; mpp tidak bisa diganti tanpa menggeser label \
                 (hapus labelnya dulu atau impor ulang irisan)"
            )));
        }
        self.labels.remove(&slide);
        let s = self.slide_mut(slide)?;
        s.mpp = mpp;
        s.reg = None;
        for other in &mut self.project.slides {
            if other.reg.as_ref().is_some_and(|r| r.fixed == slide) {
                other.reg = None;
            }
        }
        self.recompose_chain();
        Ok(())
    }

    /// Ganti jarak irisan bawaan; `z_um` yang masih bawaan dihitung ulang,
    /// isian tangan tidak disentuh (5.5).
    pub fn set_spacing(&mut self, spacing_um: f64) -> Result<(), EngineError> {
        if !(spacing_um.is_finite() && spacing_um > 0.0) {
            return Err(EngineError::invalid(format!(
                "jarak irisan tidak sah: {spacing_um}"
            )));
        }
        self.project.recon.default_spacing_um = spacing_um;
        self.project.recompute_default_z(spacing_um);
        self.project_dirty = true;
        Ok(())
    }

    /// Isi `z_um` satu irisan dengan tangan (tidak lagi dihitung otomatis).
    pub fn set_z(&mut self, slide: SlideId, z_um: f64) -> Result<(), EngineError> {
        if !z_um.is_finite() {
            return Err(EngineError::invalid("z_um tidak sah"));
        }
        let s = self.slide_mut(slide)?;
        s.z_um = z_um;
        s.z_is_default = false;
        Ok(())
    }

    /// Ubah status irisan. Catatan registrasi yang tetangganya berubah
    /// dibuang; `z_um` irisan lain tidak bergeser (5.5).
    pub fn set_status(&mut self, slide: SlideId, status: SlideStatus) -> Result<(), EngineError> {
        self.slide_mut(slide)?.status = status;
        self.drop_stale_registrations();
        self.recompose_chain();
        Ok(())
    }

    /// Pilih irisan acuan (`None` = irisan `Ok` di tengah tumpukan).
    pub fn set_reference(&mut self, reference: Option<SlideId>) -> Result<(), EngineError> {
        if let Some(r) = reference {
            if self.slide(r)?.status != SlideStatus::Ok {
                return Err(EngineError::invalid(format!(
                    "irisan {r} tidak berstatus Ok sehingga tidak bisa menjadi acuan"
                )));
            }
        }
        self.project.reference = reference;
        self.project_dirty = true;
        self.drop_stale_registrations();
        self.recompose_chain();
        Ok(())
    }

    /// Atur urutan dengan tangan. `order` harus memuat semua irisan tepat
    /// sekali. `z_um` bawaan dihitung ulang; catatan registrasi yang
    /// tetangganya berubah dibuang. Label tidak terpengaruh.
    pub fn reorder(&mut self, order: &[SlideId]) -> Result<(), EngineError> {
        self.check_permutation(order)?;
        for (i, id) in order.iter().enumerate() {
            let s = self.slide_mut(*id)?;
            if s.order != i as u32 {
                s.order = i as u32;
                s.order_source = h3d_core::OrderSource::Manual;
            }
        }
        let spacing = self.project.recon.default_spacing_um;
        self.project.recompute_default_z(spacing);
        self.drop_stale_registrations();
        self.recompose_chain();
        Ok(())
    }

    pub(crate) fn check_permutation(&self, order: &[SlideId]) -> Result<(), EngineError> {
        let mut a: Vec<SlideId> = order.to_vec();
        a.sort();
        let mut b: Vec<SlideId> = self.project.slides.iter().map(|s| s.id).collect();
        b.sort();
        if a != b {
            return Err(EngineError::invalid(
                "urutan harus memuat semua irisan proyek tepat satu kali",
            ));
        }
        Ok(())
    }

    /// Atur pewarnaan dan panel IHC irisan (8.12.3).
    pub fn set_stain(
        &mut self,
        slide: SlideId,
        stain: Stain,
        ihc: Option<IhcPanel>,
    ) -> Result<(), EngineError> {
        let s = self.slide_mut(slide)?;
        s.stain = stain;
        s.ihc = match (stain, ihc) {
            (Stain::Ihc, Some(p)) => Some(p),
            (Stain::Ihc, None) => Some(s.ihc.clone().unwrap_or_default()),
            (_, p) => p,
        };
        Ok(())
    }

    /// Isi metadata pra-analitik irisan.
    pub fn set_preanalytic(
        &mut self,
        slide: SlideId,
        meta: PreanalyticMeta,
    ) -> Result<(), EngineError> {
        self.slide_mut(slide)?.preanalytic = meta;
        Ok(())
    }

    /// Nama/inisial pengguna untuk provenance `Manual`.
    pub fn set_user_initials(&mut self, initials: &str) {
        self.project.user_initials = initials.trim().to_owned();
        self.project_dirty = true;
    }

    /// Nama dan catatan sampel.
    pub fn set_sample_info(&mut self, name: &str, notes: &str) {
        self.project.sample.name = name.to_owned();
        self.project.sample.notes = notes.to_owned();
        self.project_dirty = true;
    }

    /// Faktor susut linear (8.6.6). Hanya dilaporkan, tidak pernah diterapkan
    /// ke geometri.
    pub fn set_shrinkage(&mut self, factor: Option<f64>) -> Result<(), EngineError> {
        if let Some(f) = factor {
            if !(f.is_finite() && f > 0.0) {
                return Err(EngineError::invalid(format!("faktor susut tidak sah: {f}")));
            }
        }
        self.project.sample.shrinkage_linear = factor;
        self.project_dirty = true;
        Ok(())
    }

    /// Orientasi anatomis spesimen (divalidasi: tiga sumbu dari pasangan berbeda).
    pub fn set_orientation(
        &mut self,
        orientation: Option<SpecimenOrientation>,
    ) -> Result<(), EngineError> {
        if let Some(o) = &orientation {
            o.validate()?;
        }
        self.project.sample.orientation = orientation;
        self.project_dirty = true;
        Ok(())
    }

    /// Ganti konfigurasi rekonstruksi (cache voxel otomatis basi karena
    /// kuncinya memuat konfigurasi ini).
    pub fn set_recon_config(&mut self, cfg: ReconConfig) -> Result<(), EngineError> {
        let ok = cfg.s_xy_um.is_finite()
            && cfg.s_xy_um > 0.0
            && cfg.mpp_label.is_finite()
            && cfg.mpp_label > 0.0
            && cfg.band_px.is_finite()
            && cfg.band_px > 0.0
            && cfg.z_factor_max.is_finite()
            && cfg.z_factor_max > 0.0
            && cfg.decimate_error_factor.is_finite()
            && cfg.decimate_error_factor >= 0.0
            && cfg.default_spacing_um.is_finite()
            && cfg.default_spacing_um > 0.0;
        if !ok {
            return Err(EngineError::invalid("konfigurasi rekonstruksi tidak sah"));
        }
        self.project.recon = cfg;
        self.project_dirty = true;
        Ok(())
    }

    /// Ganti konfigurasi analisis spasial.
    pub fn set_spatial_config(&mut self, cfg: SpatialConfig) {
        self.project.spatial = cfg;
        self.project_dirty = true;
    }

    /// Isi atau hapus data klinis (8.12.2). Langsung menulis/menghapus
    /// `clinical.json`; `project.json` tidak pernah memuatnya.
    pub fn set_clinical(&mut self, c: Option<ClinicalMeta>) -> Result<(), EngineError> {
        if let Some(c) = &c {
            if !c.month_format_ok() {
                return Err(EngineError::invalid(
                    "bulan spesimen harus berbentuk YYYY-MM (tanpa hari)",
                ));
            }
        }
        self.bundle.save_clinical(c.as_ref())?;
        self.project.sample.clinical = c;
        Ok(())
    }

    /// Hapus irisan dari proyek beserta berkas label dan atributnya. Berkas
    /// slide `External` tidak disentuh.
    pub fn remove_slide(&mut self, slide: SlideId) -> Result<(), EngineError> {
        self.slide(slide)?;
        self.project.slides.retain(|s| s.id != slide);
        self.project.annotations.retain(|a| a.slide != slide);
        if self.project.reference == Some(slide) {
            self.project.reference = None;
        }
        self.labels.remove(&slide);
        self.pending.remove(&slide);
        self.changed_set().remove(&slide);
        h3d_io::remove_if_exists(&self.bundle.label_path(slide))?;
        for a in &self.project.attrs {
            self.bundle.remove_attr(slide, &a.file_stem())?;
        }
        // Rapatkan nomor urut agar tetap 0..n.
        let ids: Vec<SlideId> = self.project.slides_ordered().iter().map(|s| s.id).collect();
        for (i, id) in ids.iter().enumerate() {
            if let Some(s) = self.project.slide_mut(*id) {
                s.order = i as u32;
            }
        }
        self.drop_stale_registrations();
        self.recompose_chain();
        self.save_project()
    }

    /// Hapus `cache/` (piramida, gambar registrasi, thumbnail, voxel).
    pub fn clear_cache(&mut self) -> Result<(), EngineError> {
        self.recon = None;
        self.bundle.clear_cache()?;
        Ok(())
    }

    // ---- potret label hanya-baca ----

    /// Label semua irisan: yang ada di memori dipinjam (termasuk suntingan
    /// yang belum disimpan), sisanya dimuat dari disk. Irisan tanpa label
    /// tidak ikut.
    pub(crate) fn label_snapshot(
        &self,
    ) -> Result<Vec<(SlideId, Cow<'_, LabelStore>)>, EngineError> {
        let mut out = Vec::new();
        for s in self.project.slides_ordered() {
            if let Some(slot) = self.labels.get(&s.id) {
                if !slot.store.is_empty() {
                    out.push((s.id, Cow::Borrowed(&slot.store)));
                }
            } else if let Some(store) = self.bundle.load_labels(s.id)? {
                if !store.is_empty() {
                    out.push((s.id, Cow::Owned(store)));
                }
            }
        }
        Ok(out)
    }
}
