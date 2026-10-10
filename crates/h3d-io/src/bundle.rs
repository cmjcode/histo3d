//! Bundel proyek `.h3d` (histo3d.md 8.7.1, 8.12.9).
//!
//! ```text
//! sampel-01.h3d/
//! ├── project.json        manifes (struct Project), TANPA data klinis
//! ├── clinical.json       ClinicalMeta pseudonim; opsional
//! ├── slides/             hanya bila Embedded
//! ├── labels/NNNN.lbl     LabelStore per irisan
//! ├── attrs/NNNN.<stem>.atr
//! ├── models/
//! ├── cache/{pyramid,reg,thumbs,vox}/   bisa dihapus kapan saja
//! └── out/{mesh,volume,figures}/, analysis.json, report.md, report.csv
//! ```
//!
//! Semua penulisan atomik (`.tmp` lalu `rename`).

use std::fs;
use std::path::{Component, Path, PathBuf};

use h3d_core::{ClinicalMeta, Project, SlideId, SlideSourceRef, FORMAT_VERSION};
use h3d_segment::{decode_atr, decode_lbl, encode_atr, encode_lbl, AttrStore, LabelStore};
use serde_json::Value;

use crate::atomic::{copy_atomic, remove_if_exists, write_atomic};
use crate::error::IoError;

/// Nama berkas manifes.
pub const PROJECT_FILE: &str = "project.json";
/// Nama berkas data klinis.
pub const CLINICAL_FILE: &str = "clinical.json";
/// Cadangan ruang kosong yang harus tersisa setelah menyalin slide (byte).
pub const EMBED_FREE_SPACE_MARGIN: u64 = 64 * 1024 * 1024;

/// Folder proyek `.h3d` beserta pembantu path-nya.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectBundle {
    /// Folder akar bundel.
    pub root: PathBuf,
}

/// Nama berkas berawalan id irisan empat digit, mis. `0007`.
pub fn slide_stem(id: SlideId) -> String {
    format!("{:04}", id.0)
}

impl ProjectBundle {
    /// Buat bundel baru di `root` (folder dibuat; galat bila sudah berisi
    /// `project.json`). Manifes belum ditulis; panggil [`Self::save_project`].
    pub fn create(root: &Path) -> Result<Self, IoError> {
        let b = ProjectBundle {
            root: root.to_path_buf(),
        };
        if b.project_json().exists() {
            return Err(IoError::invalid(
                b.project_json(),
                "proyek sudah ada di folder ini",
            ));
        }
        fs::create_dir_all(root).map_err(|e| IoError::io(root, e))?;
        Ok(b)
    }

    /// Buka bundel yang sudah ada (harus memuat `project.json`).
    pub fn open(root: &Path) -> Result<Self, IoError> {
        let b = ProjectBundle {
            root: root.to_path_buf(),
        };
        if !b.project_json().is_file() {
            return Err(IoError::MissingFile(b.project_json()));
        }
        Ok(b)
    }

    // ---- path ----

    /// `project.json`.
    pub fn project_json(&self) -> PathBuf {
        self.root.join(PROJECT_FILE)
    }
    /// `clinical.json`.
    pub fn clinical_json(&self) -> PathBuf {
        self.root.join(CLINICAL_FILE)
    }
    /// `slides/`.
    pub fn slides_dir(&self) -> PathBuf {
        self.root.join("slides")
    }
    /// `labels/`.
    pub fn labels_dir(&self) -> PathBuf {
        self.root.join("labels")
    }
    /// `labels/NNNN.lbl`.
    pub fn label_path(&self, slide: SlideId) -> PathBuf {
        self.labels_dir().join(format!("{}.lbl", slide_stem(slide)))
    }
    /// `attrs/`.
    pub fn attrs_dir(&self) -> PathBuf {
        self.root.join("attrs")
    }
    /// `attrs/NNNN.<stem>.atr`.
    pub fn attr_path(&self, slide: SlideId, stem: &str) -> PathBuf {
        self.attrs_dir()
            .join(format!("{}.{}.atr", slide_stem(slide), stem))
    }
    /// `models/`.
    pub fn models_dir(&self) -> PathBuf {
        self.root.join("models")
    }
    /// `cache/`.
    pub fn cache_dir(&self) -> PathBuf {
        self.root.join("cache")
    }
    /// `cache/pyramid/`.
    pub fn cache_pyramid_dir(&self) -> PathBuf {
        self.cache_dir().join("pyramid")
    }
    /// `cache/reg/`.
    pub fn cache_reg_dir(&self) -> PathBuf {
        self.cache_dir().join("reg")
    }
    /// `cache/thumbs/`.
    pub fn cache_thumbs_dir(&self) -> PathBuf {
        self.cache_dir().join("thumbs")
    }
    /// `cache/vox/`.
    pub fn cache_vox_dir(&self) -> PathBuf {
        self.cache_dir().join("vox")
    }
    /// `out/`.
    pub fn out_dir(&self) -> PathBuf {
        self.root.join("out")
    }
    /// `out/mesh/`.
    pub fn out_mesh_dir(&self) -> PathBuf {
        self.out_dir().join("mesh")
    }
    /// `out/volume/`.
    pub fn out_volume_dir(&self) -> PathBuf {
        self.out_dir().join("volume")
    }
    /// `out/figures/`.
    pub fn out_figures_dir(&self) -> PathBuf {
        self.out_dir().join("figures")
    }
    /// `out/analysis.json`.
    pub fn analysis_json(&self) -> PathBuf {
        self.out_dir().join("analysis.json")
    }
    /// `out/report.md`.
    pub fn report_md(&self) -> PathBuf {
        self.out_dir().join("report.md")
    }
    /// `out/report.csv`.
    pub fn report_csv(&self) -> PathBuf {
        self.out_dir().join("report.csv")
    }

    // ---- project.json / clinical.json ----

    /// Muat manifes. Versi lebih baru ditolak dengan pesan jelas; versi lebih
    /// lama dimigrasikan. `clinical.json`, bila ada, dimuat ke
    /// `project.sample.clinical`; bila tidak ada → `None` tanpa galat.
    pub fn load_project(&self) -> Result<Project, IoError> {
        let path = self.project_json();
        let bytes = read_file(&path)?;
        let value: Value = serde_json::from_slice(&bytes).map_err(|e| IoError::json(&path, e))?;
        let value = migrate(value, &path)?;
        let mut project: Project =
            serde_json::from_value(value).map_err(|e| IoError::json(&path, e))?;
        project.check_version()?;
        project.sample.clinical = self.load_clinical()?;
        Ok(project)
    }

    /// Tulis `project.json` (atomik). `ClinicalMeta` TIDAK ikut ditulis
    /// (`SampleMeta::clinical` ber-`serde(skip)`); data klinis hanya ditulis
    /// lewat [`Self::save_clinical`].
    pub fn save_project(&self, project: &Project) -> Result<(), IoError> {
        write_atomic(&self.project_json(), &encode_project(project)?)
    }

    /// Muat `clinical.json`; `None` bila berkas tidak ada.
    pub fn load_clinical(&self) -> Result<Option<ClinicalMeta>, IoError> {
        let path = self.clinical_json();
        if !path.exists() {
            return Ok(None);
        }
        let bytes = read_file(&path)?;
        let c: ClinicalMeta =
            serde_json::from_slice(&bytes).map_err(|e| IoError::json(&path, e))?;
        Ok(Some(c))
    }

    /// Tulis `clinical.json` (atomik); `None` menghapus berkasnya.
    pub fn save_clinical(&self, clinical: Option<&ClinicalMeta>) -> Result<(), IoError> {
        let path = self.clinical_json();
        match clinical {
            None => remove_if_exists(&path),
            Some(c) => {
                let mut bytes =
                    serde_json::to_vec_pretty(c).map_err(|e| IoError::json(&path, e))?;
                bytes.push(b'\n');
                write_atomic(&path, &bytes)
            }
        }
    }

    // ---- label & atribut ----

    /// Muat label irisan; `None` bila belum ada berkas `.lbl`.
    pub fn load_labels(&self, slide: SlideId) -> Result<Option<LabelStore>, IoError> {
        let path = self.label_path(slide);
        if !path.exists() {
            return Ok(None);
        }
        let bytes = read_file(&path)?;
        decode_lbl(&bytes)
            .map(Some)
            .map_err(|e| IoError::invalid(&path, e.to_string()))
    }

    /// Tulis label irisan (atomik). Penyimpanan kosong menghapus berkasnya.
    pub fn save_labels(&self, slide: SlideId, store: &LabelStore) -> Result<(), IoError> {
        let path = self.label_path(slide);
        if store.is_empty() {
            return remove_if_exists(&path);
        }
        write_atomic(&path, &encode_lbl(store))
    }

    /// Muat atribut `(irisan, stem)`; `None` bila berkas tidak ada.
    /// Mengembalikan penyimpanan dan `attr_id` di header.
    pub fn load_attr(
        &self,
        slide: SlideId,
        stem: &str,
    ) -> Result<Option<(AttrStore, u16)>, IoError> {
        let path = self.attr_path(slide, stem);
        if !path.exists() {
            return Ok(None);
        }
        let bytes = read_file(&path)?;
        decode_atr(&bytes)
            .map(Some)
            .map_err(|e| IoError::invalid(&path, e.to_string()))
    }

    /// Tulis atribut `(irisan, stem)` (atomik).
    pub fn save_attr(
        &self,
        slide: SlideId,
        stem: &str,
        store: &AttrStore,
        attr_id: u16,
    ) -> Result<(), IoError> {
        write_atomic(&self.attr_path(slide, stem), &encode_atr(store, attr_id))
    }

    /// Hapus berkas atribut `(irisan, stem)` bila ada.
    pub fn remove_attr(&self, slide: SlideId, stem: &str) -> Result<(), IoError> {
        remove_if_exists(&self.attr_path(slide, stem))
    }

    // ---- sumber slide ----

    /// Path absolut berkas slide. Path `External` relatif ditafsirkan
    /// terhadap folder proyek.
    pub fn resolve_slide_path(&self, source: &SlideSourceRef) -> PathBuf {
        match source {
            SlideSourceRef::External { path, .. } => {
                if path.is_absolute() {
                    path.clone()
                } else {
                    normalize(&self.root.join(path))
                }
            }
            SlideSourceRef::Embedded { rel_path } => self.root.join(rel_path),
        }
    }

    /// Nyatakan `path` relatif terhadap folder proyek bila bisa (8.7.1);
    /// bila tidak (mis. beda volume), kembalikan path absolutnya.
    pub fn make_relative(&self, path: &Path) -> PathBuf {
        let abs_path = absolutize(path);
        let abs_root = absolutize(&self.root);
        relative_to(&abs_path, &abs_root).unwrap_or(abs_path)
    }

    /// Salin slide ke `slides/NNNN.<ext>` (atomik) setelah memeriksa ruang
    /// kosong, lalu kembalikan rujukan `Embedded`.
    pub fn embed_slide(&self, src: &Path, id: SlideId) -> Result<SlideSourceRef, IoError> {
        let meta = fs::metadata(src).map_err(|e| IoError::io(src, e))?;
        if !meta.is_file() {
            return Err(IoError::invalid(src, "bukan berkas biasa"));
        }
        let dir = self.slides_dir();
        fs::create_dir_all(&dir).map_err(|e| IoError::io(&dir, e))?;
        if let Some(available) = free_space(&dir) {
            let needed = meta.len().saturating_add(EMBED_FREE_SPACE_MARGIN);
            if available < needed {
                return Err(IoError::InsufficientSpace { needed, available });
            }
        }
        let ext = src
            .extension()
            .map(|e| e.to_string_lossy().to_ascii_lowercase())
            .filter(|e| !e.is_empty() && e.chars().all(|c| c.is_ascii_alphanumeric()));
        let name = match ext {
            Some(e) => format!("{}.{}", slide_stem(id), e),
            None => slide_stem(id),
        };
        let rel_path = PathBuf::from("slides").join(name);
        copy_atomic(src, &self.root.join(&rel_path))?;
        Ok(SlideSourceRef::Embedded { rel_path })
    }

    /// Hapus seluruh isi `cache/` (aman kapan saja; semuanya bisa dibuat ulang).
    pub fn clear_cache(&self) -> Result<(), IoError> {
        let dir = self.cache_dir();
        match fs::remove_dir_all(&dir) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(IoError::io(&dir, e)),
        }
    }
}

/// Serialisasi manifes: JSON rapi + baris baru di akhir. Deterministik.
pub fn encode_project(project: &Project) -> Result<Vec<u8>, IoError> {
    let mut bytes = serde_json::to_vec_pretty(project)
        .map_err(|e| IoError::json(Path::new(PROJECT_FILE), e))?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn read_file(path: &Path) -> Result<Vec<u8>, IoError> {
    match fs::read(path) {
        Ok(b) => Ok(b),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            Err(IoError::MissingFile(path.to_path_buf()))
        }
        Err(e) => Err(IoError::io(path, e)),
    }
}

/// Migrasi manifes dari versi lama ke [`FORMAT_VERSION`]. Tiap langkah
/// menaikkan tepat satu versi sehingga rantai migrasi mudah diperpanjang.
fn migrate(mut value: Value, path: &Path) -> Result<Value, IoError> {
    let Some(obj) = value.as_object_mut() else {
        return Err(IoError::invalid(path, "akar project.json bukan objek"));
    };
    // Manifes pra-rilis tanpa `format_version` diperlakukan sebagai versi 0.
    let mut version = match obj.get("format_version") {
        None | Some(Value::Null) => 0,
        Some(v) => v
            .as_u64()
            .ok_or_else(|| IoError::invalid(path, "format_version bukan bilangan bulat"))?,
    };
    if version > u64::from(FORMAT_VERSION) {
        return Err(IoError::FormatTooNew {
            found: version,
            supported: FORMAT_VERSION,
        });
    }
    while version < u64::from(FORMAT_VERSION) {
        match version {
            0 => migrate_v0_to_v1(obj),
            other => {
                return Err(IoError::invalid(
                    path,
                    format!("tidak ada migrasi dari versi {other}"),
                ))
            }
        }
        version += 1;
        log::info!("project.json dimigrasikan ke versi {version}");
    }
    obj.insert("format_version".into(), Value::from(FORMAT_VERSION));
    Ok(value)
}

/// v0 → v1: versi 0 adalah manifes pra-rilis tanpa nomor versi. Field baru
/// (label `kind`/`group`, `attrs`, `annotations`, `spatial`) memakai nilai
/// bawaan serde; penanda lama `interpolated` per irisan dibuang (digantikan
/// `AnnotationRecord`, 8.12.7).
fn migrate_v0_to_v1(obj: &mut serde_json::Map<String, Value>) {
    if let Some(Value::Array(slides)) = obj.get_mut("slides") {
        for s in slides {
            if let Some(so) = s.as_object_mut() {
                so.remove("interpolated");
            }
        }
    }
    // Data klinis tidak pernah boleh tinggal di manifes (aturan keras 17).
    if let Some(Value::Object(sample)) = obj.get_mut("sample") {
        sample.remove("clinical");
    }
}

/// Path absolut: `canonicalize` bila berkasnya ada, selain itu digabung
/// dengan folder kerja lalu dinormalkan.
fn absolutize(path: &Path) -> PathBuf {
    if let Ok(p) = fs::canonicalize(path) {
        return p;
    }
    // Berkas belum ada: kanonikkan induknya bila bisa.
    if let (Some(parent), Some(name)) = (path.parent(), path.file_name()) {
        if let Ok(p) = fs::canonicalize(parent) {
            return p.join(name);
        }
    }
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|d| d.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    };
    normalize(&joined)
}

/// Buang `.` dan lipat `..` secara leksikal.
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    out.push("..");
                }
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// `path` relatif terhadap `base` (keduanya absolut). `None` bila awalan
/// (drive/volume) berbeda.
fn relative_to(path: &Path, base: &Path) -> Option<PathBuf> {
    let pc: Vec<Component<'_>> = path.components().collect();
    let bc: Vec<Component<'_>> = base.components().collect();
    // Awalan (root/prefix) harus sama.
    match (pc.first(), bc.first()) {
        (Some(a), Some(b)) if a == b => {}
        _ => return None,
    }
    let common = pc.iter().zip(bc.iter()).take_while(|(a, b)| a == b).count();
    let mut out = PathBuf::new();
    for _ in common..bc.len() {
        out.push("..");
    }
    for c in &pc[common..] {
        out.push(c.as_os_str());
    }
    if out.as_os_str().is_empty() {
        return None;
    }
    Some(out)
}

/// Ruang kosong (byte) pada sistem berkas yang memuat `dir`; `None` bila
/// tidak dapat ditentukan (pemeriksaan lalu dilewati).
#[cfg(unix)]
fn free_space(dir: &Path) -> Option<u64> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let c = CString::new(dir.as_os_str().as_bytes()).ok()?;
    let mut st = std::mem::MaybeUninit::<libc::statvfs>::zeroed();
    // SAFETY: `c` adalah string C sah yang hidup selama panggilan, dan `st`
    // menunjuk ke memori bertipe `statvfs` yang boleh ditulis kernel.
    let rc = unsafe { libc::statvfs(c.as_ptr(), st.as_mut_ptr()) };
    if rc != 0 {
        return None;
    }
    // SAFETY: `statvfs` berhasil (rc == 0) sehingga struktur sudah terisi.
    let st = unsafe { st.assume_init() };
    #[allow(clippy::unnecessary_cast)]
    Some((st.f_bavail as u64).saturating_mul(st.f_frsize as u64))
}

#[cfg(not(unix))]
fn free_space(_dir: &Path) -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_paths() {
        let r = relative_to(Path::new("/a/b/slides/x.png"), Path::new("/a/b/p.h3d"));
        assert_eq!(r, Some(PathBuf::from("../slides/x.png")));
        let r = relative_to(
            Path::new("/a/b/p.h3d/slides/x.png"),
            Path::new("/a/b/p.h3d"),
        );
        assert_eq!(r, Some(PathBuf::from("slides/x.png")));
        assert_eq!(
            normalize(Path::new("/a/b/../c/./d")),
            PathBuf::from("/a/c/d")
        );
    }

    #[test]
    fn slide_stem_is_zero_padded() {
        assert_eq!(slide_stem(SlideId(7)), "0007");
        assert_eq!(slide_stem(SlideId(12345)), "12345");
    }
}
