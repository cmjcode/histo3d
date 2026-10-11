//! Lapisan platform: desktop (macOS) vs tablet (iPadOS).
//!
//! Semua keputusan "ini tablet" dan semua jalur berkas per-platform
//! dikumpulkan di sini (pola DUCAD `platform.rs`) supaya modul lain tidak
//! menebar `cfg(target_os)`.
//!
//! - Desktop: dialog berkas native lewat `rfd`.
//! - iPad: `rfd` tidak ada; berkas diambil dari folder Documents tersandbox
//!   lewat peramban dalam-aplikasi ([`list_documents`]) dan impor selalu
//!   menyalin slide ke dalam bundel (9.3).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use h3d_core::Budgets;

/// Benar bila dikompilasi untuk tablet.
pub const IS_MOBILE: bool = cfg!(target_os = "ios");

/// Akhiran folder bundel proyek.
pub const BUNDLE_EXT: &str = "h3d";

/// Ekstensi berkas slide yang bisa diimpor (huruf kecil).
pub const SLIDE_EXTENSIONS: [&str; 10] = [
    "svs", "tif", "tiff", "ndpi", "scn", "mrxs", "bif", "png", "jpg", "jpeg",
];

/// Lihat [`IS_MOBILE`]; bentuk fungsi untuk dipakai dalam ekspresi.
pub fn is_mobile() -> bool {
    IS_MOBILE
}

/// Folder dokumen yang terlihat pengguna (Files.app di iPad, `~/Documents`
/// di desktop). Bisa ditimpa dengan variabel lingkungan
/// `HISTO3D_DOCUMENTS_DIR` (tes dan uji tata letak tablet di desktop).
pub fn documents_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("HISTO3D_DOCUMENTS_DIR") {
        return PathBuf::from(dir);
    }
    #[cfg(target_vendor = "apple")]
    {
        crate::apple::documents_directory()
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        match std::env::var_os("HOME") {
            Some(home) => PathBuf::from(home).join("Documents"),
            None => PathBuf::from("Documents"),
        }
    }
}

/// Folder data aplikasi (pengaturan).
pub fn data_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("HISTO3D_DATA_DIR") {
        return PathBuf::from(dir);
    }
    #[cfg(target_vendor = "apple")]
    {
        crate::apple::app_support_directory().join("HISTO3D")
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        match std::env::var_os("HOME") {
            Some(home) => PathBuf::from(home).join(".histo3d"),
            None => PathBuf::from(".histo3d"),
        }
    }
}

/// Memori fisik perangkat (byte); `None` bila tidak dapat ditentukan.
pub fn physical_memory_bytes() -> Option<u64> {
    #[cfg(target_vendor = "apple")]
    {
        Some(crate::apple::physical_memory_bytes())
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        None
    }
}

/// Anggaran memori/thread aplikasi: anggaran platform diskalakan menurut
/// memori fisik (9.2). Satu-satunya tempat anggaran dipilih (aturan 15).
pub fn app_budgets() -> Budgets {
    let base = Budgets::for_platform();
    match physical_memory_bytes() {
        Some(bytes) if bytes > 0 => base.scaled_to_physical_memory(bytes),
        _ => base,
    }
}

static MEMORY_WARNING: AtomicBool = AtomicBool::new(false);

/// Kait peringatan memori: dipanggil observer OS (iOS
/// `UIApplicationDidReceiveMemoryWarningNotification`, lihat
/// `apple_ios::install_memory_warning_observer`) dari thread mana pun.
/// Aplikasi membacanya di awal frame berikutnya dan mengosongkan cache tile
/// CPU dan GPU (9.4).
pub fn on_memory_warning() {
    MEMORY_WARNING.store(true, Ordering::SeqCst);
}

/// Ambil dan reset sinyal peringatan memori.
pub fn take_memory_warning() -> bool {
    MEMORY_WARNING.swap(false, Ordering::SeqCst)
}

/// Satu entri peramban Documents.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DocEntry {
    /// Path lengkap.
    pub path: PathBuf,
    /// Nama tampil.
    pub name: String,
    /// Ukuran berkas (byte); 0 untuk bundel.
    pub size_bytes: u64,
}

/// Isi folder Documents yang relevan.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DocListing {
    /// Bundel proyek `.h3d` (folder berisi `project.json`).
    pub bundles: Vec<DocEntry>,
    /// Berkas gambar yang bisa diimpor.
    pub slides: Vec<DocEntry>,
}

/// Benar bila `path` berekstensi slide yang dikenal.
pub fn is_slide_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .is_some_and(|e| SLIDE_EXTENSIONS.contains(&e.as_str()))
}

/// Benar bila `path` adalah bundel proyek (folder berisi `project.json`).
pub fn is_bundle(path: &Path) -> bool {
    path.is_dir() && path.join(h3d_io::bundle::PROJECT_FILE).is_file()
}

/// Daftar bundel `.h3d` dan berkas gambar di `dir` (satu tingkat; isi
/// bundel tidak ditelusuri). Folder yang tidak terbaca menghasilkan daftar
/// kosong, bukan galat. Hasil terurut nama (urutan alami).
pub fn list_documents(dir: &Path) -> DocListing {
    let mut out = DocListing::default();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        if is_bundle(&path) {
            out.bundles.push(DocEntry {
                path,
                name,
                size_bytes: 0,
            });
        } else if path.is_file() && is_slide_file(&path) {
            let size_bytes = entry.metadata().map(|m| m.len()).unwrap_or(0);
            out.slides.push(DocEntry {
                path,
                name,
                size_bytes,
            });
        }
    }
    let by_name = |a: &DocEntry, b: &DocEntry| natural_cmp(&a.name, &b.name);
    out.bundles.sort_by(by_name);
    out.slides.sort_by(by_name);
    out
}

/// Urutan alami sederhana: deret angka dibandingkan sebagai bilangan.
fn natural_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    let key = |s: &str| -> Vec<(u8, u64, String)> {
        let mut out = Vec::new();
        let mut digits = String::new();
        let mut text = String::new();
        for c in s.chars() {
            if c.is_ascii_digit() {
                if !text.is_empty() {
                    out.push((1, 0, std::mem::take(&mut text)));
                }
                digits.push(c);
            } else {
                if !digits.is_empty() {
                    out.push((0, digits.parse().unwrap_or(u64::MAX), String::new()));
                    digits.clear();
                }
                text.extend(c.to_lowercase());
            }
        }
        if !digits.is_empty() {
            out.push((0, digits.parse().unwrap_or(u64::MAX), String::new()));
        }
        if !text.is_empty() {
            out.push((1, 0, text));
        }
        out
    };
    key(a).cmp(&key(b)).then_with(|| a.cmp(b))
}

/// Nama folder bundel yang aman dari nama proyek isian pengguna: hanya
/// huruf, angka, `-`, `_`, spasi → `_`; kosong → `proyek`.
pub fn bundle_dir_name(project_name: &str) -> String {
    let mut s: String = project_name
        .trim()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    while s.contains("__") {
        s = s.replace("__", "_");
    }
    let s = s.trim_matches('_');
    let stem = if s.is_empty() { "proyek" } else { s };
    format!("{stem}.{BUNDLE_EXT}")
}

/// Ukuran byte dalam satuan yang mudah dibaca (MB/GB, satu desimal).
pub fn fmt_bytes(bytes: u64) -> String {
    const MB: f64 = 1024.0 * 1024.0;
    let b = bytes as f64;
    if b >= 1024.0 * MB {
        format!("{:.1} GB", b / (1024.0 * MB))
    } else {
        format!("{:.1} MB", b / MB)
    }
}

// ---- dialog berkas desktop ----

/// Pilih lokasi proyek baru (desktop). `None` di iPad atau bila dibatalkan.
pub fn pick_new_project() -> Option<PathBuf> {
    #[cfg(not(target_os = "ios"))]
    {
        rfd::FileDialog::new()
            .set_file_name(format!("sampel.{BUNDLE_EXT}"))
            .add_filter("HISTO3D", &[BUNDLE_EXT])
            .save_file()
            .map(|p| {
                if p.extension().is_some_and(|e| e == BUNDLE_EXT) {
                    p
                } else {
                    p.with_extension(BUNDLE_EXT)
                }
            })
    }
    #[cfg(target_os = "ios")]
    {
        None
    }
}

/// Pilih folder proyek `.h3d` yang sudah ada (desktop).
pub fn pick_open_project() -> Option<PathBuf> {
    #[cfg(not(target_os = "ios"))]
    {
        rfd::FileDialog::new().pick_folder()
    }
    #[cfg(target_os = "ios")]
    {
        None
    }
}

/// Pilih berkas slide untuk diimpor (desktop).
pub fn pick_slides() -> Vec<PathBuf> {
    #[cfg(not(target_os = "ios"))]
    {
        rfd::FileDialog::new()
            .add_filter("Slide", &SLIDE_EXTENSIONS)
            .pick_files()
            .unwrap_or_default()
    }
    #[cfg(target_os = "ios")]
    {
        Vec::new()
    }
}

/// Pilih folder tujuan ekspor (desktop). Di iPad ekspor selalu ke `out/`
/// di dalam bundel (terlihat di Files.app).
pub fn pick_export_dir() -> Option<PathBuf> {
    #[cfg(not(target_os = "ios"))]
    {
        rfd::FileDialog::new().pick_folder()
    }
    #[cfg(target_os = "ios")]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_warning_is_edge_triggered() {
        take_memory_warning();
        on_memory_warning();
        assert!(take_memory_warning());
        assert!(!take_memory_warning());
    }

    #[test]
    fn budgets_never_exceed_platform() {
        let b = app_budgets();
        let p = Budgets::for_platform();
        assert!(b.tile_cache_cpu_bytes <= p.tile_cache_cpu_bytes);
        assert!(b.tile_cache_gpu_bytes <= p.tile_cache_gpu_bytes);
        assert!(b.decode_threads >= 1);
    }

    #[test]
    fn lists_bundles_and_slides_only() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        std::fs::create_dir(d.join("b10.h3d")).unwrap();
        std::fs::write(d.join("b10.h3d/project.json"), "{}").unwrap();
        std::fs::create_dir(d.join("b2.h3d")).unwrap();
        std::fs::write(d.join("b2.h3d/project.json"), "{}").unwrap();
        std::fs::create_dir(d.join("kosong.h3d")).unwrap();
        std::fs::write(d.join("s10.PNG"), [0u8; 10]).unwrap();
        std::fs::write(d.join("s2.tif"), [0u8; 20]).unwrap();
        std::fs::write(d.join("catatan.txt"), "x").unwrap();
        std::fs::write(d.join(".tersembunyi.png"), "x").unwrap();

        let l = list_documents(d);
        let names = |v: &[DocEntry]| v.iter().map(|e| e.name.clone()).collect::<Vec<_>>();
        assert_eq!(names(&l.bundles), ["b2.h3d", "b10.h3d"]);
        assert_eq!(names(&l.slides), ["s2.tif", "s10.PNG"]);
        assert_eq!(l.slides[0].size_bytes, 20);
        assert_eq!(list_documents(&d.join("tidak-ada")), DocListing::default());
    }

    #[test]
    fn bundle_names_are_sanitised() {
        assert_eq!(bundle_dir_name("Kasus 12/A"), "Kasus_12_A.h3d");
        assert_eq!(bundle_dir_name("  "), "proyek.h3d");
        assert_eq!(bundle_dir_name("../../x"), "x.h3d");
        assert_eq!(fmt_bytes(3 * 1024 * 1024), "3.0 MB");
        assert_eq!(fmt_bytes(5 * 1024 * 1024 * 1024 / 2), "2.5 GB");
    }
}
