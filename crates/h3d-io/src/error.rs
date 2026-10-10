//! Galat `h3d-io`.

use std::path::{Path, PathBuf};

use thiserror::Error;

/// Galat baca/tulis bundel proyek dan ekspor. Berkas rusak selalu menjadi
/// varian di sini, tidak pernah panik (aturan keras 6).
#[derive(Debug, Error)]
pub enum IoError {
    /// Galat I/O pada sebuah berkas.
    #[error("galat I/O pada {path}: {source}")]
    Io {
        /// Berkas terkait.
        path: PathBuf,
        /// Galat asal.
        #[source]
        source: std::io::Error,
    },
    /// JSON tidak bisa diurai atau ditulis.
    #[error("JSON tidak sah pada {path}: {source}")]
    Json {
        /// Berkas terkait.
        path: PathBuf,
        /// Galat asal.
        #[source]
        source: serde_json::Error,
    },
    /// Isi berkas tidak sah (terpotong, magic salah, nilai mustahil).
    #[error("berkas tidak sah {path}: {reason}")]
    InvalidFile {
        /// Berkas terkait.
        path: PathBuf,
        /// Alasan.
        reason: String,
    },
    /// Versi format proyek lebih baru dari yang didukung aplikasi ini.
    #[error(
        "proyek memakai format versi {found}, aplikasi ini hanya mendukung sampai versi \
         {supported}; perbarui aplikasi untuk membukanya"
    )]
    FormatTooNew {
        /// Versi di berkas.
        found: u64,
        /// Versi tertinggi yang didukung.
        supported: u32,
    },
    /// Berkas yang dibutuhkan tidak ada.
    #[error("berkas tidak ditemukan: {0}")]
    MissingFile(PathBuf),
    /// Ruang penyimpanan tidak cukup untuk menyalin berkas.
    #[error("ruang penyimpanan tidak cukup: butuh {needed} byte, tersedia {available} byte")]
    InsufficientSpace {
        /// Byte yang dibutuhkan.
        needed: u64,
        /// Byte yang tersedia.
        available: u64,
    },
    /// Argumen tidak sah (mis. kisi berbeda, mesh kosong).
    #[error("argumen tidak sah: {0}")]
    InvalidArgument(String),
    /// Galat tipe inti.
    #[error(transparent)]
    Core(#[from] h3d_core::CoreError),
    /// Galat label/atribut.
    #[error(transparent)]
    Seg(#[from] h3d_segment::SegError),
    /// Galat pembaca slide.
    #[error(transparent)]
    Slide(#[from] h3d_slide::SlideError),
    /// Galat volume.
    #[error(transparent)]
    Vol(#[from] h3d_volume::VolError),
}

impl IoError {
    /// Bungkus galat I/O dengan path.
    pub fn io(path: impl AsRef<Path>, source: std::io::Error) -> Self {
        IoError::Io {
            path: path.as_ref().to_path_buf(),
            source,
        }
    }

    /// Bungkus galat JSON dengan path.
    pub fn json(path: impl AsRef<Path>, source: serde_json::Error) -> Self {
        IoError::Json {
            path: path.as_ref().to_path_buf(),
            source,
        }
    }

    /// Berkas tidak sah dengan alasan.
    pub fn invalid(path: impl AsRef<Path>, reason: impl Into<String>) -> Self {
        IoError::InvalidFile {
            path: path.as_ref().to_path_buf(),
            reason: reason.into(),
        }
    }
}
