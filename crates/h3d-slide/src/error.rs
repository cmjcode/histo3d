//! Error crate `h3d-slide`. Teks `Display` bersifat teknis (untuk log);
//! pesan ke pengguna diterjemahkan lewat `h3d-i18n`.

use std::path::PathBuf;

use thiserror::Error;

/// Kesalahan saat membuka atau membaca slide.
#[derive(Debug, Error)]
pub enum SlideError {
    /// Tidak ada backend yang mengenali berkas. `attempts` berisi alasan
    /// penolakan tiap backend, berurutan sesuai urutan percobaan.
    #[error("format tidak didukung: {path} ({})", attempts.join("; "))]
    UnsupportedFormat {
        /// Berkas yang dicoba dibuka.
        path: PathBuf,
        /// Pesan penolakan per backend, mis. `"wsi: ..."`, `"tiff: ..."`.
        attempts: Vec<String>,
    },
    /// Kesalahan I/O pada berkas.
    #[error("I/O pada {path}: {source}")]
    Io {
        /// Berkas terkait.
        path: PathBuf,
        /// Error asal.
        #[source]
        source: std::io::Error,
    },
    /// Data piksel atau struktur berkas tidak bisa didekode.
    #[error("gagal mendekode: {0}")]
    Decode(String),
    /// Metadata mpp tidak ada; pengguna harus mengisinya (aturan keras 7).
    #[error("metadata mpp tidak ada")]
    MissingMpp,
    /// Indeks level di luar jumlah level slide.
    #[error("level {level} di luar rentang (slide punya {count} level)")]
    LevelOutOfRange {
        /// Level yang diminta.
        level: u32,
        /// Jumlah level yang tersedia.
        count: u32,
    },
    /// Argumen tidak sah (mis. ukuran nol, nilai tidak terhingga).
    #[error("argumen tidak sah: {0}")]
    InvalidArgument(String),
    /// Error spesifik backend.
    #[error("backend {backend}: {message}")]
    Backend {
        /// Nama backend (`"wsi"`, `"tiff"`, `"flat"`).
        backend: &'static str,
        /// Pesan teknis.
        message: String,
    },
}

impl SlideError {
    /// Bungkus error I/O dengan path berkasnya.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        SlideError::Io {
            path: path.into(),
            source,
        }
    }

    /// Error backend dengan nama dan pesan.
    pub fn backend(backend: &'static str, message: impl Into<String>) -> Self {
        SlideError::Backend {
            backend,
            message: message.into(),
        }
    }
}

impl From<image::ImageError> for SlideError {
    fn from(e: image::ImageError) -> Self {
        SlideError::Decode(e.to_string())
    }
}

impl From<tiff::TiffError> for SlideError {
    fn from(e: tiff::TiffError) -> Self {
        SlideError::Decode(e.to_string())
    }
}
