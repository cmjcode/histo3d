//! Galat `h3d-train`.

use std::path::PathBuf;

/// Galat pelatihan/ekspor.
#[derive(Debug, thiserror::Error)]
pub enum TrainError {
    /// Galat I/O dengan jalur berkas.
    #[error("I/O pada {path}: {source}")]
    Io {
        /// Berkas yang bermasalah.
        path: PathBuf,
        /// Galat asal.
        #[source]
        source: std::io::Error,
    },
    /// Gambar tidak bisa didekode.
    #[error("gambar {path} tidak sah: {reason}")]
    Image {
        /// Berkas gambar.
        path: PathBuf,
        /// Alasan.
        reason: String,
    },
    /// Dataset tidak lengkap/tidak konsisten.
    #[error("dataset: {0}")]
    Dataset(String),
    /// Parameter tidak sah.
    #[error("parameter tidak sah: {0}")]
    InvalidParameter(String),
    /// Galat model/bobot.
    #[error("model: {0}")]
    Model(String),
    /// JSON.
    #[error("JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// Galat dari h3d-segment.
    #[error(transparent)]
    Seg(#[from] h3d_segment::SegError),
    /// Galat dari h3d-imgproc.
    #[error(transparent)]
    Img(#[from] h3d_imgproc::ImgError),
}

impl TrainError {
    /// Bungkus galat I/O dengan jalurnya.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}
