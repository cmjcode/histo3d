//! Tipe galat `h3d-segment`.

use h3d_core::LabelId;
use h3d_imgproc::ImgError;
use thiserror::Error;

/// Galat segmentasi dan penyimpanan label. Berkas rusak, model tidak sah, dan
/// parameter di luar rentang selalu menjadi varian di sini, tidak pernah panik.
#[derive(Debug, Error)]
pub enum SegError {
    /// Berkas `.lbl`/`.atr`/GeoJSON tidak sah (alasan dalam bahasa Indonesia).
    #[error("berkas tidak sah: {reason}")]
    InvalidFile {
        /// Alasan penolakan.
        reason: String,
    },
    /// Galat I/O saat membaca model.
    #[error("galat I/O: {0}")]
    Io(#[from] std::io::Error),
    /// Model AI tidak sah: `model.json` gagal diurai, operator tidak didukung,
    /// atau bentuk tensor tidak cocok.
    #[error("model tidak sah: {0}")]
    Model(String),
    /// Label tidak ditemukan di penyimpanan.
    #[error("label {0} tidak ditemukan")]
    LabelNotFound(LabelId),
    /// Pekerjaan dibatalkan lewat `Progress::is_cancelled`.
    #[error("dibatalkan")]
    Cancelled,
    /// Parameter tidak sah (mpp ≤ 0, NaN, transformasi singular, dst.).
    #[error("parameter tidak sah: {0}")]
    InvalidParameter(String),
    /// Galat pemrosesan citra dari `h3d-imgproc`.
    #[error("pemrosesan citra: {0}")]
    ImgProc(ImgError),
}

impl From<ImgError> for SegError {
    fn from(e: ImgError) -> Self {
        match e {
            ImgError::Cancelled => SegError::Cancelled,
            other => SegError::ImgProc(other),
        }
    }
}

impl SegError {
    /// Pembantu `InvalidFile`.
    pub fn invalid_file(reason: impl Into<String>) -> Self {
        SegError::InvalidFile {
            reason: reason.into(),
        }
    }
}
