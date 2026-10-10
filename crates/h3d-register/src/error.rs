//! Galat registrasi.

use h3d_core::CoreError;
use h3d_imgproc::ImgError;
use thiserror::Error;

/// Galat yang dapat dikembalikan crate ini. Semua masukan luar (citra, mpp,
/// landmark) divalidasi dan menghasilkan varian di sini, bukan panik.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum RegError {
    /// Masker jaringan kosong (tidak ada piksel bernilai 1).
    #[error("masker jaringan kosong pada citra {which}")]
    EmptyMask {
        /// Citra yang bermasalah ("fixed"/"moving"/pengenal irisan).
        which: String,
    },
    /// Bentuk `gray` dan `mask` (atau dua citra yang harus sebentuk) berbeda.
    #[error("bentuk tidak cocok: diharapkan {expected:?}, diterima {got:?}")]
    ShapeMismatch {
        /// Bentuk `(tinggi, lebar)` yang diharapkan.
        expected: [usize; 2],
        /// Bentuk yang diterima.
        got: [usize; 2],
    },
    /// Transformasi tidak dapat dibalik (determinan ≈ 0 atau tak-finit).
    #[error("transformasi singular (determinan {det:e})")]
    SingularTransform {
        /// Determinan bagian linear.
        det: f64,
    },
    /// Tidak ada irisan `Ok` yang bisa dipakai, atau irisan acuan tidak ada.
    #[error("tidak ada irisan yang bisa dipakai: {0}")]
    NoUsableSlides(String),
    /// Parameter tidak sah (mpp ≤ 0, panjang buffer, jumlah landmark, dst.).
    #[error("masukan tidak sah: {0}")]
    InvalidInput(String),
    /// Dibatalkan lewat `Progress::is_cancelled`.
    #[error("registrasi dibatalkan")]
    Cancelled,
    /// Galat dari `h3d-imgproc`.
    #[error("pemrosesan citra: {0}")]
    Img(ImgError),
    /// Galat dari `h3d-core` (mis. landmark degeneratif).
    #[error("inti: {0}")]
    Core(#[from] CoreError),
}

impl From<ImgError> for RegError {
    fn from(e: ImgError) -> Self {
        match e {
            ImgError::Cancelled => RegError::Cancelled,
            other => RegError::Img(other),
        }
    }
}
