//! Tipe galat `h3d-imgproc`.

use thiserror::Error;

/// Galat pemrosesan citra. Semua fungsi yang menerima dimensi eksplisit dari
/// luar memeriksa kecocokannya dan mengembalikan varian di sini, bukan panik.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum ImgError {
    /// Bentuk/ukuran masukan tidak cocok (mis. panjang buffer ≠ `w × h × 4`).
    #[error("bentuk tidak cocok: diharapkan {expected:?}, diterima {got:?}")]
    ShapeMismatch {
        /// Bentuk yang diharapkan.
        expected: Vec<usize>,
        /// Bentuk yang diterima.
        got: Vec<usize>,
    },
    /// Masukan kosong (tidak ada piksel/elemen untuk diproses).
    #[error("masukan kosong")]
    EmptyInput,
    /// Parameter tidak sah (negatif, nol, singular, NaN, dst.).
    #[error("parameter tidak sah: {0}")]
    InvalidParameter(String),
    /// Pekerjaan dibatalkan lewat `Progress::is_cancelled`.
    #[error("dibatalkan")]
    Cancelled,
}

impl ImgError {
    /// Pembantu untuk membangun `ShapeMismatch`.
    pub fn shape(expected: &[usize], got: &[usize]) -> Self {
        ImgError::ShapeMismatch {
            expected: expected.to_vec(),
            got: got.to_vec(),
        }
    }
}
