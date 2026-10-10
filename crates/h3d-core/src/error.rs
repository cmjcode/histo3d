//! Error crate inti.

use thiserror::Error;

/// Kesalahan pada tipe inti (validasi data, aljabar transformasi).
#[derive(Debug, Error, Clone, PartialEq)]
pub enum CoreError {
    /// Matriks affine tidak bisa dibalik (determinan ≈ 0).
    #[error("transformasi affine singular (determinan {det:e})")]
    SingularAffine { det: f64 },
    /// Dua sumbu orientasi anatomis berasal dari pasangan yang sama.
    #[error(
        "orientasi spesimen tidak sah: sumbu {a} dan {b} berasal dari pasangan anatomis yang sama"
    )]
    InvalidOrientation { a: String, b: String },
    /// Jumlah pasangan landmark tidak cukup untuk model yang diminta.
    #[error("butuh minimal {needed} pasang titik, diberikan {given}")]
    NotEnoughLandmarks { needed: usize, given: usize },
    /// Landmark kolinear atau degeneratif.
    #[error("pasangan landmark degeneratif (titik kolinear atau berimpit)")]
    DegenerateLandmarks,
    /// Nilai di luar rentang yang diizinkan.
    #[error("nilai {name} di luar rentang: {value}")]
    OutOfRange { name: String, value: f64 },
    /// Versi format proyek lebih baru dari yang didukung.
    #[error("format proyek versi {found} lebih baru dari yang didukung ({supported})")]
    FormatTooNew { found: u32, supported: u32 },
}
