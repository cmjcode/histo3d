//! Tipe galat `h3d-volume`.

use h3d_imgproc::ImgError;
use thiserror::Error;

/// Galat rekonstruksi/analisis volume. Masukan dari luar tidak pernah
/// memicu panik; semua kondisi tak sah menjadi varian di sini.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum VolError {
    /// Tidak ada irisan yang memuat label yang diminta.
    #[error("tidak ada irisan berlabel untuk direkonstruksi")]
    NoLabelledSlices,
    /// Kisi tetap melampaui anggaran memori meski `s_xy` sudah dinaikkan
    /// sampai batas maksimum.
    #[error(
        "kisi terlalu besar untuk anggaran ({needed_bytes} B > {budget_bytes} B) \
         bahkan pada s_xy = {s_xy_um} µm"
    )]
    GridTooLarge {
        /// Perkiraan kebutuhan memori.
        needed_bytes: u64,
        /// Anggaran set kerja rekonstruksi.
        budget_bytes: u64,
        /// `s_xy` terakhir yang dicoba.
        s_xy_um: f64,
    },
    /// Ukuran masukan tidak konsisten (kisi berbeda, panjang buffer salah).
    #[error("ukuran tidak konsisten: {0}")]
    InconsistentSizes(String),
    /// Parameter tidak sah (nol, negatif, NaN).
    #[error("parameter tidak sah: {0}")]
    InvalidParameter(String),
    /// Dibatalkan lewat `Progress::is_cancelled`.
    #[error("dibatalkan")]
    Cancelled,
    /// Galat dari `h3d-imgproc`.
    #[error("pemrosesan citra: {0}")]
    Img(#[from] ImgError),
}
