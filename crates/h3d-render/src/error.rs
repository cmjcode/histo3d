//! Kesalahan crate render.

/// Kesalahan yang dapat dikembalikan ke aplikasi (bukan panik).
#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    /// Tekstur melebihi `device.limits().max_texture_dimension_2d`.
    #[error("tekstur {width}×{height} melebihi batas perangkat {max} piksel")]
    TextureTooLarge {
        /// Lebar yang diminta.
        width: u32,
        /// Tinggi yang diminta.
        height: u32,
        /// Batas perangkat.
        max: u32,
    },
    /// Panjang data tidak sesuai dimensi × byte per piksel.
    #[error("ukuran data tekstur tidak cocok: diharapkan {expected} byte, diterima {actual}")]
    DataSizeMismatch {
        /// Byte yang diharapkan.
        expected: usize,
        /// Byte yang diterima.
        actual: usize,
    },
    /// Mesh tidak konsisten (panjang buffer, indeks di luar jangkauan, dsb.).
    #[error("mesh tidak valid: {0}")]
    InvalidMesh(String),
    /// Satu tile lebih besar dari seluruh anggaran cache.
    #[error("tile {bytes} byte melebihi anggaran cache {max_bytes} byte")]
    TileExceedsBudget {
        /// Ukuran tile.
        bytes: u64,
        /// Anggaran cache.
        max_bytes: u64,
    },
}
