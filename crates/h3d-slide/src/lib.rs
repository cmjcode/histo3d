//! `h3d-slide`: pembaca slide (histo3d.md 8.2).
//!
//! Menyediakan trait [`SlideSource`] dengan tiga backend yang dicoba
//! berurutan oleh [`open_slide`]: [`WsiBackend`] (`wsi-rs`; fitur `wsi`),
//! [`TiffBackend`] (crate `tiff`), lalu [`FlatBackend`] (crate `image`).
//! Juga cache tile CPU berbatas byte ([`TileCache`]), pemuat tile
//! berprioritas ([`TileLoader`]), sidik jari berkas ([`fingerprint`]),
//! thumbnail ([`thumbnail`]) dan alat bantu koordinat ([`coords`]).
//!
//! Lihat histo3d.md bagian 5 untuk sistem koordinat: koordinat
//! `read_region` adalah piksel level yang diminta; pusat piksel di `+0,5`.

pub mod cache;
pub mod coords;
pub mod error;
pub mod flat;
pub mod loader;
pub mod source;
pub mod tiff_backend;
#[cfg(feature = "wsi")]
pub mod wsi;

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use image::RgbaImage;

pub use cache::{tile_bytes, TileCache, TileKey};
pub use coords::{level_for_scale, target_downsample};
pub use error::SlideError;
pub use flat::FlatBackend;
pub use loader::{read_tile, TileLoader, TileRequest, TileResult};
pub use source::{
    downsample_2x, downsample_box, downsample_of, half_size, LevelInfo, SlideInfo, SlideSource,
    BACKGROUND,
};
pub use tiff_backend::{mpp_from_resolution, parse_aperio_mpp, TiffBackend};
#[cfg(feature = "wsi")]
pub use wsi::WsiBackend;

/// Buka slide dengan mencoba backend berurutan: `wsi` (bila fitur aktif),
/// `tiff`, lalu `flat`. Bila semua gagal, [`SlideError::UnsupportedFormat`]
/// memuat alasan tiap backend.
pub fn open_slide(path: &Path) -> Result<Box<dyn SlideSource>, SlideError> {
    let meta = std::fs::metadata(path).map_err(|e| SlideError::io(path, e))?;
    let mut attempts: Vec<String> = Vec::new();

    #[cfg(feature = "wsi")]
    {
        match WsiBackend::open(path) {
            Ok(b) => return Ok(Box::new(b)),
            Err(e) => attempts.push(format!("wsi: {e}")),
        }
    }

    if !meta.is_file() {
        attempts.push("bukan berkas biasa".to_string());
        return Err(SlideError::UnsupportedFormat {
            path: path.to_path_buf(),
            attempts,
        });
    }

    match TiffBackend::open(path) {
        Ok(b) => return Ok(Box::new(b)),
        Err(e) => attempts.push(format!("tiff: {e}")),
    }
    match FlatBackend::open(path) {
        Ok(b) => return Ok(Box::new(b)),
        Err(e) => attempts.push(format!("flat: {e}")),
    }

    Err(SlideError::UnsupportedFormat {
        path: path.to_path_buf(),
        attempts,
    })
}

/// Panjang jendela sidik jari (1 MB di awal dan di akhir berkas).
pub const FINGERPRINT_WINDOW: u64 = 1024 * 1024;

/// Sidik jari berkas: `(ukuran, hash)`. Hash FNV-1a 64-bit atas ukuran
/// berkas (8 byte little-endian), 1 MB pertama dan 1 MB terakhir (jendela
/// boleh tumpang tindih untuk berkas kecil). Deterministik dan murah untuk
/// berkas bergigabyte.
pub fn fingerprint(path: &Path) -> Result<(u64, u64), SlideError> {
    let mut f = File::open(path).map_err(|e| SlideError::io(path, e))?;
    let size = f.metadata().map_err(|e| SlideError::io(path, e))?.len();
    let mut hasher = Fnv1a64::new();
    hasher.update(&size.to_le_bytes());

    let head_len = size.min(FINGERPRINT_WINDOW);
    let mut buf = vec![0u8; head_len as usize];
    f.read_exact(&mut buf)
        .map_err(|e| SlideError::io(path, e))?;
    hasher.update(&buf);

    let tail_start = size.saturating_sub(FINGERPRINT_WINDOW);
    let tail_len = size - tail_start;
    f.seek(SeekFrom::Start(tail_start))
        .map_err(|e| SlideError::io(path, e))?;
    buf.resize(tail_len as usize, 0);
    f.read_exact(&mut buf)
        .map_err(|e| SlideError::io(path, e))?;
    hasher.update(&buf);

    Ok((size, hasher.finish()))
}

/// FNV-1a 64-bit.
struct Fnv1a64(u64);

impl Fnv1a64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    fn new() -> Self {
        Self(Self::OFFSET)
    }

    fn update(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(Self::PRIME);
        }
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

/// Thumbnail: pilih level terkecil yang sisi terpanjangnya ≥ `max_side`
/// (atau level terkecil bila tidak ada), baca seluruhnya, lalu turunkan
/// dengan rata-rata kotak agar sisi terpanjang ≤ `max_side`.
pub fn thumbnail(src: &dyn SlideSource, max_side: u32) -> Result<RgbaImage, SlideError> {
    if max_side == 0 {
        return Err(SlideError::InvalidArgument(
            "max_side harus lebih dari nol".into(),
        ));
    }
    let info = src.info();
    if info.levels.is_empty() {
        return Err(SlideError::Decode("slide tanpa level".into()));
    }
    let longest = |l: &LevelInfo| l.size[0].max(l.size[1]);
    let (idx, lvl) = info
        .levels
        .iter()
        .enumerate()
        .rev()
        .find(|(_, l)| longest(l) >= max_side)
        .or_else(|| info.levels.iter().enumerate().next_back())
        .ok_or_else(|| SlideError::Decode("slide tanpa level".into()))?;

    let img = src.read_region(idx as u32, 0, 0, lvl.size[0], lvl.size[1])?;
    let side = img.width().max(img.height());
    if side <= max_side {
        return Ok(img);
    }
    let scale = f64::from(max_side) / f64::from(side);
    let ow = ((f64::from(img.width()) * scale).round() as u32).clamp(1, img.width());
    let oh = ((f64::from(img.height()) * scale).round() as u32).clamp(1, img.height());
    downsample_box(&img, ow.min(max_side), oh.min(max_side))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a_known_vector() {
        // Vektor uji FNV-1a 64 untuk "a".
        let mut h = Fnv1a64::new();
        h.update(b"a");
        assert_eq!(h.finish(), 0xaf63_dc4c_8601_ec8c);
    }
}
