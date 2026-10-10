//! Alat lepas-proyek untuk CLI: metadata slide dan thumbnail (`h3d info`,
//! `h3d thumb`).

use std::path::Path;

use h3d_io::write_atomic;
use serde::Serialize;

use crate::error::EngineError;

/// Satu level piramida.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LevelReport {
    /// Indeks level (0 = resolusi penuh).
    pub level: u32,
    /// Ukuran `[lebar, tinggi]`, piksel.
    pub size: [u32; 2],
    /// Faktor penurunan terhadap level 0.
    pub downsample: f64,
}

/// Metadata slide untuk `h3d info`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SlideReport {
    /// Nama berkas.
    pub file: String,
    /// Ukuran level 0 `[lebar, tinggi]`, piksel.
    pub size_l0: [u32; 2],
    /// µm per piksel level 0 `[x, y]`; `null` bila metadata tidak memuatnya
    /// (tidak pernah ditebak).
    pub mpp: Option<[f64; 2]>,
    /// Ukuran tile.
    pub tile_size: u32,
    /// Vendor pemindai bila diketahui.
    pub vendor: Option<String>,
    /// Level piramida.
    pub levels: Vec<LevelReport>,
    /// Ukuran berkas, byte.
    pub size_bytes: u64,
    /// Sidik jari berkas (heksadesimal).
    pub fingerprint: String,
}

/// Baca metadata sebuah slide.
pub fn inspect_slide(path: &Path) -> Result<SlideReport, EngineError> {
    let src = h3d_slide::open_slide(path)?;
    let info = src.info();
    let (size_bytes, fp) = h3d_slide::fingerprint(path)?;
    Ok(SlideReport {
        file: path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default(),
        size_l0: info.level0_size(),
        mpp: info.mpp,
        tile_size: info.tile_size,
        vendor: info.vendor.clone(),
        levels: info
            .levels
            .iter()
            .enumerate()
            .map(|(i, l)| LevelReport {
                level: i as u32,
                size: l.size,
                downsample: l.downsample,
            })
            .collect(),
        size_bytes,
        fingerprint: format!("{fp:016x}"),
    })
}

/// Tulis thumbnail PNG (sisi terpanjang ≤ `max_side`) ke `out` secara atomik.
/// Mengembalikan ukuran thumbnail `[lebar, tinggi]`.
pub fn write_thumbnail(path: &Path, out: &Path, max_side: u32) -> Result<[u32; 2], EngineError> {
    let src = h3d_slide::open_slide(path)?;
    let img = h3d_slide::thumbnail(src.as_ref(), max_side)?;
    let mut png = Vec::new();
    let enc = image::codecs::png::PngEncoder::new(&mut png);
    image::ImageEncoder::write_image(
        enc,
        img.as_raw(),
        img.width(),
        img.height(),
        image::ExtendedColorType::Rgba8,
    )
    .map_err(|e| EngineError::invalid(format!("gagal mengode PNG: {e}")))?;
    write_atomic(out, &png)?;
    Ok([img.width(), img.height()])
}
