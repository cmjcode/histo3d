//! `FlatBackend`: gambar datar (JPEG/PNG/TIFF biasa) lewat crate `image`,
//! **seluruhnya di memori**. Piramida dibangun dengan penurunan 2× (rata-rata
//! kotak) sampai sisi terpanjang ≤ [`FlatBackend::MIN_TOP_SIDE`] px.
//! Cocok untuk gambar kecil dan tes; gambar besar memakai piramida bertile
//! di disk ([`crate::tiled::TiledPyramid`]) agar RAM tidak ikut ukuran
//! gambar. mpp selalu `None` (foto kamera tidak membawa skala; aturan keras 7).

use std::path::Path;

use image::{ImageReader, RgbaImage};

use crate::error::SlideError;
use crate::source::{
    compose_on_white, crop, downsample_2x, downsample_of, half_size, LevelInfo, SlideInfo,
    SlideSource,
};

/// Sumber slide dari gambar datar dengan piramida di memori.
pub struct FlatBackend {
    info: SlideInfo,
    levels: Vec<RgbaImage>,
}

impl FlatBackend {
    /// Sisi terpanjang level teratas tidak melebihi nilai ini.
    pub const MIN_TOP_SIDE: u32 = 512;
    /// Ukuran tile yang disarankan untuk pembaca tile.
    pub const TILE_SIZE: u32 = 256;

    /// Buka berkas gambar (format ditebak dari isi, bukan ekstensi).
    pub fn open(path: &Path) -> Result<Self, SlideError> {
        let reader = ImageReader::open(path)
            .map_err(|e| SlideError::io(path, e))?
            .with_guessed_format()
            .map_err(|e| SlideError::io(path, e))?;
        if reader.format().is_none() {
            return Err(SlideError::backend("flat", "format gambar tidak dikenali"));
        }
        let img = reader.decode()?.into_rgba8();
        Self::from_image(img)
    }

    /// Bangun piramida dari gambar yang sudah didekode.
    pub fn from_image(level0: RgbaImage) -> Result<Self, SlideError> {
        if level0.width() == 0 || level0.height() == 0 {
            return Err(SlideError::Decode("gambar berukuran nol".into()));
        }
        let mut levels = vec![level0];
        loop {
            let last = levels
                .last()
                .ok_or_else(|| SlideError::Decode("piramida kosong".into()))?;
            if last.width().max(last.height()) <= Self::MIN_TOP_SIDE {
                break;
            }
            let next = downsample_2x(last);
            levels.push(next);
        }
        Self::from_levels(levels)
    }

    /// Bangun dari level yang sudah ada; memvalidasi `ceil(prev / 2)`.
    fn from_levels(levels: Vec<RgbaImage>) -> Result<Self, SlideError> {
        let first = levels
            .first()
            .ok_or_else(|| SlideError::Decode("piramida kosong".into()))?;
        let l0 = [first.width(), first.height()];
        let mut infos = Vec::with_capacity(levels.len());
        let mut expected = l0;
        for (i, img) in levels.iter().enumerate() {
            let size = [img.width(), img.height()];
            if size != expected {
                return Err(SlideError::Decode(format!(
                    "level {i} berukuran {}x{}, diharapkan {}x{}",
                    size[0], size[1], expected[0], expected[1]
                )));
            }
            infos.push(LevelInfo {
                size,
                downsample: downsample_of(l0, size),
            });
            expected = half_size(size);
        }
        Ok(Self {
            info: SlideInfo {
                levels: infos,
                mpp: None,
                tile_size: Self::TILE_SIZE,
                vendor: None,
            },
            levels,
        })
    }

    /// Akses langsung ke gambar level `level`.
    pub fn level_image(&self, level: u32) -> Result<&RgbaImage, SlideError> {
        self.levels
            .get(level as usize)
            .ok_or(SlideError::LevelOutOfRange {
                level,
                count: self.levels.len() as u32,
            })
    }
}

impl SlideSource for FlatBackend {
    fn info(&self) -> &SlideInfo {
        &self.info
    }

    fn read_region(
        &self,
        level: u32,
        x: i64,
        y: i64,
        w: u32,
        h: u32,
    ) -> Result<RgbaImage, SlideError> {
        let img = self.level_image(level)?;
        compose_on_white([img.width(), img.height()], x, y, w, h, |s| {
            Ok(crop(img, s.x, s.y, s.w, s.h))
        })
    }
}
