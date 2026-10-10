//! `WsiBackend`: pembungkus `wsi-rs` 0.6 (SVS, NDPI, SCN, BIF, DICOM WSI,
//! CZI/ZVI, MIRAX, VMS/VMU, VSI). Dipakai hanya untuk scene 0 / series 0
//! (brightfield RGB 8-bit); dataset multi-scene atau fluoresensi 16-bit
//! ditolak agar jatuh ke backend berikutnya.
//!
//! Catatan API `wsi-rs` 0.6 yang dipakai:
//! - `Slide::open_with_options(path, SlideOpenOptions)` — deterministik,
//!   tanpa `.svcache` implisit.
//! - `slide.dataset().scenes[s].series[r].levels` — `Level { dimensions:
//!   (u64, u64), downsample, tile_layout }`.
//! - `slide.dataset().properties.mpp()` → `Option<(f64, f64)>` dari
//!   `openslide.mpp-x/y`; `.vendor()` dari `openslide.vendor`.
//! - `slide.read_region_rgba(&RegionRequest)` — koordinat level, keluaran
//!   `image::RgbaImage`. Area luar gambar oleh `wsi-rs` diisi nol, sehingga
//!   region di sini dipotong ke batas gambar dan sisanya diisi putih sendiri.

use std::path::{Path, PathBuf};

use image::RgbaImage;
use wsi_rs::{
    CacheConfig, LevelIdx, RegionRequest, SampleType, SceneId, SeriesId, Slide, SlideOpenOptions,
    TileLayout,
};

use crate::error::SlideError;
use crate::source::{compose_on_white, LevelInfo, SlideInfo, SlideSource};

/// Ukuran tile yang dilaporkan bila tata letak tile `wsi-rs` tidak reguler.
const DEFAULT_TILE_SIZE: u32 = 256;

/// Sumber slide lewat `wsi-rs`.
pub struct WsiBackend {
    path: PathBuf,
    info: SlideInfo,
    slide: Slide,
}

impl WsiBackend {
    /// Buka dengan opsi deterministik bawaan `wsi-rs` (cache internal
    /// memakai bawaan pustaka).
    pub fn open(path: &Path) -> Result<Self, SlideError> {
        Self::open_with_options(path, SlideOpenOptions::deterministic())
    }

    /// Buka dengan anggaran cache tile internal `wsi-rs` dalam byte (ambil
    /// dari `h3d_core::Budgets`, jangan tulis angka di sini).
    pub fn open_with_cache_bytes(path: &Path, cache_bytes: u64) -> Result<Self, SlideError> {
        let cfg = CacheConfig::deterministic().with_shared_tile_bytes(cache_bytes);
        Self::open_with_options(
            path,
            SlideOpenOptions::deterministic().with_cache_config(cfg),
        )
    }

    fn open_with_options(path: &Path, options: SlideOpenOptions) -> Result<Self, SlideError> {
        let slide = Slide::open_with_options(path, options).map_err(wsi_err)?;
        let dataset = slide.dataset();
        let scene = dataset
            .scenes
            .first()
            .ok_or_else(|| SlideError::backend("wsi", "dataset tanpa scene"))?;
        let series = scene
            .series
            .first()
            .ok_or_else(|| SlideError::backend("wsi", "scene tanpa series"))?;
        if series.sample_type != SampleType::Uint8 {
            return Err(SlideError::backend(
                "wsi",
                format!(
                    "tipe sampel {:?} bukan 8-bit brightfield",
                    series.sample_type
                ),
            ));
        }
        if series.levels.is_empty() {
            return Err(SlideError::backend("wsi", "series tanpa level"));
        }

        let mut levels = Vec::with_capacity(series.levels.len());
        for l in &series.levels {
            let w = u32::try_from(l.dimensions.0)
                .map_err(|_| SlideError::backend("wsi", "lebar level melebihi u32"))?;
            let h = u32::try_from(l.dimensions.1)
                .map_err(|_| SlideError::backend("wsi", "tinggi level melebihi u32"))?;
            if w == 0 || h == 0 {
                return Err(SlideError::backend("wsi", "level berukuran nol"));
            }
            if !(l.downsample.is_finite() && l.downsample > 0.0) {
                return Err(SlideError::backend("wsi", "downsample level tidak sah"));
            }
            levels.push(LevelInfo {
                size: [w, h],
                downsample: l.downsample,
            });
        }
        // Urutan level harus mengecil (indeks 0 = resolusi penuh).
        if levels.windows(2).any(|p| p[1].downsample < p[0].downsample) {
            return Err(SlideError::backend("wsi", "urutan level tidak mengecil"));
        }

        let tile_size = match &series.levels[0].tile_layout {
            TileLayout::Regular { tile_width, .. } if *tile_width > 0 => *tile_width,
            TileLayout::WholeLevel {
                virtual_tile_width, ..
            } if *virtual_tile_width > 0 => *virtual_tile_width,
            _ => DEFAULT_TILE_SIZE,
        };

        let mpp = dataset
            .properties
            .mpp()
            .filter(|(x, y)| x.is_finite() && y.is_finite() && *x > 0.0 && *y > 0.0)
            .map(|(x, y)| [x, y]);
        let vendor = dataset.properties.vendor().map(str::to_string);

        Ok(Self {
            path: path.to_path_buf(),
            info: SlideInfo {
                levels,
                mpp,
                tile_size,
                vendor,
            },
            slide,
        })
    }

    /// Path berkas sumber.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Akses ke handle `wsi-rs` (mis. untuk gambar terkait/label).
    pub fn inner(&self) -> &Slide {
        &self.slide
    }
}

impl SlideSource for WsiBackend {
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
        let size = self.info.level(level)?.size;
        compose_on_white(size, x, y, w, h, |s| {
            let req =
                RegionRequest::builder(SceneId::new(0), SeriesId::new(0), LevelIdx::new(level))
                    .origin_px((i64::from(s.x), i64::from(s.y)))
                    .size_px((s.w, s.h))
                    .build()
                    .map_err(|e| SlideError::backend("wsi", e.to_string()))?;
            self.slide.read_region_rgba(&req).map_err(wsi_err)
        })
    }
}

fn wsi_err(e: wsi_rs::WsiError) -> SlideError {
    match e {
        wsi_rs::WsiError::Io(src) => SlideError::Io {
            path: PathBuf::new(),
            source: src,
        },
        wsi_rs::WsiError::LevelOutOfRange { level, count } => {
            SlideError::LevelOutOfRange { level, count }
        }
        other => SlideError::backend("wsi", other.to_string()),
    }
}
