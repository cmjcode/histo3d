//! `FlatBackend`: gambar datar (JPEG/PNG/TIFF biasa) lewat crate `image`.
//! Piramida dibangun sendiri dengan penurunan 2× (rata-rata kotak) sampai
//! sisi terpanjang ≤ [`FlatBackend::MIN_TOP_SIDE`] px dan disimpan di memori.
//! mpp selalu `None` (foto kamera tidak membawa skala; aturan keras 7).

use std::fs;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};

use image::{ImageFormat, ImageReader, RgbaImage};
use serde::{Deserialize, Serialize};

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

/// Manifes cache piramida (`pyramid.json`).
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
struct PyramidManifest {
    /// Versi format manifes.
    version: u32,
    /// Ukuran tiap level `[lebar, tinggi]`, indeks 0 = resolusi penuh.
    level_sizes: Vec<[u32; 2]>,
    /// Ukuran tile yang dilaporkan `SlideInfo`.
    tile_size: u32,
}

const MANIFEST_VERSION: u32 = 1;
const MANIFEST_NAME: &str = "pyramid.json";

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

    /// Buka lewat cache: bila `cache_dir` berisi piramida yang sah, muat
    /// darinya; bila tidak, dekode `path`, bangun piramida, lalu simpan ke
    /// `cache_dir`. Kegagalan menulis cache hanya dicatat (log), tidak
    /// menggagalkan pembukaan.
    pub fn open_with_cache(path: &Path, cache_dir: &Path) -> Result<Self, SlideError> {
        match Self::load_pyramid_cache(cache_dir) {
            Ok(b) => return Ok(b),
            Err(e) => log::debug!("cache piramida {} tidak dipakai: {e}", cache_dir.display()),
        }
        let backend = Self::open(path)?;
        if let Err(e) = backend.write_pyramid_cache(cache_dir) {
            log::warn!("gagal menulis cache piramida {}: {e}", cache_dir.display());
        }
        Ok(backend)
    }

    /// Simpan semua level sebagai PNG (`level_00.png`, …) plus `pyramid.json`
    /// ke `dir`. Penulisan atomik: `.tmp` lalu `rename` (aturan keras 10).
    pub fn write_pyramid_cache(&self, dir: &Path) -> Result<(), SlideError> {
        fs::create_dir_all(dir).map_err(|e| SlideError::io(dir, e))?;
        for (i, img) in self.levels.iter().enumerate() {
            let mut buf = Cursor::new(Vec::new());
            img.write_to(&mut buf, ImageFormat::Png)?;
            write_atomic(&dir.join(level_file_name(i)), buf.get_ref())?;
        }
        let manifest = PyramidManifest {
            version: MANIFEST_VERSION,
            level_sizes: self.info.levels.iter().map(|l| l.size).collect(),
            tile_size: self.info.tile_size,
        };
        let json = serde_json::to_vec_pretty(&manifest)
            .map_err(|e| SlideError::Decode(format!("serialisasi manifes: {e}")))?;
        write_atomic(&dir.join(MANIFEST_NAME), &json)
    }

    /// Muat piramida dari `dir` yang ditulis [`Self::write_pyramid_cache`].
    pub fn load_pyramid_cache(dir: &Path) -> Result<Self, SlideError> {
        let manifest_path = dir.join(MANIFEST_NAME);
        let json = fs::read(&manifest_path).map_err(|e| SlideError::io(&manifest_path, e))?;
        let manifest: PyramidManifest = serde_json::from_slice(&json)
            .map_err(|e| SlideError::Decode(format!("manifes cache rusak: {e}")))?;
        if manifest.version != MANIFEST_VERSION {
            return Err(SlideError::Decode(format!(
                "versi manifes cache {} tidak didukung",
                manifest.version
            )));
        }
        if manifest.level_sizes.is_empty() {
            return Err(SlideError::Decode("manifes cache tanpa level".into()));
        }
        let mut levels = Vec::with_capacity(manifest.level_sizes.len());
        for (i, size) in manifest.level_sizes.iter().enumerate() {
            let p = dir.join(level_file_name(i));
            let img = ImageReader::open(&p)
                .map_err(|e| SlideError::io(&p, e))?
                .with_guessed_format()
                .map_err(|e| SlideError::io(&p, e))?
                .decode()?
                .into_rgba8();
            if [img.width(), img.height()] != *size {
                return Err(SlideError::Decode(format!(
                    "level cache {i} berukuran {}x{}, manifes menyebut {}x{}",
                    img.width(),
                    img.height(),
                    size[0],
                    size[1]
                )));
            }
            levels.push(img);
        }
        let mut backend = Self::from_levels(levels)?;
        backend.info.tile_size = manifest.tile_size.max(1);
        Ok(backend)
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

fn level_file_name(i: usize) -> String {
    format!("level_{i:02}.png")
}

/// Tulis `bytes` ke `path` secara atomik (`.tmp` lalu `rename`).
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), SlideError> {
    let tmp: PathBuf = {
        let mut name = path
            .file_name()
            .map(|n| n.to_os_string())
            .unwrap_or_default();
        name.push(".tmp");
        path.with_file_name(name)
    };
    let mut f = fs::File::create(&tmp).map_err(|e| SlideError::io(&tmp, e))?;
    f.write_all(bytes).map_err(|e| SlideError::io(&tmp, e))?;
    f.sync_all().map_err(|e| SlideError::io(&tmp, e))?;
    drop(f);
    fs::rename(&tmp, path).map_err(|e| SlideError::io(path, e))
}
