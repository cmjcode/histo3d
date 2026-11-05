//! Pemindaian dataset Tampere (`<root>/<stack>/{Images_sub4,Masks_sub4,
//! TumorMasks_sub4}/imN.png`), pemotongan tile berlabel, dan cache tile biner.
//!
//! Kelas: 0 = latar, 1 = jaringan bukan tumor, 2 = tumor. Irisan tanpa masker
//! tumor dilewati (tumor tak berlabel akan meracuni kelas "jaringan").
//!
//! Format berkas tile (`NNNNNN.bin`, little-endian): magic `H3DT`, `u32 w`,
//! `u32 h`, `w·h·3` byte RGB, `w·h` byte kelas.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::TrainError;

/// Nama kelas, indeks = nilai kelas. Indeks 0 adalah latar (`None`).
pub const CLASS_NAMES: [Option<&str>; 3] = [None, Some("Jaringan"), Some("Tumor")];
/// Jumlah kelas.
pub const N_CLASSES: usize = CLASS_NAMES.len();
/// Faktor pengecilan folder `*_sub4` terhadap resolusi penuh.
pub const SUB4_FACTOR: f64 = 4.0;

const MAGIC: &[u8; 4] = b"H3DT";

/// Berkas satu irisan dalam dataset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlideFiles {
    /// Nama tumpukan, mis. `PTENX036-011`.
    pub stack: String,
    /// Nomor irisan dari nama berkas `imN.png`.
    pub index: u32,
    /// Gambar H&E `Images_sub4`.
    pub image: PathBuf,
    /// Masker jaringan `Masks_sub4` bila ada.
    pub tissue_mask: Option<PathBuf>,
    /// Masker tumor `TumorMasks_sub4` bila ada.
    pub tumor_mask: Option<PathBuf>,
}

/// Urai `imN.png` → `N`.
fn slide_index(name: &str) -> Option<u32> {
    let stem = name.strip_suffix(".png")?;
    let digits = stem.strip_prefix("im")?;
    digits.parse().ok()
}

/// Pindai `root` dan kembalikan daftar irisan terurut (tumpukan, indeks).
pub fn scan(root: &Path) -> Result<Vec<SlideFiles>, TrainError> {
    let mut stacks: Vec<PathBuf> = fs::read_dir(root)
        .map_err(|e| TrainError::io(root, e))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir() && p.join("Images_sub4").is_dir())
        .collect();
    stacks.sort();
    let mut out = Vec::new();
    for stack in stacks {
        let name = stack
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string();
        let img_dir = stack.join("Images_sub4");
        let mut slides: Vec<(u32, PathBuf)> = fs::read_dir(&img_dir)
            .map_err(|e| TrainError::io(&img_dir, e))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter_map(|p| {
                let n = p.file_name()?.to_str()?;
                slide_index(n).map(|i| (i, p.clone()))
            })
            .collect();
        slides.sort();
        for (index, image) in slides {
            let file = format!("im{index}.png");
            let tissue = stack.join("Masks_sub4").join(&file);
            let tumor = stack.join("TumorMasks_sub4").join(&file);
            out.push(SlideFiles {
                stack: name.clone(),
                index,
                image,
                tissue_mask: tissue.is_file().then_some(tissue),
                tumor_mask: tumor.is_file().then_some(tumor),
            });
        }
    }
    if out.is_empty() {
        return Err(TrainError::Dataset(format!(
            "tidak ada tumpukan dengan Images_sub4 di {}",
            root.display()
        )));
    }
    Ok(out)
}

/// Gambar RGB 8-bit, row-major.
#[derive(Debug, Clone)]
pub struct RgbImage {
    /// Lebar piksel.
    pub width: usize,
    /// Tinggi piksel.
    pub height: usize,
    /// `width·height·3` byte.
    pub rgb: Vec<u8>,
}

/// Masker biner (0/1), row-major.
#[derive(Debug, Clone)]
pub struct Mask {
    /// Lebar piksel.
    pub width: usize,
    /// Tinggi piksel.
    pub height: usize,
    /// `width·height` byte 0/1.
    pub data: Vec<u8>,
}

fn decode(path: &Path) -> Result<image::DynamicImage, TrainError> {
    image::open(path).map_err(|e| TrainError::Image {
        path: path.to_path_buf(),
        reason: e.to_string(),
    })
}

/// Baca PNG sebagai RGB 8-bit.
pub fn load_rgb(path: &Path) -> Result<RgbImage, TrainError> {
    let img = decode(path)?.to_rgb8();
    let (w, h) = img.dimensions();
    Ok(RgbImage {
        width: w as usize,
        height: h as usize,
        rgb: img.into_raw(),
    })
}

/// Baca PNG masker (8/16-bit, abu/RGB); piksel bukan-nol → 1.
pub fn load_mask(path: &Path) -> Result<Mask, TrainError> {
    let img = decode(path)?.to_luma16();
    let (w, h) = img.dimensions();
    Ok(Mask {
        width: w as usize,
        height: h as usize,
        data: img
            .into_raw()
            .into_iter()
            .map(|v| u8::from(v != 0))
            .collect(),
    })
}

/// Satu tile berlabel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tile {
    /// Sisi tile (persegi).
    pub size: usize,
    /// RGB `size·size·3`.
    pub rgb: Vec<u8>,
    /// Kelas `size·size`.
    pub cls: Vec<u8>,
}

/// Tulis tile ke berkas (atomik: `.tmp` lalu rename).
pub fn write_tile(path: &Path, tile: &Tile) -> Result<(), TrainError> {
    let n = tile.size * tile.size;
    if tile.rgb.len() != n * 3 || tile.cls.len() != n {
        return Err(TrainError::InvalidParameter(
            "ukuran tile tidak konsisten".into(),
        ));
    }
    let tmp = path.with_extension("tmp");
    {
        let mut f = fs::File::create(&tmp).map_err(|e| TrainError::io(&tmp, e))?;
        let w = tile.size as u32;
        f.write_all(MAGIC)
            .and_then(|_| f.write_all(&w.to_le_bytes()))
            .and_then(|_| f.write_all(&w.to_le_bytes()))
            .and_then(|_| f.write_all(&tile.rgb))
            .and_then(|_| f.write_all(&tile.cls))
            .map_err(|e| TrainError::io(&tmp, e))?;
    }
    fs::rename(&tmp, path).map_err(|e| TrainError::io(path, e))
}

/// Baca tile dari berkas.
pub fn read_tile(path: &Path) -> Result<Tile, TrainError> {
    let mut f = fs::File::open(path).map_err(|e| TrainError::io(path, e))?;
    let mut head = [0u8; 12];
    f.read_exact(&mut head)
        .map_err(|e| TrainError::io(path, e))?;
    if &head[0..4] != MAGIC {
        return Err(TrainError::Dataset(format!(
            "{} bukan berkas tile",
            path.display()
        )));
    }
    let w = u32::from_le_bytes([head[4], head[5], head[6], head[7]]) as usize;
    let h = u32::from_le_bytes([head[8], head[9], head[10], head[11]]) as usize;
    if w != h || w == 0 || w > 8192 {
        return Err(TrainError::Dataset(format!(
            "{}: ukuran tile {w}×{h} tidak sah",
            path.display()
        )));
    }
    let n = w * h;
    let mut rgb = vec![0u8; n * 3];
    let mut cls = vec![0u8; n];
    f.read_exact(&mut rgb)
        .and_then(|_| f.read_exact(&mut cls))
        .map_err(|e| TrainError::io(path, e))?;
    if cls.iter().any(|&c| c as usize >= N_CLASSES) {
        return Err(TrainError::Dataset(format!(
            "{}: nilai kelas di luar rentang",
            path.display()
        )));
    }
    Ok(Tile { size: w, rgb, cls })
}

/// Metadata satu tile dalam indeks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TileMeta {
    /// Nama berkas tile di folder cache.
    pub file: String,
    /// Tumpukan asal.
    pub stack: String,
    /// Indeks irisan asal.
    pub slide: u32,
    /// Sudut kiri-atas tile pada gambar sub4 (piksel).
    pub x: usize,
    /// Sudut kiri-atas tile pada gambar sub4 (piksel).
    pub y: usize,
    /// Fraksi piksel jaringan (kelas ≥ 1).
    pub tissue_frac: f32,
    /// Fraksi piksel tumor (kelas 2).
    pub tumor_frac: f32,
}

/// Indeks cache tile (`index.json`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TileIndex {
    /// Sisi tile.
    pub tile: usize,
    /// Langkah jendela saat memotong.
    pub stride: usize,
    /// mpp gambar sub4 (= mpp level 0 × 4) dan sumbernya.
    pub mpp: f64,
    /// Sumber nilai mpp level 0 (aturan keras 7: tidak ditebak).
    pub mpp_source: String,
    /// Nama kelas (indeks = nilai kelas).
    pub classes: Vec<Option<String>>,
    /// Daftar tile.
    pub tiles: Vec<TileMeta>,
}

impl TileIndex {
    /// Simpan `index.json` secara atomik.
    pub fn save(&self, path: &Path) -> Result<(), TrainError> {
        let tmp = path.with_extension("tmp");
        let text = serde_json::to_string_pretty(self)?;
        fs::write(&tmp, text).map_err(|e| TrainError::io(&tmp, e))?;
        fs::rename(&tmp, path).map_err(|e| TrainError::io(path, e))
    }

    /// Muat `index.json`.
    pub fn load(path: &Path) -> Result<Self, TrainError> {
        let text = fs::read_to_string(path).map_err(|e| TrainError::io(path, e))?;
        let idx: Self = serde_json::from_str(&text)?;
        if idx.tile == 0 || idx.stride == 0 || !(idx.mpp.is_finite() && idx.mpp > 0.0) {
            return Err(TrainError::Dataset(
                "index.json: tile/stride/mpp tidak sah".into(),
            ));
        }
        Ok(idx)
    }

    /// Tile dari tumpukan yang bukan `holdout` (latih) dan yang `holdout` (uji).
    pub fn split<'a>(&'a self, holdout: &[String]) -> (Vec<&'a TileMeta>, Vec<&'a TileMeta>) {
        self.tiles
            .iter()
            .partition(|t| !holdout.iter().any(|h| h == &t.stack))
    }
}

/// Parameter pemotongan tile.
#[derive(Debug, Clone)]
pub struct CutParams {
    /// Sisi tile (piksel sub4).
    pub tile: usize,
    /// Langkah jendela.
    pub stride: usize,
    /// Fraksi jaringan minimum agar tile disimpan.
    pub min_tissue: f32,
    /// mpp level 0 (µm/piksel) dan sumbernya.
    pub mpp0: f64,
    /// Sumber nilai mpp.
    pub mpp_source: String,
}

/// Bangun peta kelas satu irisan.
pub fn class_map(image: &RgbImage, tissue: &Mask, tumor: &Mask) -> Result<Vec<u8>, TrainError> {
    if tissue.width != image.width || tissue.height != image.height {
        return Err(TrainError::Dataset(format!(
            "ukuran masker jaringan {}×{} ≠ gambar {}×{}",
            tissue.width, tissue.height, image.width, image.height
        )));
    }
    if tumor.width != image.width || tumor.height != image.height {
        return Err(TrainError::Dataset(format!(
            "ukuran masker tumor {}×{} ≠ gambar {}×{}",
            tumor.width, tumor.height, image.width, image.height
        )));
    }
    Ok(tissue
        .data
        .iter()
        .zip(tumor.data.iter())
        .map(|(&t, &u)| if u != 0 { 2 } else { t })
        .collect())
}

/// Posisi awal jendela sepanjang `len` dengan `stride`, jendela terakhir
/// digeser agar tepat di ujung.
pub fn window_starts(len: usize, size: usize, stride: usize) -> Vec<usize> {
    if len <= size {
        return vec![0];
    }
    let mut v: Vec<usize> = (0..)
        .map(|i| i * stride)
        .take_while(|&s| s + size < len)
        .collect();
    v.push(len - size);
    v
}

/// Potong tile dari satu irisan; mengembalikan (tile, x, y).
pub fn cut_slide(image: &RgbImage, cls: &[u8], p: &CutParams) -> Vec<(Tile, usize, usize)> {
    let (w, h) = (image.width, image.height);
    let s = p.tile;
    let mut out = Vec::new();
    for &y in &window_starts(h, s, p.stride) {
        for &x in &window_starts(w, s, p.stride) {
            let mut rgb = Vec::with_capacity(s * s * 3);
            let mut c = Vec::with_capacity(s * s);
            let mut tissue = 0usize;
            for yy in y..(y + s).min(h) {
                let row = yy * w;
                for xx in x..(x + s).min(w) {
                    let i = row + xx;
                    rgb.extend_from_slice(&image.rgb[i * 3..i * 3 + 3]);
                    c.push(cls[i]);
                    tissue += usize::from(cls[i] != 0);
                }
            }
            // Tile di tepi gambar yang lebih kecil dari s tidak terjadi karena
            // window_starts menggeser jendela terakhir, kecuali gambar < s.
            if c.len() != s * s {
                continue;
            }
            let frac = tissue as f32 / (s * s) as f32;
            if frac < p.min_tissue {
                continue;
            }
            out.push((
                Tile {
                    size: s,
                    rgb,
                    cls: c,
                },
                x,
                y,
            ));
        }
    }
    out
}

/// Potong seluruh dataset ke `cache_dir/tiles` dan tulis `cache_dir/index.json`.
/// Irisan tanpa masker jaringan atau tumor dilewati (dilaporkan lewat `log`).
/// `log` dipanggil per irisan dengan pesan kemajuan.
pub fn cut_dataset(
    slides: &[SlideFiles],
    cache_dir: &Path,
    p: &CutParams,
    log: &dyn Fn(&str),
) -> Result<TileIndex, TrainError> {
    if p.tile == 0 || p.stride == 0 || !(p.mpp0.is_finite() && p.mpp0 > 0.0) {
        return Err(TrainError::InvalidParameter(
            "tile/stride/mpp0 harus > 0".into(),
        ));
    }
    let tiles_dir = cache_dir.join("tiles");
    fs::create_dir_all(&tiles_dir).map_err(|e| TrainError::io(&tiles_dir, e))?;
    let mut index = TileIndex {
        tile: p.tile,
        stride: p.stride,
        mpp: p.mpp0 * SUB4_FACTOR,
        mpp_source: p.mpp_source.clone(),
        classes: CLASS_NAMES.iter().map(|c| c.map(str::to_string)).collect(),
        tiles: Vec::new(),
    };
    let mut id = 0u32;
    let mut skipped = 0usize;
    for s in slides {
        let (Some(tissue_path), Some(tumor_path)) = (&s.tissue_mask, &s.tumor_mask) else {
            skipped += 1;
            log(&format!(
                "{}/im{}: masker jaringan/tumor tidak lengkap, dilewati",
                s.stack, s.index
            ));
            continue;
        };
        let image = load_rgb(&s.image)?;
        let tissue = load_mask(tissue_path)?;
        let tumor = load_mask(tumor_path)?;
        let cls = class_map(&image, &tissue, &tumor)?;
        let cut = cut_slide(&image, &cls, p);
        let n = cut.len();
        for (tile, x, y) in cut {
            let file = format!("{id:06}.bin");
            write_tile(&tiles_dir.join(&file), &tile)?;
            let total = (tile.size * tile.size) as f32;
            let tissue_px = tile.cls.iter().filter(|&&c| c != 0).count() as f32;
            let tumor_px = tile.cls.iter().filter(|&&c| c == 2).count() as f32;
            index.tiles.push(TileMeta {
                file,
                stack: s.stack.clone(),
                slide: s.index,
                x,
                y,
                tissue_frac: tissue_px / total,
                tumor_frac: tumor_px / total,
            });
            id += 1;
        }
        log(&format!("{}/im{}: {n} tile", s.stack, s.index));
    }
    if index.tiles.is_empty() {
        return Err(TrainError::Dataset(format!(
            "tidak ada tile yang dihasilkan ({skipped} irisan tanpa masker lengkap)"
        )));
    }
    index.save(&cache_dir.join("index.json"))?;
    Ok(index)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_starts_cover_extent() {
        assert_eq!(window_starts(10, 64, 48), vec![0]);
        assert_eq!(window_starts(64, 64, 48), vec![0]);
        let v = window_starts(200, 64, 48);
        assert_eq!(v, vec![0, 48, 96, 136]);
    }

    #[test]
    fn tile_roundtrip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let t = Tile {
            size: 2,
            rgb: (0..12).collect(),
            cls: vec![0, 1, 2, 1],
        };
        let p = dir.path().join("t.bin");
        write_tile(&p, &t).expect("tulis");
        assert_eq!(read_tile(&p).expect("baca"), t);
        assert!(!p.with_extension("tmp").exists());
    }

    #[test]
    fn class_map_prefers_tumor() {
        let img = RgbImage {
            width: 2,
            height: 1,
            rgb: vec![0; 6],
        };
        let tissue = Mask {
            width: 2,
            height: 1,
            data: vec![1, 0],
        };
        let tumor = Mask {
            width: 2,
            height: 1,
            data: vec![1, 1],
        };
        assert_eq!(class_map(&img, &tissue, &tumor).expect("ok"), vec![2, 2]);
        let bad = Mask {
            width: 1,
            height: 1,
            data: vec![0],
        };
        assert!(class_map(&img, &bad, &tumor).is_err());
    }

    #[test]
    fn cut_skips_background_tiles() {
        let s = 4;
        let img = RgbImage {
            width: 8,
            height: 4,
            rgb: vec![200; 8 * 4 * 3],
        };
        // Kiri jaringan penuh, kanan kosong.
        let cls: Vec<u8> = (0..32).map(|i| u8::from(i % 8 < 4)).collect();
        let p = CutParams {
            tile: s,
            stride: s,
            min_tissue: 0.5,
            mpp0: 1.0,
            mpp_source: "uji".into(),
        };
        let cut = cut_slide(&img, &cls, &p);
        assert_eq!(cut.len(), 1);
        assert_eq!((cut[0].1, cut[0].2), (0, 0));
    }
}
