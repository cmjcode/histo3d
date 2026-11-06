//! Akses piksel slide: pembukaan sumber, thumbnail, pembacaan pada resolusi
//! kerja (dalam strip), dan gambar registrasi ber-cache (`cache/reg/`).

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, MutexGuard};

use h3d_core::{Progress, SlideEntry, SlideId, SlideSourceRef};
use h3d_io::{slide_stem, write_atomic};
use h3d_register::RegImage;
use h3d_slide::{SlideSource, TiledPyramid};
use image::RgbaImage;
use ndarray::Array2;
use rayon::prelude::*;

use super::{FileHash, Session};
use crate::error::EngineError;

/// Sumber piksel terbuka per irisan: LRU berbatas jumlah
/// (`Budgets::max_open_slides`).
pub(crate) struct SourceCache {
    max: usize,
    tick: u64,
    map: HashMap<SlideId, (Arc<dyn SlideSource>, u64)>,
}

impl SourceCache {
    pub(crate) fn new(max: usize) -> Self {
        Self {
            max: max.max(1),
            tick: 0,
            map: HashMap::new(),
        }
    }

    pub(crate) fn set_max(&mut self, max: usize) {
        self.max = max.max(1);
        self.evict();
    }

    fn contains(&self, id: SlideId) -> bool {
        self.map.contains_key(&id)
    }

    fn get(&mut self, id: SlideId) -> Option<Arc<dyn SlideSource>> {
        self.tick += 1;
        let e = self.map.get_mut(&id)?;
        e.1 = self.tick;
        Some(Arc::clone(&e.0))
    }

    fn insert(&mut self, id: SlideId, src: Arc<dyn SlideSource>) {
        self.tick += 1;
        self.map.insert(id, (src, self.tick));
        self.evict();
    }

    fn remove(&mut self, id: SlideId) {
        self.map.remove(&id);
    }

    fn evict(&mut self) {
        while self.map.len() > self.max {
            let Some(&victim) = self
                .map
                .iter()
                .min_by_key(|(id, (_, t))| (*t, id.0))
                .map(|(id, _)| id)
            else {
                break;
            };
            self.map.remove(&victim);
        }
    }
}

/// Resolusi gambar registrasi (µm/px) = level terhalus bawaan registrasi.
pub const REG_IMAGE_MPP: f64 = 16.0;
/// Batas memori satu strip pembacaan (byte RGBA).
const STRIP_BYTES: u64 = 64 * 1024 * 1024;

const REG_MAGIC: &[u8; 4] = b"H3DR";
const REG_VERSION: u32 = 1;

/// Citra RGBA pada resolusi isotropik `mpp`, sudut `(0, 0)` = sudut slide.
pub(crate) struct Raster {
    pub rgba: Vec<u8>,
    pub w: usize,
    pub h: usize,
    pub mpp: f64,
}

/// Level piramida beserta resolusi fisiknya.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LevelChoice {
    pub level: u32,
    pub size: [u32; 2],
    /// µm per piksel level `[x, y]`.
    pub mpp: [f64; 2],
}

fn level_choices(src: &dyn SlideSource, mpp_l0: [f64; 2]) -> Vec<LevelChoice> {
    src.info()
        .levels
        .iter()
        .enumerate()
        .map(|(i, l)| LevelChoice {
            level: i as u32,
            size: l.size,
            mpp: [mpp_l0[0] * l.downsample, mpp_l0[1] * l.downsample],
        })
        .collect()
}

/// Level paling kasar yang resolusinya masih ≤ `target_mpp` (tidak
/// kehilangan detail); bila level 0 pun lebih kasar, level 0.
pub(crate) fn coarsest_level_at_most(
    src: &dyn SlideSource,
    mpp_l0: [f64; 2],
    target_mpp: f64,
) -> Result<LevelChoice, EngineError> {
    let levels = level_choices(src, mpp_l0);
    let first = *levels
        .first()
        .ok_or_else(|| EngineError::invalid("slide tanpa level piramida"))?;
    Ok(levels
        .into_iter()
        .rfind(|l| l.mpp[0].max(l.mpp[1]) <= target_mpp * (1.0 + 1e-9))
        .unwrap_or(first))
}

/// Level yang resolusinya paling dekat dengan `target_mpp` (jarak rasio log).
pub(crate) fn nearest_level(
    src: &dyn SlideSource,
    mpp_l0: [f64; 2],
    target_mpp: f64,
) -> Result<LevelChoice, EngineError> {
    let levels = level_choices(src, mpp_l0);
    let mut best: Option<(f64, LevelChoice)> = None;
    for l in levels {
        let m = (l.mpp[0] + l.mpp[1]) / 2.0;
        let d = (m / target_mpp).ln().abs();
        // Seri → level berindeks lebih kecil (lebih halus).
        if best.as_ref().is_none_or(|(bd, _)| d < *bd) {
            best = Some((d, l));
        }
    }
    best.map(|(_, l)| l)
        .ok_or_else(|| EngineError::invalid("slide tanpa level piramida"))
}

/// Baca seluruh slide pada resolusi isotropik ≈ `target_mpp` dengan
/// rata-rata kotak. Level piramida dipilih agar tidak pernah membaca level 0
/// slide besar sekaligus; pembacaan dilakukan per strip ≤ 64 MB.
///
/// Resolusi keluaran = `max(target_mpp, mpp level terpilih)`.
pub(crate) fn read_at_mpp(
    src: &dyn SlideSource,
    mpp_l0: [f64; 2],
    target_mpp: f64,
    p: &dyn Progress,
) -> Result<Raster, EngineError> {
    read_at_mpp_strips(src, mpp_l0, target_mpp, STRIP_BYTES, p)
}

/// [`read_at_mpp`] dengan batas memori strip eksplisit (`strip_bytes`, byte
/// RGBA sumber per strip). Parameter ini internal: produksi selalu memakai
/// [`STRIP_BYTES`]; tes menurunkannya agar jalur multi-strip teruji tanpa
/// mengalokasikan gambar raksasa. Hasil tidak bergantung pada ukuran strip.
pub(crate) fn read_at_mpp_strips(
    src: &dyn SlideSource,
    mpp_l0: [f64; 2],
    target_mpp: f64,
    strip_bytes: u64,
    p: &dyn Progress,
) -> Result<Raster, EngineError> {
    if !(target_mpp.is_finite() && target_mpp > 0.0) {
        return Err(EngineError::invalid("resolusi sasaran tidak sah"));
    }
    let lc = coarsest_level_at_most(src, mpp_l0, target_mpp)?;
    let [lw, lh] = lc.size;
    if lw == 0 || lh == 0 {
        return Err(EngineError::invalid("level slide berukuran nol"));
    }
    let mpp_out = target_mpp.max(lc.mpp[0]).max(lc.mpp[1]);
    let rx = mpp_out / lc.mpp[0];
    let ry = mpp_out / lc.mpp[1];
    let ow = ((f64::from(lw) / rx).ceil() as usize).max(1);
    let oh = ((f64::from(lh) / ry).ceil() as usize).max(1);
    let bound = |o: usize, r: f64, len: u32| -> (u32, u32) {
        let a = ((o as f64 * r).floor() as u32).min(len - 1);
        let b = (((o + 1) as f64 * r).floor() as u32).clamp(a + 1, len);
        (a, b)
    };
    let xb: Vec<(u32, u32)> = (0..ow).map(|o| bound(o, rx, lw)).collect();
    let mut rgba = vec![255u8; ow * oh * 4];

    // Jumlah baris keluaran per strip agar strip sumber ≤ STRIP_BYTES.
    let src_rows_budget = (strip_bytes / (u64::from(lw) * 4)).max(1) as f64;
    let rows_per_strip = ((src_rows_budget / ry).floor() as usize).max(1);
    let mut oy = 0usize;
    while oy < oh {
        if p.is_cancelled() {
            return Err(EngineError::Cancelled);
        }
        let oy_end = (oy + rows_per_strip).min(oh);
        let (y0, _) = bound(oy, ry, lh);
        let (_, y1) = bound(oy_end - 1, ry, lh);
        let strip = src.read_region(lc.level, 0, i64::from(y0), lw, y1 - y0)?;
        let raw = strip.as_raw();
        let stride = lw as usize * 4;
        // Baris keluaran saling bebas: dihitung paralel, hasil deterministik.
        rgba[oy * ow * 4..oy_end * ow * 4]
            .par_chunks_mut(ow * 4)
            .enumerate()
            .for_each(|(k, out_row)| {
                let o = oy + k;
                let (ya, yb) = bound(o, ry, lh);
                for (ox, &(xa, xb_)) in xb.iter().enumerate() {
                    let mut acc = [0u64; 3];
                    for yy in ya..yb {
                        let row = (yy - y0) as usize * stride;
                        for xx in xa..xb_ {
                            let i = row + xx as usize * 4;
                            acc[0] += u64::from(raw[i]);
                            acc[1] += u64::from(raw[i + 1]);
                            acc[2] += u64::from(raw[i + 2]);
                        }
                    }
                    let n = u64::from(yb - ya) * u64::from(xb_ - xa);
                    let d = ox * 4;
                    for k in 0..3 {
                        out_row[d + k] = ((acc[k] + n / 2) / n) as u8;
                    }
                }
            });
        oy = oy_end;
        p.set(oy as f32 / oh as f32, "membaca slide");
    }
    Ok(Raster {
        rgba,
        w: ow,
        h: oh,
        mpp: mpp_out,
    })
}

/// Pembuatan thumbnail yang mandiri dari `Session` (lihat
/// [`Session::thumbnail_task`]).
pub struct ThumbTask {
    source: Option<Arc<dyn SlideSource>>,
    max_side: u32,
    plain_path: PathBuf,
    tissue_path: Option<PathBuf>,
    mpp_l0: [f64; 2],
    size_l0: [u32; 2],
}

impl ThumbTask {
    /// Jalankan: pakai cache disk bila ada, selain itu baca level piramida
    /// terkecil yang cukup, turunkan, (beri masker jaringan), dan simpan ke
    /// cache. Gagal menulis cache hanya dicatat.
    pub fn run(&self) -> Result<RgbaImage, EngineError> {
        let final_path = self.tissue_path.as_ref().unwrap_or(&self.plain_path);
        if let Ok(img) = image::open(final_path) {
            return Ok(img.to_rgba8());
        }
        let mut img = match image::open(&self.plain_path) {
            Ok(i) => i.to_rgba8(),
            Err(_) => {
                let src = self
                    .source
                    .as_ref()
                    .ok_or_else(|| EngineError::invalid("sumber thumbnail tidak dibuka"))?;
                let img = h3d_slide::thumbnail(src.as_ref(), self.max_side)?;
                write_png_cache(&self.plain_path, &img, "thumbnail");
                img
            }
        };
        let Some(tissue_path) = &self.tissue_path else {
            return Ok(img);
        };
        let (w, h) = (img.width() as usize, img.height() as usize);
        if w == 0 || h == 0 {
            return Ok(img);
        }
        // mpp thumbnail = mpp level 0 × faktor penurunan.
        let mpp_thumb = self.mpp_l0[0] * f64::from(self.size_l0[0]) / w as f64;
        h3d_imgproc::apply_tissue_alpha(img.as_mut(), w, h, mpp_thumb, h3d_imgproc::FEATHER_PX)?;
        write_png_cache(tissue_path, &img, "thumbnail jaringan");
        Ok(img)
    }
}

fn write_png_cache(path: &Path, img: &RgbaImage, what: &str) {
    let mut png = Vec::new();
    let enc = image::codecs::png::PngEncoder::new(&mut png);
    if image::ImageEncoder::write_image(
        enc,
        img.as_raw(),
        img.width(),
        img.height(),
        image::ExtendedColorType::Rgba8,
    )
    .is_ok()
    {
        // Gagal menulis cache bukan galat bagi pemanggil.
        if let Err(e) = write_atomic(path, &png) {
            log::warn!("gagal menulis cache {what}: {e}");
        }
    }
}

fn build_pyramid_from_path(dir: &Path, path: &Path, p: &dyn Progress) -> Result<(), EngineError> {
    log::info!(
        "membangun piramida bertile {} → {}",
        path.display(),
        dir.display()
    );
    h3d_slide::build_tiled_pyramid_from_path(dir, path, |done, total| {
        p.set(done as f32 / total.max(1) as f32, "membangun piramida");
        !p.is_cancelled()
    })
    .map_err(map_build_err)
}

fn build_pyramid_from_source(
    dir: &Path,
    src: &dyn SlideSource,
    p: &dyn Progress,
) -> Result<(), EngineError> {
    log::info!("membangun piramida bertile dari sumber → {}", dir.display());
    h3d_slide::build_tiled_pyramid_from_source(dir, src, |done, total| {
        p.set(done as f32 / total.max(1) as f32, "membangun piramida");
        !p.is_cancelled()
    })
    .map_err(map_build_err)
}

fn map_build_err(e: h3d_slide::SlideError) -> EngineError {
    match e {
        h3d_slide::SlideError::Cancelled => EngineError::Cancelled,
        other => other.into(),
    }
}

/// Baca wilayah `w_px × h_px` yang pojok kiri-atasnya di `(x_um, y_um)` pada
/// resolusi isotropik `mpp` (untuk inferensi AI). Tiap piksel keluaran =
/// rata-rata kotak piksel level yang dicakupnya; luar slide putih.
pub(crate) fn read_region_at_mpp(
    src: &dyn SlideSource,
    mpp_l0: [f64; 2],
    x_um: f64,
    y_um: f64,
    w_px: usize,
    h_px: usize,
    mpp: f64,
) -> Result<Vec<u8>, EngineError> {
    if w_px == 0 || h_px == 0 || !(mpp.is_finite() && mpp > 0.0) {
        return Err(EngineError::invalid("wilayah baca tidak sah"));
    }
    let lc = coarsest_level_at_most(src, mpp_l0, mpp)?;
    let rx = mpp / lc.mpp[0];
    let ry = mpp / lc.mpp[1];
    let sx0 = (x_um / lc.mpp[0]).floor() as i64;
    let sy0 = (y_um / lc.mpp[1]).floor() as i64;
    let sx1 = ((x_um + w_px as f64 * mpp) / lc.mpp[0]).ceil() as i64 + 1;
    let sy1 = ((y_um + h_px as f64 * mpp) / lc.mpp[1]).ceil() as i64 + 1;
    let sw = u32::try_from(sx1 - sx0).map_err(|_| EngineError::invalid("wilayah terlalu besar"))?;
    let sh = u32::try_from(sy1 - sy0).map_err(|_| EngineError::invalid("wilayah terlalu besar"))?;
    let region = src.read_region(lc.level, sx0, sy0, sw, sh)?;
    let raw = region.as_raw();
    let stride = sw as usize * 4;
    let fx0 = x_um / lc.mpp[0] - sx0 as f64;
    let fy0 = y_um / lc.mpp[1] - sy0 as f64;
    let span = |o: usize, f0: f64, r: f64, len: u32| -> (usize, usize) {
        let a = ((f0 + o as f64 * r).floor().max(0.0) as usize).min(len as usize - 1);
        let b = ((f0 + (o + 1) as f64 * r).ceil().max(0.0) as usize).clamp(a + 1, len as usize);
        (a, b)
    };
    let mut out = vec![255u8; w_px * h_px * 4];
    out.par_chunks_mut(w_px * 4)
        .enumerate()
        .for_each(|(oy, out_row)| {
            let (ya, yb) = span(oy, fy0, ry, sh);
            for ox in 0..w_px {
                let (xa, xb) = span(ox, fx0, rx, sw);
                let mut acc = [0u64; 3];
                for yy in ya..yb {
                    for xx in xa..xb {
                        let i = yy * stride + xx * 4;
                        acc[0] += u64::from(raw[i]);
                        acc[1] += u64::from(raw[i + 1]);
                        acc[2] += u64::from(raw[i + 2]);
                    }
                }
                let n = ((yb - ya) * (xb - xa)) as u64;
                let d = ox * 4;
                for k in 0..3 {
                    out_row[d + k] = ((acc[k] + n / 2) / n) as u8;
                }
            }
        });
    Ok(out)
}

fn encode_reg(img: &RegImage, fingerprint: u64, mpp_l0: [f64; 2]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(REG_MAGIC);
    out.extend_from_slice(&REG_VERSION.to_le_bytes());
    out.extend_from_slice(&fingerprint.to_le_bytes());
    out.extend_from_slice(&mpp_l0[0].to_le_bytes());
    out.extend_from_slice(&mpp_l0[1].to_le_bytes());
    out.extend_from_slice(&img.mpp.to_le_bytes());
    out.extend_from_slice(&img.origin_um[0].to_le_bytes());
    out.extend_from_slice(&img.origin_um[1].to_le_bytes());
    out.extend_from_slice(&(img.width() as u32).to_le_bytes());
    out.extend_from_slice(&(img.height() as u32).to_le_bytes());
    let mut gray = Vec::with_capacity(img.gray.len() * 4);
    for v in img.gray.iter() {
        gray.extend_from_slice(&v.to_le_bytes());
    }
    let gz = lz4_flex::compress_prepend_size(&gray);
    out.extend_from_slice(&(gz.len() as u64).to_le_bytes());
    out.extend_from_slice(&gz);
    let mask: Vec<u8> = img.mask.iter().copied().collect();
    out.extend_from_slice(&lz4_flex::compress_prepend_size(&mask));
    out
}

/// Baca cache gambar registrasi; `None` bila berkas rusak atau bukan untuk
/// berkas slide/mpp yang sama (cache lalu dibuat ulang).
fn decode_reg(bytes: &[u8], fingerprint: u64, mpp_l0: [f64; 2]) -> Option<RegImage> {
    const HEAD: usize = 4 + 4 + 8 + 8 * 5 + 4 + 4 + 8;
    if bytes.len() < HEAD || &bytes[0..4] != REG_MAGIC {
        return None;
    }
    let u32_at =
        |o: usize| -> Option<u32> { Some(u32::from_le_bytes(bytes[o..o + 4].try_into().ok()?)) };
    let u64_at =
        |o: usize| -> Option<u64> { Some(u64::from_le_bytes(bytes[o..o + 8].try_into().ok()?)) };
    let f64_at =
        |o: usize| -> Option<f64> { Some(f64::from_le_bytes(bytes[o..o + 8].try_into().ok()?)) };
    if u32_at(4)? != REG_VERSION || u64_at(8)? != fingerprint {
        return None;
    }
    if f64_at(16)? != mpp_l0[0] || f64_at(24)? != mpp_l0[1] {
        return None;
    }
    let mpp = f64_at(32)?;
    let origin = [f64_at(40)?, f64_at(48)?];
    let w = u32_at(56)? as usize;
    let h = u32_at(60)? as usize;
    let gz_len = usize::try_from(u64_at(64)?).ok()?;
    let gz_end = HEAD.checked_add(gz_len).filter(|e| *e <= bytes.len())?;
    let n = w.checked_mul(h)?;
    let gray_bytes = lz4_flex::decompress_size_prepended(&bytes[HEAD..gz_end]).ok()?;
    let mask = lz4_flex::decompress_size_prepended(&bytes[gz_end..]).ok()?;
    if gray_bytes.len() != n.checked_mul(4)? || mask.len() != n {
        return None;
    }
    let gray: Vec<f32> = gray_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let gray = Array2::from_shape_vec((h, w), gray).ok()?;
    let mask = Array2::from_shape_vec((h, w), mask).ok()?;
    RegImage::new(gray, mask, mpp, origin).ok()
}

/// Gambar biasa (tanpa piramida) menurut ekstensi; dibaca lewat piramida
/// bertile di `cache/pyramid/`.
pub(crate) fn has_flat_extension(path: &Path) -> bool {
    path.extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .is_some_and(|e| matches!(e.as_str(), "png" | "jpg" | "jpeg" | "bmp" | "gif" | "webp"))
}

/// Sumber yang level teratasnya masih besar (TIFF datar / piramida dangkal)
/// perlu piramida bertile agar tampilan jauh tidak membaca level 0.
fn needs_tiled_pyramid(src: &dyn SlideSource) -> bool {
    src.info()
        .levels
        .last()
        .is_some_and(|top| top.size[0].max(top.size[1]) > 2 * h3d_slide::tiled::MIN_TOP_SIDE)
}

/// Progres satu pembangunan piramida di dalam [`Session::prepare_pyramids`]:
/// fraksinya disimpan di slot sendiri, lalu dijumlahkan dengan slot lain.
struct PyramidProgress<'a> {
    parent: &'a dyn Progress,
    slot: &'a AtomicU32,
    report: &'a (dyn Fn(&str) + Sync),
}

impl Progress for PyramidProgress<'_> {
    fn set(&self, fraction: f32, stage: &str) {
        self.slot
            .store(fraction.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
        (self.report)(stage);
    }
    fn is_cancelled(&self) -> bool {
        self.parent.is_cancelled()
    }
}

impl Session {
    /// Path absolut berkas slide.
    pub fn slide_path(&self, slide: SlideId) -> Result<std::path::PathBuf, EngineError> {
        Ok(self.bundle.resolve_slide_path(&self.slide(slide)?.source))
    }

    /// Irisan yang berkas slidenya tidak ditemukan (mis. berkas `External`
    /// dipindah); GUI menawarkan untuk menautkan ulang.
    pub fn missing_slides(&self) -> Vec<SlideId> {
        self.project
            .slides_ordered()
            .iter()
            .filter(|s| !self.bundle.resolve_slide_path(&s.source).is_file())
            .map(|s| s.id)
            .collect()
    }

    /// Himpunan irisan berubah (racun kunci diabaikan: isinya tetap sah).
    pub(crate) fn changed_set(&self) -> MutexGuard<'_, BTreeSet<SlideId>> {
        self.changed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Irisan `External` yang berkasnya ada tetapi tidak lagi cocok dengan
    /// ukuran + sidik jari saat impor, terurut id. Pikselnya ditolak dibaca
    /// sampai [`Self::relink_slide`] atau [`Self::accept_changed_slide`].
    pub fn changed_slides(&self) -> Vec<SlideId> {
        self.changed_set().iter().copied().collect()
    }

    /// Periksa ulang semua berkas slide `External` terhadap sidik jari
    /// tersimpan dan perbarui [`Self::changed_slides`]. Dipanggil saat proyek
    /// dibuka; GUI boleh memanggilnya lagi (mis. saat aplikasi kembali aktif).
    /// Berkas yang hilang atau tidak terbaca tidak dihitung "berubah".
    pub fn recheck_slides(&self) -> Vec<SlideId> {
        // Sidik jari (2 MB per berkas) dihitung paralel; hasil dikumpulkan
        // ke himpunan terurut sehingga deterministik.
        let found: BTreeSet<SlideId> = self
            .project
            .slides
            .par_iter()
            .filter_map(|s| {
                let SlideSourceRef::External {
                    size_bytes,
                    fingerprint,
                    ..
                } = &s.source
                else {
                    return None;
                };
                let path = self.bundle.resolve_slide_path(&s.source);
                if !path.is_file() {
                    return None;
                }
                match self.cached_fingerprint(s.id, &path) {
                    Ok(now) if now != (*size_bytes, *fingerprint) => Some(s.id),
                    Ok(_) => None,
                    Err(e) => {
                        log::warn!("sidik jari {} tidak bisa dihitung: {e}", path.display());
                        None
                    }
                }
            })
            .collect();
        let list: Vec<SlideId> = found.iter().copied().collect();
        *self.changed_set() = found;
        list
    }

    /// Sidik jari `path` untuk irisan `slide`, dihitung ulang hanya bila
    /// ukuran atau mtime berkas berubah sejak perhitungan terakhir.
    fn cached_fingerprint(&self, slide: SlideId, path: &Path) -> Result<(u64, u64), EngineError> {
        let meta = std::fs::metadata(path).map_err(|e| h3d_io::IoError::io(path, e))?;
        let (len, modified) = (meta.len(), meta.modified().ok());
        if let Ok(cache) = self.fingerprints.lock() {
            if let Some(fh) = cache.get(&slide) {
                if fh.len == len && fh.modified == modified {
                    return Ok((len, fh.hash));
                }
            }
        }
        let now = h3d_slide::fingerprint(path)?;
        if let Ok(mut cache) = self.fingerprints.lock() {
            cache.insert(
                slide,
                FileHash {
                    len,
                    modified,
                    hash: now.1,
                },
            );
        }
        Ok(now)
    }

    /// Lupakan sumber terbuka dan sidik jari irisan (berkas diganti,
    /// ditautkan ulang, atau irisan dihapus).
    pub(crate) fn invalidate_source(&self, slide: SlideId) {
        if let Ok(mut c) = self.sources.lock() {
            c.remove(slide);
        }
        if let Ok(mut c) = self.fingerprints.lock() {
            c.remove(&slide);
        }
    }

    /// Path dan sidik jari berkas slide **yang sudah diverifikasi**: berkas
    /// harus ada, dan untuk `External` ukuran + sidik jarinya harus sama
    /// dengan saat impor. Semua pembacaan piksel lewat sini, sehingga berkas
    /// yang diganti setelah proyek dibuka pun tertangkap.
    pub(crate) fn verified_source(
        &self,
        entry: &SlideEntry,
    ) -> Result<(PathBuf, u64), EngineError> {
        let path = self.bundle.resolve_slide_path(&entry.source);
        if !path.is_file() {
            return Err(h3d_io::IoError::MissingFile(path).into());
        }
        let now = self.cached_fingerprint(entry.id, &path)?;
        if let SlideSourceRef::External {
            size_bytes,
            fingerprint,
            ..
        } = &entry.source
        {
            if now != (*size_bytes, *fingerprint) {
                self.changed_set().insert(entry.id);
                return Err(EngineError::SlideChanged {
                    slide: entry.id,
                    file: entry.source.file_name(),
                });
            }
            self.changed_set().remove(&entry.id);
        }
        Ok((path, now.1))
    }

    /// Sidik jari berkas slide saat ini (untuk kunci cache); gagal bila
    /// berkas hilang atau berubah.
    pub(crate) fn source_fingerprint(&self, entry: &SlideEntry) -> Result<u64, EngineError> {
        Ok(self.verified_source(entry)?.1)
    }

    /// Terima berkas slide yang berubah sebagai berkas irisan ini: ukuran dan
    /// sidik jari tersimpan diganti dengan milik berkas saat ini, lalu
    /// `project.json` disimpan. Ukuran piksel level 0 harus sama dengan saat
    /// impor (label tersimpan dalam µm pada kisi irisan itu); bila berbeda,
    /// irisan harus diimpor ulang. Label dan catatan registrasi tidak
    /// disentuh — pengguna menyatakan bahwa ini irisan yang sama; cache
    /// thumbnail/registrasi otomatis dibuat ulang karena berkunci sidik jari.
    pub fn accept_changed_slide(&mut self, slide: SlideId) -> Result<(), EngineError> {
        let entry = self.slide(slide)?.clone();
        let SlideSourceRef::External { path: rel, .. } = &entry.source else {
            return Err(EngineError::invalid(format!(
                "irisan {slide} tersimpan di dalam proyek; tidak ada berkas luar untuk diterima"
            )));
        };
        let path = self.bundle.resolve_slide_path(&entry.source);
        if !path.is_file() {
            return Err(h3d_io::IoError::MissingFile(path).into());
        }
        let src = h3d_slide::open_slide(&path)?;
        let size_l0 = src.info().level0_size();
        if size_l0 != entry.size_l0 {
            return Err(EngineError::invalid(format!(
                "berkas {} berukuran {}×{} px, sedangkan irisan {slide} diimpor dengan {}×{} px; \
                 label tidak akan sejajar — impor ulang irisan ini",
                entry.source.file_name(),
                size_l0[0],
                size_l0[1],
                entry.size_l0[0],
                entry.size_l0[1]
            )));
        }
        let (size_bytes, fingerprint) = h3d_slide::fingerprint(&path)?;
        self.invalidate_source(slide);
        let rel = rel.clone();
        self.slide_mut(slide)?.source = SlideSourceRef::External {
            path: rel,
            size_bytes,
            fingerprint,
        };
        self.changed_set().remove(&slide);
        self.save_project()
    }

    /// Tautkan ulang irisan `External` ke `path` (berkas dipindah atau diganti
    /// nama). Ukuran + sidik jari tersimpan **tidak** diubah: bila berkas
    /// baru cocok, irisan langsung bisa dibaca dan fungsi mengembalikan
    /// `true`; bila tidak cocok, tautan tetap diganti tetapi irisan masuk
    /// [`Self::changed_slides`] dan fungsi mengembalikan `false` (pengguna
    /// lalu memilih [`Self::accept_changed_slide`] atau menautkan berkas lain).
    pub fn relink_slide(&mut self, slide: SlideId, path: &Path) -> Result<bool, EngineError> {
        let entry = self.slide(slide)?.clone();
        let SlideSourceRef::External {
            size_bytes,
            fingerprint,
            ..
        } = entry.source
        else {
            return Err(EngineError::invalid(format!(
                "irisan {slide} tersimpan di dalam proyek; tidak bisa ditautkan ulang"
            )));
        };
        if !path.is_file() {
            return Err(h3d_io::IoError::MissingFile(path.to_path_buf()).into());
        }
        let now = h3d_slide::fingerprint(path)?;
        self.invalidate_source(slide);
        let rel = self.bundle.make_relative(path);
        let duplicate = self.project.slides.iter().any(|s| {
            s.id != slide
                && matches!(&s.source, SlideSourceRef::External { path: p, .. } if *p == rel)
        });
        if duplicate {
            return Err(EngineError::invalid(format!(
                "berkas {} sudah dipakai irisan lain di proyek ini",
                path.display()
            )));
        }
        self.slide_mut(slide)?.source = SlideSourceRef::External {
            path: rel,
            size_bytes,
            fingerprint,
        };
        let matches = now == (size_bytes, fingerprint);
        if matches {
            self.changed_set().remove(&slide);
        } else {
            self.changed_set().insert(slide);
        }
        self.save_project()?;
        Ok(matches)
    }

    /// Folder piramida bertile irisan (berkunci sidik jari berkas).
    fn pyramid_dir(&self, slide: SlideId, fp: u64) -> PathBuf {
        self.bundle
            .cache_pyramid_dir()
            .join(format!("{}-{:016x}", slide_stem(slide), fp))
    }

    /// Buka sumber piksel sebuah irisan. Sumber yang sudah terbuka dipakai
    /// bersama (cache LRU `Budgets::max_open_slides`): thumbnail, registrasi,
    /// segmentasi, atribut, dan viewer tidak membuka backend masing-masing.
    /// Gambar biasa (PNG/JPEG) dan TIFF datar dibaca lewat piramida bertile
    /// di `cache/pyramid/` (dibangun saat impor; bila belum ada, dibangun di
    /// sini tanpa progres). Berkas hilang → `IoError::MissingFile`; berkas
    /// berubah → [`EngineError::SlideChanged`].
    pub fn open_source(&self, slide: SlideId) -> Result<Arc<dyn SlideSource>, EngineError> {
        let entry = self.slide(slide)?;
        let (path, fp) = self.verified_source(entry)?;
        if let Ok(mut c) = self.sources.lock() {
            if let Some(src) = c.get(slide) {
                return Ok(src);
            }
        }
        let src = self.open_source_uncached(slide, &path, fp, &h3d_core::NoProgress)?;
        if let Ok(mut c) = self.sources.lock() {
            c.insert(slide, Arc::clone(&src));
        }
        Ok(src)
    }

    /// Seperti [`Self::open_source`], tetapi **tidak pernah membangun
    /// piramida bertile**: bila irisan memerlukan piramida yang belum ada
    /// (gambar biasa, atau TIFF yang level teratasnya masih besar), hasilnya
    /// [`EngineError::PyramidPending`]. Dipakai jalur thread UI (thumbnail,
    /// viewer); GUI lalu menjalankan [`Self::prepare_pyramid`] sebagai job.
    pub fn try_open_source(&self, slide: SlideId) -> Result<Arc<dyn SlideSource>, EngineError> {
        let entry = self.slide(slide)?;
        let (path, fp) = self.verified_source(entry)?;
        if let Ok(mut c) = self.sources.lock() {
            if let Some(src) = c.get(slide) {
                return Ok(src);
            }
        }
        let dir = self.pyramid_dir(slide, fp);
        let src: Arc<dyn SlideSource> = if h3d_slide::tiled_pyramid_exists(&dir) {
            Arc::new(TiledPyramid::open(&dir)?)
        } else if has_flat_extension(&path) {
            return Err(EngineError::PyramidPending(slide));
        } else {
            let src: Arc<dyn SlideSource> = Arc::from(h3d_slide::open_slide_with_cache_bytes(
                &path,
                self.budgets.wsi_internal_cache_bytes,
            )?);
            if needs_tiled_pyramid(src.as_ref()) {
                return Err(EngineError::PyramidPending(slide));
            }
            src
        };
        if let Ok(mut c) = self.sources.lock() {
            c.insert(slide, Arc::clone(&src));
        }
        Ok(src)
    }

    /// Bangun piramida irisan `ids` sekaligus, paling banyak
    /// `Budgets::pyramid_build_jobs` bersamaan. Progres gabungan dilaporkan ke
    /// `p`; pembatalan menghentikan semua. Irisan yang gagal tidak
    /// menggagalkan yang lain: galatnya dikembalikan terurut menurut `ids`.
    pub fn prepare_pyramids(
        &self,
        ids: &[SlideId],
        p: &dyn Progress,
    ) -> Result<Vec<(SlideId, EngineError)>, EngineError> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let jobs = self.budgets.pyramid_build_jobs.clamp(1, ids.len());
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(jobs)
            .build()
            .map_err(|e| EngineError::invalid(format!("kolam thread piramida: {e}")))?;
        let fractions: Vec<AtomicU32> = ids.iter().map(|_| AtomicU32::new(0)).collect();
        let total = ids.len() as f32;
        let report = |stage: &str| {
            let sum: f32 = fractions
                .iter()
                .map(|f| f32::from_bits(f.load(Ordering::Relaxed)))
                .sum();
            p.set(sum / total, stage);
        };
        let results: Vec<Result<(), EngineError>> = pool.install(|| {
            ids.par_iter()
                .enumerate()
                .map(|(i, &id)| {
                    if p.is_cancelled() {
                        return Err(EngineError::Cancelled);
                    }
                    let sub = PyramidProgress {
                        parent: p,
                        slot: &fractions[i],
                        report: &report,
                    };
                    let r = self.prepare_pyramid(id, &sub);
                    fractions[i].store(1.0f32.to_bits(), Ordering::Relaxed);
                    report("membangun piramida");
                    r
                })
                .collect()
        });
        let mut failed = Vec::new();
        for (id, r) in ids.iter().zip(results) {
            match r {
                Ok(()) => {}
                Err(EngineError::Cancelled) => return Err(EngineError::Cancelled),
                Err(e) => failed.push((*id, e)),
            }
        }
        Ok(failed)
    }

    fn open_source_uncached(
        &self,
        slide: SlideId,
        path: &Path,
        fp: u64,
        p: &dyn Progress,
    ) -> Result<Arc<dyn SlideSource>, EngineError> {
        let dir = self.pyramid_dir(slide, fp);
        if h3d_slide::tiled_pyramid_exists(&dir) {
            match TiledPyramid::open(&dir) {
                Ok(t) => return Ok(Arc::new(t)),
                Err(e) => log::warn!(
                    "cache piramida {} rusak, dibangun ulang: {e}",
                    dir.display()
                ),
            }
        }
        if has_flat_extension(path) {
            build_pyramid_from_path(&dir, path, p)?;
            return Ok(Arc::new(TiledPyramid::open(&dir)?));
        }
        let src: Arc<dyn SlideSource> = Arc::from(h3d_slide::open_slide_with_cache_bytes(
            path,
            self.budgets.wsi_internal_cache_bytes,
        )?);
        if needs_tiled_pyramid(src.as_ref()) {
            build_pyramid_from_source(&dir, src.as_ref(), p)?;
            return Ok(Arc::new(TiledPyramid::open(&dir)?));
        }
        Ok(src)
    }

    /// Irisan yang masih memerlukan pembangunan piramida bertile (gambar
    /// biasa tanpa cache), terurut; GUI menjalankannya sebagai job saat
    /// proyek dibuka agar pembukaan pertama tidak memblokir.
    pub fn slides_needing_pyramid(&self) -> Vec<SlideId> {
        self.project
            .slides_ordered()
            .iter()
            .filter(|s| {
                if let Ok(c) = self.sources.lock() {
                    if c.contains(s.id) {
                        return false;
                    }
                }
                let Ok((path, fp)) = self.verified_source(s) else {
                    return false;
                };
                if h3d_slide::tiled_pyramid_exists(&self.pyramid_dir(s.id, fp)) {
                    return false;
                }
                if has_flat_extension(&path) {
                    return true;
                }
                // Slide berpiramida: hanya header yang dibaca. Sumber yang
                // tidak perlu piramida langsung didaftarkan ke cache agar
                // tidak dibuka dua kali.
                match h3d_slide::open_slide_with_cache_bytes(
                    &path,
                    self.budgets.wsi_internal_cache_bytes,
                ) {
                    Ok(src) => {
                        let src: Arc<dyn SlideSource> = Arc::from(src);
                        if needs_tiled_pyramid(src.as_ref()) {
                            true
                        } else {
                            if let Ok(mut c) = self.sources.lock() {
                                c.insert(s.id, src);
                            }
                            false
                        }
                    }
                    Err(_) => false,
                }
            })
            .map(|s| s.id)
            .collect()
    }

    /// Bangun piramida bertile irisan bila belum ada (dengan progres dan
    /// pembatalan), lalu daftarkan sumbernya ke cache.
    pub fn prepare_pyramid(&self, slide: SlideId, p: &dyn Progress) -> Result<(), EngineError> {
        let entry = self.slide(slide)?;
        let (path, fp) = self.verified_source(entry)?;
        let src = self.open_source_uncached(slide, &path, fp, p)?;
        if let Ok(mut c) = self.sources.lock() {
            c.insert(slide, src);
        }
        Ok(())
    }

    /// Thumbnail irisan dengan sisi terpanjang ≤ `max_side`, ber-cache di
    /// `cache/thumbs/`.
    pub fn thumbnail(&self, slide: SlideId, max_side: u32) -> Result<RgbaImage, EngineError> {
        self.thumbnail_task_with(slide, max_side, false, true)?
            .run()
    }

    /// Thumbnail irisan untuk tampilan 3D: seperti [`Session::thumbnail`],
    /// tetapi kanal alfa diisi masker jaringan (histo3d.md 8.3; 0 di
    /// kaca/latar putih, tepi dilunakkan [`h3d_imgproc::FEATHER_PX`] piksel).
    /// Citra sumber dan cache thumbnail biasa tidak diubah. Irisan tanpa mpp
    /// sah dikembalikan tanpa masker (mpp tidak pernah ditebak).
    /// Ber-cache di `cache/thumbs/*-tissue.png`.
    pub fn thumbnail_tissue(
        &self,
        slide: SlideId,
        max_side: u32,
    ) -> Result<RgbaImage, EngineError> {
        self.thumbnail_task_with(slide, max_side, true, true)?.run()
    }

    /// Thumbnail dari cache disk saja (tanpa membaca slide); `None` bila
    /// belum dibuat. Murah: dipanggil GUI tiap frame.
    pub fn thumbnail_cached(
        &self,
        slide: SlideId,
        max_side: u32,
        tissue: bool,
    ) -> Result<Option<RgbaImage>, EngineError> {
        let entry = self.slide(slide)?;
        let tissue = tissue && entry.has_valid_mpp();
        let fp = self.source_fingerprint(entry)?;
        let path = self.thumb_path(slide, fp, max_side, tissue);
        Ok(image::open(&path).ok().map(|i| i.to_rgba8()))
    }

    fn thumb_path(&self, slide: SlideId, fp: u64, max_side: u32, tissue: bool) -> PathBuf {
        // "-tissue2": masker v2 (rongga bermulut sempit dipertahankan); cache
        // masker lama dengan akhiran "-tissue" diabaikan dan dibangun ulang.
        let suffix = if tissue { "-tissue2" } else { "" };
        self.bundle.cache_thumbs_dir().join(format!(
            "{}-{:016x}-{}{suffix}.png",
            slide_stem(slide),
            fp,
            max_side
        ))
    }

    /// Tugas pembuatan thumbnail yang **tidak memegang sesi**: semua yang
    /// dibutuhkan (sumber piksel bersama, path cache, mpp) disalin, sehingga
    /// GUI bisa menjalankannya di thread latar tanpa mengunci `Session`.
    ///
    /// Jalur thread UI: piramida bertile yang belum ada **tidak** dibangun di
    /// sini; hasilnya [`EngineError::PyramidPending`] dan GUI menjalankan
    /// [`Self::prepare_pyramid`] sebagai job (lihat [`Self::try_open_source`]).
    pub fn thumbnail_task(
        &self,
        slide: SlideId,
        max_side: u32,
        tissue: bool,
    ) -> Result<ThumbTask, EngineError> {
        self.thumbnail_task_with(slide, max_side, tissue, false)
    }

    /// [`Self::thumbnail_task`] dengan pilihan `build`: benar = piramida yang
    /// belum ada dibangun di sini (jalur engine/CLI); salah = kembalikan
    /// [`EngineError::PyramidPending`] (jalur thread UI).
    fn thumbnail_task_with(
        &self,
        slide: SlideId,
        max_side: u32,
        tissue: bool,
        build: bool,
    ) -> Result<ThumbTask, EngineError> {
        let entry = self.slide(slide)?;
        let tissue = tissue && entry.has_valid_mpp();
        if !entry.has_valid_mpp() && tissue {
            log::debug!("irisan {slide} tanpa mpp: thumbnail 3D tanpa masker jaringan");
        }
        let fp = self.source_fingerprint(entry)?;
        let plain_path = self.thumb_path(slide, fp, max_side, false);
        let tissue_path = tissue.then(|| self.thumb_path(slide, fp, max_side, true));
        // Cache sudah ada: sumber tidak perlu dibuka.
        let final_path = tissue_path.as_ref().unwrap_or(&plain_path);
        let source = if final_path.is_file() {
            None
        } else if build {
            Some(self.open_source(slide)?)
        } else {
            Some(self.try_open_source(slide)?)
        };
        Ok(ThumbTask {
            source,
            max_side,
            plain_path,
            tissue_path,
            mpp_l0: entry.mpp,
            size_l0: entry.size_l0,
        })
    }

    /// Baca seluruh irisan pada resolusi ≈ `target_mpp` (lihat [`read_at_mpp`]).
    pub(crate) fn read_slide_at(
        &self,
        slide: SlideId,
        target_mpp: f64,
        p: &dyn Progress,
    ) -> Result<Raster, EngineError> {
        let entry = self.slide(slide)?;
        if !entry.has_valid_mpp() {
            return Err(EngineError::MppMissing {
                file: entry.source.file_name(),
            });
        }
        let src = self.open_source(slide)?;
        read_at_mpp(src.as_ref(), entry.mpp, target_mpp, p)
    }

    /// Gambar registrasi irisan: kanal hematoksilin 0..1 + masker jaringan
    /// pada [`REG_IMAGE_MPP`] (atau resolusi slide bila lebih kasar), ruang U.
    /// Ber-cache di `cache/reg/NNNN.reg`; cache divalidasi dengan sidik jari
    /// berkas dan mpp.
    pub fn registration_image(&self, slide: SlideId) -> Result<Arc<RegImage>, EngineError> {
        self.registration_image_with_progress(slide, &h3d_core::NoProgress)
    }

    /// [`Session::registration_image`] dengan progres pembacaan slide (hanya
    /// bila cache belum ada) dan pembatalan ([`EngineError::Cancelled`]).
    pub fn registration_image_with_progress(
        &self,
        slide: SlideId,
        p: &dyn Progress,
    ) -> Result<Arc<RegImage>, EngineError> {
        let entry = self.slide(slide)?;
        let fp = self.source_fingerprint(entry)?;
        let path = self
            .bundle
            .cache_reg_dir()
            .join(format!("{}.reg", slide_stem(slide)));
        if let Ok(bytes) = std::fs::read(&path) {
            if let Some(img) = decode_reg(&bytes, fp, entry.mpp) {
                return Ok(Arc::new(img));
            }
        }
        let raster = self.read_slide_at(slide, REG_IMAGE_MPP, p)?;
        let img = RegImage::from_rgba(
            &raster.rgba,
            raster.w,
            raster.h,
            raster.mpp,
            [0.0, 0.0],
            raster.mpp,
        )?;
        if let Err(e) = write_atomic(&path, &encode_reg(&img, fp, entry.mpp)) {
            log::warn!("gagal menulis cache gambar registrasi: {e}");
        }
        Ok(Arc::new(img))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use h3d_core::NoProgress;
    use h3d_slide::FlatBackend;

    #[test]
    fn read_at_mpp_box_average_and_strips() {
        // 64×48 px pada 2 µm → sasaran 8 µm = 16×12 px, rata-rata kotak 4×4.
        let mut img = RgbaImage::new(64, 48);
        for (x, y, p) in img.enumerate_pixels_mut() {
            *p = image::Rgba([(x / 4 * 16) as u8, (y / 4 * 20) as u8, 7, 255]);
        }
        let src = FlatBackend::from_image(img).unwrap();
        let r = read_at_mpp(&src, [2.0, 2.0], 8.0, &NoProgress).unwrap();
        assert_eq!((r.w, r.h), (16, 12));
        assert_eq!(r.mpp, 8.0);
        assert_eq!(
            &r.rgba[(5 * 16 + 3) * 4..(5 * 16 + 3) * 4 + 4],
            &[48, 100, 7, 255]
        );
        // Level 0 lebih kasar dari sasaran → dipakai apa adanya.
        let r = read_at_mpp(&src, [20.0, 20.0], 8.0, &NoProgress).unwrap();
        assert_eq!((r.w, r.h, r.mpp), (64, 48, 20.0));
    }

    /// Sumber pembungkus yang mencatat tiap `read_region` (untuk memastikan
    /// pembacaan benar-benar terbagi menjadi banyak strip).
    struct Counting<'a> {
        inner: &'a FlatBackend,
        calls: std::sync::Mutex<Vec<(i64, u32)>>,
    }

    impl SlideSource for Counting<'_> {
        fn info(&self) -> &h3d_slide::SlideInfo {
            self.inner.info()
        }
        fn read_region(
            &self,
            level: u32,
            x: i64,
            y: i64,
            w: u32,
            h: u32,
        ) -> Result<RgbaImage, h3d_slide::SlideError> {
            self.calls.lock().unwrap().push((y, h));
            self.inner.read_region(level, x, y, w, h)
        }
    }

    #[test]
    fn multi_strip_read_equals_single_strip_read() {
        // Gambar 97×131 px bertekstur tak periodik; rasio penurunan bulat
        // (4×), pecahan (2,5×), dan 1× (level dipakai apa adanya).
        let (w, h) = (97u32, 131u32);
        let mut img = RgbaImage::new(w, h);
        for (x, y, p) in img.enumerate_pixels_mut() {
            *p = image::Rgba([
                ((x * 7 + y * 13) % 251) as u8,
                ((x * x + 3 * y) % 241) as u8,
                ((x ^ y) % 239) as u8,
                255,
            ]);
        }
        let flat = FlatBackend::from_image(img).unwrap();
        for target in [8.0, 5.0, 2.0] {
            let whole = read_at_mpp(&flat, [2.0, 2.0], target, &NoProgress).unwrap();
            // Satu strip bila batasnya 64 MB.
            let one = Counting {
                inner: &flat,
                calls: Default::default(),
            };
            read_at_mpp(&one, [2.0, 2.0], target, &NoProgress).unwrap();
            assert_eq!(one.calls.lock().unwrap().len(), 1, "sasaran {target}");
            for rows in [1u64, 3, 7, 40] {
                let src = Counting {
                    inner: &flat,
                    calls: Default::default(),
                };
                let strip_bytes = u64::from(w) * 4 * rows;
                let r =
                    read_at_mpp_strips(&src, [2.0, 2.0], target, strip_bytes, &NoProgress).unwrap();
                let calls = src.calls.lock().unwrap().clone();
                assert!(
                    calls.len() > 1,
                    "sasaran {target}, {rows} baris/strip: harus multi-strip"
                );
                assert_eq!(
                    (r.w, r.h, r.mpp),
                    (whole.w, whole.h, whole.mpp),
                    "sasaran {target}, {rows} baris/strip"
                );
                assert!(
                    r.rgba == whole.rgba,
                    "sasaran {target}, {rows} baris/strip: piksel berbeda dari pembacaan satu strip"
                );
                // Strip berurutan, tidak melompati baris sumber, dan tiap
                // strip tidak melebihi anggaran (kecuali satu baris keluaran
                // yang memang butuh lebih banyak baris sumber).
                let ry = (whole.mpp / 2.0).ceil() as u32;
                let mut next = 0i64;
                for (y, hh) in &calls {
                    assert!(*y <= next, "celah sebelum baris {y}");
                    assert!(*hh >= 1 && u64::from(*hh) <= rows.max(u64::from(ry) + 1));
                    next = y + i64::from(*hh);
                }
                assert_eq!(next, i64::from(h), "strip terakhir sampai baris terakhir");
            }
        }
        // Pembatalan di tengah strip.
        struct Cancel;
        impl Progress for Cancel {
            fn set(&self, _f: f32, _s: &str) {}
            fn is_cancelled(&self) -> bool {
                true
            }
        }
        assert!(matches!(
            read_at_mpp_strips(&flat, [2.0, 2.0], 8.0, 400, &Cancel),
            Err(EngineError::Cancelled)
        ));
    }

    #[test]
    fn reg_cache_roundtrip_and_validation() {
        let gray = Array2::from_shape_fn((5, 7), |(y, x)| (x + y) as f32 / 12.0);
        let mask = Array2::from_shape_fn((5, 7), |(y, x)| u8::from(x > 1 && y > 0));
        let img = RegImage::new(gray, mask, 16.0, [0.0, 0.0]).unwrap();
        let bytes = encode_reg(&img, 42, [0.5, 0.5]);
        assert_eq!(decode_reg(&bytes, 42, [0.5, 0.5]).unwrap(), img);
        assert!(decode_reg(&bytes, 43, [0.5, 0.5]).is_none());
        assert!(decode_reg(&bytes, 42, [0.25, 0.5]).is_none());
        assert!(decode_reg(&bytes[..bytes.len() - 3], 42, [0.5, 0.5]).is_none());
        assert!(decode_reg(b"H3DR", 42, [0.5, 0.5]).is_none());
    }
}
