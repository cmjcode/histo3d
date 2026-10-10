//! Akses piksel slide: pembukaan sumber, thumbnail, pembacaan pada resolusi
//! kerja (dalam strip), dan gambar registrasi ber-cache (`cache/reg/`).

use std::path::Path;
use std::sync::Arc;

use h3d_core::{Progress, SlideEntry, SlideId};
use h3d_io::{slide_stem, write_atomic};
use h3d_register::RegImage;
use h3d_slide::{FlatBackend, SlideSource};
use image::RgbaImage;
use ndarray::Array2;

use super::Session;
use crate::error::EngineError;

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
    let src_rows_budget = (STRIP_BYTES / (u64::from(lw) * 4)).max(1) as f64;
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
        for o in oy..oy_end {
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
                let d = (o * ow + ox) * 4;
                for k in 0..3 {
                    rgba[d + k] = ((acc[k] + n / 2) / n) as u8;
                }
            }
        }
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
    for oy in 0..h_px {
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
            let d = (oy * w_px + ox) * 4;
            for k in 0..3 {
                out[d + k] = ((acc[k] + n / 2) / n) as u8;
            }
        }
    }
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
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let gray = Array2::from_shape_vec((h, w), gray).ok()?;
    let mask = Array2::from_shape_vec((h, w), mask).ok()?;
    RegImage::new(gray, mask, mpp, origin).ok()
}

fn has_flat_extension(path: &Path) -> bool {
    path.extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .is_some_and(|e| matches!(e.as_str(), "png" | "jpg" | "jpeg" | "bmp" | "gif" | "webp"))
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

    /// Sidik jari berkas slide saat ini (untuk kunci cache).
    pub(crate) fn source_fingerprint(&self, entry: &SlideEntry) -> Result<u64, EngineError> {
        let path = self.bundle.resolve_slide_path(&entry.source);
        if !path.is_file() {
            return Err(h3d_io::IoError::MissingFile(path).into());
        }
        Ok(h3d_slide::fingerprint(&path)?.1)
    }

    /// Buka sumber piksel sebuah irisan. Gambar biasa (PNG/JPEG) memakai
    /// cache piramida di `cache/pyramid/`.
    pub fn open_source(&self, slide: SlideId) -> Result<Box<dyn SlideSource>, EngineError> {
        let entry = self.slide(slide)?;
        let path = self.bundle.resolve_slide_path(&entry.source);
        if !path.is_file() {
            return Err(h3d_io::IoError::MissingFile(path).into());
        }
        if has_flat_extension(&path) {
            let fp = h3d_slide::fingerprint(&path)?.1;
            let dir =
                self.bundle
                    .cache_pyramid_dir()
                    .join(format!("{}-{:016x}", slide_stem(slide), fp));
            match FlatBackend::open_with_cache(&path, &dir) {
                Ok(b) => return Ok(Box::new(b)),
                Err(e) => log::debug!("FlatBackend gagal, mencoba backend lain: {e}"),
            }
        }
        Ok(h3d_slide::open_slide(&path)?)
    }

    /// Thumbnail irisan dengan sisi terpanjang ≤ `max_side`, ber-cache di
    /// `cache/thumbs/`.
    pub fn thumbnail(&self, slide: SlideId, max_side: u32) -> Result<RgbaImage, EngineError> {
        let entry = self.slide(slide)?;
        let fp = self.source_fingerprint(entry)?;
        let path = self.bundle.cache_thumbs_dir().join(format!(
            "{}-{:016x}-{}.png",
            slide_stem(slide),
            fp,
            max_side
        ));
        if let Ok(img) = image::open(&path) {
            return Ok(img.to_rgba8());
        }
        let src = self.open_source(slide)?;
        let img = h3d_slide::thumbnail(src.as_ref(), max_side)?;
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
            if let Err(e) = write_atomic(&path, &png) {
                log::warn!("gagal menulis cache thumbnail: {e}");
            }
        }
        Ok(img)
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
        let raster = self.read_slide_at(slide, REG_IMAGE_MPP, &h3d_core::NoProgress)?;
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
