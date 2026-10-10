//! `TiffBackend`: TIFF piramida (bertile atau berstrip) lewat crate `tiff`.
//!
//! Semua IFD dibaca sebagai kandidat level, diurutkan mengecil; IFD yang
//! rasio aspeknya berbeda dari level 0 (label, makro) dibuang. mpp diambil
//! dari `ImageDescription` Aperio (`MPP = x`) atau dari
//! `XResolution`/`YResolution` + `ResolutionUnit`; selain itu `None`.

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use image::RgbaImage;
use tiff::decoder::{ChunkType, Decoder, DecodingResult};
use tiff::tags::{ResolutionUnit, Tag};
use tiff::ColorType;

use crate::error::SlideError;
use crate::source::{compose_on_white, downsample_of, LevelInfo, SlideInfo, SlideSource};

/// Rentang mpp yang dianggap masuk akal untuk metadata resolusi TIFF.
/// Di luar rentang ini (mis. 72 dpi bawaan editor gambar) metadata dianggap
/// bukan skala fisik dan mpp dibiarkan `None` — ini penyaringan, bukan
/// tebakan.
pub const PLAUSIBLE_MPP_RANGE: (f64, f64) = (0.01, 100.0);

/// Toleransi relatif rasio aspek untuk menerima IFD sebagai level piramida.
const ASPECT_TOLERANCE: f64 = 0.02;

/// Ukuran tile yang dilaporkan bila IFD level 0 berstrip.
const DEFAULT_TILE_SIZE: u32 = 256;

#[derive(Clone, Copy, Debug)]
struct TiffLevel {
    /// Indeks IFD dalam berkas.
    ifd: usize,
    size: [u32; 2],
    chunk: ChunkType,
    /// Ukuran chunk nominal (tile, atau `lebar × rows_per_strip`).
    chunk_w: u32,
    chunk_h: u32,
    color: ColorType,
}

/// Sumber slide TIFF.
pub struct TiffBackend {
    path: PathBuf,
    info: SlideInfo,
    levels: Vec<TiffLevel>,
    decoder: Mutex<Decoder<BufReader<File>>>,
}

impl TiffBackend {
    /// Buka berkas TIFF.
    pub fn open(path: &Path) -> Result<Self, SlideError> {
        let file = File::open(path).map_err(|e| SlideError::io(path, e))?;
        let mut dec = Decoder::new(BufReader::new(file))?;

        let mut candidates = Vec::new();
        let mut ifd = 0usize;
        loop {
            match Self::describe_ifd(&mut dec, ifd) {
                Ok(Some(l)) => candidates.push(l),
                Ok(None) => {}
                Err(e) if ifd == 0 => return Err(e),
                Err(e) => {
                    log::warn!("{}: IFD {ifd} dilewati: {e}", path.display());
                }
            }
            if !dec.more_images() {
                break;
            }
            if let Err(e) = dec.next_image() {
                log::warn!("{}: rantai IFD terputus setelah {ifd}: {e}", path.display());
                break;
            }
            ifd += 1;
        }

        let levels = select_pyramid(candidates);
        let Some(level0) = levels.first().copied() else {
            return Err(SlideError::backend(
                "tiff",
                "tidak ada IFD dengan tipe piksel yang didukung",
            ));
        };

        // Metadata dari IFD level 0.
        dec.seek_to_image(level0.ifd)?;
        let description = dec
            .find_tag(Tag::ImageDescription)
            .ok()
            .flatten()
            .and_then(|v| v.into_string().ok());
        let mpp = description
            .as_deref()
            .and_then(parse_aperio_mpp)
            .map(|m| [m, m])
            .or_else(|| read_resolution_mpp(&mut dec));
        let vendor = description.as_deref().and_then(detect_vendor);

        let tile_size = match level0.chunk {
            ChunkType::Tile => level0.chunk_w,
            ChunkType::Strip => DEFAULT_TILE_SIZE,
        };
        let l0 = level0.size;
        let infos = levels
            .iter()
            .map(|l| LevelInfo {
                size: l.size,
                downsample: downsample_of(l0, l.size),
            })
            .collect();

        Ok(Self {
            path: path.to_path_buf(),
            info: SlideInfo {
                levels: infos,
                mpp,
                tile_size,
                vendor,
            },
            levels,
            decoder: Mutex::new(dec),
        })
    }

    /// Path berkas sumber.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Baca geometri IFD yang sedang aktif. `Ok(None)` bila tipe piksel atau
    /// tata letaknya tidak didukung.
    fn describe_ifd(
        dec: &mut Decoder<BufReader<File>>,
        ifd: usize,
    ) -> Result<Option<TiffLevel>, SlideError> {
        let (w, h) = dec.dimensions()?;
        if w == 0 || h == 0 {
            return Ok(None);
        }
        let color = dec.colortype()?;
        if !color_supported(color) {
            log::debug!("IFD {ifd}: tipe warna {color:?} tidak didukung");
            return Ok(None);
        }
        let planar: u16 = dec
            .find_tag_unsigned(Tag::PlanarConfiguration)?
            .unwrap_or(1);
        if planar != 1 {
            log::debug!("IFD {ifd}: konfigurasi planar tidak didukung");
            return Ok(None);
        }
        let chunk = dec.get_chunk_type();
        let (chunk_w, chunk_h) = match chunk {
            ChunkType::Tile => {
                let tw: u32 = dec
                    .get_tag_unsigned(Tag::TileWidth)
                    .map_err(|e| SlideError::Decode(format!("TileWidth: {e}")))?;
                let th: u32 = dec
                    .get_tag_unsigned(Tag::TileLength)
                    .map_err(|e| SlideError::Decode(format!("TileLength: {e}")))?;
                if tw == 0 || th == 0 {
                    return Err(SlideError::Decode("ukuran tile nol".into()));
                }
                (tw, th)
            }
            ChunkType::Strip => {
                let rps: u32 = dec
                    .find_tag_unsigned(Tag::RowsPerStrip)
                    .ok()
                    .flatten()
                    .unwrap_or(h);
                (w, rps.clamp(1, h))
            }
        };
        Ok(Some(TiffLevel {
            ifd,
            size: [w, h],
            chunk,
            chunk_w,
            chunk_h,
            color,
        }))
    }
}

impl SlideSource for TiffBackend {
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
        let lvl = *self
            .levels
            .get(level as usize)
            .ok_or(SlideError::LevelOutOfRange {
                level,
                count: self.levels.len() as u32,
            })?;
        compose_on_white(lvl.size, x, y, w, h, |s| {
            let mut dec = self
                .decoder
                .lock()
                .map_err(|_| SlideError::backend("tiff", "mutex dekoder rusak"))?;
            dec.seek_to_image(lvl.ifd)?;

            let [iw, ih] = lvl.size;
            let chunks_across = iw.div_ceil(lvl.chunk_w);
            let cx0 = s.x / lvl.chunk_w;
            let cx1 = (s.x + s.w - 1) / lvl.chunk_w;
            let cy0 = s.y / lvl.chunk_h;
            let cy1 = (s.y + s.h - 1) / lvl.chunk_h;

            let mut out = RgbaImage::new(s.w, s.h);
            for cy in cy0..=cy1 {
                for cx in cx0..=cx1 {
                    let idx = cy * chunks_across + cx;
                    // Ukuran data chunk (tanpa padding tepi).
                    let dw = lvl.chunk_w.min(iw - cx * lvl.chunk_w);
                    let dh = lvl.chunk_h.min(ih - cy * lvl.chunk_h);
                    let data = dec.read_chunk(idx)?;
                    let rgba = to_rgba(data, lvl.color, dw, dh)?;
                    blit_chunk(&rgba, (cx * lvl.chunk_w, cy * lvl.chunk_h), &mut out, s);
                }
            }
            Ok(out)
        })
    }
}

/// Salin bagian `chunk` (berasal di `origin` koordinat gambar) yang beririsan
/// dengan irisan `s` ke `out` (yang berasal di `(s.x, s.y)`).
fn blit_chunk(
    chunk: &RgbaImage,
    origin: (u32, u32),
    out: &mut RgbaImage,
    s: crate::source::Intersection,
) {
    let x0 = origin.0.max(s.x);
    let y0 = origin.1.max(s.y);
    let x1 = (origin.0 + chunk.width()).min(s.x + s.w);
    let y1 = (origin.1 + chunk.height()).min(s.y + s.h);
    if x1 <= x0 || y1 <= y0 {
        return;
    }
    let row_bytes = ((x1 - x0) * 4) as usize;
    for yy in y0..y1 {
        let src_start = (((yy - origin.1) * chunk.width() + (x0 - origin.0)) * 4) as usize;
        let dst_start = (((yy - s.y) * out.width() + (x0 - s.x)) * 4) as usize;
        out.as_mut()[dst_start..dst_start + row_bytes]
            .copy_from_slice(&chunk.as_raw()[src_start..src_start + row_bytes]);
    }
}

/// Tipe warna yang bisa diubah ke RGBA8.
fn color_supported(c: ColorType) -> bool {
    matches!(
        c,
        ColorType::Gray(8)
            | ColorType::Gray(16)
            | ColorType::RGB(8)
            | ColorType::RGB(16)
            | ColorType::RGBA(8)
            | ColorType::RGBA(16)
            | ColorType::YCbCr(8)
            | ColorType::CMYK(8)
    )
}

/// Ubah hasil dekode satu chunk `w × h` ke RGBA8.
fn to_rgba(
    data: DecodingResult,
    color: ColorType,
    w: u32,
    h: u32,
) -> Result<RgbaImage, SlideError> {
    let n = (w as usize) * (h as usize);
    let samples: Vec<u8> = match data {
        DecodingResult::U8(v) => v,
        DecodingResult::U16(v) => v.iter().map(|s| (s >> 8) as u8).collect(),
        other => {
            return Err(SlideError::Decode(format!(
                "tipe sampel {} tidak didukung",
                decoding_result_name(&other)
            )))
        }
    };
    let channels = match color {
        ColorType::Gray(_) => 1,
        ColorType::RGB(_) | ColorType::YCbCr(_) => 3,
        ColorType::RGBA(_) | ColorType::CMYK(_) => 4,
        other => {
            return Err(SlideError::Decode(format!(
                "tipe warna {other:?} tidak didukung"
            )))
        }
    };
    if samples.len() != n * channels {
        return Err(SlideError::Decode(format!(
            "chunk berisi {} sampel, diharapkan {}",
            samples.len(),
            n * channels
        )));
    }
    let mut out = Vec::with_capacity(n * 4);
    match color {
        ColorType::Gray(_) => {
            for g in samples {
                out.extend_from_slice(&[g, g, g, 255]);
            }
        }
        ColorType::RGB(_) | ColorType::YCbCr(_) => {
            for p in samples.chunks_exact(3) {
                out.extend_from_slice(&[p[0], p[1], p[2], 255]);
            }
        }
        ColorType::RGBA(_) => out = samples,
        ColorType::CMYK(_) => {
            for p in samples.chunks_exact(4) {
                let k = 255 - u32::from(p[3]);
                let f = |c: u8| ((255 - u32::from(c)) * k / 255) as u8;
                out.extend_from_slice(&[f(p[0]), f(p[1]), f(p[2]), 255]);
            }
        }
        _ => {}
    }
    RgbaImage::from_raw(w, h, out)
        .ok_or_else(|| SlideError::Decode("ukuran buffer chunk tidak cocok".into()))
}

fn decoding_result_name(r: &DecodingResult) -> &'static str {
    match r {
        DecodingResult::U8(_) => "u8",
        DecodingResult::U16(_) => "u16",
        DecodingResult::U32(_) => "u32",
        DecodingResult::U64(_) => "u64",
        DecodingResult::F16(_) => "f16",
        DecodingResult::F32(_) => "f32",
        DecodingResult::F64(_) => "f64",
        DecodingResult::I8(_) => "i8",
        DecodingResult::I16(_) => "i16",
        DecodingResult::I32(_) => "i32",
        DecodingResult::I64(_) => "i64",
    }
}

/// Pilih level piramida: urut mengecil menurut lebar, level 0 = terbesar,
/// level lain harus lebih kecil dari sebelumnya dan serasio aspek.
fn select_pyramid(mut candidates: Vec<TiffLevel>) -> Vec<TiffLevel> {
    candidates.sort_by(|a, b| (b.size[0], b.size[1], a.ifd).cmp(&(a.size[0], a.size[1], b.ifd)));
    let mut out: Vec<TiffLevel> = Vec::new();
    for c in candidates {
        let Some(first) = out.first() else {
            out.push(c);
            continue;
        };
        let aspect0 = f64::from(first.size[0]) / f64::from(first.size[1]);
        let aspect = f64::from(c.size[0]) / f64::from(c.size[1]);
        if ((aspect - aspect0) / aspect0).abs() > ASPECT_TOLERANCE {
            log::debug!("IFD {} dibuang: rasio aspek berbeda (label/makro?)", c.ifd);
            continue;
        }
        let prev = out.last().map(|l| l.size).unwrap_or(first.size);
        if c.size[0] >= prev[0] || c.size[1] >= prev[1] {
            log::debug!("IFD {} dibuang: ukuran tidak mengecil", c.ifd);
            continue;
        }
        out.push(c);
    }
    out
}

/// Ambil `MPP = x` dari `ImageDescription` Aperio, mis.
/// `"Aperio Image Library v10.0.51|AppMag = 20|MPP = 0.4990|..."`.
/// `None` bila tidak ada, tidak bisa diparse, atau tidak positif/terhingga.
pub fn parse_aperio_mpp(description: &str) -> Option<f64> {
    for field in description.split('|') {
        let field = field.trim();
        let Some(rest) = field.strip_prefix("MPP") else {
            continue;
        };
        let rest = rest.trim_start();
        let Some(value) = rest.strip_prefix('=') else {
            continue;
        };
        let v: f64 = value.trim().parse().ok()?;
        if v.is_finite() && v > 0.0 {
            return Some(v);
        }
        return None;
    }
    None
}

/// Hitung mpp dari resolusi TIFF (`piksel per unit`). `None` bila unit tidak
/// berdimensi, nilai tidak positif, atau hasil di luar
/// [`PLAUSIBLE_MPP_RANGE`].
pub fn mpp_from_resolution(xres: f64, yres: f64, unit: ResolutionUnit) -> Option<[f64; 2]> {
    let um_per_unit = match unit {
        ResolutionUnit::Inch => 25_400.0,
        ResolutionUnit::Centimeter => 10_000.0,
        _ => return None,
    };
    if !(xres.is_finite() && yres.is_finite()) || xres <= 0.0 || yres <= 0.0 {
        return None;
    }
    let mpp = [um_per_unit / xres, um_per_unit / yres];
    let (lo, hi) = PLAUSIBLE_MPP_RANGE;
    if mpp.iter().all(|m| (lo..=hi).contains(m)) {
        Some(mpp)
    } else {
        log::warn!(
            "resolusi TIFF menghasilkan mpp {:.3}/{:.3} di luar rentang wajar; diabaikan",
            mpp[0],
            mpp[1]
        );
        None
    }
}

fn read_resolution_mpp(dec: &mut Decoder<BufReader<File>>) -> Option<[f64; 2]> {
    let unit_raw: u16 = dec
        .find_tag_unsigned(Tag::ResolutionUnit)
        .ok()
        .flatten()
        .unwrap_or(2);
    let unit = ResolutionUnit::from_u16(unit_raw)?;
    let xres = value_to_f64(dec.find_tag(Tag::XResolution).ok()??)?;
    let yres = value_to_f64(dec.find_tag(Tag::YResolution).ok()??)?;
    mpp_from_resolution(xres, yres, unit)
}

/// Nilai tag numerik apa pun (RATIONAL, DOUBLE, LONG, …) ke `f64`; `None`
/// bila bukan angka atau penyebut nol.
#[allow(deprecated)]
fn value_to_f64(v: tiff::decoder::ifd::Value) -> Option<f64> {
    use tiff::decoder::ifd::Value;
    match v {
        Value::Rational(n, d) => (d != 0).then(|| f64::from(n) / f64::from(d)),
        Value::SRational(n, d) => (d != 0).then(|| f64::from(n) / f64::from(d)),
        Value::RationalBig(n, d) => (d != 0).then(|| n as f64 / d as f64),
        Value::SRationalBig(n, d) => (d != 0).then(|| n as f64 / d as f64),
        Value::Float(f) => Some(f64::from(f)),
        Value::Double(f) => Some(f),
        Value::Byte(n) => Some(f64::from(n)),
        Value::Short(n) => Some(f64::from(n)),
        Value::Unsigned(n) => Some(f64::from(n)),
        Value::UnsignedBig(n) => Some(n as f64),
        Value::Signed(n) => Some(f64::from(n)),
        Value::SignedBig(n) => Some(n as f64),
        Value::List(mut items) if items.len() == 1 => value_to_f64(items.remove(0)),
        _ => None,
    }
}

/// Vendor dari `ImageDescription`.
fn detect_vendor(description: &str) -> Option<String> {
    let d = description.trim_start();
    if d.starts_with("Aperio") {
        Some("Aperio".to_string())
    } else if d.contains("Hamamatsu") {
        Some("Hamamatsu".to_string())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_aperio_mpp_variants() {
        assert_eq!(
            parse_aperio_mpp("Aperio Image Library v10|AppMag = 20|MPP = 0.5|Left = 1"),
            Some(0.5)
        );
        assert_eq!(parse_aperio_mpp("Aperio|MPP=0.2520|"), Some(0.252));
        assert_eq!(parse_aperio_mpp("Aperio|AppMag = 40|"), None);
        assert_eq!(parse_aperio_mpp("Aperio|MPP = abc|"), None);
        assert_eq!(parse_aperio_mpp("Aperio|MPP = -1|"), None);
        assert_eq!(parse_aperio_mpp("Aperio|MPP = 0|"), None);
        assert_eq!(parse_aperio_mpp(""), None);
    }

    #[test]
    fn resolution_to_mpp() {
        let m = mpp_from_resolution(20_000.0, 20_000.0, ResolutionUnit::Centimeter).unwrap();
        assert!((m[0] - 0.5).abs() < 1e-9 && (m[1] - 0.5).abs() < 1e-9);
        let m = mpp_from_resolution(25_400.0, 50_800.0, ResolutionUnit::Inch).unwrap();
        assert!((m[0] - 1.0).abs() < 1e-9 && (m[1] - 0.5).abs() < 1e-9);
        assert_eq!(mpp_from_resolution(72.0, 72.0, ResolutionUnit::Inch), None);
        assert_eq!(mpp_from_resolution(1.0, 1.0, ResolutionUnit::None), None);
        assert_eq!(mpp_from_resolution(0.0, 1.0, ResolutionUnit::Inch), None);
    }
}
