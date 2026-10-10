//! Tipe dasar sumber slide (histo3d.md 8.2) dan alat bantu raster yang
//! dipakai semua backend.

use image::{Rgba, RgbaImage};

use crate::error::SlideError;

/// Warna latar slide: putih buram. Area di luar gambar diisi warna ini.
pub const BACKGROUND: Rgba<u8> = Rgba([255, 255, 255, 255]);

/// Satu level piramida.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LevelInfo {
    /// Ukuran level dalam piksel `[lebar, tinggi]`.
    pub size: [u32; 2],
    /// Faktor penurunan relatif level 0 (`p_l = p_0 / downsample`, 5.2).
    pub downsample: f64,
}

/// Metadata slide.
#[derive(Clone, Debug, PartialEq)]
pub struct SlideInfo {
    /// Level piramida; indeks 0 = resolusi penuh, terurut mengecil.
    pub levels: Vec<LevelInfo>,
    /// Mikrometer per piksel level 0 `[x, y]`; `None` bila metadata tidak ada.
    /// Tidak pernah ditebak (aturan keras 7).
    pub mpp: Option<[f64; 2]>,
    /// Ukuran tile asli (atau ukuran yang disarankan untuk backend tanpa tile).
    pub tile_size: u32,
    /// Vendor pemindai bila diketahui.
    pub vendor: Option<String>,
}

impl SlideInfo {
    /// Level `level`, atau error bila di luar rentang.
    pub fn level(&self, level: u32) -> Result<&LevelInfo, SlideError> {
        self.levels
            .get(level as usize)
            .ok_or(SlideError::LevelOutOfRange {
                level,
                count: self.levels.len() as u32,
            })
    }

    /// Ukuran level 0 `[lebar, tinggi]`; `[0, 0]` bila tidak ada level.
    pub fn level0_size(&self) -> [u32; 2] {
        self.levels.first().map(|l| l.size).unwrap_or([0, 0])
    }

    /// Jumlah tile `[kolom, baris]` pada `level` untuk `tile_size` slide ini.
    pub fn tile_grid(&self, level: u32) -> Result<[u32; 2], SlideError> {
        let l = self.level(level)?;
        let ts = self.tile_size.max(1);
        Ok([l.size[0].div_ceil(ts), l.size[1].div_ceil(ts)])
    }
}

/// Sumber piksel slide.
pub trait SlideSource: Send + Sync {
    /// Metadata slide.
    fn info(&self) -> &SlideInfo;

    /// Baca region. Koordinat `(x, y)` dan ukuran dalam piksel LEVEL yang
    /// diminta. Area di luar gambar diisi putih (latar slide), bukan error.
    fn read_region(
        &self,
        level: u32,
        x: i64,
        y: i64,
        w: u32,
        h: u32,
    ) -> Result<RgbaImage, SlideError>;
}

impl<T: SlideSource + ?Sized> SlideSource for Box<T> {
    fn info(&self) -> &SlideInfo {
        (**self).info()
    }

    fn read_region(
        &self,
        level: u32,
        x: i64,
        y: i64,
        w: u32,
        h: u32,
    ) -> Result<RgbaImage, SlideError> {
        (**self).read_region(level, x, y, w, h)
    }
}

impl<T: SlideSource + ?Sized> SlideSource for std::sync::Arc<T> {
    fn info(&self) -> &SlideInfo {
        (**self).info()
    }

    fn read_region(
        &self,
        level: u32,
        x: i64,
        y: i64,
        w: u32,
        h: u32,
    ) -> Result<RgbaImage, SlideError> {
        (**self).read_region(level, x, y, w, h)
    }
}

/// Irisan antara region yang diminta dan batas gambar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Intersection {
    /// Asal irisan di koordinat gambar (selalu ≥ 0).
    pub x: u32,
    pub y: u32,
    /// Ukuran irisan (> 0).
    pub w: u32,
    pub h: u32,
    /// Posisi irisan di dalam keluaran yang diminta.
    pub dst_x: u32,
    pub dst_y: u32,
}

/// Hitung irisan `[x, x+w) × [y, y+h)` dengan `[0, size.0) × [0, size.1)`.
/// `None` bila tidak beririsan.
pub(crate) fn intersect(size: [u32; 2], x: i64, y: i64, w: u32, h: u32) -> Option<Intersection> {
    let x0 = x.max(0);
    let y0 = y.max(0);
    let x1 = x.saturating_add(i64::from(w)).min(i64::from(size[0]));
    let y1 = y.saturating_add(i64::from(h)).min(i64::from(size[1]));
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some(Intersection {
        x: x0 as u32,
        y: y0 as u32,
        w: (x1 - x0) as u32,
        h: (y1 - y0) as u32,
        dst_x: (x0 - x) as u32,
        dst_y: (y0 - y) as u32,
    })
}

/// Bangun keluaran `w × h` berlatar putih; bagian yang beririsan dengan
/// gambar diisi oleh `read_inside(irisan)`, yang harus mengembalikan gambar
/// berukuran tepat `irisan.w × irisan.h`.
pub(crate) fn compose_on_white<F>(
    size: [u32; 2],
    x: i64,
    y: i64,
    w: u32,
    h: u32,
    read_inside: F,
) -> Result<RgbaImage, SlideError>
where
    F: FnOnce(Intersection) -> Result<RgbaImage, SlideError>,
{
    check_region_size(w, h)?;
    let Some(sect) = intersect(size, x, y, w, h) else {
        return Ok(RgbaImage::from_pixel(w, h, BACKGROUND));
    };
    let inner = read_inside(sect)?;
    if inner.width() != sect.w || inner.height() != sect.h {
        return Err(SlideError::Decode(format!(
            "backend mengembalikan {}x{}, diharapkan {}x{}",
            inner.width(),
            inner.height(),
            sect.w,
            sect.h
        )));
    }
    if sect.w == w && sect.h == h {
        return Ok(inner);
    }
    let mut out = RgbaImage::from_pixel(w, h, BACKGROUND);
    blit(&inner, &mut out, sect.dst_x, sect.dst_y);
    Ok(out)
}

/// Tolak region berukuran nol atau yang tidak muat di memori.
pub(crate) fn check_region_size(w: u32, h: u32) -> Result<(), SlideError> {
    if w == 0 || h == 0 {
        return Err(SlideError::InvalidArgument(
            "ukuran region harus lebih dari nol".into(),
        ));
    }
    let bytes = u64::from(w) * u64::from(h) * 4;
    if usize::try_from(bytes).is_err() {
        return Err(SlideError::InvalidArgument(format!(
            "region {w}x{h} terlalu besar"
        )));
    }
    Ok(())
}

/// Salin `src` ke `dst` pada posisi `(dx, dy)`; bagian yang melewati tepi
/// `dst` dipotong.
pub(crate) fn blit(src: &RgbaImage, dst: &mut RgbaImage, dx: u32, dy: u32) {
    let w = src.width().min(dst.width().saturating_sub(dx));
    let h = src.height().min(dst.height().saturating_sub(dy));
    for row in 0..h {
        let src_row = &src.as_raw()[(row * src.width() * 4) as usize..][..(w * 4) as usize];
        let start = ((dy + row) * dst.width() + dx) as usize * 4;
        dst.as_mut()[start..start + (w * 4) as usize].copy_from_slice(src_row);
    }
}

/// Potong `src` pada `[x, x+w) × [y, y+h)`; pemanggil menjamin batasnya sah.
pub(crate) fn crop(src: &RgbaImage, x: u32, y: u32, w: u32, h: u32) -> RgbaImage {
    let mut out = RgbaImage::new(w, h);
    for row in 0..h {
        let start = ((y + row) * src.width() + x) as usize * 4;
        let dst_start = (row * w) as usize * 4;
        out.as_mut()[dst_start..dst_start + (w * 4) as usize]
            .copy_from_slice(&src.as_raw()[start..start + (w * 4) as usize]);
    }
    out
}

/// Ukuran level berikutnya dalam piramida 2× (`ceil(prev / 2)`).
pub fn half_size(size: [u32; 2]) -> [u32; 2] {
    [size[0].div_ceil(2).max(1), size[1].div_ceil(2).max(1)]
}

/// Penurunan 2× dengan rata-rata kotak. Ukuran keluaran `ceil(prev / 2)`;
/// pada tepi ganjil kotak hanya berisi piksel yang ada (1 atau 2 piksel).
/// Deterministik: pembulatan ke terdekat.
pub fn downsample_2x(src: &RgbaImage) -> RgbaImage {
    let [ow, oh] = half_size([src.width(), src.height()]);
    let sw = src.width();
    let sh = src.height();
    let raw = src.as_raw();
    let mut out = RgbaImage::new(ow, oh);
    let dst = out.as_mut();
    for oy in 0..oh {
        let y0 = oy * 2;
        let y1 = (y0 + 1).min(sh - 1);
        for ox in 0..ow {
            let x0 = ox * 2;
            let x1 = (x0 + 1).min(sw - 1);
            let mut acc = [0u32; 4];
            let mut n = 0u32;
            for yy in y0..=y1 {
                for xx in x0..=x1 {
                    let i = ((yy * sw + xx) * 4) as usize;
                    for c in 0..4 {
                        acc[c] += u32::from(raw[i + c]);
                    }
                    n += 1;
                }
            }
            let o = ((oy * ow + ox) * 4) as usize;
            for c in 0..4 {
                dst[o + c] = ((acc[c] + n / 2) / n) as u8;
            }
        }
    }
    out
}

/// Penurunan rata-rata kotak ke ukuran `ow × oh` (harus ≤ ukuran sumber
/// dan lebih dari nol). Tiap piksel keluaran merata-ratakan kotak sumber
/// `[floor(ox·sw/ow), floor((ox+1)·sw/ow))`, minimal satu piksel.
pub fn downsample_box(src: &RgbaImage, ow: u32, oh: u32) -> Result<RgbaImage, SlideError> {
    let sw = src.width();
    let sh = src.height();
    if ow == 0 || oh == 0 || ow > sw || oh > sh {
        return Err(SlideError::InvalidArgument(format!(
            "ukuran tujuan {ow}x{oh} tidak sah untuk sumber {sw}x{sh}"
        )));
    }
    if ow == sw && oh == sh {
        return Ok(src.clone());
    }
    let bound = |o: u32, out_len: u32, src_len: u32| -> (u32, u32) {
        let a = (u64::from(o) * u64::from(src_len) / u64::from(out_len)) as u32;
        let b = ((u64::from(o) + 1) * u64::from(src_len) / u64::from(out_len)) as u32;
        (a, b.max(a + 1).min(src_len))
    };
    let raw = src.as_raw();
    let mut out = RgbaImage::new(ow, oh);
    let dst = out.as_mut();
    for oy in 0..oh {
        let (y0, y1) = bound(oy, oh, sh);
        for ox in 0..ow {
            let (x0, x1) = bound(ox, ow, sw);
            let mut acc = [0u64; 4];
            let n = u64::from(y1 - y0) * u64::from(x1 - x0);
            for yy in y0..y1 {
                let row = ((yy * sw + x0) * 4) as usize;
                for xx in 0..(x1 - x0) as usize {
                    for c in 0..4 {
                        acc[c] += u64::from(raw[row + xx * 4 + c]);
                    }
                }
            }
            let o = ((oy * ow + ox) * 4) as usize;
            for c in 0..4 {
                dst[o + c] = ((acc[c] + n / 2) / n) as u8;
            }
        }
    }
    Ok(out)
}

/// Faktor penurunan ala OpenSlide: rata-rata rasio lebar dan tinggi
/// terhadap level 0.
pub fn downsample_of(level0: [u32; 2], level: [u32; 2]) -> f64 {
    let rx = f64::from(level0[0]) / f64::from(level[0].max(1));
    let ry = f64::from(level0[1]) / f64::from(level[1].max(1));
    (rx + ry) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intersect_handles_outside_and_partial() {
        assert!(intersect([10, 10], 10, 0, 5, 5).is_none());
        assert!(intersect([10, 10], -5, -5, 5, 5).is_none());
        let s = intersect([10, 10], -2, 8, 5, 5).unwrap();
        assert_eq!((s.x, s.y, s.w, s.h, s.dst_x, s.dst_y), (0, 8, 3, 2, 2, 0));
    }

    #[test]
    fn downsample_2x_rounds_and_ceils() {
        let mut img = RgbaImage::new(3, 3);
        for (i, p) in img.pixels_mut().enumerate() {
            *p = Rgba([i as u8 * 10; 4]);
        }
        let d = downsample_2x(&img);
        assert_eq!((d.width(), d.height()), (2, 2));
        // Kotak kiri-atas: 0,10,30,40 → rata-rata 20.
        assert_eq!(d.get_pixel(0, 0)[0], 20);
        // Kotak kanan-bawah: hanya piksel (2,2) = 80.
        assert_eq!(d.get_pixel(1, 1)[0], 80);
    }
}
