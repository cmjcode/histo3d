//! Resampling affine mundur dengan aturan pusat piksel `+0,5` (histo3d.md 5.2,
//! 8.3).
//!
//! **Arah transformasi.** Semua fungsi `warp_affine_*` menerima `t_inv` yang
//! memetakan **koordinat piksel tujuan → koordinat piksel sumber** (pemetaan
//! mundur). Untuk tiap piksel tujuan `(i, j)` dihitung pusatnya
//! `(i + 0,5, j + 0,5)`, dipetakan dengan `t_inv`, lalu nilai sumber diambil
//! di posisi itu. Bila transformasi maju `T` (sumber → tujuan) yang tersedia,
//! berikan `T.inverse()`.
//!
//! Konvensi sumber: pusat piksel sumber `(x, y)` berada di `(x + 0,5, y + 0,5)`;
//! `Nearest` mengambil `floor(p)`, `Bilinear` menginterpolasi empat pusat
//! piksel di sekitar `p − 0,5`. Posisi di luar sumber mendapat `fill`.
//! `out_size = [lebar, tinggi]`; hasil `Array2` berbentuk `(tinggi, lebar)`.

use h3d_core::Affine2;
use ndarray::Array2;
use rayon::prelude::*;

use crate::error::ImgError;

/// Metode interpolasi.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interp {
    /// Tetangga terdekat (untuk label).
    Nearest,
    /// Bilinear (untuk citra kontinu).
    Bilinear,
}

fn check_out_size(out_size: [usize; 2]) -> Result<(), ImgError> {
    if out_size[0] == 0 || out_size[1] == 0 {
        return Err(ImgError::EmptyInput);
    }
    out_size[0]
        .checked_mul(out_size[1])
        .map(|_| ())
        .ok_or_else(|| ImgError::InvalidParameter("ukuran keluaran meluap".to_string()))
}

/// Bobot bilinear: `(x0, y0, fx, fy)` dengan `x0 = floor(sx − 0,5)` dst.
#[inline]
fn bilinear_setup(sx: f64, sy: f64) -> (i64, i64, f64, f64) {
    let gx = sx - 0.5;
    let gy = sy - 0.5;
    let x0 = gx.floor();
    let y0 = gy.floor();
    (x0 as i64, y0 as i64, gx - x0, gy - y0)
}

/// Sampling generik `f32` dari pengambil `get(x, y) -> Option<f32>`.
#[inline]
fn sample_f32(
    sx: f64,
    sy: f64,
    interp: Interp,
    w: usize,
    h: usize,
    get: impl Fn(usize, usize) -> f32,
    fill: f32,
) -> f32 {
    if !(sx.is_finite() && sy.is_finite()) {
        return fill;
    }
    match interp {
        Interp::Nearest => {
            let x = sx.floor();
            let y = sy.floor();
            if x < 0.0 || y < 0.0 || x >= w as f64 || y >= h as f64 {
                fill
            } else {
                get(x as usize, y as usize)
            }
        }
        Interp::Bilinear => {
            let (x0, y0, fx, fy) = bilinear_setup(sx, sy);
            let fetch = |x: i64, y: i64| -> f64 {
                if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
                    fill as f64
                } else {
                    get(x as usize, y as usize) as f64
                }
            };
            // Seluruhnya di luar → fill.
            if x0 + 1 < 0 || y0 + 1 < 0 || x0 >= w as i64 || y0 >= h as i64 {
                return fill;
            }
            let v00 = fetch(x0, y0);
            let v10 = fetch(x0 + 1, y0);
            let v01 = fetch(x0, y0 + 1);
            let v11 = fetch(x0 + 1, y0 + 1);
            let top = v00 + (v10 - v00) * fx;
            let bot = v01 + (v11 - v01) * fx;
            (top + (bot - top) * fy) as f32
        }
    }
}

/// Resampling `f32`. Lihat dokumentasi modul untuk arah `t_inv`.
pub fn warp_affine_f32(
    src: &Array2<f32>,
    t_inv: &Affine2,
    out_size: [usize; 2],
    interp: Interp,
    fill: f32,
) -> Result<Array2<f32>, ImgError> {
    check_out_size(out_size)?;
    let (h, w) = src.dim();
    let [ow, oh] = out_size;
    let mut out = vec![fill; ow * oh];
    if h > 0 && w > 0 {
        out.par_chunks_mut(ow).enumerate().for_each(|(j, row)| {
            for (i, o) in row.iter_mut().enumerate() {
                let [sx, sy] = t_inv.apply([i as f64 + 0.5, j as f64 + 0.5]);
                *o = sample_f32(sx, sy, interp, w, h, |x, y| src[[y, x]], fill);
            }
        });
    }
    Array2::from_shape_vec((oh, ow), out).map_err(|_| ImgError::EmptyInput)
}

/// Resampling `u8` (label atau citra abu-abu). Bilinear dibulatkan ke terdekat.
pub fn warp_affine_u8(
    src: &Array2<u8>,
    t_inv: &Affine2,
    out_size: [usize; 2],
    interp: Interp,
    fill: u8,
) -> Result<Array2<u8>, ImgError> {
    check_out_size(out_size)?;
    let (h, w) = src.dim();
    let [ow, oh] = out_size;
    let mut out = vec![fill; ow * oh];
    if h > 0 && w > 0 {
        out.par_chunks_mut(ow).enumerate().for_each(|(j, row)| {
            for (i, o) in row.iter_mut().enumerate() {
                let [sx, sy] = t_inv.apply([i as f64 + 0.5, j as f64 + 0.5]);
                let v = sample_f32(sx, sy, interp, w, h, |x, y| src[[y, x]] as f32, fill as f32);
                *o = v.round().clamp(0.0, 255.0) as u8;
            }
        });
    }
    Array2::from_shape_vec((oh, ow), out).map_err(|_| ImgError::EmptyInput)
}

/// Resampling RGBA (buffer row-major 4 byte/piksel). Mengembalikan buffer
/// berukuran `out_size[0] × out_size[1] × 4`.
pub fn warp_affine_rgba(
    src: &[u8],
    w: usize,
    h: usize,
    t_inv: &Affine2,
    out_size: [usize; 2],
    interp: Interp,
    fill: [u8; 4],
) -> Result<Vec<u8>, ImgError> {
    check_out_size(out_size)?;
    if w == 0 || h == 0 {
        return Err(ImgError::EmptyInput);
    }
    if src.len() != w * h * 4 {
        return Err(ImgError::shape(&[w * h * 4], &[src.len()]));
    }
    let [ow, oh] = out_size;
    let mut out = vec![0u8; ow * oh * 4];
    out.par_chunks_mut(ow * 4).enumerate().for_each(|(j, row)| {
        for i in 0..ow {
            let [sx, sy] = t_inv.apply([i as f64 + 0.5, j as f64 + 0.5]);
            for c in 0..4 {
                let v = sample_f32(
                    sx,
                    sy,
                    interp,
                    w,
                    h,
                    |x, y| src[(y * w + x) * 4 + c] as f32,
                    fill[c] as f32,
                );
                row[i * 4 + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    });
    Ok(out)
}

/// Ubah ukuran `src` ke `out_size = [lebar, tinggi]` dengan pemetaan mundur
/// berskala (pusat piksel `+0,5`).
pub fn resize_f32(
    src: &Array2<f32>,
    out_size: [usize; 2],
    interp: Interp,
) -> Result<Array2<f32>, ImgError> {
    check_out_size(out_size)?;
    let (h, w) = src.dim();
    if h == 0 || w == 0 {
        return Err(ImgError::EmptyInput);
    }
    let t_inv = Affine2::scale(w as f64 / out_size[0] as f64, h as f64 / out_size[1] as f64);
    warp_affine_f32(src, &t_inv, out_size, interp, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    fn ramp(w: usize, h: usize) -> Array2<f32> {
        Array2::from_shape_fn((h, w), |(y, x)| (y * w + x) as f32)
    }

    #[test]
    fn identity_returns_input() {
        let src = ramp(7, 5);
        for interp in [Interp::Nearest, Interp::Bilinear] {
            let out = warp_affine_f32(&src, &Affine2::IDENTITY, [7, 5], interp, -1.0).unwrap();
            assert_eq!(out, src);
        }
        let src8 = src.mapv(|v| v as u8);
        let out8 = warp_affine_u8(&src8, &Affine2::IDENTITY, [7, 5], Interp::Bilinear, 0).unwrap();
        assert_eq!(out8, src8);
    }

    #[test]
    fn integer_translation_shifts_exactly() {
        let src = ramp(6, 6);
        // Maju: tujuan = sumber + (2, 1) ⇒ t_inv = translasi (−2, −1).
        let t_inv = Affine2::translation(-2.0, -1.0);
        for interp in [Interp::Nearest, Interp::Bilinear] {
            let out = warp_affine_f32(&src, &t_inv, [6, 6], interp, -1.0).unwrap();
            for y in 0..6 {
                for x in 0..6 {
                    let want = if x >= 2 && y >= 1 {
                        src[[y - 1, x - 2]]
                    } else {
                        -1.0
                    };
                    assert_eq!(out[[y, x]], want, "({x},{y}) {interp:?}");
                }
            }
        }
    }

    #[test]
    fn pixel_center_rotation_convention() {
        // Rotasi 90° di sekitar pusat citra persegi n×n: piksel (0,0)
        // berpindah ke (n−1, 0).
        let n = 8usize;
        let src = ramp(n, n);
        let c = n as f64 / 2.0;
        let fwd = Affine2::rotation_about(FRAC_PI_2, [c, c]);
        let t_inv = fwd.inverse().unwrap();
        for interp in [Interp::Nearest, Interp::Bilinear] {
            let out = warp_affine_f32(&src, &t_inv, [n, n], interp, -1.0).unwrap();
            assert!((out[[0, n - 1]] - src[[0, 0]]).abs() < 1e-4, "{interp:?}");
            // Tidak ada piksel fill: rotasi 90° memetakan kisi ke kisi.
            assert!(out.iter().all(|v| *v >= 0.0), "{interp:?}");
            // Pemetaan umum: out[j][i] = src[y][x] dengan (x,y) = rot⁻¹(i,j).
            for j in 0..n {
                for i in 0..n {
                    let [sx, sy] = t_inv.apply([i as f64 + 0.5, j as f64 + 0.5]);
                    let want = src[[sy.floor() as usize, sx.floor() as usize]];
                    assert!((out[[j, i]] - want).abs() < 1e-4);
                }
            }
        }
    }

    #[test]
    fn bilinear_half_pixel_averages() {
        let src = Array2::from_shape_vec((1, 2), vec![0.0f32, 10.0]).unwrap();
        // Geser setengah piksel: tujuan (0) ← sumber x = 1,0 (antara pusat 0,5 dan 1,5).
        let out = warp_affine_f32(
            &src,
            &Affine2::translation(0.5, 0.0),
            [1, 1],
            Interp::Bilinear,
            0.0,
        )
        .unwrap();
        assert!((out[[0, 0]] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn rgba_and_resize() {
        let (w, h) = (4usize, 2usize);
        let src: Vec<u8> = (0..w * h * 4).map(|i| (i % 251) as u8).collect();
        let out = warp_affine_rgba(
            &src,
            w,
            h,
            &Affine2::IDENTITY,
            [w, h],
            Interp::Nearest,
            [0; 4],
        )
        .unwrap();
        assert_eq!(out, src);
        let r = resize_f32(&ramp(4, 4), [2, 2], Interp::Bilinear).unwrap();
        assert_eq!(r.dim(), (2, 2));
        // Blok 2×2 pertama: (0,1,4,5) → 2,5.
        assert!((r[[0, 0]] - 2.5).abs() < 1e-6);
        assert!(matches!(
            resize_f32(&ramp(4, 4), [0, 2], Interp::Bilinear),
            Err(ImgError::EmptyInput)
        ));
    }
}
