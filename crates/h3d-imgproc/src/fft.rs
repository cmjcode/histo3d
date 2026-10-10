//! Pembungkus tipis `rustfft`: FFT 2D dan korelasi fase (histo3d.md 8.4.2 R2).

use ndarray::Array2;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::error::ImgError;

/// FFT 2D in-place pada buffer `(h, w)` row-major; `inverse` memilih arah.
fn fft2_inplace(buf: &mut [Complex<f32>], w: usize, h: usize, inverse: bool) {
    let mut planner = FftPlanner::<f32>::new();
    let fx = if inverse {
        planner.plan_fft_inverse(w)
    } else {
        planner.plan_fft_forward(w)
    };
    let fy = if inverse {
        planner.plan_fft_inverse(h)
    } else {
        planner.plan_fft_forward(h)
    };
    // Baris.
    let mut scratch = vec![Complex::new(0.0, 0.0); fx.get_inplace_scratch_len()];
    for row in buf.chunks_mut(w) {
        fx.process_with_scratch(row, &mut scratch);
    }
    // Kolom (kumpulkan → proses → sebar).
    let mut col = vec![Complex::new(0.0, 0.0); h];
    let mut scratch = vec![Complex::new(0.0, 0.0); fy.get_inplace_scratch_len()];
    for x in 0..w {
        for y in 0..h {
            col[y] = buf[y * w + x];
        }
        fy.process_with_scratch(&mut col, &mut scratch);
        for y in 0..h {
            buf[y * w + x] = col[y];
        }
    }
}

/// FFT 2D maju dari citra riil; keluaran kompleks `(h, w)` tanpa normalisasi.
pub fn fft2_real(arr: &Array2<f32>) -> Array2<Complex<f32>> {
    let (h, w) = arr.dim();
    if h == 0 || w == 0 {
        return Array2::from_elem((h, w), Complex::new(0.0, 0.0));
    }
    let mut buf: Vec<Complex<f32>> = arr.iter().map(|&v| Complex::new(v, 0.0)).collect();
    fft2_inplace(&mut buf, w, h, false);
    Array2::from_shape_vec((h, w), buf)
        .unwrap_or_else(|_| Array2::from_elem((h, w), Complex::new(0.0, 0.0)))
}

/// IFFT 2D, dinormalkan `1/(w·h)` sehingga `ifft2(fft2_real(a)) ≈ a`.
pub fn ifft2(spec: &Array2<Complex<f32>>) -> Array2<Complex<f32>> {
    let (h, w) = spec.dim();
    if h == 0 || w == 0 {
        return spec.clone();
    }
    let mut buf: Vec<Complex<f32>> = spec.iter().copied().collect();
    fft2_inplace(&mut buf, w, h, true);
    let n = (w * h) as f32;
    for v in &mut buf {
        *v /= n;
    }
    Array2::from_shape_vec((h, w), buf).unwrap_or_else(|_| spec.clone())
}

/// Offset sub-piksel puncak dari tiga sampel `(l, c, r)`.
///
/// Puncak korelasi fase berbentuk sinc periodik, sehingga dipakai penaksir
/// rasio Foroosh dkk. (2002): `δ = r/(c + r)` bila tetangga kanan lebih
/// tinggi, `δ = −l/(c + l)` bila kiri; eksak untuk sinc ideal. Bila
/// penyebut tidak sah, jatuh ke pencocokan parabola. Hasil dijepit
/// `[-0,5, 0,5]`.
fn subpixel_offset(l: f64, c: f64, r: f64) -> f64 {
    let ratio = if r >= l {
        if c + r > 1e-12 && r >= 0.0 {
            Some(r / (c + r))
        } else {
            None
        }
    } else if c + l > 1e-12 && l >= 0.0 {
        Some(-l / (c + l))
    } else {
        None
    };
    let v = ratio.unwrap_or_else(|| {
        let denom = l - 2.0 * c + r;
        if denom.abs() < 1e-12 {
            0.0
        } else {
            0.5 * (l - r) / denom
        }
    });
    v.clamp(-0.5, 0.5)
}

/// Korelasi fase `fixed` vs `moving` (bentuk sama, tidak kosong).
///
/// `R = F_f · conj(F_m) / |F_f · conj(F_m)|`; puncak `IFFT(R)` dicari
/// (seri → indeks raster terkecil), diperhalus sub-piksel (rasio sinc Foroosh),
/// lalu dibungkus ke geseran bertanda dalam `[-n/2, n/2)`.
///
/// **Konvensi:** `(dx, dy)` sedemikian sehingga **`moving` yang digeser
/// sebesar `(dx, dy)` piksel** (isi berpindah ke +x, +y) berimpit dengan
/// `fixed`: `fixed(x, y) ≈ moving(x − dx, y − dy)`. Nilai ketiga adalah
/// tinggi puncak korelasi (≈ 1 untuk pasangan identik murni geser).
pub fn phase_correlation(
    fixed: &Array2<f32>,
    moving: &Array2<f32>,
) -> Result<(f64, f64, f32), ImgError> {
    let (h, w) = fixed.dim();
    if h == 0 || w == 0 {
        return Err(ImgError::EmptyInput);
    }
    if moving.dim() != (h, w) {
        return Err(ImgError::shape(&[h, w], &[moving.dim().0, moving.dim().1]));
    }
    let ff = fft2_real(fixed);
    let fm = fft2_real(moving);
    let mut cross = Array2::from_elem((h, w), Complex::new(0.0f32, 0.0));
    for ((c, a), b) in cross.iter_mut().zip(ff.iter()).zip(fm.iter()) {
        let p = a * b.conj();
        let n = p.norm();
        *c = if n > 1e-20 {
            p / n
        } else {
            Complex::new(0.0, 0.0)
        };
    }
    let r = ifft2(&cross);
    let mut best = f32::NEG_INFINITY;
    let mut bi = 0usize;
    for (i, v) in r.iter().enumerate() {
        if v.re > best {
            best = v.re;
            bi = i;
        }
    }
    let py = bi / w;
    let px = bi % w;
    let at = |x: usize, y: usize| r[[y % h, x % w]].re as f64;
    let ox = if w >= 3 {
        subpixel_offset(at(px + w - 1, py), at(px, py), at(px + 1, py))
    } else {
        0.0
    };
    let oy = if h >= 3 {
        subpixel_offset(at(px, py + h - 1), at(px, py), at(px, py + 1))
    } else {
        0.0
    };
    let wrap = |i: usize, n: usize| -> f64 {
        if i >= n.div_ceil(2) {
            i as f64 - n as f64
        } else {
            i as f64
        }
    };
    Ok((wrap(px, w) + ox, wrap(py, h) + oy, best))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shift_wrap(src: &Array2<f32>, dx: i64, dy: i64) -> Array2<f32> {
        let (h, w) = src.dim();
        Array2::from_shape_fn((h, w), |(y, x)| {
            let sx = (x as i64 - dx).rem_euclid(w as i64) as usize;
            let sy = (y as i64 - dy).rem_euclid(h as i64) as usize;
            src[[sy, sx]]
        })
    }

    fn pattern(w: usize, h: usize) -> Array2<f32> {
        // Pola deterministik tak periodik sederhana.
        Array2::from_shape_fn((h, w), |(y, x)| {
            let a = ((x * 7 + y * 13) % 17) as f32 / 17.0;
            let b = ((x * x + 3 * y) % 11) as f32 / 11.0;
            a * 0.6 + b * 0.4
        })
    }

    #[test]
    fn roundtrip_fft() {
        let a = pattern(8, 6);
        let back = ifft2(&fft2_real(&a));
        for (x, y) in a.iter().zip(back.iter()) {
            assert!((x - y.re).abs() < 1e-4 && y.im.abs() < 1e-4);
        }
    }

    #[test]
    fn recovers_integer_shift_exactly() {
        let moving = pattern(32, 24);
        for &(dx, dy) in &[(3i64, -5i64), (-7, 2), (0, 0), (10, 11)] {
            // fixed = moving digeser (dx, dy).
            let fixed = shift_wrap(&moving, dx, dy);
            let (gx, gy, peak) = phase_correlation(&fixed, &moving).unwrap();
            assert!((gx - dx as f64).abs() < 1e-6, "dx {gx} vs {dx}");
            assert!((gy - dy as f64).abs() < 1e-6, "dy {gy} vs {dy}");
            assert!(peak > 0.9, "puncak {peak}");
            // Konvensi: moving digeser (gx, gy) berimpit dengan fixed.
            let aligned = shift_wrap(&moving, gx.round() as i64, gy.round() as i64);
            assert_eq!(aligned, fixed);
        }
    }

    #[test]
    fn subpixel_shift_within_tenth_pixel() {
        // Tekstur padat: ratusan tonjolan Gauss kecil (σ ≈ 1,2 px) pada posisi
        // kontinu di bagian dalam citra, sehingga versi tergeser sub-piksel
        // dapat dihitung eksak dan spektrumnya lebar (seperti citra histologi
        // berjendela), bukan blob tunggal yang spektrumnya nol di frekuensi
        // tinggi.
        let (w, h) = (64usize, 64usize);
        let mut seed = 12345u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let bumps: Vec<(f64, f64, f64)> = (0..400)
            .map(|_| (8.0 + 48.0 * next(), 8.0 + 48.0 * next(), 0.5 + next()))
            .collect();
        let render = |dx: f64, dy: f64| -> Array2<f32> {
            Array2::from_shape_fn((h, w), |(y, x)| {
                let px = x as f64 + 0.5;
                let py = y as f64 + 0.5;
                let mut v = 0.0;
                for &(bx, by, amp) in &bumps {
                    let d2 = (px - bx - dx).powi(2) + (py - by - dy).powi(2);
                    if d2 < 36.0 {
                        v += amp * (-d2 / (2.0 * 1.2 * 1.2)).exp();
                    }
                }
                v as f32
            })
        };
        let moving = render(0.0, 0.0);
        for &(dx, dy) in &[(2.3f64, -1.6f64), (0.4, 0.5), (-3.75, 2.25), (1.0, -2.0)] {
            let fixed = render(dx, dy);
            let (gx, gy, _) = phase_correlation(&fixed, &moving).unwrap();
            assert!((gx - dx).abs() < 0.1, "dx {gx} vs {dx}");
            assert!((gy - dy).abs() < 0.1, "dy {gy} vs {dy}");
        }
    }

    #[test]
    fn shape_mismatch_is_error() {
        let a = pattern(4, 4);
        let b = pattern(5, 4);
        assert!(matches!(
            phase_correlation(&a, &b),
            Err(ImgError::ShapeMismatch { .. })
        ));
    }
}
