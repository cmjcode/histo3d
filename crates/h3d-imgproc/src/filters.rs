//! Filter 2D: Gauss separabel, LoG, jendela Hann, minimum lokal, jumlah kotak.
//!
//! Tepi ditangani dengan replikasi (clamp indeks).

use ndarray::Array2;
use rayon::prelude::*;

/// Kernel Gauss 1D ternormalisasi (jumlah 1) dengan radius `ceil(3σ)`.
fn gauss_kernel(sigma: f64) -> Vec<f64> {
    let r = (3.0 * sigma).ceil() as i64;
    let mut k: Vec<f64> = (-r..=r)
        .map(|i| (-(i * i) as f64 / (2.0 * sigma * sigma)).exp())
        .collect();
    let s: f64 = k.iter().sum();
    for v in &mut k {
        *v /= s;
    }
    k
}

/// Turunan kedua kernel Gauss 1D: `g''(x) = g(x)·(x² − σ²)/σ⁴`, memakai
/// normalisasi yang sama dengan `gauss_kernel`, lalu dikoreksi agar jumlahnya
/// tepat nol (respons nol pada citra konstan meski kernel terpotong).
fn gauss_kernel_dd(sigma: f64) -> Vec<f64> {
    let g = gauss_kernel(sigma);
    let r = (g.len() / 2) as i64;
    let s2 = sigma * sigma;
    let mut k: Vec<f64> = g
        .iter()
        .enumerate()
        .map(|(idx, &gv)| {
            let x = (idx as i64 - r) as f64;
            gv * (x * x - s2) / (s2 * s2)
        })
        .collect();
    let mean = k.iter().sum::<f64>() / k.len() as f64;
    for v in &mut k {
        *v -= mean;
    }
    k
}

/// Konvolusi separabel: `kx` sepanjang x (dalam baris), `ky` sepanjang y.
fn separable(arr: &Array2<f32>, kx: &[f64], ky: &[f64]) -> Array2<f32> {
    let (h, w) = arr.dim();
    if h == 0 || w == 0 {
        return arr.clone();
    }
    let rx = (kx.len() / 2) as i64;
    let ry = (ky.len() / 2) as i64;
    // Lintasan x → buffer f64 (h × w).
    let mut tmp = vec![0.0f64; h * w];
    let src: Vec<f32> = arr.iter().copied().collect();
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let s = &src[y * w..(y + 1) * w];
        for (x, o) in row.iter_mut().enumerate() {
            let mut acc = 0.0f64;
            for (k, &kv) in kx.iter().enumerate() {
                let xi = (x as i64 + k as i64 - rx).clamp(0, w as i64 - 1) as usize;
                acc += kv * s[xi] as f64;
            }
            *o = acc;
        }
    });
    // Lintasan y.
    let mut out = vec![0.0f32; h * w];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            let mut acc = 0.0f64;
            for (k, &kv) in ky.iter().enumerate() {
                let yi = (y as i64 + k as i64 - ry).clamp(0, h as i64 - 1) as usize;
                acc += kv * tmp[yi * w + x];
            }
            *o = acc as f32;
        }
    });
    Array2::from_shape_vec((h, w), out).unwrap_or_else(|_| arr.clone())
}

/// Pemulusan Gauss separabel. `sigma_px ≤ 0` → salinan.
pub fn gaussian_blur_f32(arr: &Array2<f32>, sigma_px: f64) -> Array2<f32> {
    if sigma_px.is_nan() || sigma_px <= 0.0 {
        return arr.clone();
    }
    let k = gauss_kernel(sigma_px);
    separable(arr, &k, &k)
}

/// Laplacian of Gaussian ternormalisasi skala: `σ²·(G_xx + G_yy) ∗ I`.
/// Blob terang (nilai tinggi) menghasilkan respons **negatif** di pusatnya.
/// `sigma_px ≤ 0` → nol.
pub fn log_filter(arr: &Array2<f32>, sigma_px: f64) -> Array2<f32> {
    if sigma_px.is_nan() || sigma_px <= 0.0 {
        return Array2::zeros(arr.raw_dim());
    }
    let g = gauss_kernel(sigma_px);
    let gdd = gauss_kernel_dd(sigma_px);
    let gxx = separable(arr, &gdd, &g);
    let gyy = separable(arr, &g, &gdd);
    let s2 = (sigma_px * sigma_px) as f32;
    let mut out = gxx;
    for (o, &v) in out.iter_mut().zip(gyy.iter()) {
        *o = (*o + v) * s2;
    }
    out
}

/// Jendela Hann 2D (hasil kali jendela 1D simetris), bentuk `(h, w)`.
/// Sumbu berukuran 1 bernilai 1.
pub fn hann_window_2d(w: usize, h: usize) -> Array2<f32> {
    let hann = |n: usize| -> Vec<f64> {
        if n <= 1 {
            return vec![1.0; n];
        }
        (0..n)
            .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos()))
            .collect()
    };
    let wx = hann(w);
    let wy = hann(h);
    Array2::from_shape_fn((h, w), |(y, x)| (wx[x] * wy[y]) as f32)
}

/// Minimum lokal: piksel dengan nilai `< threshold` yang tidak lebih besar
/// dari semua tetangga dalam jendela persegi radius `radius_px` (dan lebih
/// kecil secara ketat dari tetangga yang mendahuluinya dalam urutan raster,
/// agar dataran rata hanya menghasilkan satu titik). Hasil `(x, y)` dalam
/// urutan raster.
pub fn local_minima(arr: &Array2<f32>, radius_px: usize, threshold: f32) -> Vec<(usize, usize)> {
    let (h, w) = arr.dim();
    let r = radius_px as i64;
    let mut out = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let v = arr[[y, x]];
            if v.is_nan() || v >= threshold {
                continue;
            }
            let mut is_min = true;
            'outer: for dy in -r..=r {
                let yy = y as i64 + dy;
                if yy < 0 || yy >= h as i64 {
                    continue;
                }
                for dx in -r..=r {
                    let xx = x as i64 + dx;
                    if xx < 0 || xx >= w as i64 || (dx == 0 && dy == 0) {
                        continue;
                    }
                    let u = arr[[yy as usize, xx as usize]];
                    let before = dy < 0 || (dy == 0 && dx < 0);
                    if u < v || (before && u == v) {
                        is_min = false;
                        break 'outer;
                    }
                }
            }
            if is_min {
                out.push((x, y));
            }
        }
    }
    out
}

/// Jumlah nilai dalam jendela persegi `(2r+1)²` di sekitar tiap piksel
/// (tanpa padding: jendela dipotong di tepi), lewat citra integral `f64`.
pub fn box_sum(arr: &Array2<f32>, radius_px: usize) -> Array2<f32> {
    let (h, w) = arr.dim();
    if h == 0 || w == 0 {
        return arr.clone();
    }
    // Integral (h+1) × (w+1).
    let mut integ = vec![0.0f64; (h + 1) * (w + 1)];
    for y in 0..h {
        let mut rs = 0.0f64;
        for x in 0..w {
            rs += arr[[y, x]] as f64;
            integ[(y + 1) * (w + 1) + x + 1] = integ[y * (w + 1) + x + 1] + rs;
        }
    }
    let r = radius_px;
    Array2::from_shape_fn((h, w), |(y, x)| {
        let x0 = x.saturating_sub(r);
        let y0 = y.saturating_sub(r);
        let x1 = (x + r + 1).min(w);
        let y1 = (y + r + 1).min(h);
        let s = integ[y1 * (w + 1) + x1] - integ[y0 * (w + 1) + x1] - integ[y1 * (w + 1) + x0]
            + integ[y0 * (w + 1) + x0];
        s as f32
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_preserves_constant_and_mass() {
        let c = Array2::from_elem((9, 11), 3.0f32);
        let b = gaussian_blur_f32(&c, 1.5);
        assert!(b.iter().all(|v| (v - 3.0).abs() < 1e-5));
        let mut d = Array2::<f32>::zeros((21, 21));
        d[[10, 10]] = 1.0;
        let b = gaussian_blur_f32(&d, 2.0);
        let s: f32 = b.iter().sum();
        assert!((s - 1.0).abs() < 1e-4);
        assert!(b[[10, 10]] > b[[10, 11]]);
    }

    #[test]
    fn log_negative_at_bright_blob() {
        let mut img = Array2::<f32>::zeros((41, 41));
        for y in 0..41 {
            for x in 0..41 {
                let d2 = ((x as f32 - 20.0).powi(2) + (y as f32 - 20.0).powi(2)) / (2.0 * 9.0);
                img[[y, x]] = (-d2).exp();
            }
        }
        let r = log_filter(&img, 3.0);
        assert!(r[[20, 20]] < 0.0);
        let mins = local_minima(&r, 2, 0.0);
        assert_eq!(mins, vec![(20, 20)]);
        let flat = log_filter(&Array2::from_elem((9, 9), 2.0f32), 2.0);
        assert!(flat.iter().all(|v| v.abs() < 1e-4));
    }

    #[test]
    fn hann_window_shape() {
        let wnd = hann_window_2d(5, 3);
        assert_eq!(wnd.dim(), (3, 5));
        assert!((wnd[[1, 2]] - 1.0).abs() < 1e-6);
        assert!(wnd[[0, 0]].abs() < 1e-6);
        assert_eq!(hann_window_2d(1, 1)[[0, 0]], 1.0);
    }

    #[test]
    fn box_sum_matches_naive() {
        let arr = Array2::from_shape_fn((5, 6), |(y, x)| (y * 6 + x) as f32);
        let b = box_sum(&arr, 1);
        // Pusat (2,2): jumlah 3×3.
        let mut want = 0.0;
        for y in 1..4 {
            for x in 1..4 {
                want += arr[[y, x]];
            }
        }
        assert_eq!(b[[2, 2]], want);
        // Sudut (0,0): 2×2.
        assert_eq!(
            b[[0, 0]],
            arr[[0, 0]] + arr[[0, 1]] + arr[[1, 0]] + arr[[1, 1]]
        );
    }

    #[test]
    fn local_minima_plateau_yields_one() {
        let mut arr = Array2::from_elem((5, 5), 1.0f32);
        arr[[2, 2]] = -1.0;
        arr[[2, 3]] = -1.0;
        assert_eq!(local_minima(&arr, 1, 0.0), vec![(2, 2)]);
    }
}
