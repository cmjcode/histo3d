//! Penurunan resolusi 2× (piramida): rata-rata kotak untuk citra, modus untuk
//! label, rata-rata berbobot untuk atribut (histo3d.md 8.3, 8.12.5).
//!
//! Ukuran keluaran selalu `ceil(n / 2)`; blok di tepi ganjil merata-ratakan
//! piksel yang tersedia saja.

use ndarray::Array2;
use rayon::prelude::*;

use crate::error::ImgError;

/// Ukuran keluaran untuk satu sumbu.
#[inline]
pub fn half_size(n: usize) -> usize {
    n.div_ceil(2)
}

/// Turunkan RGBA 2× dengan rata-rata kotak (pembulatan ke terdekat).
/// Mengembalikan `(buffer, w2, h2)`.
pub fn downsample2_rgba(
    rgba: &[u8],
    w: usize,
    h: usize,
) -> Result<(Vec<u8>, usize, usize), ImgError> {
    if w == 0 || h == 0 {
        return Err(ImgError::EmptyInput);
    }
    if rgba.len() != w * h * 4 {
        return Err(ImgError::shape(&[w * h * 4], &[rgba.len()]));
    }
    let w2 = half_size(w);
    let h2 = half_size(h);
    let mut out = vec![0u8; w2 * h2 * 4];
    out.par_chunks_mut(w2 * 4)
        .enumerate()
        .for_each(|(y2, orow)| {
            let y0 = 2 * y2;
            let ys = if y0 + 1 < h { 2 } else { 1 };
            for x2 in 0..w2 {
                let x0 = 2 * x2;
                let xs = if x0 + 1 < w { 2 } else { 1 };
                let n = (xs * ys) as u32;
                for c in 0..4 {
                    let mut s = 0u32;
                    for dy in 0..ys {
                        for dx in 0..xs {
                            s += rgba[((y0 + dy) * w + x0 + dx) * 4 + c] as u32;
                        }
                    }
                    orow[x2 * 4 + c] = ((s + n / 2) / n) as u8;
                }
            }
        });
    Ok((out, w2, h2))
}

/// Turunkan `Array2<f32>` 2× dengan rata-rata kotak.
pub fn downsample2_f32(src: &Array2<f32>) -> Array2<f32> {
    let (h, w) = src.dim();
    let (h2, w2) = (half_size(h), half_size(w));
    Array2::from_shape_fn((h2, w2), |(y2, x2)| {
        let y0 = 2 * y2;
        let x0 = 2 * x2;
        let mut s = 0.0f64;
        let mut n = 0.0f64;
        for y in y0..(y0 + 2).min(h) {
            for x in x0..(x0 + 2).min(w) {
                s += src[[y, x]] as f64;
                n += 1.0;
            }
        }
        (s / n) as f32
    })
}

/// Turunkan label 2× dengan **modus** blok 2×2.
///
/// Aturan seri: di antara nilai yang jumlahnya sama-sama terbanyak, bila ada
/// yang bukan nol maka dipilih **nilai bukan-nol terkecil**; nol hanya
/// terpilih bila ia satu-satunya modus. Contoh: `[0, 0, 5, 5] → 5`,
/// `[0, 3, 5, 5] → 5`, `[0, 3, 5, 7] → 3`, `[0, 0, 0, 5] → 0`.
pub fn downsample2_label_mode(src: &Array2<u8>) -> Array2<u8> {
    let (h, w) = src.dim();
    let (h2, w2) = (half_size(h), half_size(w));
    Array2::from_shape_fn((h2, w2), |(y2, x2)| {
        let y0 = 2 * y2;
        let x0 = 2 * x2;
        let mut vals = [0u8; 4];
        let mut n = 0usize;
        for y in y0..(y0 + 2).min(h) {
            for x in x0..(x0 + 2).min(w) {
                vals[n] = src[[y, x]];
                n += 1;
            }
        }
        let vals = &vals[..n];
        let mut best_val = 0u8;
        let mut best_cnt = 0usize;
        for &v in vals {
            let cnt = vals.iter().filter(|&&u| u == v).count();
            let better =
                cnt > best_cnt || (cnt == best_cnt && v != 0 && (best_val == 0 || v < best_val));
            if better {
                best_cnt = cnt;
                best_val = v;
            }
        }
        best_val
    })
}

/// Turunkan atribut 2× dengan rata-rata berbobot `weights`, mengabaikan
/// piksel bernilai `nodata` (atau NaN) dan berbobot ≤ 0. Mengembalikan
/// `(nilai, jumlah bobot)`; blok tanpa data → `(nodata, 0)`. Bentuk
/// `values` dan `weights` harus sama.
pub fn downsample2_attr_weighted(
    values: &Array2<f32>,
    weights: &Array2<f32>,
    nodata: f32,
) -> Result<(Array2<f32>, Array2<f32>), ImgError> {
    if values.raw_dim() != weights.raw_dim() {
        return Err(ImgError::shape(
            &[values.dim().0, values.dim().1],
            &[weights.dim().0, weights.dim().1],
        ));
    }
    let (h, w) = values.dim();
    let (h2, w2) = (half_size(h), half_size(w));
    let is_nodata = |v: f32| v.is_nan() || v == nodata;
    let mut out_v = Array2::<f32>::from_elem((h2, w2), nodata);
    let mut out_w = Array2::<f32>::zeros((h2, w2));
    for y2 in 0..h2 {
        for x2 in 0..w2 {
            let mut sw = 0.0f64;
            let mut swv = 0.0f64;
            for y in (2 * y2)..(2 * y2 + 2).min(h) {
                for x in (2 * x2)..(2 * x2 + 2).min(w) {
                    let v = values[[y, x]];
                    let wt = weights[[y, x]];
                    if is_nodata(v) || wt.is_nan() || wt <= 0.0 {
                        continue;
                    }
                    sw += wt as f64;
                    swv += wt as f64 * v as f64;
                }
            }
            if sw > 0.0 {
                out_v[[y2, x2]] = (swv / sw) as f32;
                out_w[[y2, x2]] = sw as f32;
            }
        }
    }
    Ok((out_v, out_w))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn sizes_are_ceil_half() {
        let (_, w2, h2) = downsample2_rgba(&[0u8; 5 * 3 * 4], 5, 3).unwrap();
        assert_eq!((w2, h2), (3, 2));
        assert_eq!(downsample2_f32(&Array2::zeros((7, 1))).dim(), (4, 1));
        assert_eq!(downsample2_label_mode(&Array2::zeros((1, 1))).dim(), (1, 1));
    }

    #[test]
    fn rgba_box_average_and_odd_edge() {
        // 3×1: piksel 10, 20 | 40 → [15, 40].
        let mut rgba = vec![0u8; 3 * 4];
        rgba[0] = 10;
        rgba[4] = 20;
        rgba[8] = 40;
        let (out, w2, h2) = downsample2_rgba(&rgba, 3, 1).unwrap();
        assert_eq!((w2, h2), (2, 1));
        assert_eq!(out[0], 15);
        assert_eq!(out[4], 40);
    }

    #[test]
    fn label_mode_rules() {
        let src = arr2(&[[0u8, 0, 0, 3, 0, 3, 0, 0], [5, 5, 5, 5, 5, 7, 0, 5]]);
        let out = downsample2_label_mode(&src);
        assert_eq!(out[[0, 0]], 5); // [0,0,5,5] → 5
        assert_eq!(out[[0, 1]], 5); // [0,3,5,5] → 5
        assert_eq!(out[[0, 2]], 3); // [0,3,5,7] → 3
        assert_eq!(out[[0, 3]], 0); // [0,0,0,5] → 0
    }

    #[test]
    fn attr_weighted_ignores_nodata() {
        let nodata = -1.0f32;
        let vals = arr2(&[[10.0f32, nodata], [nodata, nodata]]);
        let wts = arr2(&[[3.0f32, 3.0], [3.0, 3.0]]);
        let (v, w) = downsample2_attr_weighted(&vals, &wts, nodata).unwrap();
        assert_eq!(v[[0, 0]], 10.0); // bukan 5 atau 2,5
        assert_eq!(w[[0, 0]], 3.0);
        let vals2 = arr2(&[[10.0f32, 20.0], [nodata, nodata]]);
        let wts2 = arr2(&[[1.0f32, 3.0], [1.0, 1.0]]);
        let (v2, w2) = downsample2_attr_weighted(&vals2, &wts2, nodata).unwrap();
        assert!((v2[[0, 0]] - 17.5).abs() < 1e-6);
        assert_eq!(w2[[0, 0]], 4.0);
        let all = arr2(&[[nodata, nodata], [nodata, nodata]]);
        let (v3, w3) = downsample2_attr_weighted(&all, &wts, nodata).unwrap();
        assert_eq!(v3[[0, 0]], nodata);
        assert_eq!(w3[[0, 0]], 0.0);
    }

    #[test]
    fn f32_average() {
        let src = arr2(&[[1.0f32, 3.0], [5.0, 7.0]]);
        assert_eq!(downsample2_f32(&src)[[0, 0]], 4.0);
    }
}
