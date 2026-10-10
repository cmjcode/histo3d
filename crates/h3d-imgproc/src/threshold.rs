//! Ambang otomatis (Otsu).

use ndarray::Array2;

/// Ambang Otsu atas `values` memakai histogram `bins` wadah pada rentang
/// `[min, max]`. Mengembalikan tepi atas wadah yang memaksimalkan variansi
/// antar-kelas; bila maksimum berupa dataran (celah kosong di antara dua
/// modus), diambil wadah **tengah** dataran itu. NaN diabaikan. Bila masukan kosong
/// atau konstan, mengembalikan nilai minimum (atau 0 bila kosong).
pub fn otsu(values: impl Iterator<Item = f32>, bins: usize) -> f32 {
    let v: Vec<f32> = values.filter(|x| !x.is_nan()).collect();
    if v.is_empty() {
        return 0.0;
    }
    let bins = bins.max(2);
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &x in &v {
        lo = lo.min(x);
        hi = hi.max(x);
    }
    if hi <= lo {
        return lo;
    }
    let width = (hi as f64 - lo as f64) / bins as f64;
    let mut hist = vec![0u64; bins];
    for &x in &v {
        let b = (((x as f64 - lo as f64) / width) as usize).min(bins - 1);
        hist[b] += 1;
    }
    let total = v.len() as f64;
    let mut sum_all = 0.0f64;
    for (i, &c) in hist.iter().enumerate() {
        sum_all += (i as f64 + 0.5) * c as f64;
    }
    let mut w_b = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut best = f64::NEG_INFINITY;
    let mut best_lo = 0usize;
    let mut best_hi = 0usize;
    for (k, &c) in hist.iter().enumerate().take(bins - 1) {
        w_b += c as f64;
        sum_b += (k as f64 + 0.5) * c as f64;
        let w_f = total - w_b;
        if w_b == 0.0 || w_f == 0.0 {
            continue;
        }
        let m_b = sum_b / w_b;
        let m_f = (sum_all - sum_b) / w_f;
        let var = w_b * w_f * (m_b - m_f) * (m_b - m_f);
        if var > best {
            best = var;
            best_lo = k;
            best_hi = k;
        } else if var == best {
            best_hi = k;
        }
    }
    let best_k = (best_lo + best_hi) / 2;
    (lo as f64 + (best_k as f64 + 1.0) * width) as f32
}

/// Otsu hanya pada piksel dengan `mask != 0` (256 wadah). Bila bentuk tidak
/// cocok atau tidak ada piksel bermasker, mengembalikan 0.
pub fn otsu_masked(arr: &Array2<f32>, mask: &Array2<u8>) -> f32 {
    if arr.raw_dim() != mask.raw_dim() {
        return 0.0;
    }
    otsu(
        arr.iter()
            .zip(mask.iter())
            .filter(|(_, m)| **m != 0)
            .map(|(v, _)| *v),
        256,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn otsu_separates_bimodal() {
        let mut v = Vec::new();
        for i in 0..1000 {
            v.push(0.1 + 0.05 * ((i % 7) as f32 / 7.0));
            v.push(0.8 + 0.05 * ((i % 5) as f32 / 5.0));
        }
        let t = otsu(v.iter().copied(), 256);
        assert!(t > 0.2 && t < 0.75, "ambang {t}");
    }

    #[test]
    fn otsu_degenerate() {
        assert_eq!(otsu(std::iter::empty(), 256), 0.0);
        assert_eq!(otsu([2.0f32; 10].iter().copied(), 256), 2.0);
    }

    #[test]
    fn otsu_masked_uses_only_mask() {
        let arr = Array2::from_shape_fn((4, 4), |(y, _)| if y < 2 { 0.1 } else { 0.9 });
        let mask = Array2::from_shape_fn((4, 4), |(y, _)| u8::from(y >= 2));
        // Hanya kelas 0,9 → konstan → min.
        assert!((otsu_masked(&arr, &mask) - 0.9).abs() < 1e-6);
    }
}
