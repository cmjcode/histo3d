//! Metrik kemiripan dan momen masker.

use ndarray::Array2;

use crate::error::ImgError;

fn check_same<A, B>(a: &Array2<A>, b: &Array2<B>) -> Result<(), ImgError> {
    if a.dim() != b.dim() {
        return Err(ImgError::shape(
            &[a.dim().0, a.dim().1],
            &[b.dim().0, b.dim().1],
        ));
    }
    if a.is_empty() {
        return Err(ImgError::EmptyInput);
    }
    Ok(())
}

/// Koefisien korelasi silang ternormalisasi (Pearson) pada piksel dengan
/// `mask != 0` (atau semua piksel bila `None`). Varians nol → 0.
pub fn ncc_masked(
    a: &Array2<f32>,
    b: &Array2<f32>,
    mask: Option<&Array2<u8>>,
) -> Result<f32, ImgError> {
    check_same(a, b)?;
    if let Some(m) = mask {
        check_same(a, m)?;
    }
    let mask_vals: Vec<u8> = match mask {
        Some(m) => m.iter().copied().collect(),
        None => vec![1u8; a.len()],
    };
    let mut n = 0.0f64;
    let (mut sa, mut sb, mut saa, mut sbb, mut sab) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for ((&x, &y), &mv) in a.iter().zip(b.iter()).zip(mask_vals.iter()) {
        if mv == 0 {
            continue;
        }
        let (x, y) = (x as f64, y as f64);
        n += 1.0;
        sa += x;
        sb += y;
        saa += x * x;
        sbb += y * y;
        sab += x * y;
    }
    if n < 2.0 {
        return Ok(0.0);
    }
    let cov = sab - sa * sb / n;
    let va = saa - sa * sa / n;
    let vb = sbb - sb * sb / n;
    if va <= 1e-20 || vb <= 1e-20 {
        return Ok(0.0);
    }
    Ok((cov / (va * vb).sqrt()).clamp(-1.0, 1.0) as f32)
}

/// Koefisien Dice `2|A∩B| / (|A| + |B|)`; keduanya kosong → 1.
pub fn dice(mask_a: &Array2<u8>, mask_b: &Array2<u8>) -> Result<f32, ImgError> {
    check_same(mask_a, mask_b)?;
    let mut inter = 0u64;
    let mut na = 0u64;
    let mut nb = 0u64;
    for (&x, &y) in mask_a.iter().zip(mask_b.iter()) {
        let xa = x != 0;
        let yb = y != 0;
        na += xa as u64;
        nb += yb as u64;
        inter += (xa && yb) as u64;
    }
    if na + nb == 0 {
        return Ok(1.0);
    }
    Ok((2.0 * inter as f64 / (na + nb) as f64) as f32)
}

/// Informasi timbal balik (nat) dari histogram bersama `bins × bins` dengan
/// pembobotan linear antar wadah (gaya Mattes sederhana). Rentang tiap citra
/// dinormalkan min–maks pada piksel bermasker. Rentang nol → 0.
pub fn mutual_information(
    a: &Array2<f32>,
    b: &Array2<f32>,
    mask: Option<&Array2<u8>>,
    bins: usize,
) -> Result<f32, ImgError> {
    check_same(a, b)?;
    if let Some(m) = mask {
        check_same(a, m)?;
    }
    let bins = bins.max(2);
    let mask_vals: Vec<u8> = match mask {
        Some(m) => m.iter().copied().collect(),
        None => vec![1u8; a.len()],
    };
    let sel: Vec<(f64, f64)> = a
        .iter()
        .zip(b.iter())
        .zip(mask_vals.iter())
        .filter(|(_, &mv)| mv != 0)
        .map(|((&x, &y), _)| (x as f64, y as f64))
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .collect();
    if sel.is_empty() {
        return Ok(0.0);
    }
    let (mut amin, mut amax, mut bmin, mut bmax) = (
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    );
    for &(x, y) in &sel {
        amin = amin.min(x);
        amax = amax.max(x);
        bmin = bmin.min(y);
        bmax = bmax.max(y);
    }
    if amax <= amin || bmax <= bmin {
        return Ok(0.0);
    }
    let mut joint = vec![0.0f64; bins * bins];
    let last = (bins - 1) as f64;
    let place = |v: f64, lo: f64, hi: f64| -> (usize, usize, f64) {
        let t = ((v - lo) / (hi - lo) * last).clamp(0.0, last);
        let i0 = t.floor() as usize;
        let i1 = (i0 + 1).min(bins - 1);
        (i0, i1, t - i0 as f64)
    };
    for &(x, y) in &sel {
        let (ia0, ia1, fa) = place(x, amin, amax);
        let (ib0, ib1, fb) = place(y, bmin, bmax);
        joint[ia0 * bins + ib0] += (1.0 - fa) * (1.0 - fb);
        joint[ia1 * bins + ib0] += fa * (1.0 - fb);
        joint[ia0 * bins + ib1] += (1.0 - fa) * fb;
        joint[ia1 * bins + ib1] += fa * fb;
    }
    let total = sel.len() as f64;
    let mut pa = vec![0.0f64; bins];
    let mut pb = vec![0.0f64; bins];
    for i in 0..bins {
        for j in 0..bins {
            let p = joint[i * bins + j] / total;
            pa[i] += p;
            pb[j] += p;
        }
    }
    let mut mi = 0.0f64;
    for i in 0..bins {
        for j in 0..bins {
            let p = joint[i * bins + j] / total;
            if p > 0.0 && pa[i] > 0.0 && pb[j] > 0.0 {
                mi += p * (p / (pa[i] * pb[j])).ln();
            }
        }
    }
    Ok(mi.max(0.0) as f32)
}

/// Momen orde ≤ 2 dari masker biner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Moments {
    /// Jumlah piksel bernilai ≠ 0.
    pub area: f64,
    /// Pusat massa `[x, y]` dalam piksel (pusat piksel `+0,5`).
    pub centroid: [f64; 2],
    /// Sudut sumbu utama (radian, `(-π/2, π/2]`, dari sumbu +x, searah y ke
    /// bawah).
    pub theta: f64,
    /// Rasio nilai eigen `λ_maks / λ_min` (≥ 1; `+∞` bila degenerate).
    pub eigen_ratio: f64,
}

/// Momen masker; galat bila masker kosong.
pub fn moments(mask: &Array2<u8>) -> Result<Moments, ImgError> {
    let (h, w) = mask.dim();
    let mut n = 0.0f64;
    let (mut sx, mut sy) = (0.0f64, 0.0f64);
    let (mut sxx, mut syy, mut sxy) = (0.0f64, 0.0f64, 0.0f64);
    for y in 0..h {
        for x in 0..w {
            if mask[[y, x]] == 0 {
                continue;
            }
            let px = x as f64 + 0.5;
            let py = y as f64 + 0.5;
            n += 1.0;
            sx += px;
            sy += py;
            sxx += px * px;
            syy += py * py;
            sxy += px * py;
        }
    }
    if n == 0.0 {
        return Err(ImgError::EmptyInput);
    }
    let cx = sx / n;
    let cy = sy / n;
    let mu20 = sxx / n - cx * cx;
    let mu02 = syy / n - cy * cy;
    let mu11 = sxy / n - cx * cy;
    let theta = 0.5 * (2.0 * mu11).atan2(mu20 - mu02);
    let tr = mu20 + mu02;
    let disc = ((mu20 - mu02).powi(2) + 4.0 * mu11 * mu11).sqrt();
    let l1 = 0.5 * (tr + disc);
    let l2 = 0.5 * (tr - disc);
    let eigen_ratio = if l2 > 1e-12 { l1 / l2 } else { f64::INFINITY };
    Ok(Moments {
        area: n,
        centroid: [cx, cy],
        theta,
        eigen_ratio,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(w: usize, h: usize) -> Array2<f32> {
        Array2::from_shape_fn((h, w), |(y, x)| ((x * 31 + y * 17) % 23) as f32)
    }

    #[test]
    fn ncc_identity_is_one() {
        let a = img(16, 12);
        assert!((ncc_masked(&a, &a, None).unwrap() - 1.0).abs() < 1e-6);
        let neg = a.mapv(|v| -v);
        assert!((ncc_masked(&a, &neg, None).unwrap() + 1.0).abs() < 1e-6);
        let mask = Array2::from_shape_fn((12, 16), |(y, _)| u8::from(y < 6));
        assert!((ncc_masked(&a, &a, Some(&mask)).unwrap() - 1.0).abs() < 1e-6);
        let c = Array2::from_elem((12, 16), 2.0f32);
        assert_eq!(ncc_masked(&a, &c, None).unwrap(), 0.0);
    }

    #[test]
    fn dice_values() {
        let a = Array2::from_shape_fn((4, 4), |(y, _)| u8::from(y < 2));
        let b = Array2::from_shape_fn((4, 4), |(y, _)| u8::from((1..3).contains(&y)));
        assert!((dice(&a, &a).unwrap() - 1.0).abs() < 1e-6);
        assert!((dice(&a, &b).unwrap() - 0.5).abs() < 1e-6);
        let z = Array2::<u8>::zeros((4, 4));
        assert_eq!(dice(&z, &z).unwrap(), 1.0);
        assert_eq!(dice(&a, &z).unwrap(), 0.0);
        assert!(dice(&a, &Array2::<u8>::zeros((3, 4))).is_err());
    }

    #[test]
    fn mutual_information_self_greater_than_noise() {
        let a = img(32, 32);
        let b = Array2::from_shape_fn((32, 32), |(y, x)| ((x * 7 + y * 29 + 5) % 19) as f32);
        let self_mi = mutual_information(&a, &a, None, 32).unwrap();
        let cross_mi = mutual_information(&a, &b, None, 32).unwrap();
        assert!(self_mi > cross_mi);
        assert!(self_mi > 1.0);
    }

    #[test]
    fn moments_of_rotated_ellipse() {
        let n = 128usize;
        let c = n as f64 / 2.0;
        for &deg in &[0.0f64, 30.0, 60.0, -45.0, 89.0] {
            let th = deg.to_radians();
            let (s, co) = th.sin_cos();
            let m = Array2::from_shape_fn((n, n), |(y, x)| {
                let dx = x as f64 + 0.5 - c;
                let dy = y as f64 + 0.5 - c;
                let u = dx * co + dy * s;
                let v = -dx * s + dy * co;
                u8::from((u / 40.0).powi(2) + (v / 15.0).powi(2) <= 1.0)
            });
            let mm = moments(&m).unwrap();
            let mut diff = (mm.theta - th).abs();
            if diff > std::f64::consts::FRAC_PI_2 {
                diff = std::f64::consts::PI - diff;
            }
            assert!(diff < 0.02, "sudut {deg}: {} vs {th}", mm.theta);
            assert!((mm.centroid[0] - c).abs() < 0.5 && (mm.centroid[1] - c).abs() < 0.5);
            let want_ratio = (40.0f64 / 15.0).powi(2);
            assert!((mm.eigen_ratio - want_ratio).abs() / want_ratio < 0.05);
        }
        assert!(matches!(
            moments(&Array2::<u8>::zeros((3, 3))),
            Err(ImgError::EmptyInput)
        ));
    }
}
