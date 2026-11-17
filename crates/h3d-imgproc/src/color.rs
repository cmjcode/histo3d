//! Dekonvolusi warna Ruifrok–Johnston (histo3d.md 8.3).
//!
//! Semua citra RGBA berurutan baris (row-major), 4 byte per piksel; hasil
//! berupa `Array2<f32>` berbentuk `(h, w)` (indeks `[y, x]`).

use std::sync::LazyLock;

use ndarray::Array2;
use rayon::prelude::*;

use crate::error::ImgError;

/// Tabel `OD(I) = −log10((I + 1) / 256)` untuk `I ∈ 0..=255`.
static OD_LUT: LazyLock<[f32; 256]> = LazyLock::new(|| {
    let mut t = [0.0f32; 256];
    for (i, v) in t.iter_mut().enumerate() {
        *v = -(((i as f64) + 1.0) / 256.0).log10() as f32;
    }
    t
});

/// Kerapatan optik satu kanal: `OD = −log10((I + 1) / 256)`.
#[inline]
pub fn od_of(intensity: u8) -> f32 {
    OD_LUT[intensity as usize]
}

/// Balikan `od_of`: intensitas 8-bit dari kerapatan optik (dijepit 0..=255).
pub fn intensity_of_od(od: f32) -> u8 {
    let i = 256.0 * 10f64.powf(-(od as f64)) - 1.0;
    i.round().clamp(0.0, 255.0) as u8
}

/// OD tiga kanal dari piksel RGB (dipakai tiga byte pertama `rgb`).
///
/// Panics bila `rgb.len() < 3`; pemanggil menjamin panjang (fungsi ini dipakai
/// pada potongan buffer yang sudah divalidasi).
#[inline]
pub fn rgb_to_od(rgb: &[u8]) -> [f32; 3] {
    [od_of(rgb[0]), od_of(rgb[1]), od_of(rgb[2])]
}

/// Vektor pewarna bawaan (belum dinormalkan) menurut Ruifrok & Johnston.
pub const HEMATOXYLIN: [f32; 3] = [0.65, 0.70, 0.29];
/// Eosin.
pub const EOSIN: [f32; 3] = [0.07, 0.99, 0.11];
/// DAB (3,3'-diaminobenzidine).
pub const DAB: [f32; 3] = [0.27, 0.57, 0.78];

/// Matriks pewarna 3×3: baris = vektor OD tiap pewarna (dinormalkan ke
/// panjang 1), beserta balikannya untuk menghitung konsentrasi.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StainMatrix {
    /// Baris = vektor pewarna ternormalisasi.
    pub stains: [[f32; 3]; 3],
    /// Balikan `stains` (presisi ganda, dihitung sekali).
    inv: [[f64; 3]; 3],
}

fn normalize3(v: [f32; 3]) -> Option<[f32; 3]> {
    let n =
        (v[0] as f64 * v[0] as f64 + v[1] as f64 * v[1] as f64 + v[2] as f64 * v[2] as f64).sqrt();
    if !n.is_finite() || n < 1e-12 {
        return None;
    }
    Some([
        (v[0] as f64 / n) as f32,
        (v[1] as f64 / n) as f32,
        (v[2] as f64 / n) as f32,
    ])
}

fn invert3(m: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if !det.is_finite() || det.abs() < 1e-9 {
        return None;
    }
    let inv = [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det,
        ],
    ];
    Some(inv)
}

impl StainMatrix {
    /// Dari tiga vektor pewarna (dinormalkan di sini). Galat bila ada vektor
    /// nol atau matriks singular (vektor saling bergantung linear).
    pub fn new(stains: [[f32; 3]; 3]) -> Result<Self, ImgError> {
        let mut norm = [[0.0f32; 3]; 3];
        for (i, s) in stains.iter().enumerate() {
            norm[i] = normalize3(*s).ok_or_else(|| {
                ImgError::InvalidParameter(format!("vektor pewarna ke-{i} bernilai nol"))
            })?;
        }
        let m64 = [
            [norm[0][0] as f64, norm[0][1] as f64, norm[0][2] as f64],
            [norm[1][0] as f64, norm[1][1] as f64, norm[1][2] as f64],
            [norm[2][0] as f64, norm[2][1] as f64, norm[2][2] as f64],
        ];
        let inv = invert3(m64)
            .ok_or_else(|| ImgError::InvalidParameter("matriks pewarna singular".to_string()))?;
        Ok(Self { stains: norm, inv })
    }

    /// Hematoksilin, eosin, dan vektor ketiga ortogonal (residu).
    pub fn he() -> Self {
        let third = cross(HEMATOXYLIN, EOSIN);
        Self::new([HEMATOXYLIN, EOSIN, third]).unwrap_or_else(|_| unreachable!())
    }

    /// Hematoksilin, DAB, dan vektor ketiga ortogonal (residu).
    pub fn h_dab() -> Self {
        let third = cross(HEMATOXYLIN, DAB);
        Self::new([HEMATOXYLIN, DAB, third]).unwrap_or_else(|_| unreachable!())
    }

    /// Hematoksilin, eosin, DAB (ketiga vektor Ruifrok sekaligus).
    pub fn h_e_dab() -> Self {
        Self::new([HEMATOXYLIN, EOSIN, DAB]).unwrap_or_else(|_| unreachable!())
    }

    /// Konsentrasi `c` dari OD: `OD = c · M` ⇒ `c = OD · M⁻¹`.
    #[inline]
    pub fn concentrations(&self, od: [f32; 3]) -> [f32; 3] {
        let mut c = [0.0f32; 3];
        for (j, cj) in c.iter_mut().enumerate() {
            let v = od[0] as f64 * self.inv[0][j]
                + od[1] as f64 * self.inv[1][j]
                + od[2] as f64 * self.inv[2][j];
            *cj = v as f32;
        }
        c
    }

    /// OD dari konsentrasi: `OD = c · M` (untuk sintesis warna, mis. fantom).
    #[inline]
    pub fn od_from_concentrations(&self, c: [f32; 3]) -> [f32; 3] {
        let mut od = [0.0f32; 3];
        for (k, o) in od.iter_mut().enumerate() {
            *o = c[0] * self.stains[0][k] + c[1] * self.stains[1][k] + c[2] * self.stains[2][k];
        }
        od
    }
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn check_rgba(rgba: &[u8], w: usize, h: usize) -> Result<(), ImgError> {
    if w == 0 || h == 0 {
        return Err(ImgError::EmptyInput);
    }
    let expected = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| ImgError::InvalidParameter("ukuran citra meluap".to_string()))?;
    if rgba.len() != expected {
        return Err(ImgError::shape(&[expected], &[rgba.len()]));
    }
    Ok(())
}

/// Dekonvolusi: tiga kanal konsentrasi pewarna (urutan sesuai baris `m`),
/// masing-masing `Array2<f32>` berbentuk `(h, w)`.
pub fn deconvolve(
    rgba: &[u8],
    w: usize,
    h: usize,
    m: &StainMatrix,
) -> Result<[Array2<f32>; 3], ImgError> {
    check_rgba(rgba, w, h)?;
    let n = w * h;
    let mut c0 = vec![0.0f32; n];
    let mut c1 = vec![0.0f32; n];
    let mut c2 = vec![0.0f32; n];
    // Paralel per baris; tiap baris menulis ke potongan keluarannya sendiri
    // sehingga hasil deterministik.
    rgba.par_chunks(w * 4)
        .zip(c0.par_chunks_mut(w))
        .zip(c1.par_chunks_mut(w))
        .zip(c2.par_chunks_mut(w))
        .for_each(|(((row, o0), o1), o2)| {
            for (i, px) in row.as_chunks::<4>().0.iter().enumerate() {
                let c = m.concentrations(rgb_to_od(px));
                o0[i] = c[0];
                o1[i] = c[1];
                o2[i] = c[2];
            }
        });
    let mk = |v: Vec<f32>| Array2::from_shape_vec((h, w), v).map_err(|_| ImgError::EmptyInput);
    Ok([mk(c0)?, mk(c1)?, mk(c2)?])
}

/// `OD_R + OD_G + OD_B` per piksel, bentuk `(h, w)`.
pub fn od_total(rgba: &[u8], w: usize, h: usize) -> Result<Array2<f32>, ImgError> {
    check_rgba(rgba, w, h)?;
    let mut out = vec![0.0f32; w * h];
    rgba.par_chunks(w * 4)
        .zip(out.par_chunks_mut(w))
        .for_each(|(row, o)| {
            for (i, px) in row.as_chunks::<4>().0.iter().enumerate() {
                let od = rgb_to_od(px);
                o[i] = od[0] + od[1] + od[2];
            }
        });
    Array2::from_shape_vec((h, w), out).map_err(|_| ImgError::EmptyInput)
}

/// Kanal hematoksilin (baris pertama `m`) yang diregangkan ke 0..1 memakai
/// persentil 1–99 (deterministik: nilai persentil dari salinan terurut).
/// Bila rentang persentil nol, seluruh kanal menjadi 0.
pub fn hematoxylin_channel(
    rgba: &[u8],
    w: usize,
    h: usize,
    m: &StainMatrix,
) -> Result<Array2<f32>, ImgError> {
    let [hch, _, _] = deconvolve(rgba, w, h, m)?;
    Ok(percentile_stretch(&hch, 0.01, 0.99))
}

/// Nilai persentil `p ∈ [0, 1]` (metode indeks terdekat) dari `values`,
/// mengabaikan NaN. `None` bila kosong.
pub fn percentile(values: &[f32], p: f32) -> Option<f32> {
    let mut v: Vec<f32> = values.iter().copied().filter(|x| !x.is_nan()).collect();
    if v.is_empty() {
        return None;
    }
    let p = p.clamp(0.0, 1.0) as f64;
    let idx = ((v.len() - 1) as f64 * p).round() as usize;
    let (_, nth, _) = v.select_nth_unstable_by(idx, |a, b| a.total_cmp(b));
    Some(*nth)
}

/// Regangkan linear sehingga persentil `lo → 0` dan `hi → 1`, dijepit 0..1.
pub fn percentile_stretch(arr: &Array2<f32>, lo: f32, hi: f32) -> Array2<f32> {
    let flat: Vec<f32> = arr.iter().copied().collect();
    let (Some(a), Some(b)) = (percentile(&flat, lo), percentile(&flat, hi)) else {
        return arr.clone();
    };
    let range = b - a;
    if range.is_nan() || range <= 0.0 {
        return Array2::zeros(arr.raw_dim());
    }
    arr.mapv(|v| ((v - a) / range).clamp(0.0, 1.0))
}

/// Sudut minimum (derajat) antara vektor H dan E hasil Macenko agar estimasi
/// dianggap sah; di bawah ini kedua pewarna tidak terpisah (jaringan satu
/// warna) dan pemanggil memakai matriks Ruifrok tetap.
pub const MACENKO_MIN_ANGLE_DEG: f64 = 10.0;
/// Jumlah piksel berpewarna minimum untuk estimasi Macenko.
pub const MACENKO_MIN_PIXELS: usize = 1000;

/// Estimasi vektor pewarna H dan E per slide dengan metode Macenko dkk.
/// (2009): OD piksel jaringan dengan `|OD| > beta` diproyeksikan ke bidang
/// dua vektor eigen kovarians terbesar; arah pada persentil sudut `alpha`
/// dan `1 − alpha` menjadi vektor ekstrem. Vektor yang lebih searah dengan
/// hematoksilin Ruifrok menjadi baris H, lainnya E, baris ketiga = silang
/// keduanya. `tissue` (bentuk `(h, w)`) membatasi piksel yang dipakai.
///
/// Mengembalikan `Ok(None)` bila piksel berpewarna `< MACENKO_MIN_PIXELS`
/// atau sudut H–E `< MACENKO_MIN_ANGLE_DEG` (mis. irisan hampir kosong atau
/// satu warna); pemanggil lalu memakai [`StainMatrix::he`]. Deterministik:
/// piksel diambil dengan langkah tetap (maks. ~200 000 sampel), eigen dengan
/// rotasi Jacobi berurutan tetap.
pub fn estimate_stains_macenko(
    rgba: &[u8],
    w: usize,
    h: usize,
    tissue: Option<&Array2<u8>>,
    beta: f32,
    alpha: f32,
) -> Result<Option<StainMatrix>, ImgError> {
    check_rgba(rgba, w, h)?;
    if let Some(t) = tissue {
        if t.dim() != (h, w) {
            return Err(ImgError::shape(&[h, w], &[t.dim().0, t.dim().1]));
        }
    }
    if !(beta.is_finite() && beta >= 0.0 && alpha.is_finite() && (0.0..0.5).contains(&alpha)) {
        return Err(ImgError::InvalidParameter(
            "beta harus ≥ 0 dan alpha di [0, 0,5)".to_string(),
        ));
    }
    let n = w * h;
    let stride = n.div_ceil(200_000).max(1);
    let mut od: Vec<[f64; 3]> = Vec::with_capacity(n / stride + 1);
    let b2 = (beta as f64) * (beta as f64);
    for i in (0..n).step_by(stride) {
        if let Some(t) = tissue {
            if t[[i / w, i % w]] == 0 {
                continue;
            }
        }
        let v = rgb_to_od(&rgba[i * 4..i * 4 + 3]);
        let v = [v[0] as f64, v[1] as f64, v[2] as f64];
        if v[0] * v[0] + v[1] * v[1] + v[2] * v[2] > b2 {
            od.push(v);
        }
    }
    if od.len() < MACENKO_MIN_PIXELS {
        return Ok(None);
    }
    // Kovarians (tanpa pemusatan: Macenko memakai momen kedua OD mentah).
    let mut cov = [[0.0f64; 3]; 3];
    for v in &od {
        for (i, row) in cov.iter_mut().enumerate() {
            for (j, c) in row.iter_mut().enumerate() {
                *c += v[i] * v[j];
            }
        }
    }
    let inv_n = 1.0 / od.len() as f64;
    for row in cov.iter_mut() {
        for c in row.iter_mut() {
            *c *= inv_n;
        }
    }
    let (vals, vecs) = jacobi_eigen_3x3(cov);
    // Dua vektor eigen terbesar.
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&a, &b| vals[b].total_cmp(&vals[a]));
    let v1 = vecs[idx[0]];
    let v2 = vecs[idx[1]];
    let mut phis: Vec<f64> = od
        .iter()
        .map(|v| {
            let a = v[0] * v1[0] + v[1] * v1[1] + v[2] * v1[2];
            let b = v[0] * v2[0] + v[1] * v2[1] + v[2] * v2[2];
            b.atan2(a)
        })
        .collect();
    phis.sort_by(f64::total_cmp);
    let pick = |p: f64| -> f64 {
        let k = ((phis.len() - 1) as f64 * p).round() as usize;
        phis[k.min(phis.len() - 1)]
    };
    let (p_lo, p_hi) = (pick(alpha as f64), pick(1.0 - alpha as f64));
    let dir = |phi: f64| -> [f64; 3] {
        let (s, c) = phi.sin_cos();
        let mut v = [
            c * v1[0] + s * v2[0],
            c * v1[1] + s * v2[1],
            c * v1[2] + s * v2[2],
        ];
        // OD pewarna tidak negatif: balik tanda bila arah menunjuk ke bawah.
        if v[0] + v[1] + v[2] < 0.0 {
            v = [-v[0], -v[1], -v[2]];
        }
        v
    };
    let (a, b) = (dir(p_lo), dir(p_hi));
    let dot = |x: [f64; 3], y: [f64; 3]| x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
    let norm = |x: [f64; 3]| dot(x, x).sqrt();
    let cos_ab = dot(a, b) / (norm(a) * norm(b)).max(1e-12);
    if !cos_ab.is_finite() || cos_ab.clamp(-1.0, 1.0).acos().to_degrees() < MACENKO_MIN_ANGLE_DEG {
        return Ok(None);
    }
    let href = [
        HEMATOXYLIN[0] as f64,
        HEMATOXYLIN[1] as f64,
        HEMATOXYLIN[2] as f64,
    ];
    let (hv, ev) = if dot(a, href) / norm(a) >= dot(b, href) / norm(b) {
        (a, b)
    } else {
        (b, a)
    };
    let to32 = |x: [f64; 3]| [x[0] as f32, x[1] as f32, x[2] as f32];
    let (hv, ev) = (to32(hv), to32(ev));
    let third = cross(hv, ev);
    match StainMatrix::new([hv, ev, third]) {
        Ok(m) => Ok(Some(m)),
        Err(_) => Ok(None),
    }
}

/// Nilai dan vektor eigen matriks simetris 3×3 dengan rotasi Jacobi siklik
/// (urutan pasangan tetap, maks. 50 sapuan). Vektor eigen dikembalikan
/// sebagai baris, dinormalkan, searah nilai eigen ke-`i`.
fn jacobi_eigen_3x3(mut a: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..50 {
        let off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
        if off < 1e-22 {
            break;
        }
        for (p, q) in [(0usize, 1usize), (0, 2), (1, 2)] {
            if a[p][q].abs() < 1e-300 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let t = if theta == 0.0 { 1.0 } else { t };
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            for row in a.iter_mut() {
                let akp = row[p];
                let akq = row[q];
                row[p] = c * akp - s * akq;
                row[q] = s * akp + c * akq;
            }
            let (rp, rq) = (a[p], a[q]);
            for k in 0..3 {
                a[p][k] = c * rp[k] - s * rq[k];
                a[q][k] = s * rp[k] + c * rq[k];
            }
            for row in v.iter_mut() {
                let vp = row[p];
                let vq = row[q];
                row[p] = c * vp - s * vq;
                row[q] = s * vp + c * vq;
            }
        }
    }
    let vals = [a[0][0], a[1][1], a[2][2]];
    // Kolom `v` = vektor eigen; kembalikan sebagai baris.
    let vecs = [
        [v[0][0], v[1][0], v[2][0]],
        [v[0][1], v[1][1], v[2][1]],
        [v[0][2], v[1][2], v[2][2]],
    ];
    (vals, vecs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn od_white_is_zero() {
        assert_eq!(od_of(255), 0.0);
        assert!(od_of(0) > 2.4);
        assert_eq!(intensity_of_od(0.0), 255);
    }

    #[test]
    fn pure_stain_recovers_single_channel() {
        let m = StainMatrix::h_dab();
        for stain in 0..2 {
            // Konsentrasi 1,0 pada satu pewarna saja.
            let mut c = [0.0f32; 3];
            c[stain] = 1.0;
            let od = m.od_from_concentrations(c);
            let px = [
                intensity_of_od(od[0]),
                intensity_of_od(od[1]),
                intensity_of_od(od[2]),
                255,
            ];
            let out = deconvolve(&px, 1, 1, &m).unwrap();
            let got = [out[0][[0, 0]], out[1][[0, 0]], out[2][[0, 0]]];
            for (k, g) in got.iter().enumerate() {
                let want = if k == stain { 1.0 } else { 0.0 };
                assert!((g - want).abs() < 0.05, "stain {stain} kanal {k}: {got:?}");
            }
        }
    }

    #[test]
    fn singular_matrix_rejected() {
        let r = StainMatrix::new([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        assert!(matches!(r, Err(ImgError::InvalidParameter(_))));
        let r = StainMatrix::new([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        assert!(matches!(r, Err(ImgError::InvalidParameter(_))));
    }

    #[test]
    fn wrong_buffer_length_is_error() {
        assert!(matches!(
            od_total(&[0u8; 7], 2, 1),
            Err(ImgError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn percentile_basic() {
        let v = [5.0, 1.0, 3.0, 2.0, 4.0];
        assert_eq!(percentile(&v, 0.0), Some(1.0));
        assert_eq!(percentile(&v, 1.0), Some(5.0));
        assert_eq!(percentile(&v, 0.5), Some(3.0));
    }

    #[test]
    fn jacobi_recovers_diagonal_and_symmetric_eigenpairs() {
        let (vals, vecs) = jacobi_eigen_3x3([[2.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 1.0]]);
        let mut sorted = vals;
        sorted.sort_by(f64::total_cmp);
        assert!((sorted[0] - 1.0).abs() < 1e-9 && (sorted[2] - 5.0).abs() < 1e-9);
        let m = [[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]];
        let (vals, vecs2) = jacobi_eigen_3x3(m);
        for (i, v) in vecs2.iter().enumerate() {
            // M·v = λ·v
            for r in 0..3 {
                let mv = m[r][0] * v[0] + m[r][1] * v[1] + m[r][2] * v[2];
                assert!((mv - vals[i] * v[r]).abs() < 1e-6, "baris {r} vektor {i}");
            }
        }
        let _ = vecs;
    }

    #[test]
    fn macenko_recovers_ruifrok_vectors_from_synthetic_he() {
        let mpp = 1.0;
        let (w, h) = (256usize, 256usize);
        let truth = crate::phantom::ellipsoid_mask([w, h], [128.0, 128.0], [110.0, 100.0]);
        let rgba = crate::phantom::render_histology(&truth, 11, mpp);
        let m = estimate_stains_macenko(&rgba, w, h, Some(&truth), 0.15, 0.01)
            .unwrap()
            .expect("jaringan cukup untuk estimasi");
        let ang = |a: [f32; 3], b: [f32; 3]| {
            let d = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) as f64;
            d.clamp(-1.0, 1.0).acos().to_degrees()
        };
        let href = StainMatrix::he().stains[0];
        let eref = StainMatrix::he().stains[1];
        assert!(
            ang(m.stains[0], href) < 15.0,
            "H {:?} vs {:?}",
            m.stains[0],
            href
        );
        assert!(
            ang(m.stains[1], eref) < 15.0,
            "E {:?} vs {:?}",
            m.stains[1],
            eref
        );
        // Terlalu sedikit piksel → None, bukan galat; parameter buruk → galat.
        let few = estimate_stains_macenko(&rgba[..16 * 4], 4, 4, None, 0.15, 0.01).unwrap();
        assert!(few.is_none());
        assert!(estimate_stains_macenko(&rgba, w, h, None, -1.0, 0.01).is_err());
        assert!(estimate_stains_macenko(&rgba, w, h, None, 0.15, 0.7).is_err());
    }
}
