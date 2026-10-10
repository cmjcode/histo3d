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
            for (i, px) in row.chunks_exact(4).enumerate() {
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
            for (i, px) in row.chunks_exact(4).enumerate() {
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
}
