//! Palet warna per slide dan masker berbasis warna (ADR 0004, mode warna):
//! warna piksel jaringan dikelompokkan di ruang CIELAB (D65) dengan k-means
//! **deterministik** (inisialisasi kuantil L*, iterasi berurutan, tanpa
//! acak), lalu pengguna memilih satu atau beberapa warna palet dan satu
//! toleransi (jarak ΔE*ab) untuk membentuk masker "warna terpilih". Dipakai
//! `glands.rs` sebagai sumber peta epitel pengganti ambang hematoksilin.
//!
//! Dipanggil dari `h3d-imgproc` (`glands.rs`), engine (`Session::palette`),
//! CLI `h3d palette`, dan laci Label (kotak warna). Tanpa I/O.

use ndarray::Array2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::error::ImgError;

/// Jumlah kelompok warna bawaan.
pub const DEFAULT_K: usize = 5;
/// Jumlah sampel piksel maksimum untuk k-means.
const MAX_SAMPLES: usize = 120_000;
/// L* minimum agar kelompok dianggap "putih" (kaca, lumen kosong, lemak).
pub const WHITE_L: f32 = 85.0;

/// Konversi sRGB 8-bit → CIELAB (D65, pengamat 2°).
pub fn rgb_to_lab(r: u8, g: u8, b: u8) -> [f32; 3] {
    #[inline]
    fn lin(c: u8) -> f64 {
        let c = c as f64 / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    }
    #[inline]
    fn f(t: f64) -> f64 {
        if t > 216.0 / 24389.0 {
            t.cbrt()
        } else {
            (24389.0 / 27.0 * t + 16.0) / 116.0
        }
    }
    let (rl, gl, bl) = (lin(r), lin(g), lin(b));
    let x = (0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl) / 0.95047;
    let y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl;
    let z = (0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl) / 1.08883;
    let (fx, fy, fz) = (f(x), f(y), f(z));
    [
        (116.0 * fy - 16.0) as f32,
        (500.0 * (fx - fy)) as f32,
        (200.0 * (fy - fz)) as f32,
    ]
}

/// Jarak Euclid di Lab (ΔE*ab).
#[inline]
pub fn delta_e(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d0 = a[0] - b[0];
    let d1 = a[1] - b[1];
    let d2 = a[2] - b[2];
    (d0 * d0 + d1 * d1 + d2 * d2).sqrt()
}

/// Satu warna palet.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PaletteColor {
    /// Pusat kelompok di Lab.
    pub lab: [f32; 3],
    /// Pusat kelompok sebagai sRGB (untuk kotak warna di UI).
    pub rgb: [u8; 3],
    /// Bagian piksel jaringan yang masuk kelompok ini (0..1).
    pub fraction: f64,
    /// Sebaran kelompok: akar rerata kuadrat ΔE ke pusatnya. Toleransi
    /// bawaan pengguna = `spread × 1,5`.
    pub spread: f32,
}

impl PaletteColor {
    /// Benar bila warna ini putih/pucat (kaca, lumen kosong, lemak).
    pub fn is_white(&self) -> bool {
        self.lab[0] >= WHITE_L
    }
}

/// Palet warna satu slide, terurut dari gelap ke terang.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct Palette {
    /// Warna, terurut L* naik.
    pub colors: Vec<PaletteColor>,
    /// Jumlah sampel piksel yang dipakai.
    pub n_samples: usize,
}

/// Preset pemilihan warna untuk kasus umum.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorPreset {
    /// Kelenjar: warna ungu-biru tidak pucat (epitel padat dan tipis).
    Gland,
    /// Sarang tumor padat: sama dengan kelenjar (perbedaannya di pemisahan).
    SolidNest,
    /// Pembuluh/darah: warna bukan-putih yang paling merah (a* + b*
    /// tertinggi; ungu hematoksilin punya b* negatif sehingga kalah).
    Vessel,
}

impl Palette {
    /// Warna tergelap yang bukan putih.
    pub fn darkest(&self) -> Option<&PaletteColor> {
        self.colors.iter().find(|c| !c.is_white())
    }

    /// Indeks warna yang dipilih preset (indeks ke `colors`).
    pub fn preset_indices(&self, preset: ColorPreset) -> Vec<usize> {
        match preset {
            ColorPreset::Gland | ColorPreset::SolidNest => {
                // Epitel H&E berwarna ungu-biru (b* ≤ 0) dan tidak pucat
                // (L* ≤ 65): mencakup inti padat gelap dan sitoplasma mauve
                // epitel tipis; stroma eosin (b* > 0 atau pucat) dan otot/darah
                // (merah, b* > 0) tidak ikut.
                let purple: Vec<usize> = self
                    .colors
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| !c.is_white() && c.lab[2] <= 0.0 && c.lab[0] <= 65.0)
                    .map(|(i, _)| i)
                    .collect();
                if !purple.is_empty() {
                    return purple;
                }
                // Cadangan: pita warna tergelap.
                let Some(d) = self.darkest() else {
                    return Vec::new();
                };
                let band = d.lab[0] + 15.0;
                self.colors
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| !c.is_white() && c.lab[0] <= band)
                    .map(|(i, _)| i)
                    .collect()
            }
            ColorPreset::Vessel => self
                .colors
                .iter()
                .enumerate()
                .filter(|(_, c)| !c.is_white())
                .max_by(|a, b| {
                    let red = |c: &PaletteColor| c.lab[1] + c.lab[2];
                    red(a.1).total_cmp(&red(b.1)).then(b.0.cmp(&a.0))
                })
                .map(|(i, _)| vec![i])
                .unwrap_or_default(),
        }
    }

    /// Warna Lab dari daftar indeks (indeks di luar rentang diabaikan).
    pub fn picks(&self, indices: &[usize]) -> Vec<[f32; 3]> {
        indices
            .iter()
            .filter_map(|&i| self.colors.get(i).map(|c| c.lab))
            .collect()
    }

    /// Toleransi bawaan untuk indeks terpilih: 1,5 × sebaran (RMS ΔE)
    /// terbesar di antara mereka, dijepit ke [8, 25].
    pub fn default_tolerance(&self, indices: &[usize]) -> f32 {
        let s = indices
            .iter()
            .filter_map(|&i| self.colors.get(i).map(|c| c.spread))
            .fold(0.0f32, f32::max);
        (s * 1.5).clamp(8.0, 25.0)
    }
}

fn check_rgba(rgba: &[u8], w: usize, h: usize) -> Result<(), ImgError> {
    if rgba.len() != w * h * 4 {
        return Err(ImgError::shape(&[w * h * 4], &[rgba.len()]));
    }
    Ok(())
}

/// Ekstrak palet `k` warna (2..=8) dari piksel jaringan (`tissue` bentuk
/// `(h, w)`; `None` = semua piksel) dengan k-means Lab deterministik:
/// sampel diambil dengan langkah tetap (≤ 120 000), pusat awal = kuantil
/// L* `(i + 0,5) / k`, maks. 40 iterasi Lloyd (berhenti saat geser pusat
/// < 0,05). Kelompok kosong mempertahankan pusat lamanya. Hasil terurut L*
/// naik. Galat bila `k` di luar rentang; `Ok(Palette::default())` bila
/// tidak ada sampel.
pub fn extract_palette(
    rgba: &[u8],
    w: usize,
    h: usize,
    tissue: Option<&Array2<u8>>,
    k: usize,
) -> Result<Palette, ImgError> {
    check_rgba(rgba, w, h)?;
    if !(2..=8).contains(&k) {
        return Err(ImgError::InvalidParameter(format!(
            "jumlah warna palet harus 2..=8: {k}"
        )));
    }
    if let Some(t) = tissue {
        if t.dim() != (h, w) {
            return Err(ImgError::shape(&[h, w], &[t.dim().0, t.dim().1]));
        }
    }
    let n = w * h;
    let stride = n.div_ceil(MAX_SAMPLES).max(1);
    let mut samples: Vec<[f32; 3]> = Vec::with_capacity(n / stride + 1);
    for i in (0..n).step_by(stride) {
        if let Some(t) = tissue {
            if t[[i / w, i % w]] == 0 {
                continue;
            }
        }
        let p = &rgba[i * 4..i * 4 + 3];
        samples.push(rgb_to_lab(p[0], p[1], p[2]));
    }
    if samples.is_empty() {
        return Ok(Palette::default());
    }
    let k = k.min(samples.len());
    // Inisialisasi kuantil L*.
    let mut by_l: Vec<usize> = (0..samples.len()).collect();
    by_l.sort_by(|&a, &b| samples[a][0].total_cmp(&samples[b][0]).then(a.cmp(&b)));
    let mut centers: Vec<[f32; 3]> = (0..k)
        .map(|i| {
            let q = ((i as f64 + 0.5) / k as f64 * (by_l.len() - 1) as f64).round() as usize;
            samples[by_l[q.min(by_l.len() - 1)]]
        })
        .collect();
    let mut assign = vec![0usize; samples.len()];
    for _ in 0..40 {
        // Penugasan (paralel per sampel; hasil tidak bergantung urutan).
        assign
            .par_iter_mut()
            .zip(samples.par_iter())
            .for_each(|(a, s)| {
                let mut best = (f32::INFINITY, 0usize);
                for (j, c) in centers.iter().enumerate() {
                    let d = delta_e(*s, *c);
                    if d < best.0 {
                        best = (d, j);
                    }
                }
                *a = best.1;
            });
        // Pusat baru (akumulasi berurutan → deterministik).
        let mut sum = vec![[0.0f64; 3]; k];
        let mut cnt = vec![0usize; k];
        for (s, &a) in samples.iter().zip(assign.iter()) {
            for d in 0..3 {
                sum[a][d] += s[d] as f64;
            }
            cnt[a] += 1;
        }
        let mut shift = 0.0f32;
        for j in 0..k {
            if cnt[j] == 0 {
                continue;
            }
            let c = [
                (sum[j][0] / cnt[j] as f64) as f32,
                (sum[j][1] / cnt[j] as f64) as f32,
                (sum[j][2] / cnt[j] as f64) as f32,
            ];
            shift = shift.max(delta_e(c, centers[j]));
            centers[j] = c;
        }
        if shift < 0.05 {
            break;
        }
    }
    // Statistik kelompok.
    let mut cnt = vec![0usize; k];
    let mut sq = vec![0.0f64; k];
    for (s, &a) in samples.iter().zip(assign.iter()) {
        cnt[a] += 1;
        let d = delta_e(*s, centers[a]);
        sq[a] += (d * d) as f64;
    }
    let total = samples.len() as f64;
    let mut colors: Vec<PaletteColor> = (0..k)
        .filter(|&j| cnt[j] > 0)
        .map(|j| PaletteColor {
            lab: centers[j],
            rgb: lab_to_rgb(centers[j]),
            fraction: cnt[j] as f64 / total,
            spread: (sq[j] / cnt[j] as f64).sqrt() as f32,
        })
        .collect();
    colors.sort_by(|a, b| a.lab[0].total_cmp(&b.lab[0]));
    Ok(Palette {
        colors,
        n_samples: samples.len(),
    })
}

/// CIELAB → sRGB 8-bit (dijepit).
pub fn lab_to_rgb(lab: [f32; 3]) -> [u8; 3] {
    let (l, a, b) = (lab[0] as f64, lab[1] as f64, lab[2] as f64);
    let fy = (l + 16.0) / 116.0;
    let fx = fy + a / 500.0;
    let fz = fy - b / 200.0;
    let finv = |t: f64| {
        let t3 = t * t * t;
        if t3 > 216.0 / 24389.0 {
            t3
        } else {
            (116.0 * t - 16.0) * 27.0 / 24389.0
        }
    };
    let x = finv(fx) * 0.95047;
    let y = finv(fy);
    let z = finv(fz) * 1.08883;
    let rl = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
    let gl = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z;
    let bl = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z;
    let gam = |c: f64| {
        let c = c.clamp(0.0, 1.0);
        let v = if c <= 0.0031308 {
            12.92 * c
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        };
        (v * 255.0).round().clamp(0.0, 255.0) as u8
    };
    [gam(rl), gam(gl), gam(bl)]
}

/// Peta jarak warna: ΔE*ab minimum tiap piksel ke salah satu `picks`
/// (bentuk `(h, w)`; paralel per baris, deterministik). `picks` kosong →
/// `f32::INFINITY` di semua piksel. Dipakai `glands.rs`: peta ini
/// dihaluskan Gauss lalu diambang dengan toleransi, sehingga campuran inti
/// gelap dan sitoplasma epitel tetap "dekat" sementara sekat tipis di antara
/// vakuola putih menjadi "jauh".
pub fn color_distance(
    rgba: &[u8],
    w: usize,
    h: usize,
    picks: &[[f32; 3]],
) -> Result<Array2<f32>, ImgError> {
    check_rgba(rgba, w, h)?;
    let mut out = vec![f32::INFINITY; w * h];
    if !picks.is_empty() && w > 0 && h > 0 {
        rgba.par_chunks(w * 4)
            .zip(out.par_chunks_mut(w))
            .for_each(|(row, o)| {
                for (i, px) in row.as_chunks::<4>().0.iter().enumerate() {
                    let lab = rgb_to_lab(px[0], px[1], px[2]);
                    o[i] = picks
                        .iter()
                        .map(|p| delta_e(lab, *p))
                        .fold(f32::INFINITY, f32::min);
                }
            });
    }
    Array2::from_shape_vec((h, w), out).map_err(|_| ImgError::EmptyInput)
}

/// Masker piksel yang jarak Lab-nya ke salah satu warna `picks` ≤ `tol`
/// (bentuk `(h, w)`; paralel per baris, deterministik). `picks` kosong →
/// masker nol.
pub fn color_mask(
    rgba: &[u8],
    w: usize,
    h: usize,
    picks: &[[f32; 3]],
    tol: f32,
) -> Result<Array2<u8>, ImgError> {
    check_rgba(rgba, w, h)?;
    if !(tol.is_finite() && tol >= 0.0) {
        return Err(ImgError::InvalidParameter(format!(
            "toleransi warna tidak sah: {tol}"
        )));
    }
    let mut out = vec![0u8; w * h];
    if !picks.is_empty() && w > 0 && h > 0 {
        rgba.par_chunks(w * 4)
            .zip(out.par_chunks_mut(w))
            .for_each(|(row, o)| {
                for (i, px) in row.as_chunks::<4>().0.iter().enumerate() {
                    let lab = rgb_to_lab(px[0], px[1], px[2]);
                    if picks.iter().any(|p| delta_e(lab, *p) <= tol) {
                        o[i] = 1;
                    }
                }
            });
    }
    Array2::from_shape_vec((h, w), out).map_err(|_| ImgError::EmptyInput)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn three_color_image() -> (Vec<u8>, usize, usize) {
        let (w, h) = (120usize, 60usize);
        let mut rgba = vec![255u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let c = if x < 40 {
                    [90u8, 40, 120]
                } else if x < 80 {
                    [235, 190, 205]
                } else {
                    [250, 250, 250]
                };
                let k = (y * w + x) * 4;
                // Sedikit variasi deterministik agar sebaran > 0.
                let d = ((x + y) % 3) as u8;
                rgba[k..k + 3].copy_from_slice(&[c[0].saturating_add(d), c[1], c[2]]);
            }
        }
        (rgba, w, h)
    }

    #[test]
    fn lab_roundtrip_and_known_values() {
        let white = rgb_to_lab(255, 255, 255);
        assert!((white[0] - 100.0).abs() < 0.5 && white[1].abs() < 0.5 && white[2].abs() < 0.5);
        let black = rgb_to_lab(0, 0, 0);
        assert!(black[0].abs() < 0.5);
        for c in [[90u8, 40, 120], [235, 190, 205], [12, 200, 77]] {
            let back = lab_to_rgb(rgb_to_lab(c[0], c[1], c[2]));
            for k in 0..3 {
                assert!(
                    (back[k] as i32 - c[k] as i32).abs() <= 1,
                    "{c:?} → {back:?}"
                );
            }
        }
    }

    #[test]
    fn palette_recovers_three_colors_sorted_dark_to_light_and_is_deterministic() {
        let (rgba, w, h) = three_color_image();
        let p = extract_palette(&rgba, w, h, None, 3).unwrap();
        assert_eq!(p.colors.len(), 3);
        assert!(p.colors[0].lab[0] < p.colors[1].lab[0] && p.colors[1].lab[0] < p.colors[2].lab[0]);
        assert!(delta_e(p.colors[0].lab, rgb_to_lab(91, 40, 120)) < 3.0);
        assert!(p.colors[2].is_white() && !p.colors[0].is_white());
        for c in &p.colors {
            assert!((c.fraction - 1.0 / 3.0).abs() < 0.02, "{}", c.fraction);
        }
        assert_eq!(p.preset_indices(ColorPreset::Gland), vec![0]);
        assert_eq!(p.preset_indices(ColorPreset::Vessel), vec![1]);
        assert_eq!(p, extract_palette(&rgba, w, h, None, 3).unwrap());
        // Masker warna gelap = kolom kiri saja.
        let m = color_mask(&rgba, w, h, &p.picks(&[0]), p.default_tolerance(&[0])).unwrap();
        let n = m.iter().filter(|v| **v != 0).count();
        assert_eq!(n, 40 * h);
        assert!(m[[0, 0]] == 1 && m[[0, 100]] == 0);
        assert_eq!(
            color_mask(&rgba, w, h, &[], 10.0)
                .unwrap()
                .iter()
                .filter(|v| **v != 0)
                .count(),
            0
        );
    }

    #[test]
    fn invalid_inputs_are_errors_and_empty_tissue_gives_empty_palette() {
        let (rgba, w, h) = three_color_image();
        assert!(extract_palette(&rgba, w, h, None, 1).is_err());
        assert!(extract_palette(&rgba, w, h, None, 9).is_err());
        assert!(extract_palette(&rgba[..8], 2, 2, None, 3).is_err());
        let tm = Array2::<u8>::zeros((h, w));
        let p = extract_palette(&rgba, w, h, Some(&tm), 3).unwrap();
        assert!(p.colors.is_empty() && p.darkest().is_none());
        assert!(color_mask(&rgba, w, h, &[[0.0; 3]], f32::NAN).is_err());
    }
}
