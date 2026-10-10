//! Deteksi inti klasik: LoG multiskala pada kanal hematoksilin
//! (histo3d.md 8.12.5 butir 2 dan 6).
//!
//! Konvensi tanda: kanal `H` tinggi di inti, sehingga respons LoG (lihat
//! `filters::log_filter`) **negatif** di pusat inti; kandidat adalah
//! **minimum lokal** respons gabungan (minimum atas skala).

use ndarray::Array2;

use crate::filters::{gaussian_blur_f32, local_minima, log_filter};
use crate::threshold::{otsu, otsu_masked};

/// Parameter deteksi inti (satuan µm).
#[derive(Debug, Clone, PartialEq)]
pub struct NucleiParams {
    /// σ pemulusan Gauss awal pada `H` (bawaan 1 µm).
    pub smooth_sigma_um: f64,
    /// Skala LoG (bawaan {2, 3, 4} µm).
    pub log_sigmas_um: Vec<f64>,
    /// Jarak supresi non-maksimum (bawaan 4 µm).
    pub nms_radius_um: f64,
    /// Ambang `H_smooth`; `None` → Otsu di dalam masker jaringan.
    pub h_threshold: Option<f32>,
}

impl Default for NucleiParams {
    fn default() -> Self {
        Self {
            smooth_sigma_um: 1.0,
            log_sigmas_um: vec![2.0, 3.0, 4.0],
            nms_radius_um: 4.0,
            h_threshold: None,
        }
    }
}

/// Satu inti terdeteksi. Koordinat dalam µm (ruang citra, pusat piksel `+0,5`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Nucleus {
    /// Posisi x (µm).
    pub x_um: f64,
    /// Posisi y (µm).
    pub y_um: f64,
    /// Skala LoG pemenang (µm) ≈ jari-jari inti.
    pub sigma_um: f32,
    /// Rata-rata `H` dalam cakram jari-jari σ.
    pub h_mean: f32,
    /// Rata-rata kanal kromogen dalam cakram jari-jari σ (diisi
    /// `measure_chromogen`; 0 sebelum itu).
    pub d_mean: f32,
    /// Respons LoG di pusat (negatif; makin kecil makin kuat).
    pub response: f32,
}

/// Rata-rata `arr` dalam cakram berjari-jari `r_px` di sekitar piksel
/// `(cx, cy)` (pusat piksel). Cakram dipotong di tepi.
pub fn disc_mean(arr: &Array2<f32>, cx: usize, cy: usize, r_px: f64) -> f32 {
    let (h, w) = arr.dim();
    let r = r_px.max(0.0);
    let ri = r.ceil() as i64;
    let r2 = r * r;
    let mut s = 0.0f64;
    let mut n = 0u32;
    for dy in -ri..=ri {
        let y = cy as i64 + dy;
        if y < 0 || y >= h as i64 {
            continue;
        }
        for dx in -ri..=ri {
            let x = cx as i64 + dx;
            if x < 0 || x >= w as i64 {
                continue;
            }
            if (dx * dx + dy * dy) as f64 <= r2 {
                s += arr[[y as usize, x as usize]] as f64;
                n += 1;
            }
        }
    }
    if n == 0 {
        arr[[cy.min(h.saturating_sub(1)), cx.min(w.saturating_sub(1))]]
    } else {
        (s / n as f64) as f32
    }
}

/// Deteksi inti pada kanal hematoksilin `h_channel` (bentuk `(h, w)`, nilai
/// OD/konsentrasi) dengan ukuran piksel `mpp` (µm/px). Hanya kandidat di
/// dalam `tissue_mask` (bila ada) dengan `H_smooth > t_H`. Hasil diurutkan
/// menurut `(y_um, x_um)`; `d_mean` = 0.
pub fn detect_nuclei(
    h_channel: &Array2<f32>,
    mpp: f64,
    tissue_mask: Option<&Array2<u8>>,
    params: &NucleiParams,
) -> Vec<Nucleus> {
    let (h, w) = h_channel.dim();
    if h == 0 || w == 0 || !(mpp.is_finite() && mpp > 0.0) || params.log_sigmas_um.is_empty() {
        return Vec::new();
    }
    let mask_ok = tissue_mask.filter(|m| m.dim() == (h, w));
    let hs = gaussian_blur_f32(h_channel, params.smooth_sigma_um / mpp);
    let t_h = params.h_threshold.unwrap_or_else(|| match mask_ok {
        Some(m) => otsu_masked(&hs, m),
        None => otsu(hs.iter().copied(), 256),
    });
    // Respons gabungan: minimum atas skala, catat skala pemenang.
    let mut best = Array2::<f32>::from_elem((h, w), f32::INFINITY);
    let mut best_sigma = Array2::<f32>::zeros((h, w));
    for &s_um in &params.log_sigmas_um {
        if s_um.is_nan() || s_um <= 0.0 {
            continue;
        }
        let r = log_filter(&hs, s_um / mpp);
        for ((b, bs), &v) in best.iter_mut().zip(best_sigma.iter_mut()).zip(r.iter()) {
            if v < *b {
                *b = v;
                *bs = s_um as f32;
            }
        }
    }
    let mut cands: Vec<(f32, usize, usize)> = local_minima(&best, 1, 0.0)
        .into_iter()
        .filter(|&(x, y)| hs[[y, x]] > t_h && mask_ok.map(|m| m[[y, x]] != 0).unwrap_or(true))
        .map(|(x, y)| (best[[y, x]], x, y))
        .collect();
    // Terkuat dulu (respons paling negatif); seri → urutan raster.
    cands.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.2.cmp(&b.2)).then(a.1.cmp(&b.1)));
    let nms_px = params.nms_radius_um / mpp;
    let nms2 = nms_px * nms_px;
    let mut kept: Vec<(usize, usize)> = Vec::new();
    for &(_, x, y) in &cands {
        let ok = kept.iter().all(|&(kx, ky)| {
            let dx = kx as f64 - x as f64;
            let dy = ky as f64 - y as f64;
            dx * dx + dy * dy > nms2
        });
        if ok {
            kept.push((x, y));
        }
    }
    let mut out: Vec<Nucleus> = kept
        .into_iter()
        .map(|(x, y)| {
            let sigma = best_sigma[[y, x]];
            Nucleus {
                x_um: (x as f64 + 0.5) * mpp,
                y_um: (y as f64 + 0.5) * mpp,
                sigma_um: sigma,
                h_mean: disc_mean(h_channel, x, y, sigma as f64 / mpp),
                d_mean: 0.0,
                response: best[[y, x]],
            }
        })
        .collect();
    out.sort_by(|a, b| a.y_um.total_cmp(&b.y_um).then(a.x_um.total_cmp(&b.x_um)));
    out
}

/// Isi `d_mean` tiap inti dengan rata-rata `d_channel` dalam cakram
/// jari-jari σ di sekitar pusatnya (koordinat µm → piksel dengan `mpp`).
pub fn measure_chromogen(nuclei: &mut [Nucleus], d_channel: &Array2<f32>, mpp: f64) {
    let (h, w) = d_channel.dim();
    if h == 0 || w == 0 || mpp.is_nan() || mpp <= 0.0 {
        return;
    }
    for n in nuclei.iter_mut() {
        let x = ((n.x_um / mpp).floor().max(0.0) as usize).min(w - 1);
        let y = ((n.y_um / mpp).floor().max(0.0) as usize).min(h - 1);
        n.d_mean = disc_mean(d_channel, x, y, n.sigma_um as f64 / mpp);
    }
}

/// Densitas inti (sel/mm²) dari jumlah dan luas jaringan (µm²). Luas ≤ 0 → 0.
pub fn nuclei_density_per_mm2(count: usize, tissue_area_um2: f64) -> f64 {
    if tissue_area_um2.is_nan() || tissue_area_um2 <= 0.0 {
        return 0.0;
    }
    count as f64 / (tissue_area_um2 / 1.0e6)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn density() {
        assert!((nuclei_density_per_mm2(50, 1.0e6) - 50.0).abs() < 1e-9);
        assert!((nuclei_density_per_mm2(10, 2.5e5) - 40.0).abs() < 1e-9);
        assert_eq!(nuclei_density_per_mm2(10, 0.0), 0.0);
    }

    #[test]
    fn detects_two_synthetic_blobs() {
        let mpp = 0.5;
        let (w, h) = (80usize, 60usize);
        let centers = [(20.0f64, 15.0f64), (55.0, 40.0)];
        let img = Array2::from_shape_fn((h, w), |(y, x)| {
            let mut v = 0.05f32;
            for &(cx, cy) in &centers {
                let d2 = (x as f64 - cx).powi(2) + (y as f64 - cy).powi(2);
                if d2 <= 36.0 {
                    v = 1.0;
                }
            }
            v
        });
        let mut nuc = detect_nuclei(&img, mpp, None, &NucleiParams::default());
        assert_eq!(nuc.len(), 2, "{nuc:?}");
        assert!((nuc[0].x_um - (20.5 * mpp)).abs() <= 1.0);
        assert!((nuc[1].y_um - (40.5 * mpp)).abs() <= 1.0);
        assert!(nuc[0].h_mean > 0.8);
        let d = Array2::from_elem((h, w), 0.4f32);
        measure_chromogen(&mut nuc, &d, mpp);
        assert!((nuc[0].d_mean - 0.4).abs() < 1e-6);
    }
}
