//! Skor IHC (histo3d.md 8.12.5 butir 3–5): inti, berbasis luas, TPS/CPS,
//! Allred.

use ndarray::Array2;

use crate::nuclei::Nucleus;

/// Ambang OD kromogen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IhcThresholds {
    /// Inti/piksel positif bila `D > t_d` (bawaan 0,15 OD).
    pub t_d: f32,
    /// Batas intensitas 1/2/3 (bawaan `[0,15; 0,35; 0,60]`).
    pub intensity: [f32; 3],
}

impl Default for IhcThresholds {
    fn default() -> Self {
        Self {
            t_d: 0.15,
            intensity: [0.15, 0.35, 0.60],
        }
    }
}

impl IhcThresholds {
    /// Kelas intensitas 0..=3 dari OD.
    pub fn intensity_class(&self, d: f32) -> u8 {
        if d >= self.intensity[2] {
            3
        } else if d >= self.intensity[1] {
            2
        } else if d >= self.intensity[0] {
            1
        } else {
            0
        }
    }
}

/// Skor IHC pola inti.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NuclearScore {
    /// Jumlah inti.
    pub n_all: u32,
    /// Jumlah inti positif (`D_mean > t_d`).
    pub n_pos: u32,
    /// Persen positif (0–100).
    pub pct_pos: f32,
    /// `Σ_i i × %_i` (0–300) atas kelas intensitas 1..=3.
    pub h_score: f32,
    /// Skor proporsi Allred 0–5.
    pub allred_proportion: u8,
    /// Skor intensitas Allred 0–3 (modus kelas intensitas inti positif).
    pub allred_intensity: u8,
    /// Jumlah keduanya (0–8).
    pub allred_total: u8,
    /// Jumlah inti per kelas intensitas 0..=3.
    pub intensity_counts: [u32; 4],
}

/// Skor proporsi Allred: 0: 0 %; 1: < 1 %; 2: 1–10 %; 3: 10–33 %; 4: 33–66 %;
/// 5: > 66 %.
pub fn allred_proportion_score(pct: f32) -> u8 {
    if pct.is_nan() || pct <= 0.0 {
        0
    } else if pct < 1.0 {
        1
    } else if pct <= 10.0 {
        2
    } else if pct <= 33.0 {
        3
    } else if pct <= 66.0 {
        4
    } else {
        5
    }
}

/// Skor inti dari daftar inti yang sudah diisi `d_mean`.
pub fn score_nuclear(nuclei: &[Nucleus], thr: &IhcThresholds) -> NuclearScore {
    let mut counts = [0u32; 4];
    let mut n_pos = 0u32;
    for n in nuclei {
        let cls = thr.intensity_class(n.d_mean);
        counts[cls as usize] += 1;
        if n.d_mean > thr.t_d {
            n_pos += 1;
        }
    }
    let n_all = nuclei.len() as u32;
    let pct = |c: u32| {
        if n_all == 0 {
            0.0
        } else {
            100.0 * c as f32 / n_all as f32
        }
    };
    let pct_pos = pct(n_pos);
    let h_score = pct(counts[1]) + 2.0 * pct(counts[2]) + 3.0 * pct(counts[3]);
    // Modus intensitas inti positif (kelas 1..=3); seri → kelas lebih tinggi.
    let mut allred_intensity = 0u8;
    let mut best = 0u32;
    for (cls, &cnt) in counts.iter().enumerate().skip(1) {
        if cnt > 0 && cnt >= best {
            best = cnt;
            allred_intensity = cls as u8;
        }
    }
    let allred_proportion = allred_proportion_score(pct_pos);
    NuclearScore {
        n_all,
        n_pos,
        pct_pos,
        h_score,
        allred_proportion,
        allred_intensity,
        allred_total: allred_proportion + allred_intensity,
        intensity_counts: counts,
    }
}

/// Skor berbasis luas (pola membran/sitoplasma, versi pertama).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AreaScore {
    /// Jumlah piksel jaringan.
    pub n_tissue: u64,
    /// Jumlah piksel positif.
    pub n_pos: u64,
    /// `100 × luas positif / luas jaringan`.
    pub pct_pos: f32,
    /// Kuantil 25/50/75 % OD piksel positif (0 bila tidak ada).
    pub intensity_quantiles: [f32; 3],
}

/// Fraksi luas positif `D > t_d` di dalam `tissue_mask` (atau seluruh citra).
pub fn score_area_based(
    d_channel: &Array2<f32>,
    tissue_mask: Option<&Array2<u8>>,
    thr: &IhcThresholds,
) -> AreaScore {
    let mask_ok = tissue_mask.filter(|m| m.dim() == d_channel.dim());
    let mut n_tissue = 0u64;
    let mut pos: Vec<f32> = Vec::new();
    for (i, &d) in d_channel.iter().enumerate() {
        if let Some(m) = mask_ok {
            if m.iter().nth(i).copied().unwrap_or(0) == 0 {
                continue;
            }
        }
        n_tissue += 1;
        if d > thr.t_d {
            pos.push(d);
        }
    }
    let n_pos = pos.len() as u64;
    let pct_pos = if n_tissue == 0 {
        0.0
    } else {
        (100.0 * n_pos as f64 / n_tissue as f64) as f32
    };
    let mut q = [0.0f32; 3];
    if !pos.is_empty() {
        pos.sort_by(|a, b| a.total_cmp(b));
        for (k, p) in [0.25f64, 0.5, 0.75].iter().enumerate() {
            let idx = (((pos.len() - 1) as f64) * p).round() as usize;
            q[k] = pos[idx];
        }
    }
    AreaScore {
        n_tissue,
        n_pos,
        pct_pos,
        intensity_quantiles: q,
    }
}

/// TPS dan CPS (PD-L1), persen. `None` bila `n_tumor_all == 0`.
/// `CPS = 100 × (tumor positif + imun positif) / semua sel tumor`, dijepit
/// ≤ 100.
pub fn tps_cps(
    n_tumor_pos: u32,
    n_tumor_all: u32,
    n_immune_pos: u32,
) -> (Option<f32>, Option<f32>) {
    if n_tumor_all == 0 {
        return (None, None);
    }
    let tps = 100.0 * n_tumor_pos as f64 / n_tumor_all as f64;
    let cps = (100.0 * (n_tumor_pos as f64 + n_immune_pos as f64) / n_tumor_all as f64).min(100.0);
    (Some(tps as f32), Some(cps as f32))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nuc(d: f32) -> Nucleus {
        Nucleus {
            x_um: 0.0,
            y_um: 0.0,
            sigma_um: 3.0,
            h_mean: 1.0,
            d_mean: d,
            response: -1.0,
        }
    }

    #[test]
    fn allred_bands() {
        assert_eq!(allred_proportion_score(0.0), 0);
        assert_eq!(allred_proportion_score(0.5), 1);
        assert_eq!(allred_proportion_score(5.0), 2);
        assert_eq!(allred_proportion_score(20.0), 3);
        assert_eq!(allred_proportion_score(50.0), 4);
        assert_eq!(allred_proportion_score(80.0), 5);
    }

    #[test]
    fn nuclear_score() {
        let thr = IhcThresholds::default();
        let n: Vec<Nucleus> = [0.0, 0.05, 0.2, 0.4, 0.4, 0.7, 0.9, 0.1, 0.0, 0.0]
            .iter()
            .map(|&d| nuc(d))
            .collect();
        let s = score_nuclear(&n, &thr);
        assert_eq!(s.n_all, 10);
        assert_eq!(s.n_pos, 5);
        assert!((s.pct_pos - 50.0).abs() < 1e-5);
        assert_eq!(s.intensity_counts, [5, 1, 2, 2]);
        assert!((s.h_score - (10.0 + 40.0 + 60.0)).abs() < 1e-4);
        assert_eq!(s.allred_proportion, 4);
        assert_eq!(s.allred_intensity, 3); // seri 2 vs 3 → 3
        assert_eq!(s.allred_total, 7);
        let empty = score_nuclear(&[], &thr);
        assert_eq!(empty.pct_pos, 0.0);
        assert_eq!(empty.allred_total, 0);
    }

    #[test]
    fn area_based() {
        let d = Array2::from_shape_fn((4, 4), |(y, _)| if y < 1 { 0.5f32 } else { 0.0 });
        let m = Array2::from_shape_fn((4, 4), |(y, _)| u8::from(y < 2));
        let s = score_area_based(&d, Some(&m), &IhcThresholds::default());
        assert_eq!(s.n_tissue, 8);
        assert_eq!(s.n_pos, 4);
        assert!((s.pct_pos - 50.0).abs() < 1e-5);
        assert_eq!(s.intensity_quantiles, [0.5, 0.5, 0.5]);
    }

    #[test]
    fn tps_cps_clamp_and_none() {
        assert_eq!(tps_cps(0, 0, 10), (None, None));
        let (t, c) = tps_cps(50, 100, 80);
        assert!((t.unwrap() - 50.0).abs() < 1e-5);
        assert_eq!(c.unwrap(), 100.0);
        let (_, c2) = tps_cps(10, 100, 5);
        assert!((c2.unwrap() - 15.0).abs() < 1e-5);
    }
}
