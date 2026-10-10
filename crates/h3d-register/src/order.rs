//! Usulan urutan dan orientasi irisan (histo3d.md 8.4.7).
//!
//! # Arah transformasi
//! - Matriks U1 menyimpan `P_{j→i}` (U irisan `j` → U irisan `i`, µm → µm)
//!   dari R1+R2 pada resolusi kasar; `P_{i→j} = P_{j→i}⁻¹`.
//! - `OrderSuggestion::orient[i]` = U irisan `i` → U irisan acuan (irisan di
//!   tengah urutan usulan), dikomposisikan sepanjang urutan (aturan 5.3):
//!   `orient_i = orient_{tetangga} ∘ P_{i→tetangga}`, `orient_ref = identitas`.
//!   Ini **bukan** `T_i`; hanya tebakan awal dan thumbnail.

use std::collections::HashMap;

use h3d_core::{Affine2, Progress, SlideId, SlideStatus};
use h3d_imgproc::moments;
use rayon::prelude::*;

use crate::builtin::coarse_align;
use crate::chain::RegInput;
use crate::error::RegError;
use crate::types::RegImage;

/// Ambang keyakinan; di bawahnya irisan masuk `uncertain`.
pub const UNCERTAIN_THRESHOLD: f32 = 0.1;
/// Resolusi (µm/px) untuk matriks kemiripan U1.
pub const SIMILARITY_MPP: f64 = 64.0;
/// Di atas jumlah irisan ini, pasangan disaring dengan deskriptor bentuk.
pub const PREFILTER_ABOVE: usize = 150;
/// Jumlah kandidat per irisan setelah penyaringan.
pub const PREFILTER_CANDIDATES: usize = 20;
/// Skor pasangan yang tidak dihitung (lebih buruk dari NCC mana pun).
const UNSCORED: f32 = -1.0;

/// Usulan urutan dan orientasi.
#[derive(Clone, Debug, PartialEq)]
pub struct OrderSuggestion {
    /// Urutan usulan: irisan `Ok` menurut lintasan, lalu irisan non-`Ok`
    /// dalam urutan semula.
    pub order: Vec<SlideId>,
    /// Rigid + cermin, U irisan → U irisan acuan (tengah urutan usulan).
    pub orient: Vec<(SlideId, Affine2)>,
    /// Keyakinan 0..1 per irisan.
    pub confidence: Vec<(SlideId, f32)>,
    /// Irisan dengan keyakinan < [`UNCERTAIN_THRESHOLD`].
    pub uncertain: Vec<SlideId>,
}

/// Matriks kemiripan simetris (U1).
struct Similarity {
    n: usize,
    /// `s[i*n + j]` = skor kandidat (NCC bermasker × Dice).
    s: Vec<f32>,
    /// `p[i*n + j]` = `P_{j→i}` (U_j → U_i, µm), bila dihitung.
    p: Vec<Option<Affine2>>,
}

impl Similarity {
    fn score(&self, i: usize, j: usize) -> f32 {
        self.s[i * self.n + j]
    }
    fn dist(&self, i: usize, j: usize) -> f64 {
        1.0 - self.score(i, j) as f64
    }
    fn p_to(&self, from: usize, to: usize) -> Option<Affine2> {
        self.p[to * self.n + from]
    }
}

/// Pasangan `(i, j)`, `i < j`, yang dihitung penuh.
fn candidate_pairs(imgs: &[RegImage]) -> Vec<(usize, usize)> {
    let n = imgs.len();
    if n <= PREFILTER_ABOVE {
        return (0..n)
            .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
            .collect();
    }
    // Deskriptor: log luas (µm²) dan log rasio eigen; jarak Euklides.
    let desc: Vec<[f64; 2]> = imgs
        .iter()
        .map(|im| match moments(&im.mask) {
            Ok(m) => [
                (m.area * im.mpp * im.mpp).max(1.0).ln(),
                m.eigen_ratio.min(1e6).ln(),
            ],
            Err(_) => [0.0, 0.0],
        })
        .collect();
    let mut set: Vec<(usize, usize)> = Vec::new();
    for i in 0..n {
        let mut d: Vec<(f64, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let dx = desc[i][0] - desc[j][0];
                let dy = desc[i][1] - desc[j][1];
                ((dx * dx + dy * dy).sqrt(), j)
            })
            .collect();
        d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        for &(_, j) in d.iter().take(PREFILTER_CANDIDATES) {
            set.push((i.min(j), i.max(j)));
        }
    }
    set.sort_unstable();
    set.dedup();
    set
}

/// U1: matriks kemiripan dengan R1+R2 pada [`SIMILARITY_MPP`].
fn similarity_matrix(imgs: &[RegImage], progress: &dyn Progress) -> Result<Similarity, RegError> {
    let n = imgs.len();
    let pairs = candidate_pairs(imgs);
    let results: Vec<Result<(f32, Affine2), RegError>> = pairs
        .par_iter()
        .map(|&(i, j)| {
            if progress.is_cancelled() {
                return Err(RegError::Cancelled);
            }
            // fixed = i, moving = j → P_{j→i}.
            let c = coarse_align(&imgs[i], &imgs[j], None, true)?;
            Ok((c.score, c.p_um))
        })
        .collect();
    let mut s = vec![UNSCORED; n * n];
    let mut p = vec![None; n * n];
    for (k, r) in results.into_iter().enumerate() {
        let (score, pji) = r?;
        let (i, j) = pairs[k];
        s[i * n + j] = score;
        s[j * n + i] = score;
        p[i * n + j] = Some(pji);
        p[j * n + i] = pji.inverse();
    }
    for i in 0..n {
        s[i * n + i] = 1.0;
        p[i * n + i] = Some(Affine2::IDENTITY);
    }
    Ok(Similarity { n, s, p })
}

fn path_cost(sim: &Similarity, path: &[usize]) -> f64 {
    path.windows(2).map(|w| sim.dist(w[0], w[1])).sum()
}

/// Tetangga-terdekat rakus dari `start`; seri dipecah dengan indeks (= urutan
/// `SlideId` karena `ids` terurut).
fn greedy_path(sim: &Similarity, start: usize) -> Vec<usize> {
    let n = sim.n;
    let mut used = vec![false; n];
    let mut path = Vec::with_capacity(n);
    let mut cur = start;
    used[cur] = true;
    path.push(cur);
    for _ in 1..n {
        let mut best: Option<(f64, usize)> = None;
        for (j, &taken) in used.iter().enumerate() {
            if taken {
                continue;
            }
            let d = sim.dist(cur, j);
            if best.is_none_or(|(bd, _)| d < bd) {
                best = Some((d, j));
            }
        }
        let Some((_, j)) = best else { break };
        used[j] = true;
        path.push(j);
        cur = j;
    }
    path
}

/// 2-opt pada lintasan terbuka sampai tidak ada perbaikan (> 1e-12).
fn two_opt(sim: &Similarity, path: &mut [usize]) {
    let n = path.len();
    if n < 3 {
        return;
    }
    loop {
        let mut improved = false;
        for i in 0..n - 1 {
            for j in (i + 1)..n {
                // Balik segmen path[i..=j]. Tepi yang berubah: (i−1,i) → (i−1,j)
                // dan (j,j+1) → (i,j+1); tepi di ujung lintasan tidak ada.
                let before = i.checked_sub(1).map(|k| path[k]);
                let after = path.get(j + 1).copied();
                let old = before.map_or(0.0, |b| sim.dist(b, path[i]))
                    + after.map_or(0.0, |a| sim.dist(path[j], a));
                let new = before.map_or(0.0, |b| sim.dist(b, path[j]))
                    + after.map_or(0.0, |a| sim.dist(path[i], a));
                if new < old - 1e-12 {
                    path[i..=j].reverse();
                    improved = true;
                }
            }
        }
        if !improved {
            break;
        }
    }
}

/// U2: lintasan Hamilton terbuka berbiaya minimum (heuristik), deterministik.
fn best_path(sim: &Similarity) -> Vec<usize> {
    let n = sim.n;
    let mut best: Option<(f64, Vec<usize>)> = None;
    for start in 0..n {
        let mut p = greedy_path(sim, start);
        two_opt(sim, &mut p);
        let c = path_cost(sim, &p);
        // Bentuk kanonik agar seri antara lintasan dan kebalikannya tetap.
        if p.first() > p.last() {
            p.reverse();
        }
        let better = match &best {
            None => true,
            Some((bc, bp)) => c < bc - 1e-12 || ((c - bc).abs() <= 1e-12 && p < *bp),
        };
        if better {
            best = Some((c, p));
        }
    }
    best.map(|(_, p)| p).unwrap_or_default()
}

/// Jumlah inversi `path` relatif terhadap posisi `pos` (urutan saat ini).
fn inversions(path: &[usize], pos: &[usize]) -> usize {
    let mut inv = 0;
    for a in 0..path.len() {
        for b in (a + 1)..path.len() {
            if pos[path[a]] > pos[path[b]] {
                inv += 1;
            }
        }
    }
    inv
}

/// Usulan urutan dan orientasi (8.4.7). `current` = urutan saat ini (semua
/// irisan). Irisan non-`Ok` tidak ikut dihitung dan ditempatkan di akhir
/// `order` dalam urutan semula. Deterministik untuk masukan sama.
pub fn suggest_order(
    slides: &[RegInput],
    current: &[SlideId],
    progress: &dyn Progress,
) -> Result<OrderSuggestion, RegError> {
    // Irisan Ok, terurut menurut posisi di `current` lalu `order`, lalu id.
    let cur_pos: HashMap<SlideId, usize> =
        current.iter().enumerate().map(|(i, s)| (*s, i)).collect();
    let mut ok: Vec<&RegInput> = slides
        .iter()
        .filter(|s| s.status == SlideStatus::Ok)
        .collect();
    ok.sort_by_key(|s| {
        (
            cur_pos.get(&s.id).copied().unwrap_or(usize::MAX),
            s.order,
            s.id,
        )
    });
    ok.dedup_by_key(|s| s.id);
    let others: Vec<SlideId> = {
        let mut v: Vec<&RegInput> = slides
            .iter()
            .filter(|s| s.status != SlideStatus::Ok)
            .collect();
        v.sort_by_key(|s| {
            (
                cur_pos.get(&s.id).copied().unwrap_or(usize::MAX),
                s.order,
                s.id,
            )
        });
        v.iter().map(|s| s.id).collect()
    };
    if ok.is_empty() {
        return Err(RegError::NoUsableSlides(
            "tidak ada irisan berstatus Ok".into(),
        ));
    }
    let ids: Vec<SlideId> = ok.iter().map(|s| s.id).collect();
    let n = ids.len();
    if n == 1 {
        let mut order = ids.clone();
        order.extend(others);
        return Ok(OrderSuggestion {
            order,
            orient: vec![(ids[0], Affine2::IDENTITY)],
            confidence: vec![(ids[0], 1.0)],
            uncertain: vec![],
        });
    }

    progress.set(0.0, "matriks kemiripan");
    let imgs: Vec<RegImage> = ok
        .iter()
        .map(|s| s.image.downsample_to(SIMILARITY_MPP))
        .collect();
    for (img, id) in imgs.iter().zip(ids.iter()) {
        crate::builtin::validate(img, &id.to_string())?;
    }
    let sim = similarity_matrix(&imgs, progress)?;
    if progress.is_cancelled() {
        return Err(RegError::Cancelled);
    }

    progress.set(0.8, "lintasan urutan");
    let mut path = best_path(&sim);
    // U3: arah dengan inversi paling sedikit terhadap urutan saat ini.
    let pos: Vec<usize> = (0..n).collect(); // indeks = posisi di `current` (ok sudah terurut).
    let fwd = inversions(&path, &pos);
    let rev_path: Vec<usize> = path.iter().rev().copied().collect();
    let rev = inversions(&rev_path, &pos);
    if rev < fwd || (rev == fwd && rev_path < path) {
        path = rev_path;
    }

    // U4: orientasi menuju irisan tengah urutan usulan.
    let mid = n / 2;
    let mut orient_by_idx: Vec<Option<Affine2>> = vec![None; n];
    orient_by_idx[path[mid]] = Some(Affine2::IDENTITY);
    for k in (0..mid).rev() {
        let (i, nb) = (path[k], path[k + 1]);
        orient_by_idx[i] = orient_by_idx[nb]
            .zip(sim.p_to(i, nb))
            .map(|(on, p)| on.compose(&p));
    }
    for k in (mid + 1)..n {
        let (i, nb) = (path[k], path[k - 1]);
        orient_by_idx[i] = orient_by_idx[nb]
            .zip(sim.p_to(i, nb))
            .map(|(on, p)| on.compose(&p));
    }

    // U5: keyakinan = (skor tetangga terpilih terbaik − alternatif terbaik) / rentang.
    let mut confidence = Vec::with_capacity(n);
    let mut uncertain = Vec::new();
    for k in 0..n {
        let i = path[k];
        let chosen: Vec<usize> = [k.checked_sub(1), (k + 1 < n).then_some(k + 1)]
            .into_iter()
            .flatten()
            .map(|q| path[q])
            .collect();
        let best_chosen = chosen
            .iter()
            .map(|&j| sim.score(i, j))
            .fold(f32::NEG_INFINITY, f32::max);
        let best_alt = (0..n)
            .filter(|&j| j != i && !chosen.contains(&j))
            .map(|j| sim.score(i, j))
            .fold(f32::NEG_INFINITY, f32::max);
        // Dinormalkan dengan rentang skor irisan ini (terpilih − terendah).
        let worst = (0..n)
            .filter(|&j| j != i)
            .map(|j| sim.score(i, j))
            .fold(f32::INFINITY, f32::min);
        let span = best_chosen - worst;
        let conf = if !best_alt.is_finite() {
            1.0
        } else if span > 1e-6 {
            ((best_chosen - best_alt) / span).clamp(0.0, 1.0)
        } else {
            0.0
        };
        confidence.push((ids[i], conf));
        if conf < UNCERTAIN_THRESHOLD {
            uncertain.push(ids[i]);
        }
    }

    let mut order: Vec<SlideId> = path.iter().map(|&i| ids[i]).collect();
    order.extend(others);
    let orient: Vec<(SlideId, Affine2)> = path
        .iter()
        .map(|&i| (ids[i], orient_by_idx[i].unwrap_or(Affine2::IDENTITY)))
        .collect();
    progress.set(1.0, "selesai");
    Ok(OrderSuggestion {
        order,
        orient,
        confidence,
        uncertain,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sim_from(d: &[&[f32]]) -> Similarity {
        let n = d.len();
        let mut s = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                s[i * n + j] = d[i][j];
            }
        }
        Similarity {
            n,
            s,
            p: vec![Some(Affine2::IDENTITY); n * n],
        }
    }

    #[test]
    fn path_recovers_linear_chain() {
        // Kemiripan menurun dengan jarak |i−j| pada urutan asli 0..5; acak label.
        let n = 6;
        let perm = [3usize, 0, 5, 1, 4, 2];
        let mut rows: Vec<Vec<f32>> = vec![vec![0.0; n]; n];
        for a in 0..n {
            for b in 0..n {
                let d = (perm[a] as i64 - perm[b] as i64).unsigned_abs() as f32;
                rows[a][b] = 1.0 - 0.15 * d;
            }
        }
        let refs: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();
        let sim = sim_from(&refs);
        let path = best_path(&sim);
        let seq: Vec<usize> = path.iter().map(|&i| perm[i]).collect();
        assert!(seq == (0..n).collect::<Vec<_>>() || seq == (0..n).rev().collect::<Vec<_>>());
    }
}
