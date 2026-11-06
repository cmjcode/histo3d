//! Pemisah objek yang bersentuhan: watershed berpenanda di atas medan jarak
//! (EDT) sebuah masker biner. Dipakai `h3d-volume::track` agar kelenjar/sel
//! yang menempel pada satu irisan tidak menjadi satu komponen.
//!
//! # Cara kerja
//! 1. EDT ke latar dihitung untuk tiap piksel masker (µm).
//! 2. Piksel masker diproses dari EDT tertinggi ke terendah (seri → indeks
//!    linear terkecil), sehingga banjir berjalan dari "puncak" ke "lembah"
//!    (sama dengan watershed atas `−EDT`), konektivitas 8, hanya di dalam
//!    masker.
//! 3. Piksel tanpa tetangga berlabel memulai wilayah baru dengan puncak di
//!    piksel itu. Piksel yang menyentuh ≥ 2 wilayah menjadi sadel: wilayah
//!    berpuncak lebih rendah **digabung** ke wilayah berpuncak lebih tinggi
//!    bila kedalaman lembahnya `puncak − EDT(sadel) < h_um` (h-maxima) atau
//!    jarak kedua puncak `< min_seed_distance_um`; selain itu keduanya tetap
//!    terpisah. Puncak wilayah yang bertahan = penanda.
//! 4. Banjir prioritas kedua dari penanda (EDT turun, seri → urutan masuk
//!    antrean, konektivitas 8, di dalam masker): tiap piksel mengambil label
//!    tetangga yang pertama mencapainya, sehingga batas jatuh di lembah EDT
//!    dan dataran EDT dibagi dari kedua sisi.
//! 5. Wilayah dinomori ulang 1..=n menurut urutan raster piksel pertamanya.
//!
//! Komponen yang hanya punya satu puncak bertahan tidak pernah dibelah.
//! Piksel di tepi larik yang tidak bersebelahan dengan latar dianggap jauh
//! dari latar (EDT tidak melihat luar larik). Deterministik: tanpa acak,
//! urutan pemrosesan tetap.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use ndarray::Array2;

use crate::edt::edt_2d_um;

struct Regions {
    parent: Vec<u32>,
    /// Indeks linear piksel puncak per wilayah.
    peak_idx: Vec<usize>,
    /// Nilai EDT puncak per wilayah.
    peak_val: Vec<f32>,
}

impl Regions {
    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            let p = self.parent[x as usize];
            self.parent[x as usize] = self.parent[p as usize];
            x = p;
        }
        x
    }
}

/// Bagi masker biner (`(h, w)`, ≠ 0 = isi) menjadi wilayah objek:
/// mengembalikan `(label per piksel 1..=n, n)`; 0 = latar. `s_xy_um` =
/// ukuran piksel. `min_seed_distance_um` = jarak minimum antar puncak,
/// `h_um` = kedalaman lembah minimum agar dua puncak dianggap dua objek;
/// nilai tidak terhingga/negatif diperlakukan sebagai 0.
pub fn split_touching(
    mask: &Array2<u8>,
    s_xy_um: f64,
    min_seed_distance_um: f64,
    h_um: f64,
) -> (Array2<u32>, usize) {
    let (h, w) = mask.dim();
    let mut labels = Array2::<u32>::zeros((h, w));
    if h == 0 || w == 0 {
        return (labels, 0);
    }
    let s = if s_xy_um.is_finite() && s_xy_um > 0.0 {
        s_xy_um
    } else {
        1.0
    };
    let h_min = if h_um.is_finite() && h_um > 0.0 {
        h_um as f32
    } else {
        0.0
    };
    let d_min2 = if min_seed_distance_um.is_finite() && min_seed_distance_um > 0.0 {
        (min_seed_distance_um / s).powi(2)
    } else {
        0.0
    };
    // EDT ke latar: EDT masker terbalik.
    let inv = mask.mapv(|v| u8::from(v == 0));
    let edt = edt_2d_um(&inv, s);
    let edt: Vec<f32> = edt.iter().copied().collect();
    let inside: Vec<bool> = mask.iter().map(|&v| v != 0).collect();

    let mut order: Vec<usize> = (0..h * w).filter(|&i| inside[i]).collect();
    order.sort_by(|&a, &b| edt[b].total_cmp(&edt[a]).then(a.cmp(&b)));

    // Label sementara per piksel: 0 = belum, selain itu indeks wilayah + 1.
    let mut lab = vec![0u32; h * w];
    let mut reg = Regions {
        parent: Vec::new(),
        peak_idx: Vec::new(),
        peak_val: Vec::new(),
    };
    let mut roots: Vec<u32> = Vec::with_capacity(8);
    for &p in &order {
        let (y, x) = (p / w, p % w);
        roots.clear();
        for dy in -1i64..=1 {
            for dx in -1i64..=1 {
                if dy == 0 && dx == 0 {
                    continue;
                }
                let ny = y as i64 + dy;
                let nx = x as i64 + dx;
                if ny < 0 || nx < 0 || ny >= h as i64 || nx >= w as i64 {
                    continue;
                }
                let q = ny as usize * w + nx as usize;
                if lab[q] == 0 {
                    continue;
                }
                let r = reg.find(lab[q] - 1);
                if !roots.contains(&r) {
                    roots.push(r);
                }
            }
        }
        if roots.is_empty() {
            let id = reg.parent.len() as u32;
            reg.parent.push(id);
            reg.peak_idx.push(p);
            reg.peak_val.push(edt[p]);
            lab[p] = id + 1;
            continue;
        }
        // Wilayah berpuncak tertinggi (seri → indeks puncak terkecil) menang.
        roots.sort_by(|&a, &b| {
            reg.peak_val[b as usize]
                .total_cmp(&reg.peak_val[a as usize])
                .then(reg.peak_idx[a as usize].cmp(&reg.peak_idx[b as usize]))
        });
        let win = roots[0];
        let (wy, wx) = (
            reg.peak_idx[win as usize] / w,
            reg.peak_idx[win as usize] % w,
        );
        for &r in &roots[1..] {
            let dyn_r = reg.peak_val[r as usize] - edt[p];
            let (ry, rx) = (reg.peak_idx[r as usize] / w, reg.peak_idx[r as usize] % w);
            let d2 = (ry as f64 - wy as f64).powi(2) + (rx as f64 - wx as f64).powi(2);
            if dyn_r < h_min || d2 < d_min2 {
                reg.parent[r as usize] = win;
            }
        }
        lab[p] = win + 1;
    }

    // Pass 2: banjir prioritas dari puncak wilayah yang bertahan; tiap
    // piksel mengambil label tetangga yang pertama mencapainya (urutan pop
    // deterministik: EDT turun, seri → indeks terkecil), sehingga batas
    // antar objek jatuh di lembah, bukan merayap mengikuti puncak tertinggi.
    let mut seeds: Vec<usize> = Vec::new();
    for r in 0..reg.parent.len() as u32 {
        if reg.find(r) == r {
            seeds.push(reg.peak_idx[r as usize]);
        }
    }
    seeds.sort_unstable();
    let mut out = vec![0u32; h * w];
    // Kunci heap: (−EDT, nomor urut masuk, piksel, label). Seri EDT (dataran)
    // dipecah oleh urutan masuk (FIFO), bukan indeks piksel, supaya dua front
    // yang bertemu di dataran bertemu di tengah, bukan disapu dari satu sisi.
    let mut heap: BinaryHeap<Reverse<(OrdF32, u64, usize, u32)>> = BinaryHeap::new();
    let mut seq: u64 = 0;
    for (k, &p) in seeds.iter().enumerate() {
        heap.push(Reverse((OrdF32(-edt[p]), seq, p, k as u32 + 1)));
        seq += 1;
    }
    while let Some(Reverse((_, _, p, l))) = heap.pop() {
        if out[p] != 0 {
            continue;
        }
        out[p] = l;
        let (y, x) = (p / w, p % w);
        for dy in -1i64..=1 {
            for dx in -1i64..=1 {
                if dy == 0 && dx == 0 {
                    continue;
                }
                let ny = y as i64 + dy;
                let nx = x as i64 + dx;
                if ny < 0 || nx < 0 || ny >= h as i64 || nx >= w as i64 {
                    continue;
                }
                let q = ny as usize * w + nx as usize;
                if inside[q] && out[q] == 0 {
                    heap.push(Reverse((OrdF32(-edt[q]), seq, q, l)));
                    seq += 1;
                }
            }
        }
    }

    // Penomoran ulang menurut urutan raster.
    let mut remap: Vec<u32> = vec![0; seeds.len() + 1];
    let mut n = 0usize;
    for (i, &l) in out.iter().enumerate() {
        if l == 0 {
            continue;
        }
        if remap[l as usize] == 0 {
            n += 1;
            remap[l as usize] = n as u32;
        }
        labels[[i / w, i % w]] = remap[l as usize];
    }
    (labels, n)
}

/// `f32` terurut total untuk kunci heap (nilai selalu terhingga atau +∞).
#[derive(Clone, Copy, Debug, PartialEq)]
struct OrdF32(f32);

impl Eq for OrdF32 {}

impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn disc_pair(gap_px: f64, r: f64) -> Array2<u8> {
        let (h, w) = (64usize, 128usize);
        let cy = h as f64 / 2.0;
        let c1 = w as f64 / 2.0 - r - gap_px / 2.0;
        let c2 = w as f64 / 2.0 + r + gap_px / 2.0;
        Array2::from_shape_fn((h, w), |(y, x)| {
            let (xf, yf) = (x as f64 + 0.5, y as f64 + 0.5);
            let d1 = (xf - c1).powi(2) + (yf - cy).powi(2);
            let d2 = (xf - c2).powi(2) + (yf - cy).powi(2);
            u8::from(d1 <= r * r || d2 <= r * r)
        })
    }

    #[test]
    fn two_touching_discs_are_split_into_two_equal_regions() {
        // Dua cakram r = 20 px yang saling tumpang 8 px.
        let m = disc_pair(-8.0, 20.0);
        let (lab, n) = split_touching(&m, 1.0, 3.0, 1.0);
        assert_eq!(n, 2);
        let a = lab.iter().filter(|&&v| v == 1).count() as f64;
        let b = lab.iter().filter(|&&v| v == 2).count() as f64;
        assert!((a - b).abs() / a.max(b) < 0.05, "{a} vs {b}");
        let total = m.iter().filter(|&&v| v != 0).count();
        assert_eq!(lab.iter().filter(|&&v| v != 0).count(), total);
        // Wilayah 1 di kiri (urutan raster).
        let x1 = lab
            .indexed_iter()
            .filter(|(_, &v)| v == 1)
            .map(|((_, x), _)| x as f64)
            .sum::<f64>()
            / a;
        let x2 = lab
            .indexed_iter()
            .filter(|(_, &v)| v == 2)
            .map(|((_, x), _)| x as f64)
            .sum::<f64>()
            / b;
        assert!(x1 < x2);
    }

    #[test]
    fn single_disc_and_elongated_ellipse_stay_whole() {
        let (h, w) = (64usize, 128usize);
        let disc = Array2::from_shape_fn((h, w), |(y, x)| {
            u8::from((x as f64 - 64.0).powi(2) + (y as f64 - 32.0).powi(2) <= 20.0 * 20.0)
        });
        let (_, n) = split_touching(&disc, 1.0, 3.0, 1.0);
        assert_eq!(n, 1);
        let ell = Array2::from_shape_fn((h, w), |(y, x)| {
            u8::from(((x as f64 - 64.0) / 50.0).powi(2) + ((y as f64 - 32.0) / 12.0).powi(2) <= 1.0)
        });
        let (_, n) = split_touching(&ell, 1.0, 3.0, 1.0);
        assert_eq!(n, 1, "elips memanjang tidak boleh dibelah");
    }

    #[test]
    fn separate_discs_and_empty_mask_and_determinism() {
        let m = disc_pair(6.0, 15.0);
        let (lab, n) = split_touching(&m, 2.0, 40.0, 8.0);
        assert_eq!(n, 2);
        let (lab2, n2) = split_touching(&m, 2.0, 40.0, 8.0);
        assert_eq!((lab, n), (lab2, n2));
        let empty = Array2::<u8>::zeros((10, 10));
        assert_eq!(split_touching(&empty, 1.0, 1.0, 1.0).1, 0);
        let (lab0, n0) = split_touching(&Array2::<u8>::zeros((0, 5)), 1.0, 1.0, 1.0);
        assert_eq!((lab0.dim(), n0), ((0, 5), 0));
    }

    #[test]
    fn large_h_merges_touching_discs() {
        let m = disc_pair(-8.0, 20.0);
        let (_, n) = split_touching(&m, 1.0, 3.0, 100.0);
        assert_eq!(n, 1);
    }
}
