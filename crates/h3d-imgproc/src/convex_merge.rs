//! Penggabungan wilayah watershed berdasarkan konveksitas (deteksi sel,
//! histo3d.md 8.12.5). Watershed berpenanda cenderung menanam beberapa benih
//! dalam satu sel besar sehingga sel terpotong menjadi beberapa keping
//! cembung. Dua wilayah bertetangga digabung bila gabungannya tetap cembung:
//! soliditas `luas / luas selubung cembung` ≥ ambang dan luasnya ≤ batas.
//! Dua inti bulat yang sekadar bersentuhan membentuk "angka 8" dengan
//! soliditas ≈ 0,88 sehingga tetap terpisah pada ambang bawaan 0,93.
//!
//! # Konvensi
//! - Selubung cembung dihitung atas **sudut piksel** (piksel `(x, y)`
//!   menempati `[x, x+1] × [y, y+1]`), sehingga satu piksel berselubung
//!   luas 1 dan soliditas bentuk persegi panjang = 1.
//! - Rakus dan deterministik: pasangan dengan soliditas gabungan tertinggi
//!   digabung lebih dulu; seri diputus dengan id terkecil. Hasil dinomori
//!   ulang 1..=n menurut id akar menaik.

use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap};

use ndarray::Array2;

type Pt = (i64, i64);

/// Gabungkan wilayah bertetangga (konektivitas 4) pada `labels` (`0` =
/// latar, `1..=n_labels`) selama gabungannya bersoliditas ≥ `min_solidity`
/// dan berluas ≤ `max_area_px` piksel. Mengembalikan jumlah wilayah baru;
/// `labels` ditulis ulang dengan nomor 1..=n. `min_solidity` di luar
/// `(0, 1]` atau tidak hingga mematikan penggabungan (label dipadatkan saja).
pub fn merge_convex_regions(
    labels: &mut Array2<u32>,
    n_labels: usize,
    max_area_px: usize,
    min_solidity: f64,
) -> usize {
    let (h, w) = labels.dim();
    let n = n_labels;
    let mut area = vec![0usize; n + 1];
    let mut pts: Vec<Vec<Pt>> = vec![Vec::new(); n + 1];
    let mut adj: Vec<BTreeSet<u32>> = vec![BTreeSet::new(); n + 1];
    for y in 0..h {
        for x in 0..w {
            let l = labels[[y, x]];
            if l == 0 || l as usize > n {
                continue;
            }
            let i = l as usize;
            area[i] += 1;
            let (xi, yi) = (x as i64, y as i64);
            // Sudut piksel ujung kiri/kanan tiap run cukup untuk selubung.
            if x == 0 || labels[[y, x - 1]] != l {
                pts[i].extend([(xi, yi), (xi, yi + 1)]);
            }
            if x + 1 == w || labels[[y, x + 1]] != l {
                pts[i].extend([(xi + 1, yi), (xi + 1, yi + 1)]);
            }
            for (nx, ny) in [(x + 1, y), (x, y + 1)] {
                if nx < w && ny < h {
                    let m = labels[[ny, nx]];
                    if m != 0 && m != l && (m as usize) <= n {
                        adj[i].insert(m);
                        adj[m as usize].insert(l);
                    }
                }
            }
        }
    }
    let mut hull: Vec<Vec<Pt>> = pts.into_iter().map(convex_hull).collect();
    let mut parent: Vec<u32> = (0..=n as u32).collect();
    let enabled = min_solidity.is_finite() && min_solidity > 0.0 && min_solidity <= 1.0;
    if enabled {
        let mut ver = vec![0u32; n + 1];
        let mut heap = BinaryHeap::new();
        let score = |a: usize, b: usize, area: &[usize], hull: &[Vec<Pt>]| -> Option<f64> {
            let tot = area[a] + area[b];
            if tot > max_area_px {
                return None;
            }
            let mut u = hull[a].clone();
            u.extend_from_slice(&hull[b]);
            let ha = hull_area(&convex_hull(u));
            let s = if ha > 0.0 { tot as f64 / ha } else { 0.0 };
            (s >= min_solidity).then_some(s)
        };
        for (a, nb) in adj.iter().enumerate().skip(1) {
            for &b in nb.range(a as u32 + 1..) {
                if let Some(s) = score(a, b as usize, &area, &hull) {
                    heap.push((s.to_bits(), Reverse(a as u32), Reverse(b), 0u32, 0u32));
                }
            }
        }
        while let Some((_, Reverse(a), Reverse(b), va, vb)) = heap.pop() {
            let (a, b) = (a as usize, b as usize);
            if parent[a] != a as u32 || parent[b] != b as u32 || ver[a] != va || ver[b] != vb {
                continue;
            }
            // Gabung b ke a (a < b); a mewarisi luas, selubung, tetangga.
            parent[b] = a as u32;
            area[a] += area[b];
            let mut u = std::mem::take(&mut hull[a]);
            u.append(&mut hull[b]);
            hull[a] = convex_hull(u);
            let nb = std::mem::take(&mut adj[b]);
            for &c in &nb {
                let c = c as usize;
                adj[c].remove(&(b as u32));
                if c != a {
                    adj[c].insert(a as u32);
                    adj[a].insert(c as u32);
                }
            }
            adj[a].remove(&(b as u32));
            adj[a].remove(&(a as u32));
            ver[a] += 1;
            let neigh: Vec<u32> = adj[a].iter().copied().collect();
            for c in neigh {
                let c = c as usize;
                if let Some(s) = score(a, c, &area, &hull) {
                    let (lo, hi) = (a.min(c), a.max(c));
                    heap.push((
                        s.to_bits(),
                        Reverse(lo as u32),
                        Reverse(hi as u32),
                        ver[lo],
                        ver[hi],
                    ));
                }
            }
        }
    }
    // Akar tiap label, lalu nomori ulang 1..=n menurut akar menaik.
    let find = |mut i: usize, parent: &[u32]| {
        while parent[i] as usize != i {
            i = parent[i] as usize;
        }
        i
    };
    let mut new_id = vec![0u32; n + 1];
    let mut next = 0u32;
    for i in 1..=n {
        if parent[i] as usize == i && area[i] > 0 {
            next += 1;
            new_id[i] = next;
        }
    }
    let map: Vec<u32> = (0..=n)
        .map(|i| if i == 0 { 0 } else { new_id[find(i, &parent)] })
        .collect();
    for v in labels.iter_mut() {
        if (*v as usize) <= n {
            *v = map[*v as usize];
        }
    }
    next as usize
}

/// Selubung cembung (rantai monoton Andrew), urutan berlawanan arah jarum
/// jam matematis, tanpa titik kolinear.
fn convex_hull(mut p: Vec<Pt>) -> Vec<Pt> {
    p.sort_unstable();
    p.dedup();
    if p.len() < 3 {
        return p;
    }
    let cross = |o: Pt, a: Pt, b: Pt| (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0);
    let mut lower: Vec<Pt> = Vec::with_capacity(p.len());
    for &q in &p {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], q) <= 0 {
            lower.pop();
        }
        lower.push(q);
    }
    let mut upper: Vec<Pt> = Vec::with_capacity(p.len());
    for &q in p.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], q) <= 0 {
            upper.pop();
        }
        upper.push(q);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// Luas poligon cembung (rumus tali sepatu, nilai mutlak).
fn hull_area(p: &[Pt]) -> f64 {
    let n = p.len();
    if n < 3 {
        return 0.0;
    }
    let s: i64 = (0..n)
        .map(|i| {
            let (a, b) = (p[i], p[(i + 1) % n]);
            a.0 * b.1 - b.0 * a.1
        })
        .sum();
    s.unsigned_abs() as f64 * 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    fn disc(a: &mut Array2<u32>, cx: f64, cy: f64, r: f64, f: impl Fn(f64, f64) -> u32) {
        for ((y, x), v) in a.indexed_iter_mut() {
            let dx = x as f64 + 0.5 - cx;
            let dy = y as f64 + 0.5 - cy;
            if dx * dx + dy * dy <= r * r {
                *v = f(dx, dy);
            }
        }
    }

    #[test]
    fn hull_of_pixel_square_has_unit_solidity() {
        let p = vec![(0, 0), (3, 0), (3, 2), (0, 2), (1, 1)];
        let h = convex_hull(p);
        assert_eq!(h.len(), 4);
        assert!((hull_area(&h) - 6.0).abs() < 1e-12);
    }

    #[test]
    fn disc_cut_into_quadrants_merges_back() {
        let mut a = Array2::<u32>::zeros((40, 40));
        disc(&mut a, 20.0, 20.0, 12.0, |dx, dy| {
            match (dx < 0.0, dy < 0.0) {
                (true, true) => 1,
                (false, true) => 2,
                (true, false) => 3,
                (false, false) => 4,
            }
        });
        let n = merge_convex_regions(&mut a, 4, 10_000, 0.93);
        assert_eq!(n, 1);
        assert!(a.iter().all(|&v| v <= 1));
    }

    #[test]
    fn touching_round_nuclei_stay_separate() {
        let mut a = Array2::<u32>::zeros((30, 50));
        disc(&mut a, 15.0, 15.0, 8.0, |_, _| 1);
        disc(&mut a, 31.0, 15.0, 8.0, |_, _| 2);
        assert_eq!(merge_convex_regions(&mut a, 2, 10_000, 0.93), 2);
    }

    #[test]
    fn l_shape_and_area_limit_block_merge() {
        // L: dua persegi panjang, gabungan soliditas 0,75.
        let mut a = Array2::<u32>::zeros((12, 12));
        for y in 1..3 {
            for x in 1..9 {
                a[[y, x]] = 1;
            }
        }
        for y in 3..9 {
            for x in 1..3 {
                a[[y, x]] = 2;
            }
        }
        assert_eq!(merge_convex_regions(&mut a.clone(), 2, 10_000, 0.93), 2);
        // Persegi dibelah dua: cembung, tetapi luas gabungan > batas.
        let mut b = Array2::<u32>::zeros((10, 10));
        for y in 0..6 {
            for x in 0..6 {
                b[[y, x]] = if x < 3 { 1 } else { 2 };
            }
        }
        assert_eq!(merge_convex_regions(&mut b.clone(), 2, 35, 0.93), 2);
        assert_eq!(merge_convex_regions(&mut b, 2, 36, 0.93), 1);
    }

    #[test]
    fn disabled_only_compacts_and_is_deterministic() {
        let mut a = Array2::<u32>::zeros((6, 6));
        a[[0, 0]] = 3;
        a[[5, 5]] = 7;
        assert_eq!(merge_convex_regions(&mut a, 7, 100, 0.0), 2);
        assert_eq!((a[[0, 0]], a[[5, 5]]), (1, 2));
        let mut q = Array2::<u32>::zeros((40, 40));
        disc(&mut q, 20.0, 20.0, 12.0, |dx, dy| {
            1 + u32::from(dx > 0.0) + 2 * u32::from(dy > 0.0)
        });
        let mut q2 = q.clone();
        merge_convex_regions(&mut q, 4, 10_000, 0.93);
        merge_convex_regions(&mut q2, 4, 10_000, 0.93);
        assert_eq!(q, q2);
    }
}
