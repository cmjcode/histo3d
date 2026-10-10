//! Komponen terhubung (dua lintasan + union-find, konektivitas 8).

use ndarray::Array2;

/// Konektivitas tetangga.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Connectivity {
    /// Tetangga sisi (4).
    Four,
    /// Tetangga sisi dan sudut (8).
    Eight,
}

struct UnionFind {
    parent: Vec<u32>,
}

impl UnionFind {
    fn new() -> Self {
        Self { parent: vec![0] }
    }
    fn make(&mut self) -> u32 {
        let id = self.parent.len() as u32;
        self.parent.push(id);
        id
    }
    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            let p = self.parent[x as usize];
            self.parent[x as usize] = self.parent[p as usize];
            x = p;
        }
        x
    }
    /// Satukan; akar yang lebih kecil menjadi induk (deterministik).
    fn union(&mut self, a: u32, b: u32) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
        self.parent[hi as usize] = lo;
    }
}

/// Label komponen dengan konektivitas pilihan. Label 1..=n diberikan menurut
/// urutan raster kemunculan pertama tiap komponen; 0 = latar.
pub fn label_components_conn(mask: &Array2<u8>, conn: Connectivity) -> (Array2<u32>, usize) {
    let (h, w) = mask.dim();
    let mut labels = Array2::<u32>::zeros((h, w));
    if h == 0 || w == 0 {
        return (labels, 0);
    }
    let mut uf = UnionFind::new();
    for y in 0..h {
        for x in 0..w {
            if mask[[y, x]] == 0 {
                continue;
            }
            let mut cur = 0u32;
            let mut consider = |l: u32, uf: &mut UnionFind| {
                if l == 0 {
                    return;
                }
                if cur == 0 {
                    cur = l;
                } else {
                    uf.union(cur, l);
                }
            };
            if x > 0 {
                consider(labels[[y, x - 1]], &mut uf);
            }
            if y > 0 {
                consider(labels[[y - 1, x]], &mut uf);
                if conn == Connectivity::Eight {
                    if x > 0 {
                        consider(labels[[y - 1, x - 1]], &mut uf);
                    }
                    if x + 1 < w {
                        consider(labels[[y - 1, x + 1]], &mut uf);
                    }
                }
            }
            if cur == 0 {
                cur = uf.make();
            }
            labels[[y, x]] = cur;
        }
    }
    // Lintasan kedua: petakan akar ke label padat menurut kemunculan pertama.
    let mut remap = vec![0u32; uf.parent.len()];
    let mut next = 0u32;
    for v in labels.iter_mut() {
        if *v == 0 {
            continue;
        }
        let r = uf.find(*v);
        if remap[r as usize] == 0 {
            next += 1;
            remap[r as usize] = next;
        }
        *v = remap[r as usize];
    }
    (labels, next as usize)
}

/// Komponen terhubung konektivitas 8. Lihat [`label_components_conn`].
pub fn label_components(mask: &Array2<u8>) -> (Array2<u32>, usize) {
    label_components_conn(mask, Connectivity::Eight)
}

/// Luas (jumlah piksel) tiap label; indeks 0 = latar, panjang `n + 1`.
pub fn component_areas(labels: &Array2<u32>, n: usize) -> Vec<usize> {
    let mut areas = vec![0usize; n + 1];
    for &l in labels.iter() {
        if (l as usize) < areas.len() {
            areas[l as usize] += 1;
        }
    }
    areas
}

/// Buang komponen (konektivitas 8) yang luasnya `< min_area` piksel.
pub fn remove_small_components(mask: &Array2<u8>, min_area: usize) -> Array2<u8> {
    let (labels, n) = label_components(mask);
    let areas = component_areas(&labels, n);
    let mut out = Array2::<u8>::zeros(mask.raw_dim());
    for (o, &l) in out.iter_mut().zip(labels.iter()) {
        if l != 0 && areas[l as usize] >= min_area {
            *o = 1;
        }
    }
    out
}

/// Komponen terbesar saja (seri → label terkecil). Kosong bila tidak ada.
pub fn largest_component(mask: &Array2<u8>) -> Array2<u8> {
    let (labels, n) = label_components(mask);
    let mut out = Array2::<u8>::zeros(mask.raw_dim());
    if n == 0 {
        return out;
    }
    let areas = component_areas(&labels, n);
    let mut best = 1usize;
    for l in 2..=n {
        if areas[l] > areas[best] {
            best = l;
        }
    }
    for (o, &l) in out.iter_mut().zip(labels.iter()) {
        if l as usize == best {
            *o = 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn counts_known_pattern() {
        // Dua komponen 8-terhubung (diagonal menyatu), tiga bila 4-terhubung.
        let m = arr2(&[
            [1u8, 0, 0, 0, 1],
            [0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
        ]);
        let (l8, n8) = label_components(&m);
        assert_eq!(n8, 3);
        assert_eq!(l8[[0, 0]], 1);
        assert_eq!(l8[[1, 1]], 1);
        assert_eq!(l8[[0, 4]], 2);
        assert_eq!(l8[[3, 0]], 3);
        let (_, n4) = label_components_conn(&m, Connectivity::Four);
        assert_eq!(n4, 4);
    }

    #[test]
    fn areas_and_filters() {
        let m = arr2(&[[1u8, 1, 0, 1], [1, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]]);
        let (l, n) = label_components(&m);
        assert_eq!(component_areas(&l, n), vec![10, 4, 1, 1]);
        let big = remove_small_components(&m, 2);
        assert_eq!(big.iter().filter(|v| **v != 0).count(), 4);
        let largest = largest_component(&m);
        assert_eq!(largest, big);
    }

    #[test]
    fn u_shape_merges() {
        // Bentuk U: label kiri dan kanan bertemu di bawah → satu komponen.
        let m = arr2(&[[1u8, 0, 1], [1, 0, 1], [1, 1, 1]]);
        let (_, n) = label_components(&m);
        assert_eq!(n, 1);
    }
}
