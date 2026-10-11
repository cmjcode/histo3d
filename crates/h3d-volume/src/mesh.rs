//! Mesh label (histo3d.md 8.6.3) dan operasinya: ukuran, topologi,
//! penghalusan Taubin, decimate QEM, las verteks, proyeksi medan.
//!
//! Konvensi: posisi µm `f64`; segitiga berlawanan arah jarum jam dilihat
//! dari luar sehingga volume bertanda positif; normal menghadap ke luar.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use h3d_core::LabelId;
use serde::{Deserialize, Serialize};

use crate::grid::StackGrid;

/// Skalar per verteks (jarak margin, hipoksia, atribut; 8.12.6).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeshScalars {
    /// Nama medan, mis. "margin_distance".
    pub name: String,
    /// Satuan, mis. "µm".
    pub unit: String,
    /// Nilai per verteks; `NaN` = NODATA.
    pub values: Vec<f32>,
}

/// Mesh segitiga satu label.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LabelMesh {
    /// Label asal.
    pub label: LabelId,
    /// Posisi verteks, µm, ruang W.
    pub positions_um: Vec<[f64; 3]>,
    /// Normal per verteks (satuan), menghadap ke luar.
    pub normals: Vec<[f32; 3]>,
    /// Indeks segitiga (3 per segitiga).
    pub indices: Vec<u32>,
    /// Skalar per verteks.
    pub scalars: Vec<MeshScalars>,
}

#[inline]
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
#[inline]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}
#[inline]
fn dist2(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = sub(a, b);
    dot(d, d)
}

/// Jarak terjauh antar pasangan titik, O(n²); deterministik.
fn farthest_pair(pts: &[[f64; 3]]) -> f64 {
    let mut best = 0.0f64;
    for i in 0..pts.len() {
        for j in (i + 1)..pts.len() {
            let d = dist2(pts[i], pts[j]);
            if d > best {
                best = d;
            }
        }
    }
    best.sqrt()
}

/// Diameter maksimum sekumpulan titik: selubung cembung (parry3d) lalu
/// pasangan terjauh antar verteks selubung. Bila selubung gagal (titik
/// koplanar/kolinear), memakai pasangan terjauh atas semua titik.
pub fn max_diameter_of_points(pts: &[[f64; 3]]) -> f64 {
    if pts.len() < 2 {
        return 0.0;
    }
    if pts.len() <= 64 {
        return farthest_pair(pts);
    }
    // Normalisasi ke pusat agar presisi f32 selubung cukup.
    let mut c = [0.0f64; 3];
    for p in pts {
        c[0] += p[0];
        c[1] += p[1];
        c[2] += p[2];
    }
    let n = pts.len() as f64;
    c = [c[0] / n, c[1] / n, c[2] / n];
    let v: Vec<parry3d::math::Vector3> = pts
        .iter()
        .map(|p| {
            parry3d::math::Vector3::new(
                (p[0] - c[0]) as f32,
                (p[1] - c[1]) as f32,
                (p[2] - c[2]) as f32,
            )
        })
        .collect();
    match parry3d::transformation::try_convex_hull(&v) {
        Ok((hull, _)) if hull.len() >= 2 => {
            let hp: Vec<[f64; 3]> = hull
                .iter()
                .map(|q| [q.x as f64 + c[0], q.y as f64 + c[1], q.z as f64 + c[2]])
                .collect();
            farthest_pair(&hp)
        }
        _ => farthest_pair(pts),
    }
}

impl LabelMesh {
    /// Mesh kosong untuk `label`.
    pub fn empty(label: LabelId) -> Self {
        LabelMesh {
            label,
            positions_um: Vec::new(),
            normals: Vec::new(),
            indices: Vec::new(),
            scalars: Vec::new(),
        }
    }

    /// Jumlah segitiga.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Iterasi segitiga sebagai `[a, b, c]` indeks.
    fn triangles(&self) -> impl Iterator<Item = [u32; 3]> + '_ {
        self.indices.chunks_exact(3).map(|t| [t[0], t[1], t[2]])
    }

    /// Volume bertanda: `(1/6) Σ a · (b × c)`; positif bila segitiga
    /// berorientasi ke luar (8.6.4).
    pub fn signed_volume_um3(&self) -> f64 {
        let p = &self.positions_um;
        let mut v = 0.0;
        for t in self.triangles() {
            let (a, b, c) = (p[t[0] as usize], p[t[1] as usize], p[t[2] as usize]);
            v += dot(a, cross(b, c));
        }
        v / 6.0
    }

    /// Jumlah luas segitiga.
    pub fn surface_area_um2(&self) -> f64 {
        let p = &self.positions_um;
        let mut s = 0.0;
        for t in self.triangles() {
            let (a, b, c) = (p[t[0] as usize], p[t[1] as usize], p[t[2] as usize]);
            s += norm(cross(sub(b, a), sub(c, a))) * 0.5;
        }
        s
    }

    /// Kotak pembatas `(min, max)`; `None` bila kosong.
    pub fn bbox(&self) -> Option<([f64; 3], [f64; 3])> {
        let mut it = self.positions_um.iter();
        let first = *it.next()?;
        let mut lo = first;
        let mut hi = first;
        for p in it {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        Some((lo, hi))
    }

    /// Diameter maksimum: jarak terjauh antar verteks pada selubung cembung.
    pub fn max_diameter_um(&self) -> f64 {
        max_diameter_of_points(&self.positions_um)
    }

    /// Jumlah komponen terhubung (verteks terhubung lewat sisi segitiga;
    /// verteks tak terpakai tidak dihitung).
    pub fn connected_components(&self) -> usize {
        let n = self.positions_um.len();
        let mut parent: Vec<u32> = (0..n as u32).collect();
        fn find(p: &mut [u32], mut x: u32) -> u32 {
            while p[x as usize] != x {
                let g = p[p[x as usize] as usize];
                p[x as usize] = g;
                x = g;
            }
            x
        }
        let mut used = vec![false; n];
        for t in self.triangles() {
            for k in 0..3 {
                used[t[k] as usize] = true;
                let a = find(&mut parent, t[k]);
                let b = find(&mut parent, t[(k + 1) % 3]);
                if a != b {
                    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                    parent[hi as usize] = lo;
                }
            }
        }
        (0..n as u32)
            .filter(|&i| used[i as usize] && find(&mut parent, i) == i)
            .count()
    }

    /// Peta sisi berarah → jumlah kemunculan.
    fn directed_edge_counts(&self) -> HashMap<(u32, u32), u32> {
        let mut m: HashMap<(u32, u32), u32> = HashMap::with_capacity(self.indices.len());
        for t in self.triangles() {
            for k in 0..3 {
                *m.entry((t[k], t[(k + 1) % 3])).or_insert(0) += 1;
            }
        }
        m
    }

    /// Benar bila tiap sisi dipakai tepat dua segitiga dengan orientasi
    /// berlawanan (manifold tertutup, orientasi konsisten).
    pub fn is_closed(&self) -> bool {
        if self.indices.is_empty() {
            return false;
        }
        let m = self.directed_edge_counts();
        m.iter()
            .all(|(&(a, b), &c)| c == 1 && m.get(&(b, a)).copied() == Some(1))
    }

    /// Karakteristik Euler `V − E + F` (verteks terpakai saja).
    pub fn euler_characteristic(&self) -> i64 {
        let mut used = vec![false; self.positions_um.len()];
        for &i in &self.indices {
            used[i as usize] = true;
        }
        let v = used.iter().filter(|&&u| u).count() as i64;
        let mut edges: HashMap<(u32, u32), ()> = HashMap::with_capacity(self.indices.len());
        for t in self.triangles() {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                edges.insert((a.min(b), a.max(b)), ());
            }
        }
        v - edges.len() as i64 + self.triangle_count() as i64
    }

    /// Daftar tetangga per verteks (terurut, unik).
    fn adjacency(&self) -> Vec<Vec<u32>> {
        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); self.positions_um.len()];
        for t in self.triangles() {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                adj[a as usize].push(b);
                adj[b as usize].push(a);
            }
        }
        for v in &mut adj {
            v.sort_unstable();
            v.dedup();
        }
        adj
    }

    /// Penghalusan Taubin: `λ` positif, `μ` negatif (`|μ| > λ`), `iters`
    /// pasang langkah. Tidak menyusutkan volume secara berarti (8.6.2).
    pub fn taubin_smooth(&mut self, lambda: f64, mu: f64, iters: u32) {
        if self.positions_um.len() < 3 || iters == 0 {
            return;
        }
        let adj = self.adjacency();
        let mut next = self.positions_um.clone();
        for _ in 0..iters {
            for &f in &[lambda, mu] {
                for (i, nb) in adj.iter().enumerate() {
                    if nb.is_empty() {
                        continue;
                    }
                    let mut m = [0.0f64; 3];
                    for &j in nb {
                        let q = self.positions_um[j as usize];
                        m[0] += q[0];
                        m[1] += q[1];
                        m[2] += q[2];
                    }
                    let n = nb.len() as f64;
                    let p = self.positions_um[i];
                    next[i] = [
                        p[0] + f * (m[0] / n - p[0]),
                        p[1] + f * (m[1] / n - p[1]),
                        p[2] + f * (m[2] / n - p[2]),
                    ];
                }
                std::mem::swap(&mut self.positions_um, &mut next);
            }
        }
    }

    /// Hitung ulang normal per verteks (rata-rata berbobot luas, satuan).
    pub fn recompute_normals(&mut self) {
        let n = self.positions_um.len();
        let mut acc = vec![[0.0f64; 3]; n];
        for t in self.triangles() {
            let p = &self.positions_um;
            let (a, b, c) = (p[t[0] as usize], p[t[1] as usize], p[t[2] as usize]);
            let fnrm = cross(sub(b, a), sub(c, a)); // panjang = 2 × luas
            for &i in &t {
                let s = &mut acc[i as usize];
                s[0] += fnrm[0];
                s[1] += fnrm[1];
                s[2] += fnrm[2];
            }
        }
        self.normals = acc
            .iter()
            .map(|v| {
                let l = norm(*v);
                if l > 0.0 {
                    [(v[0] / l) as f32, (v[1] / l) as f32, (v[2] / l) as f32]
                } else {
                    [0.0, 0.0, 1.0]
                }
            })
            .collect();
    }

    /// Balik orientasi semua segitiga.
    pub fn flip_winding(&mut self) {
        for t in self.indices.chunks_exact_mut(3) {
            t.swap(1, 2);
        }
    }

    /// Pastikan volume bertanda positif; membalik orientasi bila perlu.
    /// Mengembalikan `true` bila dibalik.
    pub fn ensure_outward(&mut self) -> bool {
        if self.signed_volume_um3() < 0.0 {
            self.flip_winding();
            true
        } else {
            false
        }
    }

    /// Las verteks yang berjarak < `eps` (kuantisasi kisi `eps`), buang
    /// segitiga degeneratif, lalu padatkan. Urutan verteks = kemunculan pertama.
    pub fn weld_vertices(&mut self, eps: f64) {
        let eps = if eps > 0.0 { eps } else { 1e-9 };
        let mut key_to_new: HashMap<[i64; 3], u32> = HashMap::new();
        let mut remap: Vec<u32> = Vec::with_capacity(self.positions_um.len());
        let mut new_pos: Vec<[f64; 3]> = Vec::new();
        for p in &self.positions_um {
            let key = [
                (p[0] / eps).round() as i64,
                (p[1] / eps).round() as i64,
                (p[2] / eps).round() as i64,
            ];
            let idx = *key_to_new.entry(key).or_insert_with(|| {
                new_pos.push(*p);
                (new_pos.len() - 1) as u32
            });
            remap.push(idx);
        }
        let mut new_idx = Vec::with_capacity(self.indices.len());
        for t in self.triangles() {
            let (a, b, c) = (
                remap[t[0] as usize],
                remap[t[1] as usize],
                remap[t[2] as usize],
            );
            if a != b && b != c && a != c {
                new_idx.extend_from_slice(&[a, b, c]);
            }
        }
        self.positions_um = new_pos;
        self.indices = new_idx;
        self.remap_scalars(&remap);
        self.compact();
        self.recompute_normals();
    }

    fn remap_scalars(&mut self, remap: &[u32]) {
        let n_new = remap.iter().map(|&r| r as usize + 1).max().unwrap_or(0);
        for s in &mut self.scalars {
            let mut v = vec![f32::NAN; n_new];
            for (old, &new) in remap.iter().enumerate() {
                if let Some(&x) = s.values.get(old) {
                    v[new as usize] = x;
                }
            }
            s.values = v;
        }
    }

    /// Buang verteks yang tidak dirujuk segitiga; urutan relatif tetap.
    #[allow(clippy::needless_range_loop)]
    pub fn compact(&mut self) {
        let n = self.positions_um.len();
        let mut used = vec![false; n];
        for &i in &self.indices {
            used[i as usize] = true;
        }
        let mut remap = vec![u32::MAX; n];
        let mut pos = Vec::new();
        let mut nrm = Vec::new();
        for i in 0..n {
            if used[i] {
                remap[i] = pos.len() as u32;
                pos.push(self.positions_um[i]);
                if let Some(nv) = self.normals.get(i) {
                    nrm.push(*nv);
                }
            }
        }
        for s in &mut self.scalars {
            let mut v = Vec::with_capacity(pos.len());
            for i in 0..n {
                if used[i] {
                    v.push(s.values.get(i).copied().unwrap_or(f32::NAN));
                }
            }
            s.values = v;
        }
        for i in &mut self.indices {
            *i = remap[*i as usize];
        }
        self.positions_um = pos;
        self.normals = if nrm.len() == self.positions_um.len() {
            nrm
        } else {
            Vec::new()
        };
    }

    /// Jadikan mesh manifold dengan "potong lalu jahit" (Guéziec dkk. 2001).
    ///
    /// Surface nets menghasilkan sisi yang dipakai empat segitiga (dan
    /// verteks tempat dua kerucut permukaan bertemu) bila sebuah muka kisi
    /// bertanda papan catur — dua bagian benda yang hanya bersentuhan pada
    /// satu sisi atau satu titik. Mesh seperti itu bukan permukaan tertutup
    /// yang sah (8.6.3; tes "tiap sisi dipakai tepat dua segitiga").
    ///
    /// 1. **Potong**: tiap verteks digandakan menjadi satu salinan per kipas
    ///    muka; dua muka sekipas bila terhubung lewat sisi manifold (tepat
    ///    dua muka, arah berlawanan) yang memuat verteks itu.
    /// 2. **Jahit**: sisi batas yang timbul dari pemotongan dipasangkan lagi
    ///    (arah berlawanan, asal sisi sama); pasangan yang sudah berbagi satu
    ///    ujung didahulukan sehingga jahitan hanya menutup celah dan tidak
    ///    mengembalikan sisi tak-manifold.
    ///
    /// Posisi salinan sama dengan aslinya (volume dan luas tidak berubah).
    /// Deterministik. Mengembalikan jumlah verteks yang ditambahkan; 0 bila
    /// mesh sudah manifold. Normal dihitung ulang bila ada perubahan.
    pub fn repair_non_manifold(&mut self) -> usize {
        let nf = self.triangle_count();
        let nv0 = self.positions_um.len();
        if nf == 0 {
            return 0;
        }
        // Sisi tak berarah → daftar (muka, searah lo→hi?).
        let mut edge_faces: HashMap<(u32, u32), Vec<(u32, bool)>> =
            HashMap::with_capacity(self.indices.len());
        for (fi, t) in self.triangles().enumerate() {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                edge_faces
                    .entry((a.min(b), a.max(b)))
                    .or_default()
                    .push((fi as u32, a < b));
            }
        }
        let manifold_other = |a: u32, b: u32, f: u32| -> Option<u32> {
            let v = edge_faces.get(&(a.min(b), a.max(b)))?;
            if v.len() != 2 || v[0].1 == v[1].1 {
                return None;
            }
            Some(if v[0].0 == f { v[1].0 } else { v[0].0 })
        };
        let mut vfaces: Vec<Vec<u32>> = vec![Vec::new(); nv0];
        for (fi, t) in self.triangles().enumerate() {
            for &v in &t {
                vfaces[v as usize].push(fi as u32);
            }
        }
        // Langkah 1: potong. `corner[3·f + k]` = indeks verteks baru.
        let faces: Vec<[u32; 3]> = self.triangles().collect();
        let mut corner: Vec<u32> = self.indices.clone();
        let mut origin: Vec<u32> = (0..nv0 as u32).collect();
        for v in 0..nv0 as u32 {
            let inc = &vfaces[v as usize];
            if inc.len() < 2 {
                continue;
            }
            // Union-find lokal atas muka di sekitar `v`.
            let mut parent: Vec<usize> = (0..inc.len()).collect();
            fn find(p: &mut [usize], mut x: usize) -> usize {
                while p[x] != x {
                    p[x] = p[p[x]];
                    x = p[x];
                }
                x
            }
            for (i, &f) in inc.iter().enumerate() {
                for &w in &faces[f as usize] {
                    if w == v {
                        continue;
                    }
                    if let Some(g) = manifold_other(v, w, f) {
                        if let Ok(j) = inc.binary_search(&g) {
                            let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                            if ri != rj {
                                parent[ri.max(rj)] = ri.min(rj);
                            }
                        }
                    }
                }
            }
            // Kipas pertama (memuat muka berindeks terkecil) memakai `v`.
            let mut comp_vertex: HashMap<usize, u32> = HashMap::new();
            for (i, &face) in inc.iter().enumerate() {
                let r = find(&mut parent, i);
                let next = if comp_vertex.is_empty() {
                    v
                } else {
                    origin.len() as u32
                };
                let id = *comp_vertex.entry(r).or_insert(next);
                if id == next && id != v {
                    origin.push(v);
                }
                let f = face as usize;
                for k in 0..3 {
                    if faces[f][k] == v {
                        corner[3 * f + k] = id;
                    }
                }
            }
        }
        let n_cut = origin.len();
        if n_cut == nv0 {
            // Tidak ada verteks singular; sisi tak-manifold mustahil tersisa
            // tanpa verteks singular, kecuali sisi batas (mesh terbuka).
            return 0;
        }
        // Langkah 2: jahit sisi batas yang berasal dari sisi asli yang sama.
        let mut directed: HashMap<(u32, u32), u32> = HashMap::with_capacity(corner.len());
        for t in corner.chunks_exact(3) {
            for k in 0..3 {
                *directed.entry((t[k], t[(k + 1) % 3])).or_insert(0) += 1;
            }
        }
        // (asal lo, asal hi, searah lo→hi?, x, y)
        let mut boundary: Vec<(u32, u32, bool, u32, u32)> = Vec::new();
        for t in corner.chunks_exact(3) {
            for k in 0..3 {
                let (x, y) = (t[k], t[(k + 1) % 3]);
                if directed.get(&(y, x)).copied().unwrap_or(0) == 0 {
                    let (ox, oy) = (origin[x as usize], origin[y as usize]);
                    boundary.push((ox.min(oy), ox.max(oy), ox < oy, x, y));
                }
            }
        }
        boundary.sort_unstable();
        boundary.dedup();
        let mut merge: Vec<u32> = (0..n_cut as u32).collect();
        fn root(m: &mut [u32], mut x: u32) -> u32 {
            while m[x as usize] != x {
                m[x as usize] = m[m[x as usize] as usize];
                x = m[x as usize];
            }
            x
        }
        let mut i = 0;
        while i < boundary.len() {
            let mut j = i;
            while j < boundary.len()
                && (boundary[j].0, boundary[j].1) == (boundary[i].0, boundary[i].1)
            {
                j += 1;
            }
            let group = &boundary[i..j];
            let fwd: Vec<(u32, u32)> = group.iter().filter(|e| e.2).map(|e| (e.3, e.4)).collect();
            let bwd: Vec<(u32, u32)> = group.iter().filter(|e| !e.2).map(|e| (e.3, e.4)).collect();
            let mut used = vec![false; bwd.len()];
            // Dua putaran: pasangan yang sudah berbagi ujung, lalu sisanya.
            for pass in 0..2 {
                for &(x, y) in &fwd {
                    // fwd: x→y (asal lo→hi); bwd: y2→x2 (asal hi→lo).
                    let (rx, ry) = (root(&mut merge, x), root(&mut merge, y));
                    let already = bwd.iter().enumerate().any(|(k, &(y2, x2))| {
                        used[k] && root(&mut merge, x2) == rx && root(&mut merge, y2) == ry
                    });
                    if already {
                        continue;
                    }
                    let pick = bwd.iter().enumerate().position(|(k, &(y2, x2))| {
                        !used[k]
                            && (pass == 1
                                || root(&mut merge, x2) == rx
                                || root(&mut merge, y2) == ry)
                    });
                    if let Some(k) = pick {
                        used[k] = true;
                        let (y2, x2) = bwd[k];
                        for (p, q) in [(x, x2), (y, y2)] {
                            let (rp, rq) = (root(&mut merge, p), root(&mut merge, q));
                            if rp != rq {
                                merge[rp.max(rq) as usize] = rp.min(rq);
                            }
                        }
                    }
                }
            }
            i = j;
        }
        for c in &mut corner {
            *c = root(&mut merge, *c);
        }
        // Tulis balik: salinan mewarisi posisi dan skalar aslinya.
        for &o in &origin[nv0..] {
            self.positions_um.push(self.positions_um[o as usize]);
        }
        for sc in &mut self.scalars {
            for &o in &origin[nv0..] {
                let v = sc.values.get(o as usize).copied().unwrap_or(f32::NAN);
                sc.values.push(v);
            }
        }
        self.normals.clear();
        self.indices = corner;
        self.compact();
        self.recompute_normals();
        self.positions_um.len().saturating_sub(nv0)
    }

    /// Decimate dengan metrik galat kuadrik (Garland–Heckbert), titik
    /// gabungan di tengah sisi, galat maksimum `max_error_um` (jarak ke
    /// bidang-bidang asal). Menolak kolaps yang merusak topologi (syarat
    /// link) atau membalik normal, sehingga mesh tertutup tetap tertutup.
    /// Deterministik: seri diputus oleh indeks verteks terkecil.
    pub fn decimate(&mut self, max_error_um: f64) {
        if max_error_um <= 0.0 || self.triangle_count() < 8 {
            return;
        }
        let mut d = Decimator::new(self, max_error_um);
        d.run();
        d.write_back(self);
        self.recompute_normals();
    }

    /// Ambil nilai medan `field` (indeks `x + nx·(y + ny·z)` atas `grid`)
    /// pada tiap verteks dengan interpolasi trilinear; verteks di luar kisi
    /// atau yang menyentuh NODATA (`NaN`) mendapat `NaN`.
    pub fn sample_field_trilinear(
        &self,
        grid: &StackGrid,
        field: &[f32],
        name: &str,
        unit: &str,
    ) -> MeshScalars {
        let values = self
            .positions_um
            .iter()
            .map(|p| sample_trilinear(grid, field, *p))
            .collect();
        MeshScalars {
            name: name.to_owned(),
            unit: unit.to_owned(),
            values,
        }
    }

    /// Tambahkan/ganti skalar bernama.
    pub fn set_scalars(&mut self, s: MeshScalars) {
        if let Some(e) = self.scalars.iter_mut().find(|e| e.name == s.name) {
            *e = s;
        } else {
            self.scalars.push(s);
        }
    }
}

/// Interpolasi trilinear medan kisi pada titik W `p`; `NaN` bila di luar
/// kisi (lebih dari setengah voxel dari pusat voxel tepi) atau sudut NODATA.
pub fn sample_trilinear(grid: &StackGrid, field: &[f32], p: [f64; 3]) -> f32 {
    let [nx, ny, nz] = grid.dims();
    if nx == 0 || ny == 0 || nz == 0 || field.len() != nx * ny * nz {
        return f32::NAN;
    }
    let g = grid.um_to_grid_continuous([p[0], p[1]]);
    let kf = match grid.z_to_continuous(p[2]) {
        Some(k) => k,
        None => return f32::NAN,
    };
    let clampf = |v: f64, n: usize| -> Option<f64> {
        if v < -0.5 || v > n as f64 - 0.5 {
            None
        } else {
            Some(v.clamp(0.0, (n as f64 - 1.0).max(0.0)))
        }
    };
    let (gx, gy, gz) = match (clampf(g[0], nx), clampf(g[1], ny), clampf(kf, nz)) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        _ => return f32::NAN,
    };
    let x0 = gx.floor() as usize;
    let y0 = gy.floor() as usize;
    let z0 = gz.floor() as usize;
    let x1 = (x0 + 1).min(nx - 1);
    let y1 = (y0 + 1).min(ny - 1);
    let z1 = (z0 + 1).min(nz - 1);
    let tx = gx - x0 as f64;
    let ty = gy - y0 as f64;
    let tz = gz - z0 as f64;
    let at = |x: usize, y: usize, z: usize| field[grid.linear(x, y, z)] as f64;
    let c = [
        at(x0, y0, z0),
        at(x1, y0, z0),
        at(x0, y1, z0),
        at(x1, y1, z0),
        at(x0, y0, z1),
        at(x1, y0, z1),
        at(x0, y1, z1),
        at(x1, y1, z1),
    ];
    if c.iter().any(|v| v.is_nan()) {
        return f32::NAN;
    }
    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
    let c00 = lerp(c[0], c[1], tx);
    let c10 = lerp(c[2], c[3], tx);
    let c01 = lerp(c[4], c[5], tx);
    let c11 = lerp(c[6], c[7], tx);
    let c0 = lerp(c00, c10, ty);
    let c1 = lerp(c01, c11, ty);
    lerp(c0, c1, tz) as f32
}

// ---------------------------------------------------------------------------
// Decimate QEM
// ---------------------------------------------------------------------------

/// Kuadrik simetris 4×4 (10 koefisien).
#[derive(Clone, Copy, Default)]
struct Quadric([f64; 10]);

impl Quadric {
    fn from_plane(n: [f64; 3], d: f64) -> Self {
        let (a, b, c) = (n[0], n[1], n[2]);
        Quadric([
            a * a,
            a * b,
            a * c,
            a * d,
            b * b,
            b * c,
            b * d,
            c * c,
            c * d,
            d * d,
        ])
    }
    fn add(&mut self, o: &Quadric) {
        for k in 0..10 {
            self.0[k] += o.0[k];
        }
    }
    fn sum(&self, o: &Quadric) -> Quadric {
        let mut q = *self;
        q.add(o);
        q
    }
    /// `vᵀ Q v` untuk `v = (x, y, z, 1)`.
    fn eval(&self, p: [f64; 3]) -> f64 {
        let q = &self.0;
        let (x, y, z) = (p[0], p[1], p[2]);
        q[0] * x * x
            + 2.0 * q[1] * x * y
            + 2.0 * q[2] * x * z
            + 2.0 * q[3] * x
            + q[4] * y * y
            + 2.0 * q[5] * y * z
            + 2.0 * q[6] * y
            + q[7] * z * z
            + 2.0 * q[8] * z
            + q[9]
    }
}

#[derive(Clone, Copy)]
struct Candidate {
    cost: f64,
    u: u32,
    v: u32,
    ver_u: u32,
    ver_v: u32,
}

impl PartialEq for Candidate {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Candidate {
    /// Heap maksimum → balik: biaya kecil lebih dulu; seri → indeks terkecil.
    fn cmp(&self, o: &Self) -> Ordering {
        o.cost
            .total_cmp(&self.cost)
            .then_with(|| o.u.cmp(&self.u))
            .then_with(|| o.v.cmp(&self.v))
    }
}

struct Decimator {
    pos: Vec<[f64; 3]>,
    quad: Vec<Quadric>,
    faces: Vec<[u32; 3]>,
    face_alive: Vec<bool>,
    vfaces: Vec<Vec<u32>>,
    version: Vec<u32>,
    alive: Vec<bool>,
    heap: BinaryHeap<Candidate>,
    max_err2: f64,
}

impl Decimator {
    fn new(m: &LabelMesh, max_error_um: f64) -> Self {
        let n = m.positions_um.len();
        let faces: Vec<[u32; 3]> = m.triangles().collect();
        let mut vfaces: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut quad = vec![Quadric::default(); n];
        for (fi, f) in faces.iter().enumerate() {
            let (a, b, c) = (
                m.positions_um[f[0] as usize],
                m.positions_um[f[1] as usize],
                m.positions_um[f[2] as usize],
            );
            let nrm = cross(sub(b, a), sub(c, a));
            let l = norm(nrm);
            if l > 0.0 {
                let nn = [nrm[0] / l, nrm[1] / l, nrm[2] / l];
                let d = -dot(nn, a);
                let q = Quadric::from_plane(nn, d);
                for &i in f {
                    quad[i as usize].add(&q);
                }
            }
            for &i in f {
                vfaces[i as usize].push(fi as u32);
            }
        }
        let mut d = Decimator {
            pos: m.positions_um.clone(),
            quad,
            face_alive: vec![true; faces.len()],
            faces,
            vfaces,
            version: vec![0; n],
            alive: vec![true; n],
            heap: BinaryHeap::new(),
            max_err2: max_error_um * max_error_um,
        };
        // Isi heap dengan semua sisi (u < v) dalam urutan tetap.
        let mut edges: Vec<(u32, u32)> = Vec::with_capacity(d.faces.len() * 3);
        for f in &d.faces {
            for k in 0..3 {
                let (a, b) = (f[k], f[(k + 1) % 3]);
                edges.push((a.min(b), a.max(b)));
            }
        }
        edges.sort_unstable();
        edges.dedup();
        for (u, v) in edges {
            d.push_edge(u, v);
        }
        d
    }

    fn push_edge(&mut self, u: u32, v: u32) {
        let (u, v) = (u.min(v), u.max(v));
        let q = self.quad[u as usize].sum(&self.quad[v as usize]);
        let mid = self.midpoint(u, v);
        let cost = q.eval(mid).max(0.0);
        if cost <= self.max_err2 {
            self.heap.push(Candidate {
                cost,
                u,
                v,
                ver_u: self.version[u as usize],
                ver_v: self.version[v as usize],
            });
        }
    }

    fn midpoint(&self, u: u32, v: u32) -> [f64; 3] {
        let a = self.pos[u as usize];
        let b = self.pos[v as usize];
        [
            (a[0] + b[0]) * 0.5,
            (a[1] + b[1]) * 0.5,
            (a[2] + b[2]) * 0.5,
        ]
    }

    /// Tetangga verteks (terurut unik) lewat muka hidup.
    fn neighbours(&self, u: u32) -> Vec<u32> {
        let mut out = Vec::new();
        for &fi in &self.vfaces[u as usize] {
            if !self.face_alive[fi as usize] {
                continue;
            }
            for &w in &self.faces[fi as usize] {
                if w != u {
                    out.push(w);
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    fn face_normal(&self, f: [u32; 3], moved: u32, newp: [f64; 3]) -> [f64; 3] {
        let p = |i: u32| {
            if i == moved {
                newp
            } else {
                self.pos[i as usize]
            }
        };
        let (a, b, c) = (p(f[0]), p(f[1]), p(f[2]));
        cross(sub(b, a), sub(c, a))
    }

    /// Kolaps `u` ke `v` (verteks `u` dibuang, `v` dipindah ke titik tengah).
    fn try_collapse(&mut self, u: u32, v: u32) -> bool {
        // Syarat link: tetangga bersama harus tepat verteks lawan dari muka
        // yang memuat sisi (u, v).
        let nu = self.neighbours(u);
        let nv = self.neighbours(v);
        let common: Vec<u32> = nu
            .iter()
            .filter(|w| nv.binary_search(w).is_ok())
            .copied()
            .collect();
        let mut shared_faces = 0usize;
        let mut opposite: Vec<u32> = Vec::new();
        for &fi in &self.vfaces[u as usize] {
            if !self.face_alive[fi as usize] {
                continue;
            }
            let f = self.faces[fi as usize];
            if f.contains(&v) {
                shared_faces += 1;
                for &w in &f {
                    if w != u && w != v {
                        opposite.push(w);
                    }
                }
            }
        }
        opposite.sort_unstable();
        opposite.dedup();
        if shared_faces == 0 || shared_faces > 2 || common != opposite {
            return false;
        }
        let newp = self.midpoint(u, v);
        // Uji balik normal dan degenerasi pada muka yang bertahan.
        for &w in &[u, v] {
            for &fi in &self.vfaces[w as usize] {
                if !self.face_alive[fi as usize] {
                    continue;
                }
                let f = self.faces[fi as usize];
                if f.contains(&u) && f.contains(&v) {
                    continue; // akan dibuang
                }
                let before = self.face_normal(f, u32::MAX, newp);
                let mut f2 = f;
                for x in &mut f2 {
                    if *x == u {
                        *x = v;
                    }
                }
                let after = self.face_normal(f2, v, newp);
                let la = norm(after);
                if la <= 1e-12 * norm(before).max(1e-300) || dot(before, after) <= 0.0 {
                    return false;
                }
            }
        }
        // Lakukan kolaps.
        let qsum = self.quad[u as usize].sum(&self.quad[v as usize]);
        self.quad[v as usize] = qsum;
        self.pos[v as usize] = newp;
        let ufaces = std::mem::take(&mut self.vfaces[u as usize]);
        for fi in ufaces {
            if !self.face_alive[fi as usize] {
                continue;
            }
            let f = &mut self.faces[fi as usize];
            if f.contains(&v) {
                self.face_alive[fi as usize] = false;
                continue;
            }
            for x in f.iter_mut() {
                if *x == u {
                    *x = v;
                }
            }
            self.vfaces[v as usize].push(fi);
        }
        self.vfaces[v as usize].retain(|&fi| self.face_alive[fi as usize]);
        self.alive[u as usize] = false;
        // Hanya u dan v yang berubah; sisi antar tetangga lain tetap sahih.
        self.version[u as usize] += 1;
        self.version[v as usize] += 1;
        for w in self.neighbours(v) {
            self.push_edge(v, w);
        }
        true
    }

    fn run(&mut self) {
        while let Some(c) = self.heap.pop() {
            if !self.alive[c.u as usize]
                || !self.alive[c.v as usize]
                || self.version[c.u as usize] != c.ver_u
                || self.version[c.v as usize] != c.ver_v
            {
                continue;
            }
            // Buang verteks berindeks lebih besar; pertahankan yang kecil.
            self.try_collapse(c.v, c.u);
        }
    }

    fn write_back(&self, m: &mut LabelMesh) {
        let mut remap = vec![u32::MAX; self.pos.len()];
        let mut pos = Vec::new();
        for (i, &a) in self.alive.iter().enumerate() {
            if a {
                remap[i] = pos.len() as u32;
                pos.push(self.pos[i]);
            }
        }
        let mut idx = Vec::new();
        for (fi, f) in self.faces.iter().enumerate() {
            if self.face_alive[fi] {
                idx.extend(f.iter().map(|&i| remap[i as usize]));
            }
        }
        m.positions_um = pos;
        m.indices = idx;
        // Skalar lama tidak lagi sahih setelah decimate.
        m.scalars.clear();
        m.compact();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tetrahedron berorientasi ke luar.
    fn tetra() -> LabelMesh {
        LabelMesh {
            label: LabelId(1),
            positions_um: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            normals: Vec::new(),
            indices: vec![0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3],
            scalars: Vec::new(),
        }
    }

    #[test]
    fn tetra_measures() {
        let mut m = tetra();
        assert!((m.signed_volume_um3() - 1.0 / 6.0).abs() < 1e-12);
        assert!(m.is_closed());
        assert_eq!(m.euler_characteristic(), 2);
        assert_eq!(m.connected_components(), 1);
        assert!((m.max_diameter_um() - 2f64.sqrt()).abs() < 1e-12);
        m.flip_winding();
        assert!(m.signed_volume_um3() < 0.0);
        assert!(m.ensure_outward());
        assert!(m.signed_volume_um3() > 0.0);
        m.recompute_normals();
        assert_eq!(m.normals.len(), 4);
    }

    #[test]
    fn weld_merges_duplicates() {
        let mut m = tetra();
        m.positions_um.push([0.0, 0.0, 1.0 + 1e-9]);
        m.indices[11] = 4;
        m.weld_vertices(1e-6);
        assert_eq!(m.positions_um.len(), 4);
        assert!(m.is_closed());
    }

    #[test]
    fn trilinear_linear_field_exact() {
        let grid = StackGrid {
            origin_um: [0.0, 0.0],
            s_xy: 10.0,
            size: [4, 5],
            z_um: vec![0.0, 20.0, 50.0],
        };
        let f = |p: [f64; 3]| (2.0 * p[0] - p[1] + 0.5 * p[2]) as f32;
        let mut field = vec![0.0f32; grid.voxel_count()];
        for z in 0..3 {
            for y in 0..5 {
                for x in 0..4 {
                    field[grid.linear(x, y, z)] = f(grid.center_um(x, y, z));
                }
            }
        }
        let p = [17.0, 23.0, 35.0];
        let v = sample_trilinear(&grid, &field, p);
        assert!((v - f(p)).abs() < 1e-3, "{v} vs {}", f(p));
        assert!(sample_trilinear(&grid, &field, [-100.0, 0.0, 0.0]).is_nan());
    }
}
