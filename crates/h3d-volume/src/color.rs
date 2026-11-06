//! Warna pulasan per verteks mesh dari irisan histologi terdekat
//! (tampilan/ekspor saja; tidak pernah masukan perhitungan).
//!
//! # Konvensi
//! - Verteks di ruang W (µm). Untuk irisan `i`, titik W dipetakan ke ruang U
//!   lewat `T_i⁻¹` (W → U, aturan keras 4) lalu dibaca sampler.
//! - Titik diambil sedikit ke dalam permukaan (`p − inset·n`) agar warna
//!   berasal dari jaringan objek, bukan latar tepat di batas.
//! - Dua irisan terdekat yang mengapit `z` dicampur linear menurut jarak Z;
//!   bila hanya satu yang berwarna, warnanya dipakai.
//! - Verteks tanpa warna diisi difusi tetangga (urutan tetap), lalu
//!   rata-rata semua warna. Deterministik.

use h3d_core::SlideTransform;
use rayon::prelude::*;

use crate::mesh::LabelMesh;

/// Sumber warna satu irisan di ruang U.
pub trait ColorSampler: Sync {
    /// Warna sRGB pada titik U `u` (µm); `None` = di luar citra atau latar
    /// (mis. alfa masker jaringan 0).
    fn rgb_at_um(&self, u: [f64; 2]) -> Option<[u8; 3]>;
}

/// Satu irisan sumber warna.
pub struct ColorSlice<'a> {
    /// `z` irisan di ruang W, µm.
    pub z_um: f64,
    /// `T_i`: U → W. Dipakai mundur (`T_i⁻¹`).
    pub transform: &'a SlideTransform,
    /// Sampler warna (ruang U).
    pub sampler: &'a dyn ColorSampler,
}

/// Iterasi difusi maksimum untuk mengisi verteks tanpa warna.
const FILL_ITERS: usize = 64;

/// Warna per verteks `mesh` dari `slices` (lihat konvensi modul). `inset_um`
/// = jarak ke dalam sepanjang `−n`. Kosong bila tidak ada satu pun verteks
/// yang mendapat warna (mis. semua irisan tanpa citra).
pub fn sample_vertex_colors(
    mesh: &LabelMesh,
    slices: &[ColorSlice<'_>],
    inset_um: f64,
) -> Vec<[u8; 3]> {
    let nv = mesh.positions_um.len();
    if nv == 0 || slices.is_empty() {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..slices.len()).collect();
    order.sort_by(|&a, &b| slices[a].z_um.total_cmp(&slices[b].z_um).then(a.cmp(&b)));
    let zs: Vec<f64> = order.iter().map(|&i| slices[i].z_um).collect();
    let sample = |k: usize, w: [f64; 2]| -> Option<[u8; 3]> {
        let s = &slices[order[k]];
        let u = s.transform.apply_inverse(w)?;
        s.sampler.rgb_at_um(u)
    };
    let inset = if inset_um.is_finite() {
        inset_um.max(0.0)
    } else {
        0.0
    };
    let raw: Vec<Option<[u8; 3]>> = (0..nv)
        .into_par_iter()
        .map(|i| {
            let p = mesh.positions_um[i];
            let n = mesh.normals.get(i).copied().unwrap_or([0.0, 0.0, 0.0]);
            let q = [
                p[0] - inset * f64::from(n[0]),
                p[1] - inset * f64::from(n[1]),
                p[2] - inset * f64::from(n[2]),
            ];
            let w = [q[0], q[1]];
            // Irisan pertama dengan z ≥ q.z (atas) dan sebelumnya (bawah).
            let hi = zs.partition_point(|&z| z < q[2]);
            let below = hi.checked_sub(1);
            let above = (hi < zs.len()).then_some(hi);
            let cb = below.and_then(|k| sample(k, w).map(|c| (k, c)));
            let ca = above.and_then(|k| sample(k, w).map(|c| (k, c)));
            match (cb, ca) {
                (Some((kb, b)), Some((ka, a))) => {
                    let span = zs[ka] - zs[kb];
                    let t = if span > 0.0 {
                        ((q[2] - zs[kb]) / span).clamp(0.0, 1.0)
                    } else {
                        0.5
                    };
                    Some(mix(b, a, t))
                }
                (Some((_, c)), None) | (None, Some((_, c))) => Some(c),
                (None, None) => {
                    // Irisan berwarna terdekat lainnya (seri → indeks kecil).
                    let mut best: Option<(f64, [u8; 3])> = None;
                    for (k, &z) in zs.iter().enumerate() {
                        let d = (z - q[2]).abs();
                        if best.is_some_and(|(bd, _)| d >= bd) {
                            continue;
                        }
                        if let Some(c) = sample(k, w) {
                            best = Some((d, c));
                        }
                    }
                    best.map(|(_, c)| c)
                }
            }
        })
        .collect();
    fill_missing(mesh, raw)
}

fn mix(a: [u8; 3], b: [u8; 3], t: f64) -> [u8; 3] {
    [0, 1, 2].map(|k| {
        let v = f64::from(a[k]) * (1.0 - t) + f64::from(b[k]) * t;
        v.round().clamp(0.0, 255.0) as u8
    })
}

/// Isi `None` dengan rata-rata tetangga berwarna (difusi berulang, urutan
/// tetap), sisanya rata-rata global. Semua `None` → kosong.
fn fill_missing(mesh: &LabelMesh, mut c: Vec<Option<[u8; 3]>>) -> Vec<[u8; 3]> {
    let nv = c.len();
    if c.iter().all(Option::is_none) {
        return Vec::new();
    }
    if c.iter().any(Option::is_none) {
        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); nv];
        for t in mesh.indices.chunks_exact(3) {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                if (a as usize) < nv && (b as usize) < nv {
                    adj[a as usize].push(b);
                    adj[b as usize].push(a);
                }
            }
        }
        for v in &mut adj {
            v.sort_unstable();
            v.dedup();
        }
        for _ in 0..FILL_ITERS {
            let next: Vec<Option<[u8; 3]>> = (0..nv)
                .map(|i| {
                    if c[i].is_some() {
                        return c[i];
                    }
                    let mut s = [0u32; 3];
                    let mut n = 0u32;
                    for &j in &adj[i] {
                        if let Some(v) = c[j as usize] {
                            for k in 0..3 {
                                s[k] += u32::from(v[k]);
                            }
                            n += 1;
                        }
                    }
                    (n > 0).then(|| s.map(|x| ((x + n / 2) / n) as u8))
                })
                .collect();
            let changed = next.iter().zip(&c).any(|(a, b)| a.is_some() != b.is_some());
            c = next;
            if !changed || c.iter().all(Option::is_some) {
                break;
            }
        }
    }
    let known: Vec<[u8; 3]> = c.iter().flatten().copied().collect();
    let n = known.len() as u64;
    let mean = [0, 1, 2].map(|k| {
        let s: u64 = known.iter().map(|v| u64::from(v[k])).sum();
        ((s + n / 2) / n.max(1)) as u8
    });
    c.into_iter().map(|v| v.unwrap_or(mean)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use h3d_core::{Affine2, LabelId};

    /// Sampler: kiri (x < 50) merah, kanan biru, di luar `[0,100]²` tidak ada.
    struct Halves;
    impl ColorSampler for Halves {
        fn rgb_at_um(&self, u: [f64; 2]) -> Option<[u8; 3]> {
            if !(0.0..100.0).contains(&u[0]) || !(0.0..100.0).contains(&u[1]) {
                return None;
            }
            Some(if u[0] < 50.0 {
                [255, 0, 0]
            } else {
                [0, 0, 255]
            })
        }
    }
    struct Solid([u8; 3]);
    impl ColorSampler for Solid {
        fn rgb_at_um(&self, _u: [f64; 2]) -> Option<[u8; 3]> {
            Some(self.0)
        }
    }

    fn quad_mesh(pts: Vec<[f64; 3]>) -> LabelMesh {
        let mut m = LabelMesh::empty(LabelId(1));
        m.indices = vec![0, 1, 2, 0, 2, 3];
        m.normals = vec![[0.0, 0.0, 1.0]; pts.len()];
        m.positions_um = pts;
        m
    }

    #[test]
    fn colour_follows_inverse_transform_and_fills_gaps() {
        let id = SlideTransform::Identity;
        let s = [ColorSlice {
            z_um: 0.0,
            transform: &id,
            sampler: &Halves,
        }];
        let m = quad_mesh(vec![
            [10.0, 10.0, 0.0],
            [90.0, 10.0, 0.0],
            [90.0, 90.0, 0.0],
            [500.0, 500.0, 0.0], // di luar citra → diisi tetangga
        ]);
        let c = sample_vertex_colors(&m, &s, 0.0);
        assert_eq!(c[0], [255, 0, 0]);
        assert_eq!(c[1], [0, 0, 255]);
        assert_eq!(c[2], [0, 0, 255]);
        // Tetangga verteks 3: 0 (merah) dan 2 (biru) → rata-rata.
        assert_eq!(c[3], [128, 0, 128]);

        // T: U → W menggeser +60 µm di x; W x = 90 → U x = 30 (merah).
        let shift = SlideTransform::Affine(Affine2::translation(60.0, 0.0));
        let s2 = [ColorSlice {
            z_um: 0.0,
            transform: &shift,
            sampler: &Halves,
        }];
        let c2 = sample_vertex_colors(&m, &s2, 0.0);
        assert_eq!(c2[1], [255, 0, 0]);
        assert_eq!(c2[2], [255, 0, 0]);
    }

    #[test]
    fn two_slices_blend_by_z() {
        let id = SlideTransform::Identity;
        let s = [
            ColorSlice {
                z_um: 100.0,
                transform: &id,
                sampler: &Solid([200, 0, 0]),
            },
            ColorSlice {
                z_um: 0.0,
                transform: &id,
                sampler: &Solid([0, 0, 200]),
            },
        ];
        let m = quad_mesh(vec![
            [0.0, 0.0, 25.0],
            [1.0, 0.0, 25.0],
            [1.0, 1.0, -10.0],
            [0.0, 1.0, 150.0],
        ]);
        let c = sample_vertex_colors(&m, &s, 0.0);
        assert_eq!(c[0], [50, 0, 150]);
        assert_eq!(c[2], [0, 0, 200]);
        assert_eq!(c[3], [200, 0, 0]);
    }

    #[test]
    fn no_colour_anywhere_gives_empty() {
        struct Nothing;
        impl ColorSampler for Nothing {
            fn rgb_at_um(&self, _u: [f64; 2]) -> Option<[u8; 3]> {
                None
            }
        }
        let id = SlideTransform::Identity;
        let s = [ColorSlice {
            z_um: 0.0,
            transform: &id,
            sampler: &Nothing,
        }];
        let m = quad_mesh(vec![
            [0.0; 3],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]);
        assert!(sample_vertex_colors(&m, &s, 0.0).is_empty());
    }
}
