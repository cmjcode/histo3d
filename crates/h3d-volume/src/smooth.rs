//! Penghalusan Gaussian arah Z atas SDF (pembungkus [`SdfProvider`]) sebelum
//! ekstraksi permukaan. Menghilangkan "tangga" antar irisan pada mesh.
//!
//! # Konvensi
//! - Hanya arah Z: SDF tiap irisan sudah halus di XY (EDT 2D), sedangkan
//!   tangga lahir dari interpolasi linear antar irisan. Bobot memakai posisi
//!   lapis `z_um` yang sebenarnya (jarak boleh tidak seragam), dinormalkan.
//! - Blur menggeser nol SDF ke dalam sebanding `σ² / r` pada bagian
//!   melengkung. Karena itu sigma efektif dibatasi `INSCRIBED_FRACTION ×`
//!   min(jari-jari dalam XY terbesar objek (−min SDF), setengah tebal Z
//!   objek), dan
//!   [`SmoothedSdf::target_volume_um3`] memberi volume SDF asli agar mesh
//!   bisa dipulihkan (`LabelMesh::restore_volume`). Objek kecil tidak hilang.
//! - Hanya tampilan geometri mesh: masker voxel dan semua pengukuran tidak
//!   pernah melalui pembungkus ini (aturan keras 19).
//! - Deterministik: fungsi murni atas lapis masukan; reduksi min/jumlah
//!   dalam urutan lapis tetap.

use rayon::prelude::*;

use crate::grid::StackGrid;
use crate::recon::SdfProvider;

/// Kernel dipotong pada `TRUNCATE × sigma`.
const TRUNCATE: f64 = 3.0;
/// Sigma efektif ≤ fraksi ini × jari-jari dalam terbesar objek.
const INSCRIBED_FRACTION: f64 = 0.5;

/// SDF yang dihaluskan Gaussian arah Z: `fill_layer(k)` = jumlah berbobot
/// lapis tetangga dalam `3σ`.
pub struct SmoothedSdf<'a> {
    inner: &'a dyn SdfProvider,
    /// Ketukan Z per lapis keluaran: `(lapis sumber, bobot)`, bobot berjumlah 1.
    taps_z: Vec<Vec<(usize, f32)>>,
    sigma_um: f64,
    target_volume_um3: f64,
}

impl<'a> SmoothedSdf<'a> {
    /// Bungkus `inner` dengan Gaussian Z sigma `sigma_um` (µm), dijepit ke
    /// `max_sigma_um` (mis. sepertiga pita SDF) dan ke `0,5 ×` jari-jari
    /// dalam objek. `sigma_um ≤ 0` atau tak finit → identitas.
    pub fn new(inner: &'a dyn SdfProvider, sigma_um: f64, max_sigma_um: f64) -> Self {
        let grid = inner.grid();
        let st = inside_stats(inner);
        let mut sigma = if sigma_um.is_finite() {
            sigma_um.clamp(0.0, max_sigma_um.max(0.0))
        } else {
            0.0
        };
        let r_in = f64::from(-st.min_sdf.min(0.0));
        sigma = sigma.min(INSCRIBED_FRACTION * r_in.min(0.5 * st.thickness_um));
        SmoothedSdf {
            inner,
            taps_z: z_taps(&grid.z_um, sigma),
            sigma_um: sigma,
            target_volume_um3: st.volume_um3,
        }
    }

    /// Sigma efektif setelah semua batas, µm.
    pub fn sigma_um(&self) -> f64 {
        self.sigma_um
    }

    /// Volume daerah `SDF < 0` sebelum penghalusan (voxel × tebal lapis), µm³.
    pub fn target_volume_um3(&self) -> f64 {
        self.target_volume_um3
    }
}

/// Tebal lapis `k`: setengah jarak ke kedua tetangga (ujung: satu sisi).
fn layer_thickness(z: &[f64], k: usize) -> f64 {
    let n = z.len();
    match (k.checked_sub(1), (k + 1 < n).then_some(k + 1)) {
        (Some(a), Some(b)) => 0.5 * (z[b] - z[a]),
        (Some(a), None) => z[k] - z[a],
        (None, Some(b)) => z[b] - z[k],
        (None, None) => 0.0,
    }
}

/// Ringkasan daerah `SDF < 0` atas semua lapis.
struct InsideStats {
    /// SDF minimum (−jari-jari dalam XY terbesar), µm.
    min_sdf: f32,
    /// Volume, µm³.
    volume_um3: f64,
    /// Jumlah tebal lapis yang memuat bagian dalam (≈ tebal Z objek), µm.
    thickness_um: f64,
}

fn inside_stats(inner: &dyn SdfProvider) -> InsideStats {
    let grid = inner.grid();
    let [nx, ny, nz] = grid.dims();
    let area = grid.s_xy * grid.s_xy;
    let per_layer: Vec<(f32, f64, f64)> = (0..nz)
        .into_par_iter()
        .map(|k| {
            let mut buf = vec![0.0f32; nx * ny];
            inner.fill_layer(k, &mut buf);
            let min = buf.iter().copied().fold(f32::INFINITY, f32::min);
            let inside = buf.iter().filter(|v| **v < 0.0).count() as f64;
            let t = layer_thickness(&grid.z_um, k).max(0.0);
            (min, inside * area * t, if inside > 0.0 { t } else { 0.0 })
        })
        .collect();
    per_layer.into_iter().fold(
        InsideStats {
            min_sdf: f32::INFINITY,
            volume_um3: 0.0,
            thickness_um: 0.0,
        },
        |a, (m, v, t)| InsideStats {
            min_sdf: a.min_sdf.min(m),
            volume_um3: a.volume_um3 + v,
            thickness_um: a.thickness_um + t,
        },
    )
}

/// Ketukan Z per lapis: lapis `j` dengan `|z_j − z_k| ≤ 3σ`, bobot Gaussian
/// atas jarak µm sebenarnya, dinormalkan.
fn z_taps(z_um: &[f64], sigma: f64) -> Vec<Vec<(usize, f32)>> {
    (0..z_um.len())
        .map(|k| {
            if sigma.is_nan() || sigma <= 1e-9 {
                return vec![(k, 1.0)];
            }
            let zk = z_um[k];
            let weight = |zj: f64| -> Option<f64> {
                let d = zj - zk;
                (d.abs() <= TRUNCATE * sigma).then(|| (-0.5 * (d / sigma).powi(2)).exp())
            };
            let mut taps: Vec<(usize, f64)> = Vec::new();
            // Lapis terurut naik: sapu ke bawah lalu ke atas dari k.
            for j in (0..=k).rev() {
                match weight(z_um[j]) {
                    Some(w) => taps.push((j, w)),
                    None => break,
                }
            }
            for (j, &zj) in z_um.iter().enumerate().skip(k + 1) {
                match weight(zj) {
                    Some(w) => taps.push((j, w)),
                    None => break,
                }
            }
            taps.sort_by_key(|t| t.0);
            let s: f64 = taps.iter().map(|t| t.1).sum();
            taps.into_iter().map(|(j, w)| (j, (w / s) as f32)).collect()
        })
        .collect()
}

impl SdfProvider for SmoothedSdf<'_> {
    fn grid(&self) -> &StackGrid {
        self.inner.grid()
    }

    fn fill_layer(&self, k: usize, out: &mut [f32]) {
        let Some(taps) = self.taps_z.get(k).filter(|t| t.len() > 1) else {
            self.inner.fill_layer(k, out);
            return;
        };
        let mut layer = vec![0.0f32; out.len()];
        out.fill(0.0);
        for &(j, w) in taps {
            self.inner.fill_layer(j, &mut layer);
            for (o, &v) in out.iter_mut().zip(&layer) {
                *o += w * v;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recon::SphereSdf;

    fn grid(n: u32, s: f64, sz: f64) -> StackGrid {
        StackGrid {
            origin_um: [0.0, 0.0],
            s_xy: s,
            size: [n, n],
            z_um: (0..n).map(|k| (k as f64 + 0.5) * sz).collect(),
        }
    }

    #[test]
    fn z_taps_follow_true_spacing() {
        let z = [0.0, 10.0, 100.0, 110.0];
        let t = z_taps(&z, 5.0);
        // σ = 5 µm: lapis 100 µm jauh di luar 3σ dari lapis 10 µm.
        assert_eq!(t[1].iter().map(|x| x.0).collect::<Vec<_>>(), vec![0, 1]);
        for taps in &t {
            let s: f32 = taps.iter().map(|x| x.1).sum();
            assert!((s - 1.0).abs() < 1e-5);
        }
        assert_eq!(z_taps(&z, 0.0)[2], vec![(2, 1.0)]);
    }

    #[test]
    fn layer_thickness_handles_ends_and_nonuniform_spacing() {
        let z = [0.0, 10.0, 30.0];
        assert_eq!(layer_thickness(&z, 0), 10.0);
        assert_eq!(layer_thickness(&z, 1), 15.0);
        assert_eq!(layer_thickness(&z, 2), 20.0);
        assert_eq!(layer_thickness(&[5.0], 0), 0.0);
    }

    #[test]
    fn constant_field_is_unchanged() {
        struct Const(StackGrid);
        impl SdfProvider for Const {
            fn grid(&self) -> &StackGrid {
                &self.0
            }
            fn fill_layer(&self, _k: usize, out: &mut [f32]) {
                out.fill(-3.5);
            }
        }
        let c = Const(grid(12, 4.0, 6.0));
        let s = SmoothedSdf::new(&c, 6.0, 100.0);
        let mut out = vec![0.0; 144];
        s.fill_layer(5, &mut out);
        assert!(out.iter().all(|v| (v + 3.5).abs() < 1e-5));
    }

    #[test]
    fn sigma_is_capped_by_inscribed_radius_and_volume_is_reported() {
        let g = grid(40, 4.0, 4.0);
        let sph = SphereSdf {
            grid: g,
            center_um: [80.0, 80.0, 80.0],
            radius_um: 10.0,
        };
        // Bola r = 10 µm: sigma 40 µm dijepit ke ≈ 0,5 × 10.
        let s = SmoothedSdf::new(&sph, 40.0, 100.0);
        assert!(s.sigma_um() <= 5.0 + 1e-6, "{}", s.sigma_um());
        assert!(s.sigma_um() > 3.0, "{}", s.sigma_um());
        let v = 4.0 / 3.0 * std::f64::consts::PI * 1000.0;
        let e = (s.target_volume_um3() - v).abs() / v;
        assert!(e < 0.25, "volume {} vs {v}", s.target_volume_um3());
    }

    #[test]
    fn sphere_zero_level_stays_close() {
        let g = grid(40, 4.0, 4.0);
        let sph = SphereSdf {
            grid: g,
            center_um: [80.0, 80.0, 80.0],
            radius_um: 50.0,
        };
        let s = SmoothedSdf::new(&sph, 4.0, 100.0);
        let mut a = vec![0.0; 1600];
        let mut b = vec![0.0; 1600];
        sph.fill_layer(20, &mut a);
        s.fill_layer(20, &mut b);
        for (x, y) in a.iter().zip(&b) {
            if x.abs() < 8.0 {
                assert!((x - y).abs() < 1.5, "{x} vs {y}");
            }
        }
    }
}
