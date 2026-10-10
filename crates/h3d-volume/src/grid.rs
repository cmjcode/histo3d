//! Kisi tumpukan `G` (histo3d.md 5.2, 8.6.1).
//!
//! Konvensi: `w = origin + (i + 0,5) × s_xy` untuk sumbu X/Y (pusat voxel di
//! +0,5) dan `z = z_um[k]` untuk lapis ke-`k`. Indeks linear buffer 3D:
//! `x + nx·(y + ny·z)` (x tercepat), sama dengan `h3d-imgproc`.

use serde::{Deserialize, Serialize};

/// Kisi volume di ruang W (µm).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StackGrid {
    /// Sudut kiri-atas kisi di ruang W (µm): tepi luar voxel `(0, 0)`.
    pub origin_um: [f64; 2],
    /// Ukuran voxel XY, µm (bawaan 16).
    pub s_xy: f64,
    /// Jumlah voxel `[nx, ny]`.
    pub size: [u32; 2],
    /// Koordinat z tiap lapis keluaran, µm, menaik.
    pub z_um: Vec<f64>,
}

impl StackGrid {
    /// Kisi dari kotak pembatas W `[xmin, ymin, xmax, ymax]` yang diperlebar
    /// `2 × band_px × s_xy` di tiap sisi, dengan lapis `z_um`.
    pub fn from_bbox(bbox_w_um: [f64; 4], s_xy: f64, band_px: f64, z_um: Vec<f64>) -> StackGrid {
        let margin = 2.0 * band_px.max(0.0) * s_xy;
        let xmin = bbox_w_um[0] - margin;
        let ymin = bbox_w_um[1] - margin;
        let xmax = bbox_w_um[2] + margin;
        let ymax = bbox_w_um[3] + margin;
        let nx = ((xmax - xmin) / s_xy).ceil().max(1.0) as u32;
        let ny = ((ymax - ymin) / s_xy).ceil().max(1.0) as u32;
        StackGrid {
            origin_um: [xmin, ymin],
            s_xy,
            size: [nx, ny],
            z_um,
        }
    }

    /// Jumlah lapis Z.
    pub fn nz(&self) -> usize {
        self.z_um.len()
    }

    /// Jumlah voxel per lapis.
    pub fn layer_len(&self) -> usize {
        self.size[0] as usize * self.size[1] as usize
    }

    /// Jumlah voxel seluruh kisi.
    pub fn voxel_count(&self) -> usize {
        self.layer_len() * self.nz()
    }

    /// Dimensi `[nx, ny, nz]` untuk fungsi 3D `h3d-imgproc`.
    pub fn dims(&self) -> [usize; 3] {
        [self.size[0] as usize, self.size[1] as usize, self.nz()]
    }

    /// Indeks linear voxel `(x, y, z)`.
    #[inline]
    pub fn linear(&self, x: usize, y: usize, z: usize) -> usize {
        x + self.size[0] as usize * (y + self.size[1] as usize * z)
    }

    /// Kebalikan [`Self::linear`]: `(x, y, z)` dari indeks linear.
    #[inline]
    pub fn delinear(&self, i: usize) -> [usize; 3] {
        let nx = self.size[0] as usize;
        let ny = self.size[1] as usize;
        [i % nx, (i / nx) % ny, i / (nx * ny)]
    }

    /// Pusat voxel `(i, j)` di ruang W: `origin + (i + 0,5) × s_xy`.
    #[inline]
    pub fn index_to_um(&self, i: u32, j: u32) -> [f64; 2] {
        [
            self.origin_um[0] + (i as f64 + 0.5) * self.s_xy,
            self.origin_um[1] + (j as f64 + 0.5) * self.s_xy,
        ]
    }

    /// Pusat voxel `(x, y, z)` dalam µm (tiga sumbu).
    #[inline]
    pub fn center_um(&self, x: usize, y: usize, z: usize) -> [f64; 3] {
        let xy = self.index_to_um(x as u32, y as u32);
        [xy[0], xy[1], self.z_um.get(z).copied().unwrap_or(0.0)]
    }

    /// Indeks voxel yang memuat titik W `w`; `None` bila di luar kisi.
    pub fn um_to_index(&self, w: [f64; 2]) -> Option<[u32; 2]> {
        let fx = (w[0] - self.origin_um[0]) / self.s_xy;
        let fy = (w[1] - self.origin_um[1]) / self.s_xy;
        if !(fx >= 0.0 && fy >= 0.0) {
            return None;
        }
        let i = fx.floor();
        let j = fy.floor();
        if i >= self.size[0] as f64 || j >= self.size[1] as f64 {
            return None;
        }
        Some([i as u32, j as u32])
    }

    /// Koordinat kisi kontinu `(gx, gy)` dengan pusat voxel pada bilangan
    /// bulat: `gx = (w − origin)/s_xy − 0,5`.
    #[inline]
    pub fn um_to_grid_continuous(&self, w: [f64; 2]) -> [f64; 2] {
        [
            (w[0] - self.origin_um[0]) / self.s_xy - 0.5,
            (w[1] - self.origin_um[1]) / self.s_xy - 0.5,
        ]
    }

    /// Jarak irisan setempat lapis `k`: `(z_{k+1} − z_{k−1}) / 2`; di ujung
    /// memakai jarak ke satu-satunya tetangga; satu lapis → `s_xy`.
    pub fn local_dz(&self, k: usize) -> f64 {
        let n = self.nz();
        if n < 2 || k >= n {
            return self.s_xy;
        }
        if k == 0 {
            self.z_um[1] - self.z_um[0]
        } else if k == n - 1 {
            self.z_um[n - 1] - self.z_um[n - 2]
        } else {
            (self.z_um[k + 1] - self.z_um[k - 1]) / 2.0
        }
    }

    /// Volume satu voxel pada lapis `k`: `s_xy² × local_dz(k)`.
    pub fn voxel_volume_um3(&self, k: usize) -> f64 {
        self.s_xy * self.s_xy * self.local_dz(k)
    }

    /// Median jarak antar lapis berurutan; `s_xy` bila kurang dari dua lapis.
    pub fn median_dz(&self) -> f64 {
        if self.nz() < 2 {
            return self.s_xy;
        }
        let mut d: Vec<f64> = self.z_um.windows(2).map(|w| w[1] - w[0]).collect();
        d.sort_by(|a, b| a.total_cmp(b));
        let m = d.len() / 2;
        if d.len() % 2 == 1 {
            d[m]
        } else {
            (d[m - 1] + d[m]) / 2.0
        }
    }

    /// Benar bila semua jarak antar lapis sama dalam toleransi `tol` (µm).
    pub fn is_uniform_z(&self, tol: f64) -> bool {
        if self.nz() < 3 {
            return true;
        }
        let m = self.median_dz();
        self.z_um
            .windows(2)
            .all(|w| ((w[1] - w[0]) - m).abs() <= tol)
    }

    /// Jarak antar lapis dalam µm pada posisi lapis kontinu `kf` (interpolasi
    /// linear antara `z_um[floor]` dan `z_um[floor + 1]`).
    pub fn z_at_continuous(&self, kf: f64) -> f64 {
        let n = self.nz();
        if n == 0 {
            return 0.0;
        }
        if kf <= 0.0 {
            return self.z_um[0] + kf * self.local_dz(0);
        }
        let k0 = kf.floor() as usize;
        if k0 + 1 >= n {
            return self.z_um[n - 1] + (kf - (n - 1) as f64) * self.local_dz(n - 1);
        }
        let t = kf - k0 as f64;
        self.z_um[k0] + t * (self.z_um[k0 + 1] - self.z_um[k0])
    }

    /// Posisi lapis kontinu untuk `z` (kebalikan [`Self::z_at_continuous`]);
    /// `None` bila `z` di luar `[z_0, z_{n−1}]` lebih dari setengah jarak lapis.
    pub fn z_to_continuous(&self, z: f64) -> Option<f64> {
        let n = self.nz();
        if n == 0 {
            return None;
        }
        if n == 1 {
            return if (z - self.z_um[0]).abs() <= self.local_dz(0) / 2.0 {
                Some(0.0)
            } else {
                None
            };
        }
        if z < self.z_um[0] {
            return if self.z_um[0] - z <= self.local_dz(0) / 2.0 {
                Some(0.0)
            } else {
                None
            };
        }
        if z > self.z_um[n - 1] {
            return if z - self.z_um[n - 1] <= self.local_dz(n - 1) / 2.0 {
                Some((n - 1) as f64)
            } else {
                None
            };
        }
        // Cari k dengan z_um[k] ≤ z ≤ z_um[k+1].
        let k = match self.z_um.binary_search_by(|p| p.total_cmp(&z)) {
            Ok(k) => return Some(k as f64),
            Err(k) => k.saturating_sub(1),
        };
        let dz = self.z_um[k + 1] - self.z_um[k];
        if dz <= 0.0 {
            return Some(k as f64);
        }
        Some(k as f64 + (z - self.z_um[k]) / dz)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_um_roundtrip() {
        let g = StackGrid {
            origin_um: [100.0, -50.0],
            s_xy: 16.0,
            size: [10, 8],
            z_um: vec![0.0, 50.0, 100.0],
        };
        assert_eq!(g.index_to_um(0, 0), [108.0, -42.0]);
        assert_eq!(g.um_to_index([108.0, -42.0]), Some([0, 0]));
        assert_eq!(g.um_to_index([99.9, 0.0]), None);
        assert_eq!(g.um_to_index([100.0 + 160.0, 0.0]), None);
        assert_eq!(g.um_to_index([100.0 + 159.9, 0.0]), Some([9, 3]));
        assert_eq!(g.voxel_volume_um3(1), 16.0 * 16.0 * 50.0);
        assert_eq!(g.voxel_volume_um3(0), 16.0 * 16.0 * 50.0);
        assert_eq!(g.median_dz(), 50.0);
        assert!(g.is_uniform_z(1e-9));
        let i = g.linear(3, 2, 1);
        assert_eq!(g.delinear(i), [3, 2, 1]);
    }

    #[test]
    fn from_bbox_adds_margin() {
        let g = StackGrid::from_bbox([0.0, 0.0, 100.0, 50.0], 10.0, 2.0, vec![0.0]);
        assert_eq!(g.origin_um, [-40.0, -40.0]);
        assert_eq!(g.size, [18, 13]);
    }

    #[test]
    fn z_continuous() {
        let g = StackGrid {
            origin_um: [0.0, 0.0],
            s_xy: 16.0,
            size: [1, 1],
            z_um: vec![0.0, 40.0, 100.0],
        };
        assert_eq!(g.z_at_continuous(0.5), 20.0);
        assert_eq!(g.z_at_continuous(1.5), 70.0);
        assert_eq!(g.z_to_continuous(70.0), Some(1.5));
        assert_eq!(g.z_to_continuous(40.0), Some(1.0));
        assert_eq!(g.z_to_continuous(-100.0), None);
    }
}
