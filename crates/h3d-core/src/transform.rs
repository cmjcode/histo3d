//! Transformasi per irisan `T_i` (U → W) (8.1).

use serde::{Deserialize, Serialize};

use crate::affine::Affine2;

/// Kisi titik kendali B-spline kubik di ruang W (fase H8). Perpindahan dalam µm.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DisplacementGrid {
    /// Posisi titik kendali (0, 0) di ruang W, µm.
    pub origin_um: [f64; 2],
    /// Jarak antar titik kendali, µm.
    pub spacing_um: [f64; 2],
    /// Jumlah titik kendali `[nx, ny]`.
    pub size: [u32; 2],
    /// Perpindahan x per titik kendali (urutan baris, `ny × nx`).
    pub dx: Vec<f32>,
    /// Perpindahan y per titik kendali.
    pub dy: Vec<f32>,
}

/// Basis B-spline kubik seragam, `t ∈ [0, 1)`.
#[inline]
fn bspline_basis(t: f64) -> [f64; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        (1.0 - t).powi(3) / 6.0,
        (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
        (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0,
        t3 / 6.0,
    ]
}

impl DisplacementGrid {
    /// Kisi nol (tanpa deformasi).
    pub fn zeros(origin_um: [f64; 2], spacing_um: [f64; 2], size: [u32; 2]) -> Self {
        let n = (size[0] as usize) * (size[1] as usize);
        Self {
            origin_um,
            spacing_um,
            size,
            dx: vec![0.0; n],
            dy: vec![0.0; n],
        }
    }

    /// Benar bila ukuran buffer sesuai `size`.
    pub fn is_consistent(&self) -> bool {
        let n = (self.size[0] as usize) * (self.size[1] as usize);
        self.dx.len() == n
            && self.dy.len() == n
            && self.spacing_um.iter().all(|s| *s > 0.0 && s.is_finite())
    }

    #[inline]
    fn ctrl(&self, ix: i64, iy: i64) -> (f64, f64) {
        // Di luar kisi: perpindahan nol (jepit ke tepi dengan nilai nol).
        if ix < 0 || iy < 0 || ix >= self.size[0] as i64 || iy >= self.size[1] as i64 {
            return (0.0, 0.0);
        }
        let i = (iy as usize) * (self.size[0] as usize) + ix as usize;
        (self.dx[i] as f64, self.dy[i] as f64)
    }

    /// Perpindahan (µm) di titik `w` (ruang W) lewat B-spline kubik.
    pub fn displacement_at(&self, w: [f64; 2]) -> [f64; 2] {
        let gx = (w[0] - self.origin_um[0]) / self.spacing_um[0];
        let gy = (w[1] - self.origin_um[1]) / self.spacing_um[1];
        let ix = gx.floor();
        let iy = gy.floor();
        let bx = bspline_basis(gx - ix);
        let by = bspline_basis(gy - iy);
        let mut dx = 0.0;
        let mut dy = 0.0;
        for (j, wy) in by.iter().enumerate() {
            for (i, wx) in bx.iter().enumerate() {
                let (cx, cy) = self.ctrl(ix as i64 + i as i64 - 1, iy as i64 + j as i64 - 1);
                let wgt = wx * wy;
                dx += wgt * cx;
                dy += wgt * cy;
            }
        }
        [dx, dy]
    }
}

/// Transformasi irisan `T_i`: memetakan ruang U (slide, µm) ke ruang W
/// (tumpukan, µm). Arah maju = `apply`; arah mundur = `apply_inverse`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SlideTransform {
    /// Irisan acuan atau belum didaftarkan.
    #[default]
    Identity,
    /// Affine global.
    Affine(Affine2),
    /// Affine global diikuti perpindahan lokal B-spline di ruang W:
    /// `w = A(u) + D(A(u))`.
    AffineBSpline {
        /// Bagian affine.
        affine: Affine2,
        /// Perpindahan lokal.
        grid: DisplacementGrid,
    },
}

impl SlideTransform {
    /// Bagian affine (identitas bila `Identity`).
    pub fn affine(&self) -> Affine2 {
        match self {
            SlideTransform::Identity => Affine2::IDENTITY,
            SlideTransform::Affine(a) => *a,
            SlideTransform::AffineBSpline { affine, .. } => *affine,
        }
    }

    /// Benar bila bukan `Identity`.
    pub fn is_registered(&self) -> bool {
        !matches!(self, SlideTransform::Identity)
    }

    /// Maju: U → W.
    pub fn apply(&self, u: [f64; 2]) -> [f64; 2] {
        match self {
            SlideTransform::Identity => u,
            SlideTransform::Affine(a) => a.apply(u),
            SlideTransform::AffineBSpline { affine, grid } => {
                let w = affine.apply(u);
                let d = grid.displacement_at(w);
                [w[0] + d[0], w[1] + d[1]]
            }
        }
    }

    /// Mundur: W → U. Untuk B-spline memakai iterasi titik tetap (perpindahan
    /// lokal kecil dibanding jarak titik kendali, konvergen cepat). `None`
    /// bila affine singular.
    pub fn apply_inverse(&self, w: [f64; 2]) -> Option<[f64; 2]> {
        match self {
            SlideTransform::Identity => Some(w),
            SlideTransform::Affine(a) => Some(a.inverse()?.apply(w)),
            SlideTransform::AffineBSpline { affine, grid } => {
                let inv = affine.inverse()?;
                // Cari w' sehingga w' + D(w') = w.
                let mut wp = w;
                for _ in 0..8 {
                    let d = grid.displacement_at(wp);
                    let next = [w[0] - d[0], w[1] - d[1]];
                    let delta = (next[0] - wp[0]).abs() + (next[1] - wp[1]).abs();
                    wp = next;
                    if delta < 1e-4 {
                        break;
                    }
                }
                Some(inv.apply(wp))
            }
        }
    }

    /// Affine balikan (W → U), bila ada. Untuk B-spline hanya bagian affine.
    pub fn affine_inverse(&self) -> Option<Affine2> {
        self.affine().inverse()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_roundtrip() {
        let t = SlideTransform::Identity;
        assert_eq!(t.apply([3.0, 4.0]), [3.0, 4.0]);
        assert_eq!(t.apply_inverse([3.0, 4.0]), Some([3.0, 4.0]));
    }

    #[test]
    fn affine_roundtrip() {
        let t = SlideTransform::Affine(Affine2::rotation_about(0.4, [100.0, 50.0]));
        let w = t.apply([10.0, 20.0]);
        let u = t.apply_inverse(w).unwrap();
        assert!((u[0] - 10.0).abs() < 1e-9 && (u[1] - 20.0).abs() < 1e-9);
    }

    #[test]
    fn bspline_zero_grid_equals_affine() {
        let a = Affine2::translation(5.0, -7.0);
        let grid = DisplacementGrid::zeros([0.0, 0.0], [1000.0, 1000.0], [8, 8]);
        let t = SlideTransform::AffineBSpline { affine: a, grid };
        assert_eq!(t.apply([100.0, 100.0]), a.apply([100.0, 100.0]));
    }

    #[test]
    fn bspline_partition_of_unity_and_inverse() {
        let mut grid = DisplacementGrid::zeros([0.0, 0.0], [1000.0, 1000.0], [8, 8]);
        // Perpindahan seragam 30 µm → di tengah kisi hasilnya tepat 30.
        for v in grid.dx.iter_mut() {
            *v = 30.0;
        }
        let d = grid.displacement_at([3500.0, 3500.0]);
        assert!((d[0] - 30.0).abs() < 1e-6 && d[1].abs() < 1e-9);
        let t = SlideTransform::AffineBSpline {
            affine: Affine2::IDENTITY,
            grid,
        };
        let w = t.apply([3500.0, 3500.0]);
        let u = t.apply_inverse(w).unwrap();
        assert!((u[0] - 3500.0).abs() < 1e-3 && (u[1] - 3500.0).abs() < 1e-3);
    }
}
