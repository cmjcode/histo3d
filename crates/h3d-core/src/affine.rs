//! Transformasi affine 2D dalam mikrometer (histo3d.md 5.4).
//!
//! ```text
//! [ a  b  tx ]
//! [ c  d  ty ]   (x, y) → (a·x + b·y + tx, c·x + d·y + ty)
//! ```
//!
//! Arah transformasi tidak melekat pada tipe ini; tiap fungsi yang memakai
//! `Affine2` wajib menyebut arahnya di doc comment (aturan keras 4).

use serde::{Deserialize, Serialize};

use crate::error::CoreError;

/// Transformasi affine 2D. Urutan elemen `m`: `a, b, tx, c, d, ty`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Affine2 {
    /// `[a, b, tx, c, d, ty]`.
    pub m: [f64; 6],
}

/// Hasil dekomposisi `Affine2` untuk tampilan dan QC.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AffineParts {
    /// Rotasi dalam radian, `(-π, π]`.
    pub rotation: f64,
    /// Skala sepanjang sumbu x dan y (setelah rotasi), selalu positif.
    pub scale: [f64; 2],
    /// Shear (koefisien y → x), tanpa satuan.
    pub shear: f64,
    /// Benar bila ada pencerminan (determinan negatif).
    pub mirrored: bool,
    /// Translasi dalam µm.
    pub translation: [f64; 2],
}

/// Ambang determinan di bawahnya affine dianggap singular.
pub const SINGULAR_EPS: f64 = 1e-12;

impl Default for Affine2 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Affine2 {
    /// Identitas.
    pub const IDENTITY: Affine2 = Affine2 {
        m: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    };

    /// Dari elemen `[a, b, tx, c, d, ty]`.
    pub const fn new(m: [f64; 6]) -> Self {
        Self { m }
    }

    /// Translasi murni.
    pub const fn translation(tx: f64, ty: f64) -> Self {
        Self {
            m: [1.0, 0.0, tx, 0.0, 1.0, ty],
        }
    }

    /// Skala murni di sekitar asal.
    pub const fn scale(sx: f64, sy: f64) -> Self {
        Self {
            m: [sx, 0.0, 0.0, 0.0, sy, 0.0],
        }
    }

    /// Rotasi `theta` radian (berlawanan arah jarum jam pada sumbu y ke bawah
    /// tampak searah jarum jam di layar) di sekitar asal.
    pub fn rotation(theta: f64) -> Self {
        let (s, c) = theta.sin_cos();
        Self {
            m: [c, -s, 0.0, s, c, 0.0],
        }
    }

    /// Rotasi `theta` radian di sekitar titik `center` (µm).
    pub fn rotation_about(theta: f64, center: [f64; 2]) -> Self {
        Self::translation(center[0], center[1])
            .compose(&Self::rotation(theta))
            .compose(&Self::translation(-center[0], -center[1]))
    }

    /// Cermin sumbu x (x → −x) di sekitar asal.
    pub const fn mirror_x() -> Self {
        Self {
            m: [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        }
    }

    /// Cermin sumbu x di sekitar garis vertikal `x = cx`.
    pub fn mirror_x_about(cx: f64) -> Self {
        Self::translation(cx, 0.0)
            .compose(&Self::mirror_x())
            .compose(&Self::translation(-cx, 0.0))
    }

    /// Terapkan ke titik.
    #[inline]
    pub fn apply(&self, p: [f64; 2]) -> [f64; 2] {
        let [a, b, tx, c, d, ty] = self.m;
        [a * p[0] + b * p[1] + tx, c * p[0] + d * p[1] + ty]
    }

    /// Terapkan hanya bagian linear (tanpa translasi), untuk vektor arah.
    #[inline]
    pub fn apply_vector(&self, v: [f64; 2]) -> [f64; 2] {
        let [a, b, _, c, d, _] = self.m;
        [a * v[0] + b * v[1], c * v[0] + d * v[1]]
    }

    /// `self ∘ other`: terapkan `other` dulu, lalu `self`.
    pub fn compose(&self, other: &Affine2) -> Affine2 {
        let [a1, b1, tx1, c1, d1, ty1] = self.m;
        let [a2, b2, tx2, c2, d2, ty2] = other.m;
        Affine2 {
            m: [
                a1 * a2 + b1 * c2,
                a1 * b2 + b1 * d2,
                a1 * tx2 + b1 * ty2 + tx1,
                c1 * a2 + d1 * c2,
                c1 * b2 + d1 * d2,
                c1 * tx2 + d1 * ty2 + ty1,
            ],
        }
    }

    /// Determinan bagian linear. Negatif = pencerminan (sah, 5.4).
    #[inline]
    pub fn determinant(&self) -> f64 {
        let [a, b, _, c, d, _] = self.m;
        a * d - b * c
    }

    /// Balikan. `None` bila determinan mendekati nol.
    pub fn inverse(&self) -> Option<Affine2> {
        let det = self.determinant();
        if !det.is_finite() || det.abs() < SINGULAR_EPS {
            return None;
        }
        let [a, b, tx, c, d, ty] = self.m;
        let ia = d / det;
        let ib = -b / det;
        let ic = -c / det;
        let id = a / det;
        Some(Affine2 {
            m: [ia, ib, -(ia * tx + ib * ty), ic, id, -(ic * tx + id * ty)],
        })
    }

    /// Seperti `inverse` tetapi mengembalikan error yang bisa dilaporkan.
    pub fn try_inverse(&self) -> Result<Affine2, CoreError> {
        self.inverse().ok_or(CoreError::SingularAffine {
            det: self.determinant(),
        })
    }

    /// Translasi `[tx, ty]`.
    #[inline]
    pub fn translation_part(&self) -> [f64; 2] {
        [self.m[2], self.m[5]]
    }

    /// Bagian linear sebagai `[a, b, c, d]`.
    #[inline]
    pub fn linear_part(&self) -> [f64; 4] {
        [self.m[0], self.m[1], self.m[3], self.m[4]]
    }

    /// Dekomposisi `L = R · (cermin) · [sx, shear·sy; 0, sy]` (QR 2×2).
    pub fn decompose(&self) -> AffineParts {
        let [a, b, tx, c, d, ty] = self.m;
        let det = a * d - b * c;
        let mirrored = det < 0.0;
        // Bila cermin, buang cermin dari kolom pertama supaya rotasi murni.
        let (a, c) = if mirrored { (-a, -c) } else { (a, c) };
        let sx = (a * a + c * c).sqrt();
        let rotation = c.atan2(a);
        let (sin_r, cos_r) = rotation.sin_cos();
        // Kolom kedua dalam kerangka yang sudah diputar balik.
        let b2 = cos_r * b + sin_r * d;
        let d2 = -sin_r * b + cos_r * d;
        let sy = d2.abs();
        let shear = if sy > 0.0 { b2 / sy } else { 0.0 };
        AffineParts {
            rotation,
            scale: [sx, sy],
            shear,
            mirrored,
            translation: [tx, ty],
        }
    }

    /// Benar bila semua elemen finit.
    pub fn is_finite(&self) -> bool {
        self.m.iter().all(|v| v.is_finite())
    }

    /// Benar bila selisih tiap elemen ≤ `eps`.
    pub fn approx_eq(&self, other: &Affine2, eps: f64) -> bool {
        self.m
            .iter()
            .zip(other.m.iter())
            .all(|(x, y)| (x - y).abs() <= eps)
    }

    /// Affine kuadrat terkecil dari ≥ 3 pasang titik `(src, dst)`:
    /// hasilnya memetakan `src → dst`. Dipakai landmark manual (8.4.5).
    pub fn fit_affine(pairs: &[([f64; 2], [f64; 2])]) -> Result<Affine2, CoreError> {
        if pairs.len() < 3 {
            return Err(CoreError::NotEnoughLandmarks {
                needed: 3,
                given: pairs.len(),
            });
        }
        // Normal equations: A^T A x = A^T b untuk x = [a, b, tx] dan [c, d, ty].
        let mut ata = [[0.0f64; 3]; 3];
        let mut atb_x = [0.0f64; 3];
        let mut atb_y = [0.0f64; 3];
        for (s, t) in pairs {
            let row = [s[0], s[1], 1.0];
            for i in 0..3 {
                for j in 0..3 {
                    ata[i][j] += row[i] * row[j];
                }
                atb_x[i] += row[i] * t[0];
                atb_y[i] += row[i] * t[1];
            }
        }
        let x = solve3(ata, atb_x).ok_or(CoreError::DegenerateLandmarks)?;
        let y = solve3(ata, atb_y).ok_or(CoreError::DegenerateLandmarks)?;
        let aff = Affine2 {
            m: [x[0], x[1], x[2], y[0], y[1], y[2]],
        };
        if !aff.is_finite() || aff.determinant().abs() < SINGULAR_EPS {
            return Err(CoreError::DegenerateLandmarks);
        }
        Ok(aff)
    }

    /// Similarity (rotasi + skala seragam + translasi, tanpa cermin) kuadrat
    /// terkecil dari ≥ 2 pasang titik; memetakan `src → dst`.
    pub fn fit_similarity(pairs: &[([f64; 2], [f64; 2])]) -> Result<Affine2, CoreError> {
        if pairs.len() < 2 {
            return Err(CoreError::NotEnoughLandmarks {
                needed: 2,
                given: pairs.len(),
            });
        }
        // Model: dst = [p -q; q p] src + [tx ty]. Linear dalam (p, q, tx, ty).
        let n = pairs.len() as f64;
        let (mut sx, mut sy, mut tx_, mut ty_) = (0.0, 0.0, 0.0, 0.0);
        let (mut sxx, mut sxt, mut syt) = (0.0, 0.0, 0.0);
        for (s, t) in pairs {
            sx += s[0];
            sy += s[1];
            tx_ += t[0];
            ty_ += t[1];
            sxx += s[0] * s[0] + s[1] * s[1];
            sxt += s[0] * t[0] + s[1] * t[1];
            syt += s[0] * t[1] - s[1] * t[0];
        }
        // Pusatkan.
        let mx = sx / n;
        let my = sy / n;
        let mtx = tx_ / n;
        let mty = ty_ / n;
        let var = sxx - n * (mx * mx + my * my);
        if var.abs() < SINGULAR_EPS {
            return Err(CoreError::DegenerateLandmarks);
        }
        let p = (sxt - n * (mx * mtx + my * mty)) / var;
        let q = (syt - n * (mx * mty - my * mtx)) / var;
        let tx = mtx - (p * mx - q * my);
        let ty = mty - (q * mx + p * my);
        let aff = Affine2 {
            m: [p, -q, tx, q, p, ty],
        };
        if !aff.is_finite() || aff.determinant().abs() < SINGULAR_EPS {
            return Err(CoreError::DegenerateLandmarks);
        }
        Ok(aff)
    }

    /// Galat RMS (µm) pemetaan `self(src)` terhadap `dst`.
    pub fn rms_error(&self, pairs: &[([f64; 2], [f64; 2])]) -> f64 {
        if pairs.is_empty() {
            return 0.0;
        }
        let sum: f64 = pairs
            .iter()
            .map(|(s, t)| {
                let p = self.apply(*s);
                (p[0] - t[0]).powi(2) + (p[1] - t[1]).powi(2)
            })
            .sum();
        (sum / pairs.len() as f64).sqrt()
    }

    /// Kotak pembatas sumbu-sejajar dari bayangan persegi `[0,w]×[0,h]` (µm).
    pub fn transform_rect(&self, w: f64, h: f64) -> ([f64; 2], [f64; 2]) {
        let corners = [
            self.apply([0.0, 0.0]),
            self.apply([w, 0.0]),
            self.apply([0.0, h]),
            self.apply([w, h]),
        ];
        let mut lo = [f64::INFINITY; 2];
        let mut hi = [f64::NEG_INFINITY; 2];
        for c in corners {
            for k in 0..2 {
                lo[k] = lo[k].min(c[k]);
                hi[k] = hi[k].max(c[k]);
            }
        }
        (lo, hi)
    }
}

/// Selesaikan sistem 3×3 dengan eliminasi Gauss berpivot parsial.
fn solve3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for col in 0..3 {
        let mut piv = col;
        for r in (col + 1)..3 {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if a[piv][col].abs() < 1e-300 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for r in (col + 1)..3 {
            let f = a[r][col] / a[col][col];
            let pivot_row = a[col];
            for (k, v) in a[r].iter_mut().enumerate().skip(col) {
                *v -= f * pivot_row[k];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = [0.0; 3];
    for i in (0..3).rev() {
        let mut s = b[i];
        for k in (i + 1)..3 {
            s -= a[i][k] * x[k];
        }
        if a[i][i].abs() < 1e-300 {
            return None;
        }
        x[i] = s / a[i][i];
    }
    if x.iter().all(|v| v.is_finite()) {
        Some(x)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn arb_affine() -> impl Strategy<Value = Affine2> {
        (
            -3.0f64..3.0,
            -3.0f64..3.0,
            -1e4f64..1e4,
            -3.0f64..3.0,
            -3.0f64..3.0,
            -1e4f64..1e4,
        )
            .prop_map(|(a, b, tx, c, d, ty)| Affine2::new([a, b, tx, c, d, ty]))
            .prop_filter("non-singular", |m| m.determinant().abs() > 1e-2)
    }

    fn arb_point() -> impl Strategy<Value = [f64; 2]> {
        (-1e4f64..1e4, -1e4f64..1e4).prop_map(|(x, y)| [x, y])
    }

    proptest! {
        #[test]
        fn inverse_of_inverse_is_identity_op(a in arb_affine()) {
            let inv = a.inverse().unwrap();
            let back = inv.inverse().unwrap();
            prop_assert!(a.approx_eq(&back, 1e-6 * (1.0 + a.m.iter().fold(0.0f64, |m, v| m.max(v.abs())))));
        }

        #[test]
        fn compose_with_inverse_is_identity(a in arb_affine(), p in arb_point()) {
            let inv = a.inverse().unwrap();
            let q = a.compose(&inv).apply(p);
            prop_assert!((q[0] - p[0]).abs() < 1e-6 * (1.0 + p[0].abs()));
            prop_assert!((q[1] - p[1]).abs() < 1e-6 * (1.0 + p[1].abs()));
        }

        #[test]
        fn compose_matches_sequential_apply(a in arb_affine(), b in arb_affine(), p in arb_point()) {
            let lhs = a.compose(&b).apply(p);
            let rhs = a.apply(b.apply(p));
            let scale = 1.0 + lhs[0].abs().max(lhs[1].abs());
            prop_assert!((lhs[0] - rhs[0]).abs() < 1e-7 * scale);
            prop_assert!((lhs[1] - rhs[1]).abs() < 1e-7 * scale);
        }

        #[test]
        fn fit_affine_recovers_known_transform(a in arb_affine(), pts in prop::collection::vec(arb_point(), 3..8)) {
            // Titik tidak boleh kolinear; cek lewat luas segitiga pertama.
            let p0 = pts[0]; let p1 = pts[1]; let p2 = pts[2];
            let area = ((p1[0]-p0[0])*(p2[1]-p0[1]) - (p2[0]-p0[0])*(p1[1]-p0[1])).abs();
            prop_assume!(area > 1e3);
            let pairs: Vec<_> = pts.iter().map(|p| (*p, a.apply(*p))).collect();
            let fit = Affine2::fit_affine(&pairs).unwrap();
            prop_assert!(fit.rms_error(&pairs) < 1e-5);
        }

        #[test]
        fn fit_similarity_recovers_known_transform(theta in -3.1f64..3.1, s in 0.5f64..2.0, tx in -1e3f64..1e3, ty in -1e3f64..1e3, pts in prop::collection::vec(arb_point(), 2..6)) {
            prop_assume!((pts[0][0]-pts[1][0]).abs() + (pts[0][1]-pts[1][1]).abs() > 10.0);
            let a = Affine2::translation(tx, ty).compose(&Affine2::rotation(theta)).compose(&Affine2::scale(s, s));
            let pairs: Vec<_> = pts.iter().map(|p| (*p, a.apply(*p))).collect();
            let fit = Affine2::fit_similarity(&pairs).unwrap();
            prop_assert!(fit.approx_eq(&a, 1e-6));
        }
    }

    #[test]
    fn singular_has_no_inverse() {
        assert!(Affine2::scale(0.0, 1.0).inverse().is_none());
        assert!(matches!(
            Affine2::scale(0.0, 1.0).try_inverse(),
            Err(CoreError::SingularAffine { .. })
        ));
    }

    #[test]
    fn mirror_has_negative_determinant_and_decomposes() {
        let m = Affine2::translation(10.0, 20.0)
            .compose(&Affine2::rotation(0.3))
            .compose(&Affine2::mirror_x())
            .compose(&Affine2::scale(2.0, 3.0));
        assert!(m.determinant() < 0.0);
        let parts = m.decompose();
        assert!(parts.mirrored);
        assert!((parts.scale[0] - 2.0).abs() < 1e-9);
        assert!((parts.scale[1] - 3.0).abs() < 1e-9);
        assert!((parts.translation[0] - 10.0).abs() < 1e-9);
    }

    #[test]
    fn decompose_plain_rotation() {
        let m = Affine2::rotation(0.75).compose(&Affine2::scale(1.5, 1.5));
        let p = m.decompose();
        assert!((p.rotation - 0.75).abs() < 1e-9);
        assert!((p.scale[0] - 1.5).abs() < 1e-9);
        assert!(p.shear.abs() < 1e-9);
        assert!(!p.mirrored);
    }

    #[test]
    fn landmark_needs_three_pairs() {
        let pairs = [([0.0, 0.0], [1.0, 1.0]), ([1.0, 0.0], [2.0, 1.0])];
        assert!(matches!(
            Affine2::fit_affine(&pairs),
            Err(CoreError::NotEnoughLandmarks {
                needed: 3,
                given: 2
            })
        ));
    }

    #[test]
    fn collinear_landmarks_rejected() {
        let pairs = [
            ([0.0, 0.0], [0.0, 0.0]),
            ([1.0, 1.0], [1.0, 1.0]),
            ([2.0, 2.0], [2.0, 2.0]),
        ];
        assert!(matches!(
            Affine2::fit_affine(&pairs),
            Err(CoreError::DegenerateLandmarks)
        ));
    }
}
