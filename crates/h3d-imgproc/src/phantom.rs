//! Fantom sintetis deterministik (histo3d.md 11.1): bentuk analitik yang
//! diiris menjadi masker 2D, render "histologi" dan IHC sintetis dengan
//! kebenaran dasar yang diketahui. RNG splitmix64 mandiri, berbenih eksplisit.
//!
//! Konvensi koordinat: `size = [w, h]` piksel; sumbu dunia (µm) berpusat di
//! tengah citra, pusat piksel `(x + 0,5, y + 0,5) × s_xy − size × s_xy / 2`.
//! Bentuk 3D berpusat di `z = 0`.

use ndarray::Array2;

use crate::color::{intensity_of_od, StainMatrix};

/// RNG splitmix64 kecil (deterministik, tanpa crate `rand`).
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    /// Dari benih.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    /// 64 bit berikutnya.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// `f64` seragam di `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// `f64` seragam di `[lo, hi)`.
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
    /// Bilangan bulat seragam di `0..n` (`n > 0`).
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }
}

/// Posisi dunia (µm) pusat piksel `(x, y)`.
#[inline]
fn world_xy(x: usize, y: usize, s_xy: f64, size: [usize; 2]) -> (f64, f64) {
    (
        (x as f64 + 0.5) * s_xy - size[0] as f64 * s_xy / 2.0,
        (y as f64 + 0.5) * s_xy - size[1] as f64 * s_xy / 2.0,
    )
}

/// Masker elips: piksel yang pusatnya memenuhi
/// `((x+0,5−cx)/rx)² + ((y+0,5−cy)/ry)² ≤ 1`. `size = [w, h]`, `center` dan
/// `radii_px` dalam piksel.
pub fn ellipsoid_mask(size: [usize; 2], center: [f64; 2], radii_px: [f64; 2]) -> Array2<u8> {
    let [w, h] = size;
    if !(radii_px[0] > 0.0 && radii_px[1] > 0.0) {
        return Array2::zeros((h, w));
    }
    Array2::from_shape_fn((h, w), |(y, x)| {
        let dx = (x as f64 + 0.5 - center[0]) / radii_px[0];
        let dy = (y as f64 + 0.5 - center[1]) / radii_px[1];
        u8::from(dx * dx + dy * dy <= 1.0)
    })
}

/// Irisan ellipsoid berjari-jari `radii_um = [rx, ry, rz]` pada ketinggian
/// `z_um`, kisi `s_xy` µm/px, citra `size = [w, h]` (ellipsoid di pusat).
pub fn slice_ellipsoid(radii_um: [f64; 3], z_um: f64, s_xy: f64, size: [usize; 2]) -> Array2<u8> {
    let [w, h] = size;
    let rz = radii_um[2];
    if rz.is_nan() || rz <= 0.0 || z_um.abs() >= rz {
        return Array2::zeros((h, w));
    }
    let k = (1.0 - (z_um / rz).powi(2)).sqrt();
    ellipsoid_mask(
        size,
        [w as f64 / 2.0, h as f64 / 2.0],
        [radii_um[0] * k / s_xy, radii_um[1] * k / s_xy],
    )
}

/// Volume analitik ellipsoid (µm³).
pub fn ellipsoid_volume(radii_um: [f64; 3]) -> f64 {
    4.0 / 3.0 * std::f64::consts::PI * radii_um[0] * radii_um[1] * radii_um[2]
}

/// Irisan torus (sumbu z, pusat di asal) dengan jari-jari mayor `major_um`
/// dan minor `minor_um` pada `z_um`.
pub fn slice_torus(
    major_um: f64,
    minor_um: f64,
    z_um: f64,
    s_xy: f64,
    size: [usize; 2],
) -> Array2<u8> {
    let [w, h] = size;
    if minor_um.is_nan() || minor_um <= 0.0 || z_um.abs() >= minor_um {
        return Array2::zeros((h, w));
    }
    let half = (minor_um * minor_um - z_um * z_um).sqrt();
    Array2::from_shape_fn((h, w), |(y, x)| {
        let (px, py) = world_xy(x, y, s_xy, size);
        let rho = (px * px + py * py).sqrt();
        u8::from((rho - major_um).abs() <= half)
    })
}

/// Volume analitik torus (µm³).
pub fn torus_volume(major_um: f64, minor_um: f64) -> f64 {
    2.0 * std::f64::consts::PI.powi(2) * major_um * minor_um * minor_um
}

/// Kapsul 3D: ruas `a–b` ditebalkan jari-jari `r` (µm).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Capsule {
    /// Ujung pertama.
    pub a: [f64; 3],
    /// Ujung kedua.
    pub b: [f64; 3],
    /// Jari-jari.
    pub r: f64,
}

impl Capsule {
    /// Jarak kuadrat titik `p` ke ruas `a–b`.
    fn dist2(&self, p: [f64; 3]) -> f64 {
        let ab = [
            self.b[0] - self.a[0],
            self.b[1] - self.a[1],
            self.b[2] - self.a[2],
        ];
        let ap = [p[0] - self.a[0], p[1] - self.a[1], p[2] - self.a[2]];
        let l2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
        let t = if l2 > 0.0 {
            ((ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / l2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let q = [ap[0] - t * ab[0], ap[1] - t * ab[1], ap[2] - t * ab[2]];
        q[0] * q[0] + q[1] * q[1] + q[2] * q[2]
    }
}

/// Irisan gabungan kapsul pada `z_um`.
pub fn slice_capsules(caps: &[Capsule], z_um: f64, s_xy: f64, size: [usize; 2]) -> Array2<u8> {
    let [w, h] = size;
    Array2::from_shape_fn((h, w), |(y, x)| {
        let (px, py) = world_xy(x, y, s_xy, size);
        u8::from(caps.iter().any(|c| c.dist2([px, py, z_um]) <= c.r * c.r))
    })
}

/// Kapsul-kapsul cabang Y: batang dari `(0,0,−trunk_len)` ke asal, dua lengan
/// sepanjang `arm_len` menyimpang `±half_angle` dari sumbu +z pada bidang xz.
pub fn branch_capsules(
    trunk_len_um: f64,
    arm_len_um: f64,
    radius_um: f64,
    half_angle_rad: f64,
) -> Vec<Capsule> {
    let (s, c) = half_angle_rad.sin_cos();
    vec![
        Capsule {
            a: [0.0, 0.0, -trunk_len_um],
            b: [0.0, 0.0, 0.0],
            r: radius_um,
        },
        Capsule {
            a: [0.0, 0.0, 0.0],
            b: [arm_len_um * s, 0.0, arm_len_um * c],
            r: radius_um,
        },
        Capsule {
            a: [0.0, 0.0, 0.0],
            b: [-arm_len_um * s, 0.0, arm_len_um * c],
            r: radius_um,
        },
    ]
}

/// Irisan cabang Y (lihat [`branch_capsules`]) pada `z_um`.
pub fn slice_branch(
    trunk_len_um: f64,
    arm_len_um: f64,
    radius_um: f64,
    half_angle_rad: f64,
    z_um: f64,
    s_xy: f64,
    size: [usize; 2],
) -> Array2<u8> {
    slice_capsules(
        &branch_capsules(trunk_len_um, arm_len_um, radius_um, half_angle_rad),
        z_um,
        s_xy,
        size,
    )
}

/// Titik-titik acak di dalam masker dengan jarak minimum `min_dist_um`
/// (penolakan dengan kisi percepatan). Mengembalikan posisi µm (koordinat
/// citra: `(x+0,5)·mpp`). Berhenti setelah `target` titik atau usaha habis.
fn scatter_points(
    mask: &Array2<u8>,
    rng: &mut Rng,
    mpp: f64,
    min_dist_um: f64,
    target: usize,
) -> Vec<(f64, f64)> {
    let (h, w) = mask.dim();
    let mut pts: Vec<(f64, f64)> = Vec::new();
    if h == 0 || w == 0 || target == 0 {
        return pts;
    }
    let cell = min_dist_um.max(1e-6);
    let gw = ((w as f64 * mpp) / cell).ceil() as usize + 1;
    let gh = ((h as f64 * mpp) / cell).ceil() as usize + 1;
    let mut grid: Vec<Vec<usize>> = vec![Vec::new(); gw * gh];
    let md2 = min_dist_um * min_dist_um;
    let max_tries = target.saturating_mul(60).max(1000);
    for _ in 0..max_tries {
        if pts.len() >= target {
            break;
        }
        let px = rng.range(0.0, w as f64);
        let py = rng.range(0.0, h as f64);
        let (ix, iy) = (px.floor() as usize, py.floor() as usize);
        if ix >= w || iy >= h || mask[[iy, ix]] == 0 {
            continue;
        }
        let (ux, uy) = (px * mpp, py * mpp);
        let (gx, gy) = ((ux / cell) as usize, (uy / cell) as usize);
        let mut ok = true;
        'n: for ny in gy.saturating_sub(1)..=(gy + 1).min(gh - 1) {
            for nx in gx.saturating_sub(1)..=(gx + 1).min(gw - 1) {
                for &i in &grid[ny * gw + nx] {
                    let (qx, qy) = pts[i];
                    if (qx - ux).powi(2) + (qy - uy).powi(2) < md2 {
                        ok = false;
                        break 'n;
                    }
                }
            }
        }
        if ok {
            grid[gy * gw + gx].push(pts.len());
            pts.push((ux, uy));
        }
    }
    pts
}

/// Lukis cakram konsentrasi pewarna ke buffer `conc` (3 kanal per piksel).
fn paint_disc(
    conc: &mut [[f32; 3]],
    w: usize,
    h: usize,
    cx_px: f64,
    cy_px: f64,
    r_px: f64,
    c: [f32; 3],
) {
    let x0 = ((cx_px - r_px).floor().max(0.0)) as usize;
    let y0 = ((cy_px - r_px).floor().max(0.0)) as usize;
    let x1 = ((cx_px + r_px).ceil().max(0.0) as usize).min(w);
    let y1 = ((cy_px + r_px).ceil().max(0.0) as usize).min(h);
    for y in y0..y1 {
        for x in x0..x1 {
            let dx = x as f64 + 0.5 - cx_px;
            let dy = y as f64 + 0.5 - cy_px;
            if dx * dx + dy * dy <= r_px * r_px {
                conc[y * w + x] = c;
            }
        }
    }
}

/// Konsentrasi → RGBA dengan matriks pewarna dan derau ±`noise` level.
fn conc_to_rgba(conc: &[[f32; 3]], m: &StainMatrix, rng: &mut Rng, noise: i32) -> Vec<u8> {
    let mut out = vec![255u8; conc.len() * 4];
    for (i, c) in conc.iter().enumerate() {
        let od = m.od_from_concentrations(*c);
        for k in 0..3 {
            let base = intensity_of_od(od[k]) as i32;
            let n = if noise > 0 {
                rng.below((2 * noise + 1) as usize) as i32 - noise
            } else {
                0
            };
            out[i * 4 + k] = (base + n).clamp(0, 255) as u8;
        }
        out[i * 4 + 3] = 255;
    }
    out
}

/// Render "histologi" H&E sintetis: latar putih, jaringan merah muda (eosin)
/// dengan bintik inti ungu gelap (hematoksilin) yang posisinya ditentukan
/// `seed`, plus derau kecil. Mengembalikan RGBA `w × h × 4`.
pub fn render_histology(mask: &Array2<u8>, seed: u64, mpp: f64) -> Vec<u8> {
    let (h, w) = mask.dim();
    let mut rng = Rng::new(seed);
    let m = StainMatrix::he();
    let mut conc = vec![[0.0f32; 3]; w * h];
    for (c, &mv) in conc.iter_mut().zip(mask.iter()) {
        if mv != 0 {
            *c = [
                0.08 + rng.range(-0.02, 0.02) as f32,
                0.28 + rng.range(-0.05, 0.05) as f32,
                0.0,
            ];
        }
    }
    let area_um2 = mask.iter().filter(|v| **v != 0).count() as f64 * mpp * mpp;
    let target = (area_um2 / 120.0) as usize; // ≈ 1 inti per 120 µm²
    let pts = scatter_points(mask, &mut rng, mpp, 6.0, target);
    for (ux, uy) in pts {
        let r_um = rng.range(2.2, 3.2);
        let c = [0.85 + rng.range(-0.1, 0.1) as f32, 0.08, 0.0];
        paint_disc(&mut conc, w, h, ux / mpp, uy / mpp, r_um / mpp, c);
    }
    conc_to_rgba(&conc, &m, &mut rng, 2)
}

/// Inti kebenaran dasar fantom IHC.
pub type TruthNucleus = (f64, f64, bool);

/// Citra IHC sintetis (hematoksilin + DAB): inti negatif biru-ungu, inti
/// positif cokelat (DAB + counterstain tipis), latar jaringan sangat pucat.
/// Tepat `round(fraksi × n)` inti dibuat positif (pemilihan berbenih).
/// Mengembalikan RGBA dan daftar `(x_um, y_um, positif)`.
pub fn synthetic_ihc(
    mask: &Array2<u8>,
    seed: u64,
    positive_fraction: f64,
    mpp: f64,
) -> (Vec<u8>, Vec<TruthNucleus>) {
    let (h, w) = mask.dim();
    let mut rng = Rng::new(seed ^ 0xA5A5_5A5A_1234_5678);
    let m = StainMatrix::h_dab();
    let mut conc = vec![[0.0f32; 3]; w * h];
    for (c, &mv) in conc.iter_mut().zip(mask.iter()) {
        if mv != 0 {
            *c = [0.06 + rng.range(-0.015, 0.015) as f32, 0.0, 0.0];
        }
    }
    let area_um2 = mask.iter().filter(|v| **v != 0).count() as f64 * mpp * mpp;
    let target = (area_um2 / 220.0) as usize;
    let pts = scatter_points(mask, &mut rng, mpp, 9.0, target);
    let n = pts.len();
    let n_pos = (positive_fraction.clamp(0.0, 1.0) * n as f64).round() as usize;
    // Pilih indeks positif lewat acak Fisher–Yates berbenih.
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.below(i + 1);
        order.swap(i, j);
    }
    let mut positive = vec![false; n];
    for &i in order.iter().take(n_pos) {
        positive[i] = true;
    }
    let mut truth = Vec::with_capacity(n);
    for (i, &(ux, uy)) in pts.iter().enumerate() {
        let r_um = rng.range(2.6, 3.4);
        let c = if positive[i] {
            [
                0.45 + rng.range(-0.05, 0.05) as f32,
                0.75 + rng.range(-0.1, 0.1) as f32,
                0.0,
            ]
        } else {
            [0.95 + rng.range(-0.1, 0.1) as f32, 0.0, 0.0]
        };
        paint_disc(&mut conc, w, h, ux / mpp, uy / mpp, r_um / mpp, c);
        truth.push((ux, uy, positive[i]));
    }
    (conc_to_rgba(&conc, &m, &mut rng, 2), truth)
}

/// Pencocokan titik deteksi vs kebenaran dalam toleransi `tol_um`
/// (serakah menurut jarak; tiap titik kebenaran dipakai sekali).
/// Mengembalikan `(tp, fp, fn)`.
pub fn match_points(
    detected: &[(f64, f64)],
    truth: &[(f64, f64)],
    tol_um: f64,
) -> (usize, usize, usize) {
    let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
    let t2 = tol_um * tol_um;
    for (i, d) in detected.iter().enumerate() {
        for (j, t) in truth.iter().enumerate() {
            let d2 = (d.0 - t.0).powi(2) + (d.1 - t.1).powi(2);
            if d2 <= t2 {
                pairs.push((d2, i, j));
            }
        }
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let mut used_d = vec![false; detected.len()];
    let mut used_t = vec![false; truth.len()];
    let mut tp = 0usize;
    for (_, i, j) in pairs {
        if !used_d[i] && !used_t[j] {
            used_d[i] = true;
            used_t[j] = true;
            tp += 1;
        }
    }
    (tp, detected.len() - tp, truth.len() - tp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_deterministic() {
        let mut a = Rng::new(5);
        let mut b = Rng::new(5);
        for _ in 0..10 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        let v = a.next_f64();
        assert!((0.0..1.0).contains(&v));
    }

    #[test]
    fn ellipsoid_slices_match_analytic_area() {
        let r = [300.0, 200.0, 100.0];
        let s = 4.0;
        let m0 = slice_ellipsoid(r, 0.0, s, [200, 200]);
        let area = m0.iter().filter(|v| **v != 0).count() as f64 * s * s;
        let want = std::f64::consts::PI * r[0] * r[1];
        assert!((area - want).abs() / want < 0.02);
        assert!(slice_ellipsoid(r, 100.0, s, [200, 200])
            .iter()
            .all(|v| *v == 0));
        // Cavalieri atas irisan rapat ≈ volume analitik.
        let dz = 2.0;
        let mut vol = 0.0;
        let mut z = -100.0 + dz / 2.0;
        while z < 100.0 {
            vol += slice_ellipsoid(r, z, s, [200, 200])
                .iter()
                .filter(|v| **v != 0)
                .count() as f64
                * s
                * s
                * dz;
            z += dz;
        }
        assert!((vol - ellipsoid_volume(r)).abs() / ellipsoid_volume(r) < 0.02);
    }

    #[test]
    fn torus_and_branch_slices() {
        let t = slice_torus(200.0, 50.0, 0.0, 4.0, [160, 160]);
        let (h, w) = t.dim();
        assert_eq!(t[[h / 2, w / 2]], 0, "lubang torus");
        assert_eq!(t[[h / 2, (w as f64 / 2.0 + 200.0 / 4.0) as usize]], 1);
        let caps = branch_capsules(200.0, 200.0, 40.0, 0.6);
        let below = slice_capsules(&caps, -100.0, 4.0, [160, 160]);
        let above = slice_capsules(&caps, 150.0, 4.0, [160, 160]);
        let (_, nb) = crate::components::label_components(&below);
        let (_, na) = crate::components::label_components(&above);
        assert_eq!(nb, 1);
        assert_eq!(na, 2);
        assert_eq!(
            slice_branch(200.0, 200.0, 40.0, 0.6, 150.0, 4.0, [160, 160]),
            above
        );
    }

    #[test]
    fn render_is_deterministic_and_white_outside() {
        let m = ellipsoid_mask([64, 48], [32.0, 24.0], [20.0, 12.0]);
        let a = render_histology(&m, 11, 0.5);
        let b = render_histology(&m, 11, 0.5);
        assert_eq!(a, b);
        assert_ne!(a, render_histology(&m, 12, 0.5));
        assert_eq!(&a[0..4], &[255, 255, 255, 255]);
        let m2 = ellipsoid_mask([128, 96], [64.0, 48.0], [50.0, 35.0]);
        let (ihc, truth) = synthetic_ihc(&m2, 3, 0.3, 1.0);
        assert_eq!(ihc.len(), 128 * 96 * 4);
        assert!(!truth.is_empty());
        let npos = truth.iter().filter(|t| t.2).count();
        assert_eq!(npos, (0.3 * truth.len() as f64).round() as usize);
    }

    #[test]
    fn match_points_counts() {
        let truth = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)];
        let det = [(0.5, 0.0), (10.2, 0.1), (50.0, 0.0)];
        assert_eq!(match_points(&det, &truth, 2.0), (2, 1, 1));
    }
}
