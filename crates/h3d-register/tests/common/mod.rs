//! Pembantu uji: fantom bertekstur dengan kebenaran dasar.
#![allow(dead_code)]

use h3d_core::Affine2;
use h3d_imgproc::color::StainMatrix;
use h3d_imgproc::phantom::{render_histology, Rng};
use h3d_imgproc::{hematoxylin_channel, warp_affine_f32, warp_affine_u8, Interp};
use h3d_register::RegImage;
use ndarray::Array2;

/// Medan tekstur 3D halus (jumlah blob Gauss berbenih) agar irisan
/// bertetangga berkorelasi. Nilai kira-kira di `[-1, 1]`.
pub struct Field3 {
    blobs: Vec<([f64; 3], [f64; 2], f64)>, // pusat, (σ_xy, σ_z), amplitudo
    waves: Vec<([f64; 3], f64, f64)>,      // vektor gelombang (rad/µm), fase, amplitudo
}

impl Field3 {
    /// `extent_um` = setengah lebar kotak pusat blob; `z_extent_um` setengah tinggi.
    pub fn new(seed: u64, extent_um: f64, z_extent_um: f64, n_blobs: usize) -> Self {
        let mut rng = Rng::new(seed);
        let blobs = (0..n_blobs)
            .map(|_| {
                let c = [
                    rng.range(-extent_um, extent_um),
                    rng.range(-extent_um, extent_um),
                    rng.range(-z_extent_um, z_extent_um),
                ];
                let s = [rng.range(150.0, 400.0), rng.range(80.0, 200.0)];
                let a = if rng.next_f64() < 0.5 { -1.0 } else { 1.0 } * rng.range(0.6, 1.0);
                (c, s, a)
            })
            .collect();
        Self {
            blobs,
            waves: Vec::new(),
        }
    }

    /// Medan dari `n` gelombang sinus 3D acak: panjang gelombang bidang
    /// `xy_um` (rentang), sepanjang z `z_um` (rentang; panjang → irisan
    /// bertetangga berkorelasi kuat dan meluruh perlahan).
    pub fn waves(seed: u64, n: usize, xy_um: (f64, f64), z_um: (f64, f64)) -> Self {
        let mut rng = Rng::new(seed);
        let tau = std::f64::consts::TAU;
        let waves = (0..n)
            .map(|_| {
                let dir = rng.range(0.0, tau);
                let kxy = tau / rng.range(xy_um.0, xy_um.1);
                let sign = if rng.next_f64() < 0.5 { -1.0 } else { 1.0 };
                let kz = sign * tau / rng.range(z_um.0, z_um.1);
                (
                    [kxy * dir.cos(), kxy * dir.sin(), kz],
                    rng.range(0.0, tau),
                    1.0 / (n as f64).sqrt() * 1.4,
                )
            })
            .collect();
        Self {
            blobs: Vec::new(),
            waves,
        }
    }

    pub fn at(&self, x: f64, y: f64, z: f64) -> f64 {
        self.blobs
            .iter()
            .map(|(c, s, a)| {
                let dxy = ((x - c[0]).powi(2) + (y - c[1]).powi(2)) / (2.0 * s[0] * s[0]);
                let dz = (z - c[2]).powi(2) / (2.0 * s[1] * s[1]);
                a * (-(dxy + dz)).exp()
            })
            .sum::<f64>()
            + self
                .waves
                .iter()
                .map(|(k, ph, a)| a * (k[0] * x + k[1] * y + k[2] * z + ph).cos())
                .sum::<f64>()
    }
}

/// Citra registrasi dari masker: hematoksilin `render_histology` (bintik inti)
/// dicampur medan tekstur bersama pada `z_um`. Asal µm di sudut (0,0) = `origin`;
/// medan dievaluasi pada koordinat berpusat citra.
pub fn textured_image(
    mask: &Array2<u8>,
    mpp: f64,
    z_um: f64,
    field: &Field3,
    seed: u64,
    origin: [f64; 2],
    field_weight: f32,
) -> RegImage {
    let (h, w) = mask.dim();
    let rgba = render_histology(mask, seed, mpp);
    let hch = hematoxylin_channel(&rgba, w, h, &StainMatrix::he()).expect("hematoksilin");
    let cx = w as f64 * mpp / 2.0;
    let cy = h as f64 * mpp / 2.0;
    let gray = Array2::from_shape_fn((h, w), |(y, x)| {
        if mask[[y, x]] == 0 {
            return 0.0f32;
        }
        let ux = (x as f64 + 0.5) * mpp - cx;
        let uy = (y as f64 + 0.5) * mpp - cy;
        let f = (0.5 + 0.5 * field.at(ux, uy, z_um).clamp(-1.0, 1.0)) as f32;
        (1.0 - field_weight) * (0.2 + 0.8 * hch[[y, x]]) + field_weight * f
    });
    RegImage::new(gray, mask.clone(), mpp, origin).expect("RegImage")
}

/// Bentuk citra baru `dst` (kanvas `size`, `mpp`, `origin`) dari `src` dengan
/// `p_src_to_dst` (U_src → U_dst, µm): `dst(u) = src(P⁻¹ u)`.
pub fn warp_image(
    src: &RegImage,
    p_src_to_dst: &Affine2,
    size: [usize; 2],
    mpp: f64,
    origin: [f64; 2],
) -> RegImage {
    let dst_px_to_um =
        Affine2::translation(origin[0], origin[1]).compose(&Affine2::scale(mpp, mpp));
    let p_inv = p_src_to_dst.inverse().expect("P invertibel");
    // t_inv: piksel dst → piksel src.
    let t_inv = src.um_to_px_affine().compose(&p_inv).compose(&dst_px_to_um);
    let gray = warp_affine_f32(&src.gray, &t_inv, size, Interp::Bilinear, 0.0).expect("warp");
    let mask = warp_affine_u8(&src.mask, &t_inv, size, Interp::Nearest, 0).expect("warp");
    RegImage::new(gray, mask, mpp, origin).expect("RegImage")
}

/// Transformasi acak berbenih `G` (U_src → U_dst) di sekitar `center` (µm):
/// rotasi 0–360°, geser ±`shift_frac × width_um`, cermin acak bila `allow_mirror`,
/// skala `[s_lo, s_hi)`. Mengembalikan `(G, mirrored)`.
pub fn random_transform(
    rng: &mut Rng,
    center: [f64; 2],
    width_um: f64,
    shift_frac: f64,
    allow_mirror: bool,
    scale: (f64, f64),
) -> (Affine2, bool) {
    let theta = rng.range(0.0, std::f64::consts::TAU);
    let tx = rng.range(-shift_frac, shift_frac) * width_um;
    let ty = rng.range(-shift_frac, shift_frac) * width_um;
    let mirror = allow_mirror && rng.next_f64() < 0.5;
    let s = rng.range(scale.0, scale.1);
    let mut g = Affine2::translation(center[0] + tx, center[1] + ty)
        .compose(&Affine2::rotation(theta))
        .compose(&Affine2::scale(s, s));
    if mirror {
        g = g.compose(&Affine2::mirror_x());
    }
    (
        g.compose(&Affine2::translation(-center[0], -center[1])),
        mirror,
    )
}

/// TRE RMS (µm) `|P_est(u) − P_true(u)|` atas pusat piksel jaringan `img`
/// (subsampel tiap `step` piksel).
pub fn tre_um(img: &RegImage, p_est: &Affine2, p_true: &Affine2, step: usize) -> f64 {
    let (h, w) = img.mask.dim();
    let mut sum = 0.0;
    let mut n = 0.0;
    for y in (0..h).step_by(step) {
        for x in (0..w).step_by(step) {
            if img.mask[[y, x]] == 0 {
                continue;
            }
            let u = img.px_to_um([x as f64 + 0.5, y as f64 + 0.5]);
            let a = p_est.apply(u);
            let b = p_true.apply(u);
            sum += (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2);
            n += 1.0;
        }
    }
    if n == 0.0 {
        f64::INFINITY
    } else {
        (sum / n).sqrt()
    }
}
