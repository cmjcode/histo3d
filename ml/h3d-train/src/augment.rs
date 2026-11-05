//! Augmentasi tile berbenih: potong acak, balik/putar 90°, jitter pewarna di
//! ruang HED (dekonvolusi `h3d_imgproc::color`), dan blur Gauss ringan.
//! Semua fungsi murni atas buffer; RNG diteruskan dari pemanggil (aturan 9).

use h3d_imgproc::color::intensity_of_od;
use h3d_imgproc::{gaussian_blur_f32, rgb_to_od, StainMatrix};
use ndarray::Array2;
use rand::Rng;

use crate::data::Tile;

/// Parameter augmentasi.
#[derive(Debug, Clone)]
pub struct AugParams {
    /// Simpangan faktor skala konsentrasi pewarna (0 = mati).
    pub stain_scale: f32,
    /// Simpangan geser konsentrasi pewarna.
    pub stain_shift: f32,
    /// Peluang blur.
    pub blur_prob: f32,
    /// Sigma blur maksimum (piksel).
    pub blur_sigma_max: f64,
}

impl Default for AugParams {
    fn default() -> Self {
        Self {
            stain_scale: 0.08,
            stain_shift: 0.03,
            blur_prob: 0.2,
            blur_sigma_max: 1.0,
        }
    }
}

/// Potong jendela `size×size` acak dari `tile`.
pub fn random_crop<R: Rng>(tile: &Tile, size: usize, rng: &mut R) -> Tile {
    let s = tile.size;
    if size >= s {
        return tile.clone();
    }
    let x0 = rng.random_range(0..=s - size);
    let y0 = rng.random_range(0..=s - size);
    crop(tile, x0, y0, size)
}

/// Potong jendela tetap.
pub fn crop(tile: &Tile, x0: usize, y0: usize, size: usize) -> Tile {
    let s = tile.size;
    let mut rgb = Vec::with_capacity(size * size * 3);
    let mut cls = Vec::with_capacity(size * size);
    for y in y0..y0 + size {
        let row = y * s;
        rgb.extend_from_slice(&tile.rgb[(row + x0) * 3..(row + x0 + size) * 3]);
        cls.extend_from_slice(&tile.cls[row + x0..row + x0 + size]);
    }
    Tile { size, rgb, cls }
}

/// Transformasi dihedral `k` (0..8): bit 0 = balik horizontal, bit 1 = balik
/// vertikal, bit 2 = transposisi. Diterapkan ke RGB dan kelas bersamaan.
pub fn dihedral(tile: &Tile, k: u8) -> Tile {
    let s = tile.size;
    let mut rgb = vec![0u8; s * s * 3];
    let mut cls = vec![0u8; s * s];
    for y in 0..s {
        for x in 0..s {
            let (mut sx, mut sy) = (x, y);
            if k & 4 != 0 {
                std::mem::swap(&mut sx, &mut sy);
            }
            if k & 1 != 0 {
                sx = s - 1 - sx;
            }
            if k & 2 != 0 {
                sy = s - 1 - sy;
            }
            let si = sy * s + sx;
            let di = y * s + x;
            rgb[di * 3..di * 3 + 3].copy_from_slice(&tile.rgb[si * 3..si * 3 + 3]);
            cls[di] = tile.cls[si];
        }
    }
    Tile { size: s, rgb, cls }
}

/// Jitter pewarna: konsentrasi H dan E dikalikan `(1 + a_i)` dan digeser `b_i`
/// (a, b acak seragam dalam ±skala/±geser). Kelas tidak berubah.
pub fn stain_jitter<R: Rng>(tile: &Tile, p: &AugParams, rng: &mut R) -> Tile {
    if p.stain_scale <= 0.0 && p.stain_shift <= 0.0 {
        return tile.clone();
    }
    let m = StainMatrix::he();
    let a = [
        rng.random_range(-p.stain_scale..=p.stain_scale),
        rng.random_range(-p.stain_scale..=p.stain_scale),
    ];
    let b = [
        rng.random_range(-p.stain_shift..=p.stain_shift),
        rng.random_range(-p.stain_shift..=p.stain_shift),
    ];
    let mut rgb = Vec::with_capacity(tile.rgb.len());
    for px in tile.rgb.chunks_exact(3) {
        let mut c = m.concentrations(rgb_to_od(px));
        c[0] = (c[0] * (1.0 + a[0]) + b[0]).max(0.0);
        c[1] = (c[1] * (1.0 + a[1]) + b[1]).max(0.0);
        let od = m.od_from_concentrations(c);
        rgb.extend(od.iter().map(|&o| intensity_of_od(o)));
    }
    Tile {
        size: tile.size,
        rgb,
        cls: tile.cls.clone(),
    }
}

/// Blur Gauss pada tiap kanal; kelas tidak berubah.
pub fn blur(tile: &Tile, sigma_px: f64) -> Tile {
    let s = tile.size;
    let mut out = vec![0u8; tile.rgb.len()];
    for ch in 0..3 {
        let arr = Array2::from_shape_fn((s, s), |(y, x)| f32::from(tile.rgb[(y * s + x) * 3 + ch]));
        let b = gaussian_blur_f32(&arr, sigma_px);
        for (i, v) in b.iter().enumerate() {
            out[i * 3 + ch] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    Tile {
        size: s,
        rgb: out,
        cls: tile.cls.clone(),
    }
}

/// Augmentasi latih lengkap: potong acak → dihedral acak → jitter pewarna →
/// blur dengan peluang.
pub fn augment<R: Rng>(tile: &Tile, crop_size: usize, p: &AugParams, rng: &mut R) -> Tile {
    let t = random_crop(tile, crop_size, rng);
    let t = dihedral(&t, rng.random_range(0..8));
    let t = stain_jitter(&t, p, rng);
    if p.blur_prob > 0.0 && rng.random_range(0.0..1.0) < p.blur_prob {
        blur(&t, rng.random_range(0.3..=p.blur_sigma_max.max(0.3)))
    } else {
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn tile3() -> Tile {
        Tile {
            size: 3,
            rgb: (0..27).collect(),
            cls: (0..9).map(|i| (i % 3) as u8).collect(),
        }
    }

    #[test]
    fn dihedral_identity_and_involution() {
        let t = tile3();
        assert_eq!(dihedral(&t, 0), t);
        for k in [1u8, 2, 3, 4] {
            assert_eq!(dihedral(&dihedral(&t, k), k), t, "k={k}");
        }
    }

    #[test]
    fn stain_jitter_keeps_white_and_classes() {
        let t = Tile {
            size: 2,
            rgb: vec![255; 12],
            cls: vec![0, 1, 2, 1],
        };
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        let j = stain_jitter(&t, &AugParams::default(), &mut rng);
        assert_eq!(j.cls, t.cls);
        // Geser konsentrasi boleh sedikit mewarnai latar putih, tetapi tidak jauh.
        assert!(j.rgb.iter().all(|&v| v >= 230), "{:?}", j.rgb);
    }

    #[test]
    fn augment_is_seeded() {
        let t = Tile {
            size: 8,
            rgb: (0..192).map(|i| (i * 7 % 256) as u8).collect(),
            cls: vec![1; 64],
        };
        let p = AugParams::default();
        let a = augment(&t, 4, &p, &mut rand_chacha::ChaCha8Rng::seed_from_u64(7));
        let b = augment(&t, 4, &p, &mut rand_chacha::ChaCha8Rng::seed_from_u64(7));
        assert_eq!(a, b);
        assert_eq!(a.size, 4);
    }
}
