//! Masker jaringan dari citra RGBA resolusi rendah (histo3d.md 8.3).

use ndarray::Array2;

use crate::color::od_total;
use crate::components::{component_areas, label_components};
use crate::edt::edt_squared_2d;
use crate::error::ImgError;
use crate::morphology::{close, fill_narrow_mouthed_holes};
use crate::threshold::otsu;

/// Radius penutupan morfologi bawaan (µm).
pub const CLOSE_RADIUS_UM: f64 = 50.0;
/// Radius cakram (µm) penentu latar "luar": latar hanya dibuang bila
/// terjangkau dari tepi citra lewat jalur selebar ≥ 2 × radius. Rongga di
/// dalam jaringan dengan mulut lebih sempit dianggap bagian jaringan.
pub const MOUTH_RADIUS_UM: f64 = 150.0;
/// Fraksi luas minimum komponen relatif terhadap komponen terbesar.
pub const MIN_COMPONENT_FRACTION: f64 = 0.005;
/// Lebar pita pelunakan tepi bawaan untuk [`tissue_alpha`] (piksel).
pub const FEATHER_PX: f32 = 2.0;

/// Masker jaringan: `OD_total` → Otsu → penutupan radius ~50 µm → isi
/// lubang, termasuk rongga bermulut sempit ([`MOUTH_RADIUS_UM`]) → buang komponen dengan luas < 0,5 % komponen terbesar.
/// `mpp_um` = ukuran piksel citra ini (µm/px, harus positif).
pub fn tissue_mask(rgba: &[u8], w: usize, h: usize, mpp_um: f64) -> Result<Array2<u8>, ImgError> {
    if !(mpp_um.is_finite() && mpp_um > 0.0) {
        return Err(ImgError::InvalidParameter(format!(
            "mpp tidak sah: {mpp_um}"
        )));
    }
    let od = od_total(rgba, w, h)?;
    let t = otsu(od.iter().copied(), 256);
    let raw = od.mapv(|v| u8::from(v > t));
    let radius_px = CLOSE_RADIUS_UM / mpp_um;
    let closed = close(&raw, radius_px);
    // Hanya latar putih di luar jaringan yang dibuang: rongga di dalam
    // jaringan yang terbuka ke luar lewat celah sempit tetap dipertahankan.
    let filled = fill_narrow_mouthed_holes(&closed, MOUTH_RADIUS_UM / mpp_um);
    let (labels, n) = label_components(&filled);
    if n == 0 {
        return Ok(filled);
    }
    let areas = component_areas(&labels, n);
    let largest = areas[1..].iter().copied().max().unwrap_or(0) as f64;
    let min_area = (largest * MIN_COMPONENT_FRACTION).ceil() as usize;
    let mut out = Array2::<u8>::zeros((h, w));
    for (o, &l) in out.iter_mut().zip(labels.iter()) {
        if l != 0 && areas[l as usize] >= min_area {
            *o = 1;
        }
    }
    Ok(out)
}

/// Kanal alfa 0..255 dari masker jaringan: 0 di latar (kaca/putih), 255 di
/// dalam jaringan, dan naik linear sepanjang pita `feather_px` dari tepi
/// masker ke dalam agar tepi tidak bergerigi. `feather_px ≤ 1` = tepi keras.
/// Dipakai untuk menampilkan irisan di 3D tanpa lembar putih; citra sumber
/// tidak diubah. Panjang hasil = `w × h`.
pub fn tissue_alpha(
    rgba: &[u8],
    w: usize,
    h: usize,
    mpp_um: f64,
    feather_px: f32,
) -> Result<Vec<u8>, ImgError> {
    let mask = tissue_mask(rgba, w, h, mpp_um)?;
    Ok(mask_to_alpha(&mask, feather_px))
}

/// Alfa 0..255 dari masker biner dengan pelunakan tepi `feather_px`
/// (lihat [`tissue_alpha`]). Masker tanpa latar sama sekali → semua 255.
pub fn mask_to_alpha(mask: &Array2<u8>, feather_px: f32) -> Vec<u8> {
    if !(feather_px.is_finite() && feather_px > 1.0) {
        return mask.iter().map(|&m| if m != 0 { 255 } else { 0 }).collect();
    }
    // Jarak tiap piksel jaringan ke latar terdekat = EDT masker terbalik.
    let inverted = mask.mapv(|m| u8::from(m == 0));
    let dist2 = edt_squared_2d(&inverted);
    mask.iter()
        .zip(dist2.iter())
        .map(|(&m, &d2)| {
            if m == 0 {
                0
            } else {
                let a = (d2.sqrt() / feather_px).clamp(0.0, 1.0);
                (a * 255.0).round() as u8
            }
        })
        .collect()
}

/// Tulis [`tissue_alpha`] ke kanal A citra RGBA8 di tempat; kanal RGB tetap.
pub fn apply_tissue_alpha(
    rgba: &mut [u8],
    w: usize,
    h: usize,
    mpp_um: f64,
    feather_px: f32,
) -> Result<(), ImgError> {
    let alpha = tissue_alpha(rgba, w, h, mpp_um, feather_px)?;
    for (px, a) in rgba.chunks_exact_mut(4).zip(alpha) {
        px[3] = a;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_covers_tissue_not_background() {
        let (w, h) = (64usize, 48usize);
        let mut rgba = vec![255u8; w * h * 4];
        // Blok "jaringan" merah muda 20..44 × 10..38 dengan sebuah lubang kecil.
        for y in 10..38 {
            for x in 20..44 {
                if (x, y) == (30, 20) {
                    continue;
                }
                let i = (y * w + x) * 4;
                rgba[i] = 220;
                rgba[i + 1] = 150;
                rgba[i + 2] = 190;
            }
        }
        // Setitik debu kecil jauh dari jaringan.
        let i = (2 * w + 60) * 4;
        rgba[i] = 100;
        rgba[i + 1] = 100;
        rgba[i + 2] = 100;
        let m = tissue_mask(&rgba, w, h, 16.0).unwrap();
        assert_eq!(m[[20, 30]], 1, "lubang terisi");
        assert_eq!(m[[24, 32]], 1);
        assert_eq!(m[[2, 2]], 0);
        assert_eq!(m[[2, 60]], 0, "debu dibuang");
    }

    #[test]
    fn bad_mpp_rejected() {
        assert!(matches!(
            tissue_mask(&[255u8; 16], 2, 2, 0.0),
            Err(ImgError::InvalidParameter(_))
        ));
    }

    #[test]
    fn alpha_is_zero_on_background_and_feathered_inside() {
        let (w, h) = (40usize, 30usize);
        let mut rgba = vec![255u8; w * h * 4];
        for y in 5..25 {
            for x in 5..35 {
                let i = (y * w + x) * 4;
                rgba[i] = 220;
                rgba[i + 1] = 150;
                rgba[i + 2] = 190;
            }
        }
        let a = tissue_alpha(&rgba, w, h, 16.0, 2.0).unwrap();
        assert_eq!(a.len(), w * h);
        assert_eq!(a[2 * w + 2], 0, "latar transparan");
        assert_eq!(a[15 * w + 20], 255, "tengah jaringan opak");
        let edge = a[5 * w + 20];
        let inner = a[6 * w + 20];
        assert!(edge > 0 && edge < 255, "tepi dilunakkan: {edge}");
        assert!(inner > edge, "alfa naik ke dalam: {inner} > {edge}");

        // Tepi keras.
        let hard = tissue_alpha(&rgba, w, h, 16.0, 0.0).unwrap();
        assert_eq!(hard[5 * w + 20], 255);
        assert_eq!(hard[2 * w + 2], 0);

        // Tulis di tempat: RGB tetap, A berubah.
        let before = rgba.clone();
        apply_tissue_alpha(&mut rgba, w, h, 16.0, 2.0).unwrap();
        for (b, n) in before.chunks_exact(4).zip(rgba.chunks_exact(4)) {
            assert_eq!(&b[..3], &n[..3]);
        }
        assert_eq!(rgba[(2 * w + 2) * 4 + 3], 0);
    }

    #[test]
    fn mask_without_background_is_fully_opaque() {
        let m = Array2::<u8>::ones((4, 4));
        assert!(mask_to_alpha(&m, 2.0).iter().all(|&a| a == 255));
    }
}
