//! Alat bantu koordinat piksel ↔ fisik (histo3d.md 5.2).
//!
//! Ruang: S0 (piksel level 0), Sl (piksel level l, `p_l = p_0 / d_l`),
//! U (µm, `u = p_0 × mpp`). Koordinat kontinu memakai tepi piksel sebagai
//! bilangan bulat, sehingga **pusat piksel indeks `i` berada di `i + 0,5`**
//! (aturan keras 5).

use crate::source::SlideInfo;

/// Pusat piksel berindeks `i` dalam koordinat kontinu.
pub fn pixel_center(i: i64) -> f64 {
    i as f64 + 0.5
}

/// Indeks piksel yang memuat koordinat kontinu `c` (`floor`).
pub fn pixel_index(c: f64) -> i64 {
    c.floor() as i64
}

/// Pusat piksel level 0 `[i, j]` ke ruang U (µm): `(i + 0,5) × mpp`.
pub fn pixel_center_um(px: [i64; 2], mpp: [f64; 2]) -> [f64; 2] {
    [pixel_center(px[0]) * mpp[0], pixel_center(px[1]) * mpp[1]]
}

/// Koordinat kontinu level 0 ke ruang U (µm).
pub fn px0_to_um(p0: [f64; 2], mpp: [f64; 2]) -> [f64; 2] {
    [p0[0] * mpp[0], p0[1] * mpp[1]]
}

/// Ruang U (µm) ke koordinat kontinu level 0.
pub fn um_to_px0(u: [f64; 2], mpp: [f64; 2]) -> [f64; 2] {
    [u[0] / mpp[0], u[1] / mpp[1]]
}

/// Ruang U (µm) ke indeks piksel level 0 yang memuat titik itu.
pub fn um_to_pixel_index(u: [f64; 2], mpp: [f64; 2]) -> [i64; 2] {
    let p = um_to_px0(u, mpp);
    [pixel_index(p[0]), pixel_index(p[1])]
}

/// Koordinat kontinu level 0 ke level dengan faktor `downsample`.
pub fn px0_to_level(p0: [f64; 2], downsample: f64) -> [f64; 2] {
    [p0[0] / downsample, p0[1] / downsample]
}

/// Koordinat kontinu level `l` ke level 0.
pub fn level_to_px0(pl: [f64; 2], downsample: f64) -> [f64; 2] {
    [pl[0] * downsample, pl[1] * downsample]
}

/// Faktor penurunan yang diinginkan agar satu piksel layar mewakili
/// `um_per_screen_px` µm pada slide ber-`mpp` (dipakai rata-rata mpp x/y).
pub fn target_downsample(um_per_screen_px: f64, mpp: [f64; 2]) -> f64 {
    let mean_mpp = (mpp[0] + mpp[1]) / 2.0;
    um_per_screen_px / mean_mpp
}

/// Level untuk skala tampilan: level **pertama** (indeks terkecil) yang
/// `downsample`-nya ≥ `um_per_screen_px / mpp`; bila tidak ada, level
/// terakhir (paling kasar). Target yang tidak sah (≤ 0, NaN) → level 0.
pub fn level_for_scale(info: &SlideInfo, um_per_screen_px: f64, mpp: [f64; 2]) -> u32 {
    let target = target_downsample(um_per_screen_px, mpp);
    if info.levels.is_empty() || !target.is_finite() || target <= 0.0 {
        return 0;
    }
    // Toleransi kecil supaya target yang tepat sama dengan downsample
    // (mis. 2,0 vs 1,9999999) tidak melompat ke level berikutnya.
    const EPS: f64 = 1e-9;
    for (i, l) in info.levels.iter().enumerate() {
        if l.downsample + EPS >= target {
            return i as u32;
        }
    }
    (info.levels.len() - 1) as u32
}

/// Asal tile `(tx, ty)` dalam piksel level.
pub fn tile_origin(tile_size: u32, tx: u32, ty: u32) -> (i64, i64) {
    (
        i64::from(tx) * i64::from(tile_size),
        i64::from(ty) * i64::from(tile_size),
    )
}

/// Rentang tile inklusif `[tx0..=tx1] × [ty0..=ty1]` yang beririsan dengan
/// persegi level `[x0, x1) × [y0, y1)`, dipotong ke kisi level. `None` bila
/// tidak ada irisan.
pub fn tiles_in_rect(
    level_size: [u32; 2],
    tile_size: u32,
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
) -> Option<[u32; 4]> {
    if tile_size == 0 || !(x0 < x1 && y0 < y1) {
        return None;
    }
    let ts = f64::from(tile_size);
    let cols = level_size[0].div_ceil(tile_size);
    let rows = level_size[1].div_ceil(tile_size);
    if cols == 0 || rows == 0 {
        return None;
    }
    let clampf = |v: f64, max: u32| -> u32 { v.max(0.0).min(f64::from(max)) as u32 };
    let tx0 = clampf((x0 / ts).floor(), cols - 1);
    let ty0 = clampf((y0 / ts).floor(), rows - 1);
    // Tepi kanan/bawah eksklusif: ambil tile yang memuat `x1 - ε`.
    let tx1 = clampf(((x1 / ts).ceil() - 1.0).max(0.0), cols - 1);
    let ty1 = clampf(((y1 / ts).ceil() - 1.0).max(0.0), rows - 1);
    if x1 <= 0.0 || y1 <= 0.0 || x0 >= f64::from(level_size[0]) || y0 >= f64::from(level_size[1]) {
        return None;
    }
    Some([tx0, ty0, tx1, ty1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::LevelInfo;

    fn info() -> SlideInfo {
        SlideInfo {
            levels: [1.0, 4.0, 16.0]
                .iter()
                .map(|&d| LevelInfo {
                    size: [(1600.0 / d) as u32, (1600.0 / d) as u32],
                    downsample: d,
                })
                .collect(),
            mpp: Some([0.25, 0.25]),
            tile_size: 256,
            vendor: None,
        }
    }

    #[test]
    fn pixel_center_convention() {
        assert_eq!(pixel_center(3), 3.5);
        assert_eq!(pixel_index(3.5), 3);
        assert_eq!(pixel_index(3.999), 3);
        assert_eq!(pixel_index(-0.5), -1);
        assert_eq!(pixel_center_um([0, 1], [0.5, 0.25]), [0.25, 0.375]);
        assert_eq!(um_to_pixel_index([0.25, 0.375], [0.5, 0.25]), [0, 1]);
    }

    #[test]
    fn level_selection() {
        let i = info();
        let mpp = [0.25, 0.25];
        assert_eq!(level_for_scale(&i, 0.1, mpp), 0); // target 0,4 → level 0
        assert_eq!(level_for_scale(&i, 0.25, mpp), 0); // target 1
        assert_eq!(level_for_scale(&i, 0.5, mpp), 1); // target 2 → downsample 4
        assert_eq!(level_for_scale(&i, 1.0, mpp), 1); // target 4 tepat
        assert_eq!(level_for_scale(&i, 1.5, mpp), 2); // target 6 → 16
        assert_eq!(level_for_scale(&i, 100.0, mpp), 2); // melebihi semua → terakhir
        assert_eq!(level_for_scale(&i, f64::NAN, mpp), 0);
    }

    #[test]
    fn tile_ranges() {
        assert_eq!(
            tiles_in_rect([1000, 600], 256, 0.0, 0.0, 1000.0, 600.0),
            Some([0, 0, 3, 2])
        );
        assert_eq!(
            tiles_in_rect([1000, 600], 256, 300.0, 100.0, 512.0, 256.0),
            Some([1, 0, 1, 0])
        );
        assert_eq!(
            tiles_in_rect([1000, 600], 256, -500.0, -500.0, 5000.0, 5000.0),
            Some([0, 0, 3, 2])
        );
        assert_eq!(
            tiles_in_rect([1000, 600], 256, 2000.0, 0.0, 3000.0, 10.0),
            None
        );
        assert_eq!(tiles_in_rect([1000, 600], 0, 0.0, 0.0, 1.0, 1.0), None);
    }
}
