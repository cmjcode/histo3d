//! Impor masker raster (PNG biner) ke [`LabelStore`] dan pembandingan Dice
//! terhadap masker raster (histo3d.md 8.5, 5.2).
//!
//! Masker berada di ruang U irisan yang sama dengan `LabelStore`, hanya
//! berbeda `mpp`. Konvensi pusat piksel `+0,5` dipakai di kedua arah:
//! piksel `(x, y)` pada mpp `k` berpusat di `((x + 0,5)·k, (y + 0,5)·k)` µm,
//! dan indeks dari µm adalah `floor(u / k)`. Semua resampling **mundur**
//! (tetangga terdekat): untuk tiap piksel tujuan, cari asalnya di sumber.

use std::path::Path;

use h3d_core::LabelId;
use ndarray::Array2;

use crate::error::SegError;
use crate::label_store::{LabelStore, PaintMode};

/// Ringkasan impor masker raster.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RasterImportSummary {
    /// Jumlah piksel `LabelStore` (pada `mpp_label`) yang berubah menjadi
    /// `label`; sudah memperhitungkan `mode` (mis. `OnlyBackground` tidak
    /// menghitung piksel yang sudah berlabel lain).
    pub filled_px: u64,
    /// Luas piksel yang diisi, µm² (`filled_px × mpp_label²`).
    pub area_um2: f64,
    /// Jumlah piksel bukan-nol di masker sumber.
    pub mask_px: u64,
}

/// Periksa `mpp_mask` dan kecocokan ukuran fisik masker dengan store:
/// `ukuran masker × mpp_mask` harus ≈ `ukuran store × mpp_label` dengan
/// toleransi satu piksel masker **ditambah** satu piksel label di tiap sumbu
/// (ukuran store dibulatkan ke atas ke kelipatan `mpp_label`, dan masker
/// yang diperkecil dari resolusi penuh dibulatkan ke bawah).
fn check_geometry(store: &LabelStore, mask: &Array2<u8>, mpp_mask: f64) -> Result<(), SegError> {
    if !(mpp_mask.is_finite() && mpp_mask > 0.0) {
        return Err(SegError::InvalidParameter(format!(
            "mpp masker tidak sah: {mpp_mask}"
        )));
    }
    let (h, w) = mask.dim();
    if h == 0 || w == 0 {
        return Err(SegError::InvalidParameter("masker kosong".to_owned()));
    }
    let store_um = store.size_um();
    let mask_um = [w as f64 * mpp_mask, h as f64 * mpp_mask];
    let tol = mpp_mask + store.mpp_label;
    for axis in 0..2 {
        if (mask_um[axis] - store_um[axis]).abs() > tol {
            return Err(SegError::InvalidParameter(format!(
                "ukuran masker {}×{} px pada mpp {mpp_mask} ({:.1}×{:.1} µm) tidak cocok dengan \
                 ukuran irisan {}×{} px pada mpp {} ({:.1}×{:.1} µm)",
                w,
                h,
                mask_um[0],
                mask_um[1],
                store.size_px[0],
                store.size_px[1],
                store.mpp_label,
                store_um[0],
                store_um[1],
            )));
        }
    }
    Ok(())
}

/// Indeks tetangga terdekat pada kisi mpp `mpp_dst` untuk pusat piksel `i`
/// pada kisi mpp `mpp_src`, dijepit ke `[0, len)`.
#[inline]
fn nearest_index(i: usize, mpp_src: f64, mpp_dst: f64, len: usize) -> usize {
    let u = (i as f64 + 0.5) * mpp_src;
    let j = (u / mpp_dst).floor();
    if j <= 0.0 {
        0
    } else {
        (j as usize).min(len.saturating_sub(1))
    }
}

/// Tulis piksel masker bukan-nol sebagai `label` ke `store` sesuai `mode`.
///
/// `mask` berindeks `[y, x]` pada `mpp_mask` dan menutupi seluruh irisan
/// (ukuran fisik harus ≈ ukuran store, toleransi satu piksel masker + satu
/// piksel label; selain
/// itu `SegError::InvalidParameter`). Pemetaan mundur: untuk tiap piksel
/// store (pusat `+0,5`) dihitung posisi µm → indeks masker terdekat. Diproses
/// tile demi tile dalam urutan `(ty, tx)` sehingga deterministik dan hemat
/// memori. Mengembalikan jumlah piksel yang benar-benar berubah.
pub fn import_mask(
    store: &mut LabelStore,
    mask: &Array2<u8>,
    mpp_mask: f64,
    label: LabelId,
    mode: PaintMode,
) -> Result<RasterImportSummary, SegError> {
    if label.is_background() {
        return Err(SegError::InvalidParameter(
            "label latar (0) tidak bisa diimpor".to_owned(),
        ));
    }
    check_geometry(store, mask, mpp_mask)?;
    let (mh, mw) = mask.dim();
    let mpp = store.mpp_label;
    let ts = store.tile_size as usize;
    let [gw, gh] = store.grid_size();
    let [sw, sh] = [store.size_px[0] as usize, store.size_px[1] as usize];
    let before = store.area_px(label);

    // Peta kolom/baris store → indeks masker, dihitung sekali.
    let col_of: Vec<usize> = (0..sw)
        .map(|x| nearest_index(x, mpp, mpp_mask, mw))
        .collect();
    let row_of: Vec<usize> = (0..sh)
        .map(|y| nearest_index(y, mpp, mpp_mask, mh))
        .collect();

    let mut tile = Array2::<u8>::zeros((ts, ts));
    for ty in 0..gh as usize {
        let y0 = ty * ts;
        let y1 = (y0 + ts).min(sh);
        for tx in 0..gw as usize {
            let x0 = tx * ts;
            let x1 = (x0 + ts).min(sw);
            tile.fill(0);
            let mut any = false;
            for y in y0..y1 {
                let my = row_of[y];
                for x in x0..x1 {
                    if mask[[my, col_of[x]]] != 0 {
                        tile[[y - y0, x - x0]] = 1;
                        any = true;
                    }
                }
            }
            if any {
                store.rasterize_mask(&tile, [x0 as i64, y0 as i64], label, mode);
            }
        }
    }

    let after = store.area_px(label);
    let filled_px = after.saturating_sub(before);
    Ok(RasterImportSummary {
        filled_px,
        area_um2: filled_px as f64 * mpp * mpp,
        mask_px: mask.iter().filter(|&&v| v != 0).count() as u64,
    })
}

/// Baca masker biner dari berkas PNG (8-bit atau 16-bit, grayscale/RGB,
/// dengan atau tanpa alfa) lewat crate `image`. Piksel yang salah satu
/// kanal warnanya bukan-nol menjadi `1`, selainnya `0`. Hasil berindeks
/// `[y, x]`. Berkas rusak atau format tak dikenal → `SegError::InvalidFile`.
pub fn load_mask_png(path: &Path) -> Result<Array2<u8>, SegError> {
    let img = image::open(path).map_err(|e| {
        SegError::invalid_file(format!("masker {} tidak bisa dibaca: {e}", path.display()))
    })?;
    let rgb = img.into_rgb16();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    if w == 0 || h == 0 {
        return Err(SegError::invalid_file(format!(
            "masker {} kosong",
            path.display()
        )));
    }
    let data: Vec<u8> = rgb
        .pixels()
        .map(|p| u8::from(p.0.iter().any(|&c| c != 0)))
        .collect();
    Array2::from_shape_vec((h, w), data).map_err(|e| {
        SegError::invalid_file(format!("masker {}: bentuk tidak sah: {e}", path.display()))
    })
}

/// Resample `label` dari `store` ke kisi masker (tetangga terdekat, pusat
/// `+0,5`) lalu hitung Dice dengan `h3d_imgproc::dice`. Masker dibaca
/// sebagai biner (`!= 0`). Syarat ukuran sama dengan [`import_mask`].
pub fn dice_against_mask(
    store: &LabelStore,
    label: LabelId,
    mask: &Array2<u8>,
    mpp_mask: f64,
) -> Result<f32, SegError> {
    check_geometry(store, mask, mpp_mask)?;
    let (mh, mw) = mask.dim();
    let mpp = store.mpp_label;
    let [sw, sh] = [store.size_px[0] as usize, store.size_px[1] as usize];
    let col_of: Vec<u32> = (0..mw)
        .map(|x| nearest_index(x, mpp_mask, mpp, sw) as u32)
        .collect();
    let row_of: Vec<u32> = (0..mh)
        .map(|y| nearest_index(y, mpp_mask, mpp, sh) as u32)
        .collect();
    let mut resampled = Array2::<u8>::zeros((mh, mw));
    for (y, mut row) in resampled.outer_iter_mut().enumerate() {
        let sy = row_of[y];
        for (x, v) in row.iter_mut().enumerate() {
            *v = u8::from(store.get(col_of[x], sy) == label.0);
        }
    }
    Ok(h3d_imgproc::dice(&resampled, mask)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn store_4x4_mpp4() -> LabelStore {
        // 4×4 px pada mpp 4 = 16×16 µm.
        LabelStore::with_size_px(4.0, [4, 4], 256)
    }

    #[test]
    fn mask_2x2_at_mpp8_fills_4x4_store_at_mpp4() {
        let mut store = store_4x4_mpp4();
        let mask = array![[1u8, 0], [0, 1]];
        let sum =
            import_mask(&mut store, &mask, 8.0, LabelId(1), PaintMode::Replace).expect("impor");
        // Dua piksel masker × (2×2) piksel store masing-masing.
        assert_eq!(sum.filled_px, 8);
        assert_eq!(sum.mask_px, 2);
        assert!((sum.area_um2 - 8.0 * 16.0).abs() < 1e-9);
        assert_eq!(store.area_px(LabelId(1)), 8);
        for y in 0..4u32 {
            for x in 0..4u32 {
                let expect = (x < 2 && y < 2) || (x >= 2 && y >= 2);
                assert_eq!(store.get(x, y) == 1, expect, "({x},{y})");
            }
        }
    }

    #[test]
    fn only_background_does_not_overwrite_other_labels() {
        let mut store = store_4x4_mpp4();
        // Label 2 memenuhi kuadran kiri-atas.
        store.rasterize_mask(
            &Array2::<u8>::ones((2, 2)),
            [0, 0],
            LabelId(2),
            PaintMode::Replace,
        );
        let mask = array![[1u8, 1], [0, 0]];
        let sum = import_mask(
            &mut store,
            &mask,
            8.0,
            LabelId(1),
            PaintMode::OnlyBackground,
        )
        .expect("impor");
        assert_eq!(sum.filled_px, 4);
        assert_eq!(store.area_px(LabelId(2)), 4);
        assert_eq!(store.area_px(LabelId(1)), 4);
        assert_eq!(store.get(0, 0), 2);
        assert_eq!(store.get(3, 0), 1);
    }

    #[test]
    fn dice_of_mask_against_itself_is_one() {
        let mut store = store_4x4_mpp4();
        let mask = array![[1u8, 0], [1, 1]];
        import_mask(&mut store, &mask, 8.0, LabelId(1), PaintMode::Replace).expect("impor");
        let d = dice_against_mask(&store, LabelId(1), &mask, 8.0).expect("dice");
        assert!((d - 1.0).abs() < 1e-6, "dice = {d}");
        // Label lain: tidak ada tumpang tindih.
        let d0 = dice_against_mask(&store, LabelId(2), &mask, 8.0).expect("dice");
        assert_eq!(d0, 0.0);
    }

    #[test]
    fn size_mismatch_is_rejected() {
        let mut store = store_4x4_mpp4();
        // 5×5 px pada mpp 8 = 40 µm; selisih 24 µm dengan store 16 µm
        // melebihi toleransi satu piksel masker + satu piksel label (12 µm).
        let mask = Array2::<u8>::ones((5, 5));
        let err = import_mask(&mut store, &mask, 8.0, LabelId(1), PaintMode::Replace)
            .expect_err("harus ditolak");
        assert!(matches!(err, SegError::InvalidParameter(_)), "{err}");
        assert!(dice_against_mask(&store, LabelId(1), &mask, 0.0).is_err());
        assert!(import_mask(&mut store, &mask, 8.0, LabelId(0), PaintMode::Replace).is_err());
    }

    #[test]
    fn load_mask_png_binarizes_and_rejects_garbage() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("m.png");
        let mut img = image::RgbImage::new(3, 2);
        img.put_pixel(0, 0, image::Rgb([1, 0, 0]));
        img.put_pixel(2, 1, image::Rgb([0, 0, 200]));
        img.save(&path).expect("tulis png");
        let m = load_mask_png(&path).expect("baca");
        assert_eq!(m.dim(), (2, 3));
        assert_eq!(m, array![[1u8, 0, 0], [0, 0, 1]]);

        let bad = dir.path().join("rusak.png");
        std::fs::write(&bad, b"bukan png").expect("tulis");
        assert!(matches!(
            load_mask_png(&bad),
            Err(SegError::InvalidFile { .. })
        ));
        assert!(load_mask_png(&dir.path().join("tidak-ada.png")).is_err());
    }
}
