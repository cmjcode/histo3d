//! Penghitungan "objek utuh" (histo3d.md 8.12.5, ADR 0003 adendum): tiap
//! wilayah hematoksilin yang tersambung menjadi **satu objek**, tanpa
//! dibelah watershed menjadi inti. Dipakai bila yang ingin dihitung adalah
//! struktur (pita/cincin epitel, gumpalan, serpihan), bukan sel satu per
//! satu; pada epitel yang sitoplasmanya ikut gelap di kanal H, watershed
//! memotong pita menjadi keping-keping yang bukan sel (lihat ADR 0003).
//!
//! Alur:
//! 1. Wilayah dibaca per tile bertumpang tindih (paralel); tiap tile
//!    menghasilkan kanal H halus dan selisih H−E, lalu hanya piksel milik
//!    tile (titik tengah tumpang tindih) ditempel ke **satu raster
//!    selebar wilayah**. Tiap tile menulis daerah yang saling lepas
//!    sehingga hasil tidak bergantung urutan thread (aturan 9).
//! 2. Satu ambang untuk seluruh wilayah: `maks(lantai, Otsu)` atas piksel
//!    jaringan, jadi tidak ada sambungan tile pada masker.
//! 3. Buka → tutup morfologi → isi hanya lubang kecil (lumen tetap lubang)
//!    → buang komponen kecil.
//! 4. Objek = komponen tersambung (konektivitas 4); kontur luar dan lubang.
//!
//! Resolusi kerja diturunkan bila raster wilayah melebihi
//! `Budgets::cell_region_bytes` ([`WholeRegionDetector::working_mpp`]);
//! mpp yang dipakai tercatat di [`CellDetection::mpp`].
//!
//! Ketidakpastian (aturan 18): skor objek = OD H halus rata-rata; `n` dan
//! `n_max` = semua objek, `n_min` = objek dengan skor ≥ 1,2 × ambang.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use h3d_core::Progress;
use h3d_imgproc::color::{deconvolve, StainMatrix};
use h3d_imgproc::Connectivity;
use ndarray::Array2;

use crate::ai::TileReader;
use crate::cells::{fnv1a, ownership, starts, CellDetection, DetectedNucleus};
use crate::error::SegError;

/// Perkiraan byte per piksel raster wilayah (H halus f32, H−E f32, masker
/// jaringan u8, masker u8, label u32, cadangan morfologi).
pub const REGION_BYTES_PER_PX: u64 = 24;

/// Nama metode objek utuh di [`CellDetection::method`].
pub const REGIONS_METHOD: &str = "regions-classic-v1";

/// Pendeteksi objek utuh: ambang kanal hematoksilin satu wilayah + komponen
/// tersambung, tanpa pemisahan watershed.
#[derive(Debug, Clone)]
pub struct WholeRegionDetector {
    /// Sisi tile pembacaan (piksel; bawaan 512).
    pub tile_size: usize,
    /// Ukuran piksel kerja, µm (bawaan 0,5; bisa dinaikkan oleh
    /// [`Self::working_mpp`]).
    pub mpp: f64,
    /// Matriks pewarna (bawaan H&E).
    pub stain: StainMatrix,
    /// σ pemulusan Gauss kanal H, µm (bawaan 1).
    pub smooth_sigma_um: f64,
    /// Lantai ambang OD hematoksilin (bawaan 0,15); bila
    /// `adaptive_threshold`, ambang = maks(lantai, Otsu wilayah).
    pub h_threshold: f32,
    /// Naikkan ambang dengan Otsu atas piksel jaringan wilayah (bawaan
    /// benar).
    pub adaptive_threshold: bool,
    /// Radius pembukaan morfologi, µm (bawaan 0,5).
    pub open_radius_um: f64,
    /// Radius penutupan morfologi, µm (bawaan 1,5): menyambung celah pucat
    /// tipis (membran) di dalam satu pita agar tidak terbelah.
    pub close_radius_um: f64,
    /// Lubang tertutup dengan luas ≤ ini diisi, µm² (bawaan 50); lubang
    /// lebih besar (lumen) tetap lubang.
    pub max_hole_area_um2: f64,
    /// Luas objek minimum, µm² (bawaan 8).
    pub min_area_um2: f64,
    /// Wajibkan OD H rata-rata ≥ OD kanal kedua (bawaan benar).
    pub h_over_e: bool,
    /// Toleransi penyederhanaan kontur (Douglas–Peucker), µm (bawaan 0,3);
    /// hanya tampilan.
    pub contour_tolerance_um: f64,
}

impl Default for WholeRegionDetector {
    fn default() -> Self {
        Self {
            tile_size: 512,
            mpp: 0.5,
            stain: StainMatrix::he(),
            smooth_sigma_um: 1.0,
            h_threshold: 0.15,
            adaptive_threshold: true,
            open_radius_um: 0.5,
            close_radius_um: 1.5,
            max_hole_area_um2: 50.0,
            min_area_um2: 8.0,
            h_over_e: true,
            contour_tolerance_um: 0.3,
        }
    }
}

impl WholeRegionDetector {
    /// Validasi nilai.
    pub fn validate(&self) -> Result<(), SegError> {
        let pos = |v: f64| v.is_finite() && v >= 0.0;
        let ok = self.tile_size >= 64
            && self.mpp.is_finite()
            && self.mpp > 0.0
            && self.h_threshold.is_finite()
            && self.h_threshold > 0.0
            && pos(self.smooth_sigma_um)
            && pos(self.open_radius_um)
            && pos(self.close_radius_um)
            && pos(self.max_hole_area_um2)
            && pos(self.min_area_um2)
            && pos(self.contour_tolerance_um);
        if !ok {
            return Err(SegError::InvalidParameter(
                "parameter objek utuh tidak sah".to_string(),
            ));
        }
        Ok(())
    }

    /// Nama metode untuk laporan.
    pub fn method_name(&self) -> String {
        REGIONS_METHOD.to_string()
    }

    /// Hash parameter stabil (termasuk mpp kerja; ukuran tile tidak ikut
    /// karena hasil tidak bergantung padanya).
    pub fn params_hash(&self) -> u64 {
        let f = |v: f64| v.to_bits().to_le_bytes().to_vec();
        let parts: Vec<Vec<u8>> = vec![
            b"h3d-cells-regions-v1".to_vec(),
            f(self.mpp),
            f(self.smooth_sigma_um),
            self.h_threshold.to_bits().to_le_bytes().to_vec(),
            vec![u8::from(self.adaptive_threshold)],
            f(self.open_radius_um),
            f(self.close_radius_um),
            f(self.max_hole_area_um2),
            f(self.min_area_um2),
            vec![u8::from(self.h_over_e)],
            f(self.contour_tolerance_um),
        ];
        let refs: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
        fnv1a(&refs)
    }

    /// Tepi tile (piksel) yang menampung kaki kernel Gauss.
    pub fn margin_px(&self) -> usize {
        let m = (3.0 * self.smooth_sigma_um / self.mpp).ceil() as usize + 2;
        m.clamp(2, self.tile_size / 4)
    }

    /// Ukuran piksel kerja untuk wilayah `roi_um = [x0, y0, x1, y1]` agar
    /// raster wilayah muat `max_bytes`: `self.mpp` bila muat, selain itu
    /// dinaikkan sebanding akar rasio luas (deterministik).
    pub fn working_mpp(&self, roi_um: [f64; 4], max_bytes: u64) -> f64 {
        let w = (roi_um[2] - roi_um[0]).max(0.0) / self.mpp;
        let h = (roi_um[3] - roi_um[1]).max(0.0) / self.mpp;
        let max_px = (max_bytes / REGION_BYTES_PER_PX).max(1) as f64;
        let px = w.ceil() * h.ceil();
        if !px.is_finite() || px <= max_px {
            return self.mpp;
        }
        // Sedikit di atas rasio pas agar pembulatan ke atas tetap muat.
        self.mpp * (px / max_px).sqrt() * 1.01
    }
}

/// Deteksi objek utuh pada wilayah `roi_um = [x0, y0, x1, y1]` (µm, ruang
/// pembaca). `read_tile(x_um, y_um, w, h)` membaca RGBA pada `det.mpp`.
/// `tissue_mask` `(masker, mpp)`: tile tanpa jaringan dilewati dan objek
/// yang kurang dari separuh luasnya di jaringan dibuang. `max_bytes`:
/// batas raster wilayah; melebihinya → error (pemanggil menurunkan
/// resolusi dengan [`WholeRegionDetector::working_mpp`] dulu).
pub fn detect_regions(
    det: &WholeRegionDetector,
    read_tile: &TileReader<'_>,
    roi_um: [f64; 4],
    tissue_mask: Option<(&Array2<u8>, f64)>,
    keep_polygons: bool,
    max_bytes: u64,
    progress: &dyn Progress,
) -> Result<CellDetection, SegError> {
    det.validate()?;
    let mpp = det.mpp;
    let s = det.tile_size;
    let m = det.margin_px();
    if roi_um.iter().any(|v| !v.is_finite()) || roi_um[2] <= roi_um[0] || roi_um[3] <= roi_um[1] {
        return Err(SegError::InvalidParameter(format!(
            "wilayah deteksi tidak sah: {roi_um:?}"
        )));
    }
    if let Some((tm, tm_mpp)) = tissue_mask {
        if !(tm_mpp.is_finite() && tm_mpp > 0.0) || tm.is_empty() {
            return Err(SegError::InvalidParameter(
                "masker jaringan / mpp-nya tidak sah".to_string(),
            ));
        }
    }
    let w_px = ((roi_um[2] - roi_um[0]) / mpp).ceil().max(1.0) as usize;
    let h_px = ((roi_um[3] - roi_um[1]) / mpp).ceil().max(1.0) as usize;
    let need = (w_px as u64)
        .saturating_mul(h_px as u64)
        .saturating_mul(REGION_BYTES_PER_PX);
    if need > max_bytes {
        return Err(SegError::InvalidParameter(format!(
            "wilayah {w_px}×{h_px} px pada {mpp} µm/px melebihi anggaran memori objek utuh"
        )));
    }
    let in_tissue = |ux: f64, uy: f64| -> bool {
        let Some((tm, tm_mpp)) = tissue_mask else {
            return true;
        };
        let (th, tw) = tm.dim();
        let tx = (ux / tm_mpp).floor();
        let ty = (uy / tm_mpp).floor();
        tx >= 0.0
            && ty >= 0.0
            && (tx as usize) < tw
            && (ty as usize) < th
            && tm[[ty as usize, tx as usize]] != 0
    };
    let tissue_has = |x0: usize, y0: usize| -> bool {
        let Some((tm, tm_mpp)) = tissue_mask else {
            return true;
        };
        let (th, tw) = tm.dim();
        let ux0 = roi_um[0] + x0 as f64 * mpp;
        let uy0 = roi_um[1] + y0 as f64 * mpp;
        let tx0 = ((ux0 / tm_mpp).floor().max(0.0) as usize).min(tw);
        let ty0 = ((uy0 / tm_mpp).floor().max(0.0) as usize).min(th);
        let tx1 = (((ux0 + s as f64 * mpp) / tm_mpp).ceil().max(0.0) as usize).min(tw);
        let ty1 = (((uy0 + s as f64 * mpp) / tm_mpp).ceil().max(0.0) as usize).min(th);
        (ty0..ty1).any(|y| (tx0..tx1).any(|x| tm[[y, x]] != 0))
    };

    // 1. Raster wilayah dari tile bertumpang tindih.
    let stride = s - 2 * m;
    let xs = starts(w_px, s, stride);
    let ys = starts(h_px, s, stride);
    let own_x = ownership(&xs, s, w_px);
    let own_y = ownership(&ys, s, h_px);
    let mut todo: Vec<(usize, usize, usize, usize)> = Vec::new();
    let mut tiles_skipped = 0usize;
    for (j, &y0) in ys.iter().enumerate() {
        for (i, &x0) in xs.iter().enumerate() {
            if tissue_has(x0, y0) {
                todo.push((i, j, x0, y0));
            } else {
                tiles_skipped += 1;
            }
        }
    }
    struct Raster {
        h: Array2<f32>,
        he: Array2<f32>,
        tissue: Array2<u8>,
    }
    let raster = Mutex::new(Raster {
        h: Array2::zeros((h_px, w_px)),
        he: Array2::zeros((h_px, w_px)),
        tissue: Array2::zeros((h_px, w_px)),
    });
    let poisoned = || SegError::InvalidParameter("raster wilayah rusak".to_string());
    let total = todo.len().max(1);
    let done = AtomicUsize::new(0);
    let run_one = |k: usize| -> Result<(), SegError> {
        if progress.is_cancelled() {
            return Err(SegError::Cancelled);
        }
        let (i, j, x0, y0) = todo[k];
        let rgba = read_tile(
            roi_um[0] + x0 as f64 * mpp,
            roi_um[1] + y0 as f64 * mpp,
            s,
            s,
        )?;
        let [hch, ech, _] = deconvolve(&rgba, s, s, &det.stain)?;
        let smooth = if det.smooth_sigma_um > 0.0 {
            h3d_imgproc::gaussian_blur_f32(&hch, det.smooth_sigma_um / mpp)
        } else {
            hch.clone()
        };
        let od = h3d_imgproc::od_total(&rgba, s, s)?;
        let (ox0, ox1) = own_x[i];
        let (oy0, oy1) = own_y[j];
        {
            let mut r = raster.lock().map_err(|_| poisoned())?;
            for gy in oy0..oy1.min(h_px) {
                let ly = gy - y0;
                for gx in ox0..ox1.min(w_px) {
                    let lx = gx - x0;
                    r.h[[gy, gx]] = smooth[[ly, lx]];
                    r.he[[gy, gx]] = hch[[ly, lx]] - ech[[ly, lx]];
                    r.tissue[[gy, gx]] = u8::from(od[[ly, lx]] > 0.1);
                }
            }
        }
        let d = done.fetch_add(1, Ordering::SeqCst) + 1;
        progress.set(0.8 * d as f32 / total as f32, "Deteksi objek");
        Ok(())
    };
    let results: Vec<Result<(), SegError>> = {
        use rayon::prelude::*;
        (0..todo.len()).into_par_iter().map(run_one).collect()
    };
    for r in results {
        r?;
    }
    let Raster {
        h: hmap,
        he,
        tissue,
    } = raster.into_inner().map_err(|_| poisoned())?;

    // 2. Satu ambang untuk seluruh wilayah.
    let n_tissue = tissue.iter().filter(|&&v| v != 0).count();
    let otsu = if det.adaptive_threshold && n_tissue > 64 {
        h3d_imgproc::otsu_masked(&hmap, &tissue)
    } else {
        0.0
    };
    drop(tissue);
    let thr = otsu.max(det.h_threshold);
    if progress.is_cancelled() {
        return Err(SegError::Cancelled);
    }

    // 3. Masker dan morfologi.
    let mut mask = hmap.mapv(|v| u8::from(v > thr));
    if det.open_radius_um > 0.0 {
        mask = h3d_imgproc::open(&mask, det.open_radius_um / mpp);
    }
    if det.close_radius_um > 0.0 {
        mask = h3d_imgproc::close(&mask, det.close_radius_um / mpp);
    }
    let hole_px = (det.max_hole_area_um2 / (mpp * mpp)).floor() as usize;
    mask = h3d_imgproc::fill_small_holes(&mask, hole_px);
    let min_px = ((det.min_area_um2 / (mpp * mpp)).ceil() as usize).max(1);
    if min_px > 1 {
        mask = h3d_imgproc::remove_small_components(&mask, min_px);
    }
    progress.set(0.85, "Deteksi objek");

    // 4. Objek = komponen tersambung.
    let (labels, n_obj) = h3d_imgproc::label_components_conn(&mask, Connectivity::Four);
    drop(mask);
    let mut area = vec![0usize; n_obj + 1];
    let mut sx = vec![0f64; n_obj + 1];
    let mut sy = vec![0f64; n_obj + 1];
    let mut sum_h = vec![0f64; n_obj + 1];
    let mut sum_he = vec![0f64; n_obj + 1];
    let mut in_t = vec![0usize; n_obj + 1];
    for ((y, x), &l) in labels.indexed_iter() {
        if l == 0 {
            continue;
        }
        let i = l as usize;
        area[i] += 1;
        sx[i] += x as f64;
        sy[i] += y as f64;
        sum_h[i] += f64::from(hmap[[y, x]]);
        sum_he[i] += f64::from(he[[y, x]]);
        if in_tissue(
            roi_um[0] + (x as f64 + 0.5) * mpp,
            roi_um[1] + (y as f64 + 0.5) * mpp,
        ) {
            in_t[i] += 1;
        }
    }
    drop(hmap);
    drop(he);
    progress.set(0.9, "Deteksi objek");
    let (outer, holes) = if keep_polygons {
        (
            h3d_imgproc::label_contours(&labels, n_obj),
            h3d_imgproc::label_hole_contours(&labels, n_obj),
        )
    } else {
        (Vec::new(), Vec::new())
    };
    drop(labels);
    let eps = (det.contour_tolerance_um / mpp) as f32;
    let smooth_poly = |c: &[[f32; 2]], cx: f64, cy: f64| -> Vec<[f32; 2]> {
        h3d_imgproc::chaikin_closed(&h3d_imgproc::simplify_closed(c, eps), 1)
            .into_iter()
            .map(|v| {
                [
                    ((f64::from(v[0]) - cx) * mpp) as f32,
                    ((f64::from(v[1]) - cy) * mpp) as f32,
                ]
            })
            .collect()
    };
    let t_strict = 1.2 * f64::from(thr);
    let mut nuclei = Vec::new();
    let mut n_min = 0usize;
    for i in 1..=n_obj {
        let a = area[i];
        if a < min_px {
            continue;
        }
        if tissue_mask.is_some() && 2 * in_t[i] < a {
            continue;
        }
        if det.h_over_e && sum_he[i] < 0.0 {
            continue;
        }
        let score = sum_h[i] / a as f64;
        if score >= t_strict {
            n_min += 1;
        }
        // Pusat piksel di +0,5 (aturan 5).
        let cx = sx[i] / a as f64 + 0.5;
        let cy = sy[i] / a as f64 + 0.5;
        let (polygon_um, holes_um) = match outer.get(i) {
            Some(c) if c.len() >= 3 => (
                smooth_poly(c, cx, cy),
                holes
                    .get(i)
                    .map(|hs| {
                        hs.iter()
                            .map(|hc| smooth_poly(hc, cx, cy))
                            .filter(|p| p.len() >= 3)
                            .collect()
                    })
                    .unwrap_or_default(),
            ),
            _ => (Vec::new(), Vec::new()),
        };
        nuclei.push(DetectedNucleus {
            center_um: [roi_um[0] + cx * mpp, roi_um[1] + cy * mpp],
            area_um2: a as f64 * mpp * mpp,
            score: score as f32,
            polygon_um,
            holes_um,
        });
    }
    progress.set(1.0, "Deteksi objek");
    let n = nuclei.len();
    Ok(CellDetection {
        nuclei,
        n,
        n_min,
        n_max: n,
        roi_um,
        mpp,
        method: det.method_name(),
        params_hash: det.params_hash(),
        tiles_run: todo.len(),
        tiles_skipped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use h3d_core::NoProgress;

    type Ring = (f64, f64, f64, f64);
    type Disc = (f64, f64, f64);

    /// Citra sintetis `w × h` px: latar putih, cincin `(cx, cy, r0, r1)` dan
    /// cakram `(cx, cy, r)` berwarna ungu hematoksilin.
    fn scene(w: usize, h: usize, rings: &[Ring], discs: &[Disc]) -> Vec<u8> {
        let mut img = vec![255u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
                let in_ring = rings.iter().any(|&(cx, cy, r0, r1)| {
                    let d = ((px - cx).powi(2) + (py - cy).powi(2)).sqrt();
                    d >= r0 && d <= r1
                });
                let in_disc = discs
                    .iter()
                    .any(|&(cx, cy, r)| (px - cx).powi(2) + (py - cy).powi(2) <= r * r);
                if in_ring || in_disc {
                    let i = (y * w + x) * 4;
                    img[i..i + 4].copy_from_slice(&[70, 40, 140, 255]);
                }
            }
        }
        img
    }

    type Reader = Box<dyn Fn(f64, f64, usize, usize) -> Result<Vec<u8>, SegError> + Sync>;

    /// Pembaca tile dari citra sintetis pada `mpp = 1`; di luar citra putih.
    fn reader(img: Vec<u8>, w: usize, h: usize) -> Reader {
        Box::new(move |x_um, y_um, tw, th| {
            let x0 = x_um.round() as i64;
            let y0 = y_um.round() as i64;
            let mut out = vec![255u8; tw * th * 4];
            for ty in 0..th {
                for tx in 0..tw {
                    let (sx, sy) = (x0 + tx as i64, y0 + ty as i64);
                    if sx >= 0 && sy >= 0 && (sx as usize) < w && (sy as usize) < h {
                        let si = (sy as usize * w + sx as usize) * 4;
                        let di = (ty * tw + tx) * 4;
                        out[di..di + 4].copy_from_slice(&img[si..si + 4]);
                    }
                }
            }
            Ok(out)
        })
    }

    /// Ambang tetap: adegan sintetis hanya berisi objek berwarna seragam,
    /// sehingga Otsu atas "jaringan" akan membelah objek itu sendiri.
    fn det(tile: usize) -> WholeRegionDetector {
        WholeRegionDetector {
            tile_size: tile,
            mpp: 1.0,
            adaptive_threshold: false,
            ..WholeRegionDetector::default()
        }
    }

    fn run(
        d: &WholeRegionDetector,
        rd: &Reader,
        roi: [f64; 4],
        polys: bool,
    ) -> Result<CellDetection, SegError> {
        detect_regions(d, rd.as_ref(), roi, None, polys, u64::MAX, &NoProgress)
    }

    #[test]
    fn ring_larger_than_tile_is_one_object_with_one_hole() {
        // Cincin berdiameter 300 px melintasi banyak tile 128 px, plus
        // serpihan di lumen dan satu cakram terpisah.
        let (w, h) = (400usize, 400usize);
        let img = scene(
            w,
            h,
            &[(200.0, 200.0, 130.0, 150.0)],
            &[(200.0, 200.0, 6.0), (380.0, 20.0, 8.0)],
        );
        let rd = reader(img, w, h);
        let r = run(&det(128), &rd, [0.0, 0.0, w as f64, h as f64], true).unwrap();
        assert_eq!(r.n, 3, "cincin + serpihan + cakram");
        assert!(r.tiles_run > 4);
        let ring = r
            .nuclei
            .iter()
            .max_by(|a, b| a.area_um2.total_cmp(&b.area_um2))
            .unwrap();
        // Tepi boleh bergeser ±1 px (pemulusan Gauss σ = 1 px + ambang).
        let pi = std::f64::consts::PI;
        let lo = pi * (149.0f64.powi(2) - 131.0f64.powi(2));
        let hi = pi * (151.0f64.powi(2) - 129.0f64.powi(2));
        assert!(
            (lo..=hi).contains(&ring.area_um2),
            "luas {} di luar [{lo}, {hi}]",
            ring.area_um2
        );
        assert_eq!(ring.holes_um.len(), 1, "lumen harus tetap satu lubang");
        assert!(ring.polygon_um.len() >= 12);
        assert!((ring.center_um[0] - 200.0).abs() < 1.0);
        assert!((ring.center_um[1] - 200.0).abs() < 1.0);
        let inner = h3d_imgproc::signed_area(&ring.holes_um[0]);
        let want_hole = std::f64::consts::PI * 130.0f64.powi(2);
        assert!(
            (inner - want_hole).abs() / want_hole < 0.05,
            "lubang {inner}"
        );
    }

    #[test]
    fn result_does_not_depend_on_tiling() {
        let (w, h) = (300usize, 260usize);
        let img = scene(
            w,
            h,
            &[(120.0, 120.0, 60.0, 75.0), (250.0, 200.0, 10.0, 22.0)],
            &[(30.0, 230.0, 9.0)],
        );
        let rd = reader(img, w, h);
        let roi = [0.0, 0.0, w as f64, h as f64];
        let a = run(&det(96), &rd, roi, true).unwrap();
        let b = run(&det(256), &rd, roi, true).unwrap();
        assert_eq!(a.n, 3);
        assert_eq!(a.nuclei, b.nuclei);
        assert_eq!(a.params_hash, b.params_hash);
        // Tanpa poligon diminta → kosong.
        let c = run(&det(96), &rd, roi, false).unwrap();
        assert_eq!(c.n, 3);
        assert!(c
            .nuclei
            .iter()
            .all(|n| n.polygon_um.is_empty() && n.holes_um.is_empty()));
    }

    #[test]
    fn separate_rings_are_separate_objects() {
        let (w, h) = (260usize, 140usize);
        let img = scene(
            w,
            h,
            &[(65.0, 70.0, 35.0, 50.0), (195.0, 70.0, 35.0, 50.0)],
            &[],
        );
        let rd = reader(img, w, h);
        let r = run(&det(128), &rd, [0.0, 0.0, w as f64, h as f64], true).unwrap();
        assert_eq!(r.n, 2);
        assert!(r.nuclei.iter().all(|n| n.holes_um.len() == 1));
    }

    #[test]
    fn invalid_inputs_are_errors() {
        let rd = reader(vec![255u8; 64 * 64 * 4], 64, 64);
        let d = det(64);
        assert!(run(&d, &rd, [0.0, 0.0, 0.0, 10.0], false).is_err());
        assert!(run(&d, &rd, [0.0, 0.0, f64::NAN, 10.0], false).is_err());
        // Melebihi anggaran.
        let roi = [0.0, 0.0, 64.0, 64.0];
        assert!(detect_regions(&d, rd.as_ref(), roi, None, false, 100, &NoProgress).is_err());
        let bad = WholeRegionDetector {
            close_radius_um: -1.0,
            ..d.clone()
        };
        assert!(bad.validate().is_err());
        let bad = WholeRegionDetector {
            mpp: 0.0,
            ..d.clone()
        };
        assert!(bad.validate().is_err());
        // Wilayah kosong (putih) → nol objek, bukan error.
        assert_eq!(run(&d, &rd, roi, false).unwrap().n, 0);
    }

    #[test]
    fn working_mpp_fits_budget() {
        let d = WholeRegionDetector::default();
        assert_eq!(d.working_mpp([0.0, 0.0, 100.0, 100.0], 1 << 30), d.mpp);
        let roi = [0.0, 0.0, 10_000.0, 8_000.0];
        let max = 64 * 1024 * 1024;
        let m = d.working_mpp(roi, max);
        assert!(m > d.mpp);
        let px = ((roi[2] / m).ceil() * (roi[3] / m).ceil()) as u64;
        assert!(px * REGION_BYTES_PER_PX <= max);
        assert_eq!(m, d.working_mpp(roi, max));
    }
}
