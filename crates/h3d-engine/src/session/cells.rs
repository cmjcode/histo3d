//! Penghitungan sel/inti otomatis per irisan (histo3d.md 8.12.5 butir 2):
//! StarDist (ONNX, `task = "nuclei"`) atau pendeteksi LoG klasik dijalankan
//! langsung pada citra slide, tanpa label. Hasil = daftar inti + jumlah
//! dengan rentang ketidakpastian (aturan 18) + densitas per mm² jaringan.
//! Mode "objek utuh" ([`CellMethod::WholeRegions`]) menghitung tiap wilayah
//! hematoksilin tersambung sebagai satu objek (tanpa pemisahan watershed).
//! Tidak mengubah label maupun proyek.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use h3d_core::{Progress, SlideId};
use h3d_imgproc::nuclei_density_per_mm2;
use h3d_segment::{
    detect_cells, detect_regions, CellDetection, ClassicNucleiDetector, NucleiDetector,
    StarDistDetector, WatershedNucleiDetector, WholeRegionDetector,
};
use serde::{Deserialize, Serialize};

use super::images::read_region_at_mpp;
use super::{ModelEntry, Session};
use crate::error::EngineError;

/// Metode deteksi sel.
#[derive(Clone, Debug)]
pub enum CellMethod {
    /// Segmentasi klasik ala QuPath: ambang H + watershed EDT (bawaan).
    Watershed(WatershedNucleiDetector),
    /// LoG klasik pada kanal hematoksilin.
    Classic(ClassicNucleiDetector),
    /// Objek utuh: tiap wilayah hematoksilin tersambung = satu objek (pita
    /// atau cincin epitel tidak dipotong menjadi keping). Resolusi kerja
    /// bisa diturunkan agar muat `Budgets::cell_region_bytes`.
    WholeRegions(WholeRegionDetector),
    /// Model StarDist di folder `models/<nama>` (atau folder luar).
    StarDist(PathBuf),
}

/// Hasil penghitungan sel satu irisan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CellCount2D {
    /// Irisan.
    pub slide: SlideId,
    /// Deteksi (inti, jumlah, rentang, metode, hash).
    pub detection: CellDetection,
    /// Luas jaringan di dalam wilayah, µm² (`None` bila tidak tersedia).
    pub tissue_area_um2: Option<f64>,
}

impl CellCount2D {
    /// Jumlah inti.
    pub fn n(&self) -> usize {
        self.detection.n
    }

    /// Benar bila hasil dari mode objek utuh (angka = objek, bukan inti).
    pub fn is_whole_regions(&self) -> bool {
        self.detection.method == h3d_segment::REGIONS_METHOD
    }

    /// Densitas inti per mm² jaringan.
    pub fn density_per_mm2(&self) -> Option<f64> {
        let a = self.tissue_area_um2.filter(|a| *a > 0.0)?;
        Some(nuclei_density_per_mm2(self.detection.n, a))
    }

    /// Median luas inti, µm² (0 bila kosong).
    pub fn median_area_um2(&self) -> f64 {
        let mut a: Vec<f64> = self.detection.nuclei.iter().map(|n| n.area_um2).collect();
        if a.is_empty() {
            return 0.0;
        }
        a.sort_by(f64::total_cmp);
        let m = a.len() / 2;
        if a.len().is_multiple_of(2) {
            (a[m - 1] + a[m]) / 2.0
        } else {
            a[m]
        }
    }
}

impl Session {
    /// Baca wilayah slide `slide` mulai `(x_um, y_um)` (ruang U) berukuran
    /// `w × h` piksel pada resolusi `mpp` sebagai RGBA (untuk pratinjau dan
    /// overlay hasil deteksi).
    pub fn read_region_rgba(
        &self,
        slide: SlideId,
        x_um: f64,
        y_um: f64,
        w: usize,
        h: usize,
        mpp: f64,
    ) -> Result<Vec<u8>, EngineError> {
        let entry = self.slide(slide)?.clone();
        let src = self.open_source(slide)?;
        read_region_at_mpp(src.as_ref(), entry.mpp, x_um, y_um, w, h, mpp)
    }

    /// Model terpasang yang bertugas `nuclei` (StarDist).
    pub fn list_nuclei_models(&self) -> Vec<ModelEntry> {
        self.list_models()
            .into_iter()
            .filter(|m| m.spec.as_ref().is_ok_and(|s| s.task == "nuclei"))
            .collect()
    }

    /// Pendeteksi StarDist dari folder `model`, dimuat sekali per sesi.
    pub fn stardist(&self, model: &Path) -> Result<Arc<StarDistDetector>, EngineError> {
        if let Ok(cache) = self.stardist.lock() {
            if let Some((dir, det)) = cache.as_ref() {
                if dir == model {
                    return Ok(Arc::clone(det));
                }
            }
        }
        let det = Arc::new(StarDistDetector::load(model)?);
        if let Ok(mut cache) = self.stardist.lock() {
            *cache = Some((model.to_path_buf(), Arc::clone(&det)));
        }
        Ok(det)
    }

    /// Hitung sel pada irisan `slide`. `roi_um` = kotak `[lo, hi]` ruang U
    /// (µm); `None` = seluruh slide. `keep_polygons` menyimpan poligon tiap
    /// inti (untuk overlay; hindari untuk seluruh slide).
    pub fn count_cells(
        &self,
        slide: SlideId,
        roi_um: Option<([f64; 2], [f64; 2])>,
        method: &CellMethod,
        keep_polygons: bool,
        p: &dyn Progress,
    ) -> Result<CellCount2D, EngineError> {
        let entry = self.slide(slide)?.clone();
        let size = entry.size_um();
        let (lo, hi) = roi_um.unwrap_or(([0.0, 0.0], size));
        let lo = [lo[0].max(0.0), lo[1].max(0.0)];
        let hi = [hi[0].min(size[0]), hi[1].min(size[1])];
        if !(hi[0] > lo[0] && hi[1] > lo[1]) {
            return Err(EngineError::invalid("wilayah hitung di luar irisan"));
        }
        let roi = [lo[0], lo[1], hi[0], hi[1]];
        if let CellMethod::WholeRegions(w) = method {
            w.validate()?;
            let max_bytes = self.budgets.cell_region_bytes;
            let det = WholeRegionDetector {
                mpp: w.working_mpp(roi, max_bytes),
                ..w.clone()
            };
            let src = self.open_source(slide)?;
            let mpp_l0 = entry.mpp;
            let mpp_det = det.mpp;
            let read_tile = |x_um: f64, y_um: f64, w: usize, h: usize| {
                read_region_at_mpp(src.as_ref(), mpp_l0, x_um, y_um, w, h, mpp_det).map_err(|e| {
                    h3d_segment::SegError::InvalidParameter(format!("gagal membaca tile: {e}"))
                })
            };
            let reg = self.registration_image(slide).ok();
            let tissue = reg.as_ref().map(|r| (&r.mask, r.mpp));
            let detection =
                detect_regions(&det, &read_tile, roi, tissue, keep_polygons, max_bytes, p)?;
            let tissue_area_um2 = reg.as_ref().map(|r| tissue_area_in(r, roi));
            return Ok(CellCount2D {
                slide,
                detection,
                tissue_area_um2,
            });
        }
        let det: Arc<dyn NucleiDetector> = match method {
            CellMethod::Watershed(w) => {
                w.validate()?;
                Arc::new(w.clone())
            }
            CellMethod::Classic(c) => {
                c.validate()?;
                Arc::new(c.clone())
            }
            CellMethod::StarDist(dir) => self.stardist(dir)?,
            CellMethod::WholeRegions(_) => {
                return Err(EngineError::invalid("metode objek utuh salah cabang"))
            }
        };
        let src = self.open_source(slide)?;
        let mpp_det = det.mpp();
        let mpp_l0 = entry.mpp;
        let read_tile = |x_um: f64, y_um: f64, w: usize, h: usize| {
            read_region_at_mpp(src.as_ref(), mpp_l0, x_um, y_um, w, h, mpp_det).map_err(|e| {
                h3d_segment::SegError::InvalidParameter(format!("gagal membaca tile: {e}"))
            })
        };
        let reg = self.registration_image(slide).ok();
        let tissue = reg.as_ref().map(|r| (&r.mask, r.mpp));
        let detection = detect_cells(det.as_ref(), &read_tile, roi, tissue, keep_polygons, p)?;
        let tissue_area_um2 = reg.as_ref().map(|r| tissue_area_in(r, roi));
        Ok(CellCount2D {
            slide,
            detection,
            tissue_area_um2,
        })
    }
}

/// Luas jaringan (µm²) masker registrasi `r` di dalam kotak `roi`
/// `[x0, y0, x1, y1]` ruang U.
fn tissue_area_in(r: &h3d_register::RegImage, roi: [f64; 4]) -> f64 {
    let (h, w) = r.mask.dim();
    let to_px = |u: f64, o: f64| ((u - o) / r.mpp).max(0.0);
    let x0 = to_px(roi[0], r.origin_um[0]).floor() as usize;
    let y0 = to_px(roi[1], r.origin_um[1]).floor() as usize;
    let x1 = (to_px(roi[2], r.origin_um[0]).ceil() as usize).min(w);
    let y1 = (to_px(roi[3], r.origin_um[1]).ceil() as usize).min(h);
    if x1 <= x0 || y1 <= y0 {
        return 0.0;
    }
    let n = r
        .mask
        .slice(ndarray::s![y0..y1, x0..x1])
        .iter()
        .filter(|&&v| v != 0)
        .count();
    n as f64 * r.mpp * r.mpp
}
