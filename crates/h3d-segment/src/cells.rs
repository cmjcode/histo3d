//! Deteksi dan penghitungan sel/inti otomatis dari citra (histo3d.md
//! 8.12.5 butir 2, 8.5.4 `"task": "nuclei"`): tanpa label manual, hasilnya
//! daftar inti (pusat, luas, poligon) yang langsung dihitung dan ditandai.
//!
//! Dua pendeteksi di balik satu trait [`NucleiDetector`]:
//! - [`StarDistDetector`]: model StarDist (Schmidt dkk. 2018) dalam ONNX via
//!   `tract` (Rust murni, CPU). Model mengeluarkan peta probabilitas objek
//!   dan `n_rays` jarak radial per piksel; inti = poligon star-convex.
//!   Pasca-proses NMS ditulis di sini: kandidat diproses dari probabilitas
//!   tertinggi, dirasterkan ke peta instance tile; kandidat yang
//!   `> nms_threshold` bagian luasnya sudah terisi inti lain dibuang.
//!   (StarDist asli memakai irisan poligon analitik; versi raster ini
//!   setara pada resolusi piksel dan sekaligus menghasilkan masker instance.)
//! - [`ClassicNucleiDetector`]: LoG multiskala pada kanal hematoksilin
//!   (`h3d_imgproc::detect_nuclei`) + saringan OD/kontras/H≥E, tanpa model;
//!   cadangan bila tidak ada `model.onnx`.
//!
//! [`detect_cells`] menjalankan pendeteksi per tile bertumpang tindih atas
//! sebuah wilayah slide. Tiap inti hanya dimiliki satu tile: batas
//! kepemilikan = titik tengah daerah tumpang tindih dua tile bertetangga,
//! sehingga tidak ada inti ganda dan tidak ada yang hilang. Tile diproses
//! paralel, hasil digabung dalam urutan tile (deterministik, aturan 9).
//!
//! Ketidakpastian (aturan 18): tiap inti membawa `score`; NMS berurutan
//! probabilitas sehingga himpunan inti pada ambang lebih ketat = himpunan
//! pada ambang longgar yang disaring `score`. Pendeteksi memberi tiga ambang
//! `[longgar, utama, ketat]` per tile → `n_max`, `n`, `n_min`.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use h3d_core::Progress;
use h3d_imgproc::color::{deconvolve, StainMatrix};
use h3d_imgproc::nuclei::{detect_nuclei, NucleiParams};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use tract_onnx::prelude::*;
use tract_onnx::tract_hir::internal::DimLike;

use crate::ai::{ModelSpec, TileReader};
use crate::error::SegError;

/// Tata letak tensor model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TensorLayout {
    /// `[N, C, H, W]` (PyTorch, bawaan).
    #[default]
    Nchw,
    /// `[N, H, W, C]` (Keras/TensorFlow; ekspor tf2onnx tanpa transposisi).
    Nhwc,
}

/// Spesifikasi head inti StarDist di `model.json` (`"nuclei": {...}`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NucleiHead {
    /// Jumlah sinar radial per inti (bawaan 32).
    pub n_rays: usize,
    /// Faktor subsampel keluaran `[gy, gx]` (bawaan `[2, 2]` seperti
    /// `2D_versatile_he`).
    pub grid: [usize; 2],
    /// Ambang probabilitas objek (bawaan 0,5).
    pub prob_threshold: f32,
    /// Ambang tumpang tindih NMS (bawaan 0,3).
    pub nms_threshold: f32,
    /// Tata letak masukan dan keluaran.
    pub layout: TensorLayout,
    /// Normalisasi persentil per tile `[p_lo, p_hi]` atas intensitas 0..255
    /// sebelum `mean`/`std` (StarDist: `[1, 99.8]`); `None` = tidak ada.
    pub percentile_norm: Option<[f32; 2]>,
    /// Selisih ambang untuk rentang ketidakpastian (bawaan 0,1).
    pub threshold_delta: f32,
    /// Tepi tile yang diabaikan (piksel; bawaan 32).
    pub margin_px: usize,
}

impl Default for NucleiHead {
    fn default() -> Self {
        Self {
            n_rays: 32,
            grid: [2, 2],
            prob_threshold: 0.5,
            nms_threshold: 0.3,
            layout: TensorLayout::Nchw,
            percentile_norm: None,
            threshold_delta: 0.1,
            margin_px: 32,
        }
    }
}

impl NucleiHead {
    /// Validasi nilai.
    pub fn validate(&self, input_size: usize) -> Result<(), SegError> {
        if self.n_rays < 3 || self.n_rays > 256 {
            return Err(SegError::Model(format!(
                "n_rays {} di luar rentang 3..=256",
                self.n_rays
            )));
        }
        if self
            .grid
            .iter()
            .any(|&g| g == 0 || !input_size.is_multiple_of(g))
        {
            return Err(SegError::Model(format!(
                "grid {:?} harus membagi input_size {input_size}",
                self.grid
            )));
        }
        let unit = |v: f32| v.is_finite() && (0.0..=1.0).contains(&v);
        if !unit(self.prob_threshold) || !unit(self.nms_threshold) {
            return Err(SegError::Model(
                "prob_threshold/nms_threshold harus di [0, 1]".to_string(),
            ));
        }
        if !(self.threshold_delta.is_finite() && self.threshold_delta >= 0.0) {
            return Err(SegError::Model("threshold_delta tidak sah".to_string()));
        }
        if let Some([lo, hi]) = self.percentile_norm {
            if !(lo.is_finite() && hi.is_finite() && 0.0 <= lo && lo < hi && hi <= 100.0) {
                return Err(SegError::Model(format!(
                    "percentile_norm {:?} tidak sah",
                    self.percentile_norm
                )));
            }
        }
        if self.margin_px * 2 >= input_size {
            return Err(SegError::Model(format!(
                "margin_px {} terlalu besar untuk input_size {input_size}",
                self.margin_px
            )));
        }
        Ok(())
    }
}

/// Satu inti terdeteksi. Koordinat µm absolut dalam ruang citra yang dibaca
/// `read_tile` (ruang U bila pembaca bekerja di U).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetectedNucleus {
    /// Pusat, µm.
    pub center_um: [f64; 2],
    /// Luas, µm² (luas poligon; cakram untuk pendeteksi klasik).
    pub area_um2: f64,
    /// Skor pendeteksi (probabilitas StarDist atau kanal H pendeteksi
    /// klasik); lebih besar = lebih yakin.
    pub score: f32,
    /// Verteks poligon relatif pusat, µm (kosong bila tidak diminta).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub polygon_um: Vec<[f32; 2]>,
    /// Kontur lubang di dalam poligon (mis. lumen di dalam cincin epitel
    /// pada mode objek utuh), relatif pusat, µm. Kosong untuk inti.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub holes_um: Vec<Vec<[f32; 2]>>,
}

impl DetectedNucleus {
    /// Jari-jari ekuivalen, µm.
    pub fn radius_um(&self) -> f64 {
        (self.area_um2 / std::f64::consts::PI).sqrt()
    }
}

/// Satu inti dalam koordinat piksel tile: `(y_px, x_px, luas_px, skor,
/// poligon relatif piksel)`.
pub type TileNucleus = (f64, f64, f64, f32, Vec<[f32; 2]>);

/// Hasil pendeteksi untuk satu tile (koordinat piksel tile).
#[derive(Debug, Clone, Default)]
pub struct TileNuclei {
    /// Inti dengan `skor ≥ thresholds[0]`.
    pub nuclei: Vec<TileNucleus>,
    /// Ambang skor `[longgar, utama, ketat]` untuk tile ini.
    pub thresholds: [f32; 3],
}

/// Pendeteksi inti per tile.
pub trait NucleiDetector: Send + Sync {
    /// Sisi tile (piksel).
    fn tile_size(&self) -> usize;
    /// Ukuran piksel masukan (µm).
    fn mpp(&self) -> f64;
    /// Tepi tile yang hasilnya tidak dipercaya (piksel).
    fn margin_px(&self) -> usize;
    /// Nama metode untuk laporan (mis. `stardist he 2D_versatile_he`).
    fn method_name(&self) -> String;
    /// Hash parameter stabil.
    fn params_hash(&self) -> u64;
    /// Deteksi pada tile RGBA `S × S`. Mengembalikan semua kandidat dengan
    /// `score ≥ thresholds[0]` (ambang longgar).
    fn detect_tile(&self, rgba: &[u8], keep_polygons: bool) -> Result<TileNuclei, SegError>;
}

pub(crate) fn fnv1a(parts: &[&[u8]]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for bytes in parts {
        h ^= bytes.len() as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
        for &b in *bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

// ---------------------------------------------------------------------------
// StarDist
// ---------------------------------------------------------------------------

/// Pendeteksi StarDist (ONNX via tract).
pub struct StarDistDetector {
    spec: ModelSpec,
    head: NucleiHead,
    model: Arc<TypedRunnableModel>,
    /// Indeks keluaran probabilitas dan jarak.
    prob_out: usize,
    dist_out: usize,
}

impl std::fmt::Debug for StarDistDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StarDistDetector")
            .field("spec", &self.spec.name)
            .field("head", &self.head)
            .finish()
    }
}

impl StarDistDetector {
    /// Muat `model.json` (`task = "nuclei"`) + `model.onnx` dari `model_dir`.
    pub fn load(model_dir: &Path) -> Result<Self, SegError> {
        let json = std::fs::read_to_string(model_dir.join("model.json"))?;
        let spec = ModelSpec::from_json(&json)?;
        let onnx = std::fs::read(model_dir.join("model.onnx"))?;
        Self::from_bytes(spec, &onnx)
    }

    /// Bangun dari spesifikasi dan byte ONNX. Model harus punya dua keluaran:
    /// probabilitas (1 kanal) dan jarak (`n_rays` kanal), masing-masing
    /// `S/grid` piksel.
    pub fn from_bytes(spec: ModelSpec, onnx: &[u8]) -> Result<Self, SegError> {
        spec.validate()?;
        if spec.task != "nuclei" {
            return Err(SegError::Model(format!(
                "tugas '{}' bukan nuclei",
                spec.task
            )));
        }
        let head = spec
            .nuclei
            .clone()
            .ok_or_else(|| SegError::Model("model.json tanpa bagian nuclei".to_string()))?;
        let s = spec.input_size;
        head.validate(s)?;
        let in_shape: TVec<usize> = match head.layout {
            TensorLayout::Nchw => tvec![1, 3, s, s],
            TensorLayout::Nhwc => tvec![1, s, s, 3],
        };
        let mut reader = std::io::Cursor::new(onnx);
        let model = tract_onnx::onnx()
            .model_for_read(&mut reader)
            .map_err(|e| SegError::Model(format!("ONNX gagal diurai: {e}")))?
            .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), in_shape))
            .map_err(|e| SegError::Model(format!("fakta masukan ditolak: {e}")))?
            .into_optimized()
            .map_err(|e| SegError::Model(format!("optimasi gagal: {e}")))?
            .into_runnable()
            .map_err(|e| SegError::Model(format!("model tidak bisa dijalankan: {e}")))?;
        let n_out = model.model().outputs.len();
        if n_out != 2 {
            return Err(SegError::Model(format!(
                "model nuclei harus punya 2 keluaran (prob, dist), didapat {n_out}"
            )));
        }
        // Tentukan keluaran mana yang probabilitas lewat jumlah kanal.
        let mut chans = [0usize; 2];
        for (i, ch) in chans.iter_mut().enumerate() {
            let fact = model
                .model()
                .output_fact(i)
                .map_err(|e| SegError::Model(format!("fakta keluaran {i}: {e}")))?;
            let shape: Vec<Option<usize>> = fact.shape.iter().map(|d| d.to_usize().ok()).collect();
            if shape.len() != 4 {
                return Err(SegError::Model(format!(
                    "keluaran {i} harus 4D, didapat {} dimensi",
                    shape.len()
                )));
            }
            let c_idx = match head.layout {
                TensorLayout::Nchw => 1,
                TensorLayout::Nhwc => 3,
            };
            *ch = shape[c_idx].unwrap_or(0);
        }
        let (prob_out, dist_out) = if chans[0] == 1 && chans[1] != 1 {
            (0, 1)
        } else if chans[1] == 1 && chans[0] != 1 {
            (1, 0)
        } else {
            return Err(SegError::Model(format!(
                "kanal keluaran {chans:?} tidak dikenali sebagai (prob=1, dist={})",
                head.n_rays
            )));
        };
        if chans[dist_out] != 0 && chans[dist_out] != head.n_rays {
            return Err(SegError::Model(format!(
                "keluaran jarak {} kanal ≠ n_rays {}",
                chans[dist_out], head.n_rays
            )));
        }
        Ok(Self {
            spec,
            head,
            model,
            prob_out,
            dist_out,
        })
    }

    /// Spesifikasi model.
    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    /// Head inti.
    pub fn head(&self) -> &NucleiHead {
        &self.head
    }

    /// Normalisasi tile RGBA → tensor masukan f32 sesuai `layout`.
    fn normalize(&self, rgba: &[u8]) -> Result<Vec<f32>, SegError> {
        let s = self.spec.input_size;
        if rgba.len() != s * s * 4 {
            return Err(SegError::InvalidParameter(format!(
                "tile {} byte, diharapkan {}",
                rgba.len(),
                s * s * 4
            )));
        }
        // Normalisasi persentil (StarDist): per kanal atas tile.
        let (lo, hi): ([f32; 3], [f32; 3]) = match self.head.percentile_norm {
            None => ([0.0; 3], [255.0; 3]),
            Some([p_lo, p_hi]) => {
                let mut lo = [0.0f32; 3];
                let mut hi = [255.0f32; 3];
                for ch in 0..3 {
                    let mut hist = [0u32; 256];
                    for px in rgba.as_chunks::<4>().0 {
                        hist[px[ch] as usize] += 1;
                    }
                    let n = (s * s) as f64;
                    let q = |p: f32| -> f32 {
                        let target = (f64::from(p) / 100.0 * n).round() as u32;
                        let mut acc = 0u32;
                        for (v, &c) in hist.iter().enumerate() {
                            acc += c;
                            if acc >= target.max(1) {
                                return v as f32;
                            }
                        }
                        255.0
                    };
                    lo[ch] = q(p_lo);
                    hi[ch] = q(p_hi).max(lo[ch] + 1e-3);
                }
                (lo, hi)
            }
        };
        let mean = self.spec.mean;
        let std = self.spec.std;
        let mut out = vec![0.0f32; 3 * s * s];
        for (i, px) in rgba.as_chunks::<4>().0.iter().enumerate() {
            for ch in 0..3 {
                let v = (px[ch] as f32 - lo[ch]) / (hi[ch] - lo[ch]);
                let v = (v - mean[ch]) / std[ch];
                match self.head.layout {
                    TensorLayout::Nchw => out[ch * s * s + i] = v,
                    TensorLayout::Nhwc => out[i * 3 + ch] = v,
                }
            }
        }
        Ok(out)
    }

    /// Jalankan model: `(prob [h×w], dist [h×w×R], h, w)` dalam urutan HWR.
    pub fn infer(&self, rgba: &[u8]) -> Result<(Vec<f32>, Vec<f32>, usize, usize), SegError> {
        let s = self.spec.input_size;
        let input = self.normalize(rgba)?;
        let shape: [usize; 4] = match self.head.layout {
            TensorLayout::Nchw => [1, 3, s, s],
            TensorLayout::Nhwc => [1, s, s, 3],
        };
        let tensor = Tensor::from_shape(&shape, &input)
            .map_err(|e| SegError::Model(format!("tensor masukan: {e}")))?;
        let outputs = self
            .model
            .run(tvec![tensor.into_tvalue()])
            .map_err(|e| SegError::Model(format!("inferensi gagal: {e}")))?;
        let h = s / self.head.grid[0];
        let w = s / self.head.grid[1];
        let r = self.head.n_rays;
        let take = |idx: usize, c: usize| -> Result<Vec<f32>, SegError> {
            let out = outputs
                .get(idx)
                .ok_or_else(|| SegError::Model("keluaran hilang".to_string()))?;
            let sh = out.shape();
            let expect: [usize; 4] = match self.head.layout {
                TensorLayout::Nchw => [1, c, h, w],
                TensorLayout::Nhwc => [1, h, w, c],
            };
            if sh != expect {
                return Err(SegError::Model(format!(
                    "bentuk keluaran {sh:?} ≠ {expect:?}"
                )));
            }
            let view = out
                .to_plain_array_view::<f32>()
                .map_err(|e| SegError::Model(format!("keluaran bukan f32: {e}")))?;
            let flat: Vec<f32> = view.iter().copied().collect();
            // Ke urutan HWC.
            Ok(match self.head.layout {
                TensorLayout::Nhwc => flat,
                TensorLayout::Nchw => {
                    let mut v = vec![0.0f32; h * w * c];
                    for k in 0..c {
                        for y in 0..h {
                            for x in 0..w {
                                v[(y * w + x) * c + k] = flat[(k * h + y) * w + x];
                            }
                        }
                    }
                    v
                }
            })
        };
        let prob = take(self.prob_out, 1)?;
        let dist = take(self.dist_out, r)?;
        Ok((prob, dist, h, w))
    }
}

/// Isi poligon (verteks `[x, y]` piksel, pusat sudah ditambahkan) di dalam
/// larik `w × h`; memanggil `f(x, y)` untuk tiap piksel di dalam (aturan
/// genap-ganjil, pusat piksel `+0,5`).
fn fill_polygon<F: FnMut(usize, usize)>(poly: &[[f32; 2]], w: usize, h: usize, mut f: F) {
    if poly.len() < 3 {
        return;
    }
    let y_min = poly.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
    let y_max = poly.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max);
    let ya = (y_min.floor().max(0.0)) as usize;
    let yb = (y_max.ceil().max(0.0) as usize).min(h);
    let mut xs: Vec<f32> = Vec::with_capacity(poly.len());
    for y in ya..yb {
        let yc = y as f32 + 0.5;
        xs.clear();
        let n = poly.len();
        for i in 0..n {
            let a = poly[i];
            let b = poly[(i + 1) % n];
            if (a[1] <= yc && b[1] > yc) || (b[1] <= yc && a[1] > yc) {
                let t = (yc - a[1]) / (b[1] - a[1]);
                xs.push(a[0] + t * (b[0] - a[0]));
            }
        }
        xs.sort_by(f32::total_cmp);
        for pair in xs.chunks(2) {
            if pair.len() < 2 {
                break;
            }
            let xa = (pair[0] - 0.5).ceil().max(0.0) as usize;
            let xb = ((pair[1] - 0.5).floor().max(-1.0) as i64 + 1).max(0) as usize;
            for x in xa..xb.min(w) {
                f(x, y);
            }
        }
    }
}

/// Luas poligon (shoelace), satuan verteks².
fn polygon_area(poly: &[[f32; 2]]) -> f64 {
    let n = poly.len();
    if n < 3 {
        return 0.0;
    }
    let mut a = 0.0f64;
    for i in 0..n {
        let p = poly[i];
        let q = poly[(i + 1) % n];
        a += f64::from(p[0]) * f64::from(q[1]) - f64::from(q[0]) * f64::from(p[1]);
    }
    (a / 2.0).abs()
}

/// Parameter NMS StarDist raster.
#[derive(Debug, Clone, Copy)]
pub struct NmsParams {
    /// Jumlah sinar.
    pub n_rays: usize,
    /// Subsampel keluaran `[gy, gx]`.
    pub grid: [usize; 2],
    /// Sisi tile.
    pub tile: usize,
    /// Ambang probabilitas.
    pub prob_threshold: f32,
    /// Ambang tumpang tindih.
    pub nms_threshold: f32,
    /// Simpan poligon.
    pub keep_polygons: bool,
}

/// NMS StarDist versi raster. `prob` `h×w`, `dist` `h×w×R` (HWR). Kandidat
/// `prob ≥ prob_threshold` diproses dari probabilitas tertinggi (seri →
/// indeks raster); yang `> nms_threshold` bagian luasnya sudah terisi
/// dibuang. Mengembalikan inti dalam koordinat piksel tile dan peta
/// instance `tile×tile` (0 = latar).
pub fn stardist_nms(
    prob: &[f32],
    dist: &[f32],
    h: usize,
    w: usize,
    p: &NmsParams,
) -> (Vec<TileNucleus>, Array2<u32>) {
    let s = p.tile;
    let n_rays = p.n_rays;
    let mut labels = Array2::<u32>::zeros((s, s));
    if prob.len() != h * w || dist.len() != h * w * n_rays {
        return (Vec::new(), labels);
    }
    let mut cands: Vec<(f32, usize)> = prob
        .iter()
        .enumerate()
        .filter(|(_, &v)| v >= p.prob_threshold)
        .map(|(i, &v)| (v, i))
        .collect();
    cands.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let angles: Vec<(f32, f32)> = (0..n_rays)
        .map(|k| {
            let phi = 2.0 * std::f32::consts::PI * k as f32 / n_rays as f32;
            (phi.cos(), phi.sin())
        })
        .collect();
    let mut out = Vec::new();
    let mut poly: Vec<[f32; 2]> = Vec::with_capacity(n_rays);
    let mut pixels: Vec<(usize, usize)> = Vec::new();
    let mut next_id = 0u32;
    for (pv, i) in cands {
        let (y, x) = (i / w, i % w);
        let cy = (y as f32 + 0.5) * p.grid[0] as f32;
        let cx = (x as f32 + 0.5) * p.grid[1] as f32;
        poly.clear();
        for (k, (c, sn)) in angles.iter().enumerate() {
            let d = dist[i * n_rays + k].max(0.0);
            poly.push([cx + d * c, cy + d * sn]);
        }
        pixels.clear();
        let mut taken = 0usize;
        fill_polygon(&poly, s, s, |px, py| {
            if labels[[py, px]] != 0 {
                taken += 1;
            }
            pixels.push((px, py));
        });
        if pixels.is_empty() {
            continue;
        }
        if taken as f32 / pixels.len() as f32 > p.nms_threshold {
            continue;
        }
        next_id += 1;
        for &(px, py) in &pixels {
            if labels[[py, px]] == 0 {
                labels[[py, px]] = next_id;
            }
        }
        let area = polygon_area(&poly);
        let rel: Vec<[f32; 2]> = if p.keep_polygons {
            poly.iter().map(|v| [v[0] - cx, v[1] - cy]).collect()
        } else {
            Vec::new()
        };
        out.push((f64::from(cy), f64::from(cx), area, pv, rel));
    }
    (out, labels)
}

impl NucleiDetector for StarDistDetector {
    fn tile_size(&self) -> usize {
        self.spec.input_size
    }
    fn mpp(&self) -> f64 {
        self.spec.mpp
    }
    fn margin_px(&self) -> usize {
        self.head.margin_px
    }
    fn method_name(&self) -> String {
        format!("stardist {} {}", self.spec.name, self.spec.version_string())
            .trim()
            .to_string()
    }
    fn params_hash(&self) -> u64 {
        fnv1a(&[
            b"h3d-cells-stardist-v1",
            self.spec.name.as_bytes(),
            self.spec.version_string().as_bytes(),
            &self.spec.mpp.to_bits().to_le_bytes(),
            &(self.head.n_rays as u64).to_le_bytes(),
            &self.head.prob_threshold.to_bits().to_le_bytes(),
            &self.head.nms_threshold.to_bits().to_le_bytes(),
            &self.head.threshold_delta.to_bits().to_le_bytes(),
        ])
    }
    fn detect_tile(&self, rgba: &[u8], keep_polygons: bool) -> Result<TileNuclei, SegError> {
        let (prob, dist, h, w) = self.infer(rgba)?;
        let t = self.head.prob_threshold;
        let d = self.head.threshold_delta;
        let thresholds = [(t - d).max(0.0), t, (t + d).min(1.0)];
        let (nuclei, _) = stardist_nms(
            &prob,
            &dist,
            h,
            w,
            &NmsParams {
                n_rays: self.head.n_rays,
                grid: self.head.grid,
                tile: self.spec.input_size,
                prob_threshold: thresholds[0],
                nms_threshold: self.head.nms_threshold,
                keep_polygons,
            },
        );
        Ok(TileNuclei { nuclei, thresholds })
    }
}

// ---------------------------------------------------------------------------
// Klasik (LoG)
// ---------------------------------------------------------------------------

/// Pendeteksi klasik: dekonvolusi pewarna → kanal H → LoG multiskala, lalu
/// tiga saringan agar tekstur stroma, sekresi, dan latar tidak ikut
/// terhitung: (1) OD hematoksilin rata-rata dalam cakram inti ≥ `min_h_od`
/// (ambang absolut, bukan Otsu per tile yang anjlok di tile kosong);
/// (2) kontras LoG ≥ `contrast_k` × derau tile (MAD respons LoG);
/// (3) bila `h_over_e`, hematoksilin ≥ kanal kedua (eosin/DAB) agar
/// sekresi dan eritrosit yang eosinofilik tidak lolos.
#[derive(Debug, Clone)]
pub struct ClassicNucleiDetector {
    /// Parameter LoG (µm). `h_threshold` di sini diabaikan; pakai
    /// `min_h_od`.
    pub params: NucleiParams,
    /// Sisi tile (piksel; bawaan 512).
    pub tile_size: usize,
    /// Ukuran piksel kerja, µm (bawaan 0,5).
    pub mpp: f64,
    /// Matriks pewarna (bawaan H&E).
    pub stain: StainMatrix,
    /// OD hematoksilin minimum rata-rata cakram inti (bawaan 0,15).
    pub min_h_od: f32,
    /// Kontras minimum respons LoG dalam satuan derau tile (bawaan 3).
    pub contrast_k: f32,
    /// Wajibkan H ≥ kanal kedua (bawaan benar).
    pub h_over_e: bool,
}

impl Default for ClassicNucleiDetector {
    fn default() -> Self {
        Self {
            params: NucleiParams::default(),
            tile_size: 512,
            mpp: 0.5,
            stain: StainMatrix::he(),
            min_h_od: 0.15,
            contrast_k: 3.0,
            h_over_e: true,
        }
    }
}

impl ClassicNucleiDetector {
    /// Validasi nilai.
    pub fn validate(&self) -> Result<(), SegError> {
        if self.tile_size < 64 || !(self.mpp.is_finite() && self.mpp > 0.0) {
            return Err(SegError::InvalidParameter(
                "tile_size ≥ 64 dan mpp > 0".to_string(),
            ));
        }
        if !(self.min_h_od.is_finite()
            && self.min_h_od >= 0.0
            && self.contrast_k.is_finite()
            && self.contrast_k >= 0.0)
        {
            return Err(SegError::InvalidParameter(
                "min_h_od/contrast_k harus terhingga dan ≥ 0".to_string(),
            ));
        }
        if self.params.log_sigmas_um.is_empty()
            || self
                .params
                .log_sigmas_um
                .iter()
                .any(|s| !(s.is_finite() && *s > 0.0))
            || !(self.params.nms_radius_um.is_finite() && self.params.nms_radius_um >= 0.0)
            || !(self.params.smooth_sigma_um.is_finite() && self.params.smooth_sigma_um >= 0.0)
        {
            return Err(SegError::InvalidParameter(
                "parameter LoG tidak sah".to_string(),
            ));
        }
        Ok(())
    }
}

impl NucleiDetector for ClassicNucleiDetector {
    fn tile_size(&self) -> usize {
        self.tile_size
    }
    fn mpp(&self) -> f64 {
        self.mpp
    }
    fn margin_px(&self) -> usize {
        let r = self
            .params
            .log_sigmas_um
            .iter()
            .copied()
            .fold(0.0, f64::max)
            .max(self.params.nms_radius_um);
        ((3.0 * r / self.mpp).ceil() as usize).clamp(4, self.tile_size / 4)
    }
    fn method_name(&self) -> String {
        "log-classic-v1".to_string()
    }
    fn params_hash(&self) -> u64 {
        let mut parts: Vec<Vec<u8>> = vec![
            b"h3d-cells-classic-v1".to_vec(),
            self.mpp.to_bits().to_le_bytes().to_vec(),
            self.params.smooth_sigma_um.to_bits().to_le_bytes().to_vec(),
            self.params.nms_radius_um.to_bits().to_le_bytes().to_vec(),
            self.min_h_od.to_bits().to_le_bytes().to_vec(),
            self.contrast_k.to_bits().to_le_bytes().to_vec(),
            vec![u8::from(self.h_over_e)],
        ];
        for s in &self.params.log_sigmas_um {
            parts.push(s.to_bits().to_le_bytes().to_vec());
        }
        let refs: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
        fnv1a(&refs)
    }
    fn detect_tile(&self, rgba: &[u8], keep_polygons: bool) -> Result<TileNuclei, SegError> {
        let s = self.tile_size;
        if rgba.len() != s * s * 4 {
            return Err(SegError::InvalidParameter(format!(
                "tile {} byte, diharapkan {}",
                rgba.len(),
                s * s * 4
            )));
        }
        let [hch, ech, _] = deconvolve(rgba, s, s, &self.stain)?;
        // Skor = kontras LoG dalam satuan derau tile; ambang longgar 0,75×,
        // utama `contrast_k`, ketat 1,5× (rentang ketidakpastian). OD H
        // minimum adalah gerbang terpisah.
        let k = self.contrast_k;
        let thresholds = [k * 0.75, k, k * 1.5];
        let loose = NucleiParams {
            h_threshold: Some(self.min_h_od),
            ..self.params.clone()
        };
        let found = detect_nuclei(&hch, self.mpp, None, &loose);
        // Derau respons LoG tile pada skala tengah: 1,4826 × MAD.
        let noise = if !found.is_empty() {
            let sig = {
                let mut v = self.params.log_sigmas_um.clone();
                v.sort_by(f64::total_cmp);
                v[v.len() / 2]
            };
            let smooth =
                h3d_imgproc::gaussian_blur_f32(&hch, self.params.smooth_sigma_um / self.mpp);
            let resp = h3d_imgproc::log_filter(&smooth, sig / self.mpp);
            let mut vals: Vec<f32> = resp.iter().copied().filter(|v| v.is_finite()).collect();
            if vals.is_empty() {
                0.0
            } else {
                let mid = vals.len() / 2;
                let med = *vals.select_nth_unstable_by(mid, f32::total_cmp).1;
                let mut dev: Vec<f32> = vals.iter().map(|v| (v - med).abs()).collect();
                let mad = *dev.select_nth_unstable_by(mid, f32::total_cmp).1;
                1.4826 * mad
            }
        } else {
            0.0
        };
        let nuclei = found
            .iter()
            .filter_map(|n| {
                let y = n.y_um / self.mpp;
                let x = n.x_um / self.mpp;
                let r_px = f64::from(n.sigma_um) / self.mpp;
                let yi = (y.floor().max(0.0) as usize).min(s - 1);
                let xi = (x.floor().max(0.0) as usize).min(s - 1);
                if self.h_over_e {
                    let e_mean = h3d_imgproc::disc_mean(&ech, xi, yi, r_px);
                    if n.h_mean < e_mean {
                        return None;
                    }
                }
                if n.h_mean < self.min_h_od {
                    return None;
                }
                // Kontras: respons LoG negatif dibagi derau tile; tanpa
                // derau terukur semua kandidat dianggap kontras tinggi.
                let score = if noise > 0.0 {
                    -n.response / noise
                } else {
                    f32::MAX
                };
                if score < thresholds[0] {
                    return None;
                }
                let poly = if keep_polygons {
                    (0..16)
                        .map(|k| {
                            let a = k as f32 / 16.0 * std::f32::consts::TAU;
                            [r_px as f32 * a.cos(), r_px as f32 * a.sin()]
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                Some((y, x, std::f64::consts::PI * r_px * r_px, score, poly))
            })
            .collect();
        Ok(TileNuclei { nuclei, thresholds })
    }
}

// ---------------------------------------------------------------------------
// Watershed (segmentasi klasik ala QuPath)
// ---------------------------------------------------------------------------

/// Pendeteksi segmentasi inti klasik tanpa model, mengikuti alur "cell
/// detection" QuPath: kanal hematoksilin (OD) → Gauss → ambang OD
/// (maks(lantai, Otsu tile)) → isi lubang + buka morfologi → **watershed
/// berpenanda di atas medan H halus + bobot × EDT** (`split_on_field`;
/// lembah intensitas maupun lekukan bentuk menjadi batas antar inti
/// berdempetan) → saring per inti (luas, sirkularitas, H ≥ E). Keluarannya poligon inti
/// (sinar dari sentroid ke tepi wilayah), bukan sekadar puncak LoG, sehingga
/// inti berdempetan di epitel terpisah.
#[derive(Debug, Clone)]
pub struct WatershedNucleiDetector {
    /// Sisi tile (piksel; bawaan 512).
    pub tile_size: usize,
    /// Ukuran piksel kerja, µm (bawaan 0,5).
    pub mpp: f64,
    /// Matriks pewarna (bawaan H&E).
    pub stain: StainMatrix,
    /// σ pemulusan Gauss kanal H, µm (bawaan 1).
    pub smooth_sigma_um: f64,
    /// Lantai ambang OD hematoksilin masker inti (bawaan 0,15); bila
    /// `adaptive_threshold`, ambang efektif per tile = maks(lantai, Otsu H
    /// di dalam jaringan tile).
    pub h_threshold: f32,
    /// Naikkan ambang dengan Otsu per tile (bawaan benar; matikan untuk IHC
    /// yang inti positifnya didominasi kromogen sehingga H bimodal palsu).
    pub adaptive_threshold: bool,
    /// Radius pembukaan morfologi, µm (bawaan 0,5).
    pub open_radius_um: f64,
    /// Jarak minimum antar puncak EDT, µm (bawaan 3).
    pub min_seed_distance_um: f64,
    /// Kedalaman lembah minimum medan pemisah (OD) agar dua puncak = dua
    /// inti (bawaan 0,03).
    pub h_od: f32,
    /// Bobot EDT dalam medan pemisah, OD per µm (bawaan 0,05): medan =
    /// H halus + bobot × jarak ke latar, sehingga inti berdempetan terpisah
    /// baik oleh lembah intensitas maupun oleh lekukan bentuk.
    pub edt_weight: f32,
    /// Luas inti minimum, µm² (bawaan 8).
    pub min_area_um2: f64,
    /// Luas inti maksimum, µm² (bawaan 300).
    pub max_area_um2: f64,
    /// Sirkularitas minimum `4πA/P²` (bawaan 0,25).
    pub min_circularity: f64,
    /// Wajibkan OD H rata-rata ≥ OD kanal kedua (bawaan benar).
    pub h_over_e: bool,
    /// Toleransi penyederhanaan kontur inti (Douglas–Peucker), µm (bawaan
    /// 0,3). Kontur mengikuti tepi wilayah watershed lalu dihaluskan
    /// Chaikin dua kali; hanya untuk tampilan, tidak memengaruhi jumlah
    /// maupun luas.
    pub contour_tolerance_um: f64,
    /// Soliditas minimum gabungan dua wilayah watershed bertetangga agar
    /// digabung menjadi satu sel (bawaan 0,93; `0` = mati). Menyatukan
    /// keping cembung satu sel yang terbelah beberapa benih; dua inti bulat
    /// yang sekadar bersentuhan (soliditas ≈ 0,88) tetap terpisah.
    pub merge_solidity: f64,
}

impl Default for WatershedNucleiDetector {
    fn default() -> Self {
        Self {
            tile_size: 512,
            mpp: 0.5,
            stain: StainMatrix::he(),
            smooth_sigma_um: 1.0,
            h_threshold: 0.15,
            adaptive_threshold: true,
            open_radius_um: 0.5,
            min_seed_distance_um: 3.0,
            h_od: 0.03,
            edt_weight: 0.05,
            min_area_um2: 8.0,
            max_area_um2: 300.0,
            min_circularity: 0.25,
            h_over_e: true,
            contour_tolerance_um: 0.3,
            merge_solidity: 0.93,
        }
    }
}

impl WatershedNucleiDetector {
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
            && pos(self.min_seed_distance_um)
            && self.h_od.is_finite()
            && self.h_od >= 0.0
            && self.edt_weight.is_finite()
            && self.edt_weight >= 0.0
            && pos(self.min_area_um2)
            && self.max_area_um2.is_finite()
            && self.max_area_um2 > self.min_area_um2
            && self.min_circularity.is_finite()
            && (0.0..=1.0).contains(&self.min_circularity)
            && pos(self.contour_tolerance_um)
            && self.merge_solidity.is_finite()
            && (0.0..=1.0).contains(&self.merge_solidity);
        if !ok {
            return Err(SegError::InvalidParameter(
                "parameter watershed inti tidak sah".to_string(),
            ));
        }
        Ok(())
    }
}

impl NucleiDetector for WatershedNucleiDetector {
    fn tile_size(&self) -> usize {
        self.tile_size
    }
    fn mpp(&self) -> f64 {
        self.mpp
    }
    fn margin_px(&self) -> usize {
        // Diameter inti terbesar + pemulusan, agar inti di daerah milik
        // tile utuh di dalam tile.
        let d_um =
            2.0 * (self.max_area_um2 / std::f64::consts::PI).sqrt() + 3.0 * self.smooth_sigma_um;
        ((d_um / self.mpp).ceil() as usize).clamp(4, self.tile_size / 4)
    }
    fn method_name(&self) -> String {
        "watershed-classic-v1".to_string()
    }
    fn params_hash(&self) -> u64 {
        let f = |v: f64| v.to_bits().to_le_bytes().to_vec();
        let parts: Vec<Vec<u8>> = vec![
            b"h3d-cells-watershed-v3".to_vec(),
            f(self.mpp),
            f(self.smooth_sigma_um),
            self.h_threshold.to_bits().to_le_bytes().to_vec(),
            vec![u8::from(self.adaptive_threshold)],
            f(self.open_radius_um),
            f(self.min_seed_distance_um),
            self.h_od.to_bits().to_le_bytes().to_vec(),
            self.edt_weight.to_bits().to_le_bytes().to_vec(),
            f(self.min_area_um2),
            f(self.max_area_um2),
            f(self.min_circularity),
            vec![u8::from(self.h_over_e)],
            f(self.contour_tolerance_um),
            f(self.merge_solidity),
        ];
        let refs: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
        fnv1a(&refs)
    }
    fn detect_tile(&self, rgba: &[u8], keep_polygons: bool) -> Result<TileNuclei, SegError> {
        let s = self.tile_size;
        if rgba.len() != s * s * 4 {
            return Err(SegError::InvalidParameter(format!(
                "tile {} byte, diharapkan {}",
                rgba.len(),
                s * s * 4
            )));
        }
        let [hch, ech, _] = deconvolve(rgba, s, s, &self.stain)?;
        let smooth = if self.smooth_sigma_um > 0.0 {
            h3d_imgproc::gaussian_blur_f32(&hch, self.smooth_sigma_um / self.mpp)
        } else {
            hch.clone()
        };
        // Ambang: Otsu atas piksel jaringan tile (OD total > 0,1), tidak
        // lebih rendah dari lantai; tile kosong jatuh ke lantai.
        let tissue = h3d_imgproc::od_total(rgba, s, s)?.mapv(|v| u8::from(v > 0.1));
        let otsu = if self.adaptive_threshold && tissue.iter().filter(|&&v| v != 0).count() > 64 {
            h3d_imgproc::otsu_masked(&smooth, &tissue)
        } else {
            0.0
        };
        let thr = otsu.max(self.h_threshold);
        let mut mask = smooth.mapv(|v| u8::from(v > thr));
        mask = h3d_imgproc::fill_holes(&mask);
        if self.open_radius_um > 0.0 {
            mask = h3d_imgproc::open(&mask, self.open_radius_um / self.mpp);
        }
        let min_px = (self.min_area_um2 / (self.mpp * self.mpp)).ceil() as usize;
        if min_px > 1 {
            mask = h3d_imgproc::remove_small_components(&mask, min_px);
        }
        // Medan pemisah: H halus + bobot × EDT (µm) ke latar masker.
        let field = if self.edt_weight > 0.0 {
            let inv = mask.mapv(|v| u8::from(v == 0));
            let edt = h3d_imgproc::edt_2d_um(&inv, self.mpp);
            let mut f = smooth.clone();
            for (v, d) in f.iter_mut().zip(edt.iter()) {
                *v += self.edt_weight * d;
            }
            f
        } else {
            smooth.clone()
        };
        let (mut labels, n_raw) = h3d_imgproc::split_on_field(
            &mask,
            &field,
            self.mpp,
            self.min_seed_distance_um,
            self.h_od,
        );
        let mut n_obj = if min_px > 2 {
            h3d_imgproc::merge_small_regions(&mut labels, n_raw, min_px / 2)
        } else {
            n_raw
        };
        // Satukan keping cembung satu sel (over-segmentasi watershed).
        if self.merge_solidity > 0.0 {
            let max_px = (self.max_area_um2 / (self.mpp * self.mpp)).floor() as usize;
            n_obj =
                h3d_imgproc::merge_convex_regions(&mut labels, n_obj, max_px, self.merge_solidity);
        }
        // Statistik per objek: luas, sentroid, bbox, keliling retakan, OD.
        let mut area = vec![0usize; n_obj + 1];
        let mut sx = vec![0f64; n_obj + 1];
        let mut sy = vec![0f64; n_obj + 1];
        let mut crack = vec![0usize; n_obj + 1];
        let mut bb = vec![[usize::MAX, usize::MAX, 0usize, 0usize]; n_obj + 1];
        let mut sum_h = vec![0.0f64; n_obj + 1];
        let mut sum_e = vec![0.0f64; n_obj + 1];
        for ((y, x), &l) in labels.indexed_iter() {
            if l == 0 {
                continue;
            }
            let i = l as usize;
            area[i] += 1;
            sx[i] += x as f64;
            sy[i] += y as f64;
            sum_h[i] += f64::from(hch[[y, x]]);
            sum_e[i] += f64::from(ech[[y, x]]);
            let b = &mut bb[i];
            b[0] = b[0].min(x);
            b[1] = b[1].min(y);
            b[2] = b[2].max(x + 1);
            b[3] = b[3].max(y + 1);
            let mut edges = 0;
            if x == 0 || labels[[y, x - 1]] != l {
                edges += 1;
            }
            if x + 1 == s || labels[[y, x + 1]] != l {
                edges += 1;
            }
            if y == 0 || labels[[y - 1, x]] != l {
                edges += 1;
            }
            if y + 1 == s || labels[[y + 1, x]] != l {
                edges += 1;
            }
            crack[i] += edges;
        }
        let mpp2 = self.mpp * self.mpp;
        let objects: Vec<h3d_imgproc::Object2D> = (1..=n_obj)
            .filter(|&i| area[i] > 0)
            .map(|i| {
                let a_um2 = area[i] as f64 * mpp2;
                let perim = crack[i] as f64 * self.mpp * std::f64::consts::FRAC_PI_4;
                let a = area[i] as f64;
                h3d_imgproc::Object2D {
                    id: i as u32,
                    area_um2: a_um2,
                    perimeter_um: perim,
                    equiv_diameter_um: 2.0 * (a_um2 / std::f64::consts::PI).sqrt(),
                    circularity: if perim > 0.0 {
                        (4.0 * std::f64::consts::PI * a_um2 / (perim * perim)).min(1.0)
                    } else {
                        0.0
                    },
                    centroid_um: [(sx[i] / a + 0.5) * self.mpp, (sy[i] / a + 0.5) * self.mpp],
                    bbox_um: [
                        bb[i][0] as f64 * self.mpp,
                        bb[i][1] as f64 * self.mpp,
                        bb[i][2] as f64 * self.mpp,
                        bb[i][3] as f64 * self.mpp,
                    ],
                    touches_border: false,
                    excluded: false,
                }
            })
            .collect();
        let cnt = area;
        let t = thr;
        // Skor = OD H rata-rata inti; ambang longgar = ambang masker, utama
        // 1,2×, ketat 1,5× (inti sejati jauh di atas ambang masker).
        let thresholds = [t, t * 1.2, t * 1.5];
        let contours = if keep_polygons {
            h3d_imgproc::label_contours(&labels, n_obj)
        } else {
            Vec::new()
        };
        let mut nuclei = Vec::with_capacity(n_obj);
        let (mut rej_area, mut rej_circ, mut rej_he) = (0usize, 0usize, 0usize);
        for o in &objects {
            let i = o.id as usize;
            if cnt[i] == 0 || o.area_um2 > self.max_area_um2 {
                rej_area += 1;
                continue;
            }
            if o.circularity < self.min_circularity {
                rej_circ += 1;
                continue;
            }
            let h_mean = sum_h[i] / cnt[i] as f64;
            let e_mean = sum_e[i] / cnt[i] as f64;
            if self.h_over_e && h_mean < e_mean {
                rej_he += 1;
                continue;
            }
            let cx = o.centroid_um[0] / self.mpp;
            let cy = o.centroid_um[1] / self.mpp;
            let poly = match contours.get(i) {
                // Kontur tepi wilayah (koordinat sudut piksel) relatif
                // sentroid (pusat piksel +0,5, sudah dalam `cx`/`cy`).
                Some(c) if keep_polygons && c.len() >= 3 => {
                    let eps = (self.contour_tolerance_um / self.mpp) as f32;
                    h3d_imgproc::chaikin_closed(&h3d_imgproc::simplify_closed(c, eps), 1)
                        .into_iter()
                        .map(|v| [v[0] - cx as f32, v[1] - cy as f32])
                        .collect()
                }
                _ => Vec::new(),
            };
            nuclei.push((
                cy,
                cx,
                o.area_um2 / (self.mpp * self.mpp),
                h_mean as f32,
                poly,
            ));
        }
        log::debug!(
            "watershed tile: masker {} px, {} objek, ditolak luas {rej_area} / sirkularitas {rej_circ} / H<E {rej_he}, lolos {}",
            mask.iter().filter(|&&v| v != 0).count(),
            n_obj,
            nuclei.len()
        );
        Ok(TileNuclei { nuclei, thresholds })
    }
}

// ---------------------------------------------------------------------------
// Penggerak tile
// ---------------------------------------------------------------------------

/// Hasil deteksi sel satu wilayah.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CellDetection {
    /// Inti dengan `score ≥` ambang utama, urut tile lalu raster.
    pub nuclei: Vec<DetectedNucleus>,
    /// Jumlah inti pada ambang utama.
    pub n: usize,
    /// Jumlah pada ambang ketat.
    pub n_min: usize,
    /// Jumlah pada ambang longgar.
    pub n_max: usize,
    /// Wilayah yang dideteksi `[x0, y0, x1, y1]`, µm.
    pub roi_um: [f64; 4],
    /// Ukuran piksel kerja, µm.
    pub mpp: f64,
    /// Nama metode.
    pub method: String,
    /// Hash parameter.
    pub params_hash: u64,
    /// Tile yang dijalankan.
    pub tiles_run: usize,
    /// Tile dilewati (tanpa jaringan).
    pub tiles_skipped: usize,
}

/// Posisi awal tile sepanjang satu sumbu (seperti `ai::tile_starts`).
pub(crate) fn starts(extent: usize, s: usize, stride: usize) -> Vec<usize> {
    if extent <= s {
        return vec![0];
    }
    let mut v = Vec::new();
    let mut x = 0usize;
    loop {
        if x + s >= extent {
            v.push(extent - s);
            break;
        }
        v.push(x);
        x += stride;
    }
    v.dedup();
    v
}

/// Batas kepemilikan tile `[lo, hi)` sepanjang satu sumbu: titik tengah
/// daerah tumpang tindih dengan tetangga; tile pertama mulai 0, tile
/// terakhir berakhir di `extent`.
pub(crate) fn ownership(starts: &[usize], s: usize, extent: usize) -> Vec<(usize, usize)> {
    let n = starts.len();
    (0..n)
        .map(|i| {
            let lo = if i == 0 {
                0
            } else {
                (starts[i - 1] + s + starts[i]).div_ceil(2)
            };
            let hi = if i + 1 == n {
                extent.min(starts[i] + s)
            } else {
                (starts[i] + s + starts[i + 1]).div_ceil(2)
            };
            (lo, hi)
        })
        .collect()
}

type TileResult = Result<(Vec<DetectedNucleus>, usize, usize), SegError>;

/// Deteksi sel pada wilayah `roi_um = [x0, y0, x1, y1]` (µm, ruang pembaca)
/// dengan `det` per tile. `read_tile(x_um, y_um, w, h)` membaca RGBA pada
/// `det.mpp()`. `tissue_mask` `(masker, mpp)`: tile tanpa jaringan dilewati.
/// Tile diproses paralel; hasil digabung dalam urutan tile.
pub fn detect_cells(
    det: &dyn NucleiDetector,
    read_tile: &TileReader<'_>,
    roi_um: [f64; 4],
    tissue_mask: Option<(&Array2<u8>, f64)>,
    keep_polygons: bool,
    progress: &dyn Progress,
) -> Result<CellDetection, SegError> {
    let mpp = det.mpp();
    let s = det.tile_size();
    let m = det.margin_px();
    if !(mpp.is_finite() && mpp > 0.0) || s == 0 || 2 * m >= s {
        return Err(SegError::InvalidParameter(
            "pendeteksi: mpp/tile/margin tidak sah".to_string(),
        ));
    }
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
    let stride = s - 2 * m;
    let xs = starts(w_px, s, stride);
    let ys = starts(h_px, s, stride);
    let own_x = ownership(&xs, s, w_px);
    let own_y = ownership(&ys, s, h_px);
    let tissue_has = |x0: usize, y0: usize| -> bool {
        let Some((tm, tm_mpp)) = tissue_mask else {
            return true;
        };
        let (th, tw) = tm.dim();
        let ux0 = roi_um[0] + x0 as f64 * mpp;
        let uy0 = roi_um[1] + y0 as f64 * mpp;
        let ux1 = ux0 + s as f64 * mpp;
        let uy1 = uy0 + s as f64 * mpp;
        let tx0 = ((ux0 / tm_mpp).floor().max(0.0) as usize).min(tw);
        let ty0 = ((uy0 / tm_mpp).floor().max(0.0) as usize).min(th);
        let tx1 = ((ux1 / tm_mpp).ceil().max(0.0) as usize).min(tw);
        let ty1 = ((uy1 / tm_mpp).ceil().max(0.0) as usize).min(th);
        (ty0..ty1).any(|y| (tx0..tx1).any(|x| tm[[y, x]] != 0))
    };
    // Daftar tile (urutan baris lalu kolom).
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
    let total = todo.len().max(1);
    let done = AtomicUsize::new(0);
    let run_one = |k: usize| -> TileResult {
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
        let t = det.detect_tile(&rgba, keep_polygons)?;
        let (ox0, ox1) = own_x[i];
        let (oy0, oy1) = own_y[j];
        let mut list = Vec::new();
        let (mut n_min, mut n_max) = (0usize, 0usize);
        for (y, x, area_px, score, poly) in t.nuclei {
            let gx = x0 as f64 + x;
            let gy = y0 as f64 + y;
            if gx < ox0 as f64 || gx >= ox1 as f64 || gy < oy0 as f64 || gy >= oy1 as f64 {
                continue;
            }
            // Pusat inti harus di dalam masker jaringan (bila ada).
            if let Some((tm, tm_mpp)) = tissue_mask {
                let (th, tw) = tm.dim();
                let tx = ((roi_um[0] + gx * mpp) / tm_mpp).floor();
                let ty = ((roi_um[1] + gy * mpp) / tm_mpp).floor();
                if tx < 0.0 || ty < 0.0 || tx as usize >= tw || ty as usize >= th {
                    continue;
                }
                if tm[[ty as usize, tx as usize]] == 0 {
                    continue;
                }
            }
            if score >= t.thresholds[0] {
                n_max += 1;
            }
            if score >= t.thresholds[2] {
                n_min += 1;
            }
            if score < t.thresholds[1] {
                continue;
            }
            list.push(DetectedNucleus {
                center_um: [roi_um[0] + gx * mpp, roi_um[1] + gy * mpp],
                area_um2: area_px * mpp * mpp,
                score,
                polygon_um: poly
                    .iter()
                    .map(|v| [v[0] * mpp as f32, v[1] * mpp as f32])
                    .collect(),
                holes_um: Vec::new(),
            });
        }
        let d = done.fetch_add(1, Ordering::SeqCst) + 1;
        progress.set(d as f32 / total as f32, "Deteksi sel");
        Ok((list, n_min, n_max))
    };
    // Paralel dengan penggabungan terurut (BTreeMap menurut indeks tile).
    let results: BTreeMap<usize, TileResult> = {
        use rayon::prelude::*;
        (0..todo.len())
            .into_par_iter()
            .map(|k| (k, run_one(k)))
            .collect()
    };
    let mut nuclei = Vec::new();
    let (mut n_min, mut n_max) = (0usize, 0usize);
    for (_, r) in results {
        let (list, a, b) = r?;
        nuclei.extend(list);
        n_min += a;
        n_max += b;
    }
    progress.set(1.0, "Deteksi sel");
    let n = nuclei.len();
    Ok(CellDetection {
        nuclei,
        n,
        n_min: n_min.min(n),
        n_max: n_max.max(n),
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
    use crate::ai::ClassSpec;
    use h3d_core::NoProgress;
    use h3d_imgproc::phantom::{ellipsoid_mask, match_points, synthetic_ihc};
    use prost::Message;
    use tract_onnx::pb;

    fn dim(n: i64) -> pb::tensor_shape_proto::Dimension {
        pb::tensor_shape_proto::Dimension {
            denotation: String::new(),
            value: Some(pb::tensor_shape_proto::dimension::Value::DimValue(n)),
        }
    }

    fn value_info(name: &str, dims: &[i64]) -> pb::ValueInfoProto {
        pb::ValueInfoProto {
            name: name.to_string(),
            r#type: Some(pb::TypeProto {
                denotation: String::new(),
                value: Some(pb::type_proto::Value::TensorType(pb::type_proto::Tensor {
                    elem_type: pb::tensor_proto::DataType::Float as i32,
                    shape: Some(pb::TensorShapeProto {
                        dim: dims.iter().map(|&d| dim(d)).collect(),
                    }),
                })),
            }),
            doc_string: String::new(),
        }
    }

    fn float_tensor(name: &str, dims: &[i64], data: &[f32]) -> pb::TensorProto {
        pb::TensorProto {
            dims: dims.to_vec(),
            data_type: pb::tensor_proto::DataType::Float as i32,
            float_data: data.to_vec(),
            name: name.to_string(),
            ..Default::default()
        }
    }

    fn conv(name: &str, x: &str, w: &str, b: &str, y: &str) -> pb::NodeProto {
        pb::NodeProto {
            input: vec![x.into(), w.into(), b.into()],
            output: vec![y.into()],
            name: name.into(),
            op_type: "Conv".into(),
            domain: String::new(),
            attribute: vec![pb::AttributeProto {
                name: "kernel_shape".into(),
                r#type: pb::attribute_proto::AttributeType::Ints as i32,
                ints: vec![1, 1],
                ..Default::default()
            }],
            doc_string: String::new(),
        }
    }

    /// Model StarDist mungil (grid 1): prob = kanal R masukan; jarak = `r`
    /// konstan untuk semua sinar. Keluaran NCHW.
    fn tiny_stardist(s: i64, n_rays: usize, r: f32) -> Vec<u8> {
        let graph = pb::GraphProto {
            node: vec![
                conv("prob", "x", "wp", "bp", "prob"),
                conv("dist", "x", "wd", "bd", "dist"),
            ],
            name: "tiny-stardist".into(),
            initializer: vec![
                float_tensor("wp", &[1, 3, 1, 1], &[1.0, 0.0, 0.0]),
                float_tensor("bp", &[1], &[0.0]),
                float_tensor("wd", &[n_rays as i64, 3, 1, 1], &vec![0.0; 3 * n_rays]),
                float_tensor("bd", &[n_rays as i64], &vec![r; n_rays]),
            ],
            input: vec![value_info("x", &[1, 3, s, s])],
            output: vec![
                value_info("prob", &[1, 1, s, s]),
                value_info("dist", &[1, n_rays as i64, s, s]),
            ],
            ..Default::default()
        };
        pb::ModelProto {
            ir_version: 8,
            producer_name: "h3d-segment-test".into(),
            graph: Some(graph),
            opset_import: vec![pb::OperatorSetIdProto {
                domain: String::new(),
                version: 13,
            }],
            ..Default::default()
        }
        .encode_to_vec()
    }

    fn nuclei_spec(s: usize) -> ModelSpec {
        ModelSpec {
            name: "sd-uji".to_string(),
            version: Some("0".to_string()),
            input_size: s,
            mpp: 0.5,
            mean: [0.0; 3],
            std: [1.0; 3],
            classes: Vec::<ClassSpec>::new(),
            task: "nuclei".to_string(),
            nuclei: Some(NucleiHead {
                n_rays: 8,
                grid: [1, 1],
                prob_threshold: 0.5,
                nms_threshold: 0.3,
                threshold_delta: 0.1,
                margin_px: 4,
                ..NucleiHead::default()
            }),
        }
    }

    #[test]
    fn polygon_fill_and_area() {
        // Persegi 4×4 dari (2,2) sampai (6,6).
        let sq = [[2.0f32, 2.0], [6.0, 2.0], [6.0, 6.0], [2.0, 6.0]];
        let mut n = 0;
        fill_polygon(&sq, 10, 10, |_, _| n += 1);
        assert_eq!(n, 16);
        assert!((polygon_area(&sq) - 16.0).abs() < 1e-6);
    }

    #[test]
    fn nms_separates_two_blobs_and_merges_duplicates() {
        let (h, w, r) = (20usize, 20usize, 8usize);
        let mut prob = vec![0.0f32; h * w];
        let dist = vec![3.0f32; h * w * r];
        // Dua puncak jauh + satu puncak ganda di sebelah puncak pertama.
        prob[5 * w + 5] = 0.9;
        prob[5 * w + 6] = 0.8; // tumpang tindih besar → dibuang
        prob[14 * w + 14] = 0.7;
        prob[2 * w + 17] = 0.3; // di bawah ambang
        let p = NmsParams {
            n_rays: r,
            grid: [1, 1],
            tile: 20,
            prob_threshold: 0.5,
            nms_threshold: 0.3,
            keep_polygons: true,
        };
        let (n, labels) = stardist_nms(&prob, &dist, h, w, &p);
        assert_eq!(n.len(), 2, "{n:?}");
        assert!((n[0].1 - 5.5).abs() < 1e-5 && (n[0].0 - 5.5).abs() < 1e-5);
        assert_eq!(n[0].4.len(), r);
        assert!(
            n[0].2 > 20.0 && n[0].2 < 29.0,
            "luas oktagon r=3: {}",
            n[0].2
        );
        assert_eq!(labels[[5, 5]], 1);
        assert_eq!(labels[[14, 14]], 2);
        assert_eq!(labels[[0, 0]], 0);
    }

    #[test]
    fn stardist_tract_end_to_end() {
        let s = 32usize;
        let det =
            StarDistDetector::from_bytes(nuclei_spec(s), &tiny_stardist(s as i64, 8, 2.5)).unwrap();
        assert_eq!(det.margin_px(), 4);
        // Tile: latar hitam, "inti" = piksel merah dengan terang berbeda.
        let mut rgba = vec![0u8; s * s * 4];
        for &(x, y, v) in &[(8usize, 8usize, 250u8), (20, 20, 200), (26, 8, 100)] {
            rgba[(y * s + x) * 4] = v;
            rgba[(y * s + x) * 4 + 3] = 255;
        }
        let t = det.detect_tile(&rgba, true).unwrap();
        // prob = R/255: 0,98, 0,78, 0,39 → ambang longgar 0,4 menyaring yang ketiga.
        assert_eq!(t.nuclei.len(), 2, "{:?}", t.nuclei);
        assert_eq!(t.thresholds, [0.4, 0.5, 0.6]);
        // Lewat penggerak tile: wilayah 2×2 tile dengan tumpang tindih.
        let reader = |x_um: f64, y_um: f64, w: usize, h: usize| {
            let mut v = vec![0u8; w * h * 4];
            // Satu inti di piksel global (24, 24) pada mpp 0,5.
            let px = 24.0 - x_um / 0.5;
            let py = 24.0 - y_um / 0.5;
            if px >= 0.0 && py >= 0.0 && (px as usize) < w && (py as usize) < h {
                let i = (py as usize * w + px as usize) * 4;
                v[i] = 255;
                v[i + 3] = 255;
            }
            Ok(v)
        };
        let r = detect_cells(
            &det,
            &reader,
            [0.0, 0.0, 28.0, 28.0],
            None,
            false,
            &NoProgress,
        )
        .unwrap();
        assert!(r.tiles_run >= 4, "{}", r.tiles_run);
        assert_eq!(r.n, 1, "{:?}", r.nuclei);
        assert_eq!((r.n_min, r.n_max), (1, 1));
        let c = r.nuclei[0].center_um;
        assert!(
            (c[0] - 12.25).abs() < 1e-6 && (c[1] - 12.25).abs() < 1e-6,
            "{c:?}"
        );
        assert!(r.nuclei[0].polygon_um.is_empty());
        assert_eq!(r.method, "stardist sd-uji 0");
    }

    #[test]
    fn stardist_rejects_wrong_spec() {
        let s = 16usize;
        let mut sp = nuclei_spec(s);
        sp.task = "segmentation".to_string();
        assert!(StarDistDetector::from_bytes(sp, &tiny_stardist(s as i64, 8, 1.0)).is_err());
        let mut sp = nuclei_spec(s);
        sp.nuclei.as_mut().unwrap().n_rays = 16; // model hanya 8 kanal
        assert!(StarDistDetector::from_bytes(sp, &tiny_stardist(s as i64, 8, 1.0)).is_err());
        let mut sp = nuclei_spec(s);
        sp.nuclei.as_mut().unwrap().grid = [3, 3];
        assert!(sp.validate().is_err());
        assert!(StarDistDetector::from_bytes(nuclei_spec(s), b"bukan onnx").is_err());
        assert!(StarDistDetector::load(Path::new("/tidak/ada")).is_err());
    }

    #[test]
    fn ownership_partitions_axis() {
        let s = 32;
        let xs = starts(100, s, 24);
        let own = ownership(&xs, s, 100);
        assert_eq!(own[0].0, 0);
        assert_eq!(own.last().unwrap().1, 100);
        for w in own.windows(2) {
            assert_eq!(w[0].1, w[1].0);
        }
        assert_eq!(ownership(&starts(20, s, 24), s, 20), vec![(0, 20)]);
    }

    #[test]
    fn classic_detector_on_ihc_phantom_matches_truth() {
        let mpp = 0.5;
        let (w, h) = (600usize, 450usize);
        let mask = ellipsoid_mask([w, h], [300.0, 225.0], [260.0, 190.0]);
        let (rgba, truth) = synthetic_ihc(&mask, 3, 0.3, mpp);
        // IHC: inti positif didominasi DAB, jadi syarat H ≥ E dimatikan.
        let det = ClassicNucleiDetector {
            tile_size: 256,
            mpp,
            stain: StainMatrix::h_dab(),
            h_over_e: false,
            ..Default::default()
        };
        det.validate().unwrap();
        let reader = |x_um: f64, y_um: f64, tw: usize, th: usize| {
            let x0 = (x_um / mpp).round() as i64;
            let y0 = (y_um / mpp).round() as i64;
            let mut v = vec![255u8; tw * th * 4];
            for y in 0..th {
                for x in 0..tw {
                    let (gx, gy) = (x0 + x as i64, y0 + y as i64);
                    if gx >= 0 && gy >= 0 && (gx as usize) < w && (gy as usize) < h {
                        let src = ((gy as usize) * w + gx as usize) * 4;
                        let dst = (y * tw + x) * 4;
                        v[dst..dst + 4].copy_from_slice(&rgba[src..src + 4]);
                    }
                }
            }
            Ok(v)
        };
        let roi = [0.0, 0.0, w as f64 * mpp, h as f64 * mpp];
        let r = detect_cells(&det, &reader, roi, None, true, &NoProgress).unwrap();
        assert!(r.tiles_run > 1);
        let det_pts: Vec<(f64, f64)> = r
            .nuclei
            .iter()
            .map(|n| (n.center_um[0], n.center_um[1]))
            .collect();
        let tru: Vec<(f64, f64)> = truth.iter().map(|t| (t.0, t.1)).collect();
        let (tp, fp, fn_) = match_points(&det_pts, &tru, 4.0);
        let f1 = 2.0 * tp as f64 / (2 * tp + fp + fn_).max(1) as f64;
        assert!(f1 >= 0.85, "F1 {f1:.3} (tp {tp} fp {fp} fn {fn_})");
        assert!(r.n_min <= r.n && r.n <= r.n_max);
        assert!(!r.nuclei[0].polygon_um.is_empty());
        assert!(r.nuclei[0].radius_um() > 1.0);
        // Deterministik.
        let r2 = detect_cells(&det, &reader, roi, None, true, &NoProgress).unwrap();
        assert_eq!(r, r2);
    }

    #[test]
    fn watershed_detector_on_ihc_phantom_and_touching_nuclei() {
        let mpp = 0.5;
        let (w, h) = (600usize, 450usize);
        let mask = ellipsoid_mask([w, h], [300.0, 225.0], [260.0, 190.0]);
        let (rgba, truth) = synthetic_ihc(&mask, 3, 0.3, mpp);
        let det = WatershedNucleiDetector {
            tile_size: 256,
            mpp,
            stain: StainMatrix::h_dab(),
            h_over_e: false,
            adaptive_threshold: false,
            ..Default::default()
        };
        det.validate().unwrap();
        let reader = |x_um: f64, y_um: f64, tw: usize, th: usize| {
            let x0 = (x_um / mpp).round() as i64;
            let y0 = (y_um / mpp).round() as i64;
            let mut v = vec![255u8; tw * th * 4];
            for y in 0..th {
                for x in 0..tw {
                    let (gx, gy) = (x0 + x as i64, y0 + y as i64);
                    if gx >= 0 && gy >= 0 && (gx as usize) < w && (gy as usize) < h {
                        let src = ((gy as usize) * w + gx as usize) * 4;
                        let dst = (y * tw + x) * 4;
                        v[dst..dst + 4].copy_from_slice(&rgba[src..src + 4]);
                    }
                }
            }
            Ok(v)
        };
        let roi = [0.0, 0.0, w as f64 * mpp, h as f64 * mpp];
        let r = detect_cells(&det, &reader, roi, None, true, &NoProgress).unwrap();
        let det_pts: Vec<(f64, f64)> = r
            .nuclei
            .iter()
            .map(|n| (n.center_um[0], n.center_um[1]))
            .collect();
        let tru: Vec<(f64, f64)> = truth.iter().map(|t| (t.0, t.1)).collect();
        let (tp, fp, fn_) = match_points(&det_pts, &tru, 4.0);
        let f1 = 2.0 * tp as f64 / (2 * tp + fp + fn_).max(1) as f64;
        assert!(f1 >= 0.85, "F1 {f1:.3} (tp {tp} fp {fp} fn {fn_})");
        // Kontur mengikuti tepi wilayah: luas poligon ≈ luas inti piksel.
        for n in &r.nuclei {
            assert!(n.area_um2 >= 8.0 && n.polygon_um.len() >= 12);
            let a = h3d_imgproc::signed_area(&n.polygon_um);
            assert!(
                (a - n.area_um2).abs() / n.area_um2 < 0.25,
                "luas poligon {a:.1} vs inti {:.1}",
                n.area_um2
            );
        }
        assert_eq!(r.method, "watershed-classic-v1");
        assert!(r.n_min <= r.n && r.n <= r.n_max);

        // Dua inti berdempetan (cakram H tinggi) pada latar putih → 2.
        let s = 64usize;
        let mut tile = vec![255u8; s * s * 4];
        for (cx, cy) in [(24.0f64, 32.0f64), (38.0, 32.0)] {
            for y in 0..s {
                for x in 0..s {
                    let (dx, dy) = (x as f64 + 0.5 - cx, y as f64 + 0.5 - cy);
                    if dx * dx + dy * dy <= 8.0 * 8.0 {
                        let i = (y * s + x) * 4;
                        tile[i..i + 3].copy_from_slice(&[60, 40, 120]);
                    }
                }
            }
        }
        let det2 = WatershedNucleiDetector {
            tile_size: s,
            mpp,
            ..Default::default()
        };
        let t = det2.detect_tile(&tile, true).unwrap();
        assert_eq!(t.nuclei.len(), 2, "{:?}", t.nuclei);
        let bad = WatershedNucleiDetector {
            max_area_um2: 1.0,
            ..Default::default()
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn detect_cells_validates_and_cancels() {
        let det = ClassicNucleiDetector::default();
        let ok = |_x: f64, _y: f64, w: usize, h: usize| Ok(vec![255u8; w * h * 4]);
        assert!(detect_cells(&det, &ok, [0.0, 0.0, 0.0, 10.0], None, false, &NoProgress).is_err());
        struct Cancel;
        impl Progress for Cancel {
            fn set(&self, _: f32, _: &str) {}
            fn is_cancelled(&self) -> bool {
                true
            }
        }
        assert!(matches!(
            detect_cells(&det, &ok, [0.0, 0.0, 100.0, 100.0], None, false, &Cancel),
            Err(SegError::Cancelled)
        ));
        let bad = ClassicNucleiDetector {
            mpp: 0.0,
            ..Default::default()
        };
        assert!(bad.validate().is_err());
    }
}
