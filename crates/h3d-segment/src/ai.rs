//! Segmentasi AI (histo3d.md 8.5.4): `ModelSpec` dari `model.json`, trait
//! [`Segmenter`], backend [`TractSegmenter`] (Rust murni, CPU), dan alur
//! inferensi seluruh slide [`infer_slide`] dengan tile bertumpang tindih dan
//! penggabungan jendela Hann.
//!
//! # Memori
//! Akumulator logit hanya setinggi satu baris tile (`S` piksel) selebar slide
//! pada mpp model: `(C + 1) × S × W_px × 4` byte. Baris piksel yang sudah tidak
//! tersentuh tile berikutnya difinalkan (argmax + softmax) dan langsung
//! diturunkan ke `mpp_label` dengan modus, sehingga peta kelas resolusi model
//! tidak pernah disimpan utuh. Keluaran tile dalam satu baris dihitung paralel
//! dalam kelompok sebesar jumlah thread, lalu dijumlahkan dalam urutan tetap
//! (deterministik).

use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;

use h3d_core::{AnnotationSource, LabelId, Progress};
use h3d_imgproc::filters::hann_window_2d;
use ndarray::Array2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tract_onnx::prelude::*;
use tract_onnx::tract_hir::internal::DimLike;

use crate::error::SegError;
use crate::proposal::Proposal;

/// Satu kelas keluaran model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClassSpec {
    /// Indeks kanal logit.
    pub index: usize,
    /// Nama label; `None` = latar.
    #[serde(default)]
    pub label: Option<String>,
}

fn default_task() -> String {
    "segmentation".to_string()
}

/// Spesifikasi model dari `model.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelSpec {
    /// Nama model.
    pub name: String,
    /// Versi model.
    #[serde(default)]
    pub version: Option<String>,
    /// Sisi tile masukan `S` (piksel).
    pub input_size: usize,
    /// Ukuran piksel masukan (µm).
    pub mpp: f64,
    /// Rata-rata normalisasi per kanal RGB.
    pub mean: [f32; 3],
    /// Simpangan baku normalisasi per kanal RGB.
    pub std: [f32; 3],
    /// Kelas keluaran.
    pub classes: Vec<ClassSpec>,
    /// Jenis tugas; bawaan `"segmentation"`.
    #[serde(default = "default_task")]
    pub task: String,
}

impl ModelSpec {
    /// Urai dan validasi `model.json`.
    pub fn from_json(text: &str) -> Result<Self, SegError> {
        let spec: ModelSpec = serde_json::from_str(text)
            .map_err(|e| SegError::Model(format!("model.json tidak sah: {e}")))?;
        spec.validate()?;
        Ok(spec)
    }

    /// Validasi nilai.
    pub fn validate(&self) -> Result<(), SegError> {
        if self.input_size == 0 || self.input_size > 8192 {
            return Err(SegError::Model(format!(
                "input_size {} di luar rentang",
                self.input_size
            )));
        }
        if !(self.mpp.is_finite() && self.mpp > 0.0) {
            return Err(SegError::Model(format!(
                "mpp model tidak sah: {}",
                self.mpp
            )));
        }
        if self.std.iter().any(|s| !(s.is_finite() && *s > 0.0))
            || self.mean.iter().any(|m| !m.is_finite())
        {
            return Err(SegError::Model("mean/std tidak sah".to_string()));
        }
        if self.classes.is_empty() {
            return Err(SegError::Model("daftar kelas kosong".to_string()));
        }
        let mut idx: Vec<usize> = self.classes.iter().map(|c| c.index).collect();
        idx.sort_unstable();
        idx.dedup();
        if idx.len() != self.classes.len() {
            return Err(SegError::Model("indeks kelas ganda".to_string()));
        }
        if self.task != "segmentation" {
            return Err(SegError::Model(format!(
                "tugas '{}' tidak didukung (hanya segmentation)",
                self.task
            )));
        }
        Ok(())
    }

    /// Jumlah kanal logit minimum yang diharapkan (indeks maksimum + 1).
    pub fn n_classes(&self) -> usize {
        self.classes.iter().map(|c| c.index + 1).max().unwrap_or(0)
    }

    /// Nama label kelas `index` bila ada.
    pub fn label_of(&self, index: usize) -> Option<&str> {
        self.classes
            .iter()
            .find(|c| c.index == index)
            .and_then(|c| c.label.as_deref())
    }

    /// Versi untuk `AnnotationSource::Ai` (string kosong bila tidak ada).
    pub fn version_string(&self) -> String {
        self.version.clone().unwrap_or_default()
    }

    /// Sumber anotasi untuk model ini.
    pub fn source(&self) -> AnnotationSource {
        AnnotationSource::Ai {
            model: self.name.clone(),
            version: self.version_string(),
        }
    }
}

/// Backend inferensi.
pub trait Segmenter: Send + Sync {
    /// Spesifikasi model.
    fn spec(&self) -> &ModelSpec;
    /// Masukan: RGB f32 NCHW `[1, 3, S, S]` (sudah dinormalisasi). Keluaran:
    /// logit `[1, C, S, S]` dengan `C ≥ spec().n_classes()`.
    fn infer(&self, input: &[f32]) -> Result<Vec<f32>, SegError>;
}

/// Backend tract (ONNX, CPU, Rust murni).
pub struct TractSegmenter {
    spec: ModelSpec,
    model: Arc<TypedRunnableModel>,
    n_out: usize,
}

impl std::fmt::Debug for TractSegmenter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TractSegmenter")
            .field("spec", &self.spec)
            .field("n_out", &self.n_out)
            .finish()
    }
}

impl TractSegmenter {
    /// Muat `model.json` + `model.onnx` dari `model_dir`. Berkas rusak atau
    /// operator tak didukung → `SegError`, tidak panik.
    pub fn load(model_dir: &Path) -> Result<Self, SegError> {
        let json = std::fs::read_to_string(model_dir.join("model.json"))?;
        let spec = ModelSpec::from_json(&json)?;
        let onnx = std::fs::read(model_dir.join("model.onnx"))?;
        Self::from_bytes(spec, &onnx)
    }

    /// Bangun dari spesifikasi dan byte ONNX.
    pub fn from_bytes(spec: ModelSpec, onnx: &[u8]) -> Result<Self, SegError> {
        spec.validate()?;
        let s = spec.input_size;
        let mut reader = std::io::Cursor::new(onnx);
        let model = tract_onnx::onnx()
            .model_for_read(&mut reader)
            .map_err(|e| SegError::Model(format!("ONNX gagal diurai: {e}")))?
            .with_input_fact(
                0,
                InferenceFact::dt_shape(f32::datum_type(), tvec![1, 3, s, s]),
            )
            .map_err(|e| SegError::Model(format!("fakta masukan ditolak: {e}")))?
            .into_optimized()
            .map_err(|e| SegError::Model(format!("optimasi gagal: {e}")))?
            .into_runnable()
            .map_err(|e| SegError::Model(format!("model tidak bisa dijalankan: {e}")))?;
        // Periksa bentuk keluaran statis bila tersedia.
        let out_fact = model
            .model()
            .output_fact(0)
            .map_err(|e| SegError::Model(format!("fakta keluaran: {e}")))?;
        let shape: Vec<Option<usize>> = out_fact.shape.iter().map(|d| d.to_usize().ok()).collect();
        if shape.len() != 4 {
            return Err(SegError::Model(format!(
                "keluaran harus 4D [1, C, S, S], didapat {} dimensi",
                shape.len()
            )));
        }
        if shape[0].is_some_and(|n| n != 1)
            || shape[2].is_some_and(|n| n != s)
            || shape[3].is_some_and(|n| n != s)
        {
            return Err(SegError::Model(format!(
                "bentuk keluaran {shape:?} tidak cocok dengan [1, C, {s}, {s}]"
            )));
        }
        let n_out = shape[1].unwrap_or(0);
        if n_out != 0 && n_out < spec.n_classes() {
            return Err(SegError::Model(format!(
                "model mengeluarkan {n_out} kanal, model.json menuntut {}",
                spec.n_classes()
            )));
        }
        Ok(Self { spec, model, n_out })
    }

    /// Jumlah kanal keluaran statis (0 bila dinamis).
    pub fn output_channels(&self) -> usize {
        self.n_out
    }
}

impl Segmenter for TractSegmenter {
    fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    fn infer(&self, input: &[f32]) -> Result<Vec<f32>, SegError> {
        let s = self.spec.input_size;
        if input.len() != 3 * s * s {
            return Err(SegError::Model(format!(
                "panjang masukan {} ≠ 3×{s}×{s}",
                input.len()
            )));
        }
        let tensor = Tensor::from_shape(&[1, 3, s, s], input)
            .map_err(|e| SegError::Model(format!("tensor masukan: {e}")))?;
        let outputs = self
            .model
            .run(tvec![tensor.into_tvalue()])
            .map_err(|e| SegError::Model(format!("inferensi gagal: {e}")))?;
        let out = outputs
            .first()
            .ok_or_else(|| SegError::Model("model tanpa keluaran".to_string()))?;
        let shape = out.shape();
        if shape.len() != 4 || shape[0] != 1 || shape[2] != s || shape[3] != s {
            return Err(SegError::Model(format!(
                "bentuk keluaran {shape:?} ≠ [1, C, {s}, {s}]"
            )));
        }
        if shape[1] < self.spec.n_classes() {
            return Err(SegError::Model(format!(
                "keluaran {} kanal < {} kelas",
                shape[1],
                self.spec.n_classes()
            )));
        }
        let view = out
            .to_plain_array_view::<f32>()
            .map_err(|e| SegError::Model(format!("keluaran bukan f32: {e}")))?;
        Ok(view.iter().copied().collect())
    }
}

/// Hasil inferensi seluruh slide.
#[derive(Debug, Clone)]
pub struct AiResult {
    /// Peta label (nilai `LabelId`) pada `mpp_label`, bentuk `(h, w)`.
    pub label_map: Array2<u8>,
    /// Ukuran piksel peta (µm).
    pub mpp: f64,
    /// Rata-rata probabilitas softmax kelas argmax atas piksel yang
    /// diinferensi (0 bila tidak ada).
    pub mean_confidence: f32,
    /// Jumlah tile yang dijalankan.
    pub tiles_run: usize,
    /// Jumlah tile yang dilewati karena masker jaringan kosong.
    pub tiles_skipped: usize,
    /// Sumber anotasi (`Ai { model, version }`).
    pub source: AnnotationSource,
}

impl AiResult {
    /// Pecah menjadi satu [`Proposal`] per label yang muncul (terurut).
    pub fn proposals(&self) -> Vec<Proposal> {
        let mut seen = [false; 256];
        for &v in self.label_map.iter() {
            seen[v as usize] = true;
        }
        (1..=255u8)
            .filter(|&l| seen[l as usize])
            .map(|l| Proposal {
                mask: self.label_map.mapv(|v| u8::from(v == l)),
                mpp: self.mpp,
                origin_um: [0.0, 0.0],
                label: LabelId(l),
                source: self.source.clone(),
            })
            .collect()
    }
}

/// Fungsi pembaca tile: `(x_um, y_um, w_px, h_px)` → RGBA `w_px × h_px × 4`
/// pada mpp model; bagian di luar slide boleh diisi apa saja (biasanya putih).
pub type TileReader<'a> = dyn Fn(f64, f64, usize, usize) -> Result<Vec<u8>, SegError> + Sync + 'a;

/// Modus label dari hitungan per nilai, aturan seri seperti
/// `downsample2_label_mode`: nilai bukan-nol terkecil menang; nol hanya bila
/// satu-satunya.
fn mode_from_counts(counts: &[u32; 256]) -> u8 {
    let mut best_val = 0u8;
    let mut best_cnt = 0u32;
    for (v, &c) in counts.iter().enumerate() {
        if c == 0 {
            continue;
        }
        let v = v as u8;
        let better = c > best_cnt || (c == best_cnt && v != 0 && (best_val == 0 || v < best_val));
        if better {
            best_cnt = c;
            best_val = v;
        }
    }
    best_val
}

/// Posisi awal tile sepanjang satu sumbu dengan langkah `stride`, tile
/// terakhir digeser agar tidak melewati `extent` (bila `extent ≥ s`).
fn tile_starts(extent: usize, s: usize, stride: usize) -> Vec<usize> {
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

/// Akumulator baris: logit berbobot `[C][S][W]` dan jumlah bobot `[S][W]`.
struct BandAcc {
    c: usize,
    s: usize,
    w: usize,
    acc: Vec<f32>,
    wsum: Vec<f32>,
}

impl BandAcc {
    fn new(c: usize, s: usize, w: usize) -> Self {
        Self {
            c,
            s,
            w,
            acc: vec![0.0; c * s * w],
            wsum: vec![0.0; s * w],
        }
    }

    /// Tambahkan logit tile `[C][S][S]` yang mulai di kolom `x0` dengan
    /// bobot `win[S×S]`. Baris tile sejajar baris band.
    fn add(&mut self, x0: usize, logits: &[f32], win: &[f32]) {
        let s = self.s;
        let w = self.w;
        let xw = s.min(w.saturating_sub(x0));
        for c in 0..self.c {
            for j in 0..s {
                let src = &logits[c * s * s + j * s..c * s * s + j * s + xw];
                let wn = &win[j * s..j * s + xw];
                let dst = &mut self.acc[c * s * w + j * w + x0..c * s * w + j * w + x0 + xw];
                for i in 0..xw {
                    dst[i] += src[i] * wn[i];
                }
            }
        }
        for j in 0..s {
            let wn = &win[j * s..j * s + xw];
            let dst = &mut self.wsum[j * w + x0..j * w + x0 + xw];
            for i in 0..xw {
                dst[i] += wn[i];
            }
        }
    }

    /// Finalkan baris `0..n`: kelas argmax (dipetakan `class_to_label`) dan
    /// jumlah probabilitas argmax; lalu geser band ke atas sebanyak `n`.
    #[allow(clippy::needless_range_loop)]
    fn finalize_rows(
        &mut self,
        n: usize,
        class_to_label: &[u8],
        out_rows: &mut VecDeque<Vec<u8>>,
        conf_sum: &mut f64,
        conf_n: &mut u64,
    ) {
        let (c, s, w) = (self.c, self.s, self.w);
        let mut scratch = vec![0.0f32; c];
        for j in 0..n.min(s) {
            let mut row = vec![0u8; w];
            for i in 0..w {
                let ws = self.wsum[j * w + i];
                if ws <= 0.0 {
                    continue;
                }
                let mut best = 0usize;
                let mut best_v = f32::NEG_INFINITY;
                for k in 0..c {
                    let v = self.acc[k * s * w + j * w + i] / ws;
                    scratch[k] = v;
                    if v > best_v {
                        best_v = v;
                        best = k;
                    }
                }
                let denom: f32 = scratch.iter().map(|v| (v - best_v).exp()).sum();
                *conf_sum += (1.0 / denom) as f64;
                *conf_n += 1;
                row[i] = class_to_label.get(best).copied().unwrap_or(0);
            }
            out_rows.push_back(row);
        }
        // Geser band.
        let n = n.min(s);
        if n == s {
            self.acc.fill(0.0);
            self.wsum.fill(0.0);
            return;
        }
        for k in 0..c {
            let base = k * s * w;
            self.acc.copy_within(base + n * w..base + s * w, base);
            self.acc[base + (s - n) * w..base + s * w].fill(0.0);
        }
        self.wsum.copy_within(n * w..s * w, 0);
        self.wsum[(s - n) * w..s * w].fill(0.0);
    }
}

/// Penurun resolusi baris demi baris ke `mpp_label` dengan modus.
struct RowDownsampler {
    factor: f64,
    w_out: usize,
    h_out: usize,
    next_out_row: usize,
    /// Baris resolusi model yang sudah final tetapi belum dikonsumsi;
    /// `first_row` = indeks global baris pertama di antrean.
    rows: VecDeque<Vec<u8>>,
    first_row: usize,
    out: Array2<u8>,
}

impl RowDownsampler {
    fn new(w_model: usize, h_model: usize, factor: f64) -> Self {
        let w_out = ((w_model as f64 / factor).ceil() as usize).max(1);
        let h_out = ((h_model as f64 / factor).ceil() as usize).max(1);
        Self {
            factor,
            w_out,
            h_out,
            next_out_row: 0,
            rows: VecDeque::new(),
            first_row: 0,
            out: Array2::zeros((h_out, w_out)),
        }
    }

    fn push_rows(&mut self, rows: &mut VecDeque<Vec<u8>>, h_model: usize) {
        self.rows.append(rows);
        self.consume(h_model);
    }

    /// Konsumsi baris keluaran yang jejaknya sudah tersedia penuh.
    fn consume(&mut self, h_model: usize) {
        let f = self.factor;
        loop {
            if self.next_out_row >= self.h_out {
                break;
            }
            let y = self.next_out_row;
            let ya = ((y as f64 * f).floor() as usize).min(h_model);
            let yb = (((y + 1) as f64 * f).ceil() as usize).clamp(ya + 1, h_model.max(ya + 1));
            let avail_end = self.first_row + self.rows.len();
            if yb > avail_end && avail_end < h_model {
                break;
            }
            let yb = yb.min(avail_end.max(ya + 1));
            let w_model = self.rows.front().map_or(0, Vec::len);
            let mut counts = [0u32; 256];
            for x in 0..self.w_out {
                counts.fill(0);
                let xa = ((x as f64 * f).floor() as usize).min(w_model);
                let xb = (((x + 1) as f64 * f).ceil() as usize).clamp(xa + 1, w_model.max(xa + 1));
                for yy in ya..yb {
                    let Some(row) = self.rows.get(yy - self.first_row) else {
                        continue;
                    };
                    for xx in xa..xb.min(row.len()) {
                        counts[row[xx] as usize] += 1;
                    }
                }
                self.out[[y, x]] = mode_from_counts(&counts);
            }
            self.next_out_row += 1;
            // Buang baris yang tidak lagi dibutuhkan baris keluaran berikutnya.
            let next_ya = ((self.next_out_row as f64 * f).floor() as usize).min(h_model);
            while self.first_row < next_ya && !self.rows.is_empty() {
                self.rows.pop_front();
                self.first_row += 1;
            }
        }
    }

    fn finish(mut self, h_model: usize) -> Array2<u8> {
        self.consume(h_model);
        self.out
    }
}

/// Jalankan model atas seluruh slide dan hasilkan peta label pada `mpp_label`.
///
/// - `read_tile(x_um, y_um, w_px, h_px)` membaca RGBA pada mpp model.
/// - `size_um` ukuran slide.
/// - `tissue_mask` `(masker (h, w), mpp_masker)`: tile tanpa jaringan dilewati.
/// - `label_map(nama)` memetakan nama kelas ke `LabelId`; kelas tanpa nama
///   atau tanpa padanan menjadi latar.
/// - Tile `S × S` bertumpang tindih 25 %; logit × jendela Hann (+ε agar tepi
///   slide tetap berbobot) dijumlahkan; argmax → label; diturunkan ke
///   `mpp_label` dengan modus.
pub fn infer_slide(
    seg: &dyn Segmenter,
    read_tile: &TileReader<'_>,
    size_um: [f64; 2],
    tissue_mask: Option<(&Array2<u8>, f64)>,
    label_map: &dyn Fn(&str) -> Option<LabelId>,
    mpp_label: f64,
    progress: &dyn Progress,
) -> Result<AiResult, SegError> {
    let spec = seg.spec();
    spec.validate()?;
    if !(mpp_label.is_finite() && mpp_label > 0.0) {
        return Err(SegError::InvalidParameter(format!(
            "mpp_label tidak sah: {mpp_label}"
        )));
    }
    if !(size_um[0].is_finite() && size_um[0] > 0.0 && size_um[1].is_finite() && size_um[1] > 0.0) {
        return Err(SegError::InvalidParameter(format!(
            "ukuran slide tidak sah: {size_um:?}"
        )));
    }
    if let Some((tm, tm_mpp)) = tissue_mask {
        if !(tm_mpp.is_finite() && tm_mpp > 0.0) || tm.is_empty() {
            return Err(SegError::InvalidParameter(
                "masker jaringan / mpp-nya tidak sah".to_string(),
            ));
        }
    }
    let s = spec.input_size;
    let mpp = spec.mpp;
    let w_model = (size_um[0] / mpp).ceil() as usize;
    let h_model = (size_um[1] / mpp).ceil() as usize;
    let c_min = spec.n_classes();
    let stride = (s - s / 4).max(1);
    let xs = tile_starts(w_model, s, stride);
    let ys = tile_starts(h_model, s, stride);
    // Lebar band = lebar yang benar-benar dicakup tile (≥ w_model).
    let w_band = w_model.max(s);

    // Jendela Hann + ε.
    let win = hann_window_2d(s, s);
    let win: Vec<f32> = win.iter().map(|&v| v + 1e-3).collect();

    // Peta kelas → label dari nama.
    let class_to_label: Vec<u8> = (0..c_min)
        .map(|k| {
            spec.label_of(k)
                .and_then(label_map)
                .map(|l| l.0)
                .unwrap_or(0)
        })
        .collect();

    let tissue_has = |x0: usize, y0: usize| -> bool {
        let Some((tm, tm_mpp)) = tissue_mask else {
            return true;
        };
        let (th, tw) = tm.dim();
        let ux0 = x0 as f64 * mpp;
        let uy0 = y0 as f64 * mpp;
        let ux1 = ux0 + s as f64 * mpp;
        let uy1 = uy0 + s as f64 * mpp;
        let tx0 = ((ux0 / tm_mpp).floor().max(0.0) as usize).min(tw);
        let ty0 = ((uy0 / tm_mpp).floor().max(0.0) as usize).min(th);
        let tx1 = ((ux1 / tm_mpp).ceil().max(0.0) as usize).min(tw);
        let ty1 = ((uy1 / tm_mpp).ceil().max(0.0) as usize).min(th);
        (ty0..ty1).any(|y| (tx0..tx1).any(|x| tm[[y, x]] != 0))
    };

    let mean = spec.mean;
    let std = spec.std;
    let normalize = |rgba: &[u8]| -> Result<Vec<f32>, SegError> {
        if rgba.len() != s * s * 4 {
            return Err(SegError::InvalidParameter(format!(
                "pembaca tile mengembalikan {} byte, diharapkan {}",
                rgba.len(),
                s * s * 4
            )));
        }
        let mut out = vec![0.0f32; 3 * s * s];
        for (i, px) in rgba.chunks_exact(4).enumerate() {
            for ch in 0..3 {
                out[ch * s * s + i] = (px[ch] as f32 / 255.0 - mean[ch]) / std[ch];
            }
        }
        Ok(out)
    };

    let mut band = BandAcc::new(c_min, s, w_band);
    let mut final_rows: VecDeque<Vec<u8>> = VecDeque::new();
    let mut down = RowDownsampler::new(w_model, h_model, mpp_label / mpp);
    let mut conf_sum = 0.0f64;
    let mut conf_n = 0u64;
    let mut tiles_run = 0usize;
    let mut tiles_skipped = 0usize;
    let total_rows = ys.len();
    let group = rayon::current_num_threads().max(1);

    for (ri, &y0) in ys.iter().enumerate() {
        if progress.is_cancelled() {
            return Err(SegError::Cancelled);
        }
        progress.set(ri as f32 / total_rows as f32, "Inferensi AI");
        let todo: Vec<usize> = xs
            .iter()
            .copied()
            .filter(|&x0| {
                let keep = tissue_has(x0, y0);
                if !keep {
                    tiles_skipped += 1;
                }
                keep
            })
            .collect();
        for chunk in todo.chunks(group) {
            let outputs: Vec<Result<Vec<f32>, SegError>> = chunk
                .par_iter()
                .map(|&x0| {
                    let rgba = read_tile(x0 as f64 * mpp, y0 as f64 * mpp, s, s)?;
                    let input = normalize(&rgba)?;
                    let logits = seg.infer(&input)?;
                    if logits.len() < c_min * s * s {
                        return Err(SegError::Model(format!(
                            "keluaran {} nilai < {}×{s}×{s}",
                            logits.len(),
                            c_min
                        )));
                    }
                    Ok(logits)
                })
                .collect();
            for (k, out) in outputs.into_iter().enumerate() {
                let logits = out?;
                band.add(chunk[k], &logits, &win);
                tiles_run += 1;
            }
        }
        // Baris yang tidak lagi tersentuh baris tile berikutnya.
        let rows_done = match ys.get(ri + 1) {
            Some(&y_next) => y_next - y0,
            None => s,
        };
        band.finalize_rows(
            rows_done,
            &class_to_label,
            &mut final_rows,
            &mut conf_sum,
            &mut conf_n,
        );
        // Potong baris di luar slide (tile terakhir mungkin melewati h_model
        // hanya bila h_model < s).
        while final_rows.len() + down.first_row + down.rows.len() > h_model {
            final_rows.pop_back();
        }
        down.push_rows(&mut final_rows, h_model);
    }
    progress.set(1.0, "Inferensi AI");
    let label_map_out = down.finish(h_model);
    let mean_confidence = if conf_n > 0 {
        (conf_sum / conf_n as f64) as f32
    } else {
        0.0
    };
    Ok(AiResult {
        label_map: label_map_out,
        mpp: mpp_label,
        mean_confidence,
        tiles_run,
        tiles_skipped,
        source: spec.source(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use h3d_core::NoProgress;
    use prost::Message;
    use tract_onnx::pb;

    fn spec(s: usize, mpp: f64) -> ModelSpec {
        ModelSpec {
            name: "uji".to_string(),
            version: Some("0.1".to_string()),
            input_size: s,
            mpp,
            mean: [0.0; 3],
            std: [1.0; 3],
            classes: vec![
                ClassSpec {
                    index: 0,
                    label: None,
                },
                ClassSpec {
                    index: 1,
                    label: Some("Tumor".to_string()),
                },
                ClassSpec {
                    index: 2,
                    label: Some("Stroma".to_string()),
                },
            ],
            task: "segmentation".to_string(),
        }
    }

    /// Segmenter tiruan: logit konstan per kelas.
    struct ConstSeg {
        spec: ModelSpec,
        logits: Vec<f32>,
    }

    impl Segmenter for ConstSeg {
        fn spec(&self) -> &ModelSpec {
            &self.spec
        }
        fn infer(&self, input: &[f32]) -> Result<Vec<f32>, SegError> {
            let s = self.spec.input_size;
            assert_eq!(input.len(), 3 * s * s);
            let mut out = Vec::with_capacity(self.logits.len() * s * s);
            for &l in &self.logits {
                out.extend(std::iter::repeat_n(l, s * s));
            }
            Ok(out)
        }
    }

    fn label_map(name: &str) -> Option<LabelId> {
        match name {
            "Tumor" => Some(LabelId(3)),
            "Stroma" => Some(LabelId(5)),
            _ => None,
        }
    }

    #[test]
    fn tile_starts_cover_extent() {
        assert_eq!(tile_starts(10, 64, 48), vec![0]);
        assert_eq!(tile_starts(64, 64, 48), vec![0]);
        let v = tile_starts(200, 64, 48);
        assert_eq!(v.first(), Some(&0));
        assert_eq!(*v.last().unwrap() + 64, 200);
        for w in v.windows(2) {
            assert!(w[1] - w[0] <= 48);
        }
    }

    #[test]
    fn constant_logits_give_seamless_constant_map() {
        let s = 32usize;
        let seg = ConstSeg {
            spec: spec(s, 2.0),
            logits: vec![0.1, 2.0, 0.5],
        };
        let reader = |_x: f64, _y: f64, w: usize, h: usize| Ok(vec![200u8; w * h * 4]);
        // 1000 × 700 µm → 500 × 350 px model → 250 × 175 px label.
        let r = infer_slide(
            &seg,
            &reader,
            [1000.0, 700.0],
            None,
            &label_map,
            4.0,
            &NoProgress,
        )
        .unwrap();
        assert_eq!(r.label_map.dim(), (175, 250));
        assert!(
            r.label_map.iter().all(|&v| v == 3),
            "ada sambungan/piksel latar"
        );
        assert!(r.tiles_run > 1);
        assert_eq!(r.tiles_skipped, 0);
        // softmax([0.1, 2.0, 0.5])[1]
        let e: Vec<f32> = [0.1f32, 2.0, 0.5].iter().map(|v| v.exp()).collect();
        let p = e[1] / e.iter().sum::<f32>();
        assert!(
            (r.mean_confidence - p).abs() < 1e-4,
            "{}",
            r.mean_confidence
        );
        let props = r.proposals();
        assert_eq!(props.len(), 1);
        assert_eq!(props[0].label, LabelId(3));
        assert!(matches!(r.source, AnnotationSource::Ai { .. }));
    }

    #[test]
    fn tissue_mask_skips_tiles_and_non_integer_factor() {
        let s = 16usize;
        let seg = ConstSeg {
            spec: spec(s, 1.0),
            logits: vec![0.0, 0.0, 3.0],
        };
        let reader = |_x: f64, _y: f64, w: usize, h: usize| Ok(vec![0u8; w * h * 4]);
        // Masker jaringan pada mpp 4: hanya separuh kiri.
        let mut tm = Array2::<u8>::zeros((25, 50));
        for y in 0..25 {
            for x in 0..20 {
                tm[[y, x]] = 1;
            }
        }
        // Slide 200 × 100 µm → model 200 × 100 px → label (mpp 3) 67 × 34.
        let r = infer_slide(
            &seg,
            &reader,
            [200.0, 100.0],
            Some((&tm, 4.0)),
            &label_map,
            3.0,
            &NoProgress,
        )
        .unwrap();
        assert_eq!(r.label_map.dim(), (34, 67));
        assert!(r.tiles_skipped > 0);
        // Kiri = Stroma (5), kanan jauh = latar.
        assert_eq!(r.label_map[[10, 5]], 5);
        assert_eq!(r.label_map[[10, 60]], 0);
    }

    #[test]
    fn reader_error_and_cancel_propagate() {
        let s = 16usize;
        let seg = ConstSeg {
            spec: spec(s, 1.0),
            logits: vec![0.0, 1.0, 0.0],
        };
        let bad = |_x: f64, _y: f64, _w: usize, _h: usize| Ok(vec![0u8; 3]);
        assert!(infer_slide(&seg, &bad, [50.0, 50.0], None, &label_map, 4.0, &NoProgress).is_err());
        struct Cancel;
        impl Progress for Cancel {
            fn set(&self, _: f32, _: &str) {}
            fn is_cancelled(&self) -> bool {
                true
            }
        }
        let ok = |_x: f64, _y: f64, w: usize, h: usize| Ok(vec![0u8; w * h * 4]);
        assert!(matches!(
            infer_slide(&seg, &ok, [50.0, 50.0], None, &label_map, 4.0, &Cancel),
            Err(SegError::Cancelled)
        ));
    }

    // ------------------------------------------------------------------
    // Model ONNX mungil dibangun lewat tipe protobuf tract (prost).
    // ------------------------------------------------------------------

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

    /// Conv 1×1: keluaran kanal 0 = R, kanal 1 = G + 0,5. Satu-satunya
    /// operator: `Conv` (opset 13).
    fn tiny_onnx(s: i64) -> Vec<u8> {
        let conv = pb::NodeProto {
            input: vec!["x".into(), "w".into(), "b".into()],
            output: vec!["y".into()],
            name: "conv".into(),
            op_type: "Conv".into(),
            domain: String::new(),
            attribute: vec![pb::AttributeProto {
                name: "kernel_shape".into(),
                r#type: pb::attribute_proto::AttributeType::Ints as i32,
                ints: vec![1, 1],
                ..Default::default()
            }],
            doc_string: String::new(),
        };
        let graph = pb::GraphProto {
            node: vec![conv],
            name: "tiny".into(),
            initializer: vec![
                float_tensor("w", &[2, 3, 1, 1], &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
                float_tensor("b", &[2], &[0.0, 0.5]),
            ],
            input: vec![value_info("x", &[1, 3, s, s])],
            output: vec![value_info("y", &[1, 2, s, s])],
            ..Default::default()
        };
        let model = pb::ModelProto {
            ir_version: 8,
            producer_name: "h3d-segment-test".into(),
            graph: Some(graph),
            opset_import: vec![pb::OperatorSetIdProto {
                domain: String::new(),
                version: 13,
            }],
            ..Default::default()
        };
        model.encode_to_vec()
    }

    fn two_class_spec(s: usize) -> ModelSpec {
        ModelSpec {
            classes: vec![
                ClassSpec {
                    index: 0,
                    label: None,
                },
                ClassSpec {
                    index: 1,
                    label: Some("Tumor".to_string()),
                },
            ],
            ..spec(s, 2.0)
        }
    }

    #[test]
    fn tract_loads_tiny_model_and_infers_constant() {
        let s = 8usize;
        let onnx = tiny_onnx(s as i64);
        assert!(onnx.len() < 1024, "model uji {} byte", onnx.len());
        let seg = TractSegmenter::from_bytes(two_class_spec(s), &onnx).unwrap();
        assert_eq!(seg.output_channels(), 2);
        let mut input = vec![0.0f32; 3 * s * s];
        input[..s * s].fill(0.2);
        input[s * s..2 * s * s].fill(0.7);
        input[2 * s * s..].fill(0.1);
        let out = seg.infer(&input).unwrap();
        assert_eq!(out.len(), 2 * s * s);
        assert!(out[..s * s].iter().all(|v| (v - 0.2).abs() < 1e-5));
        assert!(out[s * s..].iter().all(|v| (v - 1.2).abs() < 1e-5));
        // Panjang masukan salah → galat.
        assert!(seg.infer(&input[..10]).is_err());
    }

    #[test]
    fn tract_load_from_dir_and_bad_files() {
        let s = 8usize;
        let dir = tempfile::tempdir().unwrap();
        let spec_json = serde_json::to_string(&two_class_spec(s)).unwrap();
        // Tanpa model.onnx → galat I/O.
        std::fs::write(dir.path().join("model.json"), &spec_json).unwrap();
        assert!(matches!(
            TractSegmenter::load(dir.path()),
            Err(SegError::Io(_))
        ));
        // model.onnx sampah → galat model.
        std::fs::write(dir.path().join("model.onnx"), b"bukan onnx sama sekali").unwrap();
        assert!(matches!(
            TractSegmenter::load(dir.path()),
            Err(SegError::Model(_))
        ));
        // Model baik.
        std::fs::write(dir.path().join("model.onnx"), tiny_onnx(s as i64)).unwrap();
        let seg = TractSegmenter::load(dir.path()).unwrap();
        assert_eq!(seg.spec().name, "uji");
        // model.json rusak → galat model.
        std::fs::write(dir.path().join("model.json"), "{ \"name\": ").unwrap();
        assert!(matches!(
            TractSegmenter::load(dir.path()),
            Err(SegError::Model(_))
        ));
        // model.json menuntut 3 kelas padahal model hanya 2 kanal.
        std::fs::write(
            dir.path().join("model.json"),
            serde_json::to_string(&spec(s, 2.0)).unwrap(),
        )
        .unwrap();
        assert!(matches!(
            TractSegmenter::load(dir.path()),
            Err(SegError::Model(_))
        ));
        // Direktori tidak ada.
        assert!(TractSegmenter::load(Path::new("/tidak/ada/di/mana-mana")).is_err());
    }

    #[test]
    fn tract_end_to_end_slide() {
        let s = 8usize;
        let seg = TractSegmenter::from_bytes(two_class_spec(s), &tiny_onnx(s as i64)).unwrap();
        // Kiri: G tinggi → kanal 1 (G + 0,5, Tumor) menang. Kanan: R tinggi dan
        // G nol → kanal 0 (R ≈ 0,98 > 0,5) menang → latar.
        let reader = |x: f64, _y: f64, w: usize, h: usize| {
            let px: [u8; 4] = if x < 100.0 {
                [10, 250, 10, 255]
            } else {
                [250, 0, 10, 255]
            };
            let mut v = Vec::with_capacity(w * h * 4);
            for _ in 0..w * h {
                v.extend_from_slice(&px);
            }
            Ok(v)
        };
        let r = infer_slide(
            &seg,
            &reader,
            [200.0, 40.0],
            None,
            &label_map,
            4.0,
            &NoProgress,
        )
        .unwrap();
        assert_eq!(r.label_map.dim(), (10, 50));
        assert_eq!(r.label_map[[5, 2]], 3);
        assert_eq!(r.label_map[[5, 47]], 0);
    }

    #[test]
    fn model_spec_validation() {
        assert!(ModelSpec::from_json("{}").is_err());
        let mut sp = spec(16, 1.0);
        sp.std[0] = 0.0;
        assert!(sp.validate().is_err());
        let mut sp = spec(16, 1.0);
        sp.classes.push(ClassSpec {
            index: 1,
            label: None,
        });
        assert!(sp.validate().is_err());
        let json = r#"{"name":"m","input_size":64,"mpp":1.0,"mean":[0,0,0],"std":[1,1,1],
            "classes":[{"index":0,"label":null},{"index":1,"label":"Tumor"}]}"#;
        let sp = ModelSpec::from_json(json).unwrap();
        assert_eq!(sp.task, "segmentation");
        assert_eq!(sp.version, None);
        assert_eq!(sp.n_classes(), 2);
    }
}
