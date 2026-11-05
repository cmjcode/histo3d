//! Loop pelatihan: AdamW, jadwal LR pemanasan + kosinus, loss entropi silang +
//! Dice lunak (kelas latar dikecualikan dari Dice), validasi Dice per kelas
//! pada tile tumpukan holdout, checkpoint terbaik ke `out_dir/model.bpk`.
//!
//! Determinisme: pemilihan dan augmentasi tiap sampel memakai benih turunan
//! `(seed, step, i)`, sehingga urutan thread rayon tidak memengaruhi data.

use std::fs;
use std::path::{Path, PathBuf};

use burn::module::Module;
use burn::optim::{AdamWConfig, GradientsParams};
use burn::tensor::{activation, Device, Int, Tensor, TensorData};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::augment::{augment, crop, AugParams};
use crate::data::{read_tile, Tile, TileIndex, TileMeta, N_CLASSES};
use crate::model::{UNet, UNetConfig};
use crate::TrainError;

/// Rata-rata normalisasi per kanal (ImageNet; ditulis ke `model.json`).
pub const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
/// Simpangan baku normalisasi per kanal.
pub const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Konfigurasi pelatihan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    /// Jumlah langkah optimasi.
    pub steps: usize,
    /// Ukuran batch.
    pub batch: usize,
    /// Sisi potongan latih (≤ sisi tile cache, kelipatan `2^depth`).
    pub crop: usize,
    /// LR puncak.
    pub lr: f64,
    /// Benih.
    pub seed: u64,
    /// Tumpukan holdout (tidak dilatih; dipakai validasi).
    pub holdout: Vec<String>,
    /// Kanal dasar U-Net.
    pub base: usize,
    /// Kedalaman U-Net.
    pub depth: usize,
    /// Validasi tiap N langkah.
    pub eval_every: usize,
    /// Fraksi tumor minimum agar tile dihitung "tile tumor" saat penyeimbangan.
    pub tumor_min_frac: f32,
    /// Jumlah tile validasi maksimum (0 = semua).
    pub max_val_tiles: usize,
    /// Augmentasi.
    pub aug: AugParamsSer,
}

/// `AugParams` yang bisa diserialisasi.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugParamsSer {
    /// Lihat [`AugParams::stain_scale`].
    pub stain_scale: f32,
    /// Lihat [`AugParams::stain_shift`].
    pub stain_shift: f32,
    /// Lihat [`AugParams::blur_prob`].
    pub blur_prob: f32,
    /// Lihat [`AugParams::blur_sigma_max`].
    pub blur_sigma_max: f64,
}

impl From<&AugParamsSer> for AugParams {
    fn from(a: &AugParamsSer) -> Self {
        AugParams {
            stain_scale: a.stain_scale,
            stain_shift: a.stain_shift,
            blur_prob: a.blur_prob,
            blur_sigma_max: a.blur_sigma_max,
        }
    }
}

impl Default for AugParamsSer {
    fn default() -> Self {
        let d = AugParams::default();
        Self {
            stain_scale: d.stain_scale,
            stain_shift: d.stain_shift,
            blur_prob: d.blur_prob,
            blur_sigma_max: d.blur_sigma_max,
        }
    }
}

/// Metadata model yang disimpan bersama checkpoint (`train_meta.json`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainMeta {
    /// Konfigurasi arsitektur.
    pub base: usize,
    /// Kedalaman.
    pub depth: usize,
    /// Jumlah kelas.
    pub n_classes: usize,
    /// Nama kelas.
    pub classes: Vec<Option<String>>,
    /// mpp tile latih (µm/piksel).
    pub mpp: f64,
    /// Sumber mpp.
    pub mpp_source: String,
    /// Rata-rata normalisasi.
    pub mean: [f32; 3],
    /// Simpangan baku normalisasi.
    pub std: [f32; 3],
    /// Dice validasi terbaik per kelas (indeks = kelas); kosong bila belum ada validasi.
    pub best_val_dice: Vec<f32>,
    /// Langkah checkpoint terbaik.
    pub best_step: usize,
    /// Tumpukan holdout.
    pub holdout: Vec<String>,
}

impl TrainMeta {
    /// Muat `train_meta.json`.
    pub fn load(path: &Path) -> Result<Self, TrainError> {
        let text = fs::read_to_string(path).map_err(|e| TrainError::io(path, e))?;
        Ok(serde_json::from_str(&text)?)
    }

    /// Konfigurasi U-Net yang cocok dengan bobot.
    pub fn unet_config(&self) -> UNetConfig {
        UNetConfig::new()
            .with_base(self.base)
            .with_depth(self.depth)
            .with_n_classes(self.n_classes)
    }
}

fn write_atomic(path: &Path, text: &str) -> Result<(), TrainError> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, text).map_err(|e| TrainError::io(&tmp, e))?;
    fs::rename(&tmp, path).map_err(|e| TrainError::io(path, e))
}

/// Tensor masukan `[B, 3, S, S]` ternormalisasi dari tile.
pub fn to_input(tiles: &[Tile], device: &Device) -> Result<Tensor<4>, TrainError> {
    let b = tiles.len();
    let s = tiles.first().map(|t| t.size).unwrap_or(0);
    if b == 0 || s == 0 || tiles.iter().any(|t| t.size != s) {
        return Err(TrainError::InvalidParameter(
            "batch kosong atau ukuran tile beragam".into(),
        ));
    }
    let mut v = Vec::with_capacity(b * 3 * s * s);
    for t in tiles {
        for c in 0..3 {
            v.extend(
                t.rgb
                    .chunks_exact(3)
                    .map(|px| (f32::from(px[c]) / 255.0 - MEAN[c]) / STD[c]),
            );
        }
    }
    Ok(Tensor::from_data(TensorData::new(v, [b, 3, s, s]), device))
}

/// Tensor target `[B, S, S]` (kelas) dari tile.
pub fn to_target(tiles: &[Tile], device: &Device) -> Result<Tensor<3, Int>, TrainError> {
    let b = tiles.len();
    let s = tiles.first().map(|t| t.size).unwrap_or(0);
    if b == 0 || s == 0 {
        return Err(TrainError::InvalidParameter("batch kosong".into()));
    }
    let v: Vec<i32> = tiles
        .iter()
        .flat_map(|t| t.cls.iter().map(|&c| i32::from(c)))
        .collect();
    Ok(Tensor::from_data(TensorData::new(v, [b, s, s]), device))
}

/// Loss = entropi silang + (1 − Dice lunak rata-rata kelas 1..C).
pub fn seg_loss(logits: Tensor<4>, target: Tensor<3, Int>, n_classes: usize) -> Tensor<1> {
    let [b, c, _, _] = logits.dims();
    let onehot: Tensor<4> = target.one_hot::<4>(n_classes).float().permute([0, 3, 1, 2]);
    let logp = activation::log_softmax(logits.clone(), 1);
    let ce = (logp * onehot.clone()).sum_dim(1).mean().neg();
    let p = activation::softmax(logits, 1);
    let inter = (p.clone() * onehot.clone()).sum_dim(3).sum_dim(2);
    let denom = p.sum_dim(3).sum_dim(2) + onehot.sum_dim(3).sum_dim(2);
    let dice = (inter.mul_scalar(2.0).add_scalar(1.0)) / denom.add_scalar(1.0);
    let dice_fg = dice.slice([0..b, 1..c]);
    ce + dice_fg.mean().neg().add_scalar(1.0)
}

/// Dice per kelas dari prediksi dan target (CPU). Kelas tanpa piksel di
/// keduanya → 1.
pub fn dice_per_class(pred: &[u8], target: &[u8], n_classes: usize) -> Vec<f32> {
    let mut inter = vec![0u64; n_classes];
    let mut np = vec![0u64; n_classes];
    let mut nt = vec![0u64; n_classes];
    for (&p, &t) in pred.iter().zip(target) {
        np[p as usize] += 1;
        nt[t as usize] += 1;
        if p == t {
            inter[p as usize] += 1;
        }
    }
    (0..n_classes)
        .map(|k| {
            let d = np[k] + nt[k];
            if d == 0 {
                1.0
            } else {
                2.0 * inter[k] as f32 / d as f32
            }
        })
        .collect()
}

/// Argmax kanal dari logit `[B, C, S, S]` (dibaca ke CPU) → kelas per piksel.
pub fn argmax_cpu(logits: Tensor<4>) -> Result<Vec<u8>, TrainError> {
    let [b, c, h, w] = logits.dims();
    let data = logits.into_data_as::<f32>();
    let v = data
        .as_slice::<f32>()
        .map_err(|e| TrainError::Model(format!("logit bukan f32: {e:?}")))?;
    let hw = h * w;
    let mut out = Vec::with_capacity(b * hw);
    for bi in 0..b {
        for i in 0..hw {
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for k in 0..c {
                let x = v[bi * c * hw + k * hw + i];
                if x > bv {
                    bv = x;
                    best = k;
                }
            }
            out.push(best as u8);
        }
    }
    Ok(out)
}

/// Nilai skalar f32 dari tensor 1 elemen.
pub fn scalar(t: Tensor<1>) -> Result<f32, TrainError> {
    let d = t.into_data_as::<f32>();
    d.as_slice::<f32>()
        .ok()
        .and_then(|s| s.first().copied())
        .ok_or_else(|| TrainError::Model("tensor skalar kosong".into()))
}

/// LR pada langkah `step`: pemanasan linear lalu kosinus ke 5 % LR puncak.
pub fn lr_at(cfg: &TrainConfig, step: usize) -> f64 {
    let warm = (cfg.steps / 20).clamp(1, 200);
    if step < warm {
        return cfg.lr * (step + 1) as f64 / warm as f64;
    }
    let t = (step - warm) as f64 / (cfg.steps.saturating_sub(warm).max(1)) as f64;
    let cos = 0.5 * (1.0 + (std::f64::consts::PI * t).cos());
    cfg.lr * (0.05 + 0.95 * cos)
}

fn derive_seed(seed: u64, step: usize, i: usize) -> u64 {
    // SplitMix64 atas (seed, step, i).
    let mut z = seed
        ^ (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (i as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Ambil satu batch latih: setengah dari tile tumor, setengah dari sisanya
/// (bila keduanya ada), lalu augmentasi; deterministik terhadap `(seed, step)`.
pub fn sample_batch(
    tiles_dir: &Path,
    tumor: &[&TileMeta],
    other: &[&TileMeta],
    cfg: &TrainConfig,
    step: usize,
) -> Result<Vec<Tile>, TrainError> {
    let aug = AugParams::from(&cfg.aug);
    (0..cfg.batch)
        .into_par_iter()
        .map(|i| {
            let mut rng = ChaCha8Rng::seed_from_u64(derive_seed(cfg.seed, step, i));
            let pool = if (!tumor.is_empty() && (i % 2 == 0)) || other.is_empty() {
                tumor
            } else {
                other
            };
            let meta = pool[rng.random_range(0..pool.len())];
            let tile = read_tile(&tiles_dir.join(&meta.file))?;
            Ok(augment(&tile, cfg.crop, &aug, &mut rng))
        })
        .collect()
}

/// Dice validasi per kelas atas tile holdout (potongan tengah `crop`, tanpa
/// augmentasi), dihitung sebagai Dice global (akumulasi seluruh piksel).
pub fn validate(
    model: &UNet,
    tiles_dir: &Path,
    val: &[&TileMeta],
    crop_size: usize,
    device: &Device,
    log: &dyn Fn(&str),
) -> Result<Vec<f32>, TrainError> {
    let m = model.valid();
    let mut pred_all = Vec::new();
    let mut tgt_all = Vec::new();
    for (i, chunk) in val.chunks(4).enumerate() {
        let tiles: Vec<Tile> = chunk
            .par_iter()
            .map(|meta| {
                let t = read_tile(&tiles_dir.join(&meta.file))?;
                let off = (t.size.saturating_sub(crop_size)) / 2;
                Ok(crop(&t, off, off, crop_size.min(t.size)))
            })
            .collect::<Result<_, TrainError>>()?;
        let x = to_input(&tiles, device)?;
        let logits = m.forward(x).detach();
        pred_all.extend(argmax_cpu(logits)?);
        tgt_all.extend(tiles.iter().flat_map(|t| t.cls.iter().copied()));
        if i % 25 == 0 {
            log(&format!("  validasi {}/{}", (i + 1) * 4, val.len()));
        }
    }
    Ok(dice_per_class(&pred_all, &tgt_all, N_CLASSES))
}

/// Jalankan pelatihan; mengembalikan folder keluaran berisi `model.bpk`,
/// `train_meta.json`, `train_config.json`, `metrics.jsonl`.
pub fn train(
    index: &TileIndex,
    cache_dir: &Path,
    cfg: &TrainConfig,
    out_dir: &Path,
    device: &Device,
    log: &dyn Fn(&str),
) -> Result<PathBuf, TrainError> {
    let unet_cfg = UNetConfig::new()
        .with_base(cfg.base)
        .with_depth(cfg.depth)
        .with_n_classes(N_CLASSES);
    unet_cfg.validate().map_err(TrainError::InvalidParameter)?;
    let div = 1usize << cfg.depth;
    if cfg.crop == 0 || !cfg.crop.is_multiple_of(div) || cfg.crop > index.tile {
        return Err(TrainError::InvalidParameter(format!(
            "crop {} harus kelipatan {div} dan ≤ tile {}",
            cfg.crop, index.tile
        )));
    }
    if cfg.batch == 0 || cfg.steps == 0 || !(cfg.lr.is_finite() && cfg.lr > 0.0) {
        return Err(TrainError::InvalidParameter(
            "batch/steps/lr harus > 0".into(),
        ));
    }
    let tiles_dir = cache_dir.join("tiles");
    let (train_tiles, mut val_tiles) = index.split(&cfg.holdout);
    if train_tiles.is_empty() {
        return Err(TrainError::Dataset(
            "tidak ada tile latih setelah holdout".into(),
        ));
    }
    if cfg.max_val_tiles > 0 && val_tiles.len() > cfg.max_val_tiles {
        // Ambil merata sepanjang daftar (deterministik).
        let step = val_tiles.len() as f64 / cfg.max_val_tiles as f64;
        val_tiles = (0..cfg.max_val_tiles)
            .map(|i| val_tiles[(i as f64 * step) as usize])
            .collect();
    }
    let (tumor, other): (Vec<&TileMeta>, Vec<&TileMeta>) = train_tiles
        .iter()
        .partition(|t| t.tumor_frac >= cfg.tumor_min_frac);
    log(&format!(
        "tile latih {} (tumor {}, lain {}), validasi {}",
        train_tiles.len(),
        tumor.len(),
        other.len(),
        val_tiles.len()
    ));
    fs::create_dir_all(out_dir).map_err(|e| TrainError::io(out_dir, e))?;
    write_atomic(
        &out_dir.join("train_config.json"),
        &serde_json::to_string_pretty(cfg)?,
    )?;

    device.seed(cfg.seed);
    let mut model = unet_cfg.init(device).train();
    let mut optim = AdamWConfig::new().with_weight_decay(1e-4).init();
    let mut metrics = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out_dir.join("metrics.jsonl"))
        .map_err(|e| TrainError::io(out_dir.join("metrics.jsonl"), e))?;
    let mut best_score = f32::NEG_INFINITY;
    let mut meta = TrainMeta {
        base: cfg.base,
        depth: cfg.depth,
        n_classes: N_CLASSES,
        classes: index.classes.clone(),
        mpp: index.mpp,
        mpp_source: index.mpp_source.clone(),
        mean: MEAN,
        std: STD,
        best_val_dice: Vec::new(),
        best_step: 0,
        holdout: cfg.holdout.clone(),
    };
    let mut running = 0.0f32;
    let (mut t_data, mut t_step) = (0.0f64, 0.0f64);
    for step in 0..cfg.steps {
        let t0 = std::time::Instant::now();
        let batch = sample_batch(&tiles_dir, &tumor, &other, cfg, step)?;
        let x = to_input(&batch, device)?;
        let y = to_target(&batch, device)?;
        let t1 = std::time::Instant::now();
        let logits = model.forward(x);
        let loss = seg_loss(logits, y, N_CLASSES);
        let lr = lr_at(cfg, step);
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(lr, model, grads);
        let l = scalar(loss.detach())?;
        let t2 = std::time::Instant::now();
        t_data += (t1 - t0).as_secs_f64();
        t_step += (t2 - t1).as_secs_f64();
        running = if step == 0 {
            l
        } else {
            0.98 * running + 0.02 * l
        };
        if step % 20 == 0 || step + 1 == cfg.steps {
            log(&format!(
                "langkah {step}/{} loss {l:.4} (rata {running:.4}) lr {lr:.2e} | data {:.2} s, komputasi {:.2} s per langkah",
                cfg.steps,
                t_data / (step + 1) as f64,
                t_step / (step + 1) as f64
            ));
        }
        let last = step + 1 == cfg.steps;
        if cfg.eval_every > 0 && ((step + 1) % cfg.eval_every == 0 || last) && !val_tiles.is_empty()
        {
            let dice = validate(&model, &tiles_dir, &val_tiles, index.tile, device, log)?;
            let score = dice[1..].iter().sum::<f32>() / (N_CLASSES - 1) as f32;
            log(&format!(
                "  Dice validasi {dice:?} (rata non-latar {score:.4})"
            ));
            use std::io::Write;
            writeln!(
                metrics,
                "{}",
                serde_json::json!({"step": step + 1, "loss": l, "loss_ema": running, "lr": lr, "val_dice": dice})
            )
            .map_err(|e| TrainError::io(out_dir.join("metrics.jsonl"), e))?;
            if score > best_score {
                best_score = score;
                meta.best_val_dice = dice;
                meta.best_step = step + 1;
                save_checkpoint(&model, &meta, out_dir)?;
                log("  checkpoint terbaik disimpan");
            }
        } else if val_tiles.is_empty() && last {
            meta.best_step = step + 1;
            save_checkpoint(&model, &meta, out_dir)?;
        }
    }
    Ok(out_dir.to_path_buf())
}

/// Simpan bobot (`model.bpk`) dan metadata secara atomik.
pub fn save_checkpoint(model: &UNet, meta: &TrainMeta, out_dir: &Path) -> Result<(), TrainError> {
    let path = out_dir.join("model.bpk");
    let tmp = out_dir.join("model.tmp.bpk");
    model
        .clone()
        .valid()
        .save_file(&tmp)
        .map_err(|e| TrainError::Model(format!("simpan bobot: {e:?}")))?;
    fs::rename(&tmp, &path).map_err(|e| TrainError::io(&path, e))?;
    write_atomic(
        &out_dir.join("train_meta.json"),
        &serde_json::to_string_pretty(meta)?,
    )
}

/// Muat model dari folder run (`model.bpk` + `train_meta.json`) pada `device`.
pub fn load_checkpoint(run_dir: &Path, device: &Device) -> Result<(UNet, TrainMeta), TrainError> {
    let meta = TrainMeta::load(&run_dir.join("train_meta.json"))?;
    let cfg = meta.unet_config();
    cfg.validate().map_err(TrainError::InvalidParameter)?;
    let model = cfg
        .init(device)
        .try_load_file(run_dir.join("model.bpk"))
        .map_err(|e| TrainError::Model(format!("muat bobot: {e:?}")))?;
    Ok((model.valid(), meta))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dice_per_class_basic() {
        let d = dice_per_class(&[0, 1, 1, 2], &[0, 1, 2, 2], 3);
        assert!((d[0] - 1.0).abs() < 1e-6);
        assert!((d[1] - 2.0 / 3.0).abs() < 1e-6);
        assert!((d[2] - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn lr_schedule_monotone_after_warmup() {
        let cfg = TrainConfig {
            steps: 1000,
            batch: 1,
            crop: 32,
            lr: 1e-3,
            seed: 0,
            holdout: vec![],
            base: 4,
            depth: 2,
            eval_every: 0,
            tumor_min_frac: 0.05,
            max_val_tiles: 0,
            aug: AugParamsSer::default(),
        };
        assert!(lr_at(&cfg, 0) < lr_at(&cfg, 10));
        assert!((lr_at(&cfg, 50) - 1e-3).abs() < 1e-9);
        assert!(lr_at(&cfg, 500) > lr_at(&cfg, 999));
        assert!(lr_at(&cfg, 999) >= 0.05e-3 - 1e-12);
    }

    #[test]
    fn derive_seed_is_distinct() {
        assert_ne!(derive_seed(1, 0, 0), derive_seed(1, 0, 1));
        assert_ne!(derive_seed(1, 0, 0), derive_seed(1, 1, 0));
        assert_eq!(derive_seed(1, 2, 3), derive_seed(1, 2, 3));
    }
}
