//! CLI `h3d-train`: `tiles` → `train` → `export` → `eval`.
//! Pesan pengguna berbahasa Indonesia.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

use h3d_train::data::{cut_dataset, scan, CutParams, TileIndex};
use h3d_train::eval::eval_slide;
use h3d_train::export_onnx::write_model_dir;
use h3d_train::train::{load_checkpoint, train, AugParamsSer, TrainConfig};
use h3d_train::TrainError;

#[derive(Parser)]
#[command(
    name = "h3d-train",
    about = "Pelatihan model segmentasi H&E HISTO3D (Rust/burn)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Potong tile berlabel dari dataset ke cache.
    Tiles {
        /// Folder dataset (berisi `<stack>/Images_sub4`, `Masks_sub4`, `TumorMasks_sub4`).
        #[arg(long)]
        root: PathBuf,
        /// Folder cache keluaran.
        #[arg(long)]
        cache: PathBuf,
        /// Sisi tile (piksel sub4).
        #[arg(long, default_value_t = 512)]
        tile: usize,
        /// Langkah jendela.
        #[arg(long, default_value_t = 384)]
        stride: usize,
        /// Fraksi jaringan minimum.
        #[arg(long, default_value_t = 0.2)]
        min_tissue: f32,
        /// mpp level 0 (µm/piksel). Tidak ditebak: wajib diisi dari sumber resmi.
        #[arg(long)]
        mpp0: f64,
        /// Sumber nilai mpp (dicatat di index.json dan model).
        #[arg(long)]
        mpp_source: String,
        /// Batasi ke tumpukan tertentu (bisa diulang; kosong = semua).
        #[arg(long)]
        stack: Vec<String>,
    },
    /// Latih U-Net dari cache tile.
    Train {
        /// Folder cache (`index.json` + `tiles/`).
        #[arg(long)]
        cache: PathBuf,
        /// Folder keluaran run.
        #[arg(long)]
        out: PathBuf,
        /// Tumpukan holdout (bisa diulang).
        #[arg(long)]
        holdout: Vec<String>,
        /// Jumlah langkah.
        #[arg(long, default_value_t = 3000)]
        steps: usize,
        /// Ukuran batch.
        #[arg(long, default_value_t = 8)]
        batch: usize,
        /// Sisi potongan latih.
        #[arg(long, default_value_t = 256)]
        crop: usize,
        /// LR puncak.
        #[arg(long, default_value_t = 1e-3)]
        lr: f64,
        /// Benih.
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Kanal dasar U-Net.
        #[arg(long, default_value_t = 32)]
        base: usize,
        /// Kedalaman U-Net.
        #[arg(long, default_value_t = 4)]
        depth: usize,
        /// Validasi tiap N langkah.
        #[arg(long, default_value_t = 250)]
        eval_every: usize,
        /// Jumlah tile validasi maksimum (0 = semua).
        #[arg(long, default_value_t = 200)]
        max_val_tiles: usize,
        /// Fraksi tumor minimum agar tile masuk kelompok "tumor" saat penyeimbangan batch.
        #[arg(long, default_value_t = 0.005)]
        tumor_min_frac: f32,
        /// Perangkat: `auto` (GPU bila ada), `metal` (Metal langsung), atau `cpu`.
        #[arg(long, default_value = "auto")]
        device: String,
    },
    /// Ekspor run ke `model.onnx` + `model.json`.
    Export {
        /// Folder run (`model.bpk` + `train_meta.json`).
        #[arg(long)]
        run: PathBuf,
        /// Folder model keluaran.
        #[arg(long)]
        out: PathBuf,
        /// Nama model.
        #[arg(long, default_value = "tumor-he-v1")]
        name: String,
        /// Versi model.
        #[arg(long, default_value = "0.1.0")]
        version: String,
        /// Sisi tile inferensi.
        #[arg(long, default_value_t = 512)]
        input_size: usize,
    },
    /// Evaluasi folder model pada irisan tumpukan holdout (Dice vs masker manual).
    Eval {
        /// Folder model (`model.json` + `model.onnx`).
        #[arg(long)]
        model: PathBuf,
        /// Folder dataset.
        #[arg(long)]
        root: PathBuf,
        /// Tumpukan yang dievaluasi (bisa diulang).
        #[arg(long)]
        stack: Vec<String>,
        /// mpp gambar sub4 (= mpp level 0 × 4); harus sama dengan mpp model.
        #[arg(long)]
        mpp: f64,
        /// Keluaran JSON.
        #[arg(long)]
        json: bool,
    },
}

fn run(cmd: Command) -> Result<(), TrainError> {
    let log = |s: &str| eprintln!("{s}");
    match cmd {
        Command::Tiles {
            root,
            cache,
            tile,
            stride,
            min_tissue,
            mpp0,
            mpp_source,
            stack,
        } => {
            let slides: Vec<_> = scan(&root)?
                .into_iter()
                .filter(|s| stack.is_empty() || stack.contains(&s.stack))
                .collect();
            log(&format!("{} irisan ditemukan", slides.len()));
            let idx = cut_dataset(
                &slides,
                &cache,
                &CutParams {
                    tile,
                    stride,
                    min_tissue,
                    mpp0,
                    mpp_source,
                },
                &log,
            )?;
            let tumor = idx.tiles.iter().filter(|t| t.tumor_frac > 0.0).count();
            log(&format!(
                "{} tile ({tumor} memuat tumor), mpp tile {:.4}",
                idx.tiles.len(),
                idx.mpp
            ));
        }
        Command::Train {
            cache,
            out,
            holdout,
            steps,
            batch,
            crop,
            lr,
            seed,
            base,
            depth,
            eval_every,
            max_val_tiles,
            tumor_min_frac,
            device,
        } => {
            let index = TileIndex::load(&cache.join("index.json"))?;
            let cfg = TrainConfig {
                steps,
                batch,
                crop,
                lr,
                seed,
                holdout,
                base,
                depth,
                eval_every,
                tumor_min_frac,
                max_val_tiles,
                aug: AugParamsSer::default(),
            };
            let dev = match device.as_str() {
                "auto" => burn::tensor::Device::default(),
                "cpu" => burn::tensor::Device::flex(),
                "metal" => burn::tensor::Device::metal(burn::tensor::DeviceKind::DefaultDevice),
                other => {
                    return Err(TrainError::InvalidParameter(format!(
                        "perangkat '{other}' tidak dikenal (auto|metal|cpu)"
                    )))
                }
            }
            .autodiff();
            log(&format!("perangkat: {dev:?}"));
            train(&index, &cache, &cfg, &out, &dev, &log)?;
            log(&format!("selesai; run di {}", out.display()));
        }
        Command::Export {
            run,
            out,
            name,
            version,
            input_size,
        } => {
            let dev = burn::tensor::Device::flex();
            let (model, meta) = load_checkpoint(&run, &dev)?;
            let spec = write_model_dir(&model, &meta, &name, &version, input_size, &out)?;
            log(&format!(
                "ditulis {}: {} kelas, input {}, mpp {:.4}",
                out.display(),
                spec.classes.len(),
                spec.input_size,
                spec.mpp
            ));
        }
        Command::Eval {
            model,
            root,
            stack,
            mpp,
            json,
        } => {
            let seg = h3d_segment::TractSegmenter::load(&model)?;
            let slides: Vec<_> = scan(&root)?
                .into_iter()
                .filter(|s| stack.is_empty() || stack.contains(&s.stack))
                .collect();
            if slides.is_empty() {
                return Err(TrainError::Dataset("tidak ada irisan yang cocok".into()));
            }
            let mut results = Vec::new();
            for s in &slides {
                let r = eval_slide(&seg, s, mpp)?;
                if !json {
                    log(&format!(
                        "{}/im{}: Dice jaringan {:.3}, tumor {}",
                        r.stack,
                        r.slide,
                        r.dice_tissue,
                        r.dice_tumor.map_or("-".to_string(), |d| format!("{d:.3}"))
                    ));
                }
                results.push(r);
            }
            let n = results.len() as f32;
            let mean_tissue = results.iter().map(|r| r.dice_tissue).sum::<f32>() / n;
            let tumor: Vec<f32> = results.iter().filter_map(|r| r.dice_tumor).collect();
            let mean_tumor =
                (!tumor.is_empty()).then(|| tumor.iter().sum::<f32>() / tumor.len() as f32);
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "slides": results,
                        "mean_dice_tissue": mean_tissue,
                        "mean_dice_tumor": mean_tumor,
                    }))?
                );
            } else {
                log(&format!(
                    "rata-rata Dice jaringan {mean_tissue:.3}, tumor {}",
                    mean_tumor.map_or("-".to_string(), |d| format!("{d:.3}"))
                ));
            }
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    env_logger::init();
    let cli = Cli::parse();
    match run(cli.command) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("galat: {e}");
            ExitCode::from(1)
        }
    }
}
