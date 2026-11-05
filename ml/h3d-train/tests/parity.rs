//! Tes paritas dan loop latih (histo3d.md H7, versi Rust):
//! 1. U-Net burn (CPU) vs ONNX hasil ekspor yang dijalankan tract: logit sama.
//! 2. Loop latih pada data sintetis mencapai Dice tinggi (membuktikan loss,
//!    optimizer, dan pipeline data benar).

use burn::module::Module;
use burn::optim::{AdamWConfig, GradientsParams};
use burn::tensor::{Device, Tensor, TensorData};
use h3d_segment::{Segmenter, TractSegmenter};
use h3d_train::data::{Tile, N_CLASSES};
use h3d_train::export_onnx::{export_unet, model_spec};
use h3d_train::model::UNetConfig;
use h3d_train::train::{
    argmax_cpu, dice_per_class, seg_loss, to_input, to_target, TrainMeta, MEAN, STD,
};

fn meta(base: usize, depth: usize) -> TrainMeta {
    TrainMeta {
        base,
        depth,
        n_classes: N_CLASSES,
        classes: vec![None, Some("Jaringan".into()), Some("Tumor".into())],
        mpp: 2.0,
        mpp_source: "uji".into(),
        mean: MEAN,
        std: STD,
        best_val_dice: vec![],
        best_step: 0,
        holdout: vec![],
    }
}

#[test]
fn burn_and_tract_agree_on_exported_onnx() {
    let device = Device::flex();
    device.seed(3);
    let s = 32usize;
    let model = UNetConfig::new()
        .with_base(4)
        .with_depth(2)
        .init(&device)
        .valid();
    let onnx = export_unet(&model, s).expect("ekspor");
    let spec = model_spec(&meta(4, 2), "uji", "0", s);
    let seg = TractSegmenter::from_bytes(spec, &onnx).expect("tract memuat ONNX");

    // Masukan deterministik.
    let input: Vec<f32> = (0..3 * s * s)
        .map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0)
        .collect();
    let x = Tensor::<4>::from_data(TensorData::new(input.clone(), [1, 3, s, s]), &device);
    let burn_out = model.forward(x).into_data_as::<f32>();
    let burn_out = burn_out.as_slice::<f32>().expect("f32");
    let tract_out = seg.infer(&input).expect("tract infer");
    assert_eq!(burn_out.len(), tract_out.len());
    let max_diff = burn_out
        .iter()
        .zip(&tract_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_abs = burn_out.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(max_abs > 1e-3, "keluaran nol, tes tidak bermakna");
    assert!(max_diff < 1e-3, "selisih maksimum {max_diff}");
}

/// Data sintetis: latar putih; "jaringan" merah muda; "tumor" ungu gelap.
fn synthetic_tile(seed: u64, s: usize) -> Tile {
    let mut rgb = Vec::with_capacity(s * s * 3);
    let mut cls = Vec::with_capacity(s * s);
    let cx = (seed % 7) as i64 + 8;
    let cy = (seed % 5) as i64 + 8;
    for y in 0..s as i64 {
        for x in 0..s as i64 {
            let d2 = (x - cx).pow(2) + (y - cy).pow(2);
            let (c, px) = if d2 < 25 {
                (2u8, [90u8, 40, 120])
            } else if x < s as i64 / 2 + (seed % 3) as i64 {
                (1u8, [230u8, 170, 190])
            } else {
                (0u8, [250u8, 250, 250])
            };
            let n = ((x * 31 + y * 17 + seed as i64) % 11) as u8;
            rgb.extend(px.iter().map(|&v| v.saturating_sub(n)));
            cls.push(c);
        }
    }
    Tile { size: s, rgb, cls }
}

#[test]
fn training_loop_learns_synthetic_segmentation() {
    let device = Device::flex().autodiff();
    device.seed(5);
    let s = 32usize;
    let mut model = UNetConfig::new()
        .with_base(4)
        .with_depth(2)
        .init(&device)
        .train();
    let mut optim = AdamWConfig::new().init();
    let mut last_loss = f32::INFINITY;
    for step in 0..320 {
        let batch: Vec<Tile> = (0..4).map(|i| synthetic_tile(step * 4 + i, s)).collect();
        let x = to_input(&batch, &device).expect("x");
        let y = to_target(&batch, &device).expect("y");
        let loss = seg_loss(model.forward(x), y, N_CLASSES);
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(3e-3, model, grads);
        last_loss = loss.into_data_as::<f32>().as_slice::<f32>().expect("f32")[0];
    }
    let m = model.valid();
    let test: Vec<Tile> = (100..104).map(|i| synthetic_tile(i, s)).collect();
    let pred =
        argmax_cpu(m.forward(to_input(&test, &device).expect("x")).detach()).expect("argmax");
    let tgt: Vec<u8> = test.iter().flat_map(|t| t.cls.clone()).collect();
    let dice = dice_per_class(&pred, &tgt, N_CLASSES);
    assert!(last_loss.is_finite(), "loss tidak hingga");
    assert!(
        dice[1] > 0.8 && dice[2] > 0.5,
        "Dice {dice:?} (loss {last_loss})"
    );
}
