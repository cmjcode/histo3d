//! Uji registrasi berpasangan: korelasi fase, pemulihan transformasi acak
//! (TRE), `from_rgba`, pembatalan.

mod common;

use std::sync::atomic::{AtomicBool, Ordering};

use common::{random_transform, textured_image, tre_um, warp_image, Field3};
use h3d_core::{Affine2, NoProgress, Progress};
use h3d_imgproc::phantom::{render_histology, slice_ellipsoid, Rng};
use h3d_register::{
    coarse_align, BuiltinRegistrar, RegError, RegImage, RegMetric, RegModel, RegOptions, Registrar,
};

/// Fantom: elips (irisan ellipsoid) pada kanvas `size_px` × `mpp`.
fn phantom(mpp: f64, size_px: usize, seed: u64, field_weight: f32) -> RegImage {
    let mask = slice_ellipsoid([1000.0, 700.0, 600.0], 100.0, mpp, [size_px, size_px]);
    let field = Field3::new(seed ^ 0xA5A5, 900.0, 300.0, 24);
    textured_image(&mask, mpp, 100.0, &field, seed, [0.0, 0.0], field_weight)
}

#[test]
fn phase_correlation_r2_finds_integer_shift_exactly() {
    // Pada level 64 µm: fixed dan moving identik kecuali geseran bulat.
    let mpp = 64.0;
    let n = 64;
    let fixed = phantom(mpp, n, 7, 0.6);
    let shift_px = [5.0, -3.0];
    let p_true = Affine2::translation(shift_px[0] * mpp, shift_px[1] * mpp); // U_m → U_f
                                                                             // moving(u_m) = fixed(P u_m) → bentuk moving dari fixed dengan P⁻¹ (U_f → U_m).
    let moving = warp_image(&fixed, &p_true.inverse().unwrap(), [n, n], mpp, [0.0, 0.0]);
    // Kandidat awal: identitas (tanpa R1), supaya hanya R2 yang diuji.
    let c = coarse_align(&fixed, &moving, Some(Affine2::IDENTITY), false).unwrap();
    // "Persis": dibulatkan ke piksel = geseran bulat yang benar, dan sisa
    // sub-piksel (dari jendela Hann) di bawah 0,01 piksel.
    let t = c.p_um.translation_part();
    let t_px = [t[0] / mpp, t[1] / mpp];
    assert_eq!(
        [t_px[0].round(), t_px[1].round()],
        shift_px,
        "geseran {t_px:?} px"
    );
    assert!(
        (t_px[0] - shift_px[0]).abs() < 0.01 && (t_px[1] - shift_px[1]).abs() < 0.01,
        "geseran {t_px:?} px ≠ {shift_px:?}"
    );
    assert!(c
        .p_um
        .linear_part()
        .iter()
        .zip([1.0, 0.0, 0.0, 1.0])
        .all(|(a, b)| (a - b).abs() < 1e-9));
}

/// Jalankan satu benih; kembalikan TRE dalam piksel registrasi (16 µm).
fn run_seed(seed: u64, opts: &RegOptions) -> (f64, h3d_register::RegResult) {
    let mpp = 8.0;
    let n = 512;
    let width_um = n as f64 * mpp;
    let fixed = phantom(mpp, n, seed, 0.5);
    let mut rng = Rng::new(seed * 7919 + 1);
    let center = [width_um / 2.0, width_um / 2.0];
    let (g, _) = random_transform(&mut rng, center, width_um, 0.2, true, (0.95, 1.05));
    // g: U_f → U_m. P_true (U_m → U_f) = g⁻¹.
    let moving = warp_image(&fixed, &g, [n, n], mpp, [0.0, 0.0]);
    let p_true = g.inverse().unwrap();
    let res = BuiltinRegistrar
        .register(&fixed, &moving, None, opts, &NoProgress)
        .expect("registrasi");
    let fine = opts.levels_um.iter().cloned().fold(f64::INFINITY, f64::min);
    (tre_um(&moving, &res.transform, &p_true, 4) / fine, res)
}

#[test]
fn recovers_random_affine_transform_on_phantom() {
    let opts = RegOptions::default();
    let seeds: Vec<u64> = (1..=10).collect();
    let mut tres = Vec::new();
    for &s in &seeds {
        let (tre_px, res) = run_seed(s, &opts);
        eprintln!(
            "benih {s}: TRE {tre_px:.3} px, ncc {:.3}, dice {:.3}, area {:.3}, iter {}",
            res.ncc, res.mask_dice, res.area_ratio, res.iterations
        );
        tres.push(tre_px);
    }
    let mean = tres.iter().sum::<f64>() / tres.len() as f64;
    let max = tres.iter().cloned().fold(0.0, f64::max);
    eprintln!("TRE rata-rata {mean:.3} px, maks {max:.3} px");
    assert!(mean <= 2.0, "TRE rata-rata {mean:.3} px > 2 px: {tres:?}");
}

#[test]
fn rigid_and_similarity_models_recover_rigid_transform() {
    for model in [RegModel::Rigid, RegModel::Similarity] {
        let opts = RegOptions {
            model,
            ..Default::default()
        };
        let mpp = 8.0;
        let n = 512;
        let width_um = n as f64 * mpp;
        let fixed = phantom(mpp, n, 21, 0.5);
        let mut rng = Rng::new(99);
        let center = [width_um / 2.0, width_um / 2.0];
        let (g, _) = random_transform(&mut rng, center, width_um, 0.15, true, (1.0, 1.0));
        let moving = warp_image(&fixed, &g, [n, n], mpp, [0.0, 0.0]);
        let p_true = g.inverse().unwrap();
        let res = BuiltinRegistrar
            .register(&fixed, &moving, None, &opts, &NoProgress)
            .unwrap();
        let tre_px = tre_um(&moving, &res.transform, &p_true, 4) / 16.0;
        eprintln!("{model:?}: TRE {tre_px:.3} px");
        assert!(tre_px <= 2.0, "{model:?}: TRE {tre_px:.3}");
        let parts = res.transform.decompose();
        assert_eq!(parts.mirrored, g.determinant() < 0.0);
        if model == RegModel::Rigid {
            assert!((parts.scale[0] - 1.0).abs() < 1e-6 && (parts.scale[1] - 1.0).abs() < 1e-6);
        }
    }
}

#[test]
fn mutual_information_metric_recovers_transform() {
    let opts = RegOptions {
        metric: RegMetric::MutualInformation,
        model: RegModel::Rigid,
        levels_um: vec![64.0, 32.0],
        max_iterations: 60,
        ..Default::default()
    };
    let mpp = 16.0;
    let n = 256;
    let width_um = n as f64 * mpp;
    let fixed = phantom(mpp, n, 5, 0.6);
    let mut rng = Rng::new(555);
    let center = [width_um / 2.0, width_um / 2.0];
    let (g, _) = random_transform(&mut rng, center, width_um, 0.1, false, (1.0, 1.0));
    let moving = warp_image(&fixed, &g, [n, n], mpp, [0.0, 0.0]);
    let p_true = g.inverse().unwrap();
    let res = BuiltinRegistrar
        .register(&fixed, &moving, None, &opts, &NoProgress)
        .unwrap();
    let tre_px = tre_um(&moving, &res.transform, &p_true, 2) / 32.0;
    eprintln!("MI: TRE {tre_px:.3} px (32 µm)");
    assert!(tre_px <= 2.0, "MI: TRE {tre_px:.3}");
}

#[test]
fn init_is_respected_and_refined() {
    let mpp = 16.0;
    let n = 256;
    let width_um = n as f64 * mpp;
    let fixed = phantom(mpp, n, 3, 0.5);
    let mut rng = Rng::new(42);
    let center = [width_um / 2.0, width_um / 2.0];
    let (g, _) = random_transform(&mut rng, center, width_um, 0.1, true, (1.0, 1.0));
    let moving = warp_image(&fixed, &g, [n, n], mpp, [0.0, 0.0]);
    let p_true = g.inverse().unwrap();
    // Tebakan awal sedikit meleset (geser 40 µm, putar 2°).
    let init = Affine2::translation(40.0, -30.0)
        .compose(&Affine2::rotation_about(2f64.to_radians(), center))
        .compose(&p_true);
    let opts = RegOptions {
        allow_flip: false,
        ..Default::default()
    };
    let res = BuiltinRegistrar
        .register(&fixed, &moving, Some(init), &opts, &NoProgress)
        .unwrap();
    let tre_px = tre_um(&moving, &res.transform, &p_true, 2) / 16.0;
    assert!(tre_px <= 2.0, "init: TRE {tre_px:.3}");
}

#[test]
fn from_rgba_builds_mask_and_gray_in_range() {
    let mpp = 8.0;
    let n = 256;
    let mask = slice_ellipsoid([800.0, 500.0, 400.0], 0.0, mpp, [n, n]);
    let rgba = render_histology(&mask, 11, mpp);
    let img = RegImage::from_rgba(&rgba, n, n, mpp, [10.0, 20.0], 32.0).unwrap();
    assert_eq!(img.mpp, 32.0);
    assert_eq!(img.size_px(), [64, 64]);
    assert_eq!(img.origin_um, [10.0, 20.0]);
    assert!(img.gray.iter().all(|v| (0.0..=1.0).contains(v)));
    let area_src = mask.iter().filter(|v| **v != 0).count() as f64 * mpp * mpp;
    let area_dst = img.mask.iter().filter(|v| **v != 0).count() as f64 * 32.0 * 32.0;
    assert!(
        (area_dst / area_src - 1.0).abs() < 0.1,
        "luas {area_dst} vs {area_src}"
    );
    // Konversi koordinat bolak-balik dan pusat piksel +0,5.
    let u = img.px_to_um([0.5, 0.5]);
    assert_eq!(u, [10.0 + 16.0, 20.0 + 16.0]);
    let p = img.um_to_px(u);
    assert!((p[0] - 0.5).abs() < 1e-12 && (p[1] - 0.5).abs() < 1e-12);
    assert!(RegImage::from_rgba(&rgba[..10], n, n, mpp, [0.0; 2], 32.0).is_err());
    assert!(RegImage::from_rgba(&rgba, n, n, 0.0, [0.0; 2], 32.0).is_err());
}

#[test]
fn empty_mask_is_an_error() {
    let fixed = phantom(32.0, 128, 1, 0.5);
    let empty = RegImage::new(
        fixed.gray.clone(),
        ndarray::Array2::zeros(fixed.gray.dim()),
        32.0,
        [0.0, 0.0],
    )
    .unwrap();
    let r = BuiltinRegistrar.register(&fixed, &empty, None, &RegOptions::default(), &NoProgress);
    assert!(matches!(r, Err(RegError::EmptyMask { .. })), "{r:?}");
}

struct CancelAfterFirst(AtomicBool);

impl Progress for CancelAfterFirst {
    fn set(&self, _f: f32, _s: &str) {
        self.0.store(true, Ordering::SeqCst);
    }
    fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

#[test]
fn cancellation_returns_cancelled() {
    let fixed = phantom(32.0, 128, 1, 0.5);
    let moving = fixed.clone();
    let p = CancelAfterFirst(AtomicBool::new(false));
    let r = BuiltinRegistrar.register(&fixed, &moving, None, &RegOptions::default(), &p);
    assert_eq!(r, Err(RegError::Cancelled));
}
