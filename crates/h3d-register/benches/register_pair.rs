//! Bench registrasi satu pasangan affine (sumber 16 µm, piramida 64→32→16).

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use h3d_core::{Affine2, NoProgress};
use h3d_imgproc::phantom::{render_histology, slice_ellipsoid};
use h3d_imgproc::{warp_affine_f32, warp_affine_u8, Interp};
use h3d_register::{BuiltinRegistrar, RegImage, RegOptions, Registrar};

fn make_pair() -> (RegImage, RegImage) {
    let mpp = 16.0;
    let n = 256;
    let mask = slice_ellipsoid([1000.0, 700.0, 600.0], 100.0, mpp, [n, n]);
    let rgba = render_histology(&mask, 1, mpp);
    let fixed = RegImage::from_rgba(&rgba, n, n, mpp, [0.0, 0.0], mpp).expect("fixed");
    let c = [n as f64 * mpp / 2.0; 2];
    // g: U_f → U_m (putar 20°, geser, skala 1,03).
    let g = Affine2::translation(c[0] + 150.0, c[1] - 90.0)
        .compose(&Affine2::rotation(20f64.to_radians()))
        .compose(&Affine2::scale(1.03, 1.03))
        .compose(&Affine2::translation(-c[0], -c[1]));
    let t_inv = fixed
        .um_to_px_affine()
        .compose(&g.inverse().expect("g"))
        .compose(&fixed.px_to_um_affine());
    let gray = warp_affine_f32(&fixed.gray, &t_inv, [n, n], Interp::Bilinear, 0.0).expect("warp");
    let mask_m = warp_affine_u8(&fixed.mask, &t_inv, [n, n], Interp::Nearest, 0).expect("warp");
    let moving = RegImage::new(gray, mask_m, mpp, [0.0, 0.0]).expect("moving");
    (fixed, moving)
}

fn bench_register(c: &mut Criterion) {
    let (fixed, moving) = make_pair();
    let opts = RegOptions::default();
    c.bench_function("register_pair affine 16um 256px", |b| {
        b.iter(|| {
            BuiltinRegistrar
                .register(
                    black_box(&fixed),
                    black_box(&moving),
                    None,
                    &opts,
                    &NoProgress,
                )
                .expect("registrasi")
        })
    });
}

criterion_group!(benches, bench_register);
criterion_main!(benches);
