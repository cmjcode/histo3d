//! Bench EDT: 2D 512² dan edt3 anisotropik 128³.

use criterion::{criterion_group, criterion_main, Criterion};
use h3d_imgproc::edt::{edt3_anisotropic, edt_squared_2d};
use h3d_imgproc::phantom::{ellipsoid_mask, Rng};
use std::hint::black_box;

fn bench_edt(c: &mut Criterion) {
    let mask2 = ellipsoid_mask([512, 512], [256.0, 256.0], [150.0, 90.0]);
    c.bench_function("edt_squared_2d 512x512", |b| {
        b.iter(|| edt_squared_2d(black_box(&mask2)))
    });

    let n = 128usize;
    let mut rng = Rng::new(1);
    let mask3: Vec<u8> = (0..n * n * n)
        .map(|i| {
            let x = (i % n) as f64 - 64.0;
            let y = ((i / n) % n) as f64 - 64.0;
            let z = (i / (n * n)) as f64 - 64.0;
            u8::from(
                (x / 40.0).powi(2) + (y / 30.0).powi(2) + (z / 20.0).powi(2) <= 1.0
                    || rng.below(5000) == 0,
            )
        })
        .collect();
    c.bench_function("edt3_anisotropic 128^3", |b| {
        b.iter(|| edt3_anisotropic(black_box(&mask3), [n, n, n], [16.0, 16.0, 40.0]))
    });
}

criterion_group!(benches, bench_edt);
criterion_main!(benches);
