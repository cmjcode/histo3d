//! Bench surface nets: ekstraksi SDF bola 128³ + penghalusan Taubin.

use criterion::{criterion_group, criterion_main, Criterion};
use h3d_core::{LabelId, NoProgress, ReconConfig};
use h3d_volume::{extract_mesh, SphereSdf, StackGrid};
use std::hint::black_box;

fn bench_surface_nets(c: &mut Criterion) {
    let n = 128u32;
    let s = 16.0;
    let grid = StackGrid {
        origin_um: [0.0, 0.0],
        s_xy: s,
        size: [n, n],
        z_um: (0..n).map(|k| (k as f64 + 0.5) * s).collect(),
    };
    let c0 = n as f64 * s / 2.0;
    let sdf = SphereSdf {
        grid,
        center_um: [c0, c0, c0],
        radius_um: 0.4 * n as f64 * s,
    };
    // Tanpa decimate: hanya ekstraksi + Taubin 10 iterasi.
    let cfg = ReconConfig {
        decimate_error_factor: 0.0,
        ..Default::default()
    };
    let mut group = c.benchmark_group("surface_nets");
    group.sample_size(10);
    group.bench_function("sphere 128^3 extract + taubin", |b| {
        b.iter(|| extract_mesh(black_box(&sdf), LabelId(1), &cfg, &NoProgress))
    });
    group.finish();
}

criterion_group!(benches, bench_surface_nets);
criterion_main!(benches);
