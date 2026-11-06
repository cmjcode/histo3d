//! Uji ujung-ke-ujung rekonstruksi dan pengukuran (histo3d.md H5) atas
//! fantom analitik.

use h3d_core::{Budgets, NoProgress, ReconConfig, SlideId, SlideTransform};
use h3d_volume::phantom3d::LABEL_TUMOR;
use h3d_volume::{
    cavalieri, cavalieri_areas, generate_phantom, measure_label, plan_grid, reconstruct_label,
    ArrayLabelSampler, LabelMesh, Phantom, PhantomShape, PhantomSpec, SliceInput, VoxelMask,
    Warning,
};

static IDENTITY: SlideTransform = SlideTransform::Identity;

fn samplers(p: &Phantom) -> Vec<ArrayLabelSampler> {
    p.slices
        .iter()
        .map(|s| ArrayLabelSampler::from_slice(s, p.spec.s_xy))
        .collect()
}

fn inputs<'a>(p: &Phantom, smp: &'a [ArrayLabelSampler]) -> Vec<SliceInput<'a>> {
    p.slices
        .iter()
        .zip(smp)
        .enumerate()
        .map(|(i, (s, sm))| SliceInput {
            id: SlideId(i as u32),
            z_um: s.z_um,
            transform: &IDENTITY,
            labels: sm,
            annotated_labels: sm.annotated_labels(),
            reviewed_labels: Vec::new(),
        })
        .collect()
}

fn recon(p: &Phantom, cfg: &ReconConfig) -> (VoxelMask, LabelMesh) {
    let smp = samplers(p);
    let inp = inputs(p, &smp);
    let plan = plan_grid(&inp, &[LABEL_TUMOR], cfg, &Budgets::MACOS).expect("plan");
    reconstruct_label(&plan, &inp, LABEL_TUMOR, cfg, &NoProgress).expect("recon")
}

fn ellipsoid_spec() -> PhantomSpec {
    PhantomSpec {
        shape: PhantomShape::Ellipsoid {
            radii: [1500.0, 1200.0, 1000.0],
        },
        n_slices: 23,
        spacing_um: 100.0, // 1/20 diameter Z
        s_xy: 16.0,
        size_um: [3840.0, 3840.0],
        ..Default::default()
    }
}

#[test]
fn ellipsoid_volume_closed_positive_one_component() {
    let p = generate_phantom(&ellipsoid_spec());
    let cfg = ReconConfig::default();
    let smp = samplers(&p);
    let inp = inputs(&p, &smp);
    let truth = p.truth.volume_um3;

    let v_cav = cavalieri(&cavalieri_areas(&inp, LABEL_TUMOR));
    let e_cav = (v_cav - truth).abs() / truth;
    let (mask, mesh) = recon(&p, &cfg);
    let v_mesh = mesh.signed_volume_um3();
    let e_mesh = (v_mesh - truth).abs() / truth;
    let e_vox = (mask.volume_um3() - truth).abs() / truth;
    println!(
        "ellipsoid: cavalieri {:.3} %, mesh {:.3} %, voxel {:.3} %, tris {}",
        e_cav * 100.0,
        e_mesh * 100.0,
        e_vox * 100.0,
        mesh.triangle_count()
    );
    assert!(e_cav <= 0.02, "galat Cavalieri {e_cav}");
    assert!(v_mesh > 0.0, "volume bertanda harus positif");
    assert!(e_mesh <= 0.03, "galat volume mesh {e_mesh}");
    assert!(mesh.is_closed(), "mesh harus tertutup");
    assert_eq!(mesh.connected_components(), 1);
    assert_eq!(mesh.euler_characteristic(), 2);
    assert_eq!(mesh.normals.len(), mesh.positions_um.len());

    let m = measure_label(&mesh, &inp, LABEL_TUMOR, Some(1.1));
    assert_eq!(m.n_slices_annotated, 19);
    assert_eq!(m.n_components, 1);
    assert!(!m
        .warnings
        .iter()
        .any(|w| matches!(w, Warning::VolumeMismatch { .. })));
    let corrected = m.volume_corrected_um3.expect("koreksi susut");
    assert!((corrected - m.volume_cavalieri_um3 * 1.331).abs() < 1e-3 * corrected);
    assert!(
        (m.max_diameter_um - 3000.0).abs() < 100.0,
        "{}",
        m.max_diameter_um
    );
}

#[test]
fn torus_has_genus_one() {
    let p = generate_phantom(&PhantomSpec {
        shape: PhantomShape::Torus {
            major: 1000.0,
            minor: 400.0,
        },
        n_slices: 21,
        spacing_um: 50.0,
        s_xy: 16.0,
        size_um: [3840.0, 3840.0],
        ..Default::default()
    });
    let (_, mesh) = recon(&p, &ReconConfig::default());
    assert!(mesh.is_closed());
    assert_eq!(mesh.connected_components(), 1);
    assert_eq!(mesh.euler_characteristic(), 0, "torus: genus 1");
    let truth = p.truth.volume_um3;
    let e = (mesh.signed_volume_um3() - truth).abs() / truth;
    println!("torus: galat volume mesh {:.3} %", e * 100.0);
    assert!(e < 0.05, "{e}");
}

#[test]
fn branch_is_one_component() {
    let p = generate_phantom(&PhantomSpec {
        shape: PhantomShape::Branch {
            trunk_len: 800.0,
            arm_len: 900.0,
            radius: 200.0,
            half_angle: 0.6,
        },
        n_slices: 45,
        spacing_um: 50.0,
        s_xy: 16.0,
        size_um: [2560.0, 1280.0],
        ..Default::default()
    });
    let (_, mesh) = recon(&p, &ReconConfig::default());
    assert!(mesh.is_closed());
    assert_eq!(mesh.connected_components(), 1);
    assert!(mesh.signed_volume_um3() > 0.0);
}

fn mesh_bytes(m: &LabelMesh) -> Vec<u8> {
    let mut b = Vec::new();
    for p in &m.positions_um {
        for c in p {
            b.extend_from_slice(&c.to_le_bytes());
        }
    }
    for n in &m.normals {
        for c in n {
            b.extend_from_slice(&c.to_le_bytes());
        }
    }
    for i in &m.indices {
        b.extend_from_slice(&i.to_le_bytes());
    }
    b
}

#[test]
fn reconstruction_is_byte_identical() {
    let p = generate_phantom(&ellipsoid_spec());
    let cfg = ReconConfig::default();
    let (mask_a, mesh_a) = recon(&p, &cfg);
    let (mask_b, mesh_b) = recon(&p, &cfg);
    assert_eq!(mask_a.data, mask_b.data);
    assert!(!mesh_a.indices.is_empty());
    assert_eq!(mesh_bytes(&mesh_a), mesh_bytes(&mesh_b));
}

#[test]
fn plan_grid_raises_s_xy_for_small_budget() {
    let p = generate_phantom(&ellipsoid_spec());
    let smp = samplers(&p);
    let inp = inputs(&p, &smp);
    let cfg = ReconConfig::default();
    let full = plan_grid(&inp, &[LABEL_TUMOR], &cfg, &Budgets::MACOS).expect("plan");
    assert_eq!(full.s_xy, 16.0);
    assert!(full.notes.is_empty());
    // s_z ≤ k × s_xy pada semua lapis.
    // Tutup ujung berada pada setengah jarak irisan; lapis di antaranya
    // memenuhi batas s_z.
    let nz = full.grid.nz();
    assert!(full.grid.z_um[1..nz - 1]
        .windows(2)
        .all(|w| w[1] - w[0] <= cfg.z_factor_max * full.s_xy + 1e-9));

    let small = Budgets {
        recon_working_set_bytes: full.estimated_bytes / 3,
        ..Budgets::MACOS
    };
    let plan = plan_grid(&inp, &[LABEL_TUMOR], &cfg, &small).expect("plan kecil");
    assert!(plan.s_xy > 16.0, "s_xy harus naik, dapat {}", plan.s_xy);
    assert!(plan.estimated_bytes <= small.recon_working_set_bytes);
    assert!(plan.notes.iter().any(|n| n.contains("s_xy dinaikkan")));
    assert!(plan
        .warnings
        .iter()
        .any(|w| matches!(w, Warning::SxyRaised { from, .. } if *from == 16.0)));

    let tiny = Budgets {
        recon_working_set_bytes: 1024,
        ..Budgets::MACOS
    };
    assert!(matches!(
        plan_grid(&inp, &[LABEL_TUMOR], &cfg, &tiny),
        Err(h3d_volume::VolError::GridTooLarge { .. })
    ));
    assert!(matches!(
        plan_grid(&inp, &[200], &cfg, &Budgets::MACOS),
        Err(h3d_volume::VolError::NoLabelledSlices)
    ));
}

#[test]
fn large_gap_splits_object() {
    // Buang irisan tengah sehingga celah > 5 × jarak median.
    let mut p = generate_phantom(&ellipsoid_spec());
    p.slices.retain(|s| s.z_um.abs() > 350.0);
    let smp = samplers(&p);
    let inp = inputs(&p, &smp);
    let cfg = ReconConfig::default();
    let plan = plan_grid(&inp, &[LABEL_TUMOR], &cfg, &Budgets::MACOS).expect("plan");
    assert!(plan
        .warnings
        .iter()
        .any(|w| matches!(w, Warning::GapSplit { .. })));
    assert!(!plan.notes.is_empty());
    let (_, mesh) = reconstruct_label(&plan, &inp, LABEL_TUMOR, &cfg, &NoProgress).expect("recon");
    assert!(mesh.is_closed());
    assert_eq!(
        mesh.connected_components(),
        2,
        "objek harus terputus di celah"
    );
}

#[test]
fn rotated_slide_is_resampled_backward() {
    // Irisan diputar 90° di U; T (U → W) memutar balik. Hasil di W harus
    // sama dengan fantom tak diputar.
    use h3d_core::Affine2;
    let p = generate_phantom(&ellipsoid_spec());
    let c = [1920.0, 1920.0];
    let rot = SlideTransform::Affine(Affine2::rotation_about(std::f64::consts::FRAC_PI_2, c));
    let smp = samplers(&p);
    let cfg = ReconConfig::default();
    let plain = inputs(&p, &smp);
    let plan = plan_grid(&plain, &[LABEL_TUMOR], &cfg, &Budgets::MACOS).expect("plan");
    let m0 = h3d_volume::build_voxel_mask(&plan, &plain, LABEL_TUMOR, &cfg, &NoProgress).unwrap();
    let mut turned = inputs(&p, &smp);
    for s in &mut turned {
        s.transform = &rot;
    }
    let plan = plan_grid(&turned, &[LABEL_TUMOR], &cfg, &Budgets::MACOS).expect("plan putar");
    let m1 = h3d_volume::build_voxel_mask(&plan, &turned, LABEL_TUMOR, &cfg, &NoProgress).unwrap();
    // Di W, ellipsoid tampak bertukar sumbu: rx ↔ ry.
    let extent = |m: &VoxelMask, axis: usize| -> usize {
        let g = &m.grid;
        let k = g
            .z_um
            .iter()
            .position(|z| z.abs() < 1e-6)
            .expect("lapis z=0");
        let [nx, ny, _] = g.dims();
        let (na, nb) = if axis == 0 { (nx, ny) } else { (ny, nx) };
        (0..na)
            .filter(|&a| {
                (0..nb).any(|b| {
                    let (x, y) = if axis == 0 { (a, b) } else { (b, a) };
                    m.get(x, y, k) != 0
                })
            })
            .count()
    };
    let (ex0, ey0) = (extent(&m0, 0), extent(&m0, 1));
    let (ex1, ey1) = (extent(&m1, 0), extent(&m1, 1));
    assert!(ex0 > ey0);
    assert!(
        (ex1 as i64 - ey0 as i64).abs() <= 1 && (ey1 as i64 - ex0 as i64).abs() <= 1,
        "{ex0}x{ey0} → {ex1}x{ey1}"
    );
}

/// Rata-rata sudut (rad) antar normal muka yang berbagi sisi: ukuran
/// kekasaran "tangga" permukaan.
fn mean_dihedral(m: &LabelMesh) -> f64 {
    use std::collections::HashMap;
    let p = &m.positions_um;
    let face_n: Vec<[f64; 3]> = m
        .indices
        .chunks(3)
        .map(|t| {
            let (a, b, c) = (p[t[0] as usize], p[t[1] as usize], p[t[2] as usize]);
            let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let n = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-30);
            [n[0] / l, n[1] / l, n[2] / l]
        })
        .collect();
    let mut by_edge: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (f, t) in m.indices.chunks(3).enumerate() {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            by_edge.entry((a.min(b), a.max(b))).or_default().push(f);
        }
    }
    let mut sum = 0.0;
    let mut n = 0usize;
    let mut keys: Vec<_> = by_edge.keys().copied().collect();
    keys.sort_unstable();
    for k in keys {
        let fs = &by_edge[&k];
        if fs.len() == 2 {
            let (x, y) = (face_n[fs[0]], face_n[fs[1]]);
            let d = (x[0] * y[0] + x[1] * y[1] + x[2] * y[2]).clamp(-1.0, 1.0);
            sum += d.acos();
            n += 1;
        }
    }
    sum / n.max(1) as f64
}

fn organic_cfg() -> ReconConfig {
    ReconConfig {
        sdf_smooth_sigma_um: 24.0,
        subdivision_levels: 1,
        ..ReconConfig::default()
    }
}

#[test]
fn sdf_smoothing_reduces_staircase_without_touching_voxels() {
    let p = generate_phantom(&ellipsoid_spec());
    let (mask_a, mesh_a) = recon(&p, &ReconConfig::default());
    let smooth = ReconConfig {
        sdf_smooth_sigma_um: 24.0,
        ..ReconConfig::default()
    };
    let (mask_b, mesh_b) = recon(&p, &smooth);
    // Masker voxel (dasar semua pengukuran) tidak berubah sama sekali.
    assert_eq!(mask_a.data, mask_b.data);
    let (ra, rb) = (mean_dihedral(&mesh_a), mean_dihedral(&mesh_b));
    println!("sudut dihedral rata-rata: tanpa {ra:.4} rad, dengan {rb:.4} rad");
    assert!(
        rb < ra,
        "penghalusan SDF harus mengurangi kekasaran: {rb} ≥ {ra}"
    );
    let truth = p.truth.volume_um3;
    let e = (mesh_b.signed_volume_um3() - truth).abs() / truth;
    assert!(e <= 0.03, "galat volume mesh halus {e}");
    assert!(mesh_b.is_closed());
}

#[test]
fn organic_surface_keeps_volume_and_topology() {
    let p = generate_phantom(&ellipsoid_spec());
    let (_, base) = recon(&p, &ReconConfig::default());
    let (_, mesh) = recon(&p, &organic_cfg());
    let truth = p.truth.volume_um3;
    let e = (mesh.signed_volume_um3() - truth).abs() / truth;
    println!(
        "organik: galat volume {:.3} %, segitiga {} → {}",
        e * 100.0,
        base.triangle_count(),
        mesh.triangle_count()
    );
    assert!(e <= 0.03, "galat volume {e}");
    assert!(mesh.is_closed());
    assert_eq!(mesh.connected_components(), 1);
    assert_eq!(mesh.euler_characteristic(), 2);
    assert_eq!(mesh.normals.len(), mesh.positions_um.len());
    assert!(mesh.triangle_count() > 2 * base.triangle_count());
}

#[test]
fn loop_subdivision_preserves_volume_and_attributes() {
    let p = generate_phantom(&ellipsoid_spec());
    let (_, mut mesh) = recon(&p, &ReconConfig::default());
    let n0 = mesh.positions_um.len();
    mesh.colors = (0..n0).map(|i| [(i % 251) as u8, 100, 200]).collect();
    mesh.occlusion = vec![0.5; n0];
    mesh.set_scalars(h3d_volume::MeshScalars {
        name: "s".into(),
        unit: "µm".into(),
        values: (0..n0).map(|i| i as f32).collect(),
    });
    let v0 = mesh.signed_volume_um3();
    let (t0, chi0) = (mesh.triangle_count(), mesh.euler_characteristic());
    mesh.subdivide_loop(1);
    let n1 = mesh.positions_um.len();
    assert_eq!(mesh.triangle_count(), 4 * t0);
    assert_eq!(mesh.euler_characteristic(), chi0);
    assert!(mesh.is_closed());
    let dv = (mesh.signed_volume_um3() - v0).abs() / v0;
    assert!(dv < 0.005, "volume bergeser {dv}");
    assert_eq!(mesh.colors.len(), n1);
    assert_eq!(mesh.occlusion.len(), n1);
    assert_eq!(mesh.scalars[0].values.len(), n1);
    assert!(mesh.occlusion.iter().all(|&o| (o - 0.5).abs() < 1e-6));
}

#[test]
fn torus_stays_genus_one_when_organic() {
    let p = generate_phantom(&PhantomSpec {
        shape: PhantomShape::Torus {
            major: 1000.0,
            minor: 400.0,
        },
        n_slices: 21,
        spacing_um: 50.0,
        s_xy: 16.0,
        size_um: [3840.0, 3840.0],
        ..Default::default()
    });
    let (_, mut mesh) = recon(&p, &organic_cfg());
    assert!(mesh.is_closed());
    assert_eq!(mesh.euler_characteristic(), 0, "torus: genus 1");
    let truth = p.truth.volume_um3;
    let e = (mesh.signed_volume_um3() - truth).abs() / truth;
    assert!(e < 0.05, "{e}");

    // Oklusi ambien: sisi dalam cincin (menghadap lubang) lebih tertutup
    // daripada sisi luar.
    mesh.bake_occlusion(32, 2000.0);
    assert_eq!(mesh.occlusion.len(), mesh.positions_um.len());
    let (lo, hi) = mesh.bbox().expect("bbox");
    let c = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])];
    let (mut inner, mut outer) = ((0.0, 0usize), (0.0, 0usize));
    for (p, &o) in mesh.positions_um.iter().zip(&mesh.occlusion) {
        let r = ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2)).sqrt();
        if r < 800.0 {
            inner.0 += f64::from(o);
            inner.1 += 1;
        } else if r > 1300.0 {
            outer.0 += f64::from(o);
            outer.1 += 1;
        }
    }
    let (mi, mo) = (inner.0 / inner.1 as f64, outer.0 / outer.1 as f64);
    println!("oklusi: dalam {mi:.3}, luar {mo:.3}");
    assert!(
        mi + 0.1 < mo,
        "dalam {mi} harus lebih tertutup dari luar {mo}"
    );
    assert!(mo > 0.9);
}

#[test]
fn organic_reconstruction_and_occlusion_are_byte_identical() {
    let p = generate_phantom(&ellipsoid_spec());
    let (_, mut a) = recon(&p, &organic_cfg());
    let (_, mut b) = recon(&p, &organic_cfg());
    assert_eq!(mesh_bytes(&a), mesh_bytes(&b));
    a.bake_occlusion(16, 500.0);
    b.bake_occlusion(16, 500.0);
    assert_eq!(a.occlusion, b.occlusion);
    // Bola cembung: hampir semua verteks terbuka.
    let mean = a.occlusion.iter().map(|&o| f64::from(o)).sum::<f64>() / a.occlusion.len() as f64;
    assert!(mean > 0.95, "{mean}");
}
