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
