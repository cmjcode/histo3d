//! Validasi analisis spasial atas fantom 8.12.11 (histo3d.md H10b).

use h3d_core::{
    AnatDir, Budgets, LabelDef, LabelId, LabelKind, NoProgress, ReconConfig, SignedAxis, SlideId,
    SlideTransform, SpatialConfig, SpecimenOrientation, VesselKind,
};
use h3d_volume::phantom3d::{
    LABEL_CAPSULE, LABEL_MARGIN, LABEL_SURFACE, LABEL_TUMOR, LABEL_VESSEL,
};
use h3d_volume::spatial::MarginStatus;
use h3d_volume::{
    analyze, build_label_stack, extract_mesh, generate_phantom, plan_grid, project_fields_to_mesh,
    ArrayLabelSampler, LabelMesh, Phantom, PhantomShape, PhantomSpec, SliceInput, SpatialAnalysis,
    StackGrid, VoxelMask, Warning,
};

static IDENTITY: SlideTransform = SlideTransform::Identity;

fn def(label: u8) -> LabelDef {
    let (name, kind) = match label {
        LABEL_TUMOR => ("Tumor", LabelKind::Tumor),
        LABEL_MARGIN => ("Tepi", LabelKind::Margin),
        LABEL_VESSEL => ("Pembuluh", LabelKind::Vessel(VesselKind::Blood)),
        LABEL_CAPSULE => ("Kapsul", LabelKind::Capsule),
        LABEL_SURFACE => ("Permukaan", LabelKind::Surface),
        _ => ("Lain", LabelKind::Other),
    };
    LabelDef::new(LabelId(label), name, [255, 0, 0, 255], kind)
}

/// Masker voxel semua label fantom pada satu kisi, plus mesh tumor.
fn voxelize(p: &Phantom, labels: &[u8]) -> (Vec<(LabelDef, VoxelMask)>, LabelMesh) {
    let smp: Vec<ArrayLabelSampler> = p
        .slices
        .iter()
        .map(|s| ArrayLabelSampler::from_slice(s, p.spec.s_xy))
        .collect();
    let inp: Vec<SliceInput<'_>> = p
        .slices
        .iter()
        .zip(&smp)
        .enumerate()
        .map(|(i, (s, sm))| SliceInput {
            id: SlideId(i as u32),
            z_um: s.z_um,
            transform: &IDENTITY,
            labels: sm,
            annotated_labels: sm.annotated_labels(),
            reviewed_labels: Vec::new(),
        })
        .collect();
    let cfg = ReconConfig::default();
    let plan = plan_grid(&inp, labels, &cfg, &Budgets::MACOS).expect("plan");
    let mut out = Vec::new();
    let mut mesh = LabelMesh::empty(LabelId(LABEL_TUMOR));
    for &l in labels {
        let stack = build_label_stack(&plan, &inp, l, &cfg, &NoProgress).expect("stack");
        if l == LABEL_TUMOR {
            mesh = extract_mesh(&stack, LabelId(l), &cfg, &NoProgress).expect("mesh");
        }
        out.push((def(l), stack.voxel_mask()));
    }
    (out, mesh)
}

fn run(masks: &[(LabelDef, VoxelMask)], o: Option<&SpecimenOrientation>) -> SpatialAnalysis {
    analyze(masks, &[], o, &SpatialConfig::default(), None, &NoProgress).expect("analyze")
}

fn margin_phantom(d: f64) -> Phantom {
    generate_phantom(&PhantomSpec {
        shape: PhantomShape::Ellipsoid {
            radii: [800.0, 600.0, 500.0],
        },
        n_slices: 13,
        spacing_um: 100.0,
        s_xy: 16.0,
        size_um: [3200.0, 1920.0],
        margin_plane: Some(d),
        ..Default::default()
    })
}

#[test]
fn margin_plane_distance_and_direction() {
    let d = 400.0;
    let p = margin_phantom(d);
    let (masks, _) = voxelize(&p, &[LABEL_TUMOR, LABEL_MARGIN]);
    let orientation = SpecimenOrientation {
        pos_x: AnatDir::Lateral,
        pos_y: AnatDir::Anterior,
        pos_z: AnatDir::Superior,
    };
    let a = run(&masks, Some(&orientation));
    let m = a.margin.as_ref().expect("margin");
    let s_xy = a.grid.s_xy_um;
    println!(
        "margin: d_min {:.1} ± {:.1} (benar {d}), arah {:?}",
        m.d_min.value_um, m.d_min.plus_minus_um, m.direction
    );
    assert!((m.d_min.value_um - d).abs() <= s_xy, "{}", m.d_min.value_um);
    assert_eq!(m.direction, SignedAxis::PosX);
    assert_eq!(m.direction_name, Some(AnatDir::Lateral));
    assert_eq!(m.status, MarginStatus::Close);
    // Ketidakpastian: ½ voxel + TRE (bawaan s_xy).
    assert!((m.d_min.plus_minus_um - 1.5 * s_xy).abs() < 1e-9);
    assert!(!m.d_min.z_limited);
    assert!(m.nearest_margin_um[0] > m.nearest_tumor_um[0]);
    let px = m.per_direction[0].expect("+X");
    assert_eq!(px.value_um, m.d_min.value_um);
    assert!(m.per_direction[1].is_none(), "tidak ada margin ke arah -X");
    assert!(a.warnings.contains(&Warning::MarginClose {
        dir: "lateral".into()
    }));
    assert!(!a.warnings.contains(&Warning::MarginZUnassessable));
    assert!(a.fields.iter().any(|f| f.0 == "margin_distance"));
    // Ukuran staging ≈ diameter terpanjang 1600 µm.
    let size = a.size.as_ref().expect("size");
    assert!((size.pca_axes_um[0].value_um - 1600.0).abs() <= 2.0 * s_xy);
    assert!((size.max_diameter_um.value_um - 1600.0).abs() <= 2.0 * s_xy);
    assert!(size.pca_axes_um[2].z_limited);
}

#[test]
fn margin_touching_is_involved_and_far_is_clear() {
    let (masks, _) = voxelize(&margin_phantom(0.0), &[LABEL_TUMOR, LABEL_MARGIN]);
    let a = run(&masks, None);
    assert_eq!(a.margin.as_ref().unwrap().status, MarginStatus::Involved);
    assert!(a
        .warnings
        .contains(&Warning::MarginClose { dir: "+X".into() }));

    let (masks, _) = voxelize(&margin_phantom(400.0), &[LABEL_TUMOR, LABEL_MARGIN]);
    let cfg = SpatialConfig {
        margin_close_um: 200.0,
        ..Default::default()
    };
    let a = analyze(&masks, &[], None, &cfg, Some(5.0), &NoProgress).unwrap();
    assert_eq!(a.margin.as_ref().unwrap().status, MarginStatus::Clear);
    assert!(!a
        .warnings
        .iter()
        .any(|w| matches!(w, Warning::MarginClose { .. })));
    assert_eq!(a.tre_est_um, 5.0);
}

#[test]
fn two_spheres_are_two_foci_with_gap() {
    let g = 300.0;
    let p = generate_phantom(&PhantomSpec {
        shape: PhantomShape::Ellipsoid {
            radii: [400.0, 400.0, 400.0],
        },
        n_slices: 19,
        spacing_um: 50.0,
        s_xy: 16.0,
        size_um: [2560.0, 1280.0],
        foci: Some((2, g)),
        ..Default::default()
    });
    let (masks, _) = voxelize(&p, &[LABEL_TUMOR]);
    let a = run(&masks, None);
    let f = a.foci.as_ref().expect("foci");
    assert_eq!(f.n, p.truth.n_foci);
    assert_eq!(f.n, 2);
    let d = f.distances_um[0][1];
    println!("fokus: jarak {:.1} (benar {g})", d.value_um);
    assert!((d.value_um - g).abs() <= a.grid.s_xy_um, "{}", d.value_um);
    assert_eq!(f.distances_um[1][0].value_um, d.value_um);
    assert!(f.foci[0].volume_um3 >= f.foci[1].volume_um3);
    let v1 = 4.0 / 3.0 * std::f64::consts::PI * 400f64.powi(3);
    for focus in &f.foci {
        assert!(
            (focus.volume_um3 - v1).abs() / v1 < 0.05,
            "{}",
            focus.volume_um3
        );
        assert!((focus.max_diameter_um - 800.0).abs() < 48.0);
    }
}

#[test]
fn vessel_contact_area() {
    let p = generate_phantom(&PhantomSpec {
        shape: PhantomShape::Ellipsoid {
            radii: [800.0, 800.0, 600.0],
        },
        n_slices: 31,
        spacing_um: 50.0,
        s_xy: 16.0,
        size_um: [2240.0, 2240.0],
        vessel: Some(200.0),
        ..Default::default()
    });
    let truth = p.truth.contact_area_mm2.expect("kebenaran kontak");
    let (masks, _) = voxelize(&p, &[LABEL_TUMOR, LABEL_VESSEL]);
    let a = run(&masks, None);
    let c = &a.contacts[0];
    assert_eq!(c.label, LabelId(LABEL_VESSEL));
    assert!(c.result.any);
    let e = (c.result.area_mm2 - truth).abs() / truth;
    println!(
        "kontak: {:.4} mm² (benar {:.4}), galat {:.2} %, mentah {:.4}",
        c.result.area_mm2,
        truth,
        e * 100.0,
        c.result.area_voxel_faces_mm2
    );
    assert!(e <= 0.10, "galat luas kontak {e}");
    assert_eq!(c.result.patches.len(), 1);
    assert!(c.result.patches[0].layers.len() > 10);
    // Hipoksia: pembuluh darah ada → histogram dan fraksi.
    let h = a.hypoxia.as_ref().expect("hipoksia");
    assert!(h.fraction_beyond > 0.0 && h.fraction_beyond < 1.0);
    assert_eq!(
        h.histogram.iter().map(|b| b.1).sum::<u64>() as usize,
        masks[0].1.count()
    );
}

fn capsule_phantom(breach: bool) -> Phantom {
    generate_phantom(&PhantomSpec {
        shape: PhantomShape::Ellipsoid {
            radii: [500.0, 500.0, 500.0],
        },
        // Jarak irisan = 2 × s_xy → tanpa lapis interpolasi. Kulit setipis
        // 64 µm tidak bertahan utuh pada interpolasi SDF antar irisan yang
        // lubangnya berubah (pita SDF 128 µm ≫ setengah tebal kulit).
        n_slices: 51,
        spacing_um: 32.0,
        s_xy: 16.0,
        size_um: [2240.0, 2240.0],
        capsule: Some((64.0, breach)),
        ..Default::default()
    })
}

#[test]
fn capsule_breach_detected_in_both_variants() {
    for breach in [false, true] {
        let p = capsule_phantom(breach);
        let (masks, _) = voxelize(&p, &[LABEL_TUMOR, LABEL_CAPSULE]);
        let a = run(&masks, None);
        let c = a.capsule.as_ref().expect("kapsul");
        println!(
            "kapsul tembus={breach}: hasil {}, luar {:.3e} µm³, dalam {:.3e} µm³",
            c.breached, c.volume_outside_um3, c.volume_inside_um3
        );
        assert_eq!(c.breached, breach);
        assert_eq!(c.per_component, vec![breach]);
        assert!(c.volume_inside_um3 > 0.0);
        assert_eq!(c.volume_outside_um3 > 0.0, breach);
        // Kontak tumor–kapsul hanya pada varian tembus.
        let contact = a
            .contacts
            .iter()
            .find(|e| e.kind == LabelKind::Capsule)
            .expect("kontak kapsul");
        assert_eq!(contact.result.any, breach);
    }
}

#[test]
fn invasion_depth_from_flat_surface() {
    let h = 500.0;
    let p = generate_phantom(&PhantomSpec {
        shape: PhantomShape::Ellipsoid {
            radii: [400.0, 400.0, 400.0],
        },
        n_slices: 8,
        spacing_um: 100.0,
        s_xy: 16.0,
        size_um: [1280.0, 1280.0],
        surface_at_z0: Some(h),
        ..Default::default()
    });
    let (masks, _) = voxelize(&p, &[LABEL_TUMOR, LABEL_SURFACE]);
    let a = run(&masks, None);
    let d = a.depth.as_ref().expect("kedalaman");
    println!(
        "kedalaman: {:.1} ± {:.1} (benar {h}), s_z {}",
        d.depth.value_um, d.depth.plus_minus_um, a.grid.s_z_um
    );
    assert!((d.depth.value_um - h).abs() <= a.grid.s_z_um / 2.0);
    assert!(d.depth.z_limited);
    assert!((d.deepest_um[2] - h).abs() <= a.grid.s_z_um / 2.0);
    assert!(a.breslow.is_none());
    assert!(a.grid.uniform_z && !a.grid.resampled_z);
}

#[test]
fn nonuniform_spacing_is_resampled_and_noted() {
    // Satu irisan hilang → celah 200 µm dibagi 7, bukan 4 × 25 µm.
    let mut p = margin_phantom(400.0);
    p.slices.remove(4);
    let (masks, _) = voxelize(&p, &[LABEL_TUMOR, LABEL_MARGIN]);
    assert!(!masks[0].1.grid.is_uniform_z(1e-6));
    let a = run(&masks, None);
    assert!(a.grid.resampled_z && !a.grid.uniform_z);
    assert!(a
        .warnings
        .iter()
        .any(|w| matches!(w, Warning::NonUniformZResampled { dz } if (*dz - 25.0).abs() < 1e-9)));
    assert!(a.notes.iter().any(|n| n.contains("diresampel")));
    let m = a.margin.as_ref().expect("margin");
    assert!((m.d_min.value_um - 400.0).abs() <= a.grid.s_xy_um);
}

#[test]
fn analyze_twice_gives_identical_json() {
    let p = margin_phantom(400.0);
    let (masks, _) = voxelize(&p, &[LABEL_TUMOR, LABEL_MARGIN]);
    let a = serde_json::to_vec(&run(&masks, None)).expect("json");
    let b = serde_json::to_vec(&run(&masks, None)).expect("json");
    assert!(!a.is_empty());
    assert_eq!(a, b);
    let (masks2, _) = voxelize(&p, &[LABEL_TUMOR, LABEL_MARGIN]);
    let c = serde_json::to_vec(&run(&masks2, None)).expect("json");
    assert_eq!(a, c, "voxelisasi ulang + analisis harus identik");
}

#[test]
fn z_resolution_and_missing_margin_warnings() {
    // Margin hanya di irisan tengah → arah Z tak dapat dinilai.
    let mut p = margin_phantom(400.0);
    for s in &mut p.slices {
        if s.z_um.abs() > 150.0 {
            s.masks.retain(|(l, _)| *l != LABEL_MARGIN);
        }
    }
    let (masks, _) = voxelize(&p, &[LABEL_TUMOR, LABEL_MARGIN]);
    let cfg = SpatialConfig {
        margin_close_um: 10.0,
        ..Default::default()
    };
    let a = analyze(&masks, &[], None, &cfg, None, &NoProgress).unwrap();
    assert!(a.warnings.contains(&Warning::MarginZUnassessable));
    // s_z (25 µm) > 2 × 10 µm.
    assert!(a.warnings.contains(&Warning::ZResolutionCoarse));
}

#[test]
fn linear_field_projects_exactly_onto_vertices() {
    let p = margin_phantom(400.0);
    let (masks, mut mesh) = voxelize(&p, &[LABEL_TUMOR]);
    let grid: &StackGrid = &masks[0].1.grid;
    let f = |q: [f64; 3]| 0.25 * q[0] - 0.5 * q[1] + 0.75 * q[2] + 10.0;
    let mut field = vec![0f32; grid.voxel_count()];
    for (i, v) in field.iter_mut().enumerate() {
        let [x, y, z] = grid.delinear(i);
        *v = f(grid.center_um(x, y, z)) as f32;
    }
    let mut nodata = field.clone();
    nodata.fill(f32::NAN);
    project_fields_to_mesh(
        &mut mesh,
        grid,
        &[
            ("linear".into(), "µm".into(), field),
            ("kosong".into(), "".into(), nodata),
            ("salah-ukuran".into(), "".into(), vec![0.0; 3]),
        ],
    );
    assert_eq!(mesh.scalars.len(), 2);
    let s = &mesh.scalars[0];
    assert_eq!(s.values.len(), mesh.positions_um.len());
    let mut max_err = 0f64;
    for (v, q) in s.values.iter().zip(&mesh.positions_um) {
        max_err = max_err.max((*v as f64 - f(*q)).abs());
    }
    println!("proyeksi trilinear: galat maks {max_err:.2e}");
    assert!(max_err < 2e-3, "{max_err}");
    assert!(mesh.scalars[1].values.iter().all(|v| v.is_nan()));
}

#[test]
fn attribute_statistics_over_tumor() {
    use h3d_core::AttrDef;
    use h3d_volume::AttrVolume;
    let p = margin_phantom(400.0);
    let (masks, _) = voxelize(&p, &[LABEL_TUMOR]);
    let grid = masks[0].1.grid.clone();
    // Atribut = x dalam mm; separuh kiri kisi NODATA.
    let data: Vec<f32> = (0..grid.voxel_count())
        .map(|i| {
            let [x, y, z] = grid.delinear(i);
            let c = grid.center_um(x, y, z);
            if c[0] < 1600.0 {
                f32::NAN
            } else {
                (c[0] / 1000.0) as f32
            }
        })
        .collect();
    let attr = AttrDef {
        name: "x-mm".into(),
        unit: "mm".into(),
        ..Default::default()
    };
    let a = analyze(
        &masks,
        &[(attr, AttrVolume { grid, data })],
        None,
        &SpatialConfig::default(),
        None,
        &NoProgress,
    )
    .unwrap();
    let st = &a.attr_stats[0].stats;
    assert!(st.n > 0 && st.n < masks[0].1.count());
    assert!(st.p10 < st.median && st.median < st.p90);
    assert!(st.mean > 1.6 && st.mean < 2.4);
    assert!(st.heterogeneity_cv.is_some());
    assert!(!st.per_layer.is_empty());
    assert!(a.fields.iter().any(|f| f.0 == "attr:x-mm"));
}
