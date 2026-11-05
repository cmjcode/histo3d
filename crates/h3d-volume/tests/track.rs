//! Uji penautan objek antar irisan (`h3d_volume::track`) dan rekonstruksi
//! per objek atas fantom analitik.

use h3d_core::{Budgets, NoProgress, ReconConfig, SlideId, SlideTransform};
use h3d_volume::phantom3d::LABEL_TUMOR;
use h3d_volume::{
    generate_phantom, plan_grid, reconstruct_label_objects, track_label, ArrayLabelSampler,
    GridPlan, Phantom, PhantomShape, PhantomSpec, SliceInput, TrackEvent, TrackParams, TrackSet,
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

fn plan(inp: &[SliceInput<'_>], cfg: &ReconConfig) -> GridPlan {
    plan_grid(inp, &[LABEL_TUMOR], cfg, &Budgets::MACOS).expect("plan")
}

fn track(p: &Phantom, params: &TrackParams) -> TrackSet {
    let smp = samplers(p);
    let inp = inputs(p, &smp);
    let cfg = ReconConfig::default();
    let plan = plan(&inp, &cfg);
    track_label(&plan, &inp, LABEL_TUMOR, params, &NoProgress).expect("track")
}

/// Dua bola terpisah 600 µm sepanjang X.
fn two_spheres() -> PhantomSpec {
    PhantomSpec {
        shape: PhantomShape::Ellipsoid {
            radii: [500.0, 500.0, 500.0],
        },
        foci: Some((2, 600.0)),
        n_slices: 13,
        spacing_um: 100.0,
        s_xy: 16.0,
        size_um: [3200.0, 2000.0],
        ..Default::default()
    }
}

fn branch() -> PhantomSpec {
    PhantomSpec {
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
    }
}

#[test]
fn two_separate_ellipsoids_give_two_tracks_without_events() {
    let p = generate_phantom(&two_spheres());
    let set = track(&p, &TrackParams::default());
    assert_eq!(set.tracks.len(), 2, "{set:?}");
    assert_eq!(set.event_count(), 0);
    assert!(set.warnings.is_empty());
    let n_slices = p
        .slices
        .iter()
        .filter(|s| s.masks.iter().any(|(l, _)| *l == LABEL_TUMOR))
        .count();
    for t in &set.tracks {
        assert_eq!(t.members.len(), n_slices, "objek {}", t.instance_id);
        assert!(t.members.windows(2).all(|w| w[0].z_um < w[1].z_um));
        assert!(t.members.iter().all(|m| m.area_um2 > 0.0));
    }
    // Objek 1 berada di kiri (x lebih kecil) objek 2: urutan deterministik.
    let cx = |t: &h3d_volume::ObjectTrack| t.members[0].centroid_um[0];
    assert!(cx(&set.tracks[0]) < cx(&set.tracks[1]));
    assert_eq!(set.tracks[0].instance_id, 1);
    assert_eq!(set.tracks[1].instance_id, 2);
    assert_eq!(set.min_iou, 0.2);
}

#[test]
fn branch_is_born_once_then_splits_into_two() {
    let p = generate_phantom(&branch());
    let set = track(&p, &TrackParams::default());
    let splits: Vec<&TrackEvent> = set
        .tracks
        .iter()
        .flat_map(|t| t.events.iter())
        .filter(|e| matches!(e, TrackEvent::Split { .. }))
        .collect();
    assert_eq!(splits.len(), 1, "{set:?}");
    assert_eq!(set.event_count(), 1, "tidak boleh ada Merge");
    assert_eq!(set.tracks.len(), 3, "batang + dua lengan");
    let TrackEvent::Split {
        parent, children, ..
    } = splits[0]
    else {
        unreachable!()
    };
    assert_eq!(*parent, 1, "batang lahir pertama");
    assert_eq!(children, &vec![2, 3]);
    // Batang berakhir tepat sebelum anak-anaknya mulai.
    let trunk = set.track(1).expect("batang");
    let arm = set.track(2).expect("lengan");
    assert!(trunk.members.last().map(|m| m.z_um) < arm.members.first().map(|m| m.z_um));
}

#[test]
fn tracking_is_byte_identical_between_runs() {
    let p = generate_phantom(&branch());
    let a = track(&p, &TrackParams::default());
    let b = track(&p, &TrackParams::default());
    let ja = serde_json::to_string(&a).expect("json");
    let jb = serde_json::to_string(&b).expect("json");
    assert_eq!(ja, jb);
    assert_eq!(a, b);
    let back: TrackSet = serde_json::from_str(&ja).expect("ulang");
    assert_eq!(back, a);
}

#[test]
fn gap_larger_than_max_gap_breaks_the_chain() {
    // Satu bola; buang irisan tengah sehingga celah > 5 × jarak median.
    let mut single = generate_phantom(&PhantomSpec {
        foci: None,
        n_slices: 23,
        spacing_um: 50.0,
        ..two_spheres()
    });
    single.slices.retain(|s| s.z_um.abs() > 175.0);
    let smp = samplers(&single);
    let inp = inputs(&single, &smp);
    let cfg = ReconConfig::default();
    let plan = plan(&inp, &cfg);
    assert!(plan
        .warnings
        .iter()
        .any(|w| matches!(w, Warning::GapSplit { .. })));
    let set = track_label(
        &plan,
        &inp,
        LABEL_TUMOR,
        &TrackParams::default(),
        &NoProgress,
    )
    .expect("track");
    assert_eq!(set.tracks.len(), 2, "{set:?}");
    assert_eq!(set.event_count(), 0);
    assert!(set
        .warnings
        .iter()
        .any(|w| matches!(w, Warning::GapSplit { .. })));
    // Dengan celah yang lebih besar dari jarak itu, kembali satu objek.
    let wide = TrackParams {
        max_gap_um: Some(10_000.0),
        ..Default::default()
    };
    let set2 = track_label(&plan, &inp, LABEL_TUMOR, &wide, &NoProgress).expect("track");
    assert_eq!(set2.tracks.len(), 1);
    assert_ne!(set.params_hash, set2.params_hash);
}

#[test]
fn per_object_reconstruction_gives_one_closed_mesh_per_track() {
    let p = generate_phantom(&two_spheres());
    let smp = samplers(&p);
    let inp = inputs(&p, &smp);
    let cfg = ReconConfig::default();
    let plan = plan(&inp, &cfg);
    let (set, meshes) = reconstruct_label_objects(
        &plan,
        &inp,
        LABEL_TUMOR,
        &cfg,
        &TrackParams::default(),
        &NoProgress,
    )
    .expect("recon per objek");
    assert_eq!(set.tracks.len(), 2);
    assert_eq!(meshes.len(), set.tracks.len());
    let truth_each = p.truth.volume_um3 / 2.0;
    let mut sum = 0.0;
    for (m, t) in meshes.iter().zip(&set.tracks) {
        assert_eq!(m.instance_id, Some(t.instance_id));
        assert_eq!(m.label.0, LABEL_TUMOR);
        assert!(m.is_closed(), "objek {} tidak tertutup", t.instance_id);
        assert_eq!(m.connected_components(), 1);
        let v = m.signed_volume_um3();
        assert!(v > 0.0);
        sum += v;
        // Jarak irisan 1/10 diameter: galat diskretisasi sama dengan jalur
        // satu label (lihat di bawah), batas longgar terhadap analitik.
        let e = (v - truth_each).abs() / truth_each;
        assert!(e < 0.1, "objek {}: galat volume {e}", t.instance_id);
        assert_eq!(m.display_name("Tumor"), format!("Tumor#{}", t.instance_id));
    }
    // Rekonstruksi satu label utuh memakai jalur SDF yang sama: jumlah volume
    // objek harus sama dengan volume mesh utuh (dua komponen).
    let (_, whole) =
        h3d_volume::reconstruct_label(&plan, &inp, LABEL_TUMOR, &cfg, &NoProgress).expect("utuh");
    assert_eq!(whole.instance_id, None);
    assert_eq!(whole.connected_components(), 2);
    let vw = whole.signed_volume_um3();
    assert!(((sum - vw) / vw).abs() < 0.01, "{sum} vs {vw}");
}
