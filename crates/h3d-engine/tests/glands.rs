//! Uji ujung-ke-ujung objek 2D kelenjar (ADR 0004): fantom kelenjar
//! diimpor sebagai slide, disegmentasi menjadi usulan, diterima ke label,
//! lalu dihitung kembali dari `LabelStore`: tiap kelenjar harus tetap satu
//! komponen tersendiri (pemisah bertahan setelah resampling ke `mpp_label`).

use h3d_core::{LabelKind, NoProgress, SlideId};
use h3d_engine::{ColorPreset, CountParams, EpitheliumSource, GlandParams, PaintMode, Session};
use h3d_imgproc::phantom::render_glands;

fn gland_project(name: &str) -> (tempfile::TempDir, Session, SlideId, usize, f64) {
    let tmp = tempfile::tempdir().unwrap();
    let slides = tmp.path().join("slides");
    std::fs::create_dir_all(&slides).unwrap();
    let mpp = 2.0;
    let (w, h) = (400usize, 300usize);
    let (rgba, truth) = render_glands([w, h], mpp, 21);
    let path = slides.join("glands.png");
    image::RgbaImage::from_raw(w as u32, h as u32, rgba)
        .unwrap()
        .save(&path)
        .unwrap();
    let dir = tmp.path().join(format!("{name}.h3d"));
    let mut s = Session::create(&dir).unwrap();
    let ids = s
        .import_slides(&[path], Some([mpp, mpp]), &NoProgress)
        .unwrap();
    (tmp, s, ids[0], truth.n, mpp)
}

#[test]
fn gland_proposal_counts_phantom_and_survives_acceptance_as_separate_objects() {
    let (_tmp, mut s, slide, n_truth, mpp) = gland_project("kelenjar");
    let label = s
        .add_label("Kelenjar", [40, 180, 200, 255], LabelKind::Other)
        .unwrap();
    let params = GlandParams {
        mpp_um: mpp,
        ..GlandParams::default()
    };
    let gp = s
        .segment_glands(slide, label, &params, &NoProgress)
        .unwrap();
    assert_eq!(gp.summary.n, n_truth, "{:?}", gp.summary.glands);
    assert!(gp.summary.n_min <= gp.summary.n && gp.summary.n <= gp.summary.n_max);
    assert_eq!(gp.summary.n_with_lumen, n_truth - 1, "satu sarang padat");
    assert_eq!(gp.summary.label_name, "Kelenjar");
    assert!(gp.summary.stain_estimated);
    // Usulan tidak menyentuh label sebelum diterima (aturan keras 8).
    assert!(!s.slides_with_label(label).contains(&slide));
    assert!(s
        .accept_proposal(slide, &gp.proposal, PaintMode::OnlyBackground)
        .unwrap());
    // Dihitung ulang dari LabelStore (mpp_label 4 µm) tanpa pembelahan:
    // pemisah harus menjaga tiap kelenjar sebagai komponen sendiri.
    let count = s
        .count_objects_2d(
            slide,
            label,
            None,
            &CountParams {
                split_enabled: false,
                ..CountParams::default()
            },
        )
        .unwrap();
    assert_eq!(count.n(), n_truth);
    // Determinisme: dua jalan memberi usulan identik.
    let gp2 = s
        .segment_glands(slide, label, &params, &NoProgress)
        .unwrap();
    assert_eq!(gp.proposal.mask, gp2.proposal.mask);
    assert_eq!(gp.summary, gp2.summary);
    // JSON ringkasan stabil dan memuat daftar objek.
    let json = h3d_engine::to_stable_json_of(&gp.summary).unwrap();
    assert!(json.contains("\"glands\"") && json.contains("\"params_hash\""));
}

#[test]
fn invalid_params_label_and_slide_are_errors() {
    let (_tmp, mut s, slide, _, mpp) = gland_project("galat");
    let label = s
        .add_label("Kelenjar", [40, 180, 200, 255], LabelKind::Other)
        .unwrap();
    let bad = GlandParams {
        mpp_um: mpp,
        min_ring_fraction: 2.0,
        ..GlandParams::default()
    };
    assert!(s.segment_glands(slide, label, &bad, &NoProgress).is_err());
    let ok = GlandParams {
        mpp_um: mpp,
        ..GlandParams::default()
    };
    assert!(s
        .segment_glands(slide, h3d_core::LabelId(250), &ok, &NoProgress)
        .is_err());
    assert!(s
        .segment_glands(SlideId(99), label, &ok, &NoProgress)
        .is_err());
}

#[test]
fn palette_and_color_mode_find_the_same_phantom_glands() {
    let (_tmp, mut s, slide, n_truth, mpp) = gland_project("warna");
    let label = s
        .add_label("Kelenjar", [40, 180, 200, 255], LabelKind::Other)
        .unwrap();
    let pal = s.slide_palette(slide, mpp, 5, &NoProgress).unwrap();
    assert!(pal.colors.len() >= 3 && pal.n_samples > 1000);
    assert!(pal.colors.windows(2).all(|w| w[0].lab[0] <= w[1].lab[0]));
    let idx = pal.preset_indices(ColorPreset::Gland);
    assert!(!idx.is_empty(), "{pal:?}");
    let params = GlandParams {
        mpp_um: mpp,
        epithelium: EpitheliumSource::Color {
            picks: pal.picks(&idx),
            tolerance: pal.default_tolerance(&idx),
        },
        ..GlandParams::default()
    };
    let gp = s
        .segment_glands(slide, label, &params, &NoProgress)
        .unwrap();
    assert_eq!(gp.summary.n, n_truth, "{:?}", gp.summary.glands);
    assert_eq!(gp.summary.epithelium_mode, "warna");
    assert_eq!(gp.summary.color_picks.len(), idx.len());
    assert!(gp.summary.color_tolerance.is_some());
    assert!(s.slide_palette(slide, 0.0, 5, &NoProgress).is_err());
    assert!(s.slide_palette(slide, mpp, 1, &NoProgress).is_err());
}
