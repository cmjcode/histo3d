//! Uji penghitungan objek 2D per irisan: label fantom, objek bersentuhan,
//! ROI, usulan, dan keluaran JSON stabil.

mod common;

use common::open_copy;
use h3d_core::{LabelKind, SlideId};
use h3d_engine::{BorderRule, CountParams, CountSource, PaintMode};

fn circle(c: [f64; 2], r: f64) -> Vec<[f64; 2]> {
    (0..48)
        .map(|i| {
            let a = i as f64 / 48.0 * std::f64::consts::TAU;
            [c[0] + r * a.cos(), c[1] + r * a.sin()]
        })
        .collect()
}

fn params() -> CountParams {
    CountParams {
        min_seed_distance_um: 100.0,
        h_um: 20.0,
        merge_min_area_um2: 2_000.0,
        min_area_um2: 5_000.0,
        ..CountParams::default()
    }
}

#[test]
fn phantom_ellipsoid_counts_one_object_per_slice() {
    let mut s = open_copy("count-phantom");
    let tumor = s.label_by_name("Tumor").unwrap();
    let ids: Vec<SlideId> = s
        .project()
        .usable_slides_ordered()
        .iter()
        .map(|e| e.id)
        .collect();
    let mid = ids[ids.len() / 2];
    let c = s.count_objects_2d(mid, tumor, None, &params()).unwrap();
    assert_eq!(c.n(), 1, "{:?}", c.estimate.count.objects);
    assert_eq!(c.source, CountSource::Labels);
    assert_eq!(c.label_name, "Tumor");
    assert!(c.estimate.n_min <= 1 && c.estimate.n_max >= 1);
    let o = &c.estimate.count.objects[0];
    assert!(o.circularity > 0.8, "sirkularitas {}", o.circularity);
    let u = c.centroid_u(o);
    let size = s.slide(mid).unwrap().size_um();
    assert!((u[0] - size[0] / 2.0).abs() < size[0] * 0.1);
    assert!(c.tissue_area_um2.is_some_and(|a| a > 0.0));
    assert!(c.density_per_mm2().is_some_and(|d| d > 0.0));
}

#[test]
fn touching_discs_split_roi_and_proposal() {
    let mut s = open_copy("count-discs");
    let slide = s.project().usable_slides_ordered()[0].id;
    let gland = s
        .add_label("Kelenjar", [0, 200, 0, 255], LabelKind::Normal)
        .unwrap();
    // Dua cakram bersentuhan + satu cakram terpisah (µm).
    s.fill_polygon(
        slide,
        gland,
        &circle([600.0, 600.0], 150.0),
        PaintMode::Replace,
    )
    .unwrap();
    s.fill_polygon(
        slide,
        gland,
        &circle([880.0, 600.0], 150.0),
        PaintMode::Replace,
    )
    .unwrap();
    s.fill_polygon(
        slide,
        gland,
        &circle([1500.0, 1500.0], 150.0),
        PaintMode::Replace,
    )
    .unwrap();
    let p = params();
    let c = s.count_objects_2d(slide, gland, None, &p).unwrap();
    assert_eq!(c.n(), 3, "{:?}", c.estimate.count.objects);
    let no_split = CountParams {
        split_enabled: false,
        ..p.clone()
    };
    assert_eq!(
        s.count_objects_2d(slide, gland, None, &no_split)
            .unwrap()
            .n(),
        2
    );

    // ROI hanya di sekitar cakram terpisah.
    let roi = Some(([1200.0, 1200.0], [1800.0, 1800.0]));
    let c_roi = s.count_objects_2d(slide, gland, roi, &p).unwrap();
    assert_eq!(c_roi.n(), 1);
    assert_eq!(c_roi.origin_um, [1200.0, 1200.0]);
    let u = c_roi.centroid_u(&c_roi.estimate.count.objects[0]);
    assert!(
        (u[0] - 1500.0).abs() < 20.0 && (u[1] - 1500.0).abs() < 20.0,
        "{u:?}"
    );
    // ROI yang memotong cakram: aturan bingkai mengecualikan sisi kiri/bawah.
    let cut = Some(([1500.0, 1200.0], [1800.0, 1800.0]));
    let c_cut = s.count_objects_2d(slide, gland, cut, &p).unwrap();
    assert_eq!((c_cut.n(), c_cut.estimate.count.n_excluded), (0, 1));
    let inc = CountParams {
        border: BorderRule::Include,
        ..p.clone()
    };
    assert_eq!(s.count_objects_2d(slide, gland, cut, &inc).unwrap().n(), 1);
    assert!(s
        .count_objects_2d(slide, gland, Some(([5.0e6, 5.0e6], [6.0e6, 6.0e6])), &p)
        .is_err());

    // Usulan: label tidak berubah.
    let store_before = s.labels(slide).unwrap().area_px(gland);
    let mut mask = ndarray::Array2::<u8>::zeros((200, 200));
    for ((y, x), v) in mask.indexed_iter_mut() {
        let (dx, dy) = (x as f64 - 100.0, y as f64 - 100.0);
        if dx * dx + dy * dy < 50.0 * 50.0 {
            *v = 1;
        }
    }
    let proposal = h3d_engine::Proposal {
        mask,
        mpp: 4.0,
        origin_um: [2000.0, 2000.0],
        label: gland,
        source: h3d_core::AnnotationSource::Classic { params_hash: 1 },
    };
    let cp = s.count_proposal_objects(slide, &proposal, &p).unwrap();
    assert_eq!(cp.n(), 1);
    assert_eq!(cp.source, CountSource::Proposal);
    assert_eq!(s.labels(slide).unwrap().area_px(gland), store_before);

    // JSON stabil dan deterministik.
    let j1 = h3d_engine::to_stable_json_of(&c).unwrap();
    let j2 = h3d_engine::to_stable_json_of(&s.count_objects_2d(slide, gland, None, &p).unwrap())
        .unwrap();
    assert_eq!(j1, j2);
    assert!(j1.contains("\"params_hash\""));
}

#[test]
fn unknown_label_and_bad_params_are_errors() {
    let mut s = open_copy("count-errors");
    let slide = s.project().usable_slides_ordered()[0].id;
    let tumor = s.label_by_name("Tumor").unwrap();
    assert!(s
        .count_objects_2d(slide, h3d_core::LabelId(250), None, &params())
        .is_err());
    let bad = CountParams {
        h_um: f64::NAN,
        ..params()
    };
    assert!(s.count_objects_2d(slide, tumor, None, &bad).is_err());
}
