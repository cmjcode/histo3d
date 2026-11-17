//! Uji penghitungan sel otomatis: pendeteksi klasik pada fantom IHC dengan
//! jumlah inti yang diketahui, ROI, dan keluaran JSON stabil.

use h3d_core::{NoProgress, SlideId};
use h3d_engine::phantom::{write_phantom, PhantomOptions};
use h3d_engine::{
    CellMethod, ClassicNucleiDetector, Session, WatershedNucleiDetector, WholeRegionDetector,
};
use h3d_imgproc::StainMatrix;

fn ihc_project(name: &str, seed: u64) -> (tempfile::TempDir, Session, SlideId, usize, f64) {
    let tmp = tempfile::tempdir().unwrap();
    let slides = tmp.path().join("slides");
    let manifest = write_phantom(
        &slides,
        &PhantomOptions {
            slices: 3,
            size_px: 96,
            ihc: Some(0.3),
            seed,
            ..Default::default()
        },
    )
    .unwrap();
    let truth = manifest.ihc.clone().expect("kebenaran IHC");
    let dir = tmp.path().join(format!("{name}.h3d"));
    let mut s = Session::create(&dir).unwrap();
    let ids = s
        .import_slides(
            &[slides.join(&truth.file)],
            Some([truth.mpp, truth.mpp]),
            &NoProgress,
        )
        .unwrap();
    (tmp, s, ids[0], truth.n_nuclei, truth.mpp)
}

fn classic(mpp: f64) -> CellMethod {
    CellMethod::Classic(ClassicNucleiDetector {
        mpp,
        tile_size: 256,
        stain: StainMatrix::h_dab(),
        h_over_e: false,
        ..ClassicNucleiDetector::default()
    })
}

#[test]
fn classic_count_matches_phantom_truth_within_ten_percent() {
    let (_tmp, s, slide, n_truth, mpp) = ihc_project("sel", 11);
    assert!(n_truth > 50, "fantom terlalu sedikit inti: {n_truth}");
    let c = s
        .count_cells(slide, None, &classic(mpp), false, &NoProgress)
        .unwrap();
    let n = c.n() as f64;
    assert!(
        (n - n_truth as f64).abs() <= 0.1 * n_truth as f64,
        "terdeteksi {n} vs kebenaran {n_truth}"
    );
    assert!(c.detection.n_min <= c.n() && c.n() <= c.detection.n_max);
    assert!(c.detection.tiles_run >= 1);
    assert_eq!(c.detection.method, "log-classic-v1");
    assert!(c.tissue_area_um2.is_some_and(|a| a > 0.0));
    assert!(c.density_per_mm2().is_some_and(|d| d > 0.0));
    assert!(c.median_area_um2() > 0.0);
    assert!(c.detection.nuclei.iter().all(|n| n.polygon_um.is_empty()));

    // ROI separuh kiri: lebih sedikit, dan semua pusat di dalam ROI.
    let size = s.slide(slide).unwrap().size_um();
    let roi = Some(([0.0, 0.0], [size[0] / 2.0, size[1]]));
    let half = s
        .count_cells(slide, roi, &classic(mpp), true, &NoProgress)
        .unwrap();
    assert!(half.n() < c.n());
    assert!(half
        .detection
        .nuclei
        .iter()
        .all(|n| n.center_um[0] <= size[0] / 2.0 && !n.polygon_um.is_empty()));

    // Deterministik dan JSON stabil.
    let again = s
        .count_cells(slide, None, &classic(mpp), false, &NoProgress)
        .unwrap();
    assert_eq!(c, again);
    let j = h3d_engine::to_stable_json_of(&c).unwrap();
    assert!(j.contains("\"params_hash\"") && j.contains("\"method\""));
}

#[test]
fn watershed_count_matches_phantom_truth_within_ten_percent() {
    let (_tmp, s, slide, n_truth, mpp) = ihc_project("sel-ws", 11);
    let method = CellMethod::Watershed(WatershedNucleiDetector {
        mpp,
        tile_size: 256,
        stain: StainMatrix::h_dab(),
        h_over_e: false,
        adaptive_threshold: false,
        ..WatershedNucleiDetector::default()
    });
    let c = s
        .count_cells(slide, None, &method, true, &NoProgress)
        .unwrap();
    let n = c.n() as f64;
    assert!(
        (n - n_truth as f64).abs() <= 0.1 * n_truth as f64,
        "terdeteksi {n} vs kebenaran {n_truth}"
    );
    assert_eq!(c.detection.method, "watershed-classic-v1");
    // Kontur mengikuti tepi wilayah watershed, bukan sinar tetap.
    for k in &c.detection.nuclei {
        assert!(k.polygon_um.len() >= 12);
        let a = h3d_imgproc::signed_area(&k.polygon_um);
        assert!(
            (a - k.area_um2).abs() / k.area_um2 < 0.25,
            "{a} vs {}",
            k.area_um2
        );
    }
}

#[test]
fn whole_regions_never_split_and_respect_memory_budget() {
    let (_tmp, mut s, slide, n_truth, mpp) = ihc_project("sel-utuh", 11);
    let whole = |m: f64| {
        CellMethod::WholeRegions(WholeRegionDetector {
            mpp: m,
            tile_size: 256,
            stain: StainMatrix::h_dab(),
            h_over_e: false,
            adaptive_threshold: false,
            ..WholeRegionDetector::default()
        })
    };
    let c = s
        .count_cells(slide, None, &whole(mpp), true, &NoProgress)
        .unwrap();
    assert_eq!(c.detection.method, "regions-classic-v1");
    // Tanpa pemisahan: inti yang bersentuhan menyatu, jadi jumlahnya tidak
    // melebihi kebenaran (+ toleransi serpihan).
    assert!(
        c.n() > 0 && c.n() as f64 <= 1.1 * n_truth as f64,
        "{} vs kebenaran {n_truth}",
        c.n()
    );
    assert!(c.detection.n_min <= c.n() && c.n() <= c.detection.n_max);
    assert!(c.detection.nuclei.iter().all(|k| k.polygon_um.len() >= 3));
    assert_eq!(c.detection.mpp, mpp);
    let again = s
        .count_cells(slide, None, &whole(mpp), true, &NoProgress)
        .unwrap();
    assert_eq!(c, again);

    // Anggaran kecil → resolusi kerja diturunkan, bukan error.
    s.set_budgets(h3d_core::Budgets {
        cell_region_bytes: 64 * 1024,
        ..h3d_core::Budgets::MACOS
    });
    let coarse = s
        .count_cells(slide, None, &whole(mpp), false, &NoProgress)
        .unwrap();
    assert!(coarse.detection.mpp > mpp, "mpp {}", coarse.detection.mpp);
    assert_ne!(coarse.detection.params_hash, c.detection.params_hash);

    let bad = CellMethod::WholeRegions(WholeRegionDetector {
        close_radius_um: f64::NAN,
        ..WholeRegionDetector::default()
    });
    assert!(s
        .count_cells(slide, None, &bad, false, &NoProgress)
        .is_err());
}

#[test]
fn invalid_roi_model_and_params_are_errors() {
    let (_tmp, s, slide, _n, mpp) = ihc_project("sel-galat", 5);
    assert!(s
        .count_cells(
            slide,
            Some(([1.0e9, 1.0e9], [2.0e9, 2.0e9])),
            &classic(mpp),
            false,
            &NoProgress
        )
        .is_err());
    let bad = CellMethod::Classic(ClassicNucleiDetector {
        mpp: -1.0,
        ..ClassicNucleiDetector::default()
    });
    assert!(s
        .count_cells(slide, None, &bad, false, &NoProgress)
        .is_err());
    let missing = CellMethod::StarDist(std::path::PathBuf::from("/tidak/ada/model"));
    assert!(s
        .count_cells(slide, None, &missing, false, &NoProgress)
        .is_err());
    assert!(s.list_nuclei_models().is_empty());
}
