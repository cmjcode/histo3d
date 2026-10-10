//! Uji ujung-ke-ujung pada fantom IHC sintetis: deteksi inti dan skor.

use h3d_imgproc::color::{deconvolve, StainMatrix};
use h3d_imgproc::ihc::{score_nuclear, IhcThresholds};
use h3d_imgproc::nuclei::{detect_nuclei, measure_chromogen, NucleiParams};
use h3d_imgproc::phantom::{ellipsoid_mask, match_points, synthetic_ihc};
use h3d_imgproc::tissue::tissue_mask;

#[test]
fn nuclei_detection_f1_and_ihc_pct_on_phantom() {
    let mpp = 0.5;
    let (w, h) = (600usize, 450usize);
    let mask = ellipsoid_mask([w, h], [300.0, 225.0], [260.0, 190.0]);
    for &(seed, frac) in &[(1u64, 0.30f64), (2, 0.60), (3, 0.10)] {
        let (rgba, truth) = synthetic_ihc(&mask, seed, frac, mpp);
        assert!(
            truth.len() > 100,
            "fantom terlalu sedikit inti: {}",
            truth.len()
        );
        let m = StainMatrix::h_dab();
        let [hch, dch, _] = deconvolve(&rgba, w, h, &m).unwrap();
        let tmask = tissue_mask(&rgba, w, h, mpp).unwrap();
        let mut nuclei = detect_nuclei(&hch, mpp, Some(&tmask), &NucleiParams::default());
        measure_chromogen(&mut nuclei, &dch, mpp);

        let det: Vec<(f64, f64)> = nuclei.iter().map(|n| (n.x_um, n.y_um)).collect();
        let tru: Vec<(f64, f64)> = truth.iter().map(|t| (t.0, t.1)).collect();
        let (tp, fp, fn_) = match_points(&det, &tru, 4.0);
        let prec = tp as f64 / (tp + fp).max(1) as f64;
        let rec = tp as f64 / (tp + fn_).max(1) as f64;
        let f1 = 2.0 * prec * rec / (prec + rec).max(1e-9);
        eprintln!(
            "seed {seed}: F1 {f1:.3} (tp {tp} fp {fp} fn {fn_}), n_truth {}",
            truth.len()
        );
        assert!(
            f1 >= 0.9,
            "seed {seed}: F1 {f1:.3} (tp {tp} fp {fp} fn {fn_})"
        );

        let score = score_nuclear(&nuclei, &IhcThresholds::default());
        let want = 100.0 * frac;
        eprintln!("seed {seed}: pct_pos {:.2} vs {want:.2}", score.pct_pos);
        assert!(
            (score.pct_pos as f64 - want).abs() <= 2.0,
            "seed {seed}: pct_pos {} vs {want}",
            score.pct_pos
        );
    }
}
