//! Jalur atribut yang sebelumnya tidak teruji (8.12.5), di atas fantom IHC
//! sintetis dengan kromogen terkendali per inti: TPS dan CPS dengan masker
//! Tumor dan Stroma sungguhan, densitas TIL intratumoral vs peritumoral, serta
//! skor Allred dan intensitas 0–3. Kebenaran dihitung dari posisi inti yang
//! diketahui dan masker label yang benar-benar tersimpan.

use std::path::Path;

use h3d_core::{
    AttrDef, AttrId, AttrKind, AttrSource, IhcPanel, IhcScoring, LabelId, NoProgress, SlideId,
    Stain,
};
use h3d_engine::{AttrParams, PaintMode, Session};
use h3d_imgproc::phantom::{ellipsoid_mask, synthetic_ihc_with, Rng, TruthNucleusDab};

const TUMOR: LabelId = LabelId(1);
const STROMA: LabelId = LabelId(2);
/// Konsentrasi DAB per kelas intensitas 1/2/3 (ambang OD 0,15 / 0,35 / 0,60).
const DAB: [f32; 3] = [0.25, 0.47, 0.90];

fn write_png(path: &Path, rgba: &[u8], side: u32) {
    image::RgbaImage::from_raw(side, side, rgba.to_vec())
        .expect("buffer RGBA")
        .save(path)
        .expect("tulis PNG");
}

/// Proyek berisi satu irisan IHC `side × side` px pada 1 µm/px.
fn ihc_project(
    dir: &Path,
    side: usize,
    seed: u64,
    dab_of: &dyn Fn(f64, f64, &mut Rng) -> f32,
) -> (Session, SlideId, Vec<TruthNucleusDab>) {
    let c = side as f64 / 2.0;
    let mask = ellipsoid_mask(
        [side, side],
        [c, c],
        [0.47 * side as f64, 0.44 * side as f64],
    );
    let (rgba, truth) = synthetic_ihc_with(&mask, seed, 1.0, dab_of);
    let png = dir.join("ihc.png");
    write_png(&png, &rgba, side as u32);
    let mut s = Session::create(&dir.join("p.h3d")).unwrap();
    let ids = s
        .import_slides(&[png], Some([1.0, 1.0]), &NoProgress)
        .unwrap();
    s.set_stain(ids[0], Stain::Ihc, Some(IhcPanel::default()))
        .unwrap();
    (s, ids[0], truth)
}

fn circle(c: [f64; 2], r: f64) -> Vec<[f64; 2]> {
    (0..180)
        .map(|k| {
            let a = k as f64 / 180.0 * std::f64::consts::TAU;
            [c[0] + r * a.cos(), c[1] + r * a.sin()]
        })
        .collect()
}

/// Rata-rata tak berbobot dan median nilai `AttrStore` (piksel berdata).
fn store_stats(s: &Session, slide: SlideId, id: AttrId) -> (f64, f64, Vec<f32>) {
    let store = s.attr_store(slide, id).unwrap().expect("AttrStore");
    let mut v: Vec<f32> = store
        .to_array()
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .collect();
    assert!(!v.is_empty());
    v.sort_by(|a, b| a.total_cmp(b));
    let mean = v.iter().map(|x| f64::from(*x)).sum::<f64>() / v.len() as f64;
    (mean, f64::from(v[v.len() / 2]), v)
}

fn ihc_def(name: &str, slide: SlideId, scoring: IhcScoring) -> AttrDef {
    AttrDef {
        name: name.into(),
        kind: AttrKind::IhcScore { slide, scoring },
        ..Default::default()
    }
}

fn method_of(s: &Session, id: AttrId) -> String {
    match &s
        .project()
        .attrs
        .iter()
        .find(|a| a.id == id)
        .unwrap()
        .source
    {
        AttrSource::Computed { method, .. } => method.clone(),
        other => panic!("sumber {other:?}"),
    }
}

#[test]
fn tps_and_cps_with_real_tumour_and_stroma_masks() {
    let tmp = tempfile::tempdir().unwrap();
    let side = 640usize;
    // Sarang tumor berselang-seling dengan stroma (pita 24 µm), agar tiap
    // jendela agregasi 80 µm memuat keduanya. Positif: 30 % sel tumor,
    // 20 % sel imun.
    let in_tumor_band = |y: f64| ((y / 24.0).floor() as i64) % 2 == 0;
    let dab_of = move |_x: f64, y: f64, r: &mut Rng| {
        let p = if in_tumor_band(y) { 0.30 } else { 0.20 };
        if r.next_f64() < p {
            DAB[2]
        } else {
            0.0
        }
    };
    let (mut s, slide, truth) = ihc_project(tmp.path(), side, 21, &dab_of);

    // Hanya label Tumor dulu: TPS tersedia, CPS belum (butuh Stroma).
    let mut y = 0.0;
    while y < side as f64 {
        let band = [
            [0.0, y],
            [side as f64, y],
            [side as f64, y + 24.0],
            [0.0, y + 24.0],
        ];
        if in_tumor_band(y + 1.0) {
            s.fill_polygon(slide, TUMOR, &band, PaintMode::Replace)
                .unwrap();
        }
        y += 24.0;
    }
    s.save().unwrap();
    let cps0 = s
        .compute_attr(
            ihc_def("CPS tanpa stroma", slide, IhcScoring::Cps),
            &[slide],
            &NoProgress,
        )
        .unwrap();
    assert!(
        method_of(&s, cps0).contains("unavailable"),
        "{}",
        method_of(&s, cps0)
    );
    assert!(s.attr_notes().iter().any(|n| n.contains("tidak tersedia")));

    // Tambah Stroma di sisa jaringan.
    let whole = [
        [0.0, 0.0],
        [side as f64, 0.0],
        [side as f64, side as f64],
        [0.0, side as f64],
    ];
    s.fill_polygon(slide, STROMA, &whole, PaintMode::OnlyBackground)
        .unwrap();
    s.save().unwrap();

    // Kebenaran dari posisi inti dan masker label yang tersimpan.
    let labels = s.labels(slide).unwrap().clone();
    let (mut ta, mut tp, mut ip) = (0.0f64, 0.0f64, 0.0f64);
    for &(x, yv, dab) in &truth {
        match labels.get_um([x, yv]) {
            l if l == TUMOR.0 => {
                ta += 1.0;
                if dab > 0.0 {
                    tp += 1.0;
                }
            }
            l if l == STROMA.0 && dab > 0.0 => ip += 1.0,
            _ => {}
        }
    }
    assert!(
        ta > 300.0 && ip > 50.0,
        "fantom: {ta} sel tumor, {ip} imun positif"
    );
    let tps_true = 100.0 * tp / ta;
    let cps_true = (100.0 * (tp + ip) / ta).min(100.0);

    let tps = s
        .compute_attr(
            ihc_def("PD-L1 TPS", slide, IhcScoring::Tps),
            &[slide],
            &NoProgress,
        )
        .unwrap();
    assert_eq!(method_of(&s, tps), "ihc-tps-v1");
    assert!(s.attr_notes().is_empty(), "{:?}", s.attr_notes());
    let cps = s
        .compute_attr(
            ihc_def("PD-L1 CPS", slide, IhcScoring::Cps),
            &[slide],
            &NoProgress,
        )
        .unwrap();
    assert_eq!(method_of(&s, cps), "ihc-cps-v1");
    let unit = |id: AttrId| {
        s.project()
            .attrs
            .iter()
            .find(|a| a.id == id)
            .unwrap()
            .unit
            .clone()
    };
    assert_eq!((unit(tps).as_str(), unit(cps).as_str()), ("%", "CPS"));

    let (tps_mean, _, tps_v) = store_stats(&s, slide, tps);
    let (cps_mean, _, cps_v) = store_stats(&s, slide, cps);
    println!(
        "TPS: benar {tps_true:.1}, terukur {tps_mean:.1}; CPS: benar {cps_true:.1}, terukur {cps_mean:.1} \
         ({ta} sel tumor, {tp} tumor positif, {ip} imun positif)"
    );
    assert!(
        (tps_mean - tps_true).abs() <= 3.0,
        "TPS {tps_mean} vs {tps_true}"
    );
    assert!(
        (cps_mean - cps_true).abs() <= 5.0,
        "CPS {cps_mean} vs {cps_true}"
    );
    assert!(cps_mean > tps_mean + 10.0, "CPS memuat sel imun positif");
    assert!(tps_v.iter().all(|v| (0.0..=100.0).contains(v)));
    assert!(
        cps_v.iter().all(|v| (0.0..=100.0).contains(v)),
        "CPS dijepit ≤ 100"
    );

    // Laporan: baris per irisan memuat rata-rata yang sama.
    let rep = s.measure().unwrap();
    let row = |name: &str| {
        rep.attrs
            .iter()
            .find(|a| a.name == name)
            .unwrap_or_else(|| panic!("atribut {name}"))
            .per_slide[0]
            .clone()
    };
    assert!((row("PD-L1 TPS").mean - tps_mean).abs() < 1e-6);
    assert!((row("PD-L1 CPS").mean - cps_mean).abs() < 1e-6);
}

#[test]
fn til_density_intratumoral_vs_peritumoral_matches_known_nuclei() {
    let tmp = tempfile::tempdir().unwrap();
    let side = 768usize;
    let c = [side as f64 / 2.0; 2];
    let (r_tumor, band) = (170.0, 100.0);
    // CD8: 10 % positif di dalam tumor, 60 % di pita peritumoral, 20 % di luar.
    let dab_of = move |x: f64, y: f64, r: &mut Rng| {
        let d = ((x - c[0]).powi(2) + (y - c[1]).powi(2)).sqrt();
        let p = if d <= r_tumor {
            0.10
        } else if d <= r_tumor + band {
            0.60
        } else {
            0.20
        };
        if r.next_f64() < p {
            DAB[2]
        } else {
            0.0
        }
    };
    let (mut s, slide, truth) = ihc_project(tmp.path(), side, 5, &dab_of);
    s.fill_polygon(slide, TUMOR, &circle(c, r_tumor), PaintMode::Replace)
        .unwrap();
    s.save().unwrap();
    let mut cfg = s.project().spatial.clone();
    cfg.peri_band_um = band;
    s.set_spatial_config(cfg);
    s.save().unwrap();

    // Kebenaran: inti positif per mm² di dalam masker Tumor tersimpan, dan
    // di pita ≤ `band` di luar tumor.
    let labels = s.labels(slide).unwrap().clone();
    let tumor_mm2 = labels.area_um2(TUMOR) / 1e6;
    let ring_mm2 = std::f64::consts::PI * ((r_tumor + band).powi(2)) / 1e6 - tumor_mm2;
    let (mut n_intra, mut n_peri) = (0.0f64, 0.0f64);
    for &(x, y, dab) in &truth {
        if dab <= 0.0 {
            continue;
        }
        let d = ((x - c[0]).powi(2) + (y - c[1]).powi(2)).sqrt();
        if labels.get_um([x, y]) == TUMOR.0 {
            n_intra += 1.0;
        } else if d <= r_tumor + band {
            n_peri += 1.0;
        }
    }
    let (intra_true, peri_true) = (n_intra / tumor_mm2, n_peri / ring_mm2);
    assert!(n_intra > 20.0 && peri_true > 3.0 * intra_true);

    // Sebelum agregasi per kompartemen (jendela 80 µm mencampur kedua sisi
    // batas tumor): intratumoral +38,4 %, peritumoral −10,7 %.
    // Tile kerja kecil (4 × 4 tile bertumpang tindih) agar sambungan antar
    // tile ikut teruji terhadap kebenaran.
    let params = AttrParams {
        tile_px: 256,
        overlap_px: 32,
        ..Default::default()
    };
    let def = AttrDef {
        name: "CD8 TIL".into(),
        kind: AttrKind::TilDensity,
        ..Default::default()
    };
    let id = s
        .compute_attr_with(def.clone(), &[slide], &params, &NoProgress)
        .unwrap();
    assert_eq!(method_of(&s, id), "til-ihc-density-v1");
    let rep = s.measure().unwrap();
    let a = rep.attrs.iter().find(|a| a.name == "CD8 TIL").unwrap();
    assert_eq!(a.unit, "sel/mm²");
    let row = &a.per_slide[0];
    let intra = row.intratumoral.expect("intratumoral");
    let peri = row.peritumoral.expect("peritumoral");
    println!(
        "TIL: intratumoral benar {intra_true:.0}, terukur {intra:.0} sel/mm² ({:+.1} %); \
         peritumoral benar {peri_true:.0}, terukur {peri:.0} sel/mm² ({:+.1} %)",
        (intra / intra_true - 1.0) * 100.0,
        (peri / peri_true - 1.0) * 100.0
    );
    assert!(
        (intra / intra_true - 1.0).abs() <= 0.10,
        "intratumoral {intra} vs {intra_true}"
    );
    assert!(
        (peri / peri_true - 1.0).abs() <= 0.10,
        "peritumoral {peri} vs {peri_true}"
    );
    assert!(peri > 3.0 * intra, "kontras intra/peri terjaga");

    // Tile kerja tunggal memberi hasil yang sama (sambungan tile tidak
    // menggandakan atau menghilangkan inti).
    let single = s
        .compute_attr_with(
            AttrDef {
                name: "CD8 TIL satu tile".into(),
                ..def
            },
            &[slide],
            &AttrParams::default(),
            &NoProgress,
        )
        .unwrap();
    let (m_tiled, _, _) = store_stats(&s, slide, id);
    let (m_single, _, _) = store_stats(&s, slide, single);
    println!("TIL rata-rata: bertile {m_tiled:.1}, satu tile {m_single:.1}");
    assert!(
        (m_tiled / m_single - 1.0).abs() <= 0.03,
        "{m_tiled} vs {m_single}"
    );
}

#[test]
fn allred_and_intensity_scores_match_known_classes() {
    let tmp = tempfile::tempdir().unwrap();
    // (fraksi positif, kelas intensitas) → Allred = proporsi + intensitas.
    let cases: [(f64, usize, f32, f32); 3] = [
        (0.20, 1, 3.0 + 1.0, 20.0),  // 10–33 % → 3; lemah
        (0.50, 2, 4.0 + 2.0, 100.0), // 33–66 % → 4; sedang
        (0.85, 3, 5.0 + 3.0, 255.0), // > 66 % → 5; kuat
    ];
    for (k, (fraction, class, allred_want, h_want)) in cases.into_iter().enumerate() {
        let dir = tmp.path().join(format!("kasus{k}"));
        std::fs::create_dir_all(&dir).unwrap();
        let dab_of = move |_x: f64, _y: f64, r: &mut Rng| {
            if r.next_f64() < fraction {
                DAB[class - 1]
            } else {
                0.0
            }
        };
        let (mut s, slide, truth) = ihc_project(&dir, 512, 40 + k as u64, &dab_of);
        let pct_true =
            100.0 * truth.iter().filter(|t| t.2 > 0.0).count() as f64 / truth.len() as f64;
        let mut got = Vec::new();
        for (name, scoring) in [
            ("Allred", IhcScoring::Allred),
            ("Intensitas", IhcScoring::Intensity03),
            ("H-score", IhcScoring::HScore),
            ("Persen", IhcScoring::PercentPositive),
        ] {
            let id = s
                .compute_attr(ihc_def(name, slide, scoring), &[slide], &NoProgress)
                .unwrap();
            assert_eq!(method_of(&s, id), "ihc-nuclear-v1");
            got.push(store_stats(&s, slide, id));
        }
        let frac_eq = |v: &[f32], want: f32| {
            v.iter().filter(|x| (**x - want).abs() < 1e-3).count() as f64 / v.len() as f64
        };
        println!(
            "kelas {class}, {pct_true:.1} % positif: Allred median {} ({:.0} % piksel = {allred_want}), \
             intensitas median {} ({:.0} % piksel = {class}), H-score {:.1} (benar {:.1}), %pos {:.1}",
            got[0].1,
            100.0 * frac_eq(&got[0].2, allred_want),
            got[1].1,
            100.0 * frac_eq(&got[1].2, class as f32),
            got[2].0,
            pct_true * class as f64,
            got[3].0
        );
        assert_eq!(got[0].1 as f32, allred_want, "Allred kelas {class}");
        assert!(frac_eq(&got[0].2, allred_want) >= 0.7);
        assert_eq!(got[1].1 as f32, class as f32, "intensitas kelas {class}");
        assert!(frac_eq(&got[1].2, class as f32) >= 0.9);
        assert!(got[0]
            .2
            .iter()
            .all(|v| (0.0..=8.0).contains(v) && v.fract() == 0.0));
        assert!(got[1]
            .2
            .iter()
            .all(|v| (0.0..=3.0).contains(v) && v.fract() == 0.0));
        // H-score = Σ i × %_i; di sini semua positif berkelas sama.
        assert!(
            (got[2].0 - pct_true * class as f64).abs() <= 0.06 * h_want.max(60.0) as f64,
            "H-score {} vs {}",
            got[2].0,
            pct_true * class as f64
        );
        assert!(
            (got[3].0 - pct_true).abs() <= 2.5,
            "%pos {} vs {pct_true}",
            got[3].0
        );
    }
}
