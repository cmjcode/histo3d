//! Uji atribut (H10c): gambar IHC sintetis dengan fraksi positif diketahui,
//! dan jalur atribut → volume → analisis spasial → ekspor.

mod common;

use h3d_core::{
    AttrDef, AttrId, AttrKind, AttrSource, IhcPanel, IhcScoring, NoProgress, SlideId, Stain,
    StainPattern,
};
use h3d_engine::phantom::{write_phantom, PhantomOptions};
use h3d_engine::{AttrParams, Session};

/// Rata-rata berbobot `AttrStore` (bobot tidak tersimpan di berkas, jadi
/// setelah dimuat ulang tiap piksel berdata berbobot 1).
fn store_mean(store: &h3d_segment::AttrStore) -> (f64, usize) {
    let v = store.to_array();
    let w = store.weights_array();
    let mut sum = 0.0;
    let mut ws = 0.0;
    let mut n = 0;
    for (x, wt) in v.iter().zip(w.iter()) {
        if x.is_finite() {
            sum += f64::from(*x) * f64::from(*wt);
            ws += f64::from(*wt);
            n += 1;
        }
    }
    (sum / ws, n)
}

#[test]
fn synthetic_ihc_percent_positive_within_two_points() {
    let tmp = tempfile::tempdir().unwrap();
    let slides = tmp.path().join("slides");
    let fraction = 0.30;
    let manifest = write_phantom(
        &slides,
        &PhantomOptions {
            slices: 4,
            size_px: 64,
            ihc: Some(fraction),
            seed: 11,
            ..Default::default()
        },
    )
    .unwrap();
    let truth = manifest.ihc.clone().expect("kebenaran IHC");
    assert_eq!(truth.file, "ihc.png");
    let true_pct = 100.0 * truth.n_positive as f64 / truth.n_nuclei as f64;
    assert!((true_pct - 100.0 * fraction).abs() < 0.5);

    let dir = tmp.path().join("ihc.h3d");
    let mut s = Session::create(&dir).unwrap();
    let ids = s
        .import_slides(
            &[slides.join(&truth.file)],
            Some([truth.mpp, truth.mpp]),
            &NoProgress,
        )
        .unwrap();
    let slide = ids[0];
    s.set_stain(
        slide,
        Stain::Ihc,
        Some(IhcPanel {
            antibody: "Ki-67".into(),
            clone: Some("MIB-1".into()),
            ..Default::default()
        }),
    )
    .unwrap();

    let def = AttrDef {
        name: "Ki-67 % positif".into(),
        kind: AttrKind::IhcScore {
            slide,
            scoring: IhcScoring::PercentPositive,
        },
        ..Default::default()
    };
    let id = s.compute_attr(def.clone(), &[slide], &NoProgress).unwrap();
    assert_eq!(id, AttrId(0));
    let saved = s.project().attrs[0].clone();
    assert_eq!(saved.unit, "%");
    let AttrSource::Computed {
        method,
        params_hash,
    } = saved.source.clone()
    else {
        panic!("sumber harus Computed");
    };
    assert_eq!(method, "ihc-nuclear-v1");
    assert_ne!(params_hash, 0);
    assert!(s.bundle().attr_path(slide, &saved.file_stem()).is_file());

    let store = s
        .attr_store(slide, id)
        .unwrap()
        .expect("AttrStore tersimpan");
    assert_eq!(store.mpp_attr, 16.0);
    let (pct, n_px) = store_mean(&store);
    println!(
        "IHC sintetis: fraksi positif benar {true_pct:.2} %, terukur {pct:.2} % \
         ({} inti, {n_px} piksel atribut)",
        truth.n_nuclei
    );
    assert!((pct - true_pct).abs() <= 2.0, "pct_pos {pct} vs {true_pct}");

    // Laporan: tabel per irisan + parameter tercetak sebagai hash.
    let report = s.measure().unwrap();
    assert_eq!(report.attrs.len(), 1);
    assert!((report.attrs[0].per_slide[0].mean - pct).abs() < 1e-6);
    assert_eq!(report.attrs[0].params_hash, Some(params_hash));
    let md = h3d_io::render_markdown(&report, h3d_io::ClinicalScope::Omit);
    assert!(md.contains(&format!("{params_hash:016x}")));

    // Parameter sama → atribut yang sama dihitung ulang, hasil identik.
    let bytes = std::fs::read(s.bundle().attr_path(slide, &saved.file_stem())).unwrap();
    assert_eq!(
        s.compute_attr(def.clone(), &[slide], &NoProgress).unwrap(),
        id
    );
    assert_eq!(s.project().attrs.len(), 1);
    assert_eq!(
        std::fs::read(s.bundle().attr_path(slide, &saved.file_stem())).unwrap(),
        bytes,
        "penghitungan atribut deterministik"
    );

    // Parameter berbeda → atribut BARU; yang lama tidak ditimpa.
    let mut params = AttrParams::default();
    params.thresholds.t_d = 5.0;
    let id2 = s
        .compute_attr_with(def.clone(), &[slide], &params, &NoProgress)
        .unwrap();
    assert_ne!(id2, id);
    assert_eq!(s.project().attrs.len(), 2);
    assert_ne!(s.project().attrs[1].name, saved.name);
    assert_ne!(s.project().attrs[1].source, saved.source);
    assert_eq!(
        std::fs::read(s.bundle().attr_path(slide, &saved.file_stem())).unwrap(),
        bytes
    );
    let (pct_hi, _) = store_mean(&s.attr_store(slide, id2).unwrap().unwrap());
    assert!(
        pct_hi < pct,
        "ambang lebih tinggi → lebih sedikit positif ({pct_hi} vs {pct})"
    );

    // H-score: di antara %pos dan 3 × %pos.
    let h_id = s
        .compute_attr(
            AttrDef {
                name: "Ki-67 H-score".into(),
                kind: AttrKind::IhcScore {
                    slide,
                    scoring: IhcScoring::HScore,
                },
                ..Default::default()
            },
            &[slide],
            &NoProgress,
        )
        .unwrap();
    let (h, _) = store_mean(&s.attr_store(slide, h_id).unwrap().unwrap());
    assert!(h >= pct - 1.0 && h <= 3.0 * pct + 1.0, "H-score {h}");

    // Densitas inti: n / luas jaringan, dalam 20 % dari kebenaran.
    let d_id = s
        .compute_attr(
            AttrDef {
                name: "Densitas inti".into(),
                kind: AttrKind::NucleiDensity,
                ..Default::default()
            },
            &[slide],
            &NoProgress,
        )
        .unwrap();
    let (density, _) = store_mean(&s.attr_store(slide, d_id).unwrap().unwrap());
    let tissue_mm2 = std::f64::consts::PI * 225.0 * 205.0 / 1.0e6;
    let true_density = truth.n_nuclei as f64 / tissue_mm2;
    println!("densitas inti: benar {true_density:.0} sel/mm², terukur {density:.0} sel/mm²");
    assert!(
        (density - true_density).abs() / true_density < 0.2,
        "densitas {density} vs {true_density}"
    );

    // TPS tanpa masker tumor → "tidak tersedia", yang dihitung persen positif.
    let t_id = s
        .compute_attr(
            AttrDef {
                name: "PD-L1 TPS".into(),
                kind: AttrKind::IhcScore {
                    slide,
                    scoring: IhcScoring::Tps,
                },
                ..Default::default()
            },
            &[slide],
            &NoProgress,
        )
        .unwrap();
    let t_def = s
        .project()
        .attrs
        .iter()
        .find(|a| a.id == t_id)
        .unwrap()
        .clone();
    assert!(matches!(
        &t_def.source,
        AttrSource::Computed { method, .. } if method.contains("unavailable")
    ));
    assert!(s.attr_notes().iter().any(|n| n.contains("tidak tersedia")));

    // Pola membran → skor berbasis luas + peringatan di laporan.
    s.set_stain(
        slide,
        Stain::Ihc,
        Some(IhcPanel {
            antibody: "HER2".into(),
            pattern: StainPattern::Membranous,
            ..Default::default()
        }),
    )
    .unwrap();
    let m_id = s
        .compute_attr(
            AttrDef {
                name: "HER2 % luas".into(),
                kind: AttrKind::IhcScore {
                    slide,
                    scoring: IhcScoring::PercentPositive,
                },
                ..Default::default()
            },
            &[slide],
            &NoProgress,
        )
        .unwrap();
    let m_def = s
        .project()
        .attrs
        .iter()
        .find(|a| a.id == m_id)
        .unwrap()
        .clone();
    assert!(matches!(
        &m_def.source,
        AttrSource::Computed { method, .. } if method == "ihc-area-v1"
    ));
    let report = s.measure().unwrap();
    assert!(report.warnings.iter().any(|w| matches!(
        w,
        h3d_volume::Warning::MembraneAreaScore { name } if name == "HER2 % luas"
    )));
    // Densitas limfosit pada irisan non-IHC ditolak dengan pesan jelas.
    s.set_stain(slide, Stain::He, None).unwrap();
    assert!(s
        .compute_attr(
            AttrDef {
                name: "TIL".into(),
                kind: AttrKind::TilDensity,
                ..Default::default()
            },
            &[slide],
            &NoProgress,
        )
        .is_err());
    assert!(s.compute_attr(def, &[SlideId(99)], &NoProgress).is_err());
    // Proyek tetap terbuka kembali dengan semua atribut.
    let n = s.project().attrs.len();
    drop(s);
    assert_eq!(Session::open(&dir).unwrap().project().attrs.len(), n);
}

#[test]
fn attribute_volumes_flow_into_spatial_analysis_and_exports() {
    use h3d_core::LabelId;
    use h3d_engine::ExportKind;

    let mut s = common::open_copy("atribut-volume");
    let tumor = LabelId(1);
    // Irisan 11 dijadikan IHC tanpa label tumor sendiri: masker tumornya
    // harus diambil dari irisan H&E tetangga lewat T_he⁻¹ ∘ T_ihc (8.12.3).
    let own_area = s.labels(SlideId(11)).unwrap().area_px(tumor);
    assert!(own_area > 0);
    s.edit_labels(SlideId(11), None, "Kosongkan", |st| {
        for (tx, ty) in st.tile_keys() {
            st.remove_tile(tx, ty);
        }
    })
    .unwrap();
    s.set_stain(SlideId(11), Stain::Ihc, Some(IhcPanel::default()))
        .unwrap();
    s.save().unwrap();

    // TPS pada irisan IHC itu: masker tumor tersedia dari tetangga → bukan
    // jalur "tidak tersedia".
    let tps = s
        .compute_attr(
            AttrDef {
                name: "PD-L1 TPS".into(),
                kind: AttrKind::IhcScore {
                    slide: SlideId(11),
                    scoring: IhcScoring::Tps,
                },
                ..Default::default()
            },
            &[SlideId(11)],
            &NoProgress,
        )
        .unwrap();
    let def = s
        .project()
        .attrs
        .iter()
        .find(|a| a.id == tps)
        .unwrap()
        .clone();
    assert!(
        matches!(&def.source, AttrSource::Computed { method, .. } if method == "ihc-tps-v1"),
        "{:?}",
        def.source
    );

    // Densitas inti pada tiga irisan H&E berurutan → volume atribut.
    let slides = [SlideId(8), SlideId(9), SlideId(10)];
    let dens = s
        .compute_attr(
            AttrDef {
                name: "Densitas inti".into(),
                kind: AttrKind::NucleiDensity,
                ..Default::default()
            },
            &slides,
            &NoProgress,
        )
        .unwrap();
    for id in slides {
        assert!(s.attr_store(id, dens).unwrap().is_some());
    }
    assert!(s.attr_store(SlideId(3), dens).unwrap().is_none());

    let analysis = s.analyze_spatial(&NoProgress).unwrap();
    let entry = analysis
        .attr_stats
        .iter()
        .find(|e| e.name == "Densitas inti")
        .expect("statistik atribut atas tumor");
    assert!(entry.stats.n > 0);
    assert!(entry.params_hash.is_some());
    // Hanya lapis di sekitar irisan beratribut yang berisi (sisanya NODATA).
    let layers = entry.stats.per_layer.len();
    println!(
        "atribut pada kisi: {} voxel tumor sah di {layers} lapis, rata-rata {:.1} {}",
        entry.stats.n, entry.stats.mean, entry.unit
    );
    let grid_layers = s.voxel_masks().unwrap()[0].1.grid.nz();
    assert!(
        layers >= 3 && layers < grid_layers,
        "{layers} dari {grid_layers} lapis"
    );

    // Skalar atribut terproyeksi ke mesh; verteks di luar jangkauan = NaN.
    let meshes = s.meshes().unwrap();
    let sc = meshes[0]
        .scalars
        .iter()
        .find(|x| x.name == "attr:Densitas inti")
        .expect("skalar atribut pada mesh");
    assert_eq!(sc.values.len(), meshes[0].positions_um.len());
    assert!(sc.values.iter().any(|v| v.is_nan()) && sc.values.iter().any(|v| v.is_finite()));

    // Laporan: tabel per irisan dengan rata-rata intratumoral dari masker
    // tetangga (irisan 11 tidak punya label sendiri).
    let report = s.measure().unwrap();
    let tps_rows = &report.attrs.iter().find(|a| a.name == "PD-L1 TPS");
    assert!(tps_rows.is_some());
    let dens_report = report
        .attrs
        .iter()
        .find(|a| a.name == "Densitas inti")
        .unwrap();
    assert_eq!(dens_report.per_slide.len(), 3);
    assert!(dens_report
        .per_slide
        .iter()
        .all(|r| r.intratumoral.is_some()));
    assert!(report.analysis.is_some());

    // Ekspor: NIfTI atribut, GLB berwarna atribut (nama tanpa awalan), GeoJSON.
    let out = s.path().join("ekspor");
    let files = s.export_files(ExportKind::Nifti, &out).unwrap();
    let names: Vec<String> = files
        .iter()
        .map(|f| f.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert!(names.contains(&"labels.nii.gz".to_owned()), "{names:?}");
    assert!(
        names.contains(&"densitas_inti.nii.gz".to_owned()),
        "{names:?}"
    );
    let raw = h3d_io::gunzip(&std::fs::read(out.join("densitas_inti.nii.gz")).unwrap()).unwrap();
    let (hdr, data) = h3d_io::decode_nifti(&raw).unwrap();
    assert_eq!(hdr.datatype, 16);
    assert_eq!(
        data.len(),
        4 * hdr.dim[1] as usize * hdr.dim[2] as usize * hdr.dim[3] as usize
    );
    s.export(
        ExportKind::Glb {
            scalar: Some("densitas inti".into()),
        },
        &out,
    )
    .unwrap();
    let (doc, _) = h3d_io::parse_glb(&std::fs::read(out.join("mesh.glb")).unwrap()).unwrap();
    let attrs = &doc["meshes"][0]["primitives"][0]["attributes"];
    assert!(attrs.get("COLOR_0").is_some());
    assert!(attrs.get("_H3D_ATTR_DENSITAS_INTI").is_some(), "{attrs}");
    s.export(ExportKind::GeoJson, &out).unwrap();
    let gj: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("0009.geojson")).unwrap()).unwrap();
    let m = &gj["features"][0]["properties"]["measurements"];
    assert!(m["area_um2"].as_f64().unwrap() > 0.0);
    assert!(m.get("Densitas inti").is_some(), "{m}");

    // Menghitung atribut lagi membuat analisis lama basi.
    let p = AttrParams {
        window_px: 1,
        ..Default::default()
    };
    s.compute_attr_with(
        AttrDef {
            name: "Densitas inti".into(),
            kind: AttrKind::NucleiDensity,
            ..Default::default()
        },
        &slides,
        &p,
        &NoProgress,
    )
    .unwrap();
    assert!(
        s.measure().unwrap().analysis.is_none(),
        "analisis lama harus dianggap basi"
    );
    assert!(s.export(ExportKind::Analysis, &out).is_err());
}
