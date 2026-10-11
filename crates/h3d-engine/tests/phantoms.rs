//! Fantom 8.12.11 lewat mesin: tiap varian `h3d phantom` diimpor, diberi
//! label dari maskernya, lalu dianalisis; hasil dibandingkan dengan
//! kebenaran analitik. Transformasi per irisan dinolkan di sini agar yang
//! diuji geometri fantom dan jalur analisis, bukan registrasi (registrasi
//! diuji di `pipeline.rs`).

use std::path::{Path, PathBuf};

use h3d_core::{NoProgress, SlideId};
use h3d_engine::phantom::{
    read_phantom_manifest, write_phantom, PhantomManifest, PhantomOptions, PhantomShapeKind,
};
use h3d_engine::{ImportOptions, Session, SpatialAnalysis};

fn base(shape: PhantomShapeKind) -> PhantomOptions {
    PhantomOptions {
        shape,
        slices: 21,
        seed: 2,
        size_px: 128,
        mpp: 16.0,
        max_rotation_deg: 0.0,
        max_shift_frac: 0.0,
        ..Default::default()
    }
}

struct Run {
    _tmp: tempfile::TempDir,
    manifest: PhantomManifest,
    session: Session,
}

fn run(opts: &PhantomOptions) -> Run {
    let tmp = tempfile::tempdir().unwrap();
    let ph: PathBuf = tmp.path().join("ph");
    let manifest = write_phantom(&ph, opts).expect("fantom");
    assert!(
        read_phantom_manifest(&ph).unwrap() == manifest,
        "truth.json harus terbaca kembali persis"
    );
    let mut s = Session::create(&tmp.path().join("p.h3d")).unwrap();
    let paths: Vec<PathBuf> = manifest.slices.iter().map(|x| ph.join(&x.file)).collect();
    s.import_slides_with(
        &paths,
        &ImportOptions {
            mpp: Some([manifest.mpp, manifest.mpp]),
            embed: false,
            spacing_um: Some(manifest.spacing_um),
        },
        &NoProgress,
    )
    .unwrap();
    import_labels(&mut s, &ph, &manifest);
    s.save().unwrap();
    Run {
        _tmp: tmp,
        manifest,
        session: s,
    }
}

fn import_labels(s: &mut Session, ph: &Path, manifest: &PhantomManifest) {
    for (i, sl) in manifest.slices.iter().enumerate() {
        if let Some(g) = &sl.geojson {
            let sum = s.import_geojson(SlideId(i as u32), &ph.join(g)).unwrap();
            assert!(sum.imported > 0);
            assert_eq!(
                sum.skipped_unknown_label, 0,
                "nama label fantom harus dikenal proyek"
            );
        }
    }
}

fn analyze(r: &mut Run) -> SpatialAnalysis {
    r.session.analyze_spatial(&NoProgress).expect("analisis")
}

#[test]
fn ellipsoid_with_margin_plane() {
    let mut r = run(&PhantomOptions {
        margin_um: Some(240.0),
        ..base(PhantomShapeKind::Ellipsoid)
    });
    let a = analyze(&mut r);
    let m = a.margin.expect("margin");
    println!(
        "margin: d_min {:.1} ± {:.1} µm (benar 240), arah {:?}",
        m.d_min.value_um, m.d_min.plus_minus_um, m.direction
    );
    assert!((m.d_min.value_um - 240.0).abs() <= 16.0, "galat ≤ s_xy");
    assert_eq!(m.direction, h3d_core::SignedAxis::PosX);
    assert!(m.d_min.plus_minus_um >= 8.0);
    // Volume: Cavalieri vs analitik.
    let rep = r.session.measure().unwrap();
    let truth = r.manifest.analytic.volume_um3.expect("volume analitik");
    let v = rep.labels.iter().find(|l| l.name == "Tumor").unwrap();
    let e = (v.volume_cavalieri_um3 - truth).abs() / truth;
    println!("ellipsoid: galat Cavalieri {:.2} %", e * 100.0);
    assert!(e <= 0.03);
    // Gambar laporan (8.12.8 butir 3): irisan terdekat, garis luar label,
    // kedua titik terdekat, dan panah; dirujuk dari Markdown.
    let dir_a = r._tmp.path().join("laporan-a");
    let dir_b = r._tmp.path().join("laporan-b");
    let kind = h3d_engine::ExportKind::Report {
        with_clinical: false,
    };
    let files = r.session.export_files(kind.clone(), &dir_a).unwrap();
    r.session.export(kind, &dir_b).unwrap();
    let png_path = dir_a.join("figures").join("margin_tepi_reseksi.png");
    assert!(files.contains(&png_path), "berkas tertulis: {files:?}");
    let png = std::fs::read(&png_path).expect("gambar margin ada");
    assert_eq!(
        png,
        std::fs::read(dir_b.join("figures").join("margin_tepi_reseksi.png")).unwrap(),
        "gambar harus byte-identik antar dua jalan"
    );
    let figs = r.session.margin_figures().unwrap();
    assert_eq!(figs.len(), 1);
    let f = &figs[0];
    assert_eq!(f.meta.file, "figures/margin_tepi_reseksi.png");
    assert_eq!(f.meta.label, "Tepi reseksi");
    let img = image::load_from_memory(&png).unwrap().to_rgba8();
    // Slide fantom 128 px → diperbesar bulat 6× = 768 px.
    assert_eq!((img.width(), img.height()), (768, 768));
    assert_eq!(f.meta.size_px, [768, 768]);
    assert_eq!(img.as_raw(), &f.rgba, "PNG = buffer yang dirender");
    assert!((f.meta.um_per_px[0] - 16.0 / 6.0).abs() < 1e-9);
    let at = |p: [f64; 2]| img.get_pixel(p[0].floor() as u32, p[1].floor() as u32).0;
    assert_eq!(at(f.meta.tumor_px), h3d_engine::figure::TUMOR_MARKER);
    assert_eq!(at(f.meta.margin_px), h3d_engine::figure::MARGIN_MARKER);
    // Kedua titik: tumor di kiri tepi (arah +X), berjarak ≈ d_min di bidang.
    assert!(f.meta.margin_px[0] > f.meta.tumor_px[0]);
    assert!(
        (f.meta.in_plane_um - m.d_min.value_um).abs() <= 2.0 * 16.0,
        "jarak pada bidang {} vs d_min {}",
        f.meta.in_plane_um,
        m.d_min.value_um
    );
    // Irisan yang digambar = irisan berlabel terdekat ke titik tumor.
    let zs: Vec<f64> = r.session.slides().iter().map(|e| e.z_um).collect();
    let nearest = zs
        .iter()
        .map(|z| (z - m.nearest_tumor_um[2]).abs())
        .fold(f64::INFINITY, f64::min);
    assert!((f.meta.slide_z_um - m.nearest_tumor_um[2]).abs() <= nearest + 1e-9);
    // Panah: titik tengah antara kedua penanda berwarna panah.
    let mid = [
        (f.meta.tumor_px[0] + f.meta.margin_px[0]) / 2.0,
        (f.meta.tumor_px[1] + f.meta.margin_px[1]) / 2.0,
    ];
    assert_eq!(at(mid), h3d_engine::figure::ARROW);
    // Garis luar label: warna label Tumor dan Tepi reseksi muncul di gambar.
    for name in ["Tumor", "Tepi reseksi"] {
        let c = r
            .session
            .project()
            .label_by_name(name)
            .expect("label")
            .color;
        let n = img
            .pixels()
            .filter(|p| p.0 == [c[0], c[1], c[2], 255])
            .count();
        assert!(n > 50, "garis luar {name}: {n} piksel");
    }
    let md = std::fs::read_to_string(dir_a.join("report.md")).unwrap();
    assert!(
        md.contains("](figures/margin_tepi_reseksi.png)"),
        "laporan merujuk gambar dengan tautan relatif"
    );
    assert!(md.contains("1 piksel = 2.7 µm"), "keterangan memuat skala");
    let sec3 = md.find("## 3. Margin").unwrap();
    let sec4 = md.find("## 4. Invasi").unwrap();
    let link = md.find("](figures/margin_tepi_reseksi.png)").unwrap();
    assert!(sec3 < link && link < sec4, "gambar ada di bagian 3");
    // Label tepi reseksi tidak dijadikan mesh dan tidak masuk grup.
    assert!(rep
        .labels
        .iter()
        .any(|l| l.name == "Tepi reseksi" && l.volume_mesh_um3.is_none()));
    assert!(rep.groups.iter().all(|g| g.group != "Tepi reseksi"));
    assert_eq!(r.session.meshes().unwrap().len(), 1);
}

#[test]
fn two_foci_with_gap() {
    let mut r = run(&PhantomOptions {
        foci: Some((2, 320.0)),
        ..base(PhantomShapeKind::Ellipsoid)
    });
    let a = analyze(&mut r);
    let f = a.foci.expect("fokus");
    assert_eq!(f.n, 2, "jumlah fokus");
    let d = f.distances_um[0][1].value_um;
    println!("dua fokus: jarak {d:.1} µm (benar 320)");
    assert!((d - 320.0).abs() <= 32.0, "jarak antar fokus {d}");
    assert_eq!(r.manifest.analytic.n_foci, 2);
}

#[test]
fn vessel_contact_and_capsule_breach() {
    let mut r = run(&PhantomOptions {
        vessel: true,
        ..base(PhantomShapeKind::Ellipsoid)
    });
    let a = analyze(&mut r);
    let c = a
        .contacts
        .iter()
        .find(|c| c.name == "Pembuluh darah")
        .expect("kontak pembuluh");
    assert!(c.result.any);
    let truth = r
        .manifest
        .analytic
        .contact_area_mm2
        .expect("luas kontak analitik");
    println!(
        "kontak pembuluh: {:.4} mm² (benar {truth:.4})",
        c.result.area_mm2
    );
    // Sebelum normal kontak diambil dari indikator terpulus: 0,4331 mm²
    // (+10,0 %) untuk pembuluh berjari-jari 5 voxel ini.
    assert!(
        (c.result.area_mm2 - truth).abs() / truth <= 0.07,
        "luas kontak {} vs {truth}",
        c.result.area_mm2
    );
    assert!(
        a.hypoxia.is_some(),
        "proksi hipoksia tersedia bila ada pembuluh darah"
    );

    for breach in [false, true] {
        let mut r = run(&PhantomOptions {
            capsule: true,
            breach,
            slices: 41,
            ..base(PhantomShapeKind::Ellipsoid)
        });
        let a = analyze(&mut r);
        let cap = a.capsule.expect("kapsul");
        assert_eq!(
            cap.breached, breach,
            "penetrasi kapsul (varian tembus = {breach})"
        );
        assert_eq!(r.manifest.analytic.capsule_breached, Some(breach));
    }
}

#[test]
fn surface_depth() {
    let h = 600.0;
    let mut r = run(&PhantomOptions {
        surface_um: Some(h),
        ..base(PhantomShapeKind::Ellipsoid)
    });
    let a = analyze(&mut r);
    let d = a.depth.expect("kedalaman");
    let s_z = r.manifest.spacing_um;
    println!(
        "kedalaman: {:.1} ± {:.1} µm (benar {h}), s_z {s_z:.1}",
        d.depth.value_um, d.depth.plus_minus_um
    );
    assert!(
        (d.depth.value_um - h).abs() <= s_z,
        "kedalaman {}",
        d.depth.value_um
    );
    assert_eq!(r.manifest.analytic.depth_um, Some(h));
}

#[test]
fn torus_and_branch_topology_and_shuffle() {
    let mut r = run(&base(PhantomShapeKind::Torus));
    let meshes = r.session.reconstruct(&NoProgress).unwrap();
    assert_eq!(meshes.len(), 1);
    assert!(meshes[0].is_closed());
    assert_eq!(meshes[0].euler_characteristic(), 0, "torus: genus 1");
    let truth = r.manifest.analytic.volume_um3.expect("volume analitik");
    let v = meshes[0].signed_volume_um3();
    println!(
        "torus: volume mesh {:.4} mm³ (benar {:.4})",
        v / 1e9,
        truth / 1e9
    );
    assert!((v - truth).abs() / truth <= 0.08);

    let mut r = run(&base(PhantomShapeKind::Branch));
    let meshes = r.session.reconstruct(&NoProgress).unwrap();
    assert_eq!(
        meshes[0].connected_components(),
        1,
        "cabang Y: satu komponen"
    );
    assert!(meshes[0].is_closed());

    // --shuffle: nama berkas tidak lagi mengikuti urutan potong, tetapi
    // truth.json mencatat urutan sebenarnya; semua irisan tetap ada.
    let tmp = tempfile::tempdir().unwrap();
    let m = write_phantom(
        tmp.path(),
        &PhantomOptions {
            shuffle: true,
            mirror: true,
            max_rotation_deg: 180.0,
            ..base(PhantomShapeKind::Branch)
        },
    )
    .unwrap();
    let order: Vec<usize> = m.slices.iter().map(|s| s.true_index).collect();
    let mut sorted = order.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..21).collect::<Vec<_>>());
    assert_ne!(order, sorted, "urutan berkas harus teracak");
    assert!(m.slices.iter().any(|s| s.mirrored) && m.slices.iter().any(|s| !s.mirrored));
    // Opsi tidak sah → galat.
    for bad in [
        PhantomOptions {
            breach: true,
            ..base(PhantomShapeKind::Ellipsoid)
        },
        PhantomOptions {
            slices: 1,
            ..base(PhantomShapeKind::Ellipsoid)
        },
        PhantomOptions {
            ihc: Some(1.5),
            ..base(PhantomShapeKind::Ellipsoid)
        },
        PhantomOptions {
            foci: Some((9, 500.0)),
            ..base(PhantomShapeKind::Ellipsoid)
        },
    ] {
        assert!(write_phantom(&tmp.path().join("buruk"), &bad).is_err());
    }
}

/// 8.4.7 ujung-ke-ujung: irisan diacak urutannya, diputar 0–360° dan
/// dicerminkan acak; usulan urutan diterima, lalu registrasi memakai
/// `orient_hint` sebagai tebakan awal.
#[test]
fn shuffled_stack_is_reordered_by_suggestion_then_registered() {
    use h3d_core::{Affine2, OrderSource};
    use h3d_engine::RegOptions;

    let tmp = tempfile::tempdir().unwrap();
    let ph = tmp.path().join("ph");
    let manifest = write_phantom(
        &ph,
        &PhantomOptions {
            shape: PhantomShapeKind::Branch,
            slices: 30,
            seed: 4,
            size_px: 160,
            shuffle: true,
            mirror: true,
            max_rotation_deg: 180.0,
            ..Default::default()
        },
    )
    .unwrap();
    let mut s = Session::create(&tmp.path().join("p.h3d")).unwrap();
    let paths: Vec<PathBuf> = manifest.slices.iter().map(|x| ph.join(&x.file)).collect();
    s.import_slides_with(
        &paths,
        &ImportOptions {
            mpp: Some([manifest.mpp, manifest.mpp]),
            embed: false,
            spacing_um: Some(manifest.spacing_um),
        },
        &NoProgress,
    )
    .unwrap();
    import_labels(&mut s, &ph, &manifest);
    s.save().unwrap();
    let labels_before: Vec<Vec<u8>> = (0..30)
        .map(|i| std::fs::read(s.bundle().label_path(SlideId(i))).unwrap_or_default())
        .collect();

    let true_index: Vec<usize> = manifest.slices.iter().map(|x| x.true_index).collect();
    let sug = s.suggest_order(&NoProgress).unwrap();
    let seq: Vec<usize> = sug
        .order
        .iter()
        .map(|id| true_index[id.0 as usize])
        .collect();
    let adjacent = seq.windows(2).filter(|w| w[0].abs_diff(w[1]) == 1).count();
    let frac = adjacent as f64 / (seq.len() - 1) as f64;
    // Cermin relatif terhadap irisan acuan usulan (tengah urutan).
    let reference = sug.order[sug.order.len() / 2];
    let mirrored = |id: SlideId| manifest.slices[id.0 as usize].mirrored;
    let mirror_ok = sug
        .orient
        .iter()
        .filter(|(id, a)| a.decompose().mirrored == (mirrored(*id) != mirrored(reference)))
        .count();
    println!(
        "usulan urutan: {adjacent}/{} pasangan bertetangga benar ({:.1} %), cermin benar {mirror_ok}/{}, tak yakin {}",
        seq.len() - 1,
        frac * 100.0,
        sug.orient.len(),
        sug.uncertain.len()
    );
    assert!(frac >= 0.95, "pasangan bertetangga benar {frac}");
    assert!(
        mirror_ok as f64 >= 0.95 * sug.orient.len() as f64,
        "cermin benar {mirror_ok}"
    );

    // Terima semua: urutan proyek menjadi urutan usulan; label tidak tersentuh.
    let all: Vec<SlideId> = sug.order.clone();
    s.apply_order_suggestion(&sug, &all).unwrap();
    let got: Vec<SlideId> = s.slides().iter().map(|e| e.id).collect();
    assert_eq!(got, sug.order);
    assert!(s
        .slides()
        .iter()
        .any(|e| e.order_source == OrderSource::Suggested));
    assert!(s.slides().iter().all(|e| e.orient_hint.is_some()));
    assert!(s.project().z_is_monotonic());
    for i in 0..30u32 {
        let now = std::fs::read(s.bundle().label_path(SlideId(i))).unwrap_or_default();
        assert_eq!(now, labels_before[i as usize], "label irisan {i} berubah");
    }
    assert!(s.measure().unwrap().slides.order_suggested);

    // Registrasi dengan orient_hint sebagai tebakan awal; bandingkan dengan
    // transformasi sebenarnya relatif terhadap acuan.
    let sum = s.register_all(&RegOptions::default(), &NoProgress).unwrap();
    assert_eq!(sum.registered.len(), 29);
    let truth: Vec<Affine2> = manifest
        .slices
        .iter()
        .map(|x| Affine2::new(x.transform_u_to_w))
        .collect();
    let r = sum.reference.unwrap();
    let t_ref_inv = truth[r.0 as usize].inverse().unwrap();
    let field = manifest.mpp * f64::from(manifest.size_px[0]);
    let mut errs: Vec<f64> = Vec::new();
    for e in s.slides() {
        let rel = t_ref_inv.compose(&truth[e.id.0 as usize]);
        let est = e.transform.affine();
        let mut ss = 0.0;
        for (fx, fy) in [
            (0.35, 0.35),
            (0.65, 0.35),
            (0.5, 0.5),
            (0.35, 0.65),
            (0.65, 0.65),
        ] {
            let u = [fx * field, fy * field];
            let (a, b) = (est.apply(u), rel.apply(u));
            ss += (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2);
        }
        errs.push((ss / 5.0).sqrt());
    }
    errs.sort_by(|a, b| a.total_cmp(b));
    let chain_median = errs[errs.len() / 2];
    // Galat berpasangan: P_{i→tetangga} terhadap kebenaran.
    let mut pair: Vec<f64> = Vec::new();
    for e in s.slides() {
        let Some(rec) = &e.reg else { continue };
        let p_true = truth[rec.fixed.0 as usize]
            .inverse()
            .unwrap()
            .compose(&truth[e.id.0 as usize]);
        let mut ss = 0.0;
        for (fx, fy) in [
            (0.35, 0.35),
            (0.65, 0.35),
            (0.5, 0.5),
            (0.35, 0.65),
            (0.65, 0.65),
        ] {
            let u = [fx * field, fy * field];
            let (a, b) = (rec.pairwise.apply(u), p_true.apply(u));
            ss += (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2);
        }
        pair.push((ss / 5.0f64).sqrt());
    }
    pair.sort_by(|a, b| a.total_cmp(b));
    let pair_median = pair[pair.len() / 2];
    println!(
        "registrasi setelah usulan: TRE berpasangan median {pair_median:.1} µm (maks {:.1}); \
         TRE rantai median {chain_median:.1} µm (maks {:.1}); bertanda {}",
        pair[pair.len() - 1],
        errs[errs.len() - 1],
        sum.flagged.len()
    );
    // Rotasi 0–360° dan cermin ditemukan; galat berpasangan jauh di bawah
    // satu piksel registrasi dan galat rantai (15 pasang berurutan dari
    // acuan) tetap di bawah dua piksel.
    assert!(
        pair_median <= 0.5 * manifest.mpp,
        "TRE berpasangan median {pair_median} µm"
    );
    assert!(sum.flagged.is_empty());
    assert!(
        chain_median <= 2.0 * manifest.mpp,
        "TRE rantai median {chain_median} µm"
    );
}
