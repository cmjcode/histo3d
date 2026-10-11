//! Uji ujung-ke-ujung mesin atas fantom ellipsoid (H1, H3, H5, H6):
//! impor → registrasi → label dari masker fantom → rekonstruksi → ukur.

mod common;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use common::{build_project, fixed_clock, open_copy, shared, tre_rms_um, true_transforms};
use h3d_core::{
    Affine2, AnnotationSource, ClinicalMeta, LabelId, NoProgress, OrderSource, Progress,
    ReviewStatus, SlideId, SlideSourceRef, SlideStatus, SlideTransform,
};
use h3d_engine::{
    spawn_job, spawn_session_job, EngineError, ExportKind, ImportOptions, OrderSuggestion,
    PaintMode, Proposal, RegOptions, RegisterRun, Session,
};
use ndarray::Array2;

const TUMOR: LabelId = LabelId(1);

fn read_tree(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut out = BTreeMap::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            if e.path().is_file() {
                out.insert(
                    e.file_name().to_string_lossy().into_owned(),
                    std::fs::read(e.path()).unwrap(),
                );
            }
        }
    }
    out
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

/// Progres yang meminta batal setelah `limit` panggilan `set`.
struct CancelAfter {
    calls: AtomicUsize,
    limit: usize,
}

impl Progress for CancelAfter {
    fn set(&self, _f: f32, _s: &str) {
        self.calls.fetch_add(1, Ordering::SeqCst);
    }
    fn is_cancelled(&self) -> bool {
        self.calls.load(Ordering::SeqCst) >= self.limit
    }
}

struct AlwaysCancelled;

impl Progress for AlwaysCancelled {
    fn set(&self, _f: f32, _s: &str) {}
    fn is_cancelled(&self) -> bool {
        true
    }
}

#[test]
fn import_without_mpp_is_error_and_with_mpp_reopens_identically() {
    let sh = shared();
    let dir = sh.root.path().join("impor").join("p.h3d");
    std::fs::create_dir_all(dir.parent().unwrap()).unwrap();
    let mut s = Session::create(&dir).unwrap();
    // Urutan masukan dibalik: urutan proyek tetap urutan alami nama berkas.
    let mut paths = sh.slide_paths();
    paths.reverse();

    let err = s.import_slides(&paths, None, &NoProgress).unwrap_err();
    assert!(matches!(err, EngineError::MppMissing { .. }), "{err:?}");
    assert!(err.to_string().contains("mpp"));
    assert!(
        s.project().slides.is_empty(),
        "impor gagal tidak boleh mengubah proyek"
    );

    let ids = s
        .import_slides(&paths, Some([16.0, 16.0]), &NoProgress)
        .unwrap();
    assert_eq!(ids.len(), 24);
    for (i, e) in s.slides().iter().enumerate() {
        assert_eq!(e.order, i as u32);
        assert_eq!(e.source.file_name(), format!("slice_{i:03}.png"));
        assert_eq!(e.mpp, [16.0, 16.0]);
        assert_eq!(e.size_l0, [192, 192]);
        assert_eq!(e.z_um, i as f64 * 50.0, "z bawaan = urutan × jarak");
        assert!(e.z_is_default);
        assert_eq!(e.order_source, OrderSource::FileName);
        match &e.source {
            SlideSourceRef::External {
                path,
                size_bytes,
                fingerprint,
            } => {
                assert!(path.is_relative(), "path External harus relatif: {path:?}");
                assert!(*size_bytes > 0 && *fingerprint != 0);
            }
            other => panic!("diharapkan External: {other:?}"),
        }
    }
    let saved = s.project().clone();
    drop(s);
    let s2 = Session::open(&dir).unwrap();
    assert_eq!(s2.project(), &saved, "proyek harus terbuka kembali identik");
    assert!(s2.missing_slides().is_empty());
    let thumb = s2.thumbnail(SlideId(3), 64).unwrap();
    assert!(thumb.width().max(thumb.height()) <= 64);
    assert_eq!(
        s2.thumbnail(SlideId(3), 64).unwrap(),
        thumb,
        "thumbnail dari cache sama"
    );

    // --embed: berkas disalin ke dalam bundel.
    let dir3 = sh.root.path().join("impor-embed").join("p.h3d");
    std::fs::create_dir_all(dir3.parent().unwrap()).unwrap();
    let mut s3 = Session::create(&dir3).unwrap();
    s3.import_slides_with(
        &paths[..2],
        &ImportOptions {
            mpp: Some([16.0, 16.0]),
            embed: true,
            spacing_um: Some(75.0),
        },
        &NoProgress,
    )
    .unwrap();
    for e in s3.slides() {
        assert!(matches!(e.source, SlideSourceRef::Embedded { .. }));
        assert!(s3.slide_path(e.id).unwrap().starts_with(&dir3));
        assert!(s3.slide_path(e.id).unwrap().is_file());
    }
    assert_eq!(s3.slides()[1].z_um, 75.0);

    // Berkas bukan gambar → galat, proyek tidak berubah.
    let junk = sh.root.path().join("impor").join("sampah.png");
    std::fs::write(&junk, b"ini bukan PNG").unwrap();
    assert!(s3
        .import_slides(&[junk], Some([16.0, 16.0]), &NoProgress)
        .is_err());
    assert_eq!(s3.project().slides.len(), 2);
    // project.json rusak → galat saat dibuka, bukan panik.
    std::fs::write(dir3.join("project.json"), b"{ rusak").unwrap();
    assert!(Session::open(&dir3).is_err());
    assert!(Session::open(&sh.root.path().join("tidak-ada.h3d")).is_err());
}

#[test]
fn full_pipeline_volume_and_two_runs_byte_identical() {
    let sh = shared();
    // Jalan A = proyek dasar (disalin agar tidak diubah); jalan B = jalan penuh kedua.
    let mut a = open_copy("det-a");
    let mut b = build_project(sh.root.path(), &sh.manifest, "det-b");
    b.set_clock(fixed_clock());

    let (tre_mean, tre_worst) = tre_rms_um(&a, &sh.manifest);
    println!(
        "registrasi: TRE rata-rata {tre_mean:.1} µm, terburuk {tre_worst:.1} µm \
         (piksel registrasi 16 µm); proyek dasar dibangun {:.1} s",
        sh.build_seconds
    );
    // Opsi BAWAAN (affine "rigid dulu"): rantai tidak boleh hanyut. Sebelum
    // pengaman model bebas, affine memberi TRE rata-rata 26,9 µm / terburuk
    // 58,6 µm dan |det T_i| sampai 1,11 di ujung rantai.
    assert!(
        tre_mean <= 16.0,
        "TRE rata-rata {tre_mean} µm > 1 piksel registrasi"
    );
    assert!(
        tre_worst <= 2.0 * 16.0,
        "TRE terburuk {tre_worst} µm > 2 px"
    );
    assert!(a.qc_table().iter().all(|r| r.registered && !r.flagged));
    for e in a.project().usable_slides_ordered() {
        let det = e.transform.affine().determinant().abs();
        assert!(
            (det - 1.0).abs() <= 0.02,
            "irisan {}: skala kumulatif rantai |det| = {det}",
            e.id
        );
    }
    // Taksiran TRE berasal dari sisa galat terukur (bukan proksi Dice) dan
    // berada dalam faktor 2 dari TRE sebenarnya.
    assert!(a
        .project()
        .slides
        .iter()
        .filter_map(|e| e.reg.as_ref())
        .all(|r| r.residual_um.is_some() && r.residual_fit_um.is_some()));
    let tre_est = a.tre_estimate_um().expect("taksiran TRE");
    println!("taksiran TRE {tre_est:.1} µm vs sebenarnya {tre_mean:.1} µm");
    assert!(
        tre_est >= tre_mean / 2.0 && tre_est <= tre_mean * 2.0,
        "taksiran TRE {tre_est} µm di luar faktor 2 dari {tre_mean} µm"
    );

    let pa = a.path().to_path_buf();
    let pb = b.path().to_path_buf();
    assert_eq!(
        std::fs::read(pa.join("project.json")).unwrap(),
        std::fs::read(pb.join("project.json")).unwrap(),
        "project.json harus byte-identik antar dua jalan"
    );
    let la = read_tree(&pa.join("labels"));
    let lb = read_tree(&pb.join("labels"));
    assert!(la.len() >= 18, "jumlah berkas label: {}", la.len());
    assert_eq!(la, lb, ".lbl harus byte-identik antar dua jalan");

    let mut outputs: Vec<BTreeMap<String, Vec<u8>>> = Vec::new();
    let mut reports = Vec::new();
    for s in [&mut a, &mut b] {
        let meshes = s.reconstruct(&NoProgress).unwrap();
        assert_eq!(meshes.len(), 1);
        assert!(meshes[0].is_closed() && meshes[0].signed_volume_um3() > 0.0);
        let analysis = s.analyze_spatial(&NoProgress).unwrap();
        assert!(analysis.size.is_some());
        let out = s.path().join("ekspor");
        s.export(ExportKind::Glb { scalar: None }, &out).unwrap();
        s.export(ExportKind::Analysis, &out).unwrap();
        s.export(
            ExportKind::Report {
                with_clinical: false,
            },
            &out,
        )
        .unwrap();
        let mut files = read_tree(&out);
        files.insert(
            "out/analysis.json".into(),
            std::fs::read(s.bundle().analysis_json()).unwrap(),
        );
        outputs.push(files);
        reports.push(s.measure().unwrap());
    }
    for name in [
        "mesh.glb",
        "analysis.json",
        "out/analysis.json",
        "report.md",
        "report.csv",
    ] {
        assert_eq!(
            outputs[0][name], outputs[1][name],
            "{name} harus byte-identik"
        );
    }
    assert_eq!(reports[0], reports[1]);
    assert_eq!(
        outputs[0]["analysis.json"], outputs[0]["out/analysis.json"],
        "ekspor analisis = out/analysis.json"
    );

    // Volume: Cavalieri ≤ 3 % dari analitik; mesh sebagai pembanding.
    let truth = sh.manifest.analytic.volume_um3.expect("volume analitik");
    let row = &reports[0].labels[0];
    assert_eq!(row.name, "Tumor");
    let e_cav = (row.volume_cavalieri_um3 - truth).abs() / truth;
    let v_mesh = row.volume_mesh_um3.expect("volume mesh");
    let e_mesh = (v_mesh - truth).abs() / truth;
    println!(
        "volume: analitik {:.4} mm³, Cavalieri {:.4} mm³ (galat {:.2} %), mesh {:.4} mm³ \
         (galat {:.2} %), irisan beranotasi {}, diameter maks {:.0} µm ± {:.0} µm",
        truth / 1e9,
        row.volume_cavalieri_um3 / 1e9,
        e_cav * 100.0,
        v_mesh / 1e9,
        e_mesh * 100.0,
        row.n_slices_annotated,
        row.max_diameter_um.unwrap(),
        reports[0].dimension_uncertainty_um
    );
    assert!(e_cav <= 0.03, "galat Cavalieri {e_cav}");
    assert!(e_mesh <= 0.03, "galat volume mesh {e_mesh}");
    let masks = a.voxel_masks().unwrap();
    let v_vox = masks
        .iter()
        .find(|(l, _)| *l == TUMOR)
        .expect("masker voxel tumor")
        .1
        .volume_um3();
    let e_vox = (v_vox - truth).abs() / truth;
    println!(
        "volume voxel {:.4} mm³ (galat {:.2} %)",
        v_vox / 1e9,
        e_vox * 100.0
    );
    assert!(e_vox <= 0.03, "galat volume voxel {e_vox}");

    // `reconstruct` menulis out/volume/labels.nii.gz (8.7.1). Padanan
    // kriteria H5 "volume di 3D Slicer sama dalam 1 %": volume voxel NIfTI
    // (jumlah voxel berlabel × pixdim) vs Cavalieri.
    let nii_path = pa.join("out").join("volume").join("labels.nii.gz");
    assert_eq!(a.label_volume_path(), nii_path);
    let gz = std::fs::read(&nii_path).expect("reconstruct menulis labels.nii.gz");
    assert_eq!(
        gz,
        std::fs::read(pb.join("out").join("volume").join("labels.nii.gz")).unwrap(),
        "labels.nii.gz harus byte-identik antar dua jalan"
    );
    let raw = h3d_io::gunzip(&gz).unwrap();
    let (hdr, voxels) = h3d_io::decode_nifti(&raw).unwrap();
    let grid = &masks[0].1.grid;
    assert_eq!(hdr.dim[0], 3);
    assert_eq!(
        [hdr.dim[1], hdr.dim[2], hdr.dim[3]],
        [grid.size[0] as i16, grid.size[1] as i16, grid.nz() as i16]
    );
    assert_eq!((hdr.datatype, hdr.bitpix), (2, 8), "u8");
    // pixdim dalam mm: (s_xy, s_xy, s_z).
    assert!((f64::from(hdr.pixdim[1]) - grid.s_xy / 1000.0).abs() < 1e-7);
    assert!((f64::from(hdr.pixdim[2]) - grid.s_xy / 1000.0).abs() < 1e-7);
    assert!((f64::from(hdr.pixdim[3]) - grid.median_dz() / 1000.0).abs() < 1e-7);
    let n_tumor = voxels.iter().filter(|v| **v == TUMOR.0).count();
    assert!(voxels.iter().all(|v| *v == 0 || *v == TUMOR.0));
    let v_nii_mm3 = n_tumor as f64
        * f64::from(hdr.pixdim[1])
        * f64::from(hdr.pixdim[2])
        * f64::from(hdr.pixdim[3]);
    let v_cav_mm3 = row.volume_cavalieri_um3 / 1e9;
    let e_nii = (v_nii_mm3 - v_cav_mm3).abs() / v_cav_mm3;
    println!(
        "volume NIfTI {v_nii_mm3:.4} mm³ vs Cavalieri {v_cav_mm3:.4} mm³ (selisih {:.2} %)",
        e_nii * 100.0
    );
    assert!(e_nii <= 0.01, "volume NIfTI vs Cavalieri {e_nii}");
    // Berkas yang terhapus ditulis ulang oleh `reconstruct` (dari cache) dan
    // oleh `write_label_volume`.
    std::fs::remove_file(&nii_path).unwrap();
    a.reconstruct(&NoProgress).unwrap();
    assert_eq!(std::fs::read(&nii_path).unwrap(), gz);
    std::fs::remove_file(&nii_path).unwrap();
    assert_eq!(a.write_label_volume().unwrap(), nii_path);
    assert_eq!(std::fs::read(&nii_path).unwrap(), gz);
    assert!(reports[0].dimension_uncertainty_um > 0.0);
    // Peringatan laporan tidak berulang (dulu `has_draft` tercetak dua kali
    // di `measure --json`).
    for (i, w) in reports[0].warnings.iter().enumerate() {
        assert!(
            !reports[0].warnings[..i].contains(w),
            "peringatan berulang: {w:?}"
        );
    }
    assert!(
        reports[0].analysis.is_some(),
        "laporan memuat analisis yang masih sah"
    );
    assert_eq!(row.n_slices_interpolated, 0);

    // Ekspor semua format + privasi: tanpa flag tidak ada isi klinis.
    let clinical = ClinicalMeta {
        subject_pseudo_id: "ZQPSEUDO-7731-KLINIS".into(),
        age_years: Some(57),
        topography_icdo: Some("ZQTOPO-C50.4".into()),
        morphology_icdo: Some("ZQMORF-8500/3".into()),
        specimen_month: Some("2025-11".into()),
        ..Default::default()
    };
    a.set_clinical(Some(clinical.clone())).unwrap();
    a.save().unwrap();
    let project_json = std::fs::read(pa.join("project.json")).unwrap();
    let out = pa.join("ekspor-semua");
    let unknown = ExportKind::Glb {
        scalar: Some("attr:tidak-ada".into()),
    };
    assert!(
        a.export(unknown, &out).is_err(),
        "skalar tak dikenal harus galat"
    );
    for kind in [
        ExportKind::Glb { scalar: None },
        ExportKind::Stl,
        ExportKind::Obj,
        ExportKind::Nifti,
        ExportKind::GeoJson,
        ExportKind::Report {
            with_clinical: false,
        },
        ExportKind::Analysis,
    ] {
        let files = a.export_files(kind.clone(), &out).unwrap();
        assert!(!files.is_empty(), "{kind:?} tidak menulis berkas");
    }
    let files = read_tree(&out);
    for name in [
        "mesh.glb",
        "tumor_mm.stl",
        "mesh_mm.obj",
        "labels.nii.gz",
        "report.md",
        "report.csv",
        "analysis.json",
        "0012.geojson",
    ] {
        assert!(files.contains_key(name), "berkas ekspor hilang: {name}");
    }
    assert!(!files.keys().any(|k| k.ends_with(".tmp")));
    let mut all = files.clone();
    all.insert("project.json".into(), project_json);
    for (name, bytes) in &all {
        let plain = if name.ends_with(".gz") {
            h3d_io::gunzip(bytes).unwrap()
        } else {
            bytes.clone()
        };
        for marker in ["ZQPSEUDO-7731-KLINIS", "ZQTOPO-C50.4", "ZQMORF-8500/3"] {
            assert!(
                !contains(&plain, marker.as_bytes()),
                "{marker} bocor ke {name}"
            );
        }
    }
    let md = String::from_utf8(files["report.md"].clone()).unwrap();
    assert!(md.contains(h3d_core::RESEARCH_ONLY_NOTICE));
    for f in ClinicalMeta::FIELD_NAMES {
        assert!(!md.contains(f), "nama field klinis {f} muncul di laporan");
    }
    // Dengan flag: konteks klinis tercetak dan clinical.json ikut disalin.
    let out_c = pa.join("ekspor-klinis");
    a.export(
        ExportKind::Report {
            with_clinical: true,
        },
        &out_c,
    )
    .unwrap();
    let md = std::fs::read_to_string(out_c.join("report.md")).unwrap();
    assert!(md.contains("ZQPSEUDO-7731-KLINIS") && md.contains("ZQTOPO-C50.4"));
    assert!(out_c.join("clinical.json").is_file());
    // GLB berwarna skalar setelah analisis (ukuran staging tidak punya medan,
    // jadi cukup pastikan nama skalar yang salah ditolak dan yang tanpa
    // skalar valid).
    let (doc, bin) = h3d_io::parse_glb(&files["mesh.glb"]).unwrap();
    assert_eq!(
        doc["buffers"][0]["byteLength"].as_u64().unwrap() as usize,
        bin.len()
    );
    // Menghapus data klinis: berkas hilang, proyek tetap sah.
    a.set_clinical(None).unwrap();
    assert!(!pa.join("clinical.json").exists());
    assert!(Session::open(&pa)
        .unwrap()
        .project()
        .sample
        .clinical
        .is_none());
}

#[test]
fn labels_stay_in_place_after_reregistration_and_run_is_resumable() {
    let mut s = open_copy("reg-ulang");
    let before = read_tree(&s.path().join("labels"));
    let mem_before = h3d_segment::encode_lbl(s.labels(SlideId(12)).unwrap());
    let transforms_before: Vec<SlideTransform> =
        s.slides().iter().map(|e| e.transform.clone()).collect();

    // Tanpa `force`: semua catatan masih sah → tidak ada yang didaftarkan ulang.
    let sum = s.register_all(&RegOptions::default(), &NoProgress).unwrap();
    assert!(sum.registered.is_empty());
    assert_eq!(sum.skipped.len(), 23);
    assert_eq!(sum.reference, Some(SlideId(12)));

    // Dengan `force`: registrasi diulang, label tidak tersentuh.
    let sum = s
        .register_all_with(
            &RegisterRun {
                force: true,
                batch: 5,
                ..Default::default()
            },
            &NoProgress,
        )
        .unwrap();
    assert_eq!(sum.registered.len(), 23);
    s.save().unwrap();
    assert_eq!(
        read_tree(&s.path().join("labels")),
        before,
        "byte .lbl tidak boleh berubah"
    );
    assert_eq!(
        h3d_segment::encode_lbl(s.labels(SlideId(12)).unwrap()),
        mem_before
    );
    // Registrasi deterministik dan tidak bergantung pada ukuran gelombang.
    let after: Vec<SlideTransform> = s.slides().iter().map(|e| e.transform.clone()).collect();
    assert_eq!(after, transforms_before);
}

#[test]
fn interrupted_registration_resumes_from_saved_pairs() {
    let sh = shared();
    let dir = sh.root.path().join("lanjut").join("p.h3d");
    std::fs::create_dir_all(dir.parent().unwrap()).unwrap();
    let mut s = Session::create(&dir).unwrap();
    s.import_slides_with(
        &sh.slide_paths(),
        &ImportOptions {
            mpp: Some([16.0, 16.0]),
            embed: false,
            spacing_um: Some(sh.manifest.spacing_um),
        },
        &NoProgress,
    )
    .unwrap();
    // Batal setelah ±6 pasangan selesai (1 panggilan awal + 6).
    let cancel = CancelAfter {
        calls: AtomicUsize::new(0),
        limit: 7,
    };
    let run = RegisterRun {
        batch: 4,
        ..Default::default()
    };
    let err = s.register_all_with(&run, &cancel).unwrap_err();
    assert!(matches!(err, EngineError::Cancelled), "{err:?}");
    drop(s);

    // Proyek di disk konsisten dan memuat pasangan yang sudah selesai.
    let mut s = Session::open(&dir).expect("proyek harus tetap terbuka setelah batal");
    let done = s
        .project()
        .slides
        .iter()
        .filter(|e| e.reg.is_some())
        .count();
    assert!((4..23).contains(&done), "pasangan tersimpan: {done}");
    let sum = s.register_all_with(&run, &NoProgress).unwrap();
    assert_eq!(sum.skipped.len(), done, "pasangan tersimpan dilewati");
    assert_eq!(sum.registered.len(), 23 - done);
    assert!(s.needs_registration().is_empty());

    // Hasil akhir sama dengan jalan tak terputus.
    let base = Session::open(&sh.base_project()).unwrap();
    for (a, b) in s.slides().iter().zip(base.slides().iter()) {
        assert_eq!(a.transform, b.transform, "T irisan {}", a.id);
        assert_eq!(a.reg, b.reg);
    }
}

#[test]
fn landmark_correction_recomposes_downstream_only_without_reregistering() {
    let mut s = open_copy("landmark");
    let before: Vec<(SlideId, Affine2, Option<h3d_core::RegRecord>)> = s
        .slides()
        .iter()
        .map(|e| (e.id, e.transform.affine(), e.reg.clone()))
        .collect();
    let reference = s.project().effective_reference().unwrap();
    assert_eq!(reference, SlideId(12));

    // Koreksi irisan 15 (didaftarkan ke 14): P baru = geser ∘ P lama.
    let old_p = before[15].2.as_ref().unwrap().pairwise;
    assert_eq!(before[15].2.as_ref().unwrap().fixed, SlideId(14));
    let new_p = Affine2::translation(40.0, -25.0).compose(&old_p);
    let pts = [
        [600.0, 700.0],
        [2200.0, 800.0],
        [1500.0, 2300.0],
        [900.0, 1900.0],
    ];
    let pairs: Vec<([f64; 2], [f64; 2])> = pts.iter().map(|u| (*u, new_p.apply(*u))).collect();
    s.apply_landmarks(SlideId(15), &pairs).unwrap();

    let rec = s.slide(SlideId(15)).unwrap().reg.clone().unwrap();
    assert!(rec.manual && !rec.flagged);
    assert_eq!(rec.method, "manual-landmarks");
    assert!(rec.pairwise.approx_eq(&new_p, 1e-6));
    assert!(
        rec.ncc > 0.5 && rec.mask_dice > 0.9,
        "metrik QC dihitung ulang: {rec:?}"
    );
    for (id, t_old, reg_old) in &before {
        let e = s.slide(*id).unwrap();
        let t_new = e.transform.affine();
        if id.0 < 15 {
            assert_eq!(
                t_new, *t_old,
                "T irisan {id} (bukan hilir) tidak boleh berubah"
            );
        } else {
            assert!(
                !t_new.approx_eq(t_old, 1.0),
                "T irisan {id} (hilir) harus berubah"
            );
        }
        if id.0 != 15 {
            assert_eq!(
                &e.reg, reg_old,
                "catatan berpasangan irisan {id} tidak didaftarkan ulang"
            );
        }
    }
    // Komposisi: T_15 = T_14 ∘ P_baru; T_16 = T_15 ∘ P_16.
    let t14 = s.slide(SlideId(14)).unwrap().transform.affine();
    let t15 = s.slide(SlideId(15)).unwrap().transform.affine();
    let t16 = s.slide(SlideId(16)).unwrap().transform.affine();
    assert!(t15.approx_eq(&t14.compose(&new_p), 1e-6));
    let p16 = before[16].2.as_ref().unwrap().pairwise;
    assert!(t16.approx_eq(&t15.compose(&p16), 1e-6));

    // Geser/putar manual: T baru irisan 5 (U → W) diberikan langsung.
    let t5_new = Affine2::translation(-30.0, 12.0).compose(&before[5].1);
    s.apply_manual_transform(SlideId(5), t5_new).unwrap();
    assert!(s
        .slide(SlideId(5))
        .unwrap()
        .transform
        .affine()
        .approx_eq(&t5_new, 1e-6));
    assert_eq!(
        s.slide(SlideId(5)).unwrap().reg.as_ref().unwrap().method,
        "manual-transform"
    );
    for id in 6..15u32 {
        assert_eq!(
            s.slide(SlideId(id)).unwrap().transform.affine(),
            before[id as usize].1,
            "irisan {id} di antara koreksi dan acuan tidak berubah"
        );
    }
    assert!(!s
        .slide(SlideId(0))
        .unwrap()
        .transform
        .affine()
        .approx_eq(&before[0].1, 1.0));

    // Dua pasang → similarity; satu pasang / irisan acuan → galat.
    s.apply_landmarks(SlideId(20), &pairs[..2]).unwrap();
    assert!(s.apply_landmarks(SlideId(20), &pairs[..1]).is_err());
    assert!(s.apply_landmarks(reference, &pairs).is_err());
    assert!(s
        .apply_manual_transform(reference, Affine2::IDENTITY)
        .is_err());
    // Proyek tersimpan dan terbuka kembali sama.
    let saved = s.project().clone();
    let path = s.path().to_path_buf();
    drop(s);
    assert_eq!(Session::open(&path).unwrap().project(), &saved);
}

#[test]
fn accepting_order_suggestion_keeps_manual_z_and_labels_and_drops_only_affected_regs() {
    let sh = shared();
    let mut s = open_copy("urutan");
    s.set_z(SlideId(3), 999.0).unwrap();
    s.save().unwrap();
    let labels_before = read_tree(&s.path().join("labels"));
    let spacing = s.project().recon.default_spacing_um;
    assert!((spacing - sh.manifest.spacing_um).abs() < 1e-9);

    // Usulan: tukar irisan 6 dan 7; irisan 20 dan 21 juga diusulkan bertukar
    // tetapi TIDAK diterima.
    let mut order: Vec<SlideId> = (0..24).map(SlideId).collect();
    order.swap(6, 7);
    order.swap(20, 21);
    let hint = Affine2::rotation_about(0.3, [1500.0, 1500.0]);
    let suggestion = OrderSuggestion {
        order,
        orient: vec![(SlideId(6), hint), (SlideId(20), hint)],
        confidence: (0..24).map(|i| (SlideId(i), 0.9)).collect(),
        uncertain: Vec::new(),
    };
    // Tidak menerima apa pun → proyek tidak berubah.
    let untouched = s.project().clone();
    s.apply_order_suggestion(&suggestion, &[]).unwrap();
    assert_eq!(s.project(), &untouched);

    s.apply_order_suggestion(&suggestion, &[SlideId(6), SlideId(7)])
        .unwrap();
    let p = s.project();
    assert_eq!(p.slide(SlideId(6)).unwrap().order, 7);
    assert_eq!(p.slide(SlideId(7)).unwrap().order, 6);
    assert_eq!(
        p.slide(SlideId(6)).unwrap().order_source,
        OrderSource::Suggested
    );
    assert_eq!(
        p.slide(SlideId(7)).unwrap().order_source,
        OrderSource::Suggested
    );
    assert_eq!(p.slide(SlideId(6)).unwrap().orient_hint, Some(hint));
    // Yang tidak diterima tetap.
    assert_eq!(p.slide(SlideId(20)).unwrap().order, 20);
    assert_eq!(
        p.slide(SlideId(20)).unwrap().order_source,
        OrderSource::FileName
    );
    assert_eq!(p.slide(SlideId(20)).unwrap().orient_hint, None);
    // z isian tangan tidak disentuh; z bawaan mengikuti urutan baru.
    assert_eq!(p.slide(SlideId(3)).unwrap().z_um, 999.0);
    assert!(!p.slide(SlideId(3)).unwrap().z_is_default);
    assert!((p.slide(SlideId(6)).unwrap().z_um - 7.0 * spacing).abs() < 1e-9);
    assert!((p.slide(SlideId(7)).unwrap().z_um - 6.0 * spacing).abs() < 1e-9);
    assert!(
        !p.z_is_monotonic(),
        "z isian tangan membuat urutan tidak monoton"
    );
    // Hanya RegRecord irisan yang tetangganya berubah (5, 6, 7) yang dibuang.
    for e in &p.slides {
        let affected = [5, 6, 7].contains(&e.id.0);
        let is_ref = e.id == SlideId(12);
        assert_eq!(
            e.reg.is_none(),
            affected || is_ref,
            "RegRecord irisan {} (terdampak: {affected})",
            e.id
        );
    }
    assert_eq!(
        s.needs_registration(),
        vec![SlideId(5), SlideId(7), SlideId(6)]
    );
    // Rantai di hulu bagian yang putus kembali belum terdaftar; hilir utuh.
    assert!(!s.slide(SlideId(2)).unwrap().transform.is_registered());
    assert!(s.slide(SlideId(9)).unwrap().transform.is_registered());
    assert!(s.slide(SlideId(18)).unwrap().transform.is_registered());
    // Label tidak tersentuh.
    s.save().unwrap();
    assert_eq!(read_tree(&s.path().join("labels")), labels_before);

    // Registrasi lanjutan hanya mengerjakan yang perlu, memakai orient_hint.
    let sum = s.register_all(&RegOptions::default(), &NoProgress).unwrap();
    assert_eq!(sum.registered, vec![SlideId(5), SlideId(7), SlideId(6)]);
    assert!(s.needs_registration().is_empty());

    // Usulan sungguhan: deterministik dan berupa permutasi semua irisan.
    let a = s.suggest_order(&NoProgress).unwrap();
    let b = s.suggest_order(&NoProgress).unwrap();
    assert_eq!(a, b, "usulan harus identik untuk masukan sama");
    let mut ids = a.order.clone();
    ids.sort();
    assert_eq!(ids, (0..24).map(SlideId).collect::<Vec<_>>());
    // Usulan bukan permutasi proyek → ditolak.
    let bad = OrderSuggestion {
        order: vec![SlideId(0), SlideId(1)],
        orient: Vec::new(),
        confidence: Vec::new(),
        uncertain: Vec::new(),
    };
    assert!(s.apply_order_suggestion(&bad, &[SlideId(0)]).is_err());
}

fn dice(a: &Array2<u8>, b: &Array2<u8>) -> f64 {
    let mut inter = 0u64;
    let mut na = 0u64;
    let mut nb = 0u64;
    for (x, y) in a.iter().zip(b.iter()) {
        na += u64::from(*x != 0);
        nb += u64::from(*y != 0);
        inter += u64::from(*x != 0 && *y != 0);
    }
    2.0 * inter as f64 / (na + nb).max(1) as f64
}

#[test]
fn interpolation_is_a_proposal_with_provenance_and_is_excluded_from_cavalieri() {
    let mut s = open_copy("interpolasi");
    let truth_9 = s.labels(SlideId(9)).unwrap().to_array(TUMOR);
    let truth_10 = s.labels(SlideId(10)).unwrap().to_array(TUMOR);
    assert!(truth_9.iter().any(|v| *v != 0));
    // Buang label irisan 9 dan 10.
    for id in [SlideId(9), SlideId(10)] {
        let changed = s
            .edit_labels(id, None, "Kosongkan", |st| {
                for (tx, ty) in st.tile_keys() {
                    st.remove_tile(tx, ty);
                }
            })
            .unwrap();
        assert!(changed);
        assert!(
            s.project().annotation(id, TUMOR).is_none(),
            "catatan ikut dibuang"
        );
    }
    let v0 = s.measure().unwrap().labels[0].volume_cavalieri_um3;
    let n0 = s.measure().unwrap().labels[0].n_slices_annotated;

    s.interpolate_labels(TUMOR, &NoProgress).unwrap();
    // Aturan keras 8: usulan belum menyentuh label.
    assert!(s.labels(SlideId(9)).unwrap().is_empty());
    assert_eq!(s.pending_slides(), vec![SlideId(9), SlideId(10)]);
    assert_eq!(s.pending_proposals(SlideId(9)).len(), 1);
    assert!(matches!(
        s.pending_proposals(SlideId(9))[0].proposal.source,
        AnnotationSource::Interpolated { .. }
    ));
    // Menolak satu, lalu membuat ulang dan menerima semuanya.
    s.reject_pending(SlideId(10));
    assert_eq!(s.pending_slides(), vec![SlideId(9)]);
    s.reject_pending(SlideId(9));
    s.interpolate_labels(TUMOR, &NoProgress).unwrap();
    assert_eq!(s.accept_all_pending(PaintMode::OnlyBackground).unwrap(), 2);
    assert!(s.pending_slides().is_empty());

    for (id, truth) in [(SlideId(9), &truth_9), (SlideId(10), &truth_10)] {
        let rec = s
            .project()
            .annotation(id, TUMOR)
            .expect("catatan interpolasi")
            .clone();
        assert_eq!(
            rec.source,
            AnnotationSource::Interpolated {
                from: [SlideId(8), SlideId(11)]
            }
        );
        assert_eq!(rec.at, common::FIXED_TIME);
        assert_eq!(rec.review, ReviewStatus::Draft);
        let got = s.labels(id).unwrap().to_array(TUMOR);
        let d = dice(&got, truth);
        println!("interpolasi irisan {id}: Dice terhadap masker asli {d:.4}");
        assert!(d >= 0.95, "Dice interpolasi {d}");
    }
    let rep = s.measure().unwrap();
    let row = &rep.labels[0];
    assert_eq!(row.n_slices_interpolated, 2);
    assert_eq!(
        row.n_slices_annotated, n0,
        "irisan interpolasi tidak dihitung beranotasi"
    );
    assert_eq!(
        row.volume_cavalieri_um3, v0,
        "Cavalieri tidak boleh berubah oleh irisan interpolasi"
    );
    assert!(rep
        .provenance
        .by_source
        .iter()
        .any(|(k, n)| k == "interpolated" && *n == 2));
    assert!(rep.warnings.contains(&h3d_volume::Warning::HasDraft));
    // Interpolasi kedua kalinya tidak mengusulkan apa pun (irisan sudah berisi).
    s.interpolate_labels(TUMOR, &NoProgress).unwrap();
    assert!(s.pending_slides().is_empty());
    // Goresan yang tidak mengubah piksel apa pun tidak mengubah provenance.
    assert!(!s
        .paint_stroke(
            SlideId(9),
            TUMOR,
            &[[1536.0, 1536.0]],
            60.0,
            PaintMode::Replace
        )
        .unwrap());
    assert!(s
        .project()
        .annotation(SlideId(9), TUMOR)
        .unwrap()
        .source
        .is_interpolated());
    // Goresan tangan yang menambah piksel (di pojok, jauh dari tumor) di
    // atas hasil interpolasi → Manual, edited_from.
    assert!(s
        .paint_stroke(
            SlideId(9),
            TUMOR,
            &[[150.0, 150.0]],
            60.0,
            PaintMode::Replace
        )
        .unwrap());
    assert_eq!(
        s.project().annotation(SlideId(9), TUMOR).unwrap().source,
        AnnotationSource::Manual {
            by: "UJI".into(),
            edited_from: Some("interpolated".into())
        }
    );
    assert_eq!(s.measure().unwrap().labels[0].n_slices_interpolated, 1);
}

#[test]
fn vox_cache_is_reused_and_invalidated_when_labels_or_config_change() {
    let mut s = open_copy("cache");
    let vox = s.bundle().cache_vox_dir();
    let keys = |dir: &Path| -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(dir)
            .map(|rd| {
                rd.flatten()
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();
        v.sort();
        v
    };
    // Sebelum rekonstruksi: laporan tanpa besaran mesh.
    let rep = s.measure().unwrap();
    assert!(rep.labels[0].volume_mesh_um3.is_none() && rep.grid.is_none());
    assert!(s.meshes().unwrap().is_empty());

    let m1 = s.reconstruct(&NoProgress).unwrap();
    let k1 = keys(&vox);
    assert_eq!(k1.len(), 1);
    assert!(vox.join(&k1[0]).join("plan.json").is_file());
    assert!(vox.join(&k1[0]).join("001.vox").is_file());
    assert!(vox.join(&k1[0]).join("001.mesh").is_file());

    // Sesi lain memakai cache disk: mesh identik, laporan berisi besaran mesh.
    let path = s.path().to_path_buf();
    let other = Session::open(&path).unwrap();
    assert_eq!(other.meshes().unwrap(), m1);
    assert!(other.measure().unwrap().labels[0].volume_mesh_um3.is_some());
    assert_eq!(other.voxel_masks().unwrap().len(), 1);
    drop(other);

    // Label berubah → kunci baru, cache lama dibuang.
    s.paint_stroke(
        SlideId(12),
        TUMOR,
        &[[400.0, 400.0]],
        120.0,
        PaintMode::Replace,
    )
    .unwrap();
    assert!(s.meshes().unwrap().is_empty(), "mesh lama sudah basi");
    assert!(s.measure().unwrap().labels[0].volume_mesh_um3.is_none());
    let m2 = s.reconstruct(&NoProgress).unwrap();
    let k2 = keys(&vox);
    assert_eq!(k2.len(), 1);
    assert_ne!(k1, k2, "cache harus basi setelah label berubah");
    assert_ne!(m1, m2);
    // Undo mengembalikan label → kunci kembali ke semula.
    assert_eq!(s.undo(SlideId(12)).unwrap().as_deref(), Some("Kuas"));
    let m3 = s.reconstruct(&NoProgress).unwrap();
    assert_eq!(keys(&vox), k1);
    assert_eq!(m3, m1, "rekonstruksi deterministik");

    // ReconConfig berubah → kunci baru.
    let mut cfg = s.project().recon.clone();
    cfg.smooth_iterations = 3;
    s.set_recon_config(cfg).unwrap();
    s.reconstruct(&NoProgress).unwrap();
    assert_ne!(keys(&vox), k1);
    // Transformasi berubah → kunci baru.
    let k3 = keys(&vox);
    let t =
        Affine2::translation(25.0, 0.0).compose(&s.slide(SlideId(3)).unwrap().transform.affine());
    s.apply_manual_transform(SlideId(3), t).unwrap();
    s.reconstruct(&NoProgress).unwrap();
    assert_ne!(keys(&vox), k3);
    // clear_cache aman; rekonstruksi berikutnya membangun ulang.
    s.clear_cache().unwrap();
    assert!(keys(&vox).is_empty());
    assert_eq!(s.reconstruct(&NoProgress).unwrap().len(), 1);
}

#[test]
fn cancellation_returns_cancelled_and_leaves_project_consistent() {
    let mut s = open_copy("batal");
    let before = s.project().clone();
    let labels_before = read_tree(&s.path().join("labels"));
    assert!(matches!(
        s.reconstruct(&AlwaysCancelled),
        Err(EngineError::Cancelled)
    ));
    assert!(matches!(
        s.analyze_spatial(&AlwaysCancelled),
        Err(EngineError::Cancelled)
    ));
    assert!(matches!(
        s.interpolate_labels(TUMOR, &AlwaysCancelled),
        Err(EngineError::Cancelled) | Ok(())
    ));
    assert!(matches!(
        s.register_all_with(
            &RegisterRun {
                force: true,
                ..Default::default()
            },
            &AlwaysCancelled
        ),
        Err(EngineError::Cancelled)
    ));
    assert!(matches!(
        s.suggest_order(&AlwaysCancelled),
        Err(EngineError::Cancelled)
    ));
    let extra = shared().slides_dir().join("slice_000.png");
    assert!(matches!(
        s.import_slides(&[extra], Some([16.0, 16.0]), &AlwaysCancelled),
        Err(EngineError::Cancelled)
    ));
    // Tidak ada yang berubah, tidak ada cache setengah jadi, proyek terbuka.
    assert_eq!(s.project(), &before);
    assert!(!s.bundle().analysis_json().exists());
    let vox = s.bundle().cache_vox_dir();
    let complete = std::fs::read_dir(&vox)
        .map(|rd| rd.flatten().any(|e| e.path().join("plan.json").exists()))
        .unwrap_or(false);
    assert!(
        !complete,
        "cache voxel tidak boleh ditandai lengkap setelah batal"
    );
    let path = s.path().to_path_buf();
    drop(s);
    let reopened = Session::open(&path).unwrap();
    assert_eq!(reopened.project(), &before);
    assert_eq!(read_tree(&path.join("labels")), labels_before);

    // Job latar: batal lewat pegangan, lalu job normal selesai.
    let shared_session = std::sync::Arc::new(std::sync::Mutex::new(reopened));
    let started = std::sync::Arc::new(AtomicBool::new(false));
    let flag = started.clone();
    let job = spawn_session_job(&shared_session, "rekonstruksi", move |s, p| {
        flag.store(true, Ordering::SeqCst);
        // Tunggu sampai pembatalan terlihat, lalu jalankan operasi sungguhan.
        while !p.is_cancelled() {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        s.reconstruct(p)
    });
    while !started.load(Ordering::SeqCst) {
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    job.cancel();
    assert!(matches!(job.wait(), Err(EngineError::Cancelled)));

    let job = spawn_session_job(&shared_session, "rekonstruksi", |s, p| s.reconstruct(p));
    let progress = job.progress.clone();
    let meshes = job.wait().expect("job rekonstruksi");
    assert_eq!(meshes.len(), 1);
    assert_eq!(progress.fraction(), 1.0);
    assert!(!progress.stage().is_empty());
    // Panik di dalam job menjadi galat, tidak menjatuhkan proses.
    let job = spawn_job::<(), _>("panik", |_| panic!("sengaja"));
    assert!(matches!(job.wait(), Err(EngineError::InvalidState(_))));
}

#[test]
fn manual_edits_undo_and_provenance() {
    let mut s = open_copy("sunting");
    let slide = SlideId(12);
    let stroma = s.label_by_name("stroma").unwrap();
    let lbl_before = h3d_segment::encode_lbl(s.labels(slide).unwrap());
    assert!(!s.can_undo(slide));

    // Goresan pertama pada (irisan, label) membuat catatan Manual.
    assert!(s
        .paint_stroke(
            slide,
            stroma,
            &[[300.0, 300.0], [500.0, 320.0]],
            50.0,
            PaintMode::Replace
        )
        .unwrap());
    assert!(s.labels_dirty(slide) && s.is_dirty());
    let rec = s.project().annotation(slide, stroma).unwrap().clone();
    assert_eq!(
        rec.source,
        AnnotationSource::Manual {
            by: "UJI".into(),
            edited_from: None
        }
    );
    assert_eq!(rec.at, common::FIXED_TIME);
    // Undo mengembalikan keadaan persis dan membuang catatan; redo sebaliknya.
    assert_eq!(s.undo(slide).unwrap().as_deref(), Some("Kuas"));
    assert_eq!(
        h3d_segment::encode_lbl(s.labels(slide).unwrap()),
        lbl_before
    );
    assert!(s.project().annotation(slide, stroma).is_none());
    assert!(s.can_redo(slide));
    assert_eq!(s.redo(slide).unwrap().as_deref(), Some("Kuas"));
    assert!(s.project().annotation(slide, stroma).is_some());
    assert!(s.undo(SlideId(1)).unwrap().is_none());

    // Usulan AI diterima → catatan Ai; goresan tangan di atasnya → Manual + edited_from.
    let nerve = s.label_by_name("Saraf").unwrap();
    let proposal = Proposal {
        mask: Array2::from_elem((20, 20), 1u8),
        mpp: 16.0,
        origin_um: [2400.0, 2400.0],
        label: nerve,
        source: AnnotationSource::Ai {
            model: "uji".into(),
            version: "1".into(),
        },
    };
    let bytes_before = h3d_segment::encode_lbl(s.labels(slide).unwrap());
    assert!(s
        .accept_proposal_with_confidence(slide, &proposal, PaintMode::OnlyBackground, Some(0.8))
        .unwrap());
    assert_ne!(
        h3d_segment::encode_lbl(s.labels(slide).unwrap()),
        bytes_before
    );
    let rec = s.project().annotation(slide, nerve).unwrap().clone();
    assert!(matches!(rec.source, AnnotationSource::Ai { .. }));
    assert_eq!(rec.confidence, Some(0.8));
    s.mark_reviewed(slide, nerve, "YJ").unwrap();
    assert_eq!(
        s.project().annotation(slide, nerve).unwrap().review,
        ReviewStatus::Reviewed {
            by: "YJ".into(),
            at: common::FIXED_TIME.into()
        }
    );
    assert!(s
        .paint_stroke(slide, nerve, &[[2400.0, 2500.0]], 60.0, PaintMode::Replace)
        .unwrap());
    let rec = s.project().annotation(slide, nerve).unwrap().clone();
    assert_eq!(
        rec.source,
        AnnotationSource::Manual {
            by: "UJI".into(),
            edited_from: Some("ai".into())
        }
    );
    assert_eq!(
        rec.review,
        ReviewStatus::Draft,
        "suntingan membatalkan tinjauan"
    );
    // Menghapus seluruh label membuang catatannya.
    s.paint_stroke(slide, nerve, &[[2560.0, 2560.0]], 2000.0, PaintMode::Erase)
        .unwrap();
    assert!(s.project().annotation(slide, nerve).is_none());
    assert!(
        s.mark_reviewed(slide, nerve, "YJ").is_err(),
        "tidak ada anotasi untuk ditinjau"
    );
    assert!(s.mark_reviewed(slide, TUMOR, " ").is_err());

    // Segmentasi klasik hanya menghasilkan usulan; label tidak berubah.
    let bytes = h3d_segment::encode_lbl(s.labels(slide).unwrap());
    let prop = s
        .segment_classic(slide, stroma, &h3d_engine::ClassicParams::default())
        .unwrap();
    assert!(matches!(prop.source, AnnotationSource::Classic { .. }));
    assert!(!prop.is_empty());
    assert_eq!(h3d_segment::encode_lbl(s.labels(slide).unwrap()), bytes);

    // Salin ke irisan berikut: hanya mengisi latar, tercatat sebagai salinan.
    let next = s.copy_labels_to_next(slide).unwrap();
    assert_eq!(next, SlideId(13));
    let rec = s.project().annotation(next, stroma).unwrap().clone();
    assert_eq!(
        rec.source,
        AnnotationSource::Manual {
            by: "UJI".into(),
            edited_from: Some("copied".into())
        }
    );
    assert!(s.labels(next).unwrap().area_px(stroma) > 0);
    // Catatan tumor irisan 13 (impor) tidak tertimpa.
    assert!(matches!(
        s.project().annotation(next, TUMOR).unwrap().source,
        AnnotationSource::Imported { .. }
    ));

    // Tabel label.
    let id = s
        .add_label("In situ", [1, 2, 3, 255], h3d_core::LabelKind::Tumor)
        .unwrap();
    assert!(s
        .add_label("in SITU", [0; 4], h3d_core::LabelKind::Tumor)
        .is_err());
    let mut def = s.project().label(id).unwrap().clone();
    def.group = Some("Tumor".into());
    s.update_label(def).unwrap();
    s.remove_label(id).unwrap();
    assert!(
        s.remove_label(TUMOR).is_err(),
        "label yang masih dipakai tidak boleh dihapus"
    );
    assert!(s
        .paint_stroke(slide, LabelId(200), &[[1.0, 1.0]], 5.0, PaintMode::Replace)
        .is_err());

    // Simpan: label dan provenance terbuka kembali sama.
    s.save().unwrap();
    assert!(!s.is_dirty());
    let saved = s.project().clone();
    let bytes = h3d_segment::encode_lbl(s.labels(slide).unwrap());
    let path = s.path().to_path_buf();
    drop(s);
    let mut r = Session::open(&path).unwrap();
    assert_eq!(r.project(), &saved);
    assert_eq!(h3d_segment::encode_lbl(r.labels(slide).unwrap()), bytes);

    // Status irisan: mengecualikan irisan membuang catatan yang tetangganya berubah.
    r.set_status(SlideId(14), SlideStatus::Excluded).unwrap();
    assert!(r.slide(SlideId(15)).unwrap().reg.is_none());
    assert!(r.slide(SlideId(16)).unwrap().reg.is_some());
    assert_eq!(r.needs_registration(), vec![SlideId(15)]);
    let _ = true_transforms(&shared().manifest);
    let _: PathBuf = r.slide_path(SlideId(0)).unwrap();
}

/// Model tiruan: kelas 1 ("Tumor") bila kanal merah ternormalisasi gelap.
struct DarkIsTumor {
    spec: h3d_segment::ModelSpec,
}

impl h3d_segment::Segmenter for DarkIsTumor {
    fn spec(&self) -> &h3d_segment::ModelSpec {
        &self.spec
    }
    fn infer(&self, input: &[f32]) -> Result<Vec<f32>, h3d_segment::SegError> {
        let s = self.spec.input_size;
        let n = s * s;
        let mut out = vec![0.0f32; 2 * n];
        for i in 0..n {
            // input = (R/255 − 0,5) / 0,5; inti/tumor gelap → nilai kecil.
            out[n + i] = (-0.05 - input[i]) * 20.0;
        }
        Ok(out)
    }
}

#[test]
fn ai_segmentation_only_fills_pending_proposals_until_accepted() {
    let mut s = open_copy("ai");
    let slide = SlideId(12);
    let truth = s.labels(slide).unwrap().to_array(TUMOR);
    s.edit_labels(slide, None, "Kosongkan", |st| {
        for (tx, ty) in st.tile_keys() {
            st.remove_tile(tx, ty);
        }
    })
    .unwrap();
    let seg = DarkIsTumor {
        spec: h3d_segment::ModelSpec {
            name: "tiruan".into(),
            version: Some("0.1".into()),
            input_size: 64,
            mpp: 16.0,
            mean: [0.5; 3],
            std: [0.5; 3],
            classes: vec![
                h3d_segment::ClassSpec {
                    index: 0,
                    label: None,
                },
                h3d_segment::ClassSpec {
                    index: 1,
                    label: Some("Tumor".into()),
                },
            ],
            task: "segmentation".into(),
        },
    };
    s.segment_with(&seg, &[slide], &NoProgress).unwrap();
    assert!(
        s.labels(slide).unwrap().is_empty(),
        "usulan belum boleh mengubah label"
    );
    assert_eq!(s.pending_slides(), vec![slide]);
    let pending = s.pending_proposals(slide);
    assert_eq!(pending.len(), 1);
    assert!(pending[0].confidence.is_some_and(|c| c > 0.5));
    assert_eq!(pending[0].proposal.label, TUMOR);
    assert_eq!(
        s.accept_pending(slide, PaintMode::OnlyBackground).unwrap(),
        1
    );
    let rec = s.project().annotation(slide, TUMOR).unwrap().clone();
    assert_eq!(
        rec.source,
        AnnotationSource::Ai {
            model: "tiruan".into(),
            version: "0.1".into()
        }
    );
    assert!(rec.confidence.is_some());
    let got = s.labels(slide).unwrap().to_array(TUMOR);
    let d = dice(&got, &truth);
    println!("segmentasi model tiruan: Dice terhadap masker fantom {d:.3}");
    assert!(
        d > 0.5,
        "model tiruan harus menemukan daerah tumor yang lebih gelap (Dice {d})"
    );
    // Model tidak ada di disk → galat, bukan panik.
    assert!(s
        .segment_ai(Path::new("/tidak/ada/model"), &[slide], &NoProgress)
        .is_err());
    // mpp tidak bisa diganti pada irisan berlabel; pada irisan kosong bisa.
    assert!(s.set_mpp(slide, [8.0, 8.0]).is_err());
    s.set_mpp(SlideId(0), [8.0, 8.0]).unwrap();
    assert!(s.slide(SlideId(0)).unwrap().reg.is_none());
    assert!(s.slide(SlideId(1)).unwrap().reg.is_some());
}

#[test]
fn tre_estimate_falls_back_to_dice_proxy_for_records_without_residual() {
    // Proyek lama: `RegRecord` tanpa `residual_um`/`residual_fit_um`.
    let s = open_copy("tre-lama");
    let with_residual = s.tre_estimate_um().expect("taksiran TRE");
    let path = s.path().to_path_buf();
    drop(s);
    let file = path.join("project.json");
    let mut v: serde_json::Value = serde_json::from_slice(&std::fs::read(&file).unwrap()).unwrap();
    let mut stripped = 0;
    for slide in v["slides"].as_array_mut().unwrap() {
        if let Some(reg) = slide.get_mut("reg").and_then(|r| r.as_object_mut()) {
            assert!(reg.remove("residual_um").is_some());
            assert!(reg.remove("residual_fit_um").is_some());
            stripped += 1;
        }
    }
    assert_eq!(stripped, 23);
    std::fs::write(&file, serde_json::to_vec_pretty(&v).unwrap()).unwrap();
    let old = Session::open(&path).expect("proyek lama tetap terbuka");
    assert!(old
        .project()
        .slides
        .iter()
        .filter_map(|e| e.reg.as_ref())
        .all(|r| r.residual_um.is_none() && r.residual_fit_um.is_none()));
    let proxy = old.tre_estimate_um().expect("cadangan proksi Dice");
    // Proksi Dice berbatas bawah setengah piksel registrasi (8 µm).
    assert!(proxy >= 8.0, "proksi {proxy}");
    assert_eq!(old.tre_per_slide_um().len(), 23);
    println!("taksiran TRE: terukur {with_residual:.1} µm, proksi lama {proxy:.1} µm");
    // Menyimpan ulang tidak menambah field kosong ke berkas lama.
    let mut old = old;
    old.set_user_initials("X");
    old.save().unwrap();
    let text = std::fs::read_to_string(&file).unwrap();
    assert!(!text.contains("residual_um"));
}

/// PNG seukuran fantom tetapi isinya lain (untuk meniru berkas yang diganti).
fn write_other_png(path: &Path, side: u32) {
    write_png_variant(path, side, 90);
}

fn write_png_variant(path: &Path, side: u32, blue: u8) {
    let mut img = image::RgbaImage::new(side, side);
    for (x, y, p) in img.enumerate_pixels_mut() {
        *p = image::Rgba([(x % 251) as u8, (y % 241) as u8, blue, 255]);
    }
    img.save(path).unwrap();
}

#[test]
fn external_slide_fingerprint_is_enforced_until_relinked_or_accepted() {
    let sh = shared();
    // Proyek tersendiri dengan salinan tiga irisan, agar berkas fantom
    // bersama tidak tersentuh.
    let root = sh.root.path().join("sidik-jari");
    let slides = root.join("slides");
    std::fs::create_dir_all(&slides).unwrap();
    let paths: Vec<PathBuf> = (10..13)
        .map(|i| {
            let name = format!("slice_{i:03}.png");
            std::fs::copy(sh.slides_dir().join(&name), slides.join(&name)).unwrap();
            slides.join(name)
        })
        .collect();
    let dir = root.join("p.h3d");
    let mut s = Session::create(&dir).unwrap();
    let ids = s
        .import_slides(&paths, Some([16.0, 16.0]), &NoProgress)
        .unwrap();
    s.register_all(&RegOptions::default(), &NoProgress).unwrap();
    drop(s);

    // (1) Tidak berubah: tidak ada yang dilaporkan, piksel terbaca.
    let s = Session::open(&dir).unwrap();
    assert!(s.changed_slides().is_empty() && s.missing_slides().is_empty());
    assert!(s.recheck_slides().is_empty());
    let thumb = s.thumbnail(ids[1], 64).unwrap();
    s.registration_image(ids[1]).unwrap();
    drop(s);

    // (2) Berubah: isi berkas irisan ke-2 diganti (ukuran piksel sama).
    let original = std::fs::read(&paths[1]).unwrap();
    write_other_png(&paths[1], 192);
    let mut s = Session::open(&dir).expect("proyek tetap terbuka walau satu berkas berubah");
    assert_eq!(s.changed_slides(), vec![ids[1]]);
    assert!(s.missing_slides().is_empty());
    for err in [
        s.thumbnail(ids[1], 64).unwrap_err(),
        s.registration_image(ids[1]).unwrap_err(),
        s.open_source(ids[1])
            .err()
            .expect("open_source harus menolak"),
    ] {
        assert!(
            matches!(err, EngineError::SlideChanged { slide, .. } if slide == ids[1]),
            "{err:?}"
        );
        let text = err.to_string();
        assert!(
            text.contains("slice_011.png") && text.contains("berubah"),
            "{text}"
        );
    }
    // Irisan lain tetap terbaca; registrasi ulang gagal dengan galat jelas
    // dan tidak merusak proyek.
    s.thumbnail(ids[0], 64).unwrap();
    let before = s.project().clone();
    let run = RegisterRun {
        force: true,
        ..Default::default()
    };
    let err = s.register_all_with(&run, &NoProgress).unwrap_err();
    assert!(matches!(err, EngineError::SlideChanged { .. }), "{err:?}");
    assert_eq!(s.project().slides, before.slides);

    // (2a) Mengembalikan berkas asli: pemeriksaan ulang membersihkan daftar.
    std::fs::write(&paths[1], &original).unwrap();
    assert_eq!(
        s.changed_slides(),
        vec![ids[1]],
        "daftar belum diperiksa ulang"
    );
    assert!(s.recheck_slides().is_empty());
    assert_eq!(s.thumbnail(ids[1], 64).unwrap(), thumb);

    // (2b) Berkas diganti SETELAH proyek dibuka: tertangkap saat dibaca.
    write_other_png(&paths[1], 192);
    let err = s.thumbnail(ids[1], 64).unwrap_err();
    assert!(matches!(err, EngineError::SlideChanged { .. }), "{err:?}");
    assert_eq!(s.changed_slides(), vec![ids[1]]);

    // (2c) Menerima berkas baru: sidik jari tersimpan diperbarui, terbaca.
    s.accept_changed_slide(ids[1]).unwrap();
    assert!(s.changed_slides().is_empty());
    let new_thumb = s.thumbnail(ids[1], 64).unwrap();
    assert_ne!(new_thumb, thumb, "thumbnail dari berkas baru");
    drop(s);
    let mut s = Session::open(&dir).unwrap();
    assert!(s.changed_slides().is_empty(), "penerimaan tersimpan");
    // Berkas berukuran piksel lain tidak bisa diterima (label akan bergeser).
    write_other_png(&paths[1], 100);
    assert_eq!(s.recheck_slides(), vec![ids[1]]);
    let err = s.accept_changed_slide(ids[1]).unwrap_err();
    assert!(err.to_string().contains("impor ulang"), "{err}");
    assert_eq!(s.changed_slides(), vec![ids[1]]);

    // (3) Hilang: berkas irisan ke-3 dipindah.
    let moved = root.join("pindah.png");
    std::fs::rename(&paths[2], &moved).unwrap();
    assert_eq!(s.missing_slides(), vec![ids[2]]);
    assert_eq!(s.recheck_slides(), vec![ids[1]], "hilang bukan berubah");
    let err = s.thumbnail(ids[2], 64).unwrap_err();
    assert!(
        matches!(err, EngineError::Io(h3d_io::IoError::MissingFile(_))),
        "{err:?}"
    );
    // Tautkan ulang ke berkas yang sama di lokasi baru → cocok.
    assert!(s.relink_slide(ids[2], &moved).unwrap());
    assert!(s.missing_slides().is_empty());
    s.thumbnail(ids[2], 64).unwrap();
    // Tautkan irisan ke-2 ke berkas lain yang tidak cocok → tetap "berubah".
    let other = root.join("lain.png");
    write_png_variant(&other, 192, 200);
    assert!(!s.relink_slide(ids[1], &other).unwrap());
    assert_eq!(s.changed_slides(), vec![ids[1]]);
    assert!(s.thumbnail(ids[1], 64).is_err());
    // Galat pemakaian: berkas tidak ada, berkas milik irisan lain.
    assert!(s.relink_slide(ids[1], &root.join("tidak-ada.png")).is_err());
    assert!(s.relink_slide(ids[1], &moved).is_err());
    drop(s);
    let s = Session::open(&dir).unwrap();
    assert_eq!(s.changed_slides(), vec![ids[1]]);
    assert!(s.missing_slides().is_empty());
    assert_eq!(s.slide_path(ids[2]).unwrap(), moved);
}
