//! Uji bundel proyek: bolak-balik, berkas rusak, versi, klinis, tulis atomik.

mod common;

use std::fs;

use h3d_core::{ClinicalMeta, LabelId, Project, SlideId, SlideSourceRef, FORMAT_VERSION};
use h3d_io::{write_atomic, IoError, ProjectBundle};
use h3d_segment::{AttrStore, LabelStore, PaintMode};

fn new_bundle(dir: &std::path::Path) -> ProjectBundle {
    ProjectBundle::create(&dir.join("uji.h3d")).expect("buat bundel")
}

#[test]
fn project_json_roundtrip_is_exact() {
    let tmp = tempfile::tempdir().unwrap();
    let b = new_bundle(tmp.path());
    let p = common::sample_project();
    b.save_project(&p).unwrap();
    let bytes1 = fs::read(b.project_json()).unwrap();
    let back = b.load_project().unwrap();
    assert_eq!(p, back, "simpan → muat harus sama persis");
    b.save_project(&back).unwrap();
    let bytes2 = fs::read(b.project_json()).unwrap();
    assert_eq!(bytes1, bytes2, "simpan ulang harus byte-identik");
    assert!(!tmp.path().join("uji.h3d/project.json.tmp").exists());
    // Bundel bisa dibuka lagi; membuat di atas proyek yang ada ditolak.
    assert!(ProjectBundle::open(&b.root).is_ok());
    assert!(ProjectBundle::create(&b.root).is_err());
}

#[test]
fn truncated_or_garbage_project_is_error() {
    let tmp = tempfile::tempdir().unwrap();
    let b = new_bundle(tmp.path());
    b.save_project(&common::sample_project()).unwrap();
    let full = fs::read(b.project_json()).unwrap();
    for cut in [0usize, 1, full.len() / 3, full.len() / 2, full.len() - 3] {
        fs::write(b.project_json(), &full[..cut]).unwrap();
        assert!(b.load_project().is_err(), "terpotong di {cut} harus galat");
    }
    for garbage in [
        &b"\x00\xff\xfe sampah"[..],
        b"[]",
        b"42",
        b"{\"slides\": 7}",
    ] {
        fs::write(b.project_json(), garbage).unwrap();
        assert!(b.load_project().is_err(), "sampah harus galat");
    }
    fs::remove_file(b.project_json()).unwrap();
    assert!(matches!(b.load_project(), Err(IoError::MissingFile(_))));
    assert!(matches!(
        ProjectBundle::open(&b.root),
        Err(IoError::MissingFile(_))
    ));
}

#[test]
fn newer_format_version_is_rejected_and_older_is_migrated() {
    let tmp = tempfile::tempdir().unwrap();
    let b = new_bundle(tmp.path());
    let p = Project {
        format_version: FORMAT_VERSION + 1,
        ..Default::default()
    };
    b.save_project(&p).unwrap();
    match b.load_project() {
        Err(IoError::FormatTooNew { found, supported }) => {
            assert_eq!(found, u64::from(FORMAT_VERSION) + 1);
            assert_eq!(supported, FORMAT_VERSION);
        }
        other => panic!("diharapkan FormatTooNew, dapat {other:?}"),
    }
    // Manifes pra-rilis (versi 0): tanpa format_version, dengan penanda lama
    // dan blok klinis yang tidak boleh tinggal di manifes.
    let old = r#"{
        "sample": {"name": "lama", "clinical": {"subject_pseudo_id": "P-1"}},
        "slides": [{"id": 3, "mpp": [0.5, 0.5], "size_l0": [10, 10], "interpolated": true}],
        "labels": []
    }"#;
    fs::write(b.project_json(), old).unwrap();
    let p = b.load_project().expect("migrasi v0");
    assert_eq!(p.format_version, FORMAT_VERSION);
    assert_eq!(p.sample.name, "lama");
    assert_eq!(p.slides.len(), 1);
    assert!(p.sample.clinical.is_none());
}

#[test]
fn clinical_lives_only_in_clinical_json() {
    let tmp = tempfile::tempdir().unwrap();
    let b = new_bundle(tmp.path());
    let mut p = common::sample_project();
    b.save_project(&p).unwrap();
    assert!(b.load_project().unwrap().sample.clinical.is_none());
    assert!(b.load_clinical().unwrap().is_none());

    let c = ClinicalMeta {
        subject_pseudo_id: "PSEUDO-XYZZY-42".into(),
        age_years: Some(61),
        topography_icdo: Some("C50.4".into()),
        specimen_month: Some("2025-11".into()),
        ..Default::default()
    };
    p.sample.clinical = Some(c.clone());
    b.save_project(&p).unwrap();
    // save_project tidak menulis clinical.json dan tidak membocorkan isinya.
    assert!(!b.clinical_json().exists());
    let text = fs::read_to_string(b.project_json()).unwrap();
    assert!(!text.contains("PSEUDO-XYZZY-42") && !text.contains("clinical"));

    b.save_clinical(Some(&c)).unwrap();
    assert_eq!(b.load_project().unwrap().sample.clinical, Some(c));
    b.save_clinical(None).unwrap();
    assert!(!b.clinical_json().exists());
    assert!(b.load_project().unwrap().sample.clinical.is_none());
    // Menghapus yang sudah tidak ada bukan galat.
    b.save_clinical(None).unwrap();
    // clinical.json rusak → galat, bukan panik.
    fs::write(b.clinical_json(), b"{ rusak").unwrap();
    assert!(b.load_project().is_err());
}

#[test]
fn labels_and_attrs_roundtrip_and_corruption() {
    let tmp = tempfile::tempdir().unwrap();
    let b = new_bundle(tmp.path());
    let id = SlideId(7);
    assert!(b.load_labels(id).unwrap().is_none());
    let mut store = LabelStore::new(4.0, [4000.0, 3000.0]);
    store.paint_disc([1000.0, 900.0], 300.0, LabelId(1), PaintMode::Replace);
    b.save_labels(id, &store).unwrap();
    assert!(b.label_path(id).ends_with("labels/0007.lbl"));
    let back = b.load_labels(id).unwrap().expect("label ada");
    assert_eq!(back.area_px(LabelId(1)), store.area_px(LabelId(1)));
    assert_eq!(
        h3d_segment::encode_lbl(&back),
        h3d_segment::encode_lbl(&store)
    );

    let good = fs::read(b.label_path(id)).unwrap();
    for cut in [0usize, 3, 10, 27, good.len() / 2, good.len() - 1] {
        fs::write(b.label_path(id), &good[..cut]).unwrap();
        assert!(b.load_labels(id).is_err(), ".lbl terpotong di {cut}");
    }
    let mut bad = good.clone();
    bad[0] = b'X';
    fs::write(b.label_path(id), &bad).unwrap();
    assert!(b.load_labels(id).is_err(), "magic salah");
    fs::write(b.label_path(id), b"bukan berkas label sama sekali").unwrap();
    assert!(b.load_labels(id).is_err());

    // Penyimpanan kosong menghapus berkas.
    b.save_labels(id, &LabelStore::new(4.0, [4000.0, 3000.0]))
        .unwrap();
    assert!(!b.label_path(id).exists());

    let mut attr = AttrStore::new(16.0, [4000.0, 3000.0], 0.01, 0.0);
    attr.set_with_weight(10, 12, 37.5, 4);
    b.save_attr(id, "ki_67", &attr, 3).unwrap();
    assert!(b.attr_path(id, "ki_67").ends_with("attrs/0007.ki_67.atr"));
    let (a2, aid) = b.load_attr(id, "ki_67").unwrap().expect("atribut ada");
    assert_eq!(aid, 3);
    assert_eq!(a2.get(10, 12), Some(37.5));
    assert!(b.load_attr(id, "lain").unwrap().is_none());
    fs::write(b.attr_path(id, "ki_67"), b"H3DA rusak").unwrap();
    assert!(b.load_attr(id, "ki_67").is_err());
}

#[test]
fn atomic_write_leaves_no_tmp_and_failure_keeps_old_file() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("sub/dir/data.bin");
    write_atomic(&path, b"versi-1").unwrap();
    assert_eq!(fs::read(&path).unwrap(), b"versi-1");
    assert!(!tmp.path().join("sub/dir/data.bin.tmp").exists());
    write_atomic(&path, b"versi-2").unwrap();
    assert_eq!(fs::read(&path).unwrap(), b"versi-2");
    assert_eq!(fs::read_dir(tmp.path().join("sub/dir")).unwrap().count(), 1);

    // Paksa kegagalan: `.tmp` sudah ada sebagai folder sehingga tidak bisa
    // dibuat sebagai berkas. Berkas lama harus utuh.
    fs::create_dir(tmp.path().join("sub/dir/data.bin.tmp")).unwrap();
    let err = write_atomic(&path, b"versi-3-gagal");
    assert!(err.is_err());
    assert_eq!(
        fs::read(&path).unwrap(),
        b"versi-2",
        "berkas lama tidak boleh rusak"
    );
}

#[test]
fn slide_paths_relative_and_embedded() {
    let tmp = tempfile::tempdir().unwrap();
    let b = new_bundle(tmp.path());
    let slides = tmp.path().join("irisan");
    fs::create_dir_all(&slides).unwrap();
    let src = slides.join("a 01.PNG");
    fs::write(&src, b"isi slide palsu").unwrap();

    let rel = b.make_relative(&src);
    assert!(rel.is_relative(), "{rel:?}");
    assert_eq!(rel, std::path::PathBuf::from("../irisan/a 01.PNG"));
    let ext = SlideSourceRef::External {
        path: rel,
        size_bytes: 15,
        fingerprint: 0,
    };
    let resolved = b.resolve_slide_path(&ext);
    assert_eq!(fs::read(resolved).unwrap(), b"isi slide palsu");

    let emb = b.embed_slide(&src, SlideId(12)).unwrap();
    match &emb {
        SlideSourceRef::Embedded { rel_path } => {
            assert_eq!(rel_path, &std::path::PathBuf::from("slides/0012.png"));
        }
        other => panic!("diharapkan Embedded, dapat {other:?}"),
    }
    assert_eq!(
        fs::read(b.resolve_slide_path(&emb)).unwrap(),
        b"isi slide palsu"
    );
    assert!(!b.slides_dir().join("0012.png.tmp").exists());
    assert!(b
        .embed_slide(&slides.join("tidak-ada.png"), SlideId(1))
        .is_err());

    // clear_cache menghapus cache tetapi tidak menyentuh data proyek.
    fs::create_dir_all(b.cache_vox_dir()).unwrap();
    fs::write(b.cache_vox_dir().join("x"), b"1").unwrap();
    b.clear_cache().unwrap();
    assert!(!b.cache_dir().exists());
    assert!(b.slides_dir().join("0012.png").exists());
    b.clear_cache().unwrap();
}
