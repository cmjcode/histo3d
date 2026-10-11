//! Uji integrasi biner `h3d` (histo3d.md 8.9, 11.2): pipeline lengkap dari
//! fantom, keluaran emas, kode keluar, dan privasi data klinis.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

const FIXED_TIME: &str = "2026-01-02T03:04:05+07:00";

fn h3d(cwd: &Path) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_h3d"));
    c.current_dir(cwd).env("H3D_FIXED_TIME", FIXED_TIME);
    c
}

fn run(cwd: &Path, args: &[&str]) -> Output {
    h3d(cwd).args(args).output().expect("menjalankan h3d")
}

fn code(o: &Output) -> i32 {
    o.status.code().expect("kode keluar")
}

fn stdout(o: &Output) -> String {
    String::from_utf8(o.stdout.clone()).expect("stdout UTF-8")
}

fn stderr(o: &Output) -> String {
    String::from_utf8(o.stderr.clone()).expect("stderr UTF-8")
}

/// Jalankan dan tuntut kode keluar 0; kembalikan stdout.
fn ok(cwd: &Path, args: &[&str]) -> String {
    let o = run(cwd, args);
    assert_eq!(
        code(&o),
        0,
        "h3d {args:?} gagal\nstdout: {}\nstderr: {}",
        stdout(&o),
        stderr(&o)
    );
    stdout(&o)
}

/// Tuntut kegagalan dengan `expected` dan tepat satu baris galat berbahasa
/// Indonesia di stderr.
fn fails(cwd: &Path, args: &[&str], expected: i32) -> String {
    let o = run(cwd, args);
    let err = stderr(&o);
    assert_eq!(code(&o), expected, "h3d {args:?}: stderr {err}");
    assert_eq!(
        err.trim_end().lines().count(),
        1,
        "stderr harus satu baris: {err:?}"
    );
    assert!(err.starts_with("galat: "), "{err:?}");
    err
}

fn slices(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("slice_") && n.ends_with(".png"))
        .collect();
    v.sort();
    v
}

fn truth(dir: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(dir.join("truth.json")).unwrap()).unwrap()
}

/// phantom sudah ada di `<root>/ph`; jalankan new → import → register →
/// labels import → reconstruct di `<root>/<run>/p.h3d`.
fn pipeline(root: &Path, run_name: &str) -> PathBuf {
    let cwd = root.join(run_name);
    std::fs::create_dir_all(&cwd).unwrap();
    let t = truth(&root.join("ph"));
    let mpp = t["mpp"].as_f64().unwrap().to_string();
    let spacing = t["spacing_um"].as_f64().unwrap().to_string();
    ok(&cwd, &["new", "p.h3d"]);
    let names = slices(&root.join("ph"));
    let mut args: Vec<String> = vec!["import".into(), "p.h3d".into()];
    args.extend(names.iter().map(|n| format!("../ph/{n}")));
    args.extend(["--mpp".into(), mpp, "--spacing-um".into(), spacing]);
    let argv: Vec<&str> = args.iter().map(String::as_str).collect();
    let out = ok(&cwd, &argv);
    assert!(
        out.contains(&format!("{} irisan diimpor", names.len())),
        "{out}"
    );
    let out = ok(&cwd, &["register", "p.h3d", "--strict"]);
    assert!(out.contains("bertanda 0"), "{out}");
    for (i, s) in t["slices"].as_array().unwrap().iter().enumerate() {
        if let Some(g) = s["geojson"].as_str() {
            let out = ok(
                &cwd,
                &[
                    "labels",
                    "import",
                    "p.h3d",
                    "--slide",
                    &i.to_string(),
                    "--geojson",
                    &format!("../ph/{g}"),
                ],
            );
            assert!(out.contains("GeoJSON diimpor"), "{out}");
        }
    }
    let out = ok(&cwd, &["reconstruct", "p.h3d", "--smooth", "10"]);
    assert!(out.contains("Rekonstruksi selesai"), "{out}");
    // 8.7.1: `reconstruct` juga menulis volume label NIfTI ke bundel.
    assert!(out.contains("Volume label ditulis ke"), "{out}");
    let nii = cwd.join("p.h3d/out/volume/labels.nii.gz");
    let gz = std::fs::read(&nii).expect("out/volume/labels.nii.gz");
    assert_eq!(&gz[..2], &[0x1f, 0x8b], "gzip");
    cwd
}

fn phantom(root: &Path, extra: &[&str]) {
    let mut args = vec![
        "phantom",
        "--shape",
        "ellipsoid",
        "--out",
        "ph",
        "--slices",
        "20",
        "--seed",
        "5",
        "--size",
        "160",
    ];
    args.extend_from_slice(extra);
    let out = ok(root, &args);
    assert!(out.contains("Fantom ditulis"), "{out}");
    assert!(root.join("ph/truth.json").is_file());
}

#[test]
fn end_to_end_pipeline_from_phantom_through_every_export() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    phantom(root, &["--margin", "300", "--ihc", "0.4"]);
    let t = truth(&root.join("ph"));
    assert_eq!(t["slices"].as_array().unwrap().len(), 20);
    assert_eq!(t["analytic"]["d_min_um"].as_f64(), Some(300.0));

    // info / thumb.
    let info: Value = serde_json::from_str(&ok(root, &["info", "ph/slice_000.png"])).unwrap();
    assert_eq!(info["size_l0"], serde_json::json!([160, 160]));
    assert!(
        info["mpp"].is_null(),
        "PNG tidak membawa mpp; tidak boleh ditebak"
    );
    ok(
        root,
        &[
            "thumb",
            "ph/slice_003.png",
            "--out",
            "thumb.png",
            "--max",
            "64",
        ],
    );
    assert!(root.join("thumb.png").is_file());

    let cwd = pipeline(root, "a");
    let cwd = cwd.as_path();

    // qc.
    let qc: Value = serde_json::from_str(&ok(cwd, &["qc", "p.h3d", "--json"])).unwrap();
    let rows = qc.as_array().unwrap();
    assert_eq!(rows.len(), 20);
    assert_eq!(rows.iter().filter(|r| r["is_reference"] == true).count(), 1);
    assert!(rows
        .iter()
        .all(|r| r["registered"] == true && r["flagged"] == false));
    assert!(ok(cwd, &["qc", "p.h3d"]).contains("acuan"));

    // measure.
    let m: Value = serde_json::from_str(&ok(cwd, &["measure", "p.h3d", "--json"])).unwrap();
    assert_eq!(m["notice"], "Hanya untuk riset, bukan untuk diagnosis");
    let tumor = m["labels"]
        .as_array()
        .unwrap()
        .iter()
        .find(|l| l["name"] == "Tumor")
        .expect("baris Tumor");
    let analytic = t["analytic"]["volume_um3"].as_f64().unwrap();
    let cav = tumor["volume_cavalieri_um3"].as_f64().unwrap();
    let mesh = tumor["volume_mesh_um3"].as_f64().unwrap();
    println!(
        "CLI: analitik {:.4} mm³, Cavalieri {:.4} mm³ ({:+.2} %), mesh {:.4} mm³ ({:+.2} %)",
        analytic / 1e9,
        cav / 1e9,
        (cav / analytic - 1.0) * 100.0,
        mesh / 1e9,
        (mesh / analytic - 1.0) * 100.0
    );
    assert!(
        (cav - analytic).abs() / analytic <= 0.03,
        "Cavalieri {cav} vs {analytic}"
    );
    assert!(
        (mesh - analytic).abs() / analytic <= 0.08,
        "mesh {mesh} vs {analytic}"
    );
    assert!(m["dimension_uncertainty_um"].as_f64().unwrap() > 0.0);
    let text = ok(cwd, &["measure", "p.h3d"]);
    assert!(text.contains("Tumor") && text.contains("Hanya untuk riset"));

    // analyze: margin sesuai fantom, arah +X, dengan ketidakpastian.
    let a: Value = serde_json::from_str(&ok(cwd, &["analyze", "p.h3d", "--json"])).unwrap();
    assert_eq!(a["format"], "h3d-analysis");
    let margin = &a["analysis"]["margin"];
    let d = margin["d_min"]["value_um"].as_f64().expect("d_min");
    let pm = margin["d_min"]["plus_minus_um"]
        .as_f64()
        .expect("ketidakpastian d_min");
    println!(
        "CLI: margin d_min {d:.1} ± {pm:.1} µm (benar 300 µm), arah {}",
        margin["direction"]
    );
    assert!((d - 300.0).abs() <= 48.0, "d_min {d}");
    assert!(pm > 0.0);
    assert_eq!(margin["direction"], "PosX");
    assert_eq!(margin["status"], "close");
    assert_eq!(
        std::fs::read_to_string(cwd.join("p.h3d/out/analysis.json")).unwrap(),
        ok(cwd, &["analyze", "p.h3d", "--json"]),
        "analyze kedua kali identik dan sama dengan berkasnya"
    );
    assert!(ok(cwd, &["analyze", "p.h3d"]).contains("Jarak minimum ke tepi reseksi"));

    // export: semua format.
    for (format, expect) in [
        ("glb", vec!["mesh.glb"]),
        ("stl", vec!["tumor_mm.stl"]),
        ("obj", vec!["mesh_mm.obj"]),
        ("nifti", vec!["labels.nii.gz", "margin_distance.nii.gz"]),
        ("geojson", vec!["0010.geojson"]),
        ("report", vec!["report.md", "report.csv"]),
        ("analysis", vec!["analysis.json"]),
    ] {
        let out = ok(
            cwd,
            &["export", "p.h3d", "--format", format, "--out", "ekspor"],
        );
        for f in expect {
            assert!(out.contains(f), "{format}: {out}");
            let p = cwd.join("ekspor").join(f);
            assert!(
                p.is_file() && std::fs::metadata(&p).unwrap().len() > 0,
                "{f}"
            );
        }
    }
    let glb = std::fs::read(cwd.join("ekspor/mesh.glb")).unwrap();
    assert_eq!(&glb[0..4], b"glTF");
    assert_eq!(
        u32::from_le_bytes([glb[8], glb[9], glb[10], glb[11]]) as usize,
        glb.len()
    );
    let report = std::fs::read_to_string(cwd.join("ekspor/report.md")).unwrap();
    assert!(report.contains("Hanya untuk riset, bukan untuk diagnosis"));
    assert!(report.contains("## 3. Margin") && report.contains(" ± "));
    let gj: Value =
        serde_json::from_str(&std::fs::read_to_string(cwd.join("ekspor/0010.geojson")).unwrap())
            .unwrap();
    let feat = &gj["features"][0];
    assert!(
        feat["properties"]["measurements"]["area_um2"]
            .as_f64()
            .unwrap()
            > 0.0
    );
    // GLB berwarna skalar jarak margin.
    ok(
        cwd,
        &[
            "export",
            "p.h3d",
            "--format",
            "glb",
            "--out",
            "ekspor-skalar",
            "--scalar",
            "margin_distance",
        ],
    );
    let glb = std::fs::read(cwd.join("ekspor-skalar/mesh.glb")).unwrap();
    let json_len = u32::from_le_bytes([glb[12], glb[13], glb[14], glb[15]]) as usize;
    let doc: Value = serde_json::from_slice(&glb[20..20 + json_len]).unwrap();
    let attrs = &doc["meshes"][0]["primitives"][0]["attributes"];
    assert!(attrs.get("COLOR_0").is_some() && attrs.get("_H3D_MARGIN_DISTANCE").is_some());
    fails(
        cwd,
        &[
            "export",
            "p.h3d",
            "--format",
            "glb",
            "--out",
            "x",
            "--scalar",
            "tidak_ada",
        ],
        2,
    );

    // review, interpolate, suggest-order.
    let out = ok(
        cwd,
        &[
            "review", "p.h3d", "--slide", "10", "--label", "Tumor", "--by", "YJ",
        ],
    );
    assert!(out.contains("ditandai sudah ditinjau"));
    let project = std::fs::read_to_string(cwd.join("p.h3d/project.json")).unwrap();
    assert!(
        project.contains(FIXED_TIME),
        "jam beku dipakai untuk provenance"
    );
    fails(
        cwd,
        &[
            "review",
            "p.h3d",
            "--slide",
            "10",
            "--label",
            "Tidak Ada",
            "--by",
            "YJ",
        ],
        2,
    );
    let out = ok(cwd, &["interpolate", "p.h3d", "--label", "tumor"]);
    assert!(out.contains("0 irisan diisi"), "{out}");
    let s: Value = serde_json::from_str(&ok(cwd, &["suggest-order", "p.h3d", "--json"])).unwrap();
    assert_eq!(s["order"].as_array().unwrap().len(), 20);
    assert_eq!(s["applied"], false);
    assert!(ok(cwd, &["suggest-order", "p.h3d"]).contains("Urutan usulan"));

    // Atribut IHC dari gambar sintetis (proyek terpisah, mpp 1 µm).
    ok(root, &["new", "ihc.h3d"]);
    ok(root, &["import", "ihc.h3d", "ph/ihc.png", "--mpp", "1"]);
    let out = ok(
        root,
        &[
            "attrs",
            "compute",
            "ihc.h3d",
            "--kind",
            "ihc",
            "--slides",
            "0",
            "--name",
            "Ki-67 % positif",
            "--antibody",
            "Ki-67",
        ],
    );
    assert!(out.contains("ihc-nuclear-v1"), "{out}");
    let m: Value = serde_json::from_str(&ok(root, &["measure", "ihc.h3d", "--json"])).unwrap();
    let pct = m["attrs"][0]["per_slide"][0]["mean"].as_f64().unwrap();
    let n = t["ihc"]["n_nuclei"].as_f64().unwrap();
    let true_pct = 100.0 * t["ihc"]["n_positive"].as_f64().unwrap() / n;
    println!("CLI: IHC persen positif {pct:.2} % (benar {true_pct:.2} %)");
    assert!((pct - true_pct).abs() <= 2.0, "{pct} vs {true_pct}");
    assert!(ok(
        root,
        &["attrs", "compute", "ihc.h3d", "--kind", "nuclei", "--slides", "0"]
    )
    .contains("nuclei-density-v1"));
}

#[test]
fn measure_json_is_identical_across_two_seeded_runs() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    phantom(root, &["--vessel"]);
    let a = pipeline(root, "a");
    let b = pipeline(root, "b");
    let ma = ok(&a, &["measure", "p.h3d", "--json"]);
    let mb = ok(&b, &["measure", "p.h3d", "--json"]);
    assert!(ma.len() > 500, "{ma}");
    assert_eq!(ma, mb, "measure --json harus identik antar dua jalan");
    let aa = ok(&a, &["analyze", "p.h3d", "--json"]);
    let ab = ok(&b, &["analyze", "p.h3d", "--json"]);
    assert_eq!(aa, ab, "analyze --json harus identik antar dua jalan");
    let v: Value = serde_json::from_str(&aa).unwrap();
    let contact = &v["analysis"]["contacts"][0];
    assert_eq!(
        contact["result"]["any"], true,
        "pembuluh menembus tumor: {contact}"
    );
    for f in ["project.json", "out/analysis.json", "out/mesh/mesh.glb"] {
        assert_eq!(
            std::fs::read(a.join("p.h3d").join(f)).unwrap(),
            std::fs::read(b.join("p.h3d").join(f)).unwrap(),
            "{f} harus byte-identik"
        );
    }
    // Setelah analisis, laporan memuat analisis dan tetap identik.
    assert_eq!(
        ok(&a, &["measure", "p.h3d", "--json"]),
        ok(&b, &["measure", "p.h3d", "--json"])
    );
    // Fantom sendiri deterministik.
    let again = root.join("ulang");
    std::fs::create_dir_all(&again).unwrap();
    phantom(&again, &["--vessel"]);
    for f in ["truth.json", "slice_007.png", "slice_007.geojson"] {
        assert_eq!(
            std::fs::read(root.join("ph").join(f)).unwrap(),
            std::fs::read(again.join("ph").join(f)).unwrap(),
            "{f}"
        );
    }
}

fn write_disc_png(path: &Path, size: u32, center: (f64, f64), radius: f64, rgb: [u8; 3]) {
    let mut img = image::RgbaImage::from_pixel(size, size, image::Rgba([255, 255, 255, 255]));
    for (x, y, p) in img.enumerate_pixels_mut() {
        let dx = f64::from(x) + 0.5 - center.0;
        let dy = f64::from(y) + 0.5 - center.1;
        if dx * dx + dy * dy <= radius * radius {
            // Tekstur sederhana agar kanal hematoksilin tidak konstan.
            let k = ((x / 4 + y / 4) % 2) as u8 * 30;
            *p = image::Rgba([
                rgb[0].saturating_add(k),
                rgb[1].saturating_add(k),
                rgb[2],
                255,
            ]);
        }
    }
    img.save(path).unwrap();
}

#[test]
fn exit_codes_and_single_line_errors() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();

    // 0: bantuan dan versi.
    assert_eq!(code(&run(root, &["--help"])), 0);
    assert_eq!(code(&run(root, &["--version"])), 0);
    assert!(stdout(&run(root, &["phantom", "--help"])).contains("--shuffle"));

    // 1: kesalahan pemakaian.
    fails(root, &[], 1);
    fails(root, &["perintah-tidak-ada"], 1);
    fails(root, &["new"], 1);
    fails(
        root,
        &["export", "x.h3d", "--format", "docx", "--out", "o"],
        1,
    );
    fails(root, &["import", "x.h3d", "a.png", "--mpp", "abc"], 1);
    fails(root, &["import", "x.h3d", "a.png", "--mpp", "-1"], 1);
    fails(root, &["import", "x.h3d"], 1);
    fails(root, &["register", "x.h3d", "--model", "bspline"], 1);
    fails(root, &["phantom", "--shape", "kubus", "--out", "p"], 1);

    // 2: berkas hilang atau rusak.
    let err = fails(root, &["measure", "tidak-ada.h3d"], 2);
    assert!(err.contains("tidak bisa membuka proyek"), "{err}");
    fails(root, &["info", "tidak-ada.svs"], 2);
    std::fs::write(root.join("sampah.png"), b"bukan gambar").unwrap();
    fails(root, &["info", "sampah.png"], 2);
    fails(root, &["thumb", "sampah.png", "--out", "t.png"], 2);
    ok(root, &["new", "rusak.h3d"]);
    fails(root, &["new", "rusak.h3d"], 2);
    std::fs::write(root.join("rusak.h3d/project.json"), b"{\"slides\": [1, 2").unwrap();
    fails(root, &["measure", "rusak.h3d"], 2);
    fails(root, &["qc", "rusak.h3d"], 2);
    std::fs::write(
        root.join("rusak.h3d/project.json"),
        b"{\"format_version\": 999, \"slides\": []}",
    )
    .unwrap();
    let err = fails(root, &["measure", "rusak.h3d"], 2);
    assert!(err.contains("999"), "{err}");

    // 2: impor PNG tanpa mpp, dengan pesan mpp; proyek tidak berubah.
    write_disc_png(
        &root.join("s1.png"),
        128,
        (64.0, 64.0),
        46.0,
        [120, 60, 150],
    );
    write_disc_png(
        &root.join("s2.png"),
        128,
        (30.0, 28.0),
        9.0,
        [200, 140, 190],
    );
    ok(root, &["new", "p.h3d"]);
    let err = fails(root, &["import", "p.h3d", "s1.png", "s2.png"], 2);
    assert!(err.contains("mpp") && err.contains("--mpp"), "{err}");
    assert!(
        ok(root, &["qc", "p.h3d"]).lines().count() == 1,
        "proyek harus tetap kosong"
    );
    fails(root, &["import", "p.h3d", "sampah.png", "--mpp", "16"], 2);
    fails(root, &["reconstruct", "p.h3d"], 2);
    fails(
        root,
        &["export", "p.h3d", "--format", "glb", "--out", "o"],
        2,
    );
    fails(
        root,
        &[
            "labels",
            "import",
            "p.h3d",
            "--slide",
            "0",
            "--geojson",
            "tak-ada.geojson",
        ],
        2,
    );
    fails(root, &["phantom", "--out", "ph", "--breach"], 2);

    // 3: QC gagal dengan --strict; tanpa --strict tetap 0.
    ok(
        root,
        &["import", "p.h3d", "s1.png", "s2.png", "--mpp", "16"],
    );
    let o = run(root, &["register", "p.h3d"]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    assert!(stdout(&o).contains("bertanda 1"), "{}", stdout(&o));
    let o = run(root, &["register", "p.h3d", "--strict", "--force"]);
    assert_eq!(code(&o), 3, "stdout {} stderr {}", stdout(&o), stderr(&o));
    let err = stderr(&o);
    assert!(err.starts_with("galat: ") && err.contains("QC"), "{err}");
    assert_eq!(err.trim_end().lines().count(), 1);
    let qc: Value = serde_json::from_str(&ok(root, &["qc", "p.h3d", "--json"])).unwrap();
    assert_eq!(
        qc.as_array()
            .unwrap()
            .iter()
            .filter(|r| r["flagged"] == true)
            .count(),
        1
    );
}

#[test]
fn clinical_set_report_and_strip() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    ok(root, &["new", "k.h3d"]);
    let clinical = serde_json::json!({
        "subject_pseudo_id": "ZQPSEUDO-CLI-90817",
        "age_years": 64,
        "sex": "female",
        "topography_icdo": "ZQTOPO-C18.7",
        "morphology_icdo": "ZQMORF-8140/3",
        "grade": { "system": "ZQSYS-WHO", "value": "ZQVAL-2" },
        "molecular": [{ "gene": "ZQGENE-KRAS", "result": "ZQRES-G12D", "method": null }],
        "specimen_month": "2025-11"
    });
    std::fs::write(
        root.join("klinis.json"),
        serde_json::to_vec_pretty(&clinical).unwrap(),
    )
    .unwrap();
    let markers = [
        "ZQPSEUDO-CLI-90817",
        "ZQTOPO-C18.7",
        "ZQMORF-8140/3",
        "ZQSYS-WHO",
        "ZQVAL-2",
        "ZQGENE-KRAS",
        "ZQRES-G12D",
    ];
    let fields = [
        "subject_pseudo_id",
        "age_years",
        "sex",
        "topography_icdo",
        "morphology_icdo",
        "laterality",
        "procedure",
        "grade",
        "ptnm",
        "neoadjuvant",
        "molecular",
        "specimen_month",
    ];

    let o = run(root, &["clinical", "set", "k.h3d", "--json", "klinis.json"]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    for m in markers {
        assert!(
            !stdout(&o).contains(m) && !stderr(&o).contains(m),
            "isi klinis tercetak"
        );
    }
    assert!(root.join("k.h3d/clinical.json").is_file());
    let project = std::fs::read_to_string(root.join("k.h3d/project.json")).unwrap();
    for m in markers {
        assert!(!project.contains(m), "{m} bocor ke project.json");
    }
    for f in fields {
        assert!(
            !project.contains(f),
            "field klinis {f} muncul di project.json"
        );
    }

    // Tanpa flag: tidak ada isi klinis di laporan maupun keluaran perintah.
    let o = run(
        root,
        &["export", "k.h3d", "--format", "report", "--out", "tanpa"],
    );
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    for name in ["report.md", "report.csv"] {
        let text = std::fs::read_to_string(root.join("tanpa").join(name)).unwrap();
        assert!(text.contains("Hanya untuk riset, bukan untuk diagnosis"));
        for m in markers {
            assert!(
                !text.contains(m),
                "{m} bocor ke {name} tanpa --with-clinical"
            );
        }
        for f in fields {
            assert!(!text.contains(f), "field klinis {f} muncul di {name}");
        }
        assert!(!text.contains("2025-11"));
    }
    assert!(!root.join("tanpa/clinical.json").exists());
    let measure = ok(root, &["measure", "k.h3d", "--json"]);
    for m in markers {
        assert!(!measure.contains(m));
    }

    // Dengan flag: konteks klinis tercetak dan clinical.json ikut diekspor.
    ok(
        root,
        &[
            "export",
            "k.h3d",
            "--format",
            "report",
            "--out",
            "dengan",
            "--with-clinical",
        ],
    );
    let text = std::fs::read_to_string(root.join("dengan/report.md")).unwrap();
    for m in markers {
        assert!(
            text.contains(m),
            "{m} seharusnya ada dengan --with-clinical"
        );
    }
    assert!(text.contains("64") && text.contains("perempuan") && text.contains("2025-11"));
    assert!(root.join("dengan/clinical.json").is_file());

    // strip: berkas hilang, proyek tetap terbuka, laporan tetap bisa dibuat.
    let out = ok(root, &["clinical", "strip", "k.h3d"]);
    assert!(out.contains("dihapus"));
    assert!(!root.join("k.h3d/clinical.json").exists());
    ok(root, &["measure", "k.h3d"]);
    ok(
        root,
        &[
            "export",
            "k.h3d",
            "--format",
            "report",
            "--out",
            "sesudah",
            "--with-clinical",
        ],
    );
    let text = std::fs::read_to_string(root.join("sesudah/report.md")).unwrap();
    for m in markers {
        assert!(!text.contains(m));
    }
    ok(root, &["clinical", "strip", "k.h3d"]);

    // Masukan klinis tidak sah → kode 2, satu baris.
    std::fs::write(root.join("buruk.json"), b"{ bukan json").unwrap();
    fails(
        root,
        &["clinical", "set", "k.h3d", "--json", "buruk.json"],
        2,
    );
    std::fs::write(root.join("salah-ketik.json"), br#"{"subject_id": "X"}"#).unwrap();
    let err = fails(
        root,
        &["clinical", "set", "k.h3d", "--json", "salah-ketik.json"],
        2,
    );
    assert!(err.contains("subject_id"), "{err}");
    std::fs::write(
        root.join("hari.json"),
        br#"{"subject_pseudo_id": "P", "specimen_month": "2025-11-03"}"#,
    )
    .unwrap();
    fails(
        root,
        &["clinical", "set", "k.h3d", "--json", "hari.json"],
        2,
    );
    assert!(!root.join("k.h3d/clinical.json").exists());
}

#[test]
fn changed_and_missing_slide_files_warn_on_stderr_and_relink_recovers() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    for (i, cx) in [(0, 60.0), (1, 64.0), (2, 68.0)] {
        write_disc_png(
            &root.join(format!("s{i}.png")),
            128,
            (cx, 64.0),
            40.0,
            [150, 90, 170],
        );
    }
    ok(root, &["new", "p.h3d"]);
    ok(
        root,
        &[
            "import", "p.h3d", "s0.png", "s1.png", "s2.png", "--mpp", "16",
        ],
    );
    // Tidak berubah: stderr kosong.
    let o = run(root, &["qc", "p.h3d"]);
    assert_eq!(code(&o), 0);
    assert_eq!(stderr(&o), "", "tanpa peringatan bila berkas utuh");

    // Berubah: isi s1.png diganti. Proyek tetap terbuka (kode 0) dengan satu
    // baris peringatan; perintah yang butuh pikselnya gagal dengan kode 2.
    write_disc_png(&root.join("s1.png"), 128, (30.0, 30.0), 20.0, [90, 60, 200]);
    let o = run(root, &["qc", "p.h3d"]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let err = stderr(&o);
    assert_eq!(err.trim_end().lines().count(), 1, "{err:?}");
    assert!(
        err.starts_with("peringatan: berkas slide irisan 1 (s1.png) berubah sejak diimpor"),
        "{err:?}"
    );
    let o = run(root, &["register", "p.h3d"]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    let err = stderr(&o);
    let last = err.trim_end().lines().last().unwrap_or_default().to_owned();
    assert!(
        last.starts_with("galat: ") && last.contains("berubah sejak diimpor"),
        "{err:?}"
    );

    // Hilang: s2.png dipindah → peringatan kedua, berbeda bunyinya.
    std::fs::rename(root.join("s2.png"), root.join("pindah.png")).unwrap();
    let o = run(root, &["qc", "p.h3d"]);
    assert_eq!(code(&o), 0);
    let err = stderr(&o);
    assert_eq!(err.trim_end().lines().count(), 2, "{err:?}");
    assert!(
        err.contains("peringatan: berkas slide irisan 2 (s2.png) tidak ditemukan"),
        "{err:?}"
    );

    // Pemulihan: tautkan ulang yang pindah, terima yang berubah.
    fails_with_warnings(root, &["relink", "p.h3d", "--slide", "2"], 2);
    let out = ok(
        root,
        &["relink", "p.h3d", "--slide", "2", "--path", "pindah.png"],
    );
    assert!(out.contains("sidik jari cocok"), "{out}");
    let out = ok(root, &["relink", "p.h3d", "--slide", "1", "--accept"]);
    assert!(out.contains("diterima"), "{out}");
    let o = run(root, &["qc", "p.h3d"]);
    assert_eq!((code(&o), stderr(&o).as_str()), (0, ""));
    // Menautkan ke berkas yang tidak cocok tanpa --accept: tetap ditolak baca.
    write_disc_png(
        &root.join("lain.png"),
        128,
        (90.0, 90.0),
        25.0,
        [60, 60, 210],
    );
    let out = ok(
        root,
        &["relink", "p.h3d", "--slide", "0", "--path", "lain.png"],
    );
    assert!(out.contains("sidik jari BERBEDA"), "{out}");
    // Berkas milik irisan lain ditolak.
    fails_with_warnings(
        root,
        &["relink", "p.h3d", "--slide", "0", "--path", "s1.png"],
        2,
    );
    let o = run(root, &["qc", "p.h3d"]);
    assert!(stderr(&o).contains("irisan 0"), "{}", stderr(&o));
}

/// Seperti `fails`, tetapi mengizinkan baris peringatan sebelum baris galat.
fn fails_with_warnings(cwd: &Path, args: &[&str], expected: i32) -> String {
    let o = run(cwd, args);
    let err = stderr(&o);
    assert_eq!(code(&o), expected, "h3d {args:?}: stderr {err}");
    let last = err.trim_end().lines().last().unwrap_or_default().to_owned();
    assert!(last.starts_with("galat: "), "{err:?}");
    assert!(
        err.trim_end()
            .lines()
            .rev()
            .skip(1)
            .all(|l| l.starts_with("peringatan: ")),
        "{err:?}"
    );
    last
}

// ------------------------------------------------------------ berkas emas

/// `testdata/golden/` di akar workspace.
fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("testdata")
        .join("golden")
}

/// Platform pembuat berkas emas, mis. `aarch64-macos`.
fn platform() -> String {
    format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS)
}

/// Jalankan pipeline lengkap atas fantom berbenih dan kembalikan
/// `(measure --json, out/analysis.json)`.
fn golden_outputs(phantom_args: &[&str]) -> (String, String) {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let mut args = vec!["phantom", "--shape", "ellipsoid", "--out", "ph"];
    args.extend_from_slice(phantom_args);
    ok(root, &args);
    let cwd = pipeline(root, "emas");
    ok(&cwd, &["analyze", "p.h3d"]);
    let measure = ok(&cwd, &["measure", "p.h3d", "--json"]);
    let analysis = std::fs::read_to_string(cwd.join("p.h3d/out/analysis.json")).unwrap();
    (measure, analysis)
}

/// Bandingkan dua nilai JSON: struktur dan teks harus sama; bilangan boleh
/// berbeda `tol` relatif (dipakai hanya di platform selain pembuat berkas
/// emas, tempat FFT/libm bisa berbeda pada bit terakhir).
fn json_close(a: &Value, b: &Value, tol: f64, path: &str) -> Result<(), String> {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            let (x, y) = (
                x.as_f64().unwrap_or(f64::NAN),
                y.as_f64().unwrap_or(f64::NAN),
            );
            if (x - y).abs() <= tol * x.abs().max(y.abs()).max(1.0) {
                Ok(())
            } else {
                Err(format!("{path}: {x} ≠ {y}"))
            }
        }
        (Value::Array(x), Value::Array(y)) => {
            if x.len() != y.len() {
                return Err(format!("{path}: panjang {} ≠ {}", x.len(), y.len()));
            }
            x.iter()
                .zip(y)
                .enumerate()
                .try_for_each(|(i, (p, q))| json_close(p, q, tol, &format!("{path}[{i}]")))
        }
        (Value::Object(x), Value::Object(y)) => {
            if x.len() != y.len() || x.keys().any(|k| !y.contains_key(k)) {
                return Err(format!("{path}: kunci berbeda"));
            }
            x.iter()
                .try_for_each(|(k, p)| json_close(p, &y[k], tol, &format!("{path}.{k}")))
        }
        // Kunci hash masukan bergantung pada bit transformasi; hanya
        // dibandingkan persis di platform pembuat.
        (Value::String(_), Value::String(_)) if path.ends_with(".input_key") => Ok(()),
        _ if a == b => Ok(()),
        _ => Err(format!("{path}: {a} ≠ {b}")),
    }
}

/// Baris pertama yang berbeda, untuk pesan galat yang berguna.
fn first_diff(want: &str, got: &str) -> String {
    for (i, (w, g)) in want.lines().zip(got.lines()).enumerate() {
        if w != g {
            return format!("baris {}:\n  emas : {w}\n  kini : {g}", i + 1);
        }
    }
    format!(
        "jumlah baris berbeda: emas {}, kini {}",
        want.lines().count(),
        got.lines().count()
    )
}

/// Tes emas (11.2): keluaran CLI atas dua fantom berbenih dibandingkan byte
/// demi byte dengan berkas tersimpan di `testdata/golden/`. Selisih berarti
/// PERILAKU berubah; bila memang disengaja, segarkan dengan
/// `H3D_UPDATE_GOLDEN=1 cargo test -p h3d-cli golden` dan tinjau diff-nya
/// (lihat `testdata/README.md`).
#[test]
fn golden_files_match_cli_output_of_seeded_phantoms() {
    let dir = golden_dir();
    let update = std::env::var_os("H3D_UPDATE_GOLDEN").is_some_and(|v| v == "1");
    let cases: [(&str, &[&str]); 2] = [
        (
            "ellipsoid_margin",
            &[
                "--slices", "20", "--seed", "5", "--size", "128", "--margin", "240",
            ],
        ),
        (
            "two_foci_vessel",
            &[
                "--slices", "20", "--seed", "6", "--size", "128", "--foci", "2", "--gap", "120",
                "--vessel",
            ],
        ),
    ];
    if update {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("PLATFORM"), format!("{}\n", platform())).unwrap();
    }
    let made_on = std::fs::read_to_string(dir.join("PLATFORM"))
        .map(|s| s.trim().to_owned())
        .unwrap_or_default();
    for (name, args) in cases {
        let (measure, analysis) = golden_outputs(args);
        // Jaminan isi: fantom memang menguji hal yang dimaksud.
        let v: Value = serde_json::from_str(&analysis).unwrap();
        if name == "ellipsoid_margin" {
            assert!(v["analysis"]["margin"]["d_min"]["value_um"]
                .as_f64()
                .is_some());
        } else {
            assert_eq!(v["analysis"]["foci"]["n"], 2, "dua fokus");
            assert_eq!(v["analysis"]["contacts"][0]["result"]["any"], true);
        }
        for (suffix, got) in [("measure.json", &measure), ("analysis.json", &analysis)] {
            let path = dir.join(format!("{name}.{suffix}"));
            if update {
                std::fs::write(&path, got).unwrap();
                continue;
            }
            let want = std::fs::read_to_string(&path).unwrap_or_else(|e| {
                panic!(
                    "berkas emas {} tidak terbaca ({e}); buat dengan H3D_UPDATE_GOLDEN=1",
                    path.display()
                )
            });
            if made_on == platform() {
                assert!(
                    want == *got,
                    "{name}.{suffix} berbeda dari berkas emas — perilaku berubah. {}\n\
                     Bila disengaja: H3D_UPDATE_GOLDEN=1 cargo test -p h3d-cli golden",
                    first_diff(&want, got)
                );
            } else {
                // Platform lain: struktur sama, bilangan dalam 0,1 %.
                let a: Value = serde_json::from_str(&want).unwrap();
                let b: Value = serde_json::from_str(got).unwrap();
                if let Err(e) = json_close(&a, &b, 1e-3, name) {
                    panic!(
                        "{name}.{suffix} menyimpang dari berkas emas (dibuat di {made_on}, kini {}): {e}",
                        platform()
                    );
                }
            }
        }
    }
}

#[test]
fn json_close_tolerates_only_small_numeric_differences() {
    let a: Value = serde_json::json!({"x": [1.0, 2.0], "s": "a", "input_key": "aa"});
    let b: Value = serde_json::json!({"x": [1.0005, 2.0], "s": "a", "input_key": "bb"});
    assert!(json_close(&a, &b, 1e-3, "r").is_ok());
    assert!(json_close(&a, &b, 1e-5, "r").is_err());
    let c: Value = serde_json::json!({"x": [1.0, 2.0], "s": "b", "input_key": "aa"});
    assert!(json_close(&a, &c, 1e-3, "r").is_err());
    let d: Value = serde_json::json!({"x": [1.0], "s": "a", "input_key": "aa"});
    assert!(json_close(&a, &d, 1e-3, "r").is_err());
    assert!(first_diff("a\nb\n", "a\nc\n").contains("baris 2"));
}
