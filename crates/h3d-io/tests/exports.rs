//! Uji ekspor: GLB, STL, OBJ, NIfTI, laporan, analysis.json, cache voxel.

mod common;

use h3d_core::{ClinicalMeta, Colormap, LabelId, Sex, RESEARCH_ONLY_NOTICE};
use h3d_io::glb::GLB_NODATA;
use h3d_io::nifti::{DT_FLOAT32, DT_UINT8};
use h3d_io::{
    decode_mesh, decode_nifti, decode_stl, decode_vox, encode_f32_nifti, encode_glb,
    encode_label_nifti, encode_mesh, encode_obj, encode_stl, encode_vox, gunzip, parse_glb,
    render_csv, render_markdown, scalar_attribute_name, stl_file_name, ClinicalScope, GlbMesh,
    GlbScalar,
};
use h3d_volume::{MeshScalars, StackGrid};
use serde_json::Value;

fn component_count(ty: &str) -> usize {
    match ty {
        "SCALAR" => 1,
        "VEC3" => 3,
        "VEC4" => 4,
        other => panic!("tipe accessor tak dikenal: {other}"),
    }
}

fn component_size(ct: u64) -> usize {
    match ct {
        5121 => 1,
        5125 | 5126 => 4,
        other => panic!("componentType tak dikenal: {other}"),
    }
}

/// Validasi struktural glTF: semua accessor muat di bufferView-nya, semua
/// bufferView muat di buffer, rata 4 byte.
fn check_gltf_consistency(doc: &Value, bin: &[u8]) {
    assert_eq!(doc["asset"]["version"], "2.0");
    assert_eq!(
        doc["buffers"][0]["byteLength"].as_u64().unwrap() as usize,
        bin.len()
    );
    assert_eq!(bin.len() % 4, 0);
    let views = doc["bufferViews"].as_array().unwrap();
    for v in views {
        let off = v["byteOffset"].as_u64().unwrap() as usize;
        let len = v["byteLength"].as_u64().unwrap() as usize;
        assert_eq!(off % 4, 0, "bufferView harus rata 4 byte");
        assert!(off + len <= bin.len());
    }
    for a in doc["accessors"].as_array().unwrap() {
        let view = &views[a["bufferView"].as_u64().unwrap() as usize];
        let count = a["count"].as_u64().unwrap() as usize;
        let need = count
            * component_count(a["type"].as_str().unwrap())
            * component_size(a["componentType"].as_u64().unwrap());
        assert_eq!(
            need,
            view["byteLength"].as_u64().unwrap() as usize,
            "panjang accessor harus sama dengan bufferView-nya"
        );
    }
}

fn read_f32(bin: &[u8], doc: &Value, accessor: u64) -> Vec<f32> {
    let a = &doc["accessors"][accessor as usize];
    let v = &doc["bufferViews"][a["bufferView"].as_u64().unwrap() as usize];
    let off = v["byteOffset"].as_u64().unwrap() as usize;
    let len = v["byteLength"].as_u64().unwrap() as usize;
    bin[off..off + len]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_u32(bin: &[u8], doc: &Value, accessor: u64) -> Vec<u32> {
    let a = &doc["accessors"][accessor as usize];
    let v = &doc["bufferViews"][a["bufferView"].as_u64().unwrap() as usize];
    let off = v["byteOffset"].as_u64().unwrap() as usize;
    let len = v["byteLength"].as_u64().unwrap() as usize;
    bin[off..off + len]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn signed_volume(pos: &[[f64; 3]], tris: impl Iterator<Item = [usize; 3]>) -> f64 {
    let mut v = 0.0;
    for [a, b, c] in tris {
        let (a, b, c) = (pos[a], pos[b], pos[c]);
        v += a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]);
    }
    v / 6.0
}

#[test]
fn glb_parses_back_with_consistent_buffers() {
    let mut mesh = common::sphere_mesh();
    let n = mesh.positions_um.len();
    assert!(n > 100 && mesh.is_closed());
    // Skalar dengan satu NODATA.
    let mut values: Vec<f32> = (0..n).map(|i| i as f32).collect();
    values[5] = f32::NAN;
    mesh.set_scalars(MeshScalars {
        name: "margin distance".into(),
        unit: "µm".into(),
        values,
    });
    let meshes = [GlbMesh {
        name: "Tumor",
        color: [220, 40, 40, 255],
        mesh: &mesh,
    }];

    // Tanpa skalar: warna label sebagai material.
    let plain = encode_glb(&meshes, None).unwrap();
    assert_eq!(&plain[0..4], b"glTF");
    assert_eq!(
        u32::from_le_bytes([plain[4], plain[5], plain[6], plain[7]]),
        2
    );
    assert_eq!(
        u32::from_le_bytes([plain[8], plain[9], plain[10], plain[11]]) as usize,
        plain.len()
    );
    assert_eq!(plain.len() % 4, 0);
    let (doc, bin) = parse_glb(&plain).unwrap();
    check_gltf_consistency(&doc, &bin);
    let prim = &doc["meshes"][0]["primitives"][0];
    let pos_acc = prim["attributes"]["POSITION"].as_u64().unwrap();
    let idx_acc = prim["indices"].as_u64().unwrap();
    assert_eq!(
        doc["accessors"][pos_acc as usize]["count"]
            .as_u64()
            .unwrap() as usize,
        n
    );
    assert_eq!(
        doc["accessors"][idx_acc as usize]["count"]
            .as_u64()
            .unwrap() as usize,
        mesh.indices.len()
    );
    assert!(prim["attributes"].get("COLOR_0").is_none());
    let base = &doc["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"];
    assert!((base[0].as_f64().unwrap() - 220.0 / 255.0).abs() < 1e-9);

    // min/max accessor POSITION cocok dengan data, satuan meter.
    let pos = read_f32(&bin, &doc, pos_acc);
    let acc = &doc["accessors"][pos_acc as usize];
    for k in 0..3 {
        let lo = pos
            .iter()
            .skip(k)
            .step_by(3)
            .copied()
            .fold(f32::INFINITY, f32::min);
        let hi = pos
            .iter()
            .skip(k)
            .step_by(3)
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(acc["min"][k].as_f64().unwrap(), f64::from(lo));
        assert_eq!(acc["max"][k].as_f64().unwrap(), f64::from(hi));
    }
    let extent_m = acc["max"][0].as_f64().unwrap() - acc["min"][0].as_f64().unwrap();
    assert!(
        (extent_m - 2.0 * common::RADIUS_UM * 1e-6).abs() < 40e-6,
        "{extent_m}"
    );

    // Orientasi segitiga tetap menghadap keluar di kerangka glTF, dan volume
    // sama dengan mesh asal (µm³ → m³).
    let idx = read_u32(&bin, &doc, idx_acc);
    let p64: Vec<[f64; 3]> = pos
        .chunks_exact(3)
        .map(|c| [f64::from(c[0]), f64::from(c[1]), f64::from(c[2])])
        .collect();
    let vol_m3 = signed_volume(
        &p64,
        idx.chunks_exact(3)
            .map(|t| [t[0] as usize, t[1] as usize, t[2] as usize]),
    );
    let expect = mesh.signed_volume_um3() * 1e-18;
    assert!(vol_m3 > 0.0);
    assert!(
        ((vol_m3 - expect) / expect).abs() < 1e-3,
        "{vol_m3} vs {expect}"
    );

    // Dengan skalar: COLOR_0 + _H3D_<NAMA>, NODATA bukan NaN.
    let sc = GlbScalar {
        name: "margin distance",
        colormap: Colormap::Viridis,
        range: None,
    };
    let colored = encode_glb(&meshes, Some(&sc)).unwrap();
    let (doc, bin) = parse_glb(&colored).unwrap();
    check_gltf_consistency(&doc, &bin);
    let attrs = &doc["meshes"][0]["primitives"][0]["attributes"];
    let attr_name = scalar_attribute_name("margin distance");
    assert_eq!(attr_name, "_H3D_MARGIN_DISTANCE");
    let col = &doc["accessors"][attrs["COLOR_0"].as_u64().unwrap() as usize];
    assert_eq!(col["count"].as_u64().unwrap() as usize, n);
    assert_eq!(col["normalized"], true);
    let raw = read_f32(&bin, &doc, attrs[attr_name.as_str()].as_u64().unwrap());
    assert_eq!(raw.len(), n);
    assert!(raw.iter().all(|v| v.is_finite()));
    assert_eq!(raw[5], GLB_NODATA);
    assert_eq!(raw[7], 7.0);

    // Deterministik.
    assert_eq!(colored, encode_glb(&meshes, Some(&sc)).unwrap());
    // Berkas terpotong/rusak → galat.
    assert!(parse_glb(&colored[..colored.len() - 1]).is_err());
    assert!(parse_glb(b"bukan glb").is_err());
    // Tidak ada mesh berisi → galat, bukan berkas kosong.
    let empty = h3d_volume::LabelMesh::empty(LabelId(2));
    assert!(encode_glb(
        &[GlbMesh {
            name: "kosong",
            color: [0; 4],
            mesh: &empty
        }],
        None
    )
    .is_err());
}

#[test]
fn stl_triangle_count_and_volume() {
    let mesh = common::sphere_mesh();
    let bytes = encode_stl(&mesh).unwrap();
    assert_eq!(bytes.len(), 84 + 50 * mesh.triangle_count());
    assert!(!bytes.starts_with(b"solid"));
    assert!(std::str::from_utf8(&bytes[..80])
        .unwrap()
        .contains("units=mm"));
    let tris = decode_stl(&bytes).unwrap();
    assert_eq!(tris.len(), mesh.triangle_count());
    let mut pos: Vec<[f64; 3]> = Vec::new();
    for t in &tris {
        for v in t {
            pos.push([f64::from(v[0]), f64::from(v[1]), f64::from(v[2])]);
        }
    }
    let vol_mm3 = signed_volume(&pos, (0..tris.len()).map(|i| [3 * i, 3 * i + 1, 3 * i + 2]));
    let expect = mesh.signed_volume_um3() / 1e9;
    assert!(vol_mm3 > 0.0, "orientasi keluar harus terjaga");
    assert!(
        ((vol_mm3 - expect) / expect).abs() < 1e-3,
        "{vol_mm3} vs {expect}"
    );
    let analytic = 4.0 / 3.0 * std::f64::consts::PI * common::RADIUS_UM.powi(3) / 1e9;
    assert!(
        ((vol_mm3 - analytic) / analytic).abs() < 0.03,
        "{vol_mm3} vs {analytic}"
    );
    assert_eq!(stl_file_name("Tumor invasif", 1), "tumor_invasif_mm.stl");
    assert!(decode_stl(&bytes[..bytes.len() - 7]).is_err());
}

#[test]
fn obj_has_one_group_per_label_in_mm() {
    let mesh = common::sphere_mesh();
    let mut second = mesh.clone();
    second.label = LabelId(2);
    let text = encode_obj(&[("Tumor", &mesh), ("Kapsul", &second)]).unwrap();
    assert!(text.contains("# units: mm"));
    assert_eq!(text.lines().filter(|l| l.starts_with("g ")).count(), 2);
    assert!(text.contains("g tumor\n") && text.contains("g kapsul\n"));
    let n = mesh.positions_um.len();
    assert_eq!(text.lines().filter(|l| l.starts_with("v ")).count(), 2 * n);
    assert_eq!(
        text.lines().filter(|l| l.starts_with("f ")).count(),
        2 * mesh.triangle_count()
    );
    // Indeks grup kedua digeser sebanyak verteks grup pertama.
    let max_index = text
        .lines()
        .filter(|l| l.starts_with("f "))
        .flat_map(|l| l[2..].split(' '))
        .map(|tok| tok.split("//").next().unwrap().parse::<usize>().unwrap())
        .max()
        .unwrap();
    assert_eq!(max_index, 2 * n);
    // Rentang x ≈ diameter bola dalam mm.
    let xs: Vec<f64> = text
        .lines()
        .filter(|l| l.starts_with("v "))
        .map(|l| l.split(' ').nth(1).unwrap().parse().unwrap())
        .collect();
    let ext =
        xs.iter().cloned().fold(f64::MIN, f64::max) - xs.iter().cloned().fold(f64::MAX, f64::min);
    assert!((ext - 0.4).abs() < 0.04, "{ext}");
}

#[test]
fn nifti_header_and_voxel_sum() {
    let grid = common::sphere_grid();
    let mask = common::sphere_mask(&grid);
    let out = encode_label_nifti(&[(3, &mask)]).unwrap();
    assert!(out.notes.is_empty(), "kisi seragam tidak butuh catatan");
    assert_eq!(&out.gz[..2], &[0x1f, 0x8b]);
    let raw = gunzip(&out.gz).unwrap();
    let (h, data) = decode_nifti(&raw).unwrap();
    assert_eq!(h.dim, [3, 40, 40, 40, 1, 1, 1, 1]);
    assert_eq!(h.datatype, DT_UINT8);
    assert_eq!(h.bitpix, 8);
    assert_eq!(h.xyzt_units, 2);
    assert!((h.pixdim[1] - 0.016).abs() < 1e-7);
    assert!((h.pixdim[2] - 0.016).abs() < 1e-7);
    assert!((h.pixdim[3] - 0.016).abs() < 1e-7);
    assert_eq!(h.vox_offset, 352.0);
    assert_eq!(h.sform_code, 1);
    assert!(h.srow[1][1] < 0.0, "sumbu y dibalik agar tangan-kanan");
    let sum: u64 = data.iter().map(|&v| u64::from(v)).sum();
    assert_eq!(
        sum,
        3 * mask.count() as u64,
        "jumlah voxel harus sama dengan masker"
    );
    assert!(data.iter().all(|&v| v == 0 || v == 3));

    // Verifikasi independen dengan crate `nifti`.
    let hdr = nifti::NiftiHeader::from_reader(&raw[..]).expect("header NIfTI sah");
    assert_eq!(hdr.dim, [3, 40, 40, 40, 1, 1, 1, 1]);
    assert_eq!(hdr.datatype, 2);
    assert_eq!(hdr.bitpix, 8);
    assert!((hdr.pixdim[3] - 0.016).abs() < 1e-7);

    // Volume f32 dengan NODATA.
    let mut values = vec![1.5f32; grid.voxel_count()];
    values[0] = f32::NAN;
    let out = encode_f32_nifti(&grid, &values).unwrap();
    let raw = gunzip(&out.gz).unwrap();
    let (h, data) = decode_nifti(&raw).unwrap();
    assert_eq!(h.datatype, DT_FLOAT32);
    assert_eq!(h.bitpix, 32);
    assert_eq!(data.len(), 4 * grid.voxel_count());
    let first = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    let second = f32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    assert!(first.is_nan() && second == 1.5);

    // Jarak lapis tidak seragam → diresampel ke median, dengan catatan.
    let uneven = StackGrid {
        origin_um: [0.0, 0.0],
        s_xy: 16.0,
        size: [4, 4],
        z_um: vec![0.0, 20.0, 40.0, 100.0],
    };
    let mut m = h3d_volume::VoxelMask::empty(uneven.clone());
    m.data.iter_mut().for_each(|v| *v = 1);
    let out = encode_label_nifti(&[(1, &m)]).unwrap();
    assert_eq!(out.notes.len(), 1);
    assert!(out.notes[0].contains("median"));
    let raw = gunzip(&out.gz).unwrap();
    let (h, _) = decode_nifti(&raw).unwrap();
    assert!((h.pixdim[3] - 0.020).abs() < 1e-7);
    assert_eq!(h.dim[3], 6, "0..100 µm pada 20 µm = 6 lapis");

    // Masukan tak sah → galat.
    assert!(encode_label_nifti(&[]).is_err());
    assert!(encode_f32_nifti(&grid, &values[..10]).is_err());
    assert!(decode_nifti(&raw[..100]).is_err());
    // Deterministik (gzip tanpa cap waktu).
    assert_eq!(
        encode_label_nifti(&[(3, &mask)]).unwrap().gz,
        encode_label_nifti(&[(3, &mask)]).unwrap().gz
    );
}

#[test]
fn report_has_notice_sections_and_no_clinical_without_flag() {
    let r = common::sample_report();
    let c = ClinicalMeta {
        subject_pseudo_id: "PSEUDO-KUNCI-771".into(),
        age_years: Some(63),
        sex: Some(Sex::Female),
        topography_icdo: Some("C50.4-TOPO".into()),
        morphology_icdo: Some("8500/3-MORFO".into()),
        specimen_month: Some("2025-11".into()),
        ..Default::default()
    };
    let md = render_markdown(&r, ClinicalScope::Omit);
    let csv = render_csv(&r, ClinicalScope::Omit);
    for text in [&md, &csv] {
        assert!(text.contains(RESEARCH_ONLY_NOTICE));
        for marker in [
            "PSEUDO-KUNCI-771",
            "C50.4-TOPO",
            "8500/3-MORFO",
            "2025-11",
            "perempuan",
        ] {
            assert!(!text.contains(marker), "bocor: {marker}");
        }
        for field in ClinicalMeta::FIELD_NAMES {
            assert!(!text.contains(field), "nama field klinis bocor: {field}");
        }
    }
    // Bagian 1–8 muncul berurutan.
    let mut last = 0;
    for s in h3d_io::strings::SECTIONS {
        let at = md
            .find(&format!("## {s}"))
            .unwrap_or_else(|| panic!("bagian hilang: {s}"));
        assert!(at > last, "urutan bagian salah pada {s}");
        last = at;
    }
    // Ukuran 3D dicetak dengan ketidakpastian.
    assert!(md.contains("0.400 ± 0.024 mm"), "{md}");
    assert!(csv.contains("2,diameter_maks,Tumor,0.400,0.024,mm"));
    // Tidak ada teks anjuran.
    for banned in ["disarankan", "sebaiknya", "perlu reseksi", "rekomendasi"] {
        assert!(
            !md.to_lowercase().contains(banned),
            "teks anjuran: {banned}"
        );
    }

    // Cakupan ringkas: hanya pseudo-id, organ, diagnosis.
    let brief = render_markdown(&r, ClinicalScope::Brief(&c));
    assert!(brief.contains("PSEUDO-KUNCI-771") && brief.contains("C50.4-TOPO"));
    assert!(brief.contains("8500/3-MORFO"));
    assert!(!brief.contains("perempuan") && !brief.contains("2025-11") && !brief.contains("63"));
    // Cakupan penuh.
    let full = render_markdown(&r, ClinicalScope::Full(&c));
    assert!(full.contains("perempuan") && full.contains("2025-11") && full.contains("63"));
    let full_csv = render_csv(&r, ClinicalScope::Full(&c));
    assert!(full_csv.contains("PSEUDO-KUNCI-771") && full_csv.contains("perempuan"));
}

#[test]
fn vox_and_mesh_cache_roundtrip_and_corruption() {
    let grid = common::sphere_grid();
    let mut mask = common::sphere_mask(&grid);
    mask.annotated_layer[3] = Some((h3d_core::SlideId(9), true));
    let bytes = encode_vox(&mask).unwrap();
    let p = std::path::Path::new("x.vox");
    assert_eq!(decode_vox(&bytes, p).unwrap(), mask);
    assert!(decode_vox(&bytes[..bytes.len() / 2], p).is_err());
    assert!(decode_vox(b"H3DV", p).is_err());

    let mut mesh = common::sphere_mesh();
    mesh.set_scalars(MeshScalars {
        name: "d".into(),
        unit: "µm".into(),
        values: vec![2.5; mesh.positions_um.len()],
    });
    let bytes = encode_mesh(&mesh).unwrap();
    let back = decode_mesh(&bytes, p).unwrap();
    assert_eq!(back, mesh);
    assert!(decode_mesh(&bytes[..bytes.len() - 5], p).is_err());
    assert!(decode_mesh(&encode_vox(&mask).unwrap(), p).is_err());
}
