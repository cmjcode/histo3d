//! Tes properti (aturan keras 17, H10a): isi `ClinicalMeta` tidak pernah
//! muncul di `project.json` maupun di ekspor apa pun yang dibuat tanpa
//! `with_clinical`.

mod common;

use h3d_core::{
    ClinicalMeta, Grade, LabelId, Laterality, MolecularResult, Procedure, Ptnm, Sex, Therapy,
};
use h3d_io::{
    encode_f32_nifti, encode_glb, encode_label_nifti, encode_obj, encode_project, encode_stl,
    export_slide_geojson, gunzip, parse_glb, render_csv, render_markdown, to_stable_json_of,
    ClinicalScope, GlbMesh, ProjectBundle,
};
use h3d_segment::{LabelStore, PaintMode};
use proptest::prelude::*;

/// Penanda khas: awalan tetap + 12 karakter acak, mustahil muncul kebetulan.
fn marker(tag: &'static str) -> impl Strategy<Value = String> {
    "[A-Z0-9]{12}".prop_map(move |s| format!("ZQ{tag}{s}"))
}

fn arb_clinical() -> impl Strategy<Value = ClinicalMeta> {
    (
        (
            marker("ID"),
            proptest::option::of(0u8..120),
            proptest::option::of(prop_oneof![
                Just(Sex::Female),
                Just(Sex::Male),
                Just(Sex::Other),
                Just(Sex::Unknown)
            ]),
            proptest::option::of(marker("TOPO")),
            proptest::option::of(marker("MORF")),
            proptest::option::of(prop_oneof![
                Just(Laterality::Left),
                Just(Laterality::Right),
                Just(Laterality::Bilateral),
                Just(Laterality::NotApplicable)
            ]),
        ),
        (
            proptest::option::of(prop_oneof![
                Just(Procedure::Biopsy),
                Just(Procedure::Excision),
                Just(Procedure::Resection),
                marker("PROC").prop_map(Procedure::Other)
            ]),
            proptest::option::of((marker("GSYS"), marker("GVAL"))),
            proptest::option::of((
                marker("T"),
                marker("N"),
                marker("M"),
                0u8..12,
                marker("PFX"),
            )),
            proptest::option::of((any::<bool>(), proptest::option::of(marker("REG")))),
            proptest::collection::vec(
                (
                    marker("GENE"),
                    marker("RES"),
                    proptest::option::of(marker("METH")),
                ),
                0..3,
            ),
            proptest::option::of(marker("MONTH")),
        ),
    )
        .prop_map(
            |((id, age, sex, topo, morf, lat), (proc_, grade, ptnm, neo, mol, month))| {
                ClinicalMeta {
                    subject_pseudo_id: id,
                    age_years: age,
                    sex,
                    topography_icdo: topo,
                    morphology_icdo: morf,
                    laterality: lat,
                    procedure: proc_,
                    grade: grade.map(|(system, value)| Grade { system, value }),
                    ptnm: ptnm.map(|(t, n, m, edition, prefix)| Ptnm {
                        t,
                        n,
                        m,
                        edition,
                        prefix,
                    }),
                    neoadjuvant: neo.map(|(given, regimen)| Therapy { given, regimen }),
                    molecular: mol
                        .into_iter()
                        .map(|(gene, result, method)| MolecularResult {
                            gene,
                            result,
                            method,
                        })
                        .collect(),
                    specimen_month: month,
                }
            },
        )
}

/// Semua string penanda di dalam `ClinicalMeta`.
fn markers_of(c: &ClinicalMeta) -> Vec<String> {
    let mut v = vec![c.subject_pseudo_id.clone()];
    v.extend(c.topography_icdo.clone());
    v.extend(c.morphology_icdo.clone());
    if let Some(Procedure::Other(s)) = &c.procedure {
        v.push(s.clone());
    }
    if let Some(g) = &c.grade {
        v.push(g.system.clone());
        v.push(g.value.clone());
    }
    if let Some(p) = &c.ptnm {
        v.extend([p.t.clone(), p.n.clone(), p.m.clone(), p.prefix.clone()]);
    }
    if let Some(n) = &c.neoadjuvant {
        v.extend(n.regimen.clone());
    }
    for m in &c.molecular {
        v.push(m.gene.clone());
        v.push(m.result.clone());
        v.extend(m.method.clone());
    }
    v.extend(c.specimen_month.clone());
    v
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && haystack.windows(needle.len()).any(|w| w == needle)
}

/// Ekspor yang tidak bergantung pada data klinis dibuat sekali saja.
struct FixedExports {
    /// `(nama, byte lengkap untuk pencarian penanda, bagian teks/header untuk
    /// pencarian nama field)`.
    items: Vec<(&'static str, Vec<u8>, Vec<u8>)>,
}

fn fixed_exports() -> &'static FixedExports {
    static CELL: std::sync::OnceLock<FixedExports> = std::sync::OnceLock::new();
    CELL.get_or_init(build_fixed_exports)
}

fn build_fixed_exports() -> FixedExports {
    let mesh = common::sphere_mesh();
    let grid = common::sphere_grid();
    let mask = common::sphere_mask(&grid);
    let mut items = Vec::new();

    let glb = encode_glb(
        &[GlbMesh {
            name: "Tumor",
            color: [220, 40, 40, 255],
            mesh: &mesh,
        }],
        None,
    )
    .unwrap();
    let (doc, _) = parse_glb(&glb).unwrap();
    items.push(("glb", glb, serde_json::to_vec(&doc).unwrap()));

    let stl = encode_stl(&mesh).unwrap();
    let header = stl[..80].to_vec();
    items.push(("stl", stl, header));

    let obj = encode_obj(&[("Tumor", &mesh)]).unwrap().into_bytes();
    items.push(("obj", obj.clone(), obj));

    let nii = gunzip(&encode_label_nifti(&[(1, &mask)]).unwrap().gz).unwrap();
    let header = nii[..352].to_vec();
    items.push(("nifti-label", nii, header));
    let vals = vec![0.5f32; grid.voxel_count()];
    let nii = gunzip(&encode_f32_nifti(&grid, &vals).unwrap().gz).unwrap();
    let header = nii[..352].to_vec();
    items.push(("nifti-f32", nii, header));

    let mut store = LabelStore::new(4.0, [2000.0, 2000.0]);
    store.paint_disc([900.0, 900.0], 300.0, LabelId(1), PaintMode::Replace);
    let gj = export_slide_geojson(
        &store,
        &[(LabelId(1), "Tumor")],
        [0.5, 0.5],
        &vec![(LabelId(1), vec![("Ki-67 % positif".to_owned(), 12.5)])],
    )
    .into_bytes();
    items.push(("geojson", gj.clone(), gj));
    FixedExports { items }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, .. ProptestConfig::default() })]

    #[test]
    fn clinical_never_leaks_into_project_json_or_exports(c in arb_clinical()) {
        let markers = markers_of(&c);

        // project.json: lewat serialisasi langsung dan lewat bundel di disk.
        let mut project = common::sample_project();
        project.sample.clinical = Some(c.clone());
        let tmp = tempfile::tempdir().unwrap();
        let bundle = ProjectBundle::create(&tmp.path().join("p.h3d")).unwrap();
        bundle.save_project(&project).unwrap();
        bundle.save_clinical(Some(&c)).unwrap();
        let on_disk = std::fs::read(bundle.project_json()).unwrap();
        prop_assert_eq!(&on_disk, &encode_project(&project).unwrap());

        // Laporan tanpa flag, dan JSON laporan (dipakai `measure --json`).
        let report = common::sample_report();
        let md = render_markdown(&report, ClinicalScope::Omit).into_bytes();
        let csv = render_csv(&report, ClinicalScope::Omit).into_bytes();
        let json = to_stable_json_of(&report).unwrap().into_bytes();

        let mut all: Vec<(&str, Vec<u8>, Vec<u8>)> = vec![
            ("project.json", on_disk.clone(), on_disk),
            ("report.md", md.clone(), md),
            ("report.csv", csv.clone(), csv),
            ("report.json", json.clone(), json),
        ];
        all.extend(fixed_exports().items.iter().cloned());

        for (name, bytes, text) in &all {
            for m in &markers {
                prop_assert!(!contains(bytes, m.as_bytes()), "penanda {} bocor ke {}", m, name);
            }
            for f in ClinicalMeta::FIELD_NAMES {
                prop_assert!(!contains(text, f.as_bytes()), "nama field {} muncul di {}", f, name);
            }
        }

        // Kendali positif: dengan flag, penanda memang tercetak; data klinis
        // tetap bisa dimuat kembali utuh dari clinical.json.
        let full = render_markdown(&report, ClinicalScope::Full(&c));
        prop_assert!(full.contains(&c.subject_pseudo_id));
        let loaded = bundle.load_project().unwrap();
        prop_assert_eq!(loaded.sample.clinical, Some(c));
    }
}
