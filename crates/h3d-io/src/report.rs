//! Laporan Markdown + CSV (8.7.3, 8.12.8).
//!
//! Struktur data [`Report`] diisi `h3d-engine` (`Session::measure`); modul
//! ini hanya merendernya. Bagian laporan berurutan tetap 1–8. Tiap jarak
//! atau ukuran 3D dicetak sebagai `nilai ± ketidakpastian satuan`; arah
//! dicetak sebagai sumbu mentah (+X) di samping nama anatomisnya. Laporan
//! selalu memuat [`h3d_core::RESEARCH_ONLY_NOTICE`] dan tidak pernah memuat
//! teks rekomendasi tindakan.
//!
//! Privasi (aturan keras 17): data klinis hanya dicetak sesuai
//! [`ClinicalScope`]; ekspor tanpa `--with-clinical` memakai
//! [`ClinicalScope::Omit`] sehingga tidak ada satu pun isi `ClinicalMeta`
//! yang keluar.

use std::fmt::Write as _;

use h3d_core::{
    ClinicalMeta, LabelId, LabelKind, Laterality, Procedure, ReconConfig, Sex, SignedAxis, SlideId,
    SpatialConfig, SpecimenOrientation, RESEARCH_ONLY_NOTICE,
};
use h3d_volume::{SpatialAnalysis, Uncertainty, Warning};
use serde::{Deserialize, Serialize};

use crate::strings as t;

/// Seberapa banyak konteks klinis yang boleh dicetak.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClinicalScope<'a> {
    /// Tidak ada data klinis sama sekali (bawaan semua ekspor).
    Omit,
    /// Hanya id pseudonim, organ/topografi, dan diagnosis (8.12.9); untuk
    /// tampilan di dalam aplikasi, bukan ekspor berkas.
    Brief(&'a ClinicalMeta),
    /// Semua field (`--with-clinical`).
    Full(&'a ClinicalMeta),
}

/// Ringkasan irisan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct SlideSummary {
    /// Jumlah irisan.
    pub total: usize,
    /// Irisan `Ok`.
    pub usable: usize,
    /// Irisan terdaftar (punya `T_i` atau acuan).
    pub registered: usize,
    /// Irisan gagal QC.
    pub flagged: Vec<SlideId>,
    /// Irisan acuan.
    pub reference: Option<SlideId>,
    /// Jarak irisan median, µm.
    pub median_spacing_um: Option<f64>,
    /// Urutan berasal dari usulan otomatis yang diterima (8.4.7).
    pub order_suggested: bool,
}

/// Pengukuran satu label (8.6.4).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LabelReport {
    /// Label.
    pub label: LabelId,
    /// Nama.
    pub name: String,
    /// Jenis.
    pub kind: LabelKind,
    /// Grup.
    pub group: String,
    /// Volume Cavalieri dari irisan beranotasi non-interpolasi, µm³.
    pub volume_cavalieri_um3: f64,
    /// Volume Cavalieri dari irisan `Reviewed` saja; `None` bila < 2 irisan.
    pub volume_reviewed_um3: Option<f64>,
    /// `V × f³` bila faktor susut diisi (tidak diterapkan ke geometri).
    pub volume_corrected_um3: Option<f64>,
    /// Volume mesh bertanda, µm³.
    pub volume_mesh_um3: Option<f64>,
    /// Luas permukaan mesh, µm².
    pub surface_area_um2: Option<f64>,
    /// Kotak pembatas mesh `(min, max)`, µm.
    pub bbox_um: Option<([f64; 3], [f64; 3])>,
    /// Diameter maksimum mesh, µm.
    pub max_diameter_um: Option<f64>,
    /// Jumlah komponen mesh.
    pub n_components: Option<usize>,
    /// Irisan beranotasi (bukan interpolasi).
    pub n_slices_annotated: usize,
    /// Irisan hasil interpolasi.
    pub n_slices_interpolated: usize,
    /// Irisan berstatus `Reviewed`.
    pub n_slices_reviewed: usize,
}

/// Anggota grup.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GroupMember {
    /// Nama label.
    pub name: String,
    /// Volume Cavalieri, µm³.
    pub volume_um3: f64,
    /// Volume anggota / volume grup.
    pub fraction: f64,
}

/// Agregasi per grup (8.12.4).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GroupReport {
    /// Nama grup.
    pub group: String,
    /// Jumlah volume anggota, µm³.
    pub volume_um3: f64,
    /// Anggota.
    pub members: Vec<GroupMember>,
}

/// Ringkasan provenance (8.12.7).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct ProvenanceReport {
    /// Jumlah catatan (irisan × label).
    pub total: usize,
    /// `(sumber, jumlah)` dalam urutan tetap.
    pub by_source: Vec<(String, usize)>,
    /// Jumlah catatan `Reviewed`.
    pub reviewed: usize,
    /// Catatan `Draft`: `(irisan, nama label)`.
    pub drafts: Vec<(SlideId, String)>,
}

/// Baris atribut per irisan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttrSlideRow {
    /// Irisan.
    pub slide: SlideId,
    /// Rata-rata berbobot.
    pub mean: f64,
    /// Jumlah bobot.
    pub weight: f64,
    /// Rata-rata di dalam masker tumor.
    pub intratumoral: Option<f64>,
    /// Rata-rata di pita peritumoral.
    pub peritumoral: Option<f64>,
}

/// Atribut dalam laporan (8.12.5).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttrReport {
    /// Nama.
    pub name: String,
    /// Satuan.
    pub unit: String,
    /// Metode.
    pub method: String,
    /// Hash parameter (aturan keras 19).
    pub params_hash: Option<u64>,
    /// Tabel per irisan.
    pub per_slide: Vec<AttrSlideRow>,
    /// Rentang rata-rata antar irisan (maks − min).
    pub range_between_slides: Option<f64>,
}

/// Ringkasan kisi rekonstruksi.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GridInfo {
    /// `s_xy`, µm.
    pub s_xy_um: f64,
    /// `s_z` median, µm.
    pub s_z_um: f64,
    /// `[nx, ny]`.
    pub size: [u32; 2],
    /// Jumlah lapis.
    pub nz: usize,
}

/// Gambar laporan (8.12.8 butir 3): irisan terdekat ke titik tumor–tepi
/// terdekat, dengan garis luar label, kedua titik, dan panah di antaranya.
/// Angka ada di sini (dan di keterangan Markdown), bukan di dalam gambar.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReportFigure {
    /// Path PNG relatif terhadap `report.md`, mis. `figures/margin_tepi_reseksi.png`.
    pub file: String,
    /// Nama label tepi reseksi yang memuat titik tepi terdekat.
    pub label: String,
    /// Irisan yang digambar (terdekat pada sumbu z ke titik tumor terdekat).
    pub slide: SlideId,
    /// Nama berkas irisan itu.
    pub slide_file: String,
    /// `z` irisan itu, µm.
    pub slide_z_um: f64,
    /// Ukuran gambar `[lebar, tinggi]`, piksel.
    pub size_px: [u32; 2],
    /// µm ruang irisan per piksel gambar `[x, y]`.
    pub um_per_px: [f64; 2],
    /// Titik tumor terdekat di gambar, piksel (kontinu; ruang irisan itu).
    pub tumor_px: [f64; 2],
    /// Titik tepi terdekat di gambar, piksel.
    pub margin_px: [f64; 2],
    /// Jarak kedua titik pada bidang irisan, µm.
    pub in_plane_um: f64,
    /// Selisih `z` kedua titik (tepi − tumor), µm.
    pub dz_um: f64,
}

/// Laporan lengkap. Tidak memuat data klinis.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Report {
    /// Versi aplikasi.
    pub app_version: String,
    /// Kalimat "Hanya untuk riset".
    pub notice: String,
    /// Nama sampel.
    pub sample_name: String,
    /// Catatan sampel.
    pub sample_notes: String,
    /// Orientasi spesimen.
    pub orientation: Option<SpecimenOrientation>,
    /// Faktor susut linear.
    pub shrinkage_linear: Option<f64>,
    /// Ringkasan irisan.
    pub slides: SlideSummary,
    /// Pengukuran per label.
    pub labels: Vec<LabelReport>,
    /// Agregasi per grup.
    pub groups: Vec<GroupReport>,
    /// Provenance.
    pub provenance: ProvenanceReport,
    /// Atribut.
    pub attrs: Vec<AttrReport>,
    /// Analisis spasial (bila sudah dijalankan untuk keadaan saat ini).
    pub analysis: Option<SpatialAnalysis>,
    /// Gambar margin yang sudah ditulis di samping laporan (kosong sampai
    /// ekspor laporan menulisnya).
    #[serde(default)]
    pub figures: Vec<ReportFigure>,
    /// Parameter rekonstruksi.
    pub recon: ReconConfig,
    /// Parameter analisis spasial.
    pub spatial: SpatialConfig,
    /// Kisi rekonstruksi.
    pub grid: Option<GridInfo>,
    /// `± (½ voxel terbesar + taksiran TRE)` untuk ukuran turunan mesh, µm.
    pub dimension_uncertainty_um: f64,
    /// Catatan.
    pub notes: Vec<String>,
    /// Peringatan.
    pub warnings: Vec<Warning>,
}

impl Report {
    /// Laporan kosong dengan nama sampel dan versi aplikasi.
    pub fn new(sample_name: &str, app_version: &str) -> Self {
        Report {
            app_version: app_version.to_owned(),
            notice: RESEARCH_ONLY_NOTICE.to_owned(),
            sample_name: sample_name.to_owned(),
            sample_notes: String::new(),
            orientation: None,
            shrinkage_linear: None,
            slides: SlideSummary::default(),
            labels: Vec::new(),
            groups: Vec::new(),
            provenance: ProvenanceReport::default(),
            attrs: Vec::new(),
            analysis: None,
            figures: Vec::new(),
            recon: ReconConfig::default(),
            spatial: SpatialConfig::default(),
            grid: None,
            dimension_uncertainty_um: 0.0,
            notes: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Semua peringatan (laporan + analisis), tanpa duplikat, urutan tetap.
    pub fn all_warnings(&self) -> Vec<Warning> {
        let mut out: Vec<Warning> = Vec::new();
        let analysis = self.analysis.iter().flat_map(|a| a.warnings.iter());
        for w in self.warnings.iter().chain(analysis) {
            if !out.contains(w) {
                out.push(w.clone());
            }
        }
        out
    }

    /// Semua catatan (laporan + analisis), tanpa duplikat.
    pub fn all_notes(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        let analysis = self.analysis.iter().flat_map(|a| a.notes.iter());
        for n in self.notes.iter().chain(analysis) {
            if !out.contains(n) {
                out.push(n.clone());
            }
        }
        out
    }
}

// ---------------------------------------------------------------- format

/// µm → mm, tiga desimal.
pub fn fmt_mm(um: f64) -> String {
    format!("{:.3}", um / 1000.0)
}

/// µm³ → mm³, empat desimal.
pub fn fmt_mm3(um3: f64) -> String {
    format!("{:.4}", um3 / 1.0e9)
}

/// µm² → mm², empat desimal.
pub fn fmt_mm2(um2: f64) -> String {
    format!("{:.4}", um2 / 1.0e6)
}

/// Jarak beserta ketidakpastiannya: `nilai ± u mm`.
pub fn fmt_distance(value_um: f64, plus_minus_um: f64) -> String {
    format!("{} ± {} mm", fmt_mm(value_um), fmt_mm(plus_minus_um))
}

/// Jarak [`Uncertainty`]; menandai bila dibatasi resolusi Z.
pub fn fmt_uncertainty(u: &Uncertainty) -> String {
    let mut s = fmt_distance(u.value_um, u.plus_minus_um);
    if u.z_limited {
        let _ = write!(s, " ({})", t::Z_LIMITED);
    }
    s
}

fn fmt_point_mm(p: [f64; 3]) -> String {
    format!("({}, {}, {})", fmt_mm(p[0]), fmt_mm(p[1]), fmt_mm(p[2]))
}

fn opt<T>(v: Option<T>, f: impl Fn(T) -> String) -> String {
    v.map(f).unwrap_or_else(|| t::NOT_AVAILABLE.to_owned())
}

/// Sel tabel Markdown: `|` dan baris baru dinetralkan.
fn cell(s: &str) -> String {
    s.replace('|', "\\|").replace(['\n', '\r'], " ")
}

fn md_table(out: &mut String, header: &[&str], rows: &[Vec<String>]) {
    let _ = writeln!(
        out,
        "| {} |",
        header
            .iter()
            .map(|h| cell(h))
            .collect::<Vec<_>>()
            .join(" | ")
    );
    let _ = writeln!(
        out,
        "|{}|",
        header.iter().map(|_| " --- ").collect::<Vec<_>>().join("|")
    );
    for r in rows {
        let _ = writeln!(
            out,
            "| {} |",
            r.iter().map(|c| cell(c)).collect::<Vec<_>>().join(" | ")
        );
    }
    out.push('\n');
}

fn yes_no(b: bool) -> &'static str {
    if b {
        t::YES
    } else {
        t::NO
    }
}

fn orientation_rows(o: &SpecimenOrientation) -> Vec<(SignedAxis, String)> {
    SignedAxis::ALL
        .iter()
        .map(|a| (*a, t::direction_text(*a, o.name_of(*a))))
        .collect()
}

/// Baris konteks klinis `(nama field, nilai)` sesuai cakupan.
fn clinical_rows(scope: ClinicalScope<'_>) -> Vec<(&'static str, String)> {
    let (c, full) = match scope {
        ClinicalScope::Omit => return Vec::new(),
        ClinicalScope::Brief(c) => (c, false),
        ClinicalScope::Full(c) => (c, true),
    };
    let mut rows: Vec<(&'static str, String)> = Vec::new();
    rows.push((t::CLIN_PSEUDO_ID, c.subject_pseudo_id.clone()));
    if let Some(v) = &c.topography_icdo {
        rows.push((t::CLIN_TOPOGRAPHY, v.clone()));
    }
    if let Some(v) = &c.morphology_icdo {
        rows.push((t::CLIN_MORPHOLOGY, v.clone()));
    }
    if !full {
        return rows;
    }
    if let Some(v) = c.age_years {
        rows.push((t::CLIN_AGE, v.to_string()));
    }
    if let Some(v) = c.sex {
        let s = match v {
            Sex::Female => "perempuan",
            Sex::Male => "laki-laki",
            Sex::Other => "lainnya",
            Sex::Unknown => "tidak diketahui",
        };
        rows.push((t::CLIN_GENDER, s.to_owned()));
    }
    if let Some(v) = c.laterality {
        let s = match v {
            Laterality::Left => "kiri",
            Laterality::Right => "kanan",
            Laterality::Bilateral => "bilateral",
            Laterality::NotApplicable => "tidak berlaku",
        };
        rows.push((t::CLIN_SIDE, s.to_owned()));
    }
    if let Some(v) = &c.procedure {
        let s = match v {
            Procedure::Biopsy => "biopsi".to_owned(),
            Procedure::Excision => "eksisi".to_owned(),
            Procedure::Resection => "reseksi".to_owned(),
            Procedure::Other(o) => o.clone(),
        };
        rows.push((t::CLIN_SPECIMEN_TYPE, s));
    }
    if let Some(g) = &c.grade {
        rows.push((t::CLIN_DEGREE, format!("{} {}", g.system, g.value)));
    }
    if let Some(p) = &c.ptnm {
        rows.push((
            t::CLIN_STAGE,
            format!(
                "{}T{} N{} M{} (AJCC {})",
                p.prefix, p.t, p.n, p.m, p.edition
            ),
        ));
    }
    if let Some(n) = &c.neoadjuvant {
        let mut s = yes_no(n.given).to_owned();
        if let Some(r) = &n.regimen {
            let _ = write!(s, " ({r})");
        }
        rows.push((t::CLIN_PRIOR_THERAPY, s));
    }
    if !c.molecular.is_empty() {
        let s = c
            .molecular
            .iter()
            .map(|m| match &m.method {
                Some(me) => format!("{} {} ({})", m.gene, m.result, me),
                None => format!("{} {}", m.gene, m.result),
            })
            .collect::<Vec<_>>()
            .join("; ");
        rows.push((t::CLIN_GENE_RESULTS, s));
    }
    if let Some(v) = &c.specimen_month {
        rows.push((t::CLIN_MONTH, v.clone()));
    }
    rows
}

// -------------------------------------------------------------- markdown

/// Render laporan Markdown (bagian 1–8 berurutan tetap).
pub fn render_markdown(r: &Report, clinical: ClinicalScope<'_>) -> String {
    let mut o = String::new();
    let _ = writeln!(o, "# {} — {}\n", t::TITLE, cell(&r.sample_name));
    let _ = writeln!(o, "> **{RESEARCH_ONLY_NOTICE}**\n");

    // 1. Konteks sampel dan orientasi.
    let _ = writeln!(o, "## {}\n", t::SECTIONS[0]);
    let _ = writeln!(o, "- {}: {}", t::SAMPLE_NAME, cell(&r.sample_name));
    if !r.sample_notes.is_empty() {
        let _ = writeln!(o, "- {}: {}", t::SAMPLE_NOTES, cell(&r.sample_notes));
    }
    let rows = clinical_rows(clinical);
    if rows.is_empty() {
        let _ = writeln!(o, "- {}", t::CLINICAL_OMITTED);
    }
    for (k, v) in rows {
        let _ = writeln!(o, "- {}: {}", k, cell(&v));
    }
    match &r.orientation {
        Some(or) => {
            let list = orientation_rows(or)
                .into_iter()
                .map(|(_, s)| s)
                .collect::<Vec<_>>()
                .join(", ");
            let _ = writeln!(o, "- {}: {}", t::ORIENTATION, list);
        }
        None => {
            let _ = writeln!(o, "- {}", t::ORIENTATION_UNSET);
        }
    }
    o.push('\n');

    // 2. Komponen tumor: tabel pengukuran per label, grup, fokus.
    let _ = writeln!(o, "## {}\n", t::SECTIONS[1]);
    let u = r.dimension_uncertainty_um;
    let rows: Vec<Vec<String>> = r
        .labels
        .iter()
        .map(|l| {
            vec![
                l.name.clone(),
                t::kind_name(l.kind).to_owned(),
                l.group.clone(),
                fmt_mm3(l.volume_cavalieri_um3),
                opt(l.volume_reviewed_um3, fmt_mm3),
                opt(l.volume_corrected_um3, fmt_mm3),
                opt(l.volume_mesh_um3, fmt_mm3),
                opt(l.surface_area_um2, fmt_mm2),
                opt(l.max_diameter_um, |d| fmt_distance(d, u)),
                opt(l.n_components, |n| n.to_string()),
                format!(
                    "{} / {} / {}",
                    l.n_slices_annotated, l.n_slices_interpolated, l.n_slices_reviewed
                ),
            ]
        })
        .collect();
    if rows.is_empty() {
        let _ = writeln!(o, "{}\n", t::NONE);
    } else {
        md_table(&mut o, &t::MEASURE_HEADER, &rows);
    }
    let _ = writeln!(o, "{}\n", t::CAVALIERI_NOTE);
    if let Some(f) = r.shrinkage_linear {
        let _ = writeln!(
            o,
            "{}\n",
            t::SHRINKAGE_NOTE.replace("{f}", &format!("{f:.4}"))
        );
    }
    if r.grid.is_none() && !r.labels.is_empty() {
        let _ = writeln!(o, "{}\n", t::MESH_MISSING_NOTE);
    }
    let rows: Vec<Vec<String>> = r
        .groups
        .iter()
        .flat_map(|g| {
            g.members.iter().map(move |m| {
                vec![
                    g.group.clone(),
                    m.name.clone(),
                    fmt_mm3(m.volume_um3),
                    format!("{:.1} %", m.fraction * 100.0),
                ]
            })
        })
        .collect();
    if !rows.is_empty() {
        md_table(&mut o, &t::GROUP_HEADER, &rows);
    }
    if let Some(a) = &r.analysis {
        if let Some(p) = a.fractions.necrosis_percent {
            let _ = writeln!(o, "- {}: {:.1} %", t::NECROSIS_PERCENT, p);
        }
        if let Some(f) = &a.foci {
            let _ = writeln!(o, "- {}: {}\n", t::FOCI_COUNT, f.n);
            let rows: Vec<Vec<String>> = f
                .foci
                .iter()
                .enumerate()
                .map(|(i, fc)| {
                    vec![
                        (i + 1).to_string(),
                        fmt_mm3(fc.volume_um3),
                        fmt_distance(fc.max_diameter_um, u),
                        fmt_point_mm(fc.centroid_um),
                    ]
                })
                .collect();
            md_table(&mut o, &t::FOCI_HEADER, &rows);
            for (i, row) in f.distances_um.iter().enumerate() {
                for (j, d) in row.iter().enumerate().skip(i + 1) {
                    let _ = writeln!(
                        o,
                        "- {} {}–{}: {}",
                        t::FOCI_DISTANCE,
                        i + 1,
                        j + 1,
                        fmt_uncertainty(d)
                    );
                }
            }
            o.push('\n');
        }
    }

    // 3. Margin.
    let _ = writeln!(o, "## {}\n", t::SECTIONS[2]);
    match r.analysis.as_ref() {
        None => {
            let _ = writeln!(o, "{}\n", t::ANALYSIS_MISSING);
        }
        Some(a) => match &a.margin {
            None => {
                let _ = writeln!(o, "{}\n", t::MARGIN_MISSING);
            }
            Some(m) => {
                let _ = writeln!(o, "- {}: {}", t::MARGIN_DMIN, fmt_uncertainty(&m.d_min));
                let _ = writeln!(
                    o,
                    "- {}: {}",
                    t::DIRECTION,
                    t::direction_text(m.direction, m.direction_name)
                );
                let _ = writeln!(o, "- {}: {}", t::STATUS, t::margin_status_name(m.status));
                let _ = writeln!(
                    o,
                    "- {}: {}",
                    t::NEAREST_TUMOR,
                    fmt_point_mm(m.nearest_tumor_um)
                );
                let _ = writeln!(
                    o,
                    "- {}: {}\n",
                    t::NEAREST_MARGIN,
                    fmt_point_mm(m.nearest_margin_um)
                );
                let rows: Vec<Vec<String>> = SignedAxis::ALL
                    .iter()
                    .zip(m.per_direction.iter())
                    .map(|(axis, d)| {
                        let name = r.orientation.as_ref().and_then(|or| or.name_of(*axis));
                        vec![
                            t::direction_text(*axis, name),
                            d.as_ref()
                                .map(fmt_uncertainty)
                                .unwrap_or_else(|| t::NOT_AVAILABLE.to_owned()),
                        ]
                    })
                    .collect();
                md_table(&mut o, &t::MARGIN_DIR_HEADER, &rows);
                for f in &r.figures {
                    // Tautan relatif terhadap report.md; angka di keterangan.
                    let _ = writeln!(
                        o,
                        "![{}]({})\n",
                        cell(&t::figure_alt(&f.label)),
                        f.file.replace(' ', "%20")
                    );
                    let _ = writeln!(
                        o,
                        "*{}*\n",
                        t::figure_caption(
                            &cell(&f.label),
                            f.slide.0,
                            &cell(&f.slide_file),
                            &fmt_mm(f.slide_z_um),
                            &fmt_uncertainty(&m.d_min),
                            &fmt_mm(f.in_plane_um),
                            &fmt_mm(f.dz_um),
                            (f.um_per_px[0] + f.um_per_px[1]) / 2.0,
                        )
                    );
                }
            }
        },
    }

    // 4. Invasi.
    let _ = writeln!(o, "## {}\n", t::SECTIONS[3]);
    match r.analysis.as_ref() {
        None => {
            let _ = writeln!(o, "{}\n", t::ANALYSIS_MISSING);
        }
        Some(a) => {
            if a.contacts.is_empty() && a.capsule.is_none() {
                let _ = writeln!(o, "{}\n", t::INVASION_NONE);
            }
            if !a.contacts.is_empty() {
                let rows: Vec<Vec<String>> = a
                    .contacts
                    .iter()
                    .map(|c| {
                        vec![
                            c.name.clone(),
                            t::kind_name(c.kind).to_owned(),
                            yes_no(c.result.any).to_owned(),
                            format!("{:.4}", c.result.area_mm2),
                            c.result.patches.len().to_string(),
                        ]
                    })
                    .collect();
                md_table(&mut o, &t::INVASION_HEADER, &rows);
                let rows: Vec<Vec<String>> = a
                    .contacts
                    .iter()
                    .flat_map(|c| {
                        c.result.patches.iter().enumerate().map(move |(i, p)| {
                            vec![
                                c.name.clone(),
                                (i + 1).to_string(),
                                format!("{:.4}", p.area_mm2),
                                fmt_point_mm(p.centroid_um),
                            ]
                        })
                    })
                    .collect();
                if !rows.is_empty() {
                    md_table(&mut o, &t::PATCH_HEADER, &rows);
                }
            }
            if let Some(c) = &a.capsule {
                let _ = writeln!(o, "- {}: {}", t::CAPSULE_BREACH, yes_no(c.breached));
                let _ = writeln!(
                    o,
                    "- {}: {}",
                    t::CAPSULE_OUTSIDE,
                    fmt_mm3(c.volume_outside_um3)
                );
            }
            if let Some(h) = &a.hypoxia {
                let _ = writeln!(
                    o,
                    "- {}: {:.1} %",
                    t::HYPOXIA.replace("{t}", &format!("{:.0}", h.threshold_um)),
                    h.fraction_beyond * 100.0
                );
            }
            o.push('\n');
        }
    }

    // 5. Ukuran staging.
    let _ = writeln!(o, "## {}\n", t::SECTIONS[4]);
    match r.analysis.as_ref() {
        None => {
            let _ = writeln!(o, "{}\n", t::ANALYSIS_MISSING);
        }
        Some(a) => {
            match &a.size {
                None => {
                    let _ = writeln!(o, "{}", t::SIZE_MISSING);
                }
                Some(s) => {
                    let _ = writeln!(
                        o,
                        "- {}: {}",
                        t::SIZE_MAX_DIAMETER,
                        fmt_uncertainty(&s.max_diameter_um)
                    );
                    let axes = s
                        .pca_axes_um
                        .iter()
                        .map(fmt_uncertainty)
                        .collect::<Vec<_>>()
                        .join(" × ");
                    let _ = writeln!(o, "- {}: {}", t::SIZE_PCA, axes);
                }
            }
            if let Some(d) = &a.depth {
                let _ = writeln!(o, "- {}: {}", t::DEPTH, fmt_uncertainty(&d.depth));
            }
            if let Some(d) = &a.breslow {
                let _ = writeln!(o, "- {}: {}", t::BRESLOW, fmt_uncertainty(&d.depth));
            }
            o.push('\n');
        }
    }

    // 6. Atribut.
    let _ = writeln!(o, "## {}\n", t::SECTIONS[5]);
    if r.attrs.is_empty() {
        let _ = writeln!(o, "{}\n", t::ATTR_NONE);
    }
    for at in &r.attrs {
        let _ = writeln!(o, "### {} ({})\n", cell(&at.name), cell(&at.unit));
        let _ = writeln!(o, "- {}: {}", t::ATTR_METHOD, cell(&at.method));
        let _ = writeln!(
            o,
            "- {}: {}",
            t::ATTR_HASH,
            opt(at.params_hash, |h| format!("{h:016x}"))
        );
        if at.method.contains("area") {
            let _ = writeln!(o, "- {}", t::ATTR_AREA_BASED);
        }
        if at.method.contains("unavailable") {
            let _ = writeln!(o, "- {}", t::ATTR_TPS_UNAVAILABLE);
        }
        o.push('\n');
        let rows: Vec<Vec<String>> = at
            .per_slide
            .iter()
            .map(|s| {
                vec![
                    s.slide.to_string(),
                    format!("{:.2}", s.mean),
                    format!("{:.0}", s.weight),
                    opt(s.intratumoral, |v| format!("{v:.2}")),
                    opt(s.peritumoral, |v| format!("{v:.2}")),
                ]
            })
            .collect();
        if !rows.is_empty() {
            md_table(&mut o, &t::ATTR_SLIDE_HEADER, &rows);
        }
        if let Some(a) = &r.analysis {
            if let Some(e) = a.attr_stats.iter().find(|e| e.name == at.name) {
                let s = &e.stats;
                md_table(
                    &mut o,
                    &t::ATTR_STATS_HEADER,
                    &[vec![
                        s.n.to_string(),
                        format!("{:.2}", s.mean),
                        format!("{:.2}", s.median),
                        format!("{:.2}", s.p10),
                        format!("{:.2}", s.p90),
                        opt(s.heterogeneity_cv, |v| format!("{v:.3}")),
                        e.unit.clone(),
                    ]],
                );
            }
        }
    }

    // 7. Provenance.
    let _ = writeln!(o, "## {}\n", t::SECTIONS[6]);
    let total = r.provenance.total.max(1) as f64;
    let rows: Vec<Vec<String>> = r
        .provenance
        .by_source
        .iter()
        .map(|(s, n)| {
            vec![
                t::source_name(s).to_owned(),
                n.to_string(),
                format!("{:.1} %", *n as f64 / total * 100.0),
            ]
        })
        .collect();
    if rows.is_empty() {
        let _ = writeln!(o, "{}\n", t::NONE);
    } else {
        md_table(&mut o, &t::PROV_HEADER, &rows);
    }
    let _ = writeln!(
        o,
        "- {}: {} / {}",
        t::PROV_REVIEWED,
        r.provenance.reviewed,
        r.provenance.total
    );
    let drafts = if r.provenance.drafts.is_empty() {
        t::NONE.to_owned()
    } else {
        r.provenance
            .drafts
            .iter()
            .map(|(s, l)| format!("{s}:{l}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let _ = writeln!(o, "- {}: {}", t::PROV_DRAFTS, cell(&drafts));
    if r.slides.order_suggested {
        let _ = writeln!(o, "- {}", t::ORDER_SUGGESTED);
    }
    o.push('\n');

    // 8. Parameter, versi, peringatan, kalimat riset.
    let _ = writeln!(o, "## {}\n", t::SECTIONS[7]);
    let _ = writeln!(o, "- {}: {}", t::APP_VERSION, cell(&r.app_version));
    let _ = writeln!(
        o,
        "- {}: total {}, dipakai {}, terdaftar {}, gagal QC {}, acuan {}, jarak median {}",
        t::SLIDES_SUMMARY,
        r.slides.total,
        r.slides.usable,
        r.slides.registered,
        r.slides.flagged.len(),
        opt(r.slides.reference, |s| s.to_string()),
        opt(r.slides.median_spacing_um, |v| format!("{v:.1} µm")),
    );
    let c = &r.recon;
    let _ = writeln!(
        o,
        "- {}: s_xy = {} µm, z_factor_max = {}, band_px = {}, max_gap_um = {}, \
         smooth_iterations = {}, decimate_error_factor = {}, mpp_label = {} µm",
        t::PARAMS_RECON,
        c.s_xy_um,
        c.z_factor_max,
        c.band_px,
        opt(c.max_gap_um, |v| v.to_string()),
        c.smooth_iterations,
        c.decimate_error_factor,
        c.mpp_label,
    );
    let s = &r.spatial;
    let _ = writeln!(
        o,
        "- {}: margin_close_um = {}, peri_band_um = {}, hypoxia_um = {}, min_focus_um3 = {}, \
         ihc_hetero_warn = {}, vessel_hist_bin_um = {}",
        t::PARAMS_SPATIAL,
        s.margin_close_um,
        s.peri_band_um,
        s.hypoxia_um,
        opt(s.min_focus_um3, |v| v.to_string()),
        s.ihc_hetero_warn,
        s.vessel_hist_bin_um,
    );
    if let Some(g) = &r.grid {
        let _ = writeln!(
            o,
            "- {}: {} × {} × {} voxel, s_xy = {} µm, s_z median = {:.2} µm",
            t::GRID,
            g.size[0],
            g.size[1],
            g.nz,
            g.s_xy_um,
            g.s_z_um
        );
    }
    let _ = writeln!(
        o,
        "- {}: ± {} mm — {}",
        t::DIM_UNCERTAINTY,
        fmt_mm(r.dimension_uncertainty_um),
        t::DIM_UNCERTAINTY_NOTE
    );
    for at in &r.attrs {
        let _ = writeln!(
            o,
            "- {} — {}: {} ({} {})",
            t::ATTR_PARAMS,
            cell(&at.name),
            cell(&at.method),
            t::ATTR_HASH,
            opt(at.params_hash, |h| format!("{h:016x}"))
        );
    }
    o.push('\n');
    let notes = r.all_notes();
    let _ = writeln!(o, "### {}\n", t::NOTES);
    if notes.is_empty() {
        let _ = writeln!(o, "- {}", t::NONE);
    }
    for n in &notes {
        let _ = writeln!(o, "- {}", cell(n));
    }
    o.push('\n');
    let warnings = r.all_warnings();
    let _ = writeln!(o, "### {}\n", t::WARNINGS);
    if warnings.is_empty() {
        let _ = writeln!(o, "- {}", t::NONE);
    }
    for w in &warnings {
        let _ = writeln!(o, "- {}", cell(&t::warning_text(w)));
    }
    o.push('\n');
    let _ = writeln!(o, "---\n\n**{RESEARCH_ONLY_NOTICE}**");
    o
}

// ------------------------------------------------------------------- csv

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_owned()
    }
}

struct Csv(String);

impl Csv {
    fn row(&mut self, section: &str, item: &str, label: &str, value: &str, pm: &str, unit: &str) {
        let fields = [section, item, label, value, pm, unit];
        let line = fields
            .iter()
            .map(|f| csv_field(f))
            .collect::<Vec<_>>()
            .join(",");
        self.0.push_str(&line);
        self.0.push('\n');
    }

    fn dist(&mut self, section: &str, item: &str, label: &str, u: &Uncertainty) {
        self.row(
            section,
            item,
            label,
            &fmt_mm(u.value_um),
            &fmt_mm(u.plus_minus_um),
            "mm",
        );
    }
}

/// Render laporan CSV: kolom `bagian,butir,label,nilai,ketidakpastian,satuan`.
/// Bersumber dari data yang sama dengan Markdown (dan `analysis.json`).
pub fn render_csv(r: &Report, clinical: ClinicalScope<'_>) -> String {
    let mut c = Csv(String::new());
    c.row(
        "bagian",
        "butir",
        "label",
        "nilai",
        "ketidakpastian",
        "satuan",
    );
    c.row("0", "pemberitahuan", "", RESEARCH_ONLY_NOTICE, "", "");

    c.row("1", "nama_sampel", "", &r.sample_name, "", "");
    for (k, v) in clinical_rows(clinical) {
        c.row("1", k, "", &v, "", "");
    }
    if let Some(or) = &r.orientation {
        for (axis, text) in orientation_rows(or) {
            c.row("1", "orientasi", axis.as_str(), &text, "", "");
        }
    }

    let u = r.dimension_uncertainty_um;
    for l in &r.labels {
        let n = l.name.as_str();
        c.row(
            "2",
            "volume_cavalieri",
            n,
            &fmt_mm3(l.volume_cavalieri_um3),
            "",
            "mm3",
        );
        if let Some(v) = l.volume_reviewed_um3 {
            c.row("2", "volume_ditinjau", n, &fmt_mm3(v), "", "mm3");
        }
        if let Some(v) = l.volume_corrected_um3 {
            c.row("2", "volume_terkoreksi_susut", n, &fmt_mm3(v), "", "mm3");
        }
        if let Some(v) = l.volume_mesh_um3 {
            c.row("2", "volume_mesh", n, &fmt_mm3(v), "", "mm3");
        }
        if let Some(v) = l.surface_area_um2 {
            c.row("2", "luas_permukaan", n, &fmt_mm2(v), "", "mm2");
        }
        if let Some(v) = l.max_diameter_um {
            c.row("2", "diameter_maks", n, &fmt_mm(v), &fmt_mm(u), "mm");
        }
        if let Some(v) = l.n_components {
            c.row("2", "komponen", n, &v.to_string(), "", "");
        }
        c.row(
            "2",
            "irisan_anotasi",
            n,
            &l.n_slices_annotated.to_string(),
            "",
            "",
        );
        c.row(
            "2",
            "irisan_interpolasi",
            n,
            &l.n_slices_interpolated.to_string(),
            "",
            "",
        );
        c.row(
            "2",
            "irisan_ditinjau",
            n,
            &l.n_slices_reviewed.to_string(),
            "",
            "",
        );
    }
    for g in &r.groups {
        c.row(
            "2",
            "volume_grup",
            &g.group,
            &fmt_mm3(g.volume_um3),
            "",
            "mm3",
        );
        for m in &g.members {
            c.row(
                "2",
                "fraksi_dalam_grup",
                &format!("{}/{}", g.group, m.name),
                &format!("{:.4}", m.fraction),
                "",
                "",
            );
        }
    }

    if let Some(a) = &r.analysis {
        if let Some(f) = &a.foci {
            c.row("2", "jumlah_fokus", "", &f.n.to_string(), "", "");
            for (i, fc) in f.foci.iter().enumerate() {
                let id = (i + 1).to_string();
                c.row("2", "volume_fokus", &id, &fmt_mm3(fc.volume_um3), "", "mm3");
                c.row(
                    "2",
                    "diameter_fokus",
                    &id,
                    &fmt_mm(fc.max_diameter_um),
                    &fmt_mm(u),
                    "mm",
                );
            }
            for (i, row) in f.distances_um.iter().enumerate() {
                for (j, d) in row.iter().enumerate().skip(i + 1) {
                    c.dist("2", "jarak_antar_fokus", &format!("{}-{}", i + 1, j + 1), d);
                }
            }
        }
        if let Some(p) = a.fractions.necrosis_percent {
            c.row("2", "nekrosis", "", &format!("{p:.2}"), "", "%");
        }
        if let Some(m) = &a.margin {
            c.dist("3", "margin_d_min", "", &m.d_min);
            c.row(
                "3",
                "margin_arah",
                "",
                &t::direction_text(m.direction, m.direction_name),
                "",
                "",
            );
            c.row(
                "3",
                "margin_status",
                "",
                t::margin_status_name(m.status),
                "",
                "",
            );
            for (axis, d) in SignedAxis::ALL.iter().zip(m.per_direction.iter()) {
                let name = r.orientation.as_ref().and_then(|or| or.name_of(*axis));
                if let Some(d) = d {
                    c.dist("3", "margin_per_arah", &t::direction_text(*axis, name), d);
                }
            }
        }
        for ct in &a.contacts {
            c.row("4", "kontak", &ct.name, yes_no(ct.result.any), "", "");
            c.row(
                "4",
                "luas_kontak",
                &ct.name,
                &format!("{:.4}", ct.result.area_mm2),
                "",
                "mm2",
            );
            c.row(
                "4",
                "tambalan",
                &ct.name,
                &ct.result.patches.len().to_string(),
                "",
                "",
            );
        }
        if let Some(cp) = &a.capsule {
            c.row("4", "penetrasi_kapsul", "", yes_no(cp.breached), "", "");
            c.row(
                "4",
                "volume_luar_kapsul",
                "",
                &fmt_mm3(cp.volume_outside_um3),
                "",
                "mm3",
            );
        }
        if let Some(h) = &a.hypoxia {
            c.row(
                "4",
                "fraksi_jauh_dari_pembuluh",
                &format!("> {:.0} um", h.threshold_um),
                &format!("{:.4}", h.fraction_beyond),
                "",
                "",
            );
        }
        if let Some(s) = &a.size {
            c.dist("5", "diameter_maks", "", &s.max_diameter_um);
            for (i, ax) in s.pca_axes_um.iter().enumerate() {
                c.dist("5", "sumbu_pca", &(i + 1).to_string(), ax);
            }
        }
        if let Some(d) = &a.depth {
            c.dist("5", "kedalaman_invasi", "", &d.depth);
        }
        if let Some(d) = &a.breslow {
            c.dist("5", "ketebalan_breslow", "", &d.depth);
        }
        for e in &a.attr_stats {
            c.row(
                "6",
                "rata_rata_tumor",
                &e.name,
                &format!("{:.4}", e.stats.mean),
                "",
                &e.unit,
            );
            c.row(
                "6",
                "median_tumor",
                &e.name,
                &format!("{:.4}", e.stats.median),
                "",
                &e.unit,
            );
            if let Some(cv) = e.stats.heterogeneity_cv {
                c.row(
                    "6",
                    "heterogenitas_cv",
                    &e.name,
                    &format!("{cv:.4}"),
                    "",
                    "",
                );
            }
        }
    }

    for at in &r.attrs {
        c.row("6", "metode", &at.name, &at.method, "", "");
        if let Some(h) = at.params_hash {
            c.row(
                "6",
                "hash_parameter",
                &at.name,
                &format!("{h:016x}"),
                "",
                "",
            );
        }
        for s in &at.per_slide {
            c.row(
                "6",
                "rata_rata_irisan",
                &format!("{}/{}", at.name, s.slide),
                &format!("{:.4}", s.mean),
                "",
                &at.unit,
            );
        }
    }

    for (s, n) in &r.provenance.by_source {
        c.row(
            "7",
            "sumber_anotasi",
            t::source_name(s),
            &n.to_string(),
            "",
            "",
        );
    }
    c.row(
        "7",
        "anotasi_ditinjau",
        "",
        &r.provenance.reviewed.to_string(),
        "",
        "",
    );
    c.row(
        "7",
        "anotasi_draft",
        "",
        &r.provenance.drafts.len().to_string(),
        "",
        "",
    );

    c.row("8", "versi_aplikasi", "", &r.app_version, "", "");
    c.row("8", "s_xy", "", &r.recon.s_xy_um.to_string(), "", "um");
    c.row(
        "8",
        "mpp_label",
        "",
        &r.recon.mpp_label.to_string(),
        "",
        "um",
    );
    c.row(
        "8",
        "margin_close_um",
        "",
        &r.spatial.margin_close_um.to_string(),
        "",
        "um",
    );
    c.row(
        "8",
        "ketidakpastian_ukuran",
        "",
        &fmt_mm(r.dimension_uncertainty_um),
        "",
        "mm",
    );
    for n in r.all_notes() {
        c.row("8", "catatan", "", &n, "", "");
    }
    for w in r.all_warnings() {
        c.row(
            "8",
            "peringatan",
            w.i18n_key(),
            &t::warning_text(&w),
            "",
            "",
        );
    }
    c.row("8", "pemberitahuan", "", RESEARCH_ONLY_NOTICE, "", "");
    c.0
}
