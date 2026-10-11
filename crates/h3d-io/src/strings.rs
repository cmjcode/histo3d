//! Semua teks laporan (bahasa Indonesia) di satu tempat, agar kelak mudah
//! dipindahkan ke `h3d-i18n`. Tidak ada teks rekomendasi tindakan di sini
//! (aturan keras 18): hanya nama besaran, status, dan peringatan.

use h3d_core::{AnatDir, LabelKind, SignedAxis, VesselKind};
use h3d_volume::{MarginStatus, Warning};

/// Judul laporan.
pub const TITLE: &str = "Laporan HISTO3D";
/// Judul bagian 1–8 (urutan tetap, 8.12.8).
pub const SECTIONS: [&str; 8] = [
    "1. Konteks sampel dan orientasi",
    "2. Komponen tumor",
    "3. Margin",
    "4. Invasi",
    "5. Ukuran staging",
    "6. Atribut",
    "7. Provenance anotasi",
    "8. Parameter, versi, dan peringatan",
];

/// Teks "tidak tersedia".
pub const NOT_AVAILABLE: &str = "tidak tersedia";
/// Teks "tidak ada".
pub const NONE: &str = "tidak ada";
/// Teks "ya".
pub const YES: &str = "ya";
/// Teks "tidak".
pub const NO: &str = "tidak";

/// Nama sampel.
pub const SAMPLE_NAME: &str = "Nama sampel";
/// Catatan sampel.
pub const SAMPLE_NOTES: &str = "Catatan";
/// Orientasi spesimen.
pub const ORIENTATION: &str = "Orientasi spesimen";
/// Orientasi tidak diisi.
pub const ORIENTATION_UNSET: &str =
    "Orientasi anatomis tidak diisi; arah dilaporkan sebagai sumbu mentah (+X, −X, …).";
/// Data klinis tidak disertakan.
pub const CLINICAL_OMITTED: &str = "Konteks klinis tidak disertakan dalam ekspor ini.";
/// Id pseudonim.
pub const CLIN_PSEUDO_ID: &str = "Id pseudonim subjek";
/// Organ / topografi.
pub const CLIN_TOPOGRAPHY: &str = "Organ / topografi (ICD-O)";
/// Diagnosis / morfologi.
pub const CLIN_MORPHOLOGY: &str = "Diagnosis / morfologi (ICD-O)";
/// Usia.
pub const CLIN_AGE: &str = "Usia (tahun)";
/// Jenis kelamin.
pub const CLIN_GENDER: &str = "Jenis kelamin";
/// Sisi tubuh.
pub const CLIN_SIDE: &str = "Sisi tubuh";
/// Tindakan pengambilan.
pub const CLIN_SPECIMEN_TYPE: &str = "Jenis pengambilan spesimen";
/// Derajat histologis.
pub const CLIN_DEGREE: &str = "Derajat histologis";
/// Stadium patologis.
pub const CLIN_STAGE: &str = "Stadium patologis (TNM)";
/// Terapi sebelum operasi.
pub const CLIN_PRIOR_THERAPY: &str = "Terapi sebelum operasi";
/// Hasil uji gen.
pub const CLIN_GENE_RESULTS: &str = "Hasil uji gen";
/// Bulan spesimen.
pub const CLIN_MONTH: &str = "Bulan spesimen";

/// Kepala tabel pengukuran per label.
pub const MEASURE_HEADER: [&str; 11] = [
    "Label",
    "Jenis",
    "Grup",
    "Volume Cavalieri (mm³)",
    "Volume, irisan ditinjau saja (mm³)",
    "Volume terkoreksi susut (mm³)",
    "Volume mesh (mm³)",
    "Luas permukaan (mm²)",
    "Diameter maks (mm)",
    "Komponen",
    "Irisan (anotasi / interpolasi / ditinjau)",
];
/// Catatan tentang Cavalieri.
pub const CAVALIERI_NOTE: &str = "Volume Cavalieri (angka utama) hanya memakai irisan yang \
benar-benar dianotasi; irisan hasil interpolasi tidak ikut dihitung.";
/// Catatan koreksi susut.
pub const SHRINKAGE_NOTE: &str = "Faktor susut linear f = {f}; volume terkoreksi = V × f³. \
Koreksi hanya dilaporkan berdampingan dan tidak diterapkan ke geometri.";
/// Mesh belum ada.
pub const MESH_MISSING_NOTE: &str = "Mesh belum direkonstruksi (atau cache sudah basi); \
besaran mesh tidak tersedia.";
/// Kepala tabel grup.
pub const GROUP_HEADER: [&str; 4] = ["Grup", "Anggota", "Volume (mm³)", "Fraksi dalam grup"];
/// Jumlah fokus.
pub const FOCI_COUNT: &str = "Jumlah fokus tumor";
/// Kepala tabel fokus.
pub const FOCI_HEADER: [&str; 4] = [
    "Fokus",
    "Volume (mm³)",
    "Diameter maks (mm)",
    "Centroid (mm)",
];
/// Jarak antar fokus.
pub const FOCI_DISTANCE: &str = "Jarak antar fokus";
/// Persen nekrosis.
pub const NECROSIS_PERCENT: &str = "Nekrosis (% dari tumor + nekrosis dalam pita tumor)";

/// Analisis belum dijalankan.
pub const ANALYSIS_MISSING: &str =
    "Analisis spasial belum dijalankan untuk keadaan proyek saat ini.";
/// Margin tidak dapat dinilai.
pub const MARGIN_MISSING: &str =
    "Tidak dapat dinilai: butuh label berjenis Tumor dan Tepi reseksi.";
/// Jarak minimum.
pub const MARGIN_DMIN: &str = "Jarak minimum tumor ke tepi reseksi";
/// Arah.
pub const DIRECTION: &str = "Arah";
/// Status.
pub const STATUS: &str = "Status";
/// Titik terdekat tumor.
pub const NEAREST_TUMOR: &str = "Titik tumor terdekat (mm)";
/// Titik terdekat margin.
pub const NEAREST_MARGIN: &str = "Titik tepi terdekat (mm)";
/// Kepala tabel margin per arah.
pub const MARGIN_DIR_HEADER: [&str; 2] = ["Arah", "Jarak minimum"];
/// Teks alternatif gambar margin.
pub fn figure_alt(label: &str) -> String {
    format!("Irisan terdekat ke titik tumor–tepi terdekat ({label})")
}

/// Keterangan gambar margin. Angka dicetak di sini karena gambar tidak
/// memuat teks. Tidak ada kalimat rekomendasi (aturan keras 18).
#[allow(clippy::too_many_arguments)]
pub fn figure_caption(
    label: &str,
    slide: u32,
    slide_file: &str,
    slide_z_mm: &str,
    d_min: &str,
    in_plane_mm: &str,
    dz_mm: &str,
    um_per_px: f64,
) -> String {
    format!(
        "Gambar: irisan {slide} ({slide_file}, z = {slide_z_mm} mm) pada ruangnya sendiri, \
         dengan garis luar label. Titik magenta = titik tumor terdekat; titik biru = titik \
         terdekat pada \"{label}\"; panah dari tumor ke tepi. Jarak minimum 3D {d_min}; \
         pada bidang irisan ini kedua titik berjarak {in_plane_mm} mm dan selisih z-nya \
         {dz_mm} mm. Skala: 1 piksel = {um_per_px:.1} µm."
    )
}

/// Penanda resolusi Z.
pub const Z_LIMITED: &str = "resolusi Z = jarak irisan";

/// Kepala tabel invasi.
pub const INVASION_HEADER: [&str; 5] = [
    "Struktur",
    "Jenis",
    "Kontak",
    "Luas kontak (mm²)",
    "Tambalan",
];
/// Tidak ada struktur.
pub const INVASION_NONE: &str =
    "Tidak ada label pembuluh, saraf, atau kapsul yang berisi untuk dinilai.";
/// Kepala tabel tambalan.
pub const PATCH_HEADER: [&str; 4] = ["Struktur", "Tambalan", "Luas (mm²)", "Centroid (mm)"];
/// Penetrasi kapsul.
pub const CAPSULE_BREACH: &str = "Penetrasi kapsul";
/// Volume tumor di luar kapsul.
pub const CAPSULE_OUTSIDE: &str = "Volume tumor di luar kapsul (mm³)";
/// Proksi hipoksia.
pub const HYPOXIA: &str = "Fraksi voxel tumor berjarak > {t} µm dari pembuluh darah";

/// Diameter maksimum.
pub const SIZE_MAX_DIAMETER: &str = "Diameter maksimum";
/// Sumbu PCA.
pub const SIZE_PCA: &str = "Ukuran pada sumbu utama PCA";
/// Kedalaman invasi.
pub const DEPTH: &str = "Kedalaman invasi (acuan Permukaan)";
/// Breslow.
pub const BRESLOW: &str = "Ketebalan Breslow (acuan garis Rujukan)";
/// Ukuran tidak tersedia.
pub const SIZE_MISSING: &str = "Tidak dapat dinilai: butuh label berjenis Tumor.";

/// Tidak ada atribut.
pub const ATTR_NONE: &str = "Tidak ada atribut yang dihitung.";
/// Kepala tabel atribut per irisan.
pub const ATTR_SLIDE_HEADER: [&str; 5] = [
    "Irisan",
    "Rata-rata",
    "Bobot (sel/piksel)",
    "Intratumoral",
    "Peritumoral",
];
/// Statistik atribut atas tumor.
pub const ATTR_STATS_HEADER: [&str; 7] = [
    "n voxel",
    "Rata-rata",
    "Median",
    "p10",
    "p90",
    "Heterogenitas (CV kubus 1 mm³)",
    "Satuan",
];
/// Metode.
pub const ATTR_METHOD: &str = "Metode";
/// Hash parameter.
pub const ATTR_HASH: &str = "Hash parameter";
/// Parameter skor.
pub const ATTR_PARAMS: &str = "Parameter skor";
/// Skor berbasis luas.
pub const ATTR_AREA_BASED: &str = "Skor berbasis luas, bukan per sel.";
/// TPS/CPS tidak tersedia.
pub const ATTR_TPS_UNAVAILABLE: &str = "TPS/CPS tidak tersedia (butuh masker Tumor dan Stroma); \
yang tersimpan adalah persen positif.";

/// Kepala tabel provenance.
pub const PROV_HEADER: [&str; 3] = ["Sumber", "Jumlah (irisan × label)", "Fraksi"];
/// Sudah ditinjau.
pub const PROV_REVIEWED: &str = "Anotasi sudah ditinjau";
/// Belum ditinjau.
pub const PROV_DRAFTS: &str = "Anotasi belum ditinjau (Draft)";
/// Urutan dari usulan.
pub const ORDER_SUGGESTED: &str = "Urutan irisan berasal dari usulan otomatis yang diterima.";

/// Versi aplikasi.
pub const APP_VERSION: &str = "Versi aplikasi";
/// Parameter rekonstruksi.
pub const PARAMS_RECON: &str = "Parameter rekonstruksi";
/// Parameter analisis spasial.
pub const PARAMS_SPATIAL: &str = "Parameter analisis spasial";
/// Ringkasan irisan.
pub const SLIDES_SUMMARY: &str = "Irisan";
/// Kisi.
pub const GRID: &str = "Kisi volume";
/// Ketidakpastian ukuran.
pub const DIM_UNCERTAINTY: &str = "Ketidakpastian jarak/ukuran 3D";
/// Penjelasan ketidakpastian.
pub const DIM_UNCERTAINTY_NOTE: &str =
    "± (½ ukuran voxel pada sumbu dominan + taksiran galat registrasi)";
/// Catatan.
pub const NOTES: &str = "Catatan";
/// Peringatan.
pub const WARNINGS: &str = "Peringatan";

/// Nama sumber anotasi.
pub fn source_name(short: &str) -> &'static str {
    match short {
        "manual" => "Manual",
        "classic" => "Segmentasi klasik",
        "ai" => "Model AI",
        "interpolated" => "Interpolasi",
        "imported" => "Impor berkas",
        _ => "Lainnya",
    }
}

/// Nama jenis label.
pub fn kind_name(kind: LabelKind) -> &'static str {
    match kind {
        LabelKind::Tumor => "Tumor",
        LabelKind::Necrosis => "Nekrosis",
        LabelKind::Stroma => "Stroma",
        LabelKind::Normal => "Normal",
        LabelKind::Vessel(VesselKind::Blood) => "Pembuluh darah",
        LabelKind::Vessel(VesselKind::Lymph) => "Pembuluh limfe",
        LabelKind::Vessel(VesselKind::Unknown) => "Pembuluh",
        LabelKind::Nerve => "Saraf",
        LabelKind::Capsule => "Kapsul",
        LabelKind::Margin => "Tepi reseksi",
        LabelKind::Surface => "Permukaan",
        LabelKind::Reference => "Garis rujukan",
        LabelKind::Other => "Lainnya",
    }
}

/// Nama arah anatomis.
pub fn anat_name(dir: AnatDir) -> &'static str {
    match dir {
        AnatDir::Anterior => "anterior",
        AnatDir::Posterior => "posterior",
        AnatDir::Superior => "superior",
        AnatDir::Inferior => "inferior",
        AnatDir::Medial => "medial",
        AnatDir::Lateral => "lateral",
        AnatDir::Proximal => "proksimal",
        AnatDir::Distal => "distal",
        AnatDir::Left => "kiri",
        AnatDir::Right => "kanan",
        AnatDir::Unknown => "tidak diketahui",
    }
}

/// Arah untuk laporan: sumbu mentah selalu dicetak, nama anatomis di
/// sebelahnya bila orientasi diketahui, mis. `+X (anterior)`.
pub fn direction_text(axis: SignedAxis, name: Option<AnatDir>) -> String {
    match name {
        Some(d) if d != AnatDir::Unknown => format!("{} ({})", axis.as_str(), anat_name(d)),
        _ => axis.as_str().to_owned(),
    }
}

/// Nama status margin (status, bukan anjuran).
pub fn margin_status_name(s: MarginStatus) -> &'static str {
    match s {
        MarginStatus::Involved => "Terlibat (d_min ≤ ½ s_xy)",
        MarginStatus::Close => "Dekat (d_min di bawah ambang)",
        MarginStatus::Clear => "Bebas (d_min di atas ambang)",
    }
}

/// Teks peringatan (padanan kunci `warn-*` di `h3d-i18n`).
pub fn warning_text(w: &Warning) -> String {
    match w {
        Warning::MarginClose { dir } => format!("Tumor dekat tepi reseksi arah {dir}"),
        Warning::MarginZUnassessable => "Jarak ke tepi pada arah Z tidak dapat dinilai".to_owned(),
        Warning::IhcHeterogeneity => {
            "Heterogenitas IHC tinggi; satu irisan mungkin tidak representatif".to_owned()
        }
        Warning::MostlyInterpolated => "Sebagian besar bentuk berasal dari interpolasi".to_owned(),
        Warning::HasDraft => "Angka memuat anotasi yang belum ditinjau".to_owned(),
        Warning::ZResolutionCoarse => "Resolusi Z lebih kasar dari ambang margin".to_owned(),
        Warning::MembraneAreaScore { name } => {
            format!("Skor {name} dihitung per luas, bukan per sel")
        }
        Warning::VolumeMismatch { percent } => {
            format!("Volume mesh berbeda {percent:.1}% dari Cavalieri")
        }
        Warning::SxyRaised { from, to } => {
            format!("Resolusi kerja dinaikkan dari {from:.0} ke {to:.0} µm agar muat di memori")
        }
        Warning::GapSplit { from_z, to_z, gap } => {
            format!("Objek diputus antara z {from_z:.0} dan {to_z:.0} µm (celah {gap:.0} µm)")
        }
        Warning::NonUniformZResampled { dz } => {
            format!("Jarak irisan tidak seragam; volume diresampel ke {dz:.1} µm")
        }
    }
}
