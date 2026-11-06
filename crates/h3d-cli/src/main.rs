//! `h3d`: antarmuka baris perintah HISTO3D (histo3d.md 8.9).
//!
//! Kode keluar: 0 sukses, 1 kesalahan pemakaian, 2 kesalahan data/berkas,
//! 3 QC gagal (ada irisan bertanda dan `--strict` aktif). Galat dicetak
//! sebagai satu baris berbahasa Indonesia ke stderr. Keluaran `--json`
//! stabil (kunci terurut, format bilangan tetap). Data klinis tidak pernah
//! dicetak ke stdout maupun log.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{anyhow, bail, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use h3d_core::{
    AttrDef, AttrKind, ClinicalMeta, IhcScoring, LabelKind, NoProgress, SlideId, SlideStatus,
    Stain, StainPattern,
};
use h3d_engine::phantom::{write_phantom, PhantomOptions, PhantomShapeKind};
use h3d_engine::{
    clock_from_env, to_stable_json_of, AttrParams, ClassicParams, ExportKind, ImportOptions,
    PaintMode, RegMetric, RegModel, RegOptions, RegisterRun, Session, StainChannel,
};
use serde_json::json;

/// Kode keluar: kesalahan pemakaian.
const EXIT_USAGE: u8 = 1;
/// Kode keluar: kesalahan data/berkas.
const EXIT_DATA: u8 = 2;
/// Kode keluar: QC gagal dengan `--strict`.
const EXIT_QC: u8 = 3;

#[derive(Parser)]
#[command(
    name = "h3d",
    version,
    about = "HISTO3D: rekonstruksi 3D tumor dari irisan histologi serial. Hanya untuk riset, bukan untuk diagnosis."
)]
struct Cli {
    /// Tampilkan log rinci (ulangi untuk lebih rinci).
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    verbose: u8,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Tampilkan level, mpp, dan ukuran sebuah slide (JSON).
    Info {
        /// Berkas slide.
        slide: PathBuf,
    },
    /// Buat thumbnail PNG sebuah slide.
    Thumb {
        /// Berkas slide.
        slide: PathBuf,
        /// Berkas PNG keluaran.
        #[arg(long)]
        out: PathBuf,
        /// Sisi terpanjang maksimum, piksel.
        #[arg(long, default_value_t = 1024)]
        max: u32,
    },
    /// Buat proyek baru (folder `.h3d`).
    New {
        /// Folder proyek.
        project: PathBuf,
    },
    /// Impor slide ke proyek (urutan = urutan alami nama berkas).
    Import {
        /// Folder proyek.
        project: PathBuf,
        /// Berkas slide.
        #[arg(required = true)]
        slides: Vec<PathBuf>,
        /// µm per piksel bila metadata berkas tidak memuatnya: `0.5` atau `0.5,0.5`.
        #[arg(long, value_parser = parse_mpp)]
        mpp: Option<[f64; 2]>,
        /// Jarak antar irisan, µm.
        #[arg(long = "spacing-um")]
        spacing_um: Option<f64>,
        /// Salin berkas slide ke dalam proyek.
        #[arg(long)]
        embed: bool,
    },
    /// Usulkan urutan dan orientasi irisan dari kemiripan gambar.
    #[command(name = "suggest-order")]
    SuggestOrder {
        /// Folder proyek.
        project: PathBuf,
        /// Keluaran JSON.
        #[arg(long)]
        json: bool,
        /// Terima seluruh usulan.
        #[arg(long)]
        apply: bool,
    },
    /// Daftarkan semua irisan ke irisan acuan.
    Register {
        /// Folder proyek.
        project: PathBuf,
        /// Model transformasi.
        #[arg(long, value_enum, default_value_t = ModelArg::Affine)]
        model: ModelArg,
        /// Metrik kemiripan.
        #[arg(long, value_enum, default_value_t = MetricArg::Ncc)]
        metric: MetricArg,
        /// Id irisan acuan (bawaan: tengah tumpukan).
        #[arg(long)]
        reference: Option<u32>,
        /// Keluar dengan kode 3 bila ada irisan gagal QC.
        #[arg(long)]
        strict: bool,
        /// Daftarkan ulang walau catatan lama masih sah.
        #[arg(long)]
        force: bool,
    },
    /// Tabel QC registrasi per irisan.
    Qc {
        /// Folder proyek.
        project: PathBuf,
        /// Keluaran JSON.
        #[arg(long)]
        json: bool,
    },
    /// Operasi label.
    Labels {
        #[command(subcommand)]
        command: LabelsCommand,
    },
    /// Segmentasi AI (ONNX) lalu terima hasilnya pada piksel yang masih kosong.
    Segment {
        /// Folder proyek.
        project: PathBuf,
        /// Folder model (`model.json` + `model.onnx`).
        #[arg(long)]
        model: PathBuf,
        /// Id irisan dipisah koma (bawaan: semua irisan Ok).
        #[arg(long, value_delimiter = ',')]
        slides: Vec<u32>,
    },
    /// Segmentasi klasik (ambang kanal pewarna + morfologi) pada irisan,
    /// lalu terima hasilnya hanya pada piksel yang masih kosong. Label
    /// dibuat (jenis `other`) bila belum ada.
    Classic {
        /// Folder proyek.
        project: PathBuf,
        /// Nama label tujuan.
        #[arg(long)]
        label: String,
        /// Id irisan dipisah koma (bawaan: semua irisan Ok).
        #[arg(long, value_delimiter = ',')]
        slides: Vec<u32>,
        /// Kanal pewarna yang diambangkan.
        #[arg(long, value_enum, default_value_t = ChannelArg::Hematoxylin)]
        channel: ChannelArg,
        /// Ambang 0..1 pada kanal yang diregangkan; tanpa nilai = Otsu.
        #[arg(long)]
        threshold: Option<f32>,
        /// Ambil piksel di bawah ambang.
        #[arg(long)]
        invert: bool,
        /// Radius pembukaan morfologi, µm.
        #[arg(long = "open-um", default_value_t = 4.0)]
        open_um: f64,
        /// Radius penutupan morfologi, µm.
        #[arg(long = "close-um", default_value_t = 8.0)]
        close_um: f64,
        /// Luas komponen minimum, µm².
        #[arg(long = "min-area-um2", default_value_t = 500.0)]
        min_area_um2: f64,
        /// Isi lubang (lumen kelenjar ikut masuk).
        #[arg(long = "fill-holes")]
        fill_holes: bool,
    },
    /// Ubah status irisan (ok, excluded, damaged).
    Status {
        /// Folder proyek.
        project: PathBuf,
        /// Id irisan.
        #[arg(long)]
        slide: u32,
        /// Status baru.
        #[arg(long, value_enum)]
        status: StatusArg,
    },
    /// Interpolasi bentuk label di antara irisan beranotasi.
    Interpolate {
        /// Folder proyek.
        project: PathBuf,
        /// Nama label.
        #[arg(long)]
        label: String,
    },
    /// Rekonstruksi mesh 3D semua label.
    Reconstruct {
        /// Folder proyek.
        project: PathBuf,
        /// Ukuran voxel XY, µm.
        #[arg(long)]
        sxy: Option<f64>,
        /// Iterasi penghalusan Taubin.
        #[arg(long)]
        smooth: Option<u32>,
        /// Galat decimate relatif terhadap ukuran voxel.
        #[arg(long)]
        decimate: Option<f64>,
        /// Rekonstruksi per objek: komponen label ditautkan antar irisan dan
        /// tiap objek menjadi mesh terpisah `<Label>#<id>`.
        #[arg(long = "per-object")]
        per_object: bool,
        /// Pisahkan objek yang menempel dalam satu irisan (watershed atas
        /// jarak ke tepi) sebelum penautan; menyiratkan `--per-object`.
        #[arg(long)]
        split: bool,
        /// Jarak minimum antar pusat objek saat memisahkan, µm.
        #[arg(long = "split-distance")]
        split_distance: Option<f64>,
        /// Kedalaman lembah minimum saat memisahkan, µm.
        #[arg(long = "split-h")]
        split_h: Option<f64>,
        /// Irisan hilang berurutan maksimum yang masih dijembatani saat
        /// menautkan objek.
        #[arg(long = "max-skip")]
        max_skip: Option<u32>,
        /// Sigma penghalusan Gaussian arah Z SDF sebelum ekstraksi, µm (0 =
        /// mati); hanya bentuk mesh, pengukuran tetap dari masker voxel.

        #[arg(long = "sdf-smooth")]
        sdf_smooth: Option<f64>,
        /// Tingkat subdivisi Loop setelah decimate (0..=2).
        #[arg(long)]
        subdivide: Option<u32>,
        /// Jangan warnai mesh dengan warna pulasan irisan.
        #[arg(long = "no-stain-colors")]
        no_stain_colors: bool,
    },
    /// Tabel pengukuran per label.
    Measure {
        /// Folder proyek.
        project: PathBuf,
        /// Keluaran JSON.
        #[arg(long)]
        json: bool,
    },
    /// Ekspor mesh, volume, kontur, laporan, atau analisis.
    Export {
        /// Folder proyek.
        project: PathBuf,
        /// Format keluaran.
        #[arg(long, value_enum)]
        format: FormatArg,
        /// Folder keluaran.
        #[arg(long)]
        out: PathBuf,
        /// Skalar untuk warna verteks GLB (mis. `margin_distance` atau nama atribut).
        #[arg(long)]
        scalar: Option<String>,
        /// Sertakan data klinis (hanya laporan).
        #[arg(long = "with-clinical")]
        with_clinical: bool,
    },
    /// Data klinis pseudonim (`clinical.json`).
    Clinical {
        #[command(subcommand)]
        command: ClinicalCommand,
    },
    /// Atribut skalar per irisan.
    Attrs {
        #[command(subcommand)]
        command: AttrsCommand,
    },
    /// Analisis spasial 3D; menulis `out/analysis.json`.
    Analyze {
        /// Folder proyek.
        project: PathBuf,
        /// Keluaran JSON.
        #[arg(long)]
        json: bool,
    },
    /// Tandai anotasi sudah ditinjau.
    Review {
        /// Folder proyek.
        project: PathBuf,
        /// Id irisan.
        #[arg(long)]
        slide: u32,
        /// Nama label.
        #[arg(long)]
        label: String,
        /// Nama atau inisial peninjau.
        #[arg(long)]
        by: String,
    },
    /// Tautkan ulang berkas slide yang dipindah, atau terima berkas yang berubah.
    Relink {
        /// Folder proyek.
        project: PathBuf,
        /// Id irisan.
        #[arg(long)]
        slide: u32,
        /// Lokasi baru berkas slide.
        #[arg(long)]
        path: Option<PathBuf>,
        /// Terima berkas yang sidik jarinya berbeda dari saat impor.
        #[arg(long)]
        accept: bool,
    },
    /// Buat data uji sintetis dengan kebenaran dasar.
    Phantom(PhantomArgs),
    /// Tautkan objek (instance) satu label antar irisan; menulis `out/tracks.json`.
    Track {
        /// Folder proyek.
        project: PathBuf,
        /// Nama label.
        #[arg(long)]
        label: String,
        /// Ambang IoU penautan (0..1; bawaan dari konfigurasi proyek, 0,2).
        #[arg(long = "min-iou")]
        min_iou: Option<f64>,
        /// Keluaran JSON.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
enum LabelsCommand {
    /// Kosongkan piksel sebuah label pada irisan (definisi label tetap);
    /// dipakai sebelum mengulang segmentasi dengan parameter lain.
    Clear {
        /// Folder proyek.
        project: PathBuf,
        /// Nama label.
        #[arg(long)]
        label: String,
        /// Id irisan dipisah koma (bawaan: semua irisan yang memuat label).
        #[arg(long, value_delimiter = ',')]
        slides: Vec<u32>,
    },
    /// Impor anotasi GeoJSON (QuPath) ke satu irisan.
    Import {
        /// Folder proyek.
        project: PathBuf,
        /// Id irisan.
        #[arg(long)]
        slide: u32,
        /// Berkas GeoJSON.
        #[arg(long)]
        geojson: PathBuf,
    },
    /// Impor masker raster PNG (piksel bukan-nol → label) ke satu irisan,
    /// hanya pada piksel yang masih kosong. Label dibuat bila belum ada.
    #[command(name = "import-mask")]
    ImportMask {
        /// Folder proyek.
        project: PathBuf,
        /// Id irisan.
        #[arg(long)]
        slide: u32,
        /// Berkas masker PNG (8/16-bit, grayscale/RGB; bukan-nol = label).
        #[arg(long)]
        mask: PathBuf,
        /// µm per piksel masker (wajib; tidak pernah ditebak).
        #[arg(long)]
        mpp: f64,
        /// Nama label tujuan.
        #[arg(long)]
        label: String,
    },
    /// Bandingkan label satu irisan dengan masker raster PNG (koefisien Dice).
    Diff {
        /// Folder proyek.
        project: PathBuf,
        /// Id irisan.
        #[arg(long)]
        slide: u32,
        /// Nama label.
        #[arg(long)]
        label: String,
        /// Berkas masker PNG pembanding.
        #[arg(long)]
        mask: PathBuf,
        /// µm per piksel masker.
        #[arg(long)]
        mpp: f64,
        /// Keluaran JSON.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
enum ClinicalCommand {
    /// Tulis `clinical.json` dari berkas JSON.
    Set {
        /// Folder proyek.
        project: PathBuf,
        /// Berkas JSON `ClinicalMeta`.
        #[arg(long)]
        json: PathBuf,
    },
    /// Hapus `clinical.json`.
    Strip {
        /// Folder proyek.
        project: PathBuf,
    },
}

#[derive(Subcommand)]
enum AttrsCommand {
    /// Hitung atribut untuk irisan terpilih.
    Compute {
        /// Folder proyek.
        project: PathBuf,
        /// Jenis atribut.
        #[arg(long, value_enum)]
        kind: AttrKindArg,
        /// Id irisan dipisah koma.
        #[arg(long, value_delimiter = ',', required = true)]
        slides: Vec<u32>,
        /// Ambang OD kromogen `t_D`.
        #[arg(long = "t-d")]
        t_d: Option<f32>,
        /// Nama atribut.
        #[arg(long)]
        name: Option<String>,
        /// Skema skor IHC.
        #[arg(long, value_enum, default_value_t = ScoringArg::Pct)]
        scoring: ScoringArg,
        /// Pola pewarnaan IHC (menandai irisan sebagai IHC dengan pola ini).
        #[arg(long, value_enum)]
        pattern: Option<PatternArg>,
        /// Nama antibodi untuk panel IHC irisan.
        #[arg(long)]
        antibody: Option<String>,
    },
}

#[derive(Args)]
struct PhantomArgs {
    /// Bentuk tumor.
    #[arg(long, value_enum, default_value_t = ShapeArg::Ellipsoid)]
    shape: ShapeArg,
    /// Folder keluaran.
    #[arg(long)]
    out: PathBuf,
    /// Jumlah irisan.
    #[arg(long, default_value_t = 60)]
    slices: usize,
    /// Benih.
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Acak nama berkas sehingga tidak mencerminkan urutan potong.
    #[arg(long)]
    shuffle: bool,
    /// Bidang tepi reseksi pada jarak D µm dari permukaan tumor.
    #[arg(long)]
    margin: Option<f64>,
    /// Jumlah fokus tumor (bersama --gap).
    #[arg(long)]
    foci: Option<usize>,
    /// Celah antar fokus, µm.
    #[arg(long)]
    gap: Option<f64>,
    /// Tambahkan silinder pembuluh menembus tumor.
    #[arg(long)]
    vessel: bool,
    /// Tambahkan kulit kapsul di sekeliling tumor.
    #[arg(long)]
    capsule: bool,
    /// Tumor menembus kapsul.
    #[arg(long)]
    breach: bool,
    /// Lapisan permukaan di z = 0 dan tumor sampai kedalaman H µm.
    #[arg(long)]
    surface: Option<f64>,
    /// Tambahkan gambar IHC sintetis dengan fraksi inti positif P (0..1).
    #[arg(long)]
    ihc: Option<f64>,
    /// Sisi gambar irisan, piksel.
    #[arg(long, default_value_t = 256)]
    size: u32,
    /// Ukuran piksel gambar irisan, µm.
    #[arg(long, default_value_t = 16.0)]
    mpp: f64,
    /// Rotasi acak per irisan dalam ± derajat ini.
    #[arg(long = "max-rot-deg", default_value_t = 10.0)]
    max_rot_deg: f64,
    /// Geser acak per irisan dalam ± fraksi sisi gambar.
    #[arg(long = "max-shift", default_value_t = 0.04)]
    max_shift: f64,
    /// Izinkan cermin acak per irisan.
    #[arg(long)]
    mirror: bool,
}

#[derive(Clone, Copy, ValueEnum)]
enum ChannelArg {
    Hematoxylin,
    Eosin,
    Dab,
}

impl From<ChannelArg> for StainChannel {
    fn from(c: ChannelArg) -> Self {
        match c {
            ChannelArg::Hematoxylin => StainChannel::Hematoxylin,
            ChannelArg::Eosin => StainChannel::Eosin,
            ChannelArg::Dab => StainChannel::Dab,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum StatusArg {
    Ok,
    Excluded,
    Damaged,
}

impl From<StatusArg> for SlideStatus {
    fn from(s: StatusArg) -> Self {
        match s {
            StatusArg::Ok => SlideStatus::Ok,
            StatusArg::Excluded => SlideStatus::Excluded,
            StatusArg::Damaged => SlideStatus::Damaged,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum ModelArg {
    Rigid,
    Similarity,
    Affine,
}

#[derive(Clone, Copy, ValueEnum)]
enum MetricArg {
    Ncc,
    Mi,
}

#[derive(Clone, Copy, ValueEnum)]
enum FormatArg {
    Glb,
    Stl,
    Obj,
    Nifti,
    Geojson,
    Report,
    Analysis,
}

#[derive(Clone, Copy, ValueEnum)]
enum AttrKindArg {
    Ihc,
    Nuclei,
    Til,
}

#[derive(Clone, Copy, ValueEnum)]
enum ScoringArg {
    Pct,
    Hscore,
    Allred,
    Tps,
    Cps,
    Intensity,
}

#[derive(Clone, Copy, ValueEnum)]
enum PatternArg {
    Nuclear,
    Membranous,
    Cytoplasmic,
}

#[derive(Clone, Copy, ValueEnum)]
enum ShapeArg {
    Ellipsoid,
    Torus,
    Branch,
}

fn parse_mpp(s: &str) -> std::result::Result<[f64; 2], String> {
    let parts: Vec<&str> = s.split(',').map(str::trim).collect();
    let parse = |t: &str| -> std::result::Result<f64, String> {
        t.parse::<f64>()
            .ok()
            .filter(|v| v.is_finite() && *v > 0.0)
            .ok_or_else(|| format!("mpp harus bilangan positif, bukan \"{t}\""))
    };
    match parts.as_slice() {
        [one] => {
            let v = parse(one)?;
            Ok([v, v])
        }
        [x, y] => Ok([parse(x)?, parse(y)?]),
        _ => Err("mpp berbentuk `0.5` atau `0.5,0.5`".to_owned()),
    }
}

/// Hasil perintah: kode keluar selain 0 untuk keadaan yang bukan galat
/// teknis (QC gagal dengan `--strict`).
enum Outcome {
    Ok,
    QcFailed(usize),
}

fn open(project: &Path) -> Result<Session> {
    let mut s = Session::open(project)
        .with_context(|| format!("tidak bisa membuka proyek {}", project.display()))?;
    s.set_clock(clock_from_env());
    warn_about_slide_files(&s);
    Ok(s)
}

/// Satu baris peringatan di stderr per irisan yang berkas slidenya berubah
/// sejak impor atau tidak ditemukan. Proyek tetap terbuka; perintah yang
/// tidak membaca piksel irisan itu tetap berjalan.
fn warn_about_slide_files(s: &Session) {
    let name = |id: SlideId| {
        s.slide(id)
            .map(|e| e.source.file_name())
            .unwrap_or_default()
    };
    for id in s.changed_slides() {
        eprintln!(
            "peringatan: berkas slide irisan {id} ({}) berubah sejak diimpor; pikselnya tidak \
             dibaca sampai ditautkan ulang atau diterima (h3d relink)",
            name(id)
        );
    }
    for id in s.missing_slides() {
        eprintln!(
            "peringatan: berkas slide irisan {id} ({}) tidak ditemukan; tautkan ulang dengan \
             h3d relink --path",
            name(id)
        );
    }
}

/// Id label bernama `name`; bila belum ada di tabel, buat label baru jenis
/// `Other` dengan warna deterministik dari id-nya.
fn label_or_create(s: &mut Session, name: &str) -> Result<h3d_core::LabelId> {
    if let Ok(id) = s.label_by_name(name) {
        return Ok(id);
    }
    let next = s
        .project()
        .next_label_id()
        .ok_or_else(|| anyhow!("tabel label penuh (maksimum 255)"))?;
    // Warna berbeda per id, tanpa acak: putar rona lewat perkalian modulo.
    let hue = (u32::from(next.0) * 47) % 256;
    let color = [
        (hue.wrapping_mul(3) % 200 + 40) as u8,
        (hue.wrapping_mul(7) % 200 + 40) as u8,
        (hue.wrapping_mul(11) % 200 + 40) as u8,
        255,
    ];
    let id = s.add_label(name, color, LabelKind::Other)?;
    eprintln!(
        "label \"{}\" belum ada; dibuat dengan id {}",
        name.trim(),
        id
    );
    Ok(id)
}

fn print_json<T: serde::Serialize>(value: &T) -> Result<()> {
    print!("{}", to_stable_json_of(value)?);
    Ok(())
}

fn mm3(um3: f64) -> f64 {
    um3 / 1.0e9
}

const SLIDE_EXTENSIONS: [&str; 10] = [
    "svs", "tif", "tiff", "ndpi", "scn", "mrxs", "bif", "png", "jpg", "jpeg",
];

fn expand_slide_paths(paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut out = Vec::with_capacity(paths.len());
    for p in paths {
        if p.is_dir() {
            if let Ok(entries) = std::fs::read_dir(p) {
                let mut found: Vec<PathBuf> = entries
                    .flatten()
                    .map(|e| e.path())
                    .filter(|ep| {
                        ep.is_file()
                            && !ep
                                .file_name()
                                .is_some_and(|n| n.to_string_lossy().starts_with('.'))
                            && ep
                                .extension()
                                .and_then(|e| e.to_str())
                                .map(|e| e.to_ascii_lowercase())
                                .is_some_and(|e| SLIDE_EXTENSIONS.contains(&e.as_str()))
                    })
                    .collect();
                if !found.is_empty() {
                    found.sort();
                    out.extend(found);
                    continue;
                }
            }
        }
        out.push(p.clone());
    }
    out
}

fn run(cli: Cli) -> Result<Outcome> {
    match cli.command {
        Command::Info { slide } => {
            let report = h3d_engine::tools::inspect_slide(&slide)
                .with_context(|| format!("tidak bisa membaca slide {}", slide.display()))?;
            print_json(&report)?;
        }
        Command::Thumb { slide, out, max } => {
            let size = h3d_engine::tools::write_thumbnail(&slide, &out, max)
                .with_context(|| format!("tidak bisa membuat thumbnail {}", slide.display()))?;
            println!(
                "Thumbnail {}×{} ditulis ke {}",
                size[0],
                size[1],
                out.display()
            );
        }
        Command::New { project } => {
            Session::create(&project)
                .with_context(|| format!("tidak bisa membuat proyek {}", project.display()))?;
            println!("Proyek dibuat: {}", project.display());
        }
        Command::Import {
            project,
            slides,
            mpp,
            spacing_um,
            embed,
        } => {
            let mut s = open(&project)?;
            let expanded = expand_slide_paths(&slides);
            let opts = ImportOptions {
                mpp,
                embed,
                spacing_um,
                ..Default::default()
            };
            let ids = match s.import_slides_with(&expanded, &opts, &NoProgress) {
                Ok(ids) => ids,
                Err(h3d_engine::EngineError::Slide(first_err)) if expanded.len() > 1 => {
                    let mut valid = Vec::new();
                    let mut skipped = Vec::new();
                    for path in &expanded {
                        match h3d_engine::tools::inspect_slide(path) {
                            Ok(_) => valid.push(path.clone()),
                            Err(e) => skipped.push((path.clone(), e)),
                        }
                    }
                    if valid.is_empty() {
                        return Err(h3d_engine::EngineError::Slide(first_err).into());
                    }
                    let ids = s.import_slides_with(&valid, &opts, &NoProgress)?;
                    for (path, err) in skipped {
                        eprintln!(
                            "peringatan: melewati berkas slide {} yang belum lengkap atau rusak: {err}",
                            path.display()
                        );
                    }
                    ids
                }
                Err(e) => return Err(e.into()),
            };
            println!("{} irisan diimpor:", ids.len());
            for id in ids {
                let e = s.slide(id)?;
                println!(
                    "  id {:>3}  urutan {:>3}  z {:>9.1} µm  mpp {:.4}×{:.4}  {}",
                    e.id,
                    e.order,
                    e.z_um,
                    e.mpp[0],
                    e.mpp[1],
                    e.source.file_name()
                );
            }
        }
        Command::SuggestOrder {
            project,
            json,
            apply,
        } => {
            let mut s = open(&project)?;
            let sug = s.suggest_order(&NoProgress)?;
            let current: Vec<SlideId> = s.slides().iter().map(|e| e.id).collect();
            let changed: Vec<SlideId> = sug
                .order
                .iter()
                .enumerate()
                .filter(|(i, id)| current.get(*i) != Some(*id))
                .map(|(_, id)| *id)
                .collect();
            if apply {
                let all: Vec<SlideId> = sug.order.clone();
                s.apply_order_suggestion(&sug, &all)?;
            }
            if json {
                let orient: Vec<serde_json::Value> = sug
                    .orient
                    .iter()
                    .map(|(id, a)| {
                        let parts = a.decompose();
                        json!({
                            "slide": id.0,
                            "rotation_deg": parts.rotation.to_degrees(),
                            "mirrored": parts.mirrored,
                        })
                    })
                    .collect();
                print_json(&json!({
                    "order": sug.order.iter().map(|i| i.0).collect::<Vec<_>>(),
                    "changed": changed.iter().map(|i| i.0).collect::<Vec<_>>(),
                    "orient": orient,
                    "confidence": sug.confidence.iter().map(|(i, c)| json!({"slide": i.0, "confidence": c})).collect::<Vec<_>>(),
                    "uncertain": sug.uncertain.iter().map(|i| i.0).collect::<Vec<_>>(),
                    "applied": apply,
                }))?;
            } else {
                println!(
                    "Urutan usulan: {}",
                    sug.order
                        .iter()
                        .map(|i| i.to_string())
                        .collect::<Vec<_>>()
                        .join(" ")
                );
                println!("Irisan yang posisinya berubah: {}", changed.len());
                if !sug.uncertain.is_empty() {
                    println!(
                        "Kurang yakin: {}",
                        sug.uncertain
                            .iter()
                            .map(|i| i.to_string())
                            .collect::<Vec<_>>()
                            .join(" ")
                    );
                }
                println!("Catatan: arah tumpukan (atas–bawah) tidak dapat ditentukan dari gambar.");
                if apply {
                    println!(
                        "Usulan diterima; {} irisan perlu didaftarkan ulang.",
                        s.needs_registration().len()
                    );
                }
            }
        }
        Command::Register {
            project,
            model,
            metric,
            reference,
            strict,
            force,
        } => {
            let mut s = open(&project)?;
            let options = RegOptions {
                model: match model {
                    ModelArg::Rigid => RegModel::Rigid,
                    ModelArg::Similarity => RegModel::Similarity,
                    ModelArg::Affine => RegModel::Affine,
                },
                metric: match metric {
                    MetricArg::Ncc => RegMetric::Ncc,
                    MetricArg::Mi => RegMetric::MutualInformation,
                },
                ..Default::default()
            };
            let sum = s.register_all_with(
                &RegisterRun {
                    options,
                    reference: reference.map(SlideId),
                    force,
                    ..Default::default()
                },
                &NoProgress,
            )?;
            println!(
                "Registrasi selesai: acuan {}, didaftarkan {}, dilewati {}, bertanda {}",
                sum.reference.map_or("-".to_owned(), |r| r.to_string()),
                sum.registered.len(),
                sum.skipped.len(),
                sum.flagged.len()
            );
            if !sum.flagged.is_empty() {
                println!(
                    "Irisan perlu diperiksa: {}",
                    sum.flagged
                        .iter()
                        .map(|i| i.to_string())
                        .collect::<Vec<_>>()
                        .join(" ")
                );
                if strict {
                    return Ok(Outcome::QcFailed(sum.flagged.len()));
                }
            }
        }
        Command::Qc { project, json } => {
            let s = open(&project)?;
            let rows = s.qc_table();
            if json {
                print_json(&rows)?;
            } else {
                println!("  id urutan  ke    NCC   Dice  rasio  tanda  metode");
                for r in rows {
                    let f = |v: Option<f32>| v.map_or("    -".to_owned(), |x| format!("{x:5.3}"));
                    println!(
                        "{:>4} {:>6} {:>3}  {}  {}  {}  {:<5}  {}",
                        r.slide,
                        r.order,
                        r.fixed.map_or("-".to_owned(), |i| i.to_string()),
                        f(r.ncc),
                        f(r.mask_dice),
                        f(r.area_ratio),
                        if r.flagged { "YA" } else { "-" },
                        if r.is_reference {
                            "acuan".to_owned()
                        } else {
                            r.method.unwrap_or_else(|| "belum terdaftar".to_owned())
                        }
                    );
                }
            }
        }
        Command::Labels {
            command:
                LabelsCommand::Import {
                    project,
                    slide,
                    geojson,
                },
        } => {
            let mut s = open(&project)?;
            let sum = s.import_geojson(SlideId(slide), &geojson)?;
            s.save()?;
            println!(
                "GeoJSON diimpor ke irisan {slide}: {} dari {} fitur ({} label tak dikenal, {} geometri dilewati)",
                sum.imported, sum.features, sum.skipped_unknown_label, sum.skipped_geometry
            );
        }
        Command::Labels {
            command:
                LabelsCommand::ImportMask {
                    project,
                    slide,
                    mask,
                    mpp,
                    label,
                },
        } => {
            let mut s = open(&project)?;
            let id = label_or_create(&mut s, &label)?;
            let sum = s.import_mask(SlideId(slide), &mask, mpp, id)?;
            s.save()?;
            println!(
                "Masker diimpor ke irisan {slide} sebagai label \"{}\": {} piksel diisi ({:.1} µm², \
                 hanya pada piksel kosong) dari {} piksel masker",
                label.trim(),
                sum.filled_px,
                sum.area_um2,
                sum.mask_px
            );
        }
        Command::Labels {
            command:
                LabelsCommand::Clear {
                    project,
                    label,
                    slides,
                },
        } => {
            let mut s = open(&project)?;
            let id = s.label_by_name(&label)?;
            let ids: Vec<SlideId> = if slides.is_empty() {
                s.slides_with_label(id)
            } else {
                slides.into_iter().map(SlideId).collect()
            };
            let mut n = 0usize;
            for &slide in &ids {
                let size = s.slide(slide)?.size_um();
                // Poligon seluas irisan dengan mode hapus: hanya
                // piksel bernilai `label` yang menjadi latar.
                let rect = [
                    [0.0, 0.0],
                    [size[0], 0.0],
                    [size[0], size[1]],
                    [0.0, size[1]],
                ];
                if s.fill_polygon(slide, id, &rect, PaintMode::Erase)? {
                    n += 1;
                }
            }
            s.save()?;
            println!("Label \"{label}\" dikosongkan pada {n} irisan");
        }
        Command::Labels {
            command:
                LabelsCommand::Diff {
                    project,
                    slide,
                    label,
                    mask,
                    mpp,
                    json,
                },
        } => {
            let mut s = open(&project)?;
            let id = s.label_by_name(&label)?;
            let dice = s.dice_against_mask(SlideId(slide), id, &mask, mpp)?;
            if json {
                print_json(&json!({
                    "slide": slide,
                    "label": label.trim(),
                    "mask": mask.display().to_string(),
                    "mpp_mask": mpp,
                    "dice": dice,
                }))?;
            } else {
                println!(
                    "Dice label \"{}\" irisan {slide} terhadap masker {}: {dice:.4}",
                    label.trim(),
                    mask.display()
                );
            }
        }
        Command::Segment {
            project,
            model,
            slides,
        } => {
            let mut s = open(&project)?;
            let ids: Vec<SlideId> = if slides.is_empty() {
                s.project()
                    .usable_slides_ordered()
                    .iter()
                    .map(|e| e.id)
                    .collect()
            } else {
                slides.into_iter().map(SlideId).collect()
            };
            s.segment_ai(&model, &ids, &NoProgress)?;
            // Menjalankan perintah ini adalah persetujuan eksplisit pengguna;
            // tetap saja label yang sudah ada tidak ditimpa.
            let n = s.accept_all_pending(PaintMode::OnlyBackground)?;
            s.save()?;
            println!(
                "Segmentasi AI: {} irisan diproses, {n} usulan diterima (hanya pada piksel kosong)",
                ids.len()
            );
        }
        Command::Classic {
            project,
            label,
            slides,
            channel,
            threshold,
            invert,
            open_um,
            close_um,
            min_area_um2,
            fill_holes,
        } => {
            let mut s = open(&project)?;
            let id = match s.label_by_name(&label) {
                Ok(id) => id,
                Err(_) => s.add_label(&label, [40, 180, 200, 255], LabelKind::Other)?,
            };
            let ids: Vec<SlideId> = if slides.is_empty() {
                s.project()
                    .usable_slides_ordered()
                    .iter()
                    .map(|e| e.id)
                    .collect()
            } else {
                slides.into_iter().map(SlideId).collect()
            };
            let params = ClassicParams {
                channel: channel.into(),
                threshold,
                invert,
                open_radius_um: open_um,
                close_radius_um: close_um,
                min_area_um2,
                fill_holes,
            };
            let mut n = 0usize;
            for &slide in &ids {
                let proposal = s.segment_classic(slide, id, &params)?;
                // Menjalankan perintah ini adalah persetujuan eksplisit
                // pengguna; label yang sudah ada tetap tidak ditimpa.
                if s.accept_proposal(slide, &proposal, PaintMode::OnlyBackground)? {
                    n += 1;
                }
                eprintln!("  irisan {slide}: selesai");
            }
            s.save()?;
            println!(
                "Segmentasi klasik \"{label}\": {} irisan diproses, {n} diterima (hanya pada piksel kosong)",
                ids.len()
            );
        }
        Command::Status {
            project,
            slide,
            status,
        } => {
            let mut s = open(&project)?;
            s.set_status(SlideId(slide), status.into())?;
            s.save()?;
            println!("Status irisan {slide} diubah");
        }
        Command::Interpolate { project, label } => {
            let mut s = open(&project)?;
            let id = s.label_by_name(&label)?;
            s.interpolate_labels(id, &NoProgress)?;
            let slides = s.pending_slides();
            let n = s.accept_all_pending(PaintMode::OnlyBackground)?;
            s.save()?;
            println!(
                "Interpolasi label \"{label}\": {n} irisan diisi ({}). Irisan hasil interpolasi tidak ikut volume Cavalieri.",
                slides
                    .iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
        Command::Reconstruct {
            project,
            sxy,
            smooth,
            decimate,
            per_object,
            split,
            split_distance,
            split_h,
            max_skip,
            sdf_smooth,
            subdivide,
            no_stain_colors,
        } => {
            let mut s = open(&project)?;
            let mut cfg = s.project().recon.clone();
            if split {
                cfg.per_object = true;
                cfg.track_split.enabled = true;
            }
            if let Some(v) = split_distance {
                cfg.track_split.min_seed_distance_um = v;
            }
            if let Some(v) = split_h {
                cfg.track_split.h_um = v;
            }
            if let Some(v) = max_skip {
                cfg.track_max_skip = v;
            }
            if let Some(v) = sxy {
                cfg.s_xy_um = v;
            }
            if let Some(v) = smooth {
                cfg.smooth_iterations = v;
            }
            if let Some(v) = decimate {
                cfg.decimate_error_factor = v;
            }
            if per_object {
                cfg.per_object = true;
            }
            if let Some(v) = sdf_smooth {
                cfg.sdf_smooth_sigma_um = v;
            }
            if let Some(v) = subdivide {
                cfg.subdivision_levels = v;
            }
            if no_stain_colors {
                cfg.stain_colors = false;
            }
            s.set_recon_config(cfg)?;

            s.save()?;
            let meshes = s.reconstruct(&NoProgress)?;
            let out = s.bundle().out_mesh_dir();
            s.export(ExportKind::Glb { scalar: None }, &out)?;
            println!("Rekonstruksi selesai: {} mesh", meshes.len());
            for m in &meshes {
                let name = s
                    .project()
                    .label(m.label)
                    .map_or_else(|| m.label.to_string(), |l| l.name.clone());
                let name = m.display_name(&name);
                println!(
                    "  {:<16} {:>8} verteks {:>8} segitiga  volume mesh {:.4} mm³",
                    name,
                    m.positions_um.len(),
                    m.triangle_count(),
                    mm3(m.signed_volume_um3())
                );
            }
            println!("Mesh ditulis ke {}", out.join("mesh.glb").display());
            // 8.7.1: volume label di kisi tumpukan untuk validasi silang.
            let nii = s.write_label_volume()?;
            println!("Volume label ditulis ke {}", nii.display());
        }
        Command::Measure { project, json } => {
            let s = open(&project)?;
            let report = s.measure()?;
            if json {
                print_json(&report)?;
            } else {
                println!("{}", h3d_core::RESEARCH_ONLY_NOTICE);
                println!(
                    "Label              Cavalieri mm³     Mesh mm³   Irisan (anot/interp/tinjau)"
                );
                for l in &report.labels {
                    println!(
                        "{:<18} {:>13.4} {:>12} {:>9} / {} / {}",
                        l.name,
                        mm3(l.volume_cavalieri_um3),
                        l.volume_mesh_um3
                            .map_or("-".to_owned(), |v| format!("{:.4}", mm3(v))),
                        l.n_slices_annotated,
                        l.n_slices_interpolated,
                        l.n_slices_reviewed
                    );
                }
                if !report.objects.is_empty() {
                    println!(
                        "Objek               Mesh mm³    Luas mm²   Diameter mm  Irisan  Dijembatani"
                    );
                    for ob in &report.objects {
                        println!(
                            "{:<18} {:>11.4} {:>11.4} {:>13.3} {:>7} {:>12}",
                            ob.name,
                            mm3(ob.volume_mesh_um3),
                            ob.surface_area_um2 / 1.0e6,
                            ob.max_diameter_um / 1000.0,
                            ob.n_slices.map_or("-".to_owned(), |n| n.to_string()),
                            ob.n_bridged_slices
                                .map_or("-".to_owned(), |n| n.to_string())
                        );
                    }
                    if let Some(h) = report.objects.iter().find_map(|o| o.track_params_hash) {
                        println!("Hash parameter penautan: {h:016x}");
                    }
                }
                println!(
                    "Ketidakpastian ukuran 3D: ± {:.3} mm",
                    report.dimension_uncertainty_um / 1000.0
                );
                for w in report.all_warnings() {
                    println!("Peringatan: {}", h3d_engine::warning_text(&w));
                }
            }
        }
        Command::Export {
            project,
            format,
            out,
            scalar,
            with_clinical,
        } => {
            let s = open(&project)?;
            let kind = match format {
                FormatArg::Glb => ExportKind::Glb { scalar },
                FormatArg::Stl => ExportKind::Stl,
                FormatArg::Obj => ExportKind::Obj,
                FormatArg::Nifti => ExportKind::Nifti,
                FormatArg::Geojson => ExportKind::GeoJson,
                FormatArg::Report => ExportKind::Report { with_clinical },
                FormatArg::Analysis => ExportKind::Analysis,
            };
            let files = s.export_files(kind, &out)?;
            println!("{} berkas ditulis ke {}:", files.len(), out.display());
            for f in files {
                println!(
                    "  {}",
                    f.file_name()
                        .map_or_else(String::new, |n| n.to_string_lossy().into_owned())
                );
            }
        }
        Command::Clinical { command } => match command {
            ClinicalCommand::Set { project, json } => {
                let mut s = open(&project)?;
                let text = std::fs::read_to_string(&json)
                    .with_context(|| format!("tidak bisa membaca {}", json.display()))?;
                // Nama field diperiksa dulu: salah ketik tidak boleh diam-diam
                // menghasilkan data klinis kosong.
                let value: serde_json::Value = serde_json::from_str(&text)
                    .map_err(|e| anyhow!("berkas klinis {} tidak sah: {e}", json.display()))?;
                let obj = value.as_object().ok_or_else(|| {
                    anyhow!("berkas klinis {} harus berupa objek JSON", json.display())
                })?;
                if let Some(unknown) = obj
                    .keys()
                    .find(|k| !ClinicalMeta::FIELD_NAMES.contains(&k.as_str()))
                {
                    bail!(
                        "berkas klinis {} memuat field tak dikenal \"{unknown}\"",
                        json.display()
                    );
                }
                let meta: ClinicalMeta = serde_json::from_value(value)
                    .map_err(|e| anyhow!("berkas klinis {} tidak sah: {e}", json.display()))?;
                s.set_clinical(Some(meta))?;
                // Isi data klinis sengaja tidak dicetak.
                println!("clinical.json ditulis.");
            }
            ClinicalCommand::Strip { project } => {
                let mut s = open(&project)?;
                s.set_clinical(None)?;
                println!("clinical.json dihapus.");
            }
        },
        Command::Attrs {
            command:
                AttrsCommand::Compute {
                    project,
                    kind,
                    slides,
                    t_d,
                    name,
                    scoring,
                    pattern,
                    antibody,
                },
        } => {
            let mut s = open(&project)?;
            let ids: Vec<SlideId> = slides.into_iter().map(SlideId).collect();
            let scoring = match scoring {
                ScoringArg::Pct => IhcScoring::PercentPositive,
                ScoringArg::Hscore => IhcScoring::HScore,
                ScoringArg::Allred => IhcScoring::Allred,
                ScoringArg::Tps => IhcScoring::Tps,
                ScoringArg::Cps => IhcScoring::Cps,
                ScoringArg::Intensity => IhcScoring::Intensity03,
            };
            let attr_kind = match kind {
                AttrKindArg::Ihc => AttrKind::IhcScore {
                    slide: *ids
                        .first()
                        .ok_or_else(|| anyhow!("tidak ada irisan yang dipilih"))?,
                    scoring,
                },
                AttrKindArg::Nuclei => AttrKind::NucleiDensity,
                AttrKindArg::Til => AttrKind::TilDensity,
            };
            // `--kind ihc|til` menandai irisan sebagai IHC (panel dari flag).
            if matches!(kind, AttrKindArg::Ihc | AttrKindArg::Til) {
                for id in &ids {
                    let e = s.slide(*id)?.clone();
                    let mut panel = e.ihc.clone().unwrap_or_default();
                    if let Some(a) = &antibody {
                        panel.antibody = a.clone();
                    }
                    if let Some(p) = pattern {
                        panel.pattern = match p {
                            PatternArg::Nuclear => StainPattern::Nuclear,
                            PatternArg::Membranous => StainPattern::Membranous,
                            PatternArg::Cytoplasmic => StainPattern::Cytoplasmic,
                        };
                    }
                    panel.scoring = scoring;
                    if e.stain != Stain::Ihc || e.ihc.as_ref() != Some(&panel) {
                        s.set_stain(*id, Stain::Ihc, Some(panel))?;
                    }
                }
            }
            let mut params = AttrParams::default();
            if let Some(t) = t_d {
                if !(t.is_finite() && t > 0.0) {
                    bail!("--t-d harus bilangan positif");
                }
                params.thresholds.t_d = t;
            }
            let def = AttrDef {
                name: name.unwrap_or_default(),
                kind: attr_kind,
                ..Default::default()
            };
            let id = s.compute_attr_with(def, &ids, &params, &NoProgress)?;
            let report = s.measure()?;
            let saved = s
                .project()
                .attrs
                .iter()
                .find(|a| a.id == id)
                .ok_or_else(|| anyhow!("atribut tidak tersimpan"))?
                .clone();
            println!("Atribut \"{}\" (id {}) dihitung:", saved.name, saved.id);
            if let Some(a) = report.attrs.iter().find(|a| a.name == saved.name) {
                println!(
                    "  metode {}  hash parameter {}",
                    a.method,
                    a.params_hash
                        .map_or("-".to_owned(), |h| format!("{h:016x}"))
                );
                for row in &a.per_slide {
                    println!(
                        "  irisan {:>3}  rata-rata {:.2} {}",
                        row.slide, row.mean, a.unit
                    );
                }
            }
            for n in s.attr_notes() {
                println!("Catatan: {n}");
            }
        }
        Command::Analyze { project, json } => {
            let mut s = open(&project)?;
            let a = s.analyze_spatial(&NoProgress)?;
            let path = s.bundle().analysis_json();
            if json {
                let text = std::fs::read_to_string(&path)
                    .with_context(|| format!("tidak bisa membaca {}", path.display()))?;
                print!("{text}");
            } else {
                println!("{}", h3d_core::RESEARCH_ONLY_NOTICE);
                println!("Volume tumor (voxel): {:.4} mm³", mm3(a.tumor_volume_um3));
                if let Some(m) = &a.margin {
                    println!(
                        "Jarak minimum ke tepi reseksi: {} arah {}",
                        h3d_engine::fmt_uncertainty(&m.d_min),
                        h3d_engine::direction_text(m.direction, m.direction_name)
                    );
                }
                if let Some(f) = &a.foci {
                    println!("Jumlah fokus: {}", f.n);
                }
                for c in &a.contacts {
                    println!(
                        "Kontak {}: {} ({:.4} mm²)",
                        c.name,
                        if c.result.any { "ya" } else { "tidak" },
                        c.result.area_mm2
                    );
                }
                if let Some(c) = &a.capsule {
                    println!(
                        "Penetrasi kapsul: {}",
                        if c.breached { "ya" } else { "tidak" }
                    );
                }
                if let Some(d) = &a.depth {
                    println!(
                        "Kedalaman invasi: {}",
                        h3d_engine::fmt_uncertainty(&d.depth)
                    );
                }
                if let Some(sz) = &a.size {
                    println!(
                        "Diameter maksimum: {}",
                        h3d_engine::fmt_uncertainty(&sz.max_diameter_um)
                    );
                }
                for w in &a.warnings {
                    println!("Peringatan: {}", h3d_engine::warning_text(w));
                }
                println!("Hasil lengkap ditulis ke {}", path.display());
            }
        }
        Command::Review {
            project,
            slide,
            label,
            by,
        } => {
            let mut s = open(&project)?;
            let id = s.label_by_name(&label)?;
            s.mark_reviewed(SlideId(slide), id, &by)?;
            println!("Anotasi \"{label}\" pada irisan {slide} ditandai sudah ditinjau.");
        }
        Command::Relink {
            project,
            slide,
            path,
            accept,
        } => {
            if path.is_none() && !accept {
                bail!("relink butuh --path BERKAS dan/atau --accept");
            }
            let mut s = open(&project)?;
            let id = SlideId(slide);
            let mut matches = true;
            if let Some(p) = &path {
                matches = s.relink_slide(id, p)?;
                println!(
                    "Irisan {slide} ditautkan ke {}{}",
                    p.display(),
                    if matches {
                        " (sidik jari cocok dengan saat impor)"
                    } else {
                        " (sidik jari BERBEDA dari saat impor)"
                    }
                );
            }
            if accept {
                s.accept_changed_slide(id)?;
                println!("Berkas irisan {slide} diterima; sidik jari tersimpan diperbarui.");
            } else if !matches {
                println!(
                    "Piksel irisan {slide} belum dibaca: jalankan lagi dengan --accept bila ini \
                     memang berkas irisan tersebut."
                );
            }
        }
        Command::Phantom(a) => {
            let foci = match (a.foci, a.gap) {
                (Some(n), g) => Some((n, g.unwrap_or(200.0))),
                (None, Some(_)) => bail!("--gap butuh --foci"),
                (None, None) => None,
            };
            let opts = PhantomOptions {
                shape: match a.shape {
                    ShapeArg::Ellipsoid => PhantomShapeKind::Ellipsoid,
                    ShapeArg::Torus => PhantomShapeKind::Torus,
                    ShapeArg::Branch => PhantomShapeKind::Branch,
                },
                slices: a.slices,
                seed: a.seed,
                shuffle: a.shuffle,
                margin_um: a.margin,
                foci,
                vessel: a.vessel,
                capsule: a.capsule,
                breach: a.breach,
                surface_um: a.surface,
                ihc: a.ihc,
                size_px: a.size,
                mpp: a.mpp,
                spacing_um: None,
                max_rotation_deg: a.max_rot_deg,
                max_shift_frac: a.max_shift,
                mirror: a.mirror,
            };
            let m = write_phantom(&a.out, &opts)?;
            println!(
                "Fantom ditulis ke {}: {} irisan {}×{} px, mpp {} µm, jarak irisan {:.3} µm",
                a.out.display(),
                m.slices.len(),
                m.size_px[0],
                m.size_px[1],
                m.mpp,
                m.spacing_um
            );
            match m.analytic.volume_um3 {
                Some(v) => println!(
                    "Volume tumor analitik: {:.4} mm³ (lihat truth.json)",
                    mm3(v)
                ),
                None => println!("Bentuk ini tidak punya volume analitik (lihat truth.json)."),
            }
            println!(
                "Lanjutkan dengan: h3d import PROYEK.h3d {}/slice_*.png --mpp {} --spacing-um {:.6}",
                a.out.display(),
                m.mpp,
                m.spacing_um
            );
        }
        Command::Track {
            project,
            label,
            min_iou,
            json,
        } => {
            if let Some(v) = min_iou {
                if !(v.is_finite() && (0.0..=1.0).contains(&v)) {
                    bail!("--min-iou harus bilangan di antara 0 dan 1");
                }
            }
            let mut s = open(&project)?;
            let id = s.label_by_name(&label)?;
            let set = s.track(id, min_iou, &NoProgress)?;
            if json {
                print_json(&set)?;
            } else {
                println!(
                    "Penautan label \"{label}\": {} objek, {} peristiwa (IoU ≥ {:.2}, celah maks {:.0} µm, s_xy {} µm)",
                    set.tracks.len(),
                    set.event_count(),
                    set.min_iou,
                    set.max_gap_um,
                    set.s_xy_um
                );
                println!(
                    "  id  irisan   z awal (µm)   z akhir (µm)   luas rata-rata (mm²)  peristiwa"
                );
                for t in &set.tracks {
                    let (z0, z1) = t.z_range_um().unwrap_or((f64::NAN, f64::NAN));
                    let mean_area = if t.members.is_empty() {
                        0.0
                    } else {
                        t.members.iter().map(|m| m.area_um2).sum::<f64>() / t.members.len() as f64
                    };
                    let events: Vec<String> = t
                        .events
                        .iter()
                        .map(|e| match e {
                            h3d_engine::TrackEvent::Split { children, .. } => format!(
                                "pecah→{}",
                                children
                                    .iter()
                                    .map(|c| c.to_string())
                                    .collect::<Vec<_>>()
                                    .join(",")
                            ),
                            h3d_engine::TrackEvent::Merge { parents, .. } => format!(
                                "gabung←{}",
                                parents
                                    .iter()
                                    .map(|c| c.to_string())
                                    .collect::<Vec<_>>()
                                    .join(",")
                            ),
                            h3d_engine::TrackEvent::Bridged { skipped, .. } => {
                                format!("jembatan({} irisan)", skipped.len())
                            }
                        })
                        .collect();
                    println!(
                        "{:>4} {:>7} {:>13.0} {:>14.0} {:>22.4}  {}",
                        t.instance_id,
                        t.members.len(),
                        z0,
                        z1,
                        mean_area / 1.0e6,
                        if events.is_empty() {
                            "-".to_owned()
                        } else {
                            events.join(" ")
                        }
                    );
                }
                for w in &set.warnings {
                    println!("Peringatan: {}", h3d_engine::warning_text(w));
                }
                println!("Hasil ditulis ke {}", s.bundle().tracks_json().display());
            }
        }
    }
    Ok(Outcome::Ok)
}

/// Satu baris pesan dari rantai galat (tanpa baris baru).
fn one_line(e: &anyhow::Error) -> String {
    format!("{e:#}")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() -> ExitCode {
    let cli = match Cli::try_parse() {
        Ok(c) => c,
        Err(e) => {
            use clap::error::ErrorKind;
            return match e.kind() {
                ErrorKind::DisplayHelp | ErrorKind::DisplayVersion => {
                    let _ = e.print();
                    ExitCode::SUCCESS
                }
                _ => {
                    // Ringkas pesan clap menjadi satu baris.
                    let text = e.render().to_string();
                    let first = text
                        .lines()
                        .next()
                        .unwrap_or_default()
                        .trim_start_matches("error: ")
                        .to_owned();
                    eprintln!(
                        "galat: argumen tidak sah: {first} (jalankan `h3d --help` untuk bantuan)"
                    );
                    ExitCode::from(EXIT_USAGE)
                }
            };
        }
    };
    let level = match cli.verbose {
        0 => log::LevelFilter::Warn,
        1 => log::LevelFilter::Info,
        _ => log::LevelFilter::Debug,
    };
    let _ = env_logger::Builder::new().filter_level(level).try_init();
    match run(cli) {
        Ok(Outcome::Ok) => ExitCode::SUCCESS,
        Ok(Outcome::QcFailed(n)) => {
            eprintln!("galat: {n} irisan gagal QC registrasi (--strict aktif)");
            ExitCode::from(EXIT_QC)
        }
        Err(e) => {
            eprintln!("galat: {}", one_line(&e));
            ExitCode::from(EXIT_DATA)
        }
    }
}
