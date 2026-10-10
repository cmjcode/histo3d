//! Entri irisan dan metadata per irisan (8.1, 8.2, 8.12.3).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::affine::Affine2;
use crate::ids::SlideId;
use crate::transform::SlideTransform;

/// Rujukan ke berkas gambar irisan.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SlideSourceRef {
    /// Berkas di luar proyek (desktop). `path` relatif terhadap folder proyek
    /// bila memungkinkan.
    External {
        /// Lokasi berkas.
        path: PathBuf,
        /// Ukuran berkas saat impor.
        size_bytes: u64,
        /// Hash ukuran + 1 MB pertama + 1 MB terakhir.
        fingerprint: u64,
    },
    /// Berkas disalin ke dalam bundel proyek (wajib di iPad).
    Embedded {
        /// Path relatif terhadap folder proyek, mis. `slides/0007.svs`.
        rel_path: PathBuf,
    },
}

impl SlideSourceRef {
    /// Nama berkas untuk tampilan.
    pub fn file_name(&self) -> String {
        let p = match self {
            SlideSourceRef::External { path, .. } => path,
            SlideSourceRef::Embedded { rel_path } => rel_path,
        };
        p.file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default()
    }
}

/// Status irisan dalam tumpukan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SlideStatus {
    /// Dipakai.
    #[default]
    Ok,
    /// Dikecualikan pengguna.
    Excluded,
    /// Rusak; dilompati registrasi.
    Damaged,
}

/// Pewarnaan irisan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Stain {
    /// Hematoksilin-eosin.
    #[default]
    He,
    /// Imunohistokimia.
    Ihc,
    /// Lainnya.
    Other,
}

/// Asal urutan irisan (8.4.7).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum OrderSource {
    /// Dari nama berkas saat impor.
    #[default]
    FileName,
    /// Diatur tangan.
    Manual,
    /// Usulan otomatis yang diterima.
    Suggested,
}

/// Catatan registrasi berpasangan satu irisan terhadap `fixed`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RegRecord {
    /// Irisan tempat irisan ini didaftarkan.
    pub fixed: SlideId,
    /// Nama metode, mis. "builtin-ecc-affine", "manual-landmarks".
    pub method: String,
    /// `P_{i→fixed}`: ruang U irisan ini → ruang U `fixed`, µm.
    pub pairwise: Affine2,
    /// NCC bermasker setelah registrasi.
    pub ncc: f32,
    /// Dice masker jaringan setelah registrasi.
    pub mask_dice: f32,
    /// `|det|` bagian linear.
    pub area_ratio: f32,
    /// Gagal ambang QC (8.4.5).
    pub flagged: bool,
    /// Hasil koreksi manual.
    pub manual: bool,
}

/// Metadata pra-analitik per irisan (8.12.3).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct PreanalyticMeta {
    /// Blok parafin asal.
    pub block_id: Option<String>,
    /// Ketebalan potong, µm (biasanya 3–5). Metadata saja; tidak dipakai geometri (5.5).
    pub section_thickness_um: Option<f64>,
    /// Lama fiksasi, jam.
    pub fixation_hours: Option<f64>,
    /// Fiksatif, mis. "NBF 10%".
    pub fixative: Option<String>,
    /// Bulan pewarnaan, "YYYY-MM".
    pub stained_month: Option<String>,
    /// Pemindai, dari metadata slide bila ada.
    pub scanner: Option<String>,
    /// Objektif, mis. "20x".
    pub objective: Option<String>,
}

/// Kromogen IHC.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Chromogen {
    /// Diaminobenzidin (cokelat).
    #[default]
    Dab,
    /// AEC (merah).
    Aec,
    /// Fast Red.
    FastRed,
    /// Lainnya.
    Other,
}

/// Pola pewarnaan IHC.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum StainPattern {
    /// Inti.
    #[default]
    Nuclear,
    /// Membran.
    Membranous,
    /// Sitoplasma.
    Cytoplasmic,
}

/// Skema skor IHC.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum IhcScoring {
    /// Persen sel positif (Ki-67, ER/PR, CD8).
    #[default]
    PercentPositive,
    /// H-score 0–300.
    HScore,
    /// Allred: proporsi 0–5 + intensitas 0–3.
    Allred,
    /// PD-L1 sel tumor.
    Tps,
    /// PD-L1 (tumor + imun) / tumor × 100.
    Cps,
    /// HER2 gaya 0/1+/2+/3+.
    Intensity03,
}

/// Panel IHC per irisan (8.12.3); wajib bila `stain == Ihc`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct IhcPanel {
    /// Antibodi, mis. "Ki-67".
    pub antibody: String,
    /// Klon, mis. "MIB-1".
    pub clone: Option<String>,
    /// Kromogen.
    pub chromogen: Chromogen,
    /// Pola pewarnaan.
    pub pattern: StainPattern,
    /// Skema skor.
    pub scoring: IhcScoring,
    /// Vektor OD pewarna hasil kalibrasi (H, kromogen, sisa); `None` = bawaan Ruifrok.
    pub od_vectors: Option<[[f32; 3]; 3]>,
}

impl Default for IhcPanel {
    fn default() -> Self {
        Self {
            antibody: String::new(),
            clone: None,
            chromogen: Chromogen::Dab,
            pattern: StainPattern::Nuclear,
            scoring: IhcScoring::PercentPositive,
            od_vectors: None,
        }
    }
}

/// Satu irisan dalam proyek.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SlideEntry {
    /// Pengenal.
    pub id: SlideId,
    /// Sumber berkas.
    pub source: SlideSourceRef,
    /// Urutan dalam tumpukan (0 = pertama).
    pub order: u32,
    /// Asal urutan.
    pub order_source: OrderSource,
    /// Usulan orientasi yang diterima (8.4.7): U irisan → U irisan acuan,
    /// hanya dipakai sebagai `init` registrasi dan thumbnail; bukan `T_i`.
    pub orient_hint: Option<Affine2>,
    /// Posisi Z, µm. Bawaan `order × jarak_irisan`.
    pub z_um: f64,
    /// Benar bila `z_um` masih nilai bawaan (belum diisi tangan).
    pub z_is_default: bool,
    /// Mikrometer per piksel level 0, `[x, y]`. WAJIB terisi; tidak ditebak.
    pub mpp: [f64; 2],
    /// Ukuran level 0 dalam piksel.
    pub size_l0: [u32; 2],
    /// Pewarnaan.
    pub stain: Stain,
    /// Status.
    pub status: SlideStatus,
    /// `T_i` (U → W).
    pub transform: SlideTransform,
    /// Catatan registrasi berpasangan.
    pub reg: Option<RegRecord>,
    /// Pra-analitik.
    pub preanalytic: PreanalyticMeta,
    /// Panel IHC; wajib bila `stain == Ihc`.
    pub ihc: Option<IhcPanel>,
}

impl Default for SlideEntry {
    fn default() -> Self {
        Self {
            id: SlideId(0),
            source: SlideSourceRef::Embedded {
                rel_path: PathBuf::new(),
            },
            order: 0,
            order_source: OrderSource::FileName,
            orient_hint: None,
            z_um: 0.0,
            z_is_default: true,
            mpp: [0.0, 0.0],
            size_l0: [0, 0],
            stain: Stain::He,
            status: SlideStatus::Ok,
            transform: SlideTransform::Identity,
            reg: None,
            preanalytic: PreanalyticMeta::default(),
            ihc: None,
        }
    }
}

impl SlideEntry {
    /// Ukuran fisik level 0 dalam µm `[w, h]`.
    pub fn size_um(&self) -> [f64; 2] {
        [
            self.size_l0[0] as f64 * self.mpp[0],
            self.size_l0[1] as f64 * self.mpp[1],
        ]
    }

    /// Benar bila irisan ikut dipakai (`Ok`).
    pub fn is_usable(&self) -> bool {
        self.status == SlideStatus::Ok
    }

    /// Benar bila mpp sah (positif dan finit di kedua sumbu).
    pub fn has_valid_mpp(&self) -> bool {
        self.mpp.iter().all(|m| *m > 0.0 && m.is_finite())
    }
}
