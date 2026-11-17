//! Segmentasi kelenjar/struktur jaringan 2D berpandu lumen (histo3d.md
//! 8.5.3, ADR 0004): satu wilayah per kelenjar (cincin epitel **beserta**
//! lumennya), sarang tumor padat, atau pembuluh, sebagai satuan objek untuk
//! penautan antar irisan dan rekonstruksi 3D. Berbeda dari deteksi inti
//! (`nuclei.rs`, ADR 0003) yang satuannya sel.
//!
//! # Alur (semua pada skala kerja `mpp`, disarankan 1–2 µm/px)
//! 1. **Pewarna.** Vektor H/E per slide diestimasi Macenko
//!    ([`estimate_stains_macenko`]); bila gagal, Ruifrok tetap.
//! 2. **Peta epitel.** Kanal H dihaluskan Gauss `epithelium_sigma_um` lalu
//!    diambang (Otsu di dalam jaringan, atau nilai tetap): pada skala ini
//!    inti melebur menjadi tekstur "epitel padat".
//! 3. **Lumen**, dua sumber. (a) **Lubang cincin epitel tertutup**: peta
//!    epitel ditutup `close_radius_um`, lubangnya berluas
//!    `min_lumen_area_um2..=max_lumen_area_um2` menjadi lumen apa pun
//!    isinya (sekresi merah muda, sel lepas), sehingga kelenjar berisi
//!    sekresi tidak terbelah; uji rasio cincin di bawah tetap berlaku agar
//!    lingkaran serat/debu tipis di kaca gugur. (b) **Piksel terang** (OD total di bawah
//!    `lumen_od_max` dan ambang Otsu latar) di dalam jaringan, untuk cincin
//!    terbuka/tipis; tiap kandidat diuji **rasio cincin**: bagian cincin
//!    selebar `ring_width_um` di sekelilingnya yang berupa epitel harus
//!    ≥ `min_ring_fraction`, dan epitel cincin harus cukup tebal
//!    (`min_ring_thickness_um`). Vakuola lemak (sekat tipis) dan celah
//!    stroma gugur di sini.
//! 4. **Masker kelenjar** = epitel ∪ lumen lolos, dibuka `open_radius_um`
//!    (memutus jembatan tipis), ditutup `close_radius_um` (menyambung
//!    cincin), lubang kecil `< min_lumen_area_um2` diisi.
//! 5. **Pemisahan.** Komponen yang memuat lumen dibagi dengan watershed
//!    berpenanda ([`split_on_field`]) di atas medan `−jarak ke lumen
//!    terdekat`: tiap lumen menjadi penanda dan epitel jatuh ke lumen
//!    terdekatnya; dua lumen yang lembah di antaranya lebih dangkal dari
//!    `lumen_valley_um` (kribriform) tetap satu objek. Komponen tanpa lumen
//!    (sarang padat) dibelah dengan watershed EDT ([`split_touching`],
//!    `solid_seed_distance_um`, `solid_valley_um`). Keping lebih kecil dari
//!    `min_area_um2` digabung ke tetangga terluas, sisanya dibuang; objek
//!    tanpa lumen yang lebih tipis dari `min_solid_thickness_um` (serpihan
//!    tepi, lipatan) dan objek yang sebagian besar abu-abu/hitam tak
//!    berwarna (`max_gray_fraction`; serat debu, tinta di kaca) dibuang.
//! 6. **Bentuk star-convex** (opsional, `star_hull`). Tiap objek diganti
//!    oleh *star hull*-nya dari pusat (piksel EDT maksimum): jari-jari
//!    terjauh tepi per sudut (2°), dihaluskan Gauss melingkar
//!    `smooth_sigma_deg`, dirasterkan kembali tanpa menimpa objek lain dan
//!    tanpa keluar jaringan. Ini "cembung yang masih mengikuti bentuk
//!    nyata": cekungan radial terisi, lekuk besar (bentuk U) ikut terisi
//!    menjadi satu badan, tetapi dua kelenjar bertetangga tidak pernah
//!    disatukan. Soliditas bentuk mentah tetap dilaporkan.
//!
//! Objek dinomori `1..=n` menurut urutan raster piksel pertamanya;
//! deterministik (tanpa acak, paralel hanya di dalam primitif yang sudah
//! deterministik). Semua ukuran µm; sentroid memakai pusat piksel `+0,5`.
//! Jumlah dilaporkan dengan rentang (`n_min`, `n_max`) dari varian saringan
//! luas (×1,25) dan rasio cincin (±0,15) tanpa menjalankan ulang (aturan
//! keras 18).

use ndarray::{s, Array2};
use serde::{Deserialize, Serialize};

use crate::color::{
    deconvolve, estimate_stains_macenko, od_total, percentile_stretch, StainMatrix,
};
use crate::components::{label_components, merge_small_regions};
use crate::edt::{edt_2d_um, edt_squared_2d};
use crate::error::ImgError;
use crate::filters::gaussian_blur_f32;
use crate::morphology::{close, dilate_disc, fill_holes, open};
use crate::palette::color_distance;
use crate::threshold::{otsu, otsu_masked};
use crate::tissue::tissue_mask;
use crate::watershed::{split_on_field, split_touching};

/// Cara menentukan vektor pewarna.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum StainEstimate {
    /// Estimasi per slide (Macenko dkk. 2009); jatuh ke Ruifrok bila gagal.
    #[default]
    Macenko,
    /// Vektor Ruifrok & Johnston tetap.
    Ruifrok,
}

/// Sumber peta epitel.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum EpitheliumSource {
    /// Kanal hematoksilin hasil dekonvolusi, diambang (Otsu atau tetap).
    #[default]
    Hematoxylin,
    /// Warna palet terpilih (CIELAB) dengan toleransi ΔE: piksel yang
    /// jaraknya ke salah satu `picks` ≤ `tolerance` adalah epitel. Mode
    /// "objek dari warna": pengguna memilih kotak warna dan satu toleransi.
    Color {
        /// Warna terpilih (Lab).
        picks: Vec<[f32; 3]>,
        /// Toleransi ΔE*ab.
        tolerance: f32,
    },
}

/// Parameter segmentasi kelenjar (satuan µm).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GlandParams {
    /// Skala kerja, µm/px (bawaan 2). Dipakai pemanggil untuk membaca
    /// slide; fungsi segmentasi menerima `mpp` citra yang diberikan.
    pub mpp_um: f64,
    /// Sumber vektor pewarna.
    pub stain: StainEstimate,
    /// Sumber peta epitel (hematoksilin atau warna palet).
    pub epithelium: EpitheliumSource,
    /// Sigma Gauss kanal H untuk peta epitel, µm (bawaan 4).
    pub epithelium_sigma_um: f64,
    /// Ambang peta epitel pada kanal H teregang 0..1; `None` = Otsu di
    /// dalam jaringan.
    pub epithelium_threshold: Option<f32>,
    /// OD total maksimum piksel lumen (bawaan 0,15; kaca ≈ 0, stroma pucat
    /// ≳ 0,3). Ambang efektif = min(Otsu latar, nilai ini).
    pub lumen_od_max: f64,
    /// Luas lumen minimum, µm² (bawaan 50); lubang lebih kecil diisi.
    pub min_lumen_area_um2: f64,
    /// Luas maksimum lubang cincin epitel yang dianggap lumen, µm² (bawaan
    /// 250 000 ≈ lingkaran Ø 560 µm); kantong stroma lebih besar yang
    /// terkurung rantai kelenjar tidak dijadikan lumen.
    pub max_lumen_area_um2: f64,
    /// Lebar cincin uji di sekeliling lumen, µm (bawaan 10).
    pub ring_width_um: f64,
    /// Bagian minimum cincin yang berupa epitel agar lumen dianggap milik
    /// kelenjar (bawaan 0,5).
    pub min_ring_fraction: f64,
    /// Ketebalan minimum epitel cincin, µm (bawaan 5): epitel dierosi
    /// `max(ketebalan/2, 1 px)` dan sisanya harus menutupi ≥ `MIN_THICK_RING_FRACTION`
    /// cincin. Sekat lemak dan membran tipis (2–3 µm) gugur, epitel satu
    /// lapis (≥ 6 µm) lolos.
    pub min_ring_thickness_um: f64,
    /// Radius pembukaan masker kelenjar, µm (bawaan 2; 0 = lewati).
    pub open_radius_um: f64,
    /// Radius penutupan masker kelenjar, µm (bawaan 5; 0 = lewati).
    pub close_radius_um: f64,
    /// Luas objek minimum, µm² (bawaan 300).
    pub min_area_um2: f64,
    /// Ketebalan minimum (2 × EDT maksimum) objek **tanpa lumen**, µm
    /// (bawaan 8): serpihan tipis di tepi jaringan atau lipatan dibuang.
    pub min_solid_thickness_um: f64,
    /// Kedalaman lembah minimum antara dua lumen agar menjadi dua objek,
    /// µm (bawaan 4).
    pub lumen_valley_um: f64,
    /// Sarang padat: jarak minimum antar puncak EDT, µm (bawaan 40).
    pub solid_seed_distance_um: f64,
    /// Sarang padat: kedalaman lembah EDT minimum, µm (bawaan 8).
    pub solid_valley_um: f64,
    /// Ganti bentuk tiap objek dengan star hull halusnya (bawaan benar).
    pub star_hull: bool,
    /// Sigma penghalusan jari-jari melingkar, derajat (bawaan 6; 0 = tanpa
    /// penghalusan, hanya star hull).
    pub smooth_sigma_deg: f64,
    /// Bagian maksimum piksel non-lumen objek yang **abu-abu/hitam** (kroma
    /// RGB < 30 dan intensitas < 180; bawaan 0,5). Serat debu, tinta, dan
    /// artefak gelap tak berwarna di kaca bukan jaringan H&E dan dibuang.
    pub max_gray_fraction: f64,
    /// Batas pembengkakan star hull: bila luas hull > faktor ini × luas
    /// mentah (bawaan 1,5), bentuk mentah dipertahankan. Mencegah objek tipis
    /// memanjang/melengkung menjadi kipas atau tetesan besar.
    pub star_hull_max_area_factor: f64,
}

impl Default for GlandParams {
    fn default() -> Self {
        Self {
            mpp_um: 2.0,
            stain: StainEstimate::Macenko,
            epithelium: EpitheliumSource::Hematoxylin,
            epithelium_sigma_um: 4.0,
            epithelium_threshold: None,
            lumen_od_max: 0.15,
            min_lumen_area_um2: 50.0,
            max_lumen_area_um2: 250_000.0,
            ring_width_um: 10.0,
            min_ring_fraction: 0.5,
            min_ring_thickness_um: 5.0,
            open_radius_um: 2.0,
            close_radius_um: 5.0,
            min_area_um2: 300.0,
            min_solid_thickness_um: 8.0,
            lumen_valley_um: 4.0,
            solid_seed_distance_um: 40.0,
            solid_valley_um: 8.0,
            star_hull: true,
            smooth_sigma_deg: 6.0,
            max_gray_fraction: 0.5,
            star_hull_max_area_factor: 1.5,
        }
    }
}

/// Selisih rasio cincin untuk rentang ketidakpastian jumlah.
pub const RING_FRACTION_DELTA: f64 = 0.15;
/// Faktor luas minimum untuk batas bawah jumlah.
pub const MIN_AREA_STRICT_FACTOR: f64 = 1.25;
/// Jumlah bin sudut star hull (2° per bin).
const STAR_BINS: usize = 180;
/// Bagian minimum cincin yang masih epitel setelah erosi ketebalan.
pub const MIN_THICK_RING_FRACTION: f64 = 0.25;

impl GlandParams {
    /// Periksa nilai: semua terhingga dan tidak negatif, `mpp_um > 0`,
    /// rasio di `[0, 1]`, ambang (bila ada) di `[0, 1]`.
    pub fn validate(&self) -> Result<(), ImgError> {
        if !(self.mpp_um.is_finite() && self.mpp_um > 0.0) {
            return Err(ImgError::InvalidParameter(format!(
                "mpp_um harus positif: {}",
                self.mpp_um
            )));
        }
        for (name, v) in [
            ("epithelium_sigma_um", self.epithelium_sigma_um),
            ("lumen_od_max", self.lumen_od_max),
            ("min_lumen_area_um2", self.min_lumen_area_um2),
            ("max_lumen_area_um2", self.max_lumen_area_um2),
            ("ring_width_um", self.ring_width_um),
            ("min_ring_thickness_um", self.min_ring_thickness_um),
            ("open_radius_um", self.open_radius_um),
            ("close_radius_um", self.close_radius_um),
            ("min_area_um2", self.min_area_um2),
            ("min_solid_thickness_um", self.min_solid_thickness_um),
            ("lumen_valley_um", self.lumen_valley_um),
            ("solid_seed_distance_um", self.solid_seed_distance_um),
            ("solid_valley_um", self.solid_valley_um),
            ("smooth_sigma_deg", self.smooth_sigma_deg),
            ("star_hull_max_area_factor", self.star_hull_max_area_factor),
            ("max_gray_fraction", self.max_gray_fraction),
        ] {
            if !(v.is_finite() && v >= 0.0) {
                return Err(ImgError::InvalidParameter(format!(
                    "{name} harus terhingga dan ≥ 0: {v}"
                )));
            }
        }
        if !(self.min_ring_fraction.is_finite() && (0.0..=1.0).contains(&self.min_ring_fraction)) {
            return Err(ImgError::InvalidParameter(
                "min_ring_fraction harus di [0, 1]".to_string(),
            ));
        }
        if let Some(t) = self.epithelium_threshold {
            if !(t.is_finite() && (0.0..=1.0).contains(&t)) {
                return Err(ImgError::InvalidParameter(
                    "epithelium_threshold harus di [0, 1]".to_string(),
                ));
            }
        }
        if let EpitheliumSource::Color { picks, tolerance } = &self.epithelium {
            if picks.is_empty() {
                return Err(ImgError::InvalidParameter(
                    "mode warna butuh minimal satu warna terpilih".to_string(),
                ));
            }
            if !(tolerance.is_finite() && *tolerance > 0.0)
                || picks.iter().flatten().any(|v| !v.is_finite())
            {
                return Err(ImgError::InvalidParameter(
                    "toleransi/warna palet harus terhingga dan toleransi > 0".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Hash FNV-1a 64-bit yang stabil atas semua parameter (untuk
    /// `AnnotationSource::Classic { params_hash }` dan laporan).
    pub fn params_hash(&self) -> u64 {
        let mut b: Vec<u8> = Vec::with_capacity(128);
        b.extend_from_slice(b"h3d-glands-v1");
        b.extend_from_slice(&self.mpp_um.to_bits().to_le_bytes());
        b.push(match self.stain {
            StainEstimate::Macenko => 0,
            StainEstimate::Ruifrok => 1,
        });
        b.extend_from_slice(&self.epithelium_sigma_um.to_bits().to_le_bytes());
        match self.epithelium_threshold {
            None => b.push(0),
            Some(t) => {
                b.push(1);
                b.extend_from_slice(&t.to_bits().to_le_bytes());
            }
        }
        for v in [
            self.lumen_od_max,
            self.min_lumen_area_um2,
            self.max_lumen_area_um2,
            self.ring_width_um,
            self.min_ring_fraction,
            self.min_ring_thickness_um,
            self.open_radius_um,
            self.close_radius_um,
            self.min_area_um2,
            self.min_solid_thickness_um,
            self.lumen_valley_um,
            self.solid_seed_distance_um,
            self.solid_valley_um,
            self.smooth_sigma_deg,
            self.star_hull_max_area_factor,
            self.max_gray_fraction,
        ] {
            b.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        b.push(u8::from(self.star_hull));
        match &self.epithelium {
            EpitheliumSource::Hematoxylin => b.push(0),
            EpitheliumSource::Color { picks, tolerance } => {
                b.push(1);
                b.extend_from_slice(&tolerance.to_bits().to_le_bytes());
                for p in picks {
                    for v in p {
                        b.extend_from_slice(&v.to_bits().to_le_bytes());
                    }
                }
            }
        }
        fnv1a(&b)
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Satu objek kelenjar/struktur hasil segmentasi.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Gland {
    /// Nomor objek `1..=n` (nilai di peta label).
    pub id: u32,
    /// Luas akhir (setelah star hull bila aktif), µm².
    pub area_um2: f64,
    /// Luas bentuk mentah sebelum star hull, µm².
    pub raw_area_um2: f64,
    /// Luas lumen di dalam objek, µm² (0 untuk sarang padat).
    pub lumen_area_um2: f64,
    /// Sentroid objek relatif pojok kiri-atas citra, µm (pusat piksel +0,5).
    pub centroid_um: [f64; 2],
    /// Kotak pembatas piksel `[x0, y0, x1, y1)`.
    pub bbox_px: [u32; 4],
    /// Benar bila objek memuat lumen yang lolos uji cincin.
    pub has_lumen: bool,
    /// Rasio cincin epitel tertinggi di antara lumennya; `None` untuk sarang
    /// padat.
    pub ring_fraction: Option<f64>,
    /// Soliditas bentuk mentah = luas / luas convex hull (≤ 1).
    pub solidity: f64,
}

/// Hasil segmentasi kelenjar satu citra.
#[derive(Clone, Debug, PartialEq)]
pub struct GlandSegmentation {
    /// Peta label `1..=n`, bentuk `(h, w)`; 0 = bukan objek.
    pub labels: Array2<u32>,
    /// Jumlah objek.
    pub n: usize,
    /// Objek, terurut menurut `id`.
    pub glands: Vec<Gland>,
    /// Hash parameter.
    pub params_hash: u64,
    /// Matriks pewarna yang dipakai (baris H, E, residu; ternormalisasi).
    pub stains: [[f32; 3]; 3],
    /// Benar bila `stains` hasil estimasi Macenko (bukan Ruifrok cadangan).
    pub stain_estimated: bool,
    /// Lumen yang gugur uji cincin tetapi berada dalam
    /// `RING_FRACTION_DELTA` di bawah ambang (masuk batas atas jumlah).
    pub near_miss_lumens: usize,
    /// Batas bawah jumlah: saringan luas `× MIN_AREA_STRICT_FACTOR` dan
    /// rasio cincin `+ RING_FRACTION_DELTA`.
    pub n_min: usize,
    /// Batas atas jumlah: `n + near_miss_lumens`.
    pub n_max: usize,
}

impl GlandSegmentation {
    /// Median luas objek, µm² (0 bila kosong).
    pub fn median_area_um2(&self) -> f64 {
        let mut a: Vec<f64> = self.glands.iter().map(|g| g.area_um2).collect();
        if a.is_empty() {
            return 0.0;
        }
        a.sort_by(f64::total_cmp);
        let m = a.len() / 2;
        if a.len().is_multiple_of(2) {
            (a[m - 1] + a[m]) / 2.0
        } else {
            a[m]
        }
    }

    /// Jumlah objek berlumen.
    pub fn n_with_lumen(&self) -> usize {
        self.glands.iter().filter(|g| g.has_lumen).count()
    }
}

/// Segmentasi kelenjar pada citra RGBA `w × h` berukuran piksel `mpp` (µm).
/// `tissue` (bentuk `(h, w)`) membatasi ke jaringan; `None` = dihitung
/// dengan [`tissue_mask`]. Lihat dokumentasi modul untuk alurnya.
pub fn segment_glands(
    rgba: &[u8],
    w: usize,
    h: usize,
    mpp: f64,
    tissue: Option<&Array2<u8>>,
    params: &GlandParams,
) -> Result<GlandSegmentation, ImgError> {
    params.validate()?;
    if !(mpp.is_finite() && mpp > 0.0) {
        return Err(ImgError::InvalidParameter(format!("mpp tidak sah: {mpp}")));
    }
    if let Some(t) = tissue {
        if t.dim() != (h, w) {
            return Err(ImgError::shape(&[h, w], &[t.dim().0, t.dim().1]));
        }
    }
    let tissue_owned;
    let tissue: &Array2<u8> = match tissue {
        Some(t) => t,
        None => {
            tissue_owned = tissue_mask(rgba, w, h, mpp)?;
            &tissue_owned
        }
    };
    let px = |um: f64| um / mpp;
    let area_px = |um2: f64| (um2 / (mpp * mpp)).ceil().max(0.0) as usize;

    // 1. Pewarna.
    let (matrix, stain_estimated) = match params.stain {
        StainEstimate::Ruifrok => (StainMatrix::he(), false),
        StainEstimate::Macenko => {
            match estimate_stains_macenko(rgba, w, h, Some(tissue), 0.15, 0.01)? {
                Some(m) => (m, true),
                None => (StainMatrix::he(), false),
            }
        }
    };

    // 2. Peta epitel: kanal H diambang, atau warna palet terpilih. Keduanya
    // dihaluskan Gauss `epithelium_sigma_um` supaya inti melebur jadi
    // tekstur epitel.
    let (hs, thr) = match &params.epithelium {
        EpitheliumSource::Hematoxylin => {
            let [hch, _, _] = deconvolve(rgba, w, h, &matrix)?;
            let hs = if params.epithelium_sigma_um > 0.0 {
                gaussian_blur_f32(&hch, px(params.epithelium_sigma_um))
            } else {
                hch
            };
            let hs = percentile_stretch(&hs, 0.01, 0.99);
            let thr = match params.epithelium_threshold {
                Some(t) => t,
                None => otsu_masked(&hs, tissue),
            };
            (hs, thr)
        }
        EpitheliumSource::Color { picks, tolerance } => {
            // Jarak warna dijepit ke 1,5 × toleransi agar kaca/putih tidak
            // mendominasi rerata (tepi cincin tipis tetap masuk), dihaluskan, lalu dibalik (`tol − d`) supaya
            // ambang "> 0" berarti rerata jarak ≤ toleransi.
            let cap = 1.5 * *tolerance;
            let d = color_distance(rgba, w, h, picks)?.mapv(|v| v.min(cap));
            let ds = if params.epithelium_sigma_um > 0.0 {
                gaussian_blur_f32(&d, px(params.epithelium_sigma_um))
            } else {
                d
            };
            (ds.mapv(|v| *tolerance - v), 0.0f32)
        }
    };
    let mut epi = Array2::<u8>::zeros((h, w));
    for ((y, x), &v) in hs.indexed_iter() {
        if v > thr && tissue[[y, x]] != 0 {
            epi[[y, x]] = 1;
        }
    }

    let min_lumen_px = area_px(params.min_lumen_area_um2).max(1);
    let max_lumen_px = area_px(params.max_lumen_area_um2).max(min_lumen_px);
    let mut lumen_keep = Array2::<u8>::zeros((h, w));
    // Rasio cincin per piksel lumen lolos (untuk atribut objek).
    let mut lumen_frac = Array2::<f32>::zeros((h, w));

    // Uji rasio cincin: bagian cincin selebar `ring_px` di sekeliling
    // komponen `l` dari `lab` yang berupa epitel. Dipakai untuk kedua sumber
    // lumen, sehingga lingkaran serat/debu di kaca (cincin sangat tipis) dan
    // vakuola lemak (sekat tanpa epitel) sama-sama gugur.
    let ring_px = px(params.ring_width_um).max(1.0);
    let pad = ring_px.ceil() as usize + 1;
    let mut near_miss = 0usize;
    // Epitel "tebal": sisa setelah erosi setengah ketebalan minimum.
    let epi_thick = if params.min_ring_thickness_um > 0.0 {
        crate::morphology::erode_disc(&epi, (px(params.min_ring_thickness_um) * 0.5).max(1.0))
    } else {
        epi.clone()
    };
    let mut keep_lumens =
        |lab: &Array2<u32>, n: usize, lumen_keep: &mut Array2<u8>, lumen_frac: &mut Array2<f32>| {
            let mut bbox: Vec<[usize; 4]> = vec![[usize::MAX, usize::MAX, 0, 0]; n + 1];
            let mut area = vec![0usize; n + 1];
            for ((y, x), &l) in lab.indexed_iter() {
                if l == 0 {
                    continue;
                }
                let bb = &mut bbox[l as usize];
                bb[0] = bb[0].min(x);
                bb[1] = bb[1].min(y);
                bb[2] = bb[2].max(x + 1);
                bb[3] = bb[3].max(y + 1);
                area[l as usize] += 1;
            }
            for l in 1..=n {
                if !(min_lumen_px..=max_lumen_px).contains(&area[l]) {
                    continue;
                }
                let [bx0, by0, bx1, by1] = bbox[l];
                let x0 = bx0.saturating_sub(pad);
                let y0 = by0.saturating_sub(pad);
                let x1 = (bx1 + pad).min(w);
                let y1 = (by1 + pad).min(h);
                let crop = lab
                    .slice(s![y0..y1, x0..x1])
                    .mapv(|v| u8::from(v == l as u32));
                let dil = dilate_disc(&crop, ring_px);
                let epi_c = epi.slice(s![y0..y1, x0..x1]);
                let thick_c = epi_thick.slice(s![y0..y1, x0..x1]);
                let (mut ring_n, mut ring_epi, mut ring_thick) = (0usize, 0usize, 0usize);
                for (((d, c), e), t) in dil
                    .iter()
                    .zip(crop.iter())
                    .zip(epi_c.iter())
                    .zip(thick_c.iter())
                {
                    if *d != 0 && *c == 0 {
                        ring_n += 1;
                        if *e != 0 {
                            ring_epi += 1;
                        }
                        if *t != 0 {
                            ring_thick += 1;
                        }
                    }
                }
                if ring_n == 0 {
                    continue;
                }
                let frac = ring_epi as f64 / ring_n as f64;
                let thick = ring_thick as f64 / ring_n as f64;
                if thick < MIN_THICK_RING_FRACTION {
                    continue;
                }
                if frac >= params.min_ring_fraction {
                    for ((y, x), &v) in lab.slice(s![y0..y1, x0..x1]).indexed_iter() {
                        if v == l as u32 {
                            lumen_keep[[y0 + y, x0 + x]] = 1;
                            lumen_frac[[y0 + y, x0 + x]] = frac as f32;
                        }
                    }
                } else if frac >= params.min_ring_fraction - RING_FRACTION_DELTA {
                    near_miss += 1;
                }
            }
        };

    // 3a. Lumen = lubang cincin epitel tertutup, apa pun isinya.
    {
        let closed_epi = if params.close_radius_um > 0.0 {
            close(&epi, px(params.close_radius_um))
        } else {
            epi.clone()
        };
        let filled = fill_holes(&closed_epi);
        // Pulau epitel yang terkurung di dalam lubang (sel lepas, debris
        // gelap di lumen) bukan dinding kelenjar: hanya komponen epitel yang
        // menyentuh luar (bukan-`filled`) yang dianggap cincin; sisanya ikut
        // menjadi lumen supaya satu kelenjar tidak terbelah oleh debrisnya.
        let (epi_lab, n_epi) = label_components(&closed_epi);
        let mut touches_outside = vec![false; n_epi + 1];
        for ((y, x), &l) in epi_lab.indexed_iter() {
            if l == 0 {
                continue;
            }
            let at_border = y == 0 || x == 0 || y + 1 == h || x + 1 == w;
            let near_outside = at_border
                || filled[[y - 1, x]] == 0
                || filled[[y + 1, x]] == 0
                || filled[[y, x - 1]] == 0
                || filled[[y, x + 1]] == 0;
            if near_outside {
                touches_outside[l as usize] = true;
            }
        }
        let holes = Array2::from_shape_fn((h, w), |ix| {
            let l = epi_lab[ix] as usize;
            u8::from(filled[ix] != 0 && (l == 0 || !touches_outside[l]) && tissue[ix] != 0)
        });
        let (hole_lab, n_hole) = label_components(&holes);
        keep_lumens(&hole_lab, n_hole, &mut lumen_keep, &mut lumen_frac);
    }

    // 3b. Lumen terang (cincin terbuka/tipis) yang lolos uji cincin.
    let od = od_total(rgba, w, h)?;
    let t_bg = otsu(od.iter().copied(), 256).min(params.lumen_od_max as f32);
    let lumen_raw = Array2::from_shape_fn((h, w), |(y, x)| {
        u8::from(tissue[[y, x]] != 0 && lumen_keep[[y, x]] == 0 && od[[y, x]] < t_bg)
    });
    let (lum_lab, n_lum) = label_components(&lumen_raw);
    keep_lumens(&lum_lab, n_lum, &mut lumen_keep, &mut lumen_frac);

    // 4. Masker kelenjar.
    let mut g = Array2::from_shape_fn((h, w), |ix| u8::from(epi[ix] != 0 || lumen_keep[ix] != 0));
    if params.open_radius_um > 0.0 {
        g = open(&g, px(params.open_radius_um));
        // Lumen lolos tidak boleh hilang oleh pembukaan.
        for (gv, &lk) in g.iter_mut().zip(lumen_keep.iter()) {
            if lk != 0 {
                *gv = 1;
            }
        }
    }
    if params.close_radius_um > 0.0 {
        g = close(&g, px(params.close_radius_um));
    }
    {
        let filled = fill_holes(&g);
        let holes = Array2::from_shape_fn((h, w), |ix| u8::from(filled[ix] != 0 && g[ix] == 0));
        let (hl, hn) = label_components(&holes);
        let mut harea = vec![0usize; hn + 1];
        for &v in hl.iter() {
            harea[v as usize] += 1;
        }
        for (gv, &v) in g.iter_mut().zip(hl.iter()) {
            if v != 0 && harea[v as usize] < min_lumen_px {
                *gv = 1;
            }
        }
    }
    for (gv, &t) in g.iter_mut().zip(tissue.iter()) {
        if t == 0 {
            *gv = 0;
        }
    }

    // 5. Pemisahan per komponen.
    let min_area_px = area_px(params.min_area_um2).max(1);
    let (comp_lab, n_comp) = label_components(&g);
    let mut comp_bbox: Vec<[usize; 4]> = vec![[usize::MAX, usize::MAX, 0, 0]; n_comp + 1];
    for ((y, x), &l) in comp_lab.indexed_iter() {
        if l == 0 {
            continue;
        }
        let b = &mut comp_bbox[l as usize];
        b[0] = b[0].min(x);
        b[1] = b[1].min(y);
        b[2] = b[2].max(x + 1);
        b[3] = b[3].max(y + 1);
    }
    let mut raw = Array2::<u32>::zeros((h, w));
    let mut next = 0u32;
    for (c, &[x0, y0, x1, y1]) in comp_bbox.iter().enumerate().skip(1) {
        let comp = comp_lab
            .slice(s![y0..y1, x0..x1])
            .mapv(|v| u8::from(v == c as u32));
        let lum_c = Array2::from_shape_fn(comp.raw_dim(), |(y, x)| {
            u8::from(comp[[y, x]] != 0 && lumen_keep[[y0 + y, x0 + x]] != 0)
        });
        let has_lumen = lum_c.iter().any(|&v| v != 0);
        let (mut lab, n) = if has_lumen {
            let field = edt_2d_um(&lum_c, mpp).mapv(|d| -d);
            split_on_field(&comp, &field, mpp, 0.0, params.lumen_valley_um as f32)
        } else {
            split_touching(
                &comp,
                mpp,
                params.solid_seed_distance_um,
                params.solid_valley_um,
            )
        };
        let n = if n > 1 {
            merge_small_regions(&mut lab, n, min_area_px)
        } else {
            n
        };
        if n == 0 {
            continue;
        }
        for ((y, x), &v) in lab.indexed_iter() {
            if v != 0 {
                raw[[y0 + y, x0 + x]] = next + v;
            }
        }
        next += n as u32;
    }
    // Saring luas dan ketebalan (objek tanpa lumen), nomori ulang menurut
    // urutan raster.
    let nr = next as usize;
    let mut area = vec![0usize; nr + 1];
    let mut has_lum = vec![false; nr + 1];
    let mut gray = vec![0usize; nr + 1];
    let mut non_lumen = vec![0usize; nr + 1];
    let mut rb: Vec<[usize; 4]> = vec![[usize::MAX, usize::MAX, 0, 0]; nr + 1];
    for ((y, x), &v) in raw.indexed_iter() {
        let i = v as usize;
        area[i] += 1;
        if i == 0 {
            continue;
        }
        if lumen_keep[[y, x]] != 0 {
            has_lum[i] = true;
        } else {
            non_lumen[i] += 1;
            let k = (y * w + x) * 4;
            let (r, g, b) = (rgba[k], rgba[k + 1], rgba[k + 2]);
            let hi = r.max(g).max(b);
            let lo = r.min(g).min(b);
            if hi - lo < 30 && hi < 180 {
                gray[i] += 1;
            }
        }
        let b = &mut rb[i];
        b[0] = b[0].min(x);
        b[1] = b[1].min(y);
        b[2] = b[2].max(x + 1);
        b[3] = b[3].max(y + 1);
    }
    // Artefak tak berwarna (serat debu, tinta) di kaca: bukan jaringan H&E.
    for i in 1..=nr {
        if non_lumen[i] > 0 && gray[i] as f64 > params.max_gray_fraction * non_lumen[i] as f64 {
            area[i] = 0;
        }
    }
    let half_t2 = (px(params.min_solid_thickness_um) * 0.5).powi(2) as f32;
    if half_t2 > 0.0 {
        for i in 1..=nr {
            if has_lum[i] || area[i] < min_area_px {
                continue;
            }
            let [x0, y0, x1, y1] = rb[i];
            let inv = raw
                .slice(s![y0..y1, x0..x1])
                .mapv(|v| u8::from(v != i as u32));
            let dmax = edt_squared_2d(&inv).iter().copied().fold(0.0f32, f32::max);
            if dmax < half_t2 {
                area[i] = 0;
            }
        }
    }
    let mut remap = vec![0u32; next as usize + 1];
    let mut n = 0u32;
    for v in raw.iter_mut() {
        let i = *v as usize;
        if i == 0 {
            continue;
        }
        if area[i] < min_area_px {
            *v = 0;
            continue;
        }
        if remap[i] == 0 {
            n += 1;
            remap[i] = n;
        }
        *v = remap[i];
    }
    let n = n as usize;

    // Statistik bentuk mentah.
    let mut stats = RawStats::new(n);
    for ((y, x), &v) in raw.indexed_iter() {
        if v != 0 {
            stats.add(v as usize, x, y);
        }
    }
    let solidity = raw_solidity(&raw, n, &stats);

    // 6. Star hull.
    let labels = if params.star_hull && n > 0 {
        star_hull_labels(
            &raw,
            n,
            &stats,
            tissue,
            params.smooth_sigma_deg,
            params.star_hull_max_area_factor,
        )
    } else {
        raw
    };

    // Atribut akhir.
    let mut fin = RawStats::new(n);
    let mut lumen_px = vec![0usize; n + 1];
    let mut ring = vec![0f32; n + 1];
    for ((y, x), &v) in labels.indexed_iter() {
        if v == 0 {
            continue;
        }
        let i = v as usize;
        fin.add(i, x, y);
        if lumen_keep[[y, x]] != 0 {
            lumen_px[i] += 1;
            ring[i] = ring[i].max(lumen_frac[[y, x]]);
        }
    }
    let a_um2 = mpp * mpp;
    let glands: Vec<Gland> = (1..=n)
        .map(|i| Gland {
            id: i as u32,
            area_um2: fin.area[i] as f64 * a_um2,
            raw_area_um2: stats.area[i] as f64 * a_um2,
            lumen_area_um2: lumen_px[i] as f64 * a_um2,
            centroid_um: [
                (fin.sx[i] / fin.area[i].max(1) as f64 + 0.5) * mpp,
                (fin.sy[i] / fin.area[i].max(1) as f64 + 0.5) * mpp,
            ],
            bbox_px: [
                fin.bbox[i][0] as u32,
                fin.bbox[i][1] as u32,
                fin.bbox[i][2] as u32,
                fin.bbox[i][3] as u32,
            ],
            has_lumen: lumen_px[i] > 0,
            ring_fraction: (lumen_px[i] > 0).then_some(ring[i] as f64),
            solidity: solidity[i],
        })
        .collect();
    let strict_area = params.min_area_um2 * MIN_AREA_STRICT_FACTOR;
    let strict_ring = params.min_ring_fraction + RING_FRACTION_DELTA;
    let n_min = glands
        .iter()
        .filter(|g| {
            g.raw_area_um2 >= strict_area && g.ring_fraction.is_none_or(|f| f >= strict_ring)
        })
        .count();
    Ok(GlandSegmentation {
        labels,
        n,
        glands,
        params_hash: params.params_hash(),
        stains: matrix.stains,
        stain_estimated,
        near_miss_lumens: near_miss,
        n_min,
        n_max: n + near_miss,
    })
}

/// Masker biner dari peta instance dengan **pemisah** antar objek: piksel
/// objek yang berjarak Chebyshev `≤ gap_px` dari piksel objek **lain**
/// menjadi 0, sehingga dua objek bertetangga tetap dua komponen terhubung
/// setelah diresampling ke resolusi label (pemanggil memilih `gap_px` ≥
/// rasio resolusi + 1). `gap_px = 0` → masker tanpa pemisah.
pub fn instances_to_mask(labels: &Array2<u32>, gap_px: usize) -> Array2<u8> {
    let (h, w) = labels.dim();
    let g = gap_px as i64;
    Array2::from_shape_fn((h, w), |(y, x)| {
        let v = labels[[y, x]];
        if v == 0 {
            return 0;
        }
        if g == 0 {
            return 1;
        }
        for dy in -g..=g {
            for dx in -g..=g {
                let (ny, nx) = (y as i64 + dy, x as i64 + dx);
                if ny < 0 || nx < 0 || ny >= h as i64 || nx >= w as i64 {
                    continue;
                }
                let o = labels[[ny as usize, nx as usize]];
                if o != 0 && o != v {
                    return 0;
                }
            }
        }
        1
    })
}

struct RawStats {
    area: Vec<usize>,
    sx: Vec<f64>,
    sy: Vec<f64>,
    bbox: Vec<[usize; 4]>,
}

impl RawStats {
    fn new(n: usize) -> Self {
        Self {
            area: vec![0; n + 1],
            sx: vec![0.0; n + 1],
            sy: vec![0.0; n + 1],
            bbox: vec![[usize::MAX, usize::MAX, 0, 0]; n + 1],
        }
    }

    fn add(&mut self, i: usize, x: usize, y: usize) {
        self.area[i] += 1;
        self.sx[i] += x as f64;
        self.sy[i] += y as f64;
        let b = &mut self.bbox[i];
        b[0] = b[0].min(x);
        b[1] = b[1].min(y);
        b[2] = b[2].max(x + 1);
        b[3] = b[3].max(y + 1);
    }
}

/// Benar bila piksel `(x, y)` di `sub` bernilai `id` dan menyentuh
/// bukan-`id` (4-tetangga) atau tepi larik.
fn is_edge(sub: &Array2<u32>, x: usize, y: usize, id: u32) -> bool {
    let (sh, sw) = sub.dim();
    y == 0
        || x == 0
        || y + 1 == sh
        || x + 1 == sw
        || sub[[y - 1, x]] != id
        || sub[[y + 1, x]] != id
        || sub[[y, x - 1]] != id
        || sub[[y, x + 1]] != id
}

/// Soliditas tiap objek: luas / luas convex hull titik sudut pikselnya.
fn raw_solidity(labels: &Array2<u32>, n: usize, st: &RawStats) -> Vec<f64> {
    let mut out = vec![1.0f64; n + 1];
    for (i, o) in out.iter_mut().enumerate().skip(1) {
        let [x0, y0, x1, y1] = st.bbox[i];
        if st.area[i] == 0 || x1 <= x0 || y1 <= y0 {
            continue;
        }
        let id = i as u32;
        let sub = labels.slice(s![y0..y1, x0..x1]).to_owned();
        let (sh, sw) = sub.dim();
        let mut pts: Vec<(f64, f64)> = Vec::new();
        for y in 0..sh {
            for x in 0..sw {
                if sub[[y, x]] == id && is_edge(&sub, x, y, id) {
                    let (fx, fy) = (x as f64, y as f64);
                    pts.push((fx, fy));
                    pts.push((fx + 1.0, fy));
                    pts.push((fx, fy + 1.0));
                    pts.push((fx + 1.0, fy + 1.0));
                }
            }
        }
        let hull_area = convex_hull_area(&mut pts);
        if hull_area > 0.0 {
            *o = (st.area[i] as f64 / hull_area).min(1.0);
        }
    }
    out
}

/// Luas convex hull (rantai monoton Andrew); `pts` diurutkan di tempat.
fn convex_hull_area(pts: &mut Vec<(f64, f64)>) -> f64 {
    if pts.len() < 3 {
        return 0.0;
    }
    pts.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    pts.dedup();
    if pts.len() < 3 {
        return 0.0;
    }
    let cross = |o: (f64, f64), a: (f64, f64), b: (f64, f64)| {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };
    let mut lower: Vec<(f64, f64)> = Vec::new();
    for &p in pts.iter() {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<(f64, f64)> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    let hull = lower;
    let mut a = 0.0;
    for k in 0..hull.len() {
        let (x1, y1) = hull[k];
        let (x2, y2) = hull[(k + 1) % hull.len()];
        a += x1 * y2 - x2 * y1;
    }
    (a * 0.5).abs()
}

/// Ganti tiap objek dengan star hull halusnya (lihat dokumentasi modul
/// langkah 6). Objek diproses menurut nomor; piksel hanya diambil bila belum
/// milik objek lain di peta asal maupun hasil, dan berada di jaringan.
fn star_hull_labels(
    raw: &Array2<u32>,
    n: usize,
    st: &RawStats,
    tissue: &Array2<u8>,
    sigma_deg: f64,
    max_area_factor: f64,
) -> Array2<u32> {
    let (h, w) = raw.dim();
    let mut out = Array2::<u32>::zeros((h, w));
    let mut added: Vec<(usize, usize)> = Vec::new();
    let bins = STAR_BINS;
    let bin_deg = 360.0 / bins as f64;
    for i in 1..=n {
        let id = i as u32;
        let [x0, y0, x1, y1] = st.bbox[i];
        if st.area[i] == 0 || x1 <= x0 || y1 <= y0 {
            continue;
        }
        let sub = raw.slice(s![y0..y1, x0..x1]).to_owned();
        // Pusat: piksel EDT maksimum (seri → urutan raster).
        let edt = edt_squared_2d(&sub.mapv(|v| u8::from(v != id)));
        let mut best = (-1.0f32, 0usize, 0usize);
        for ((y, x), &d) in edt.indexed_iter() {
            if sub[[y, x]] == id && d > best.0 {
                best = (d, x, y);
            }
        }
        let (cx, cy) = (best.1 as f64 + 0.5, best.2 as f64 + 0.5);
        // Jari-jari terjauh tepi per bin sudut.
        let mut r = vec![f64::NAN; bins];
        let (sh, sw) = sub.dim();
        for y in 0..sh {
            for x in 0..sw {
                if sub[[y, x]] != id || !is_edge(&sub, x, y, id) {
                    continue;
                }
                let (dx, dy) = (x as f64 + 0.5 - cx, y as f64 + 0.5 - cy);
                let d = (dx * dx + dy * dy).sqrt() + 0.5;
                let ang = dy.atan2(dx).to_degrees().rem_euclid(360.0);
                let b = ((ang / bin_deg).floor() as usize).min(bins - 1);
                if r[b].is_nan() || d > r[b] {
                    r[b] = d;
                }
            }
        }
        if r.iter().all(|v| v.is_nan()) {
            continue;
        }
        fill_circular_gaps(&mut r);
        // Penghalusan Gauss melingkar.
        if sigma_deg > 0.0 {
            let sb = sigma_deg / bin_deg;
            let half = (3.0 * sb).ceil() as i64;
            let mut sm = vec![0.0f64; bins];
            for (b, o) in sm.iter_mut().enumerate() {
                let (mut acc, mut wsum) = (0.0, 0.0);
                for k in -half..=half {
                    let wgt = (-(k as f64 * k as f64) / (2.0 * sb * sb)).exp();
                    let j = (b as i64 + k).rem_euclid(bins as i64) as usize;
                    acc += wgt * r[j];
                    wsum += wgt;
                }
                *o = acc / wsum;
            }
            r = sm;
        }
        let rmax = r.iter().copied().fold(0.0f64, f64::max);
        // Rasterisasi dalam koordinat citra penuh.
        let (gcx, gcy) = (x0 as f64 + cx, y0 as f64 + cy);
        let gx0 = ((gcx - rmax).floor().max(0.0)) as usize;
        let gy0 = ((gcy - rmax).floor().max(0.0)) as usize;
        let gx1 = (((gcx + rmax).ceil().max(0.0)) as usize).min(w);
        let gy1 = (((gcy + rmax).ceil().max(0.0)) as usize).min(h);
        // Piksel mentah selalu dipertahankan; piksel tambahan dikumpulkan
        // dulu dan hanya dipakai bila hull tidak membengkak melewati batas.
        added.clear();
        for y in gy0..gy1 {
            for x in gx0..gx1 {
                let o = raw[[y, x]];
                if (o != 0 && o != id) || out[[y, x]] != 0 || tissue[[y, x]] == 0 {
                    continue;
                }
                if o == id {
                    out[[y, x]] = id;
                    continue;
                }
                let (dx, dy) = (x as f64 + 0.5 - gcx, y as f64 + 0.5 - gcy);
                let d = (dx * dx + dy * dy).sqrt();
                let ang = dy.atan2(dx).to_degrees().rem_euclid(360.0);
                let fb = ang / bin_deg;
                let b0 = (fb.floor() as usize).min(bins - 1);
                let b1 = (b0 + 1) % bins;
                let t = fb - b0 as f64;
                let rr = r[b0] * (1.0 - t) + r[b1] * t;
                if d <= rr {
                    added.push((x, y));
                }
            }
        }
        let limit = (st.area[i] as f64 * max_area_factor.max(1.0)) as usize;
        if st.area[i] + added.len() <= limit {
            for &(x, y) in &added {
                out[[y, x]] = id;
            }
        }
    }
    out
}

/// Isi entri NaN (bin sudut tanpa piksel tepi) dengan **minimum** dua
/// tetangga terdekat yang diketahui (melingkar). Konservatif: arah tanpa
/// tepi berarti objek tipis di arah itu, bukan sektor yang harus diisi.
fn fill_circular_gaps(r: &mut [f64]) {
    let n = r.len();
    let known: Vec<usize> = (0..n).filter(|&i| !r[i].is_nan()).collect();
    if known.is_empty() || known.len() == n {
        return;
    }
    for (k, &a) in known.iter().enumerate() {
        let b = known[(k + 1) % known.len()];
        let span = if b > a { b - a } else { b + n - a };
        let v = r[a].min(r[b]);
        for step in 1..span {
            r[(a + step) % n] = v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phantom::render_glands;

    fn dice_label(a: &Array2<u32>, ia: u32, b: &Array2<u32>, ib: u32) -> f64 {
        let (mut inter, mut na, mut nb) = (0usize, 0usize, 0usize);
        for (&x, &y) in a.iter().zip(b.iter()) {
            let (p, q) = (x == ia, y == ib);
            na += usize::from(p);
            nb += usize::from(q);
            inter += usize::from(p && q);
        }
        if na + nb == 0 {
            return 0.0;
        }
        2.0 * inter as f64 / (na + nb) as f64
    }

    /// Objek hasil yang paling tumpang tindih dengan kelenjar kebenaran `t`.
    fn best_match(seg: &GlandSegmentation, truth: &Array2<u32>, t: u32) -> (u32, f64) {
        let mut best = (0u32, 0.0f64);
        for g in &seg.glands {
            let d = dice_label(&seg.labels, g.id, truth, t);
            if d > best.1 {
                best = (g.id, d);
            }
        }
        best
    }

    fn phantom() -> (
        Vec<u8>,
        crate::phantom::GlandPhantomTruth,
        usize,
        usize,
        f64,
    ) {
        let mpp = 2.0;
        let (w, h) = (400usize, 300usize);
        let (rgba, truth) = render_glands([w, h], mpp, 3);
        (rgba, truth, w, h, mpp)
    }

    #[test]
    fn params_validate_and_hash() {
        let p = GlandParams::default();
        assert!(p.validate().is_ok());
        assert_eq!(p.params_hash(), GlandParams::default().params_hash());
        let q = GlandParams {
            min_ring_fraction: 0.6,
            ..GlandParams::default()
        };
        assert_ne!(p.params_hash(), q.params_hash());
        for bad in [
            GlandParams {
                mpp_um: 0.0,
                ..GlandParams::default()
            },
            GlandParams {
                min_area_um2: -1.0,
                ..GlandParams::default()
            },
            GlandParams {
                min_ring_fraction: 1.5,
                ..GlandParams::default()
            },
            GlandParams {
                epithelium_threshold: Some(f32::NAN),
                ..GlandParams::default()
            },
        ] {
            assert!(bad.validate().is_err());
        }
        assert!(segment_glands(&[0u8; 16], 2, 2, 0.0, None, &p).is_err());
        let tm = Array2::<u8>::zeros((3, 3));
        assert!(segment_glands(&[0u8; 16], 2, 2, 1.0, Some(&tm), &p).is_err());
    }

    #[test]
    fn raw_shapes_match_phantom_glands_and_ignore_fat() {
        let (rgba, truth, w, h, mpp) = phantom();
        let params = GlandParams {
            star_hull: false,
            ..GlandParams::default()
        };
        let seg = segment_glands(&rgba, w, h, mpp, None, &params).unwrap();
        assert_eq!(seg.n, truth.n, "jumlah objek: {:?}", seg.glands);
        assert!(seg.n_min <= seg.n && seg.n <= seg.n_max);
        assert!(seg.stain_estimated, "Macenko harus berhasil pada fantom");
        // Tiap kelenjar kebenaran punya pasangan dengan Dice tinggi.
        let mut used = Vec::new();
        for t in 1..=truth.n as u32 {
            let (id, d) = best_match(&seg, &truth.labels, t);
            assert!(d >= 0.8, "kelenjar {t}: Dice {d:.3}");
            assert!(!used.contains(&id), "objek {id} dipakai dua kelenjar");
            used.push(id);
        }
        let (solid_id, _) = best_match(&seg, &truth.labels, truth.solid);
        let solid = &seg.glands[(solid_id - 1) as usize];
        assert!(!solid.has_lumen && solid.ring_fraction.is_none());
        let (g1, _) = best_match(&seg, &truth.labels, 1);
        assert!(seg.glands[(g1 - 1) as usize].has_lumen);
        // Tidak ada objek yang menutupi lemak.
        let fat_hit = seg
            .labels
            .iter()
            .zip(truth.fat.iter())
            .filter(|(l, f)| **l != 0 && **f != 0)
            .count();
        let fat_n = truth.fat.iter().filter(|v| **v != 0).count();
        assert!(fat_hit * 20 < fat_n, "lemak tertutup {fat_hit}/{fat_n}");
        // Determinisme.
        let seg2 = segment_glands(&rgba, w, h, mpp, None, &params).unwrap();
        assert_eq!(seg.labels, seg2.labels);
    }

    #[test]
    fn star_hull_keeps_count_fills_u_and_never_merges_neighbours() {
        let (rgba, truth, w, h, mpp) = phantom();
        let raw = segment_glands(
            &rgba,
            w,
            h,
            mpp,
            None,
            &GlandParams {
                star_hull: false,
                ..GlandParams::default()
            },
        )
        .unwrap();
        let seg = segment_glands(&rgba, w, h, mpp, None, &GlandParams::default()).unwrap();
        assert_eq!(seg.n, raw.n);
        let (u_id, _) = best_match(&raw, &truth.labels, truth.u_shaped);
        let u_raw = &raw.glands[(u_id - 1) as usize];
        let u_hull = &seg.glands[(u_id - 1) as usize];
        assert!(
            u_raw.solidity < 0.9,
            "U mentah harus cekung: {}",
            u_raw.solidity
        );
        assert!(
            u_hull.area_um2 > u_raw.area_um2,
            "star hull mengisi bukaan U: {} vs {}",
            u_hull.area_um2,
            u_raw.area_um2
        );
        // Bentuk mentah selalu termuat dalam hull-nya.
        for (&a, &b) in raw.labels.iter().zip(seg.labels.iter()) {
            if a != 0 {
                assert_eq!(a, b);
            }
        }
        let [a, b] = truth.touching;
        let (ia, _) = best_match(&seg, &truth.labels, a);
        let (ib, _) = best_match(&seg, &truth.labels, b);
        assert_ne!(ia, ib);
        let m = instances_to_mask(&seg.labels, 1);
        let (_, nc) = label_components(&m);
        assert_eq!(nc, seg.n, "pemisah 1 px harus menjaga jumlah komponen");
        let m0 = instances_to_mask(&seg.labels, 0);
        assert_eq!(
            m0.iter().filter(|v| **v != 0).count(),
            seg.labels.iter().filter(|v| **v != 0).count()
        );
    }

    #[test]
    fn gray_fibre_loop_on_glass_is_rejected() {
        // Cincin hitam-abu tak berwarna (serat debu) di kaca putih: bukan
        // kelenjar walau membentuk lubang tertutup.
        let (w, h) = (160usize, 160usize);
        let mut rgba = vec![255u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let (dx, dy) = (x as f64 + 0.5 - 80.0, y as f64 + 0.5 - 80.0);
                let d = (dx * dx + dy * dy).sqrt();
                if (40.0..=44.0).contains(&d) {
                    let k = (y * w + x) * 4;
                    rgba[k..k + 3].copy_from_slice(&[60, 62, 64]);
                }
            }
        }
        let seg = segment_glands(&rgba, w, h, 2.0, None, &GlandParams::default()).unwrap();
        assert_eq!(seg.n, 0, "{:?}", seg.glands);
        let lenient = GlandParams {
            max_gray_fraction: 1.0,
            ..GlandParams::default()
        };
        let seg2 = segment_glands(&rgba, w, h, 2.0, None, &lenient).unwrap();
        assert!(seg2.n >= 1, "tanpa saringan abu-abu cincin ini lolos");
    }

    #[test]
    fn color_mode_with_dark_palette_picks_matches_phantom_glands() {
        let (rgba, truth, w, h, mpp) = phantom();
        let pal = crate::palette::extract_palette(&rgba, w, h, None, 5).unwrap();
        let idx = pal.preset_indices(crate::palette::ColorPreset::Gland);
        assert!(!idx.is_empty());
        let params = GlandParams {
            epithelium: EpitheliumSource::Color {
                picks: pal.picks(&idx),
                tolerance: pal.default_tolerance(&idx),
            },
            star_hull: false,
            ..GlandParams::default()
        };
        let seg = segment_glands(&rgba, w, h, mpp, None, &params).unwrap();
        assert_eq!(seg.n, truth.n, "{:?}", seg.glands);
        for t in 1..=truth.n as u32 {
            let (_, d) = best_match(&seg, &truth.labels, t);
            assert!(d >= 0.8, "kelenjar {t}: Dice {d:.3}");
        }
        assert_ne!(seg.params_hash, GlandParams::default().params_hash());
        let bad = GlandParams {
            epithelium: EpitheliumSource::Color {
                picks: vec![],
                tolerance: 10.0,
            },
            ..GlandParams::default()
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn blank_image_gives_no_objects() {
        let (w, h) = (64usize, 48usize);
        let white = vec![255u8; w * h * 4];
        let seg = segment_glands(&white, w, h, 2.0, None, &GlandParams::default()).unwrap();
        assert_eq!(seg.n, 0);
        assert!(seg.glands.is_empty());
        assert_eq!(seg.median_area_um2(), 0.0);
    }

    #[test]
    fn convex_hull_area_of_unit_square_and_gap_filling() {
        let mut pts = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)];
        assert!((convex_hull_area(&mut pts) - 1.0).abs() < 1e-12);
        let mut r = vec![1.0, f64::NAN, f64::NAN, 4.0, f64::NAN];
        fill_circular_gaps(&mut r);
        assert_eq!(r, vec![1.0, 1.0, 1.0, 4.0, 1.0]);
    }

    #[test]
    fn star_hull_never_balloons_thin_curved_objects() {
        // Strip tipis melengkung (seperempat cincin tebal 4 px) di stroma
        // pucat tanpa lumen: hull harus tetap ≤ 1,5 × luas mentah.
        let (w, h, mpp) = (160usize, 160usize, 2.0);
        let mut conc = vec![[0.06f32, 0.25, 0.0]; w * h];
        for y in 0..h {
            for x in 0..w {
                let (dx, dy) = (x as f64 + 0.5 - 20.0, y as f64 + 0.5 - 20.0);
                let d = (dx * dx + dy * dy).sqrt();
                if (100.0..=106.0).contains(&d) && dx > 0.0 && dy > 0.0 {
                    conc[y * w + x] = [0.9, 0.3, 0.0];
                }
            }
        }
        let m = crate::color::StainMatrix::he();
        let mut rgba = vec![255u8; w * h * 4];
        for (i, c) in conc.iter().enumerate() {
            let od = m.od_from_concentrations(*c);
            for k in 0..3 {
                rgba[i * 4 + k] = crate::color::intensity_of_od(od[k]);
            }
        }
        let tissue = Array2::<u8>::ones((h, w));
        let params = GlandParams {
            min_solid_thickness_um: 0.0,
            stain: StainEstimate::Ruifrok,
            ..GlandParams::default()
        };
        let seg = segment_glands(&rgba, w, h, mpp, Some(&tissue), &params).unwrap();
        assert!(seg.n >= 1, "strip harus terdeteksi");
        for g in &seg.glands {
            assert!(
                g.area_um2 <= g.raw_area_um2 * params.star_hull_max_area_factor + 1.0,
                "objek {} membengkak: {} vs mentah {}",
                g.id,
                g.area_um2,
                g.raw_area_um2
            );
        }
    }
}
