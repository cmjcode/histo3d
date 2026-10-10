//! Fantom sintetis ber-kebenaran-dasar (histo3d.md 11.1, 8.12.11).
//!
//! Bentuk 3D analitik (`h3d_volume::phantom3d`) diiris, dirender menjadi
//! "histologi" H&E sintetis yang teksturnya berkorelasi antar irisan
//! bertetangga, lalu tiap irisan diberi transformasi berbenih yang dicatat.
//! Keluaran: PNG per irisan, GeoJSON masker per irisan (untuk
//! `h3d labels import`), dan `truth.json` berisi transformasi dan nilai
//! analitik, sehingga seluruh pipeline bisa dijalankan dari CLI.

use std::path::{Path, PathBuf};

use h3d_core::{Affine2, LabelId};
use h3d_imgproc::color::intensity_of_od;
use h3d_imgproc::phantom::{ellipsoid_mask, synthetic_ihc, Rng};
use h3d_imgproc::StainMatrix;
use h3d_io::write_atomic;
use h3d_segment::{LabelStore, PaintMode};
use h3d_volume::phantom3d::{
    LABEL_CAPSULE, LABEL_MARGIN, LABEL_SURFACE, LABEL_TUMOR, LABEL_VESSEL,
};
use h3d_volume::{generate_phantom, PhantomShape, PhantomSpec};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::error::EngineError;

/// Bentuk dasar fantom.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhantomShapeKind {
    /// Ellipsoid.
    Ellipsoid,
    /// Torus (genus 1).
    Torus,
    /// Cabang Y.
    Branch,
}

/// Opsi pembuatan fantom.
#[derive(Clone, Debug, PartialEq)]
pub struct PhantomOptions {
    /// Bentuk.
    pub shape: PhantomShapeKind,
    /// Jumlah irisan.
    pub slices: usize,
    /// Benih semua keacakan.
    pub seed: u64,
    /// Acak nama berkas sehingga tidak mencerminkan urutan potong.
    pub shuffle: bool,
    /// Bidang `Margin` pada jarak ini (µm) dari permukaan tumor arah +X.
    pub margin_um: Option<f64>,
    /// `(jumlah, celah µm)`: beberapa fokus bola.
    pub foci: Option<(usize, f64)>,
    /// Silinder `Vessel` menembus tumor.
    pub vessel: bool,
    /// Kulit bola `Capsule` di sekeliling tumor.
    pub capsule: bool,
    /// Tumor menembus kapsul (butuh `capsule`).
    pub breach: bool,
    /// Lapisan `Surface` di z = 0 dan tumor sampai kedalaman ini (µm).
    pub surface_um: Option<f64>,
    /// Tambahkan `ihc.png`: gambar IHC sintetis dengan fraksi inti positif ini.
    pub ihc: Option<f64>,
    /// Sisi gambar irisan, piksel.
    pub size_px: u32,
    /// Ukuran piksel gambar irisan, µm.
    pub mpp: f64,
    /// Jarak irisan; `None` = dipilih agar tumpukan meliputi seluruh bentuk.
    pub spacing_um: Option<f64>,
    /// Rotasi acak per irisan dalam ±nilai ini (derajat).
    pub max_rotation_deg: f64,
    /// Geser acak per irisan dalam ±fraksi sisi gambar.
    pub max_shift_frac: f64,
    /// Izinkan cermin acak per irisan.
    pub mirror: bool,
}

impl Default for PhantomOptions {
    fn default() -> Self {
        PhantomOptions {
            shape: PhantomShapeKind::Ellipsoid,
            slices: 60,
            seed: 1,
            shuffle: false,
            margin_um: None,
            foci: None,
            vessel: false,
            capsule: false,
            breach: false,
            surface_um: None,
            ihc: None,
            size_px: 256,
            mpp: 16.0,
            spacing_um: None,
            max_rotation_deg: 10.0,
            max_shift_frac: 0.04,
            mirror: false,
        }
    }
}

/// Satu irisan di `truth.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PhantomSliceTruth {
    /// Nama berkas PNG.
    pub file: String,
    /// Nama berkas GeoJSON masker; `None` bila irisan tanpa label.
    pub geojson: Option<String>,
    /// Posisi dalam urutan potong sebenarnya (0 = paling bawah).
    pub true_index: usize,
    /// Posisi z sebenarnya, µm.
    pub z_um: f64,
    /// `T_i` sebenarnya: U irisan (µm) → ruang kanonik W (µm), `[a, b, tx, c, d, ty]`.
    pub transform_u_to_w: [f64; 6],
    /// Rotasi yang dipakai, derajat.
    pub rotation_deg: f64,
    /// Irisan dicerminkan.
    pub mirrored: bool,
}

/// Nilai analitik.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct PhantomAnalytic {
    /// Volume tumor, µm³; `None` bila bentuknya tidak punya rumus tertutup.
    pub volume_um3: Option<f64>,
    /// Jarak margin minimum, µm.
    pub d_min_um: Option<f64>,
    /// Jumlah fokus.
    pub n_foci: usize,
    /// Celah antar fokus, µm.
    pub foci_gap_um: Option<f64>,
    /// Luas kontak tumor–pembuluh, mm².
    pub contact_area_mm2: Option<f64>,
    /// Kapsul tembus?
    pub capsule_breached: Option<bool>,
    /// Kedalaman invasi, µm.
    pub depth_um: Option<f64>,
}

/// Kebenaran gambar IHC sintetis.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PhantomIhcTruth {
    /// Nama berkas.
    pub file: String,
    /// Ukuran piksel, µm.
    pub mpp: f64,
    /// Fraksi positif yang diminta.
    pub positive_fraction: f64,
    /// Jumlah inti yang digambar.
    pub n_nuclei: usize,
    /// Jumlah inti positif.
    pub n_positive: usize,
}

/// Isi `truth.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PhantomManifest {
    /// Selalu `"h3d-phantom"`.
    pub format: String,
    /// Versi format.
    pub version: u32,
    /// Bentuk.
    pub shape: PhantomShapeKind,
    /// Benih.
    pub seed: u64,
    /// Ukuran piksel gambar irisan, µm.
    pub mpp: f64,
    /// Ukuran gambar `[w, h]`, piksel.
    pub size_px: [u32; 2],
    /// Jarak irisan, µm.
    pub spacing_um: f64,
    /// Nama berkas diacak?
    pub shuffled: bool,
    /// Nama label `(id fantom, nama di tabel label bawaan proyek)`.
    pub labels: Vec<(u8, String)>,
    /// Irisan, terurut nama berkas.
    pub slices: Vec<PhantomSliceTruth>,
    /// Nilai analitik.
    pub analytic: PhantomAnalytic,
    /// Gambar IHC sintetis, bila diminta.
    pub ihc: Option<PhantomIhcTruth>,
}

/// Nama label fantom di tabel label bawaan proyek.
pub fn phantom_label_name(id: u8) -> &'static str {
    match id {
        LABEL_TUMOR => "Tumor",
        LABEL_MARGIN => "Tepi reseksi",
        LABEL_VESSEL => "Pembuluh darah",
        LABEL_CAPSULE => "Kapsul",
        LABEL_SURFACE => "Permukaan",
        _ => "Lainnya",
    }
}

/// Hash 64-bit per piksel untuk derau deterministik yang tidak bergantung
/// pada urutan iterasi.
fn hash3(seed: u64, a: u64, b: u64, c: u64) -> u64 {
    let mut z = seed
        ^ a.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ b.wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ c.wrapping_mul(0x1656_67B1_9E37_79F9);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Medan tekstur halus 3D bernilai 0..1: jumlah gelombang **berdiri**
/// berbenih, `cos(k_xy · p + φ) × cos(k_z · z + ψ)`.
///
/// Pola di bidang irisan tidak bergeser terhadap z; hanya kontrasnya yang
/// berubah. Gelombang berjalan miring (`cos(k_xy · p + k_z · z)`) sengaja
/// tidak dipakai: polanya bergeser lateral dari irisan ke irisan, dan
/// registrasi berpasangan akan "meluruskan" pergeseran itu (bias searah yang
/// menumpuk di sepanjang rantai, lihat 8.4.6) sehingga kebenaran dasar
/// transformasi tidak lagi bisa dipakai menilai registrasi. Panjang gelombang
/// sepanjang z beberapa kali jarak irisan, sehingga irisan bertetangga
/// berkorelasi kuat tetapi tetap dapat dibedakan urutannya.
struct Texture {
    /// `(k_x, k_y, fase bidang, k_z, fase z)`.
    waves: Vec<([f64; 2], f64, f64, f64)>,
}

impl Texture {
    fn new(seed: u64, field_um: f64) -> Self {
        let mut rng = Rng::new(seed ^ 0x7E57_7E57_7E57_7E57);
        let tau = std::f64::consts::TAU;
        let waves = (0..9)
            .map(|_| {
                let dir = rng.range(0.0, tau);
                let kxy = tau / rng.range(0.06 * field_um, 0.20 * field_um);
                let phase = rng.range(0.0, tau);
                let kz = tau / rng.range(0.25 * field_um, 0.7 * field_um);
                let phase_z = rng.range(0.0, tau);
                ([kxy * dir.cos(), kxy * dir.sin()], phase, kz, phase_z)
            })
            .collect();
        Texture { waves }
    }

    fn at(&self, x: f64, y: f64, z: f64) -> f64 {
        let n = self.waves.len() as f64;
        let s: f64 = self
            .waves
            .iter()
            .map(|(k, ph, kz, phz)| (k[0] * x + k[1] * y + ph).cos() * (kz * z + phz).cos())
            .sum();
        (0.5 + 0.5 * s / n.sqrt() * 1.2).clamp(0.0, 1.0)
    }
}

/// Garis luar jaringan di ruang kanonik: elips + lobus kecil (asimetris dan
/// kiral, sehingga rotasi dan cermin dapat ditentukan).
fn in_tissue(w: [f64; 2], field_um: f64) -> bool {
    let c = field_um / 2.0;
    let (x, y) = (w[0] - c, w[1] - c);
    let e = (x / (0.36 * field_um)).powi(2) + (y / (0.30 * field_um)).powi(2);
    let lx = x - 0.20 * field_um;
    let ly = y + 0.17 * field_um;
    e <= 1.0 || lx * lx + ly * ly <= (0.12 * field_um).powi(2)
}

fn png_bytes(rgba: &[u8], w: u32, h: u32) -> Result<Vec<u8>, EngineError> {
    let mut out = Vec::new();
    let enc = image::codecs::png::PngEncoder::new(&mut out);
    image::ImageEncoder::write_image(enc, rgba, w, h, image::ExtendedColorType::Rgba8)
        .map_err(|e| EngineError::invalid(format!("gagal mengode PNG: {e}")))?;
    Ok(out)
}

/// Tebal kulit kapsul dalam piksel.
const CAPSULE_THICKNESS_PX: f64 = 4.0;

/// Jari-jari tumor untuk fantom kapsul: sebesar mungkin sehingga kulit
/// kapsul (jari-jari luar `1,3·r0 + 7 px`) tercakup tumpukan berjarak irisan
/// `2 × mpp`, dan tidak melebihi 0,18 × sisi bidang.
fn capsule_r0(o: &PhantomOptions, field: f64) -> Result<f64, EngineError> {
    let s = o.mpp;
    let spacing = o.spacing_um.unwrap_or(2.0 * s);
    let half_stack = (o.slices.max(2) - 1) as f64 * spacing / 2.0;
    let r0 = ((half_stack / 1.08 - (3.0 + CAPSULE_THICKNESS_PX) * s) / 1.3).min(0.18 * field);
    if r0 < 5.0 * s {
        return Err(EngineError::invalid(
            "fantom kapsul butuh lebih banyak irisan agar kulit kapsul tercakup (coba --slices 31 atau lebih)",
        ));
    }
    Ok(r0)
}

fn spec_for(o: &PhantomOptions) -> Result<(PhantomSpec, f64), EngineError> {
    let field = f64::from(o.size_px) * o.mpp;
    let n = o.slices;
    let s = o.mpp;
    let mut spec = PhantomSpec {
        n_slices: n,
        s_xy: s,
        size_um: [field, field],
        seed: o.seed,
        ..Default::default()
    };
    // Setengah tinggi (z) yang harus dicakup tumpukan.
    let half_z;
    match o.shape {
        PhantomShapeKind::Ellipsoid => {
            let radii = if o.foci.is_some() {
                [0.10 * field; 3]
            } else if o.capsule {
                let r0 = capsule_r0(o, field)?;
                [r0, 0.9 * r0, 0.85 * r0]
            } else {
                [0.28 * field, 0.22 * field, 0.19 * field]
            };
            spec.shape = PhantomShape::Ellipsoid { radii };
            half_z = if o.foci.is_some() { radii[0] } else { radii[2] };
        }
        PhantomShapeKind::Torus => {
            let (major, minor) = (0.20 * field, 0.08 * field);
            spec.shape = PhantomShape::Torus { major, minor };
            half_z = minor;
        }
        PhantomShapeKind::Branch => {
            let (trunk, arm, radius, half_angle) =
                (0.22 * field, 0.24 * field, 0.055 * field, 0.5f64);
            spec.shape = PhantomShape::Branch {
                trunk_len: trunk,
                arm_len: arm,
                radius,
                half_angle,
            };
            half_z = (trunk + radius).max(arm * half_angle.cos() + radius);
        }
    }
    let mut half_z = half_z;
    if let Some((k, g)) = o.foci {
        if o.shape != PhantomShapeKind::Ellipsoid || k == 0 || !(g.is_finite() && g >= 0.0) {
            return Err(EngineError::invalid(
                "--foci butuh bentuk ellipsoid, jumlah ≥ 1, dan celah ≥ 0",
            ));
        }
        let width = k as f64 * 0.20 * field + (k - 1) as f64 * g;
        if width > 0.70 * field {
            return Err(EngineError::invalid("fokus tidak muat di bidang gambar"));
        }
        spec.foci = Some((k, g));
    }
    if let Some(d) = o.margin_um {
        if !(d.is_finite() && d >= 0.0) {
            return Err(EngineError::invalid("--margin harus ≥ 0"));
        }
        spec.margin_plane = Some(d);
    }
    if o.vessel {
        spec.vessel = Some(0.04 * field);
    }
    if o.capsule {
        if o.shape != PhantomShapeKind::Ellipsoid || o.foci.is_some() {
            return Err(EngineError::invalid(
                "--capsule butuh bentuk ellipsoid tunggal",
            ));
        }
        let t = CAPSULE_THICKNESS_PX * s;
        spec.capsule = Some((t, o.breach));
        let r0 = capsule_r0(o, field)?;
        half_z = r0 * 1.3 + 3.0 * s + t;
    } else if o.breach {
        return Err(EngineError::invalid("--breach butuh --capsule"));
    }
    let spacing = if let Some(h) = o.surface_um {
        if !(h.is_finite() && h > 0.0) || n < 5 {
            return Err(EngineError::invalid(
                "--surface harus > 0 dan butuh minimal 5 irisan",
            ));
        }
        spec.surface_at_z0 = Some(h);
        if o.shape == PhantomShapeKind::Ellipsoid && !o.capsule && o.foci.is_none() {
            spec.shape = PhantomShape::Ellipsoid {
                radii: [0.22 * field, 0.18 * field, h],
            };
        }
        o.spacing_um.unwrap_or(h / (n as f64 - 3.0))
    } else {
        // Sedikit melewati ujung bentuk agar irisan terluar kosong. Kulit
        // kapsul yang tipis hanya utuh bila tidak ada lapis interpolasi di
        // antara irisan, jadi jarak irisannya dipatok 2 × ukuran piksel.
        o.spacing_um.unwrap_or(if o.capsule {
            2.0 * s
        } else {
            2.0 * half_z * 1.08 / (n.max(2) - 1) as f64
        })
    };
    if !(spacing.is_finite() && spacing > 0.0) {
        return Err(EngineError::invalid("jarak irisan tidak sah"));
    }
    spec.spacing_um = spacing;
    Ok((spec, field))
}

/// Tulis fantom ke `out_dir`: `slice_NNN.png`, `slice_NNN.geojson`,
/// `truth.json`, dan (bila diminta) `ihc.png`. Deterministik untuk opsi sama.
pub fn write_phantom(out_dir: &Path, o: &PhantomOptions) -> Result<PhantomManifest, EngineError> {
    if o.slices < 2 || o.slices > 2000 {
        return Err(EngineError::invalid("jumlah irisan harus 2..=2000"));
    }
    if o.size_px < 32 || o.size_px > 8192 || !(o.mpp.is_finite() && o.mpp > 0.0) {
        return Err(EngineError::invalid(
            "ukuran gambar atau mpp fantom tidak sah",
        ));
    }
    let transforms_ok = o.max_rotation_deg.is_finite()
        && o.max_shift_frac.is_finite()
        && (0.0..=0.2).contains(&o.max_shift_frac);
    if !transforms_ok {
        return Err(EngineError::invalid(
            "rentang transformasi fantom tidak sah",
        ));
    }
    if let Some(p) = o.ihc {
        if !(0.0..=1.0).contains(&p) {
            return Err(EngineError::invalid("--ihc harus di antara 0 dan 1"));
        }
    }
    let (spec, field) = spec_for(o)?;
    let phantom = generate_phantom(&spec);
    let n = phantom.slices.len();
    let size = o.size_px as usize;
    let mpp = o.mpp;
    let center = [field / 2.0, field / 2.0];
    let texture = Texture::new(o.seed, field);
    let stain = StainMatrix::he();

    // Transformasi sebenarnya per irisan (urutan potong), berbenih.
    let mut rng = Rng::new(o.seed ^ 0x51CE_0FF5_E7A1_1E55);
    let mut transforms: Vec<(Affine2, f64, bool)> = Vec::with_capacity(n);
    for _ in 0..n {
        let rot_deg = rng.range(-o.max_rotation_deg, o.max_rotation_deg);
        let tx = rng.range(-o.max_shift_frac, o.max_shift_frac) * field;
        let ty = rng.range(-o.max_shift_frac, o.max_shift_frac) * field;
        let mirrored = o.mirror && rng.next_f64() < 0.5;
        let mut g = Affine2::translation(center[0] + tx, center[1] + ty)
            .compose(&Affine2::rotation(rot_deg.to_radians()));
        if mirrored {
            g = g.compose(&Affine2::mirror_x());
        }
        let t = g.compose(&Affine2::translation(-center[0], -center[1]));
        transforms.push((t, rot_deg, mirrored));
    }
    // Nama berkas: urutan potong, atau permutasi berbenih bila `shuffle`.
    let mut file_index: Vec<usize> = (0..n).collect();
    if o.shuffle {
        let mut r = Rng::new(o.seed ^ 0x5AFF_1E00_5AFF_1E00);
        for i in (1..n).rev() {
            let j = r.below(i + 1);
            file_index.swap(i, j);
        }
    }

    std::fs::create_dir_all(out_dir).map_err(|e| h3d_io::IoError::io(out_dir, e))?;
    let mut slices: Vec<PhantomSliceTruth> = Vec::with_capacity(n);
    let mut labels_seen: Vec<u8> = Vec::new();
    for (k, slice) in phantom.slices.iter().enumerate() {
        let (t, rot_deg, mirrored) = transforms[k];
        let z = slice.z_um;
        let mut rgba = vec![255u8; size * size * 4];
        let mut label_map = Array2::<u8>::zeros((size, size));
        for j in 0..size {
            for i in 0..size {
                // Pemetaan mundur: pusat piksel U → titik kanonik W.
                let u = [(i as f64 + 0.5) * mpp, (j as f64 + 0.5) * mpp];
                let w = t.apply(u);
                let cx = (w[0] / mpp).floor();
                let cy = (w[1] / mpp).floor();
                let mut label = 0u8;
                if cx >= 0.0 && cy >= 0.0 && (cx as usize) < size && (cy as usize) < size {
                    for (id, m) in &slice.masks {
                        if m[[cy as usize, cx as usize]] != 0 {
                            label = *id;
                            break;
                        }
                    }
                }
                label_map[[j, i]] = label;
                if !in_tissue(w, field) && label != LABEL_TUMOR {
                    continue;
                }
                let f = texture.at(w[0], w[1], z);
                let noise = |salt: u64| {
                    (hash3(o.seed, k as u64, (j * size + i) as u64, salt) % 1000) as f64 / 1000.0
                        - 0.5
                };
                let tumor_boost = if label == LABEL_TUMOR { 0.22 } else { 0.0 };
                let h = (0.10 + 0.45 * f + tumor_boost + 0.03 * noise(1)) as f32;
                let e = (0.30 + 0.04 * noise(2)) as f32;
                let od = stain.od_from_concentrations([h, e, 0.0]);
                let d = (j * size + i) * 4;
                for c in 0..3 {
                    rgba[d + c] = intensity_of_od(od[c]);
                }
            }
        }
        let fi = file_index[k];
        let file = format!("slice_{fi:03}.png");
        write_atomic(
            &out_dir.join(&file),
            &png_bytes(&rgba, o.size_px, o.size_px)?,
        )?;

        // Masker → GeoJSON (koordinat piksel level 0 irisan ini).
        let mut present: Vec<u8> = label_map.iter().copied().filter(|v| *v != 0).collect();
        present.sort_unstable();
        present.dedup();
        let geojson = if present.is_empty() {
            None
        } else {
            let mut store = LabelStore::with_size_px(mpp, [o.size_px, o.size_px], 256);
            for id in &present {
                let mask = label_map.mapv(|v| u8::from(v == *id));
                store.rasterize_mask(&mask, [0, 0], LabelId(*id), PaintMode::Replace);
                if !labels_seen.contains(id) {
                    labels_seen.push(*id);
                }
            }
            let named: Vec<(LabelId, &str)> = present
                .iter()
                .map(|id| (LabelId(*id), phantom_label_name(*id)))
                .collect();
            let text = h3d_segment::export_geojson(&store, &named, [mpp, mpp], None);
            let name = format!("slice_{fi:03}.geojson");
            write_atomic(&out_dir.join(&name), text.as_bytes())?;
            Some(name)
        };
        slices.push(PhantomSliceTruth {
            file,
            geojson,
            true_index: k,
            z_um: z,
            transform_u_to_w: t.m,
            rotation_deg: rot_deg,
            mirrored,
        });
    }
    slices.sort_by(|a, b| a.file.cmp(&b.file));
    labels_seen.sort_unstable();

    let ihc = match o.ihc {
        None => None,
        Some(fraction) => {
            let side = 512usize;
            let mask = ellipsoid_mask([side, side], [256.0, 256.0], [225.0, 205.0]);
            let (rgba, truth) = synthetic_ihc(&mask, o.seed, fraction, 1.0);
            let file = "ihc.png".to_owned();
            write_atomic(
                &out_dir.join(&file),
                &png_bytes(&rgba, side as u32, side as u32)?,
            )?;
            Some(PhantomIhcTruth {
                file,
                mpp: 1.0,
                positive_fraction: fraction,
                n_nuclei: truth.len(),
                n_positive: truth.iter().filter(|t| t.2).count(),
            })
        }
    };

    let t = &phantom.truth;
    let manifest = PhantomManifest {
        format: "h3d-phantom".to_owned(),
        version: 1,
        shape: o.shape,
        seed: o.seed,
        mpp,
        size_px: [o.size_px, o.size_px],
        spacing_um: spec.spacing_um,
        shuffled: o.shuffle,
        labels: labels_seen
            .iter()
            .map(|id| (*id, phantom_label_name(*id).to_owned()))
            .collect(),
        slices,
        analytic: PhantomAnalytic {
            volume_um3: (t.volume_um3.is_finite() && t.volume_um3 > 0.0).then_some(t.volume_um3),
            d_min_um: t.d_min,
            n_foci: t.n_foci,
            foci_gap_um: t.foci_gap_um,
            contact_area_mm2: t.contact_area_mm2,
            capsule_breached: t.capsule_breached,
            depth_um: t.depth_um,
        },
        ihc,
    };
    // Presisi penuh (bukan format enam desimal): transformasi kebenaran
    // harus terbaca kembali persis.
    let truth_path = out_dir.join("truth.json");
    let mut text = serde_json::to_string_pretty(&manifest)
        .map_err(|e| h3d_io::IoError::json(&truth_path, e))?;
    text.push('\n');
    write_atomic(&truth_path, text.as_bytes())?;
    Ok(manifest)
}

/// Baca `truth.json` sebuah folder fantom.
pub fn read_phantom_manifest(dir: &Path) -> Result<PhantomManifest, EngineError> {
    let path: PathBuf = dir.join("truth.json");
    let text = std::fs::read_to_string(&path).map_err(|e| h3d_io::IoError::io(&path, e))?;
    let m: PhantomManifest =
        serde_json::from_str(&text).map_err(|e| h3d_io::IoError::json(&path, e))?;
    if m.format != "h3d-phantom" {
        return Err(h3d_io::IoError::invalid(&path, "bukan truth.json fantom HISTO3D").into());
    }
    Ok(m)
}
