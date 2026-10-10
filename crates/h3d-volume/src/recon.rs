//! Rekonstruksi 3D per label (histo3d.md 8.6.2): perencanaan kisi,
//! resampling mundur, SDF 2D per irisan, lapis Z interpolasi, ekstraksi
//! permukaan per chunk dengan `fast-surface-nets`, pasca-proses mesh.
//!
//! # Konvensi
//! - `SliceInput::transform` adalah `T_i`: **U → W**. Resampling memakai
//!   `T_i⁻¹` (W → U) pada pusat voxel kisi (aturan keras 4).
//! - SDF **negatif di dalam**, satuan µm, dijepit `±band`.
//! - Buffer 3D: indeks linear `x + nx·(y + ny·z)`.
//! - Deterministik: hasil paralel ditulis ke lokasi tetap atau digabung
//!   dalam urutan chunk tetap.

use std::collections::HashMap;

use fast_surface_nets::{surface_nets, SurfaceNetsBuffer};
use h3d_core::{Affine2, Budgets, LabelId, Progress, ReconConfig, SlideId, SlideTransform};
use h3d_imgproc::edt::signed_distance_2d_um;
use ndarray::Array2;
use ndshape::RuntimeShape;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::error::VolError;
use crate::grid::StackGrid;
use crate::measure::Warning;
use crate::mesh::LabelMesh;

/// Sumber label raster satu irisan, dibaca di ruang U (µm). Abstraksi atas
/// `LabelStore` milik `h3d-segment` agar crate ini tidak bergantung padanya.
pub trait LabelSampler: Sync {
    /// Label pada titik U `u` (µm); 0 = latar atau di luar.
    fn label_at_um(&self, u: [f64; 2]) -> u8;
    /// Resolusi label, µm/px.
    fn mpp_label(&self) -> f64;
    /// Kotak pembatas `[xmin, ymin, xmax, ymax]` (µm, ruang U) piksel
    /// berlabel `label`; `None` bila label tidak ada di irisan ini.
    fn bbox_um(&self, label: u8) -> Option<[f64; 4]>;
}

/// Satu irisan masukan rekonstruksi.
pub struct SliceInput<'a> {
    /// Pengenal irisan.
    pub id: SlideId,
    /// Posisi z, µm.
    pub z_um: f64,
    /// `T_i`: U → W.
    pub transform: &'a SlideTransform,
    /// Sumber label.
    pub labels: &'a dyn LabelSampler,
    /// Label yang ada di irisan ini: `(label, hasil interpolasi?)`.
    pub annotated_labels: Vec<(u8, bool)>,
    /// Label yang anotasinya berstatus `Reviewed` (untuk laporan).
    pub reviewed_labels: Vec<u8>,
}

impl SliceInput<'_> {
    /// Benar bila irisan memuat `label` (interpolasi atau tidak).
    pub fn has_label(&self, label: u8) -> bool {
        self.annotated_labels.iter().any(|&(l, _)| l == label)
    }

    /// `Some(interpolated)` bila irisan memuat `label`.
    pub fn label_flag(&self, label: u8) -> Option<bool> {
        self.annotated_labels
            .iter()
            .find(|&&(l, _)| l == label)
            .map(|&(_, i)| i)
    }
}

/// `s_xy` maksimum yang dicoba saat menaikkan resolusi otomatis (µm).
pub const MAX_S_XY_UM: f64 = 64.0;
/// Ukuran chunk ekstraksi (voxel per sisi).
pub const CHUNK: usize = 64;

/// Hasil perencanaan kisi.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GridPlan {
    /// Kisi keluaran.
    pub grid: StackGrid,
    /// `s_xy` terpilih (mungkin lebih besar dari `cfg.s_xy_um`).
    pub s_xy: f64,
    /// Celah maksimum sebelum objek diputus, µm.
    pub max_gap_um: f64,
    /// Jarak irisan median, µm.
    pub median_spacing_um: f64,
    /// Perkiraan memori set kerja, byte.
    pub estimated_bytes: u64,
    /// Catatan untuk laporan (bahasa Indonesia).
    pub notes: Vec<String>,
    /// Peringatan terstruktur.
    pub warnings: Vec<Warning>,
}

fn median(v: &mut [f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let m = v.len() / 2;
    Some(if v.len() % 2 == 1 {
        v[m]
    } else {
        (v[m - 1] + v[m]) / 2.0
    })
}

/// Lapis z keluaran dari daftar z irisan beranotasi (terurut, unik).
fn plan_z_layers(
    annotated_z: &[f64],
    half_spacing: f64,
    max_gap: f64,
    s_z_target: f64,
    warnings: &mut Vec<Warning>,
) -> Vec<f64> {
    let m = annotated_z.len();
    let mut z = Vec::new();
    if m == 0 {
        return z;
    }
    // Lapis menuju tutup: `cap` dibagi rata agar tiap langkah ≤ s_z target.
    let cap_steps = |cap: f64| -> usize { (cap / s_z_target).ceil().max(1.0) as usize };
    let n_end = cap_steps(half_spacing);
    for j in (1..=n_end).rev() {
        z.push(annotated_z[0] - half_spacing * j as f64 / n_end as f64);
    }
    for i in 0..m {
        z.push(annotated_z[i]);
        if i + 1 < m {
            let gap = annotated_z[i + 1] - annotated_z[i];
            if gap > max_gap {
                let cap = half_spacing.min(gap / 3.0);
                let n = cap_steps(cap);
                for j in 1..=n {
                    z.push(annotated_z[i] + cap * j as f64 / n as f64);
                }
                for j in (1..=n).rev() {
                    z.push(annotated_z[i + 1] - cap * j as f64 / n as f64);
                }
                warnings.push(Warning::GapSplit {
                    from_z: annotated_z[i],
                    to_z: annotated_z[i + 1],
                    gap,
                });
            } else if gap > 0.0 {
                let n = (gap / s_z_target).ceil().max(1.0) as usize;
                for j in 1..n {
                    z.push(annotated_z[i] + gap * j as f64 / n as f64);
                }
            }
        }
    }
    for j in 1..=n_end {
        z.push(annotated_z[m - 1] + half_spacing * j as f64 / n_end as f64);
    }
    z
}

/// Perkiraan memori set kerja: SDF 2D semua irisan sumber + jendela geser
/// `CHUNK + 2` lapis `f32` + masker voxel `u8` per label.
fn estimate_bytes(nx: u64, ny: u64, nz: u64, n_src: u64, n_labels: u64) -> u64 {
    let layer = nx * ny;
    let window = nz.min(CHUNK as u64 + 2);
    layer * (4 * (n_src + window) + nz * n_labels.max(1))
}

fn next_s_xy(s: f64) -> f64 {
    if s < 24.0 {
        24.0
    } else if s < 32.0 {
        32.0
    } else {
        s * 2.0
    }
}

/// Rencanakan kisi untuk `labels_to_recon` (8.6.1, 8.6.5): kotak pembatas
/// gabungan label setelah `T_i`, margin `2 × band`, lapis Z dengan
/// `s_z ≤ k × s_xy`, pemutusan pada celah besar, tutup ujung pada setengah
/// jarak irisan, dan kenaikan `s_xy` otomatis bila melampaui anggaran.
pub fn plan_grid(
    inputs: &[SliceInput<'_>],
    labels_to_recon: &[u8],
    cfg: &ReconConfig,
    budget: &Budgets,
) -> Result<GridPlan, VolError> {
    let positive = |v: f64| v.is_finite() && v > 0.0;
    if !positive(cfg.s_xy_um) || !positive(cfg.z_factor_max) {
        return Err(VolError::InvalidParameter(
            "s_xy_um dan z_factor_max harus positif".into(),
        ));
    }
    // Kotak pembatas gabungan di W dan daftar z beranotasi.
    let mut bbox: Option<[f64; 4]> = None;
    let mut annotated_z: Vec<f64> = Vec::new();
    for s in inputs {
        let mut any = false;
        for &label in labels_to_recon {
            if !s.has_label(label) {
                continue;
            }
            let Some(b) = s.labels.bbox_um(label) else {
                continue;
            };
            any = true;
            for corner in [[b[0], b[1]], [b[2], b[1]], [b[0], b[3]], [b[2], b[3]]] {
                let w = s.transform.apply(corner);
                bbox = Some(match bbox {
                    None => [w[0], w[1], w[0], w[1]],
                    Some(bb) => [
                        bb[0].min(w[0]),
                        bb[1].min(w[1]),
                        bb[2].max(w[0]),
                        bb[3].max(w[1]),
                    ],
                });
            }
        }
        if any {
            annotated_z.push(s.z_um);
        }
    }
    let Some(bbox) = bbox else {
        return Err(VolError::NoLabelledSlices);
    };
    annotated_z.sort_by(|a, b| a.total_cmp(b));
    annotated_z.dedup();
    let mut diffs: Vec<f64> = annotated_z
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|d| *d > 0.0)
        .collect();
    let median_spacing = median(&mut diffs).unwrap_or(cfg.default_spacing_um);
    let median_spacing = if median_spacing > 0.0 {
        median_spacing
    } else {
        cfg.default_spacing_um.max(1.0)
    };
    let max_gap = cfg.max_gap_um.unwrap_or(5.0 * median_spacing);
    let half = median_spacing / 2.0;

    let mut notes = Vec::new();
    let mut s_xy = cfg.s_xy_um;
    let n_src = annotated_z.len() as u64;
    let n_labels = labels_to_recon.len() as u64;
    loop {
        let mut warnings = Vec::new();
        let z = plan_z_layers(
            &annotated_z,
            half,
            max_gap,
            cfg.z_factor_max * s_xy,
            &mut warnings,
        );
        let grid = StackGrid::from_bbox(bbox, s_xy, cfg.band_px, z);
        let est = estimate_bytes(
            grid.size[0] as u64,
            grid.size[1] as u64,
            grid.nz() as u64,
            n_src,
            n_labels,
        );
        if est <= budget.recon_working_set_bytes {
            if s_xy != cfg.s_xy_um {
                warnings.insert(
                    0,
                    Warning::SxyRaised {
                        from: cfg.s_xy_um,
                        to: s_xy,
                    },
                );
            }
            for w in &warnings {
                if let Warning::GapSplit { from_z, to_z, gap } = w {
                    notes.push(format!(
                        "Celah {gap:.0} µm antara irisan z = {from_z:.0} dan z = {to_z:.0} \
                         melampaui batas {max_gap:.0} µm; objek diputus di situ."
                    ));
                }
            }
            return Ok(GridPlan {
                grid,
                s_xy,
                max_gap_um: max_gap,
                median_spacing_um: median_spacing,
                estimated_bytes: est,
                notes,
                warnings,
            });
        }
        let next = next_s_xy(s_xy);
        if next > MAX_S_XY_UM {
            return Err(VolError::GridTooLarge {
                needed_bytes: est,
                budget_bytes: budget.recon_working_set_bytes,
                s_xy_um: s_xy,
            });
        }
        notes.push(format!(
            "Perkiraan memori {:.0} MB melampaui anggaran {:.0} MB; s_xy dinaikkan dari {s_xy} ke {next} µm.",
            est as f64 / 1_048_576.0,
            budget.recon_working_set_bytes as f64 / 1_048_576.0
        ));
        log::info!("s_xy dinaikkan {s_xy} → {next} µm (perkiraan {est} B)");
        s_xy = next;
    }
}

/// Pemetaan mundur W → U yang sudah dipra-hitung.
enum InverseMap<'a> {
    Affine(Affine2),
    General(&'a SlideTransform),
}

impl InverseMap<'_> {
    fn new(t: &SlideTransform, id: SlideId) -> Result<InverseMap<'_>, VolError> {
        match t {
            SlideTransform::Identity => Ok(InverseMap::Affine(Affine2::IDENTITY)),
            SlideTransform::Affine(a) => a.inverse().map(InverseMap::Affine).ok_or_else(|| {
                VolError::InvalidParameter(format!("transformasi irisan {id} singular"))
            }),
            SlideTransform::AffineBSpline { affine, .. } => {
                if affine.inverse().is_none() {
                    return Err(VolError::InvalidParameter(format!(
                        "transformasi irisan {id} singular"
                    )));
                }
                Ok(InverseMap::General(t))
            }
        }
    }

    #[inline]
    fn apply(&self, w: [f64; 2]) -> Option<[f64; 2]> {
        match self {
            InverseMap::Affine(a) => Some(a.apply(w)),
            InverseMap::General(t) => t.apply_inverse(w),
        }
    }
}

/// Resampling masker satu irisan ke kisi: `u = T⁻¹(w)` pada tiap pusat voxel.
fn resample_one(
    grid: &StackGrid,
    input: &SliceInput<'_>,
    label: u8,
) -> Result<Array2<u8>, VolError> {
    let inv = InverseMap::new(input.transform, input.id)?;
    let nx = grid.size[0] as usize;
    let ny = grid.size[1] as usize;
    let mut data = vec![0u8; nx * ny];
    data.par_chunks_mut(nx).enumerate().for_each(|(j, row)| {
        for (i, px) in row.iter_mut().enumerate() {
            let w = grid.index_to_um(i as u32, j as u32);
            if let Some(u) = inv.apply(w) {
                *px = u8::from(input.labels.label_at_um(u) == label);
            }
        }
    });
    Array2::from_shape_vec((ny, nx), data).map_err(|e| VolError::InconsistentSizes(e.to_string()))
}

/// Masker label `label` di kisi untuk tiap irisan masukan (`None` bila irisan
/// tidak memuat label). Pemetaan mundur `u = T_i⁻¹(w)`; paralel per baris.
pub fn resample_masks(
    grid: &StackGrid,
    inputs: &[SliceInput<'_>],
    label: u8,
    progress: &dyn Progress,
) -> Result<Vec<Option<Array2<u8>>>, VolError> {
    let mut out = Vec::with_capacity(inputs.len());
    for (k, s) in inputs.iter().enumerate() {
        if progress.is_cancelled() {
            return Err(VolError::Cancelled);
        }
        progress.set(
            k as f32 / inputs.len().max(1) as f32,
            "Resampling masker label",
        );
        out.push(if s.has_label(label) {
            Some(resample_one(grid, s, label)?)
        } else {
            None
        });
    }
    Ok(out)
}

/// Asal satu lapis keluaran untuk sebuah label.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum LayerSource {
    /// Irisan masukan ke-`input` (indeks dalam daftar masukan).
    Annotated {
        /// Indeks masukan.
        input: usize,
    },
    /// Interpolasi linear SDF antara dua irisan masukan.
    Interpolated {
        /// Irisan bawah.
        a: usize,
        /// Irisan atas.
        b: usize,
        /// Bobot `t ∈ (0, 1)` ke arah `b`.
        t: f64,
    },
    /// Menuju tutup ujung/celah: `(1 − t)·sdf + t·band`, `t ∈ (0, 1)`.
    TowardCap {
        /// Irisan terdekat.
        input: usize,
        /// Bobot ke arah tutup.
        t: f64,
    },
    /// Tutup ujung / celah: SDF = `+band`, masker kosong.
    Cap,
}

/// Tentukan asal tiap lapis kisi untuk `label`. `cap_um` = jarak tutup
/// ujung dari irisan terluar (setengah jarak irisan median); pada celah
/// yang diputus jarak tutup = `min(cap_um, celah / 3)`.
pub fn layer_sources(
    grid: &StackGrid,
    inputs: &[SliceInput<'_>],
    label: u8,
    max_gap_um: f64,
    cap_um: f64,
) -> Vec<LayerSource> {
    // Irisan yang memuat label, terurut z (seri → indeks terkecil).
    let mut ann: Vec<(f64, usize)> = inputs
        .iter()
        .enumerate()
        .filter(|(_, s)| s.has_label(label))
        .map(|(i, s)| (s.z_um, i))
        .collect();
    ann.sort_by(|p, q| p.0.total_cmp(&q.0).then(p.1.cmp(&q.1)));
    ann.dedup_by(|a, b| (a.0 - b.0).abs() <= 1e-6);
    const EPS: f64 = 1e-6;
    grid.z_um
        .iter()
        .map(|&z| {
            if let Some(&(_, i)) = ann.iter().find(|(za, _)| (za - z).abs() <= EPS) {
                return LayerSource::Annotated { input: i };
            }
            let below = ann.iter().rev().find(|(za, _)| *za < z);
            let above = ann.iter().find(|(za, _)| *za > z);
            match (below, above) {
                (Some(&(za, ia)), Some(&(zb, ib))) if zb - za <= max_gap_um && zb > za => {
                    LayerSource::Interpolated {
                        a: ia,
                        b: ib,
                        t: (z - za) / (zb - za),
                    }
                }
                (below, above) => {
                    // Tidak dijembatani: menuju tutup dari irisan terdekat
                    // (seri → irisan bawah).
                    let cap = match (below, above) {
                        (Some(&(za, _)), Some(&(zb, _))) => cap_um.min((zb - za) / 3.0),
                        _ => cap_um,
                    };
                    let db = below.map(|&(za, i)| (z - za, i));
                    let da = above.map(|&(zb, i)| (zb - z, i));
                    let nearest = match (db, da) {
                        (Some(b), Some(a)) => Some(if a.0 < b.0 { a } else { b }),
                        (b, a) => b.or(a),
                    };
                    match nearest {
                        Some((d, input)) if cap > 0.0 && d < cap - EPS => {
                            LayerSource::TowardCap { input, t: d / cap }
                        }
                        _ => LayerSource::Cap,
                    }
                }
            }
        })
        .collect()
}

/// Masker voxel satu label pada semua lapis keluaran (termasuk lapis
/// interpolasi). Indeks linear `x + nx·(y + ny·z)`; nilai 0/1.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VoxelMask {
    /// Kisi.
    pub grid: StackGrid,
    /// Data voxel.
    pub data: Vec<u8>,
    /// Per lapis: `Some((irisan, interpolasi?))` bila lapis berasal dari
    /// irisan beranotasi, `None` untuk lapis interpolasi/tutup.
    pub annotated_layer: Vec<Option<(SlideId, bool)>>,
}

impl VoxelMask {
    /// Masker kosong atas `grid`.
    pub fn empty(grid: StackGrid) -> Self {
        let n = grid.voxel_count();
        let nz = grid.nz();
        VoxelMask {
            grid,
            data: vec![0; n],
            annotated_layer: vec![None; nz],
        }
    }

    /// Nilai voxel `(x, y, z)`.
    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> u8 {
        self.data[self.grid.linear(x, y, z)]
    }

    /// Jumlah voxel bernilai 1.
    pub fn count(&self) -> usize {
        self.data.iter().filter(|&&v| v != 0).count()
    }

    /// Volume voxel bernilai 1 (µm³) memakai jarak lapis setempat.
    pub fn volume_um3(&self) -> f64 {
        let layer = self.grid.layer_len();
        let mut v = 0.0;
        for k in 0..self.grid.nz() {
            let c = self.data[k * layer..(k + 1) * layer]
                .iter()
                .filter(|&&m| m != 0)
                .count();
            v += c as f64 * self.grid.voxel_volume_um3(k);
        }
        v
    }

    /// Benar bila ada voxel 1 pada lapis `k`.
    pub fn layer_any(&self, k: usize) -> bool {
        let layer = self.grid.layer_len();
        self.data[k * layer..(k + 1) * layer]
            .iter()
            .any(|&m| m != 0)
    }

    /// Gabungan (OR) dua masker pada kisi yang sama.
    pub fn union_with(&mut self, other: &VoxelMask) -> Result<(), VolError> {
        if other.data.len() != self.data.len() || other.grid.size != self.grid.size {
            return Err(VolError::InconsistentSizes(
                "masker voxel berbeda kisi".into(),
            ));
        }
        for (a, &b) in self.data.iter_mut().zip(&other.data) {
            *a |= u8::from(b != 0);
        }
        Ok(())
    }
}

/// Penyedia lapis SDF untuk ekstraksi permukaan. Implementasi menghasilkan
/// lapis sesuai kebutuhan (jendela geser), sehingga SDF 3D penuh tidak
/// pernah dialokasikan.
pub trait SdfProvider: Sync {
    /// Kisi.
    fn grid(&self) -> &StackGrid;
    /// Isi `out` (panjang `nx × ny`, indeks `x + nx·y`) dengan SDF lapis `k`
    /// dalam µm, negatif di dalam.
    fn fill_layer(&self, k: usize, out: &mut [f32]);
}

/// Tumpukan SDF satu label: masker dan SDF 2D per irisan beranotasi, asal
/// tiap lapis keluaran.
pub struct LabelStack {
    /// Kisi.
    pub grid: StackGrid,
    /// Label.
    pub label: u8,
    /// Lebar pita SDF, µm.
    pub band_um: f32,
    /// Asal tiap lapis.
    pub layers: Vec<LayerSource>,
    /// Masker per irisan masukan.
    pub masks: Vec<Option<Array2<u8>>>,
    /// SDF per irisan masukan (µm, negatif di dalam, dijepit `±band`).
    pub sdfs: Vec<Option<Array2<f32>>>,
    /// `(id, interpolasi?)` per irisan masukan untuk label ini.
    pub slide_flags: Vec<Option<(SlideId, bool)>>,
}

impl LabelStack {
    /// Masker voxel pada semua lapis keluaran.
    pub fn voxel_mask(&self) -> VoxelMask {
        let layer = self.grid.layer_len();
        let mut m = VoxelMask::empty(self.grid.clone());
        let mut buf = vec![0f32; layer];
        for (k, src) in self.layers.iter().enumerate() {
            let out = &mut m.data[k * layer..(k + 1) * layer];
            match *src {
                LayerSource::Annotated { input } => {
                    if let Some(mask) = &self.masks[input] {
                        for (o, &v) in out.iter_mut().zip(mask.iter()) {
                            *o = u8::from(v != 0);
                        }
                    }
                    m.annotated_layer[k] = self.slide_flags[input];
                }
                LayerSource::Interpolated { .. } | LayerSource::TowardCap { .. } => {
                    self.fill_layer(k, &mut buf);
                    for (o, &v) in out.iter_mut().zip(&buf) {
                        *o = u8::from(v < 0.0);
                    }
                }
                LayerSource::Cap => {}
            }
        }
        m
    }
}

impl SdfProvider for LabelStack {
    fn grid(&self) -> &StackGrid {
        &self.grid
    }

    fn fill_layer(&self, k: usize, out: &mut [f32]) {
        let band = self.band_um;
        match self.layers.get(k).copied().unwrap_or(LayerSource::Cap) {
            LayerSource::Annotated { input } => match &self.sdfs[input] {
                Some(s) => {
                    for (o, &v) in out.iter_mut().zip(s.iter()) {
                        *o = v;
                    }
                }
                None => out.fill(band),
            },
            LayerSource::Interpolated { a, b, t } => match (&self.sdfs[a], &self.sdfs[b]) {
                (Some(sa), Some(sb)) => {
                    let t = t as f32;
                    for ((o, &va), &vb) in out.iter_mut().zip(sa.iter()).zip(sb.iter()) {
                        *o = (1.0 - t) * va + t * vb;
                    }
                }
                _ => out.fill(band),
            },
            LayerSource::TowardCap { input, t } => match &self.sdfs[input] {
                Some(s) => {
                    let t = t as f32;
                    for (o, &v) in out.iter_mut().zip(s.iter()) {
                        *o = (1.0 - t) * v + t * band;
                    }
                }
                None => out.fill(band),
            },
            LayerSource::Cap => out.fill(band),
        }
    }
}

/// Bangun [`LabelStack`] untuk `label`: resampling (langkah 1), SDF 2D
/// (langkah 2), asal lapis (langkah 3–5).
pub fn build_label_stack(
    plan: &GridPlan,
    inputs: &[SliceInput<'_>],
    label: u8,
    cfg: &ReconConfig,
    progress: &dyn Progress,
) -> Result<LabelStack, VolError> {
    let grid = plan.grid.clone();
    let band_um = (cfg.band_px.max(1.0) * grid.s_xy) as f32;
    let masks = resample_masks(&grid, inputs, label, progress)?;
    if progress.is_cancelled() {
        return Err(VolError::Cancelled);
    }
    progress.set(0.5, "Menghitung SDF 2D");
    let sdfs: Vec<Option<Array2<f32>>> = masks
        .par_iter()
        .map(|m| {
            m.as_ref()
                .map(|mask| signed_distance_2d_um(mask, grid.s_xy, band_um))
        })
        .collect();
    let slide_flags = inputs
        .iter()
        .map(|s| s.label_flag(label).map(|i| (s.id, i)))
        .collect();
    let layers = layer_sources(
        &grid,
        inputs,
        label,
        plan.max_gap_um,
        plan.median_spacing_um / 2.0,
    );
    progress.set(1.0, "Tumpukan SDF siap");
    Ok(LabelStack {
        grid,
        label,
        band_um,
        layers,
        masks,
        sdfs,
        slide_flags,
    })
}

/// Masker voxel label pada semua lapis keluaran (termasuk interpolasi).
pub fn build_voxel_mask(
    plan: &GridPlan,
    inputs: &[SliceInput<'_>],
    label: u8,
    cfg: &ReconConfig,
    progress: &dyn Progress,
) -> Result<VoxelMask, VolError> {
    Ok(build_label_stack(plan, inputs, label, cfg, progress)?.voxel_mask())
}

/// Hasil mentah satu chunk: kunci voxel global, posisi kisi global, indeks lokal.
struct ChunkOut {
    keys: Vec<[u32; 3]>,
    positions: Vec<[f64; 3]>,
    indices: Vec<u32>,
}

#[inline]
fn key_u64(k: [u32; 3]) -> u64 {
    // 21 bit per sumbu cukup (≤ 2 juta voxel per sisi).
    ((k[0] as u64) & 0x1F_FFFF) | (((k[1] as u64) & 0x1F_FFFF) << 21) | ((k[2] as u64) << 42)
}

/// Nilai SDF lapis bantalan di luar kisi (µm, positif = luar).
const PAD_SDF: f32 = 1.0e6;

/// Ekstraksi permukaan (langkah 6–11): chunk 64³ dengan tumpang 1 voxel,
/// `fast-surface-nets` per chunk, las verteks lewat kunci voxel global,
/// skala ke µm, Taubin, decimate, normal. Segitiga diorientasi ke luar
/// (volume bertanda positif).
///
/// Kisi diperlakukan seolah dikelilingi bantalan SDF positif (1 lapis di
/// sisi minimum, 2 lapis di sisi maksimum tiap sumbu) karena surface nets
/// tidak membuat muka pada bidang batas larik; dengan begitu permukaan
/// selalu tertutup walau SDF masih negatif di tepi kisi.
pub fn extract_mesh(
    sdf: &dyn SdfProvider,
    label: LabelId,
    cfg: &ReconConfig,
    progress: &dyn Progress,
) -> Result<LabelMesh, VolError> {
    let grid = sdf.grid().clone();
    let [nx, ny, nz] = grid.dims();
    let mut mesh = LabelMesh::empty(label);
    if nx == 0 || ny == 0 || nz == 0 {
        return Ok(mesh);
    }
    let layer = nx * ny;
    // Lattice diperluas: indeks e = indeks nyata + 1; ukuran n + 3.
    let ext = [nx + 3, ny + 3, nz + 3];
    let n_cubes = ext.map(|e| e - 1);
    let n_chunks = n_cubes.map(|c| c.div_ceil(CHUNK));
    // Rentang lattice (inklusif) chunk ke-`c` pada sumbu `a`: satu lapis
    // tumpang di sisi minimum; sisi maksimum = sudut kubus terakhir.
    let range = |c: usize, a: usize| -> (usize, usize) {
        let c0 = c * CHUNK;
        let c1 = ((c + 1) * CHUNK).min(n_cubes[a]);
        (c0.saturating_sub(1), c1)
    };
    let mut key_to_vertex: HashMap<u64, u32> = HashMap::new();
    let mut slab: Vec<f32> = Vec::new();

    for cz in 0..n_chunks[2] {
        if progress.is_cancelled() {
            return Err(VolError::Cancelled);
        }
        progress.set(
            0.8 * cz as f32 / n_chunks[2] as f32,
            "Ekstraksi permukaan (surface nets)",
        );
        let (minz, maxz) = range(cz, 2);
        let nlz = maxz - minz + 1;
        // Jendela geser: hanya lapis chunk ini yang ada di memori.
        slab.resize(nlz * layer, 0.0);
        slab.par_chunks_mut(layer).enumerate().for_each(|(j, out)| {
            let ez = minz + j;
            if ez >= 1 && ez <= nz {
                sdf.fill_layer(ez - 1, out);
            } else {
                out.fill(PAD_SDF);
            }
        });

        let mut chunk_list: Vec<(usize, usize)> = Vec::new();
        for cy in 0..n_chunks[1] {
            for cx in 0..n_chunks[0] {
                chunk_list.push((cx, cy));
            }
        }
        let slab_ref = &slab;
        let outs: Vec<ChunkOut> = chunk_list
            .par_iter()
            .map(|&(cx, cy)| {
                let (minx, maxx) = range(cx, 0);
                let (miny, maxy) = range(cy, 1);
                let (lx, ly, lz) = (maxx - minx + 1, maxy - miny + 1, nlz);
                let mut local = vec![PAD_SDF; lx * ly * lz];
                // Bagian nyata sumbu x: indeks diperluas 1..=nx.
                let rx0 = minx.max(1);
                let rx1 = maxx.min(nx);
                if rx0 <= rx1 {
                    for z in 0..lz {
                        for y in 0..ly {
                            let ey = miny + y;
                            if ey < 1 || ey > ny {
                                continue;
                            }
                            let src = z * layer + (ey - 1) * nx + (rx0 - 1);
                            let dst = (z * ly + y) * lx + (rx0 - minx);
                            let len = rx1 - rx0 + 1;
                            local[dst..dst + len].copy_from_slice(&slab_ref[src..src + len]);
                        }
                    }
                }
                let shape = RuntimeShape::<u32, 3>::new([lx as u32, ly as u32, lz as u32]);
                let mut buf = SurfaceNetsBuffer::default();
                surface_nets(
                    &local,
                    &shape,
                    [0, 0, 0],
                    [lx as u32 - 1, ly as u32 - 1, lz as u32 - 1],
                    &mut buf,
                );
                let keys = buf
                    .surface_points
                    .iter()
                    .map(|p| [p[0] + minx as u32, p[1] + miny as u32, p[2] + minz as u32])
                    .collect();
                // Posisi dalam koordinat lattice nyata (diperluas − 1).
                let positions = buf
                    .positions
                    .iter()
                    .map(|p| {
                        [
                            p[0] as f64 + minx as f64 - 1.0,
                            p[1] as f64 + miny as f64 - 1.0,
                            p[2] as f64 + minz as f64 - 1.0,
                        ]
                    })
                    .collect();
                ChunkOut {
                    keys,
                    positions,
                    indices: buf.indices,
                }
            })
            .collect();

        // Gabung dalam urutan chunk tetap (deterministik).
        for out in outs {
            let mut local_to_global = Vec::with_capacity(out.keys.len());
            for (k, p) in out.keys.iter().zip(&out.positions) {
                let key = key_u64(*k);
                let idx = *key_to_vertex.entry(key).or_insert_with(|| {
                    mesh.positions_um.push(*p);
                    (mesh.positions_um.len() - 1) as u32
                });
                local_to_global.push(idx);
            }
            mesh.indices
                .extend(out.indices.iter().map(|&i| local_to_global[i as usize]));
        }
    }
    // Verteks tumpang yang tidak dipakai segitiga mana pun dibuang.
    mesh.compact();

    // Langkah 8: ke µm.
    for p in &mut mesh.positions_um {
        let z = grid.z_at_continuous(p[2]);
        *p = [
            grid.origin_um[0] + (p[0] + 0.5) * grid.s_xy,
            grid.origin_um[1] + (p[1] + 0.5) * grid.s_xy,
            z,
        ];
    }
    // Orientasi: volume bertanda harus positif.
    mesh.ensure_outward();
    progress.set(0.85, "Penghalusan Taubin");
    mesh.taubin_smooth(0.5, -0.53, cfg.smooth_iterations);
    if progress.is_cancelled() {
        return Err(VolError::Cancelled);
    }
    progress.set(0.9, "Decimate");
    mesh.decimate(cfg.decimate_error_factor * grid.s_xy);
    mesh.recompute_normals();
    progress.set(1.0, "Mesh selesai");
    Ok(mesh)
}

/// Rekonstruksi satu label lengkap: masker voxel dan mesh.
pub fn reconstruct_label(
    plan: &GridPlan,
    inputs: &[SliceInput<'_>],
    label: u8,
    cfg: &ReconConfig,
    progress: &dyn Progress,
) -> Result<(VoxelMask, LabelMesh), VolError> {
    let sub1 = h3d_core::SubProgress::new(progress, 0.0, 0.3);
    let stack = build_label_stack(plan, inputs, label, cfg, &sub1)?;
    let mask = stack.voxel_mask();
    let sub2 = h3d_core::SubProgress::new(progress, 0.3, 1.0);
    let mesh = extract_mesh(&stack, LabelId(label), cfg, &sub2)?;
    Ok((mask, mesh))
}

/// SDF analitik bola untuk uji dan bench: pusat dan jari-jari dalam µm.
pub struct SphereSdf {
    /// Kisi.
    pub grid: StackGrid,
    /// Pusat, µm.
    pub center_um: [f64; 3],
    /// Jari-jari, µm.
    pub radius_um: f64,
}

impl SdfProvider for SphereSdf {
    fn grid(&self) -> &StackGrid {
        &self.grid
    }
    fn fill_layer(&self, k: usize, out: &mut [f32]) {
        let nx = self.grid.size[0] as usize;
        let z = self.grid.z_um.get(k).copied().unwrap_or(0.0);
        for (i, o) in out.iter_mut().enumerate() {
            let w = self.grid.index_to_um((i % nx) as u32, (i / nx) as u32);
            let d = ((w[0] - self.center_um[0]).powi(2)
                + (w[1] - self.center_um[1]).powi(2)
                + (z - self.center_um[2]).powi(2))
            .sqrt()
                - self.radius_um;
            *o = d as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use h3d_core::NoProgress;

    fn sphere_grid(n: u32, s: f64) -> StackGrid {
        StackGrid {
            origin_um: [0.0, 0.0],
            s_xy: s,
            size: [n, n],
            z_um: (0..n).map(|k| (k as f64 + 0.5) * s).collect(),
        }
    }

    #[test]
    fn sphere_sdf_mesh_closed_positive_volume() {
        let n = 40;
        let s = 10.0;
        let grid = sphere_grid(n, s);
        let c = n as f64 * s / 2.0;
        let r = 120.0;
        let sdf = SphereSdf {
            grid,
            center_um: [c, c, c],
            radius_um: r,
        };
        let cfg = ReconConfig {
            smooth_iterations: 0,
            decimate_error_factor: 0.0,
            ..Default::default()
        };
        let mesh = extract_mesh(&sdf, LabelId(1), &cfg, &NoProgress).unwrap();
        assert!(mesh.is_closed(), "mesh tidak tertutup");
        assert_eq!(mesh.connected_components(), 1);
        assert_eq!(mesh.euler_characteristic(), 2);
        let v = mesh.signed_volume_um3();
        let truth = 4.0 / 3.0 * std::f64::consts::PI * r * r * r;
        assert!(v > 0.0);
        assert!(((v - truth) / truth).abs() < 0.03, "{v} vs {truth}");
    }

    #[test]
    fn multi_chunk_extraction_is_seamless() {
        // 70 voxel per sisi → 2 chunk per sumbu.
        let n = 70;
        let s = 10.0;
        let grid = sphere_grid(n, s);
        let c = n as f64 * s / 2.0;
        let sdf = SphereSdf {
            grid,
            center_um: [c, c, c],
            radius_um: 300.0,
        };
        let cfg = ReconConfig {
            smooth_iterations: 2,
            decimate_error_factor: 0.5,
            ..Default::default()
        };
        let mesh = extract_mesh(&sdf, LabelId(1), &cfg, &NoProgress).unwrap();
        assert!(mesh.is_closed(), "mesh lintas chunk tidak tertutup");
        assert_eq!(mesh.connected_components(), 1);
        assert_eq!(mesh.euler_characteristic(), 2);
        let truth = 4.0 / 3.0 * std::f64::consts::PI * 300f64.powi(3);
        let v = mesh.signed_volume_um3();
        assert!(((v - truth) / truth).abs() < 0.03, "{v} vs {truth}");
    }

    #[test]
    fn z_layers_insert_intermediates_and_caps() {
        let mut w = Vec::new();
        let z = plan_z_layers(&[0.0, 100.0, 200.0], 50.0, 500.0, 32.0, &mut w);
        assert_eq!(z.first(), Some(&-50.0));
        assert_eq!(z.last(), Some(&250.0));
        assert!(w.is_empty());
        // 100/32 → 4 interval → 3 lapis sisipan per celah.
        // Tutup: 50/32 → 2 langkah per ujung.
        assert_eq!(z.len(), 2 * 2 + 3 + 2 * 3);
        let z2 = plan_z_layers(&[0.0, 1000.0], 50.0, 500.0, 32.0, &mut w);
        assert_eq!(w.len(), 1);
        assert_eq!(
            z2,
            vec![-50.0, -25.0, 0.0, 25.0, 50.0, 950.0, 975.0, 1000.0, 1025.0, 1050.0]
        );
    }
}
