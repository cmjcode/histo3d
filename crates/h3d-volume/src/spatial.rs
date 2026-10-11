//! Analisis spasial 3D (histo3d.md 8.12.6) atas masker voxel kisi tumpukan
//! — bukan mesh (aturan keras 19). Semua jarak dilaporkan bersama
//! [`Uncertainty`] (aturan keras 18); tidak ada teks rekomendasi tindakan.
//!
//! Determinisme: pemindaian dalam urutan `(z, y, x)` = indeks linear
//! menaik; minimum/maksimum diperbarui hanya bila **lebih baik secara
//! ketat**, sehingga seri jatuh ke indeks terkecil. `edt3_argmin` dari
//! `h3d-imgproc` memakai aturan yang sama.

use std::collections::VecDeque;

use h3d_core::{
    AnatDir, AttrDef, LabelDef, LabelId, LabelKind, Progress, SignedAxis, SpatialConfig,
    SpecimenOrientation, VesselKind,
};
use h3d_imgproc::edt::{edt3_anisotropic, edt3_argmin, NO_INDEX};
use serde::{Deserialize, Serialize};

use crate::error::VolError;
use crate::grid::StackGrid;
use crate::measure::Warning;
use crate::mesh::LabelMesh;
use crate::recon::VoxelMask;

/// Nilai jarak/ukuran 3D beserta ketidakpastiannya (8.12.6).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Uncertainty {
    /// Nilai, µm.
    pub value_um: f64,
    /// `± (½ ukuran voxel pada sumbu dominan + TRE_est)`, µm.
    pub plus_minus_um: f64,
    /// Benar bila sumbu dominan adalah Z ("resolusi Z = jarak irisan").
    pub z_limited: bool,
}

/// Ketidakpastian jarak: setengah voxel pada sumbu dominan ditambah TRE.
pub fn distance_uncertainty(
    value_um: f64,
    s_xy: f64,
    s_z: f64,
    dominant_axis: SignedAxis,
    tre_est: f64,
) -> Uncertainty {
    let z = matches!(dominant_axis, SignedAxis::PosZ | SignedAxis::NegZ);
    let half = if z { s_z } else { s_xy } / 2.0;
    Uncertainty {
        value_um,
        plus_minus_um: half + tre_est.max(0.0),
        z_limited: z,
    }
}

#[inline]
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Komponen terhubung 3D (26- atau 6-konektivitas) dengan BFS urutan pindai
/// tetap; label 1..=n menurut kemunculan pertama, 0 = latar.
fn label_components_3d(mask: &[u8], grid: &StackGrid, conn26: bool) -> (Vec<u32>, usize) {
    let [nx, ny, nz] = grid.dims();
    let mut labels = vec![0u32; mask.len()];
    let mut n = 0usize;
    let mut queue: VecDeque<usize> = VecDeque::new();
    for start in 0..mask.len() {
        if mask[start] == 0 || labels[start] != 0 {
            continue;
        }
        n += 1;
        labels[start] = n as u32;
        queue.push_back(start);
        while let Some(i) = queue.pop_front() {
            let [x, y, z] = grid.delinear(i);
            for dz in -1i64..=1 {
                for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        if dx == 0 && dy == 0 && dz == 0 {
                            continue;
                        }
                        if !conn26 && (dx.abs() + dy.abs() + dz.abs()) != 1 {
                            continue;
                        }
                        let (xx, yy, zz) = (x as i64 + dx, y as i64 + dy, z as i64 + dz);
                        if xx < 0
                            || yy < 0
                            || zz < 0
                            || xx >= nx as i64
                            || yy >= ny as i64
                            || zz >= nz as i64
                        {
                            continue;
                        }
                        let j = grid.linear(xx as usize, yy as usize, zz as usize);
                        if mask[j] != 0 && labels[j] == 0 {
                            labels[j] = n as u32;
                            queue.push_back(j);
                        }
                    }
                }
            }
        }
    }
    (labels, n)
}

/// Resampel masker ke jarak lapis seragam (`s_z` median) dengan tetangga
/// terdekat (seri → lapis berindeks lebih kecil). Masker yang sudah seragam
/// dikembalikan apa adanya (klon).
pub fn resample_uniform_z(mask: &VoxelMask) -> VoxelMask {
    let g = &mask.grid;
    if g.nz() < 2 || g.is_uniform_z(1e-6) {
        return mask.clone();
    }
    let dz = g.median_dz();
    let z0 = g.z_um[0];
    let z1 = g.z_um[g.nz() - 1];
    let n = ((z1 - z0) / dz).round().max(0.0) as usize + 1;
    let new_z: Vec<f64> = (0..n).map(|k| z0 + k as f64 * dz).collect();
    let grid = StackGrid {
        origin_um: g.origin_um,
        s_xy: g.s_xy,
        size: g.size,
        z_um: new_z.clone(),
    };
    let layer = g.layer_len();
    let mut out = VoxelMask::empty(grid);
    for (k, &z) in new_z.iter().enumerate() {
        // Lapis asal terdekat.
        let mut best = 0usize;
        let mut bd = f64::INFINITY;
        for (j, &zj) in g.z_um.iter().enumerate() {
            let d = (zj - z).abs();
            if d < bd {
                bd = d;
                best = j;
            }
        }
        out.data[k * layer..(k + 1) * layer]
            .copy_from_slice(&mask.data[best * layer..(best + 1) * layer]);
        out.annotated_layer[k] = mask.annotated_layer.get(best).copied().flatten();
    }
    out
}

/// Volume atribut pada kisi (`NaN` = NODATA), indeks `x + nx·(y + ny·z)`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttrVolume {
    /// Kisi.
    pub grid: StackGrid,
    /// Nilai per voxel.
    pub data: Vec<f32>,
}

impl AttrVolume {
    /// Resampel ke jarak lapis seragam (tetangga terdekat), seperti
    /// [`resample_uniform_z`].
    pub fn resample_uniform_z(&self) -> AttrVolume {
        let g = &self.grid;
        if g.nz() < 2 || g.is_uniform_z(1e-6) {
            return self.clone();
        }
        let dz = g.median_dz();
        let z0 = g.z_um[0];
        let z1 = g.z_um[g.nz() - 1];
        let n = ((z1 - z0) / dz).round().max(0.0) as usize + 1;
        let new_z: Vec<f64> = (0..n).map(|k| z0 + k as f64 * dz).collect();
        let layer = g.layer_len();
        let mut data = vec![f32::NAN; layer * n];
        for (k, &z) in new_z.iter().enumerate() {
            let mut best = 0usize;
            let mut bd = f64::INFINITY;
            for (j, &zj) in g.z_um.iter().enumerate() {
                let d = (zj - z).abs();
                if d < bd {
                    bd = d;
                    best = j;
                }
            }
            data[k * layer..(k + 1) * layer]
                .copy_from_slice(&self.data[best * layer..(best + 1) * layer]);
        }
        AttrVolume {
            grid: StackGrid {
                origin_um: g.origin_um,
                s_xy: g.s_xy,
                size: g.size,
                z_um: new_z,
            },
            data,
        }
    }
}

fn check_same_grid(a: &VoxelMask, b: &VoxelMask) -> Result<(), VolError> {
    if a.grid.size != b.grid.size || a.grid.nz() != b.grid.nz() || a.data.len() != b.data.len() {
        return Err(VolError::InconsistentSizes(
            "masker voxel tidak pada kisi yang sama".into(),
        ));
    }
    Ok(())
}

fn spacing(grid: &StackGrid) -> [f64; 3] {
    [grid.s_xy, grid.s_xy, grid.median_dz()]
}

/// Jarak antar **permukaan** dua himpunan voxel dari jarak antar **pusat**
/// voxel terdekat: dikurangi satu ukuran voxel pada sumbu dominan (setengah
/// voxel di tiap sisi), dijepit ke 0. Tanpa koreksi ini dua voxel yang
/// bersentuhan berjarak `s_xy`, bukan 0, sehingga status `Involved`
/// (`d_min ≤ ½ s_xy`) tidak pernah tercapai dan jarak bias `+1` voxel.
fn surface_gap(center_dist_um: f64, axis: SignedAxis, sp: [f64; 3]) -> f64 {
    let vox = if matches!(axis, SignedAxis::PosZ | SignedAxis::NegZ) {
        sp[2]
    } else {
        sp[0]
    };
    (center_dist_um - vox).max(0.0)
}

// ---------------------------------------------------------------------------
// Margin
// ---------------------------------------------------------------------------

/// Status margin.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MarginStatus {
    /// `d_min ≤ ½ s_xy`.
    Involved,
    /// `d_min < margin_close_um`.
    Close,
    /// Lainnya.
    Clear,
}

/// Hasil analisis margin.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MarginResult {
    /// Jarak minimum tumor → margin.
    pub d_min: Uncertainty,
    /// Sumbu dominan arah tumor → margin terdekat.
    pub direction: SignedAxis,
    /// Nama anatomis arah itu bila orientasi diketahui.
    pub direction_name: Option<AnatDir>,
    /// Status.
    pub status: MarginStatus,
    /// Pusat voxel tumor terdekat, µm.
    pub nearest_tumor_um: [f64; 3],
    /// Pusat voxel margin terdekat, µm.
    pub nearest_margin_um: [f64; 3],
    /// Jarak minimum per arah, urutan [`SignedAxis::ALL`].
    pub per_direction: [Option<Uncertainty>; 6],
    /// Medan `d_M` (µm) atas kisi untuk proyeksi ke mesh: jarak pusat voxel
    /// ke pusat voxel margin terdekat (tanpa koreksi setengah voxel).
    #[serde(skip)]
    pub vertex_field: Vec<f32>,
}

/// Analisis margin: `d_M = edt3(Margin)`, `d_min` atas voxel tumor, arah
/// dari argmin, status, per arah. `None` bila tumor atau margin kosong.
/// `d_min` dan nilai per arah adalah jarak antar permukaan voxel (lihat
/// `surface_gap`).
pub fn margin(
    tumor: &VoxelMask,
    margin_mask: &VoxelMask,
    orientation: Option<&SpecimenOrientation>,
    cfg: &SpatialConfig,
    tre_est: f64,
) -> Result<Option<MarginResult>, VolError> {
    check_same_grid(tumor, margin_mask)?;
    let grid = &tumor.grid;
    if tumor.count() == 0 || margin_mask.count() == 0 {
        return Ok(None);
    }
    let sp = spacing(grid);
    let (d2, arg) = edt3_argmin(&margin_mask.data, grid.dims(), sp)?;
    let field: Vec<f32> = d2.iter().map(|v| v.sqrt()).collect();
    let mut best_i = usize::MAX;
    let mut best = f64::INFINITY;
    let mut per_dir: [Option<(f64, usize)>; 6] = [None; 6];
    for (i, &m) in tumor.data.iter().enumerate() {
        if m == 0 || arg[i] == NO_INDEX {
            continue;
        }
        let [x, y, z] = grid.delinear(i);
        let [mx, my, mz] = grid.delinear(arg[i] as usize);
        let v = sub3(grid.center_um(mx, my, mz), grid.center_um(x, y, z));
        let axis = SignedAxis::dominant(v);
        let d = surface_gap(field[i] as f64, axis, sp);
        let bin = SignedAxis::ALL.iter().position(|a| *a == axis).unwrap_or(0);
        match per_dir[bin] {
            Some((pd, _)) if pd <= d => {}
            _ => per_dir[bin] = Some((d, i)),
        }
        if d < best {
            best = d;
            best_i = i;
        }
    }
    if best_i == usize::MAX {
        return Ok(None);
    }
    let [x, y, z] = grid.delinear(best_i);
    let [mx, my, mz] = grid.delinear(arg[best_i] as usize);
    let pt = grid.center_um(x, y, z);
    let pm = grid.center_um(mx, my, mz);
    let direction = SignedAxis::dominant(sub3(pm, pt));
    let d_min = distance_uncertainty(best, sp[0], sp[2], direction, tre_est);
    let status = if best <= 0.5 * grid.s_xy {
        MarginStatus::Involved
    } else if best < cfg.margin_close_um {
        MarginStatus::Close
    } else {
        MarginStatus::Clear
    };
    let mut per_direction: [Option<Uncertainty>; 6] = [None; 6];
    for (k, axis) in SignedAxis::ALL.iter().enumerate() {
        if let Some((d, _)) = per_dir[k] {
            per_direction[k] = Some(distance_uncertainty(d, sp[0], sp[2], *axis, tre_est));
        }
    }
    Ok(Some(MarginResult {
        d_min,
        direction,
        direction_name: orientation.and_then(|o| o.name_of(direction)),
        status,
        nearest_tumor_um: pt,
        nearest_margin_um: pm,
        per_direction,
        vertex_field: field,
    }))
}

// ---------------------------------------------------------------------------
// Kontak / invasi
// ---------------------------------------------------------------------------

/// Tambalan kontak.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Patch {
    /// Centroid (berbobot luas), µm.
    pub centroid_um: [f64; 3],
    /// Luas kontak, mm².
    pub area_mm2: f64,
    /// Indeks lapis yang terlibat (terurut).
    pub layers: Vec<usize>,
}

/// Hasil kontak tumor–struktur.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContactResult {
    /// Ada kontak?
    pub any: bool,
    /// Luas kontak terkoreksi normal setempat, mm² (lihat [`contact`]).
    pub area_mm2: f64,
    /// Luas kontak mentah: jumlah luas muka voxel, mm² (bias tangga ≤ 27 %).
    pub area_voxel_faces_mm2: f64,
    /// Tambalan kontak.
    pub patches: Vec<Patch>,
}

/// Sigma pemulusan indikator struktur untuk normal kontak, dalam satuan
/// `s_xy` (isotropik secara fisik; pada sumbu z dikonversi ke lapis).
pub const CONTACT_NORMAL_SIGMA_VOXELS: f64 = 1.5;

/// Pemulusan Gauss 3D separabel atas buffer `f32` berindeks
/// `x + nx·(y + ny·z)`; `sigma` per sumbu dalam voxel (≤ 0 = sumbu dilewati),
/// tepi direplikasi.
fn gaussian_smooth_3d(data: &mut [f32], dims: [usize; 3], sigma: [f64; 3]) {
    let [nx, ny, nz] = dims;
    let strides = [1usize, nx, nx * ny];
    for axis in 0..3 {
        let sg = sigma[axis];
        let n = dims[axis];
        if sg.is_nan() || sg <= 0.0 || n < 2 {
            continue;
        }
        let r = (3.0 * sg).ceil() as i64;
        let kernel: Vec<f64> = (-r..=r)
            .map(|i| (-(i * i) as f64 / (2.0 * sg * sg)).exp())
            .collect();
        let ksum: f64 = kernel.iter().sum();
        let stride = strides[axis];
        // Garis sepanjang `axis`: titik awal = semua indeks dengan koordinat
        // sumbu itu nol.
        let (na, nb) = match axis {
            0 => (ny, nz),
            1 => (nx, nz),
            _ => (nx, ny),
        };
        let mut line = vec![0f64; n];
        for b in 0..nb {
            for a in 0..na {
                let start = match axis {
                    0 => nx * (a + ny * b),
                    1 => a + nx * ny * b,
                    _ => a + nx * b,
                };
                for (i, l) in line.iter_mut().enumerate() {
                    *l = f64::from(data[start + i * stride]);
                }
                for i in 0..n {
                    let mut acc = 0.0;
                    for (j, w) in kernel.iter().enumerate() {
                        let q = (i as i64 + j as i64 - r).clamp(0, n as i64 - 1) as usize;
                        acc += w * line[q];
                    }
                    data[start + i * stride] = (acc / ksum) as f32;
                }
            }
        }
    }
}

/// Kontak via pasangan voxel bertetangga muka (6-konektivitas), satu tumor
/// dan satu struktur. Luas muka `s_xy × s_xy` (muka ±Z) atau `s_xy × s_z`
/// setempat (muka ±X/±Y). Luas dilaporkan dua kali: mentah (jumlah muka) dan
/// terkoreksi, yakni tiap muka dibobot `|n_sumbu|` dengan `n` = normal
/// permukaan struktur **di muka itu** — penaksir luas permukaan yang tak bias
/// untuk permukaan halus (Σ muka × |n·e| = ∫ (nx²+ny²+nz²) dA = ∫ dA).
///
/// Normal = gradien indikator struktur yang dipulus Gauss
/// (σ = [`CONTACT_NORMAL_SIGMA_VOXELS`] × `s_xy`, isotropik secara fisik),
/// rata-rata gradien di voxel tumor dan voxel struktur yang mengapit muka.
/// Gradien `edt3(struktur)` tidak dipakai: arahnya terkuantisasi (menunjuk
/// pusat voxel terdekat) dan derau arah itu selalu menaikkan `Σ|n_k|`,
/// sehingga keliling cakram digital terukur +6…+10 % (jari-jari 3–16 voxel);
/// dengan indikator terpulus biasnya ≤ 0,5 % ± 2 %.
/// Tambalan = komponen 26-terhubung voxel tumor yang berkontak.
pub fn contact(tumor: &VoxelMask, structure: &VoxelMask) -> Result<ContactResult, VolError> {
    check_same_grid(tumor, structure)?;
    let grid = &tumor.grid;
    let [nx, ny, nz] = grid.dims();
    let sp = spacing(grid);
    let mut smooth: Vec<f32> = structure.data.iter().map(|&v| f32::from(v != 0)).collect();
    let sg_xy = CONTACT_NORMAL_SIGMA_VOXELS;
    let sg_z = if sp[2] > 0.0 {
        CONTACT_NORMAL_SIGMA_VOXELS * sp[0] / sp[2]
    } else {
        0.0
    };
    gaussian_smooth_3d(&mut smooth, grid.dims(), [sg_xy, sg_xy, sg_z]);
    let at = |x: i64, y: i64, z: i64| -> f64 {
        let xx = x.clamp(0, nx as i64 - 1) as usize;
        let yy = y.clamp(0, ny as i64 - 1) as usize;
        let zz = z.clamp(0, nz as i64 - 1) as usize;
        f64::from(smooth[grid.linear(xx, yy, zz)])
    };
    // Gradien fisik (per µm) indikator terpulus di voxel `(x, y, z)`.
    let gradient = |x: i64, y: i64, z: i64| -> [f64; 3] {
        let gx = (at(x + 1, y, z) - at(x - 1, y, z)) / (2.0 * grid.s_xy);
        let gy = (at(x, y + 1, z) - at(x, y - 1, z)) / (2.0 * grid.s_xy);
        let zu = (z + 1).min(nz as i64 - 1);
        let zl = (z - 1).max(0);
        let dzz = grid.z_um[zu as usize] - grid.z_um[zl as usize];
        let gz = if dzz > 0.0 {
            (at(x, y, zu) - at(x, y, zl)) / dzz
        } else {
            0.0
        };
        [gx, gy, gz]
    };
    let mut contact_mask = vec![0u8; tumor.data.len()];
    let mut raw_area = vec![0f64; tumor.data.len()];
    let mut corr_area = vec![0f64; tumor.data.len()];
    const DIRS: [(i64, i64, i64); 6] = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ];
    for (i, &t) in tumor.data.iter().enumerate() {
        if t == 0 {
            continue;
        }
        let [x, y, z] = grid.delinear(i);
        let (xi, yi, zi) = (x as i64, y as i64, z as i64);
        let mut g_here: Option<[f64; 3]> = None;
        for (k, &(dx, dy, dz)) in DIRS.iter().enumerate() {
            let (xx, yy, zz) = (xi + dx, yi + dy, zi + dz);
            if xx < 0 || yy < 0 || zz < 0 || xx >= nx as i64 || yy >= ny as i64 || zz >= nz as i64 {
                continue;
            }
            let j = grid.linear(xx as usize, yy as usize, zz as usize);
            if structure.data[j] == 0 {
                continue;
            }
            let dz_loc = grid.local_dz(z);
            let face = if k < 4 {
                grid.s_xy * dz_loc
            } else {
                grid.s_xy * grid.s_xy
            };
            // Normal di muka = rata-rata gradien kedua voxel pengapitnya.
            let ga = *g_here.get_or_insert_with(|| gradient(xi, yi, zi));
            let gb = gradient(xx, yy, zz);
            let g = [ga[0] + gb[0], ga[1] + gb[1], ga[2] + gb[2]];
            let l = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
            let w = if l > 1e-12 { g[k / 2].abs() / l } else { 1.0 };
            contact_mask[i] = 1;
            raw_area[i] += face;
            corr_area[i] += face * w;
        }
    }
    let (labels, n) = label_components_3d(&contact_mask, grid, true);
    let mut patches: Vec<Patch> = Vec::with_capacity(n);
    let mut acc: Vec<([f64; 3], f64, f64, Vec<usize>)> = vec![([0.0; 3], 0.0, 0.0, Vec::new()); n];
    for (i, &l) in labels.iter().enumerate() {
        if l == 0 {
            continue;
        }
        let e = &mut acc[l as usize - 1];
        let [x, y, z] = grid.delinear(i);
        let c = grid.center_um(x, y, z);
        let a = corr_area[i];
        e.0[0] += c[0] * a;
        e.0[1] += c[1] * a;
        e.0[2] += c[2] * a;
        e.1 += a;
        e.2 += raw_area[i];
        if e.3.last() != Some(&z) {
            e.3.push(z);
        }
    }
    let mut total_corr = 0.0;
    let mut total_raw = 0.0;
    for (c, a, raw, mut layers) in acc {
        layers.sort_unstable();
        layers.dedup();
        total_corr += a;
        total_raw += raw;
        let centroid = if a > 0.0 {
            [c[0] / a, c[1] / a, c[2] / a]
        } else {
            [0.0; 3]
        };
        patches.push(Patch {
            centroid_um: centroid,
            area_mm2: a / 1e6,
            layers,
        });
    }
    patches.sort_by(|p, q| q.area_mm2.total_cmp(&p.area_mm2));
    Ok(ContactResult {
        any: total_raw > 0.0,
        area_mm2: total_corr / 1e6,
        area_voxel_faces_mm2: total_raw / 1e6,
        patches,
    })
}

// ---------------------------------------------------------------------------
// Penetrasi kapsul
// ---------------------------------------------------------------------------

/// Hasil penetrasi kapsul.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CapsuleResult {
    /// Ada komponen tumor yang menembus?
    pub breached: bool,
    /// Per komponen tumor (26-konektivitas, urutan kemunculan): menembus?
    pub per_component: Vec<bool>,
    /// Volume tumor di luar kapsul, µm³.
    pub volume_outside_um3: f64,
    /// Volume tumor di dalam kapsul, µm³.
    pub volume_inside_um3: f64,
}

const FACE_NEIGHBOURS: [(i64, i64, i64); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

/// Penetrasi kapsul. `outside` = flood-fill 6-konektivitas dari batas kisi
/// (urutan pindai z, y, x) atas voxel yang **bukan kapsul dan bukan tumor**;
/// latar yang tidak terjangkau = `inside` (terkurung). Tumor ikut menjadi
/// penghalang karena tumor yang menembus kapsul menyumbat lubangnya sendiri:
/// bila flood boleh melewati tumor, seluruh isi kapsul ikut menjadi `outside`
/// dan penetrasi justru tak terdeteksi.
///
/// Komponen tumor (26-konektivitas) yang bertetangga muka dengan latar
/// `inside` **dan** latar `outside` = menembus. Volume luar/dalam: komponen
/// yang hanya menyentuh `outside` dihitung luar seluruhnya, yang tidak
/// menyentuh `outside` dalam seluruhnya; pada komponen yang menembus tiap
/// voxel dihitung luar bila lebih dekat ke latar `outside` daripada ke
/// latar `inside` (seri → dalam).
pub fn capsule_penetration(
    tumor: &VoxelMask,
    capsule: &VoxelMask,
) -> Result<CapsuleResult, VolError> {
    check_same_grid(tumor, capsule)?;
    let grid = &tumor.grid;
    let [nx, ny, nz] = grid.dims();
    let n_vox = capsule.data.len();
    let free = |i: usize| capsule.data[i] == 0 && tumor.data[i] == 0;
    let in_grid = |x: i64, y: i64, z: i64| {
        x >= 0 && y >= 0 && z >= 0 && x < nx as i64 && y < ny as i64 && z < nz as i64
    };
    let mut outside = vec![0u8; n_vox];
    let mut queue: VecDeque<usize> = VecDeque::new();
    for i in 0..n_vox {
        let [x, y, z] = grid.delinear(i);
        let border = x == 0 || y == 0 || z == 0 || x + 1 == nx || y + 1 == ny || z + 1 == nz;
        if !border || !free(i) || outside[i] != 0 {
            continue;
        }
        outside[i] = 1;
        queue.push_back(i);
        while let Some(j) = queue.pop_front() {
            let [jx, jy, jz] = grid.delinear(j);
            for (dx, dy, dz) in FACE_NEIGHBOURS {
                let (xx, yy, zz) = (jx as i64 + dx, jy as i64 + dy, jz as i64 + dz);
                if !in_grid(xx, yy, zz) {
                    continue;
                }
                let k = grid.linear(xx as usize, yy as usize, zz as usize);
                if free(k) && outside[k] == 0 {
                    outside[k] = 1;
                    queue.push_back(k);
                }
            }
        }
    }
    // Latar terkurung.
    let inside: Vec<u8> = (0..n_vox)
        .map(|i| u8::from(free(i) && outside[i] == 0))
        .collect();

    let (labels, n) = label_components_3d(&tumor.data, grid, true);
    let mut touch_in = vec![false; n];
    let mut touch_out = vec![false; n];
    for (i, &l) in labels.iter().enumerate() {
        if l == 0 {
            continue;
        }
        let c = l as usize - 1;
        let [x, y, z] = grid.delinear(i);
        for (dx, dy, dz) in FACE_NEIGHBOURS {
            let (xx, yy, zz) = (x as i64 + dx, y as i64 + dy, z as i64 + dz);
            if !in_grid(xx, yy, zz) {
                // Tumor di tepi kisi berbatasan dengan bagian luar.
                touch_out[c] = true;
                continue;
            }
            let k = grid.linear(xx as usize, yy as usize, zz as usize);
            if outside[k] != 0 {
                touch_out[c] = true;
            } else if inside[k] != 0 {
                touch_in[c] = true;
            }
        }
    }
    let per_component: Vec<bool> = (0..n).map(|k| touch_in[k] && touch_out[k]).collect();
    let breached = per_component.iter().any(|&b| b);
    // Medan jarak hanya diperlukan bila ada komponen yang menembus.
    let dists = if breached {
        let sp = spacing(grid);
        Some((
            edt3_anisotropic(&outside, grid.dims(), sp)?,
            edt3_anisotropic(&inside, grid.dims(), sp)?,
        ))
    } else {
        None
    };
    let mut v_out = 0.0;
    let mut v_in = 0.0;
    for (i, &l) in labels.iter().enumerate() {
        if l == 0 {
            continue;
        }
        let c = l as usize - 1;
        let vv = grid.voxel_volume_um3(grid.delinear(i)[2]);
        let is_out = match (&dists, per_component[c]) {
            (Some((d_out, d_in)), true) => d_out[i] < d_in[i],
            _ => touch_out[c],
        };
        if is_out {
            v_out += vv;
        } else {
            v_in += vv;
        }
    }
    Ok(CapsuleResult {
        breached,
        per_component,
        volume_outside_um3: v_out,
        volume_inside_um3: v_in,
    })
}

// ---------------------------------------------------------------------------
// Multifokalitas
// ---------------------------------------------------------------------------

/// Satu fokus tumor.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Focus {
    /// Volume, µm³ (jumlah voxel × volume voxel setempat).
    pub volume_um3: f64,
    /// Centroid pusat voxel, µm.
    pub centroid_um: [f64; 3],
    /// Kotak pembatas pusat voxel `(min, max)`, µm.
    pub bbox_um: ([f64; 3], [f64; 3]),
    /// Diameter maksimum (selubung cembung voxel permukaan), µm.
    pub max_diameter_um: f64,
    /// Jumlah voxel.
    pub n_voxels: usize,
    /// Rentang lapis `(pertama, terakhir)`.
    pub layers: (usize, usize),
}

/// Hasil multifokalitas.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FociResult {
    /// Jumlah fokus setelah penyaringan.
    pub n: usize,
    /// Fokus terurut volume menurun.
    pub foci: Vec<Focus>,
    /// Matriks jarak antar fokus (min `edt3(fokus A)` atas voxel fokus B);
    /// diagonal 0. Kosong bila fokus > [`MAX_FOCI_DISTANCE_MATRIX`].
    pub distances_um: Vec<Vec<Uncertainty>>,
}

/// Jumlah fokus maksimum yang matriks jaraknya dihitung.
pub const MAX_FOCI_DISTANCE_MATRIX: usize = 16;

/// Voxel permukaan: bernilai 1 dengan tetangga muka 0 atau di batas kisi.
fn surface_voxels(mask: &[u8], grid: &StackGrid) -> Vec<[f64; 3]> {
    let [nx, ny, nz] = grid.dims();
    let mut out = Vec::new();
    for (i, &m) in mask.iter().enumerate() {
        if m == 0 {
            continue;
        }
        let [x, y, z] = grid.delinear(i);
        let is_surface = x == 0
            || y == 0
            || z == 0
            || x + 1 == nx
            || y + 1 == ny
            || z + 1 == nz
            || mask[grid.linear(x - 1, y, z)] == 0
            || mask[grid.linear(x + 1, y, z)] == 0
            || mask[grid.linear(x, y - 1, z)] == 0
            || mask[grid.linear(x, y + 1, z)] == 0
            || mask[grid.linear(x, y, z - 1)] == 0
            || mask[grid.linear(x, y, z + 1)] == 0;
        if is_surface {
            out.push(grid.center_um(x, y, z));
        }
    }
    out
}

/// Komponen 26-terhubung gabungan grup `Tumor`; buang `< min_focus_um3`
/// (bawaan `(3 × s_xy)³`); urut volume menurun (seri → kemunculan pertama).
pub fn foci(tumor: &VoxelMask, cfg: &SpatialConfig, tre_est: f64) -> Result<FociResult, VolError> {
    let grid = &tumor.grid;
    let min_v = cfg
        .min_focus_um3
        .unwrap_or_else(|| (3.0 * grid.s_xy).powi(3));
    let (labels, n) = label_components_3d(&tumor.data, grid, true);
    let mut vol = vec![0f64; n];
    let mut cnt = vec![0usize; n];
    for (i, &l) in labels.iter().enumerate() {
        if l != 0 {
            let z = grid.delinear(i)[2];
            vol[l as usize - 1] += grid.voxel_volume_um3(z);
            cnt[l as usize - 1] += 1;
        }
    }
    let mut keep: Vec<usize> = (0..n).filter(|&k| vol[k] >= min_v).collect();
    keep.sort_by(|&a, &b| vol[b].total_cmp(&vol[a]).then(a.cmp(&b)));
    let mut foci_out = Vec::with_capacity(keep.len());
    let mut masks: Vec<Vec<u8>> = Vec::with_capacity(keep.len());
    for &k in &keep {
        let comp: Vec<u8> = labels
            .iter()
            .map(|&l| u8::from(l as usize == k + 1))
            .collect();
        let mut c = [0.0f64; 3];
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        let mut zlo = usize::MAX;
        let mut zhi = 0usize;
        for (i, &m) in comp.iter().enumerate() {
            if m == 0 {
                continue;
            }
            let [x, y, z] = grid.delinear(i);
            let p = grid.center_um(x, y, z);
            for a in 0..3 {
                c[a] += p[a];
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
            zlo = zlo.min(z);
            zhi = zhi.max(z);
        }
        let nvox = cnt[k] as f64;
        let surf = surface_voxels(&comp, grid);
        foci_out.push(Focus {
            volume_um3: vol[k],
            centroid_um: [c[0] / nvox, c[1] / nvox, c[2] / nvox],
            bbox_um: (lo, hi),
            max_diameter_um: crate::mesh::max_diameter_of_points(&surf),
            n_voxels: cnt[k],
            layers: (zlo, zhi),
        });
        masks.push(comp);
    }
    let m = foci_out.len();
    let sp = spacing(grid);
    let mut distances = Vec::new();
    if (2..=MAX_FOCI_DISTANCE_MATRIX).contains(&m) {
        distances =
            vec![vec![distance_uncertainty(0.0, sp[0], sp[2], SignedAxis::PosX, tre_est); m]; m];
        for a in 0..m {
            let (d2, arg) = edt3_argmin(&masks[a], grid.dims(), sp)?;
            for b in 0..m {
                if a == b {
                    continue;
                }
                let mut best = f64::INFINITY;
                let mut bi = usize::MAX;
                for (i, &mb) in masks[b].iter().enumerate() {
                    if mb != 0 && arg[i] != NO_INDEX {
                        let d = (d2[i] as f64).sqrt();
                        if d < best {
                            best = d;
                            bi = i;
                        }
                    }
                }
                if bi != usize::MAX {
                    let [x, y, z] = grid.delinear(bi);
                    let [ax, ay, az] = grid.delinear(arg[bi] as usize);
                    let v = sub3(grid.center_um(ax, ay, az), grid.center_um(x, y, z));
                    let axis = SignedAxis::dominant(v);
                    distances[b][a] = distance_uncertainty(
                        surface_gap(best, axis, sp),
                        sp[0],
                        sp[2],
                        axis,
                        tre_est,
                    );
                }
            }
        }
    }
    Ok(FociResult {
        n: m,
        foci: foci_out,
        distances_um: distances,
    })
}

// ---------------------------------------------------------------------------
// Kedalaman / Breslow
// ---------------------------------------------------------------------------

/// Hasil kedalaman invasi (atau ketebalan Breslow).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DepthResult {
    /// Kedalaman = max `edt3(acuan)` atas voxel tumor.
    pub depth: Uncertainty,
    /// Titik terdalam (pusat voxel), µm.
    pub deepest_um: [f64; 3],
    /// Indeks lapis titik terdalam.
    pub layer: usize,
}

/// Kedalaman: `d_S = edt3(acuan)`; maksimum atas voxel tumor (seri → indeks
/// terkecil). `None` bila tumor atau acuan kosong.
pub fn depth(
    tumor: &VoxelMask,
    reference: &VoxelMask,
    tre_est: f64,
) -> Result<Option<DepthResult>, VolError> {
    check_same_grid(tumor, reference)?;
    let grid = &tumor.grid;
    if tumor.count() == 0 || reference.count() == 0 {
        return Ok(None);
    }
    let sp = spacing(grid);
    let (d2, arg) = edt3_argmin(&reference.data, grid.dims(), sp)?;
    let mut best = f64::NEG_INFINITY;
    let mut bi = usize::MAX;
    for (i, &m) in tumor.data.iter().enumerate() {
        if m != 0 && arg[i] != NO_INDEX {
            let d = (d2[i] as f64).sqrt();
            if d > best {
                best = d;
                bi = i;
            }
        }
    }
    if bi == usize::MAX {
        return Ok(None);
    }
    let [x, y, z] = grid.delinear(bi);
    let [ax, ay, az] = grid.delinear(arg[bi] as usize);
    let p = grid.center_um(x, y, z);
    let v = sub3(p, grid.center_um(ax, ay, az));
    Ok(Some(DepthResult {
        depth: distance_uncertainty(best, sp[0], sp[2], SignedAxis::dominant(v), tre_est),
        deepest_um: p,
        layer: z,
    }))
}

// ---------------------------------------------------------------------------
// Ukuran staging
// ---------------------------------------------------------------------------

/// Hasil ukuran staging.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SizeResult {
    /// Diameter maksimum selubung cembung pusat voxel permukaan, µm.
    pub max_diameter_um: Uncertainty,
    /// Ukuran pada tiga sumbu utama PCA (menurun), µm.
    pub pca_axes_um: [Uncertainty; 3],
    /// Arah sumbu utama (vektor satuan), urutan sama.
    pub pca_directions: [[f64; 3]; 3],
    /// Centroid pusat voxel, µm.
    pub centroid_um: [f64; 3],
}

/// Dekomposisi eigen matriks simetris 3×3 (Jacobi); mengembalikan
/// `(nilai, vektor kolom)` terurut menurun.
#[allow(clippy::needless_range_loop)]
fn eigen_sym3(mut a: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..50 {
        let (mut p, mut q) = (0, 1);
        let mut m = a[0][1].abs();
        if a[0][2].abs() > m {
            m = a[0][2].abs();
            p = 0;
            q = 2;
        }
        if a[1][2].abs() > m {
            m = a[1][2].abs();
            p = 1;
            q = 2;
        }
        if m < 1e-12 {
            break;
        }
        let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
        let t = if theta == 0.0 { 1.0 } else { t };
        let c = 1.0 / (t * t + 1.0).sqrt();
        let s = t * c;
        for k in 0..3 {
            let akp = a[k][p];
            let akq = a[k][q];
            a[k][p] = c * akp - s * akq;
            a[k][q] = s * akp + c * akq;
        }
        for k in 0..3 {
            let apk = a[p][k];
            let aqk = a[q][k];
            a[p][k] = c * apk - s * aqk;
            a[q][k] = s * apk + c * aqk;
        }
        for k in 0..3 {
            let vkp = v[k][p];
            let vkq = v[k][q];
            v[k][p] = c * vkp - s * vkq;
            v[k][q] = s * vkp + c * vkq;
        }
    }
    let mut order = [0usize, 1, 2];
    order.sort_by(|&i, &j| a[j][j].total_cmp(&a[i][i]).then(i.cmp(&j)));
    let vals = [
        a[order[0]][order[0]],
        a[order[1]][order[1]],
        a[order[2]][order[2]],
    ];
    let mut vecs = [[0.0; 3]; 3];
    for (n, &o) in order.iter().enumerate() {
        let mut e = [v[0][o], v[1][o], v[2][o]];
        // Tanda tetap: komponen absolut terbesar positif.
        let mut big = 0;
        for k in 1..3 {
            if e[k].abs() > e[big].abs() {
                big = k;
            }
        }
        if e[big] < 0.0 {
            e = [-e[0], -e[1], -e[2]];
        }
        vecs[n] = e;
    }
    (vals, vecs)
}

/// Diameter maksimum (selubung cembung voxel permukaan) dan ukuran pada
/// tiga sumbu utama PCA pusat voxel tumor. `None` bila tumor kosong.
pub fn staging_size(tumor: &VoxelMask, tre_est: f64) -> Option<SizeResult> {
    let grid = &tumor.grid;
    let n = tumor.count();
    if n == 0 {
        return None;
    }
    let sp = spacing(grid);
    let mut c = [0.0f64; 3];
    for (i, &m) in tumor.data.iter().enumerate() {
        if m != 0 {
            let [x, y, z] = grid.delinear(i);
            let p = grid.center_um(x, y, z);
            c[0] += p[0];
            c[1] += p[1];
            c[2] += p[2];
        }
    }
    let nf = n as f64;
    c = [c[0] / nf, c[1] / nf, c[2] / nf];
    let mut cov = [[0.0f64; 3]; 3];
    for (i, &m) in tumor.data.iter().enumerate() {
        if m != 0 {
            let [x, y, z] = grid.delinear(i);
            let d = sub3(grid.center_um(x, y, z), c);
            for a in 0..3 {
                for b in 0..3 {
                    cov[a][b] += d[a] * d[b] / nf;
                }
            }
        }
    }
    let (_, dirs) = eigen_sym3(cov);
    let surf = surface_voxels(&tumor.data, grid);
    let mut ext = [Uncertainty {
        value_um: 0.0,
        plus_minus_um: 0.0,
        z_limited: false,
    }; 3];
    for (k, e) in dirs.iter().enumerate() {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for p in &surf {
            let d = sub3(*p, c);
            let t = d[0] * e[0] + d[1] * e[1] + d[2] * e[2];
            lo = lo.min(t);
            hi = hi.max(t);
        }
        let axis = SignedAxis::dominant(*e);
        // Ukuran = rentang proyeksi pusat voxel + satu voxel (dua setengah tepi).
        let vox = if matches!(axis, SignedAxis::PosZ | SignedAxis::NegZ) {
            sp[2]
        } else {
            sp[0]
        };
        ext[k] = distance_uncertainty((hi - lo) + vox, sp[0], sp[2], axis, tre_est);
    }
    let diam = crate::mesh::max_diameter_of_points(&surf);
    Some(SizeResult {
        max_diameter_um: distance_uncertainty(diam, sp[0], sp[2], SignedAxis::PosX, tre_est),
        pca_axes_um: ext,
        pca_directions: dirs,
        centroid_um: c,
    })
}

// ---------------------------------------------------------------------------
// Fraksi komponen
// ---------------------------------------------------------------------------

/// Satu baris tabel fraksi.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FractionRow {
    /// Nama grup.
    pub group: String,
    /// Label anggota.
    pub label: LabelId,
    /// Nama label.
    pub name: String,
    /// Jenis.
    pub kind: LabelKind,
    /// Volume anggota, µm³.
    pub volume_um3: f64,
    /// Volume anggota / volume grup (0 bila grup kosong).
    pub fraction_of_group: f64,
}

/// Tabel fraksi.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct FractionsResult {
    /// Baris per anggota, urutan masukan.
    pub rows: Vec<FractionRow>,
    /// Volume per grup, urutan kemunculan.
    pub group_volumes: Vec<(String, f64)>,
    /// Nekrosis % = `Necrosis ∩ pita tumor / (tumor + nekrosis dalam pita)`,
    /// pita = jarak ≤ `peri_band_um` dari tumor.
    pub necrosis_percent: Option<f64>,
}

/// Fraksi volume anggota terhadap grupnya (`LabelDef::group_name`), dan
/// persen nekrosis.
pub fn fractions(
    masks_by_label: &[(LabelDef, VoxelMask)],
    tumor_union: Option<&VoxelMask>,
    cfg: &SpatialConfig,
) -> Result<FractionsResult, VolError> {
    let mut groups: Vec<(String, f64)> = Vec::new();
    let vols: Vec<f64> = masks_by_label.iter().map(|(_, m)| m.volume_um3()).collect();
    for ((def, _), &v) in masks_by_label.iter().zip(&vols) {
        let g = def.group_name().to_owned();
        match groups.iter_mut().find(|(n, _)| *n == g) {
            Some(e) => e.1 += v,
            None => groups.push((g, v)),
        }
    }
    let rows = masks_by_label
        .iter()
        .zip(&vols)
        .map(|((def, _), &v)| {
            let g = def.group_name().to_owned();
            let gv = groups
                .iter()
                .find(|(n, _)| *n == g)
                .map(|e| e.1)
                .unwrap_or(0.0);
            FractionRow {
                group: g,
                label: def.id,
                name: def.name.clone(),
                kind: def.kind,
                volume_um3: v,
                fraction_of_group: if gv > 0.0 { v / gv } else { 0.0 },
            }
        })
        .collect();
    let mut necrosis_percent = None;
    if let Some(t) = tumor_union {
        let necro: Vec<&VoxelMask> = masks_by_label
            .iter()
            .filter(|(d, _)| d.kind == LabelKind::Necrosis)
            .map(|(_, m)| m)
            .collect();
        if !necro.is_empty() && t.count() > 0 {
            let grid = &t.grid;
            let sp = spacing(grid);
            let d2 = edt3_anisotropic(&t.data, grid.dims(), sp)?;
            let band2 = (cfg.peri_band_um * cfg.peri_band_um) as f32;
            let mut v_necro = 0.0;
            let mut seen = vec![0u8; t.data.len()];
            for m in &necro {
                check_same_grid(t, m)?;
                for (i, &n) in m.data.iter().enumerate() {
                    if n != 0 && seen[i] == 0 && d2[i] <= band2 {
                        seen[i] = 1;
                        v_necro += grid.voxel_volume_um3(grid.delinear(i)[2]);
                    }
                }
            }
            let v_t = t.volume_um3();
            if v_t + v_necro > 0.0 {
                necrosis_percent = Some(100.0 * v_necro / (v_t + v_necro));
            }
        }
    }
    Ok(FractionsResult {
        rows,
        group_volumes: groups,
        necrosis_percent,
    })
}

// ---------------------------------------------------------------------------
// Proksi hipoksia
// ---------------------------------------------------------------------------

/// Hasil proksi hipoksia.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HypoxiaResult {
    /// Fraksi voxel tumor dengan `d_V > hypoxia_um`.
    pub fraction_beyond: f64,
    /// Ambang yang dipakai, µm.
    pub threshold_um: f64,
    /// Histogram `(tepi bawah bin µm, jumlah voxel)`.
    pub histogram: Vec<(f64, u64)>,
    /// Medan `d_V` (µm) untuk proyeksi ke mesh.
    #[serde(skip)]
    pub field: Vec<f32>,
}

/// `d_V = edt3(Vessel)`; histogram atas voxel tumor (bin `vessel_hist_bin_um`)
/// dan fraksi `d_V > hypoxia_um`. `None` bila pembuluh atau tumor kosong.
pub fn hypoxia(
    tumor: &VoxelMask,
    vessel_blood: &VoxelMask,
    cfg: &SpatialConfig,
) -> Result<Option<HypoxiaResult>, VolError> {
    check_same_grid(tumor, vessel_blood)?;
    let grid = &tumor.grid;
    if tumor.count() == 0 || vessel_blood.count() == 0 {
        return Ok(None);
    }
    let d2 = edt3_anisotropic(&vessel_blood.data, grid.dims(), spacing(grid))?;
    let field: Vec<f32> = d2.iter().map(|v| v.sqrt()).collect();
    let bin = if cfg.vessel_hist_bin_um > 0.0 {
        cfg.vessel_hist_bin_um
    } else {
        25.0
    };
    let mut hist: Vec<u64> = Vec::new();
    let mut n = 0u64;
    let mut beyond = 0u64;
    for (i, &m) in tumor.data.iter().enumerate() {
        if m == 0 {
            continue;
        }
        let d = field[i] as f64;
        if !d.is_finite() {
            continue;
        }
        n += 1;
        if d > cfg.hypoxia_um {
            beyond += 1;
        }
        let b = (d / bin).floor() as usize;
        if b >= hist.len() {
            hist.resize(b + 1, 0);
        }
        hist[b] += 1;
    }
    Ok(Some(HypoxiaResult {
        fraction_beyond: if n > 0 { beyond as f64 / n as f64 } else { 0.0 },
        threshold_um: cfg.hypoxia_um,
        histogram: hist
            .iter()
            .enumerate()
            .map(|(k, &c)| (k as f64 * bin, c))
            .collect(),
        field,
    }))
}

// ---------------------------------------------------------------------------
// Statistik atribut
// ---------------------------------------------------------------------------

/// Statistik atribut per region.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttrStats {
    /// Jumlah voxel sah.
    pub n: usize,
    /// Rata-rata.
    pub mean: f64,
    /// Median.
    pub median: f64,
    /// Persentil 10.
    pub p10: f64,
    /// Persentil 90.
    pub p90: f64,
    /// Heterogenitas: koefisien variasi rata-rata kubus 1 mm³ (`None` bila
    /// kurang dari dua kubus).
    pub heterogeneity_cv: Option<f64>,
    /// `(z, rata-rata)` per lapis yang punya voxel sah.
    pub per_layer: Vec<(f64, f64)>,
}

fn percentile_sorted(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let r = p * (v.len() - 1) as f64;
    let lo = r.floor() as usize;
    let hi = r.ceil() as usize;
    v[lo] + (v[hi] - v[lo]) * (r - lo as f64)
}

/// Statistik `attr` atas voxel `mask` (voxel `NaN` diabaikan). `None` bila
/// tidak ada voxel sah.
pub fn attr_stats(attr: &AttrVolume, mask: &VoxelMask) -> Result<Option<AttrStats>, VolError> {
    if attr.data.len() != mask.data.len() || attr.grid.size != mask.grid.size {
        return Err(VolError::InconsistentSizes(
            "volume atribut dan masker berbeda kisi".into(),
        ));
    }
    let grid = &mask.grid;
    let mut vals: Vec<f64> = Vec::new();
    let cube_vox = (1000.0 / grid.s_xy).round().max(1.0) as usize;
    let z0 = grid.z_um.first().copied().unwrap_or(0.0);
    let mut cubes: std::collections::BTreeMap<(usize, usize, usize), (f64, usize)> =
        std::collections::BTreeMap::new();
    let mut layers: Vec<(f64, usize)> = vec![(0.0, 0); grid.nz()];
    for (i, &m) in mask.data.iter().enumerate() {
        if m == 0 {
            continue;
        }
        let v = attr.data[i] as f64;
        if !v.is_finite() {
            continue;
        }
        vals.push(v);
        let [x, y, z] = grid.delinear(i);
        let zc = ((grid.z_um[z] - z0) / 1000.0).floor().max(0.0) as usize;
        let e = cubes
            .entry((x / cube_vox, y / cube_vox, zc))
            .or_insert((0.0, 0));
        e.0 += v;
        e.1 += 1;
        layers[z].0 += v;
        layers[z].1 += 1;
    }
    if vals.is_empty() {
        return Ok(None);
    }
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    let mut sorted = vals.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let cube_means: Vec<f64> = cubes.values().map(|(s, n)| s / *n as f64).collect();
    let heterogeneity_cv = if cube_means.len() >= 2 {
        let cm = cube_means.iter().sum::<f64>() / cube_means.len() as f64;
        let var = cube_means.iter().map(|m| (m - cm).powi(2)).sum::<f64>()
            / (cube_means.len() - 1) as f64;
        if cm.abs() > 0.0 {
            Some(var.sqrt() / cm.abs())
        } else {
            None
        }
    } else {
        None
    };
    let per_layer = layers
        .iter()
        .enumerate()
        .filter(|(_, (_, n))| *n > 0)
        .map(|(k, (s, n))| (grid.z_um[k], s / *n as f64))
        .collect();
    Ok(Some(AttrStats {
        n: vals.len(),
        mean,
        median: percentile_sorted(&sorted, 0.5),
        p10: percentile_sorted(&sorted, 0.1),
        p90: percentile_sorted(&sorted, 0.9),
        heterogeneity_cv,
        per_layer,
    }))
}

// ---------------------------------------------------------------------------
// Agregat
// ---------------------------------------------------------------------------

/// Ringkasan kisi untuk laporan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GridSummary {
    /// `s_xy`, µm.
    pub s_xy_um: f64,
    /// `s_z` median, µm.
    pub s_z_um: f64,
    /// `[nx, ny]`.
    pub size: [u32; 2],
    /// Jumlah lapis.
    pub nz: usize,
    /// Kisi masukan sudah seragam di Z?
    pub uniform_z: bool,
    /// Kisi diresampel ke `s_z` median?
    pub resampled_z: bool,
}

/// Kontak per struktur.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContactEntry {
    /// Label struktur.
    pub label: LabelId,
    /// Nama label.
    pub name: String,
    /// Jenis.
    pub kind: LabelKind,
    /// Hasil.
    pub result: ContactResult,
}

/// Statistik satu atribut.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttrStatsEntry {
    /// Nama atribut.
    pub name: String,
    /// Satuan.
    pub unit: String,
    /// Hash parameter (dicetak di laporan, aturan 19).
    pub params_hash: Option<u64>,
    /// Statistik atas gabungan tumor.
    pub stats: AttrStats,
}

/// Hasil analisis spasial lengkap (`analysis.json`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpatialAnalysis {
    /// Parameter yang dipakai.
    pub params: SpatialConfig,
    /// Ringkasan kisi.
    pub grid: GridSummary,
    /// TRE yang dipakai untuk ketidakpastian, µm.
    pub tre_est_um: f64,
    /// Catatan.
    pub notes: Vec<String>,
    /// Peringatan (8.12.8).
    pub warnings: Vec<Warning>,
    /// Volume gabungan tumor, µm³ (voxel).
    pub tumor_volume_um3: f64,
    /// Margin.
    pub margin: Option<MarginResult>,
    /// Kontak per struktur.
    pub contacts: Vec<ContactEntry>,
    /// Penetrasi kapsul.
    pub capsule: Option<CapsuleResult>,
    /// Multifokalitas.
    pub foci: Option<FociResult>,
    /// Kedalaman invasi (acuan `Surface`).
    pub depth: Option<DepthResult>,
    /// Ketebalan Breslow (acuan `Reference`).
    pub breslow: Option<DepthResult>,
    /// Ukuran staging.
    pub size: Option<SizeResult>,
    /// Fraksi komponen.
    pub fractions: FractionsResult,
    /// Proksi hipoksia.
    pub hypoxia: Option<HypoxiaResult>,
    /// Statistik atribut.
    pub attr_stats: Vec<AttrStatsEntry>,
    /// Medan untuk proyeksi ke mesh: `(nama, satuan, nilai per voxel)`.
    #[serde(skip)]
    pub fields: Vec<(String, String, Vec<f32>)>,
}

fn union_of(masks: &[&VoxelMask], grid: &StackGrid) -> Result<Option<VoxelMask>, VolError> {
    if masks.is_empty() {
        return Ok(None);
    }
    let mut u = VoxelMask::empty(grid.clone());
    for m in masks {
        u.union_with(m)?;
    }
    Ok(Some(u))
}

/// Jalankan semua analisis yang prasyarat `kind`-nya terpenuhi (tabel 8.12.6).
/// `tre_est`: median TRE QC registrasi; `None` → `s_xy`.
pub fn analyze(
    masks: &[(LabelDef, VoxelMask)],
    attrs: &[(AttrDef, AttrVolume)],
    orientation: Option<&SpecimenOrientation>,
    cfg: &SpatialConfig,
    tre_est: Option<f64>,
    progress: &dyn Progress,
) -> Result<SpatialAnalysis, VolError> {
    let Some((_, first)) = masks.first() else {
        return Err(VolError::InvalidParameter("tidak ada masker voxel".into()));
    };
    for (_, m) in masks {
        check_same_grid(first, m)?;
        if m.grid.z_um != first.grid.z_um {
            return Err(VolError::InconsistentSizes("lapis z masker berbeda".into()));
        }
    }
    let mut notes = Vec::new();
    let mut warnings = Vec::new();
    let uniform = first.grid.is_uniform_z(1e-6);
    // Resampel ke s_z median bila perlu.
    let owned: Vec<(LabelDef, VoxelMask)>;
    let masks: &[(LabelDef, VoxelMask)] = if uniform {
        masks
    } else {
        owned = masks
            .iter()
            .map(|(d, m)| (d.clone(), resample_uniform_z(m)))
            .collect();
        let dz = owned[0].1.grid.median_dz();
        notes.push(format!(
            "Jarak irisan tidak seragam; kisi diresampel ke s_z median {dz:.1} µm untuk analisis spasial."
        ));
        warnings.push(Warning::NonUniformZResampled { dz });
        &owned
    };
    let owned_attrs: Vec<(AttrDef, AttrVolume)>;
    let attrs: &[(AttrDef, AttrVolume)] = if uniform {
        attrs
    } else {
        owned_attrs = attrs
            .iter()
            .map(|(d, a)| (d.clone(), a.resample_uniform_z()))
            .collect();
        &owned_attrs
    };
    let grid = masks[0].1.grid.clone();
    let s_z = grid.median_dz();
    let tre = tre_est
        .filter(|t| t.is_finite() && *t >= 0.0)
        .unwrap_or(grid.s_xy);
    let mut fields: Vec<(String, String, Vec<f32>)> = Vec::new();

    let by_kind = |pred: &dyn Fn(LabelKind) -> bool| -> Vec<&VoxelMask> {
        masks
            .iter()
            .filter(|(d, _)| pred(d.kind))
            .map(|(_, m)| m)
            .collect()
    };
    let cancelled = || progress.is_cancelled();

    progress.set(0.05, "Gabungan tumor");
    let tumor = union_of(&by_kind(&|k| k == LabelKind::Tumor), &grid)?;
    let tumor_volume = tumor.as_ref().map(|t| t.volume_um3()).unwrap_or(0.0);

    // Margin.
    let mut margin_res = None;
    if let (Some(t), Some(mm)) = (
        tumor.as_ref(),
        union_of(&by_kind(&|k| k == LabelKind::Margin), &grid)?,
    ) {
        progress.set(0.15, "Margin");
        if let Some(r) = margin(t, &mm, orientation, cfg, tre)? {
            if r.status != MarginStatus::Clear {
                let dir = r
                    .direction_name
                    .map(|d| d.as_str().to_owned())
                    .unwrap_or_else(|| r.direction.as_str().to_owned());
                warnings.push(Warning::MarginClose { dir });
            }
            // Margin harus ada di lapis berlabel tumor pertama dan terakhir.
            let layers: Vec<usize> = (0..grid.nz()).filter(|&k| t.layer_any(k)).collect();
            if let (Some(&f), Some(&l)) = (layers.first(), layers.last()) {
                if !mm.layer_any(f) || !mm.layer_any(l) {
                    warnings.push(Warning::MarginZUnassessable);
                }
            }
            fields.push((
                "margin_distance".to_owned(),
                "µm".to_owned(),
                r.vertex_field.clone(),
            ));
            margin_res = Some(r);
        }
    }
    if s_z > 2.0 * cfg.margin_close_um {
        warnings.push(Warning::ZResolutionCoarse);
    }
    if cancelled() {
        return Err(VolError::Cancelled);
    }

    // Kontak per struktur.
    let mut contacts = Vec::new();
    if let Some(t) = tumor.as_ref() {
        progress.set(0.3, "Kontak struktur");
        for (d, m) in masks {
            if matches!(
                d.kind,
                LabelKind::Vessel(_) | LabelKind::Nerve | LabelKind::Capsule
            ) {
                contacts.push(ContactEntry {
                    label: d.id,
                    name: d.name.clone(),
                    kind: d.kind,
                    result: contact(t, m)?,
                });
            }
        }
    }
    if cancelled() {
        return Err(VolError::Cancelled);
    }

    // Kapsul.
    let mut capsule_res = None;
    if let (Some(t), Some(c)) = (
        tumor.as_ref(),
        union_of(&by_kind(&|k| k == LabelKind::Capsule), &grid)?,
    ) {
        progress.set(0.45, "Penetrasi kapsul");
        capsule_res = Some(capsule_penetration(t, &c)?);
    }

    // Fokus.
    let mut foci_res = None;
    if let Some(t) = tumor.as_ref() {
        progress.set(0.55, "Multifokalitas");
        foci_res = Some(foci(t, cfg, tre)?);
    }
    if cancelled() {
        return Err(VolError::Cancelled);
    }

    // Kedalaman dan Breslow.
    let mut depth_res = None;
    let mut breslow_res = None;
    if let Some(t) = tumor.as_ref() {
        if let Some(s) = union_of(&by_kind(&|k| k == LabelKind::Surface), &grid)? {
            progress.set(0.65, "Kedalaman invasi");
            depth_res = depth(t, &s, tre)?;
        }
        if let Some(r) = union_of(&by_kind(&|k| k == LabelKind::Reference), &grid)? {
            progress.set(0.7, "Ketebalan Breslow");
            breslow_res = depth(t, &r, tre)?;
        }
    }

    // Ukuran.
    let size_res = tumor.as_ref().and_then(|t| {
        progress.set(0.75, "Ukuran staging");
        staging_size(t, tre)
    });

    // Fraksi.
    progress.set(0.8, "Fraksi komponen");
    let fractions_res = fractions(masks, tumor.as_ref(), cfg)?;

    // Hipoksia.
    let mut hypoxia_res = None;
    if let (Some(t), Some(v)) = (
        tumor.as_ref(),
        union_of(
            &by_kind(&|k| k == LabelKind::Vessel(VesselKind::Blood)),
            &grid,
        )?,
    ) {
        progress.set(0.85, "Proksi hipoksia");
        if let Some(h) = hypoxia(t, &v, cfg)? {
            fields.push((
                "vessel_distance".to_owned(),
                "µm".to_owned(),
                h.field.clone(),
            ));
            hypoxia_res = Some(h);
        }
    }
    if cancelled() {
        return Err(VolError::Cancelled);
    }

    // Atribut.
    let mut attr_entries = Vec::new();
    if let Some(t) = tumor.as_ref() {
        progress.set(0.9, "Statistik atribut");
        for (d, a) in attrs {
            if let Some(st) = attr_stats(a, t)? {
                let params_hash = match &d.source {
                    h3d_core::AttrSource::Computed { params_hash, .. } => Some(*params_hash),
                    _ => None,
                };
                fields.push((format!("attr:{}", d.name), d.unit.clone(), a.data.clone()));
                attr_entries.push(AttrStatsEntry {
                    name: d.name.clone(),
                    unit: d.unit.clone(),
                    params_hash,
                    stats: st,
                });
            }
        }
    }

    progress.set(1.0, "Analisis selesai");
    Ok(SpatialAnalysis {
        params: cfg.clone(),
        grid: GridSummary {
            s_xy_um: grid.s_xy,
            s_z_um: s_z,
            size: grid.size,
            nz: grid.nz(),
            uniform_z: uniform,
            resampled_z: !uniform,
        },
        tre_est_um: tre,
        notes,
        warnings,
        tumor_volume_um3: tumor_volume,
        margin: margin_res,
        contacts,
        capsule: capsule_res,
        foci: foci_res,
        depth: depth_res,
        breslow: breslow_res,
        size: size_res,
        fractions: fractions_res,
        hypoxia: hypoxia_res,
        attr_stats: attr_entries,
        fields,
    })
}

/// Proyeksikan medan kisi ke verteks mesh (interpolasi trilinear; NODATA →
/// `NaN`) sebagai `MeshScalars`.
pub fn project_fields_to_mesh(
    mesh: &mut LabelMesh,
    grid: &StackGrid,
    fields: &[(String, String, Vec<f32>)],
) {
    for (name, unit, data) in fields {
        if data.len() != grid.voxel_count() {
            continue;
        }
        let s = mesh.sample_field_trilinear(grid, data, name, unit);
        mesh.set_scalars(s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(n: usize, s: f64) -> StackGrid {
        StackGrid {
            origin_um: [0.0, 0.0],
            s_xy: s,
            size: [n as u32, n as u32],
            z_um: (0..n).map(|k| k as f64 * s).collect(),
        }
    }

    fn mask_fn(g: &StackGrid, f: impl Fn([f64; 3]) -> bool) -> VoxelMask {
        let mut m = VoxelMask::empty(g.clone());
        for i in 0..m.data.len() {
            let [x, y, z] = g.delinear(i);
            m.data[i] = u8::from(f(g.center_um(x, y, z)));
        }
        m
    }

    #[test]
    fn components_26_and_6() {
        let g = grid(4, 1.0);
        let mut m = VoxelMask::empty(g.clone());
        m.data[g.linear(0, 0, 0)] = 1;
        m.data[g.linear(1, 1, 1)] = 1; // diagonal: 26-terhubung, 6-terpisah
        assert_eq!(label_components_3d(&m.data, &g, true).1, 1);
        assert_eq!(label_components_3d(&m.data, &g, false).1, 2);
    }

    #[test]
    fn contact_area_of_small_cylinders_is_unbiased() {
        // Silinder struktur berjari-jari 3–12 voxel dengan pusat di luar
        // kisi-simetri, dikelilingi tumor. Luas benar = 2πr × panjang.
        // Normal dari gradien EDT (arah terkuantisasi) memberi +6…+10 %;
        // jumlah muka mentah +27 % (= 4/π).
        let s = 16.0;
        let n = 40usize;
        let dz = 24.0;
        let g = StackGrid {
            origin_um: [0.0, 0.0],
            s_xy: s,
            size: [n as u32, n as u32],
            z_um: (0..12).map(|k| k as f64 * dz).collect(),
        };
        let length = 12.0 * dz;
        for (r_vox, off) in [
            (3.0, [0.3, 0.7]),
            (5.12, [0.13, 0.41]),
            (5.12, [0.5, 0.5]),
            (7.3, [0.81, 0.27]),
            (12.0, [0.37, 0.59]),
        ] {
            let c = [(20.0 + off[0]) * s, (20.0 + off[1]) * s];
            let r = r_vox * s;
            let inside = |p: [f64; 3]| (p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) <= r * r;
            let vessel = mask_fn(&g, inside);
            let tumor = mask_fn(&g, |p| !inside(p));
            let res = contact(&tumor, &vessel).unwrap();
            let truth = 2.0 * std::f64::consts::PI * r * length / 1e6;
            let e = res.area_mm2 / truth - 1.0;
            let e_raw = res.area_voxel_faces_mm2 / truth - 1.0;
            println!(
                "r = {r_vox} voxel: terkoreksi {:+.1} %, mentah {:+.1} %",
                e * 100.0,
                e_raw * 100.0
            );
            assert!(e.abs() <= 0.05, "r = {r_vox}: galat luas kontak {e}");
            assert!(e_raw > 0.15, "r = {r_vox}: bias tangga mentah {e_raw}");
            assert_eq!(res.patches.len(), 1);
            assert_eq!(res.patches[0].layers.len(), 12);
        }
        // Bidang datar tegak lurus sumbu: kedua taksiran sama dan persis.
        let wall = mask_fn(&g, |p| p[0] < 10.0 * s);
        let rest = mask_fn(&g, |p| p[0] >= 10.0 * s);
        let res = contact(&rest, &wall).unwrap();
        let truth = n as f64 * s * length / 1e6;
        assert!((res.area_voxel_faces_mm2 - truth).abs() < 1e-9);
        assert!((res.area_mm2 - truth).abs() / truth < 1e-6);
    }

    #[test]
    fn gaussian_smooth_3d_preserves_constants_and_mass() {
        let dims = [7usize, 5, 4];
        let mut c = vec![2.5f32; 7 * 5 * 4];
        gaussian_smooth_3d(&mut c, dims, [1.5, 1.5, 0.8]);
        assert!(c.iter().all(|v| (v - 2.5).abs() < 1e-6));
        // Impuls di tengah: tersebar, simetris, dan jumlahnya tetap.
        let mut d = vec![0f32; 21 * 21 * 21];
        let mid = 10 + 21 * (10 + 21 * 10);
        d[mid] = 1.0;
        gaussian_smooth_3d(&mut d, [21, 21, 21], [1.5, 1.5, 1.5]);
        let sum: f32 = d.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "jumlah {sum}");
        assert!(d[mid] < 1.0 && d[mid] > d[mid + 1]);
        assert!((d[mid + 1] - d[mid - 1]).abs() < 1e-7);
        assert!((d[mid + 1] - d[mid + 21]).abs() < 1e-7);
        assert!((d[mid + 1] - d[mid + 21 * 21]).abs() < 1e-7);
        // Sigma nol pada satu sumbu = sumbu itu tidak disentuh.
        let mut e = vec![0f32; 3 * 3 * 3];
        e[13] = 1.0;
        gaussian_smooth_3d(&mut e, [3, 3, 3], [0.0, 0.0, 0.0]);
        assert_eq!(e[13], 1.0);
    }

    #[test]
    fn uncertainty_axis() {
        let u = distance_uncertainty(100.0, 16.0, 50.0, SignedAxis::NegZ, 10.0);
        assert!(u.z_limited);
        assert_eq!(u.plus_minus_um, 35.0);
        let u = distance_uncertainty(100.0, 16.0, 50.0, SignedAxis::PosX, 10.0);
        assert!(!u.z_limited);
        assert_eq!(u.plus_minus_um, 18.0);
    }

    #[test]
    fn eigen_identity_like() {
        let (vals, _) = eigen_sym3([[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]);
        assert_eq!(vals, [3.0, 2.0, 1.0]);
        let (vals, vecs) = eigen_sym3([[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 1.0]]);
        assert!((vals[0] - 3.0).abs() < 1e-9 && (vals[2] - 1.0).abs() < 1e-9);
        assert!((vecs[0][0].abs() - vecs[0][1].abs()).abs() < 1e-9);
    }

    #[test]
    fn resample_nonuniform_z() {
        let g = StackGrid {
            origin_um: [0.0, 0.0],
            s_xy: 10.0,
            size: [2, 2],
            z_um: vec![0.0, 10.0, 20.0, 40.0],
        };
        let mut m = VoxelMask::empty(g);
        m.data[8..12].fill(1); // lapis 2
        let r = resample_uniform_z(&m);
        assert_eq!(r.grid.z_um, vec![0.0, 10.0, 20.0, 30.0, 40.0]);
        assert_eq!(r.data[4..8], [0, 0, 0, 0]);
        assert_eq!(r.data[8..12], [1, 1, 1, 1]);
        // z = 30: seri antara lapis 20 dan 40 → lapis berindeks lebih kecil.
        assert_eq!(r.data[12..16], [1, 1, 1, 1]);
        assert_eq!(r.data[16..20], [0, 0, 0, 0]);
    }

    #[test]
    fn staging_size_of_box() {
        let g = grid(20, 10.0);
        let t = mask_fn(&g, |p| {
            (20.0..=140.0).contains(&p[0])
                && (60.0..=100.0).contains(&p[1])
                && (80.0..=90.0).contains(&p[2])
        });
        let r = staging_size(&t, 0.0).unwrap();
        assert!((r.pca_axes_um[0].value_um - 120.0).abs() < 1e-6);
        assert!((r.pca_axes_um[1].value_um - 40.0).abs() < 1e-6);
        assert!((r.pca_axes_um[2].value_um - 20.0).abs() < 1e-6);
        assert!(r.pca_axes_um[2].z_limited);
    }
}
