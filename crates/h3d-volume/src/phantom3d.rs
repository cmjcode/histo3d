//! Fantom 3D berkebenaran dasar (histo3d.md 8.12.11, 11.1) untuk uji dan
//! CLI `h3d phantom`, serta [`ArrayLabelSampler`] agar seluruh pipeline bisa
//! dijalankan atas masker sintetis dengan transformasi identitas.
//!
//! Koordinat fantom = ruang U = ruang W (identitas): piksel `(x, y)`
//! berpusat di `((x + 0,5)·s_xy, (y + 0,5)·s_xy)`; bentuk berpusat di
//! `size_um / 2`. Sumbu z: tiap irisan punya `z_um` eksplisit.
//!
//! Label tetap: 1 = Tumor, 2 = Margin, 3 = Vessel, 4 = Capsule, 5 = Surface.

use ndarray::Array2;

use crate::recon::LabelSampler;

/// Label `Tumor` di fantom.
pub const LABEL_TUMOR: u8 = 1;
/// Label `Margin`.
pub const LABEL_MARGIN: u8 = 2;
/// Label `Vessel`.
pub const LABEL_VESSEL: u8 = 3;
/// Label `Capsule`.
pub const LABEL_CAPSULE: u8 = 4;
/// Label `Surface`.
pub const LABEL_SURFACE: u8 = 5;

/// Bentuk dasar tumor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PhantomShape {
    /// Ellipsoid berjari-jari `[rx, ry, rz]` µm.
    Ellipsoid {
        /// Jari-jari.
        radii: [f64; 3],
    },
    /// Torus sumbu z.
    Torus {
        /// Jari-jari mayor.
        major: f64,
        /// Jari-jari minor.
        minor: f64,
    },
    /// Cabang Y (lihat `h3d_imgproc::phantom::branch_capsules`).
    Branch {
        /// Panjang batang.
        trunk_len: f64,
        /// Panjang lengan.
        arm_len: f64,
        /// Jari-jari.
        radius: f64,
        /// Setengah sudut bukaan, radian.
        half_angle: f64,
    },
}

/// Spesifikasi fantom.
#[derive(Clone, Debug, PartialEq)]
pub struct PhantomSpec {
    /// Bentuk tumor.
    pub shape: PhantomShape,
    /// Jumlah irisan.
    pub n_slices: usize,
    /// Jarak irisan, µm.
    pub spacing_um: f64,
    /// Ukuran piksel masker, µm.
    pub s_xy: f64,
    /// Ukuran bidang, µm `[w, h]`.
    pub size_um: [f64; 2],
    /// Bidang `Margin` `x ≥ rx + d` (jarak `d` dari permukaan ke arah +X).
    pub margin_plane: Option<f64>,
    /// `(jumlah, celah)`: fokus bola sejajar sumbu X dengan celah `g` µm;
    /// menggantikan bentuk dasar (jari-jari `radii[0]`).
    pub foci: Option<(usize, f64)>,
    /// Silinder `Vessel` sumbu z berjari-jari ini melalui pusat.
    pub vessel: Option<f64>,
    /// `(tebal, tembus?)`: kulit bola `Capsule` di sekeliling tumor.
    pub capsule: Option<(f64, bool)>,
    /// Lapisan `Surface` datar di `z = 0`, tumor silinder sampai kedalaman `h`.
    pub surface_at_z0: Option<f64>,
    /// Benih (dicadangkan untuk variasi acak).
    pub seed: u64,
}

impl Default for PhantomSpec {
    fn default() -> Self {
        PhantomSpec {
            shape: PhantomShape::Ellipsoid {
                radii: [1500.0, 1200.0, 1000.0],
            },
            n_slices: 21,
            spacing_um: 100.0,
            s_xy: 16.0,
            size_um: [6400.0, 6400.0],
            margin_plane: None,
            foci: None,
            vessel: None,
            capsule: None,
            surface_at_z0: None,
            seed: 1,
        }
    }
}

/// Satu irisan fantom.
#[derive(Clone, Debug, PartialEq)]
pub struct PhantomSlice {
    /// Posisi z, µm.
    pub z_um: f64,
    /// Masker per label (hanya label yang ada di irisan ini).
    pub masks: Vec<(u8, Array2<u8>)>,
}

/// Kebenaran analitik.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct PhantomTruth {
    /// Volume tumor, µm³.
    pub volume_um3: f64,
    /// Jarak margin minimum, µm (bila ada bidang margin).
    pub d_min: Option<f64>,
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

/// Fantom lengkap.
#[derive(Clone, Debug, PartialEq)]
pub struct Phantom {
    /// Spesifikasi.
    pub spec: PhantomSpec,
    /// Irisan terurut z.
    pub slices: Vec<PhantomSlice>,
    /// Kebenaran.
    pub truth: PhantomTruth,
}

fn px_count(size_um: [f64; 2], s: f64) -> [usize; 2] {
    [
        (size_um[0] / s).round().max(1.0) as usize,
        (size_um[1] / s).round().max(1.0) as usize,
    ]
}

/// Masker dari predikat atas koordinat dunia `(x, y)` µm.
fn mask_from(size_px: [usize; 2], s: f64, f: impl Fn(f64, f64) -> bool) -> Array2<u8> {
    Array2::from_shape_fn((size_px[1], size_px[0]), |(y, x)| {
        u8::from(f((x as f64 + 0.5) * s, (y as f64 + 0.5) * s))
    })
}

fn any(m: &Array2<u8>) -> bool {
    m.iter().any(|&v| v != 0)
}

/// Bangun fantom dari spesifikasi. Deterministik.
pub fn generate(spec: &PhantomSpec) -> Phantom {
    let s = spec.s_xy;
    let size_px = px_count(spec.size_um, s);
    let cx = spec.size_um[0] / 2.0;
    let cy = spec.size_um[1] / 2.0;
    let n = spec.n_slices.max(1);
    // Susunan z: berpusat di 0, kecuali fantom kedalaman (permukaan di z=0).
    let z0 = if spec.surface_at_z0.is_some() {
        -spec.spacing_um
    } else {
        -((n - 1) as f64) / 2.0 * spec.spacing_um
    };
    let mut truth = PhantomTruth {
        n_foci: 1,
        ..Default::default()
    };
    let radii = match spec.shape {
        PhantomShape::Ellipsoid { radii } => radii,
        PhantomShape::Torus { major, minor } => [major + minor, major + minor, minor],
        PhantomShape::Branch { radius, .. } => [radius, radius, radius],
    };
    let r0 = radii[0];

    // Fokus: pusat sepanjang X.
    let foci_centers: Vec<f64> = match spec.foci {
        Some((k, g)) if k >= 1 => {
            let pitch = 2.0 * r0 + g;
            let total = pitch * (k - 1) as f64;
            (0..k)
                .map(|i| cx - total / 2.0 + i as f64 * pitch)
                .collect()
        }
        _ => vec![cx],
    };
    if let Some((k, g)) = spec.foci {
        truth.n_foci = k;
        truth.foci_gap_um = Some(g);
    }
    // Kapsul: kulit bola di sekeliling tumor.
    let capsule_inner = spec.capsule.map(|_| r0 * 1.3 + 3.0 * s);
    let breach_len = spec
        .capsule
        .filter(|(_, breach)| *breach)
        .map(|(t, _)| r0 * 1.3 + 3.0 * s + t + 6.0 * s);
    truth.capsule_breached = spec.capsule.map(|(_, b)| b);

    let mut slices = Vec::with_capacity(n);
    for k in 0..n {
        let z = z0 + k as f64 * spec.spacing_um;
        let mut masks: Vec<(u8, Array2<u8>)> = Vec::new();

        // Tumor.
        let tumor: Array2<u8> = if let Some(h) = spec.surface_at_z0 {
            // Silinder sumbu z berjari-jari rx dari z ∈ (0, h].
            if z > 0.0 && z <= h + 1e-9 {
                mask_from(size_px, s, |x, y| {
                    ((x - cx) / radii[0]).powi(2) + ((y - cy) / radii[1]).powi(2) <= 1.0
                })
            } else {
                Array2::zeros((size_px[1], size_px[0]))
            }
        } else {
            match spec.shape {
                PhantomShape::Ellipsoid { radii } => {
                    let r = if spec.foci.is_some() {
                        [radii[0], radii[0], radii[0]]
                    } else {
                        radii
                    };
                    let centers = &foci_centers;
                    let mut m = mask_from(size_px, s, |x, y| {
                        centers.iter().any(|&fx| {
                            ((x - fx) / r[0]).powi(2)
                                + ((y - cy) / r[1]).powi(2)
                                + (z / r[2]).powi(2)
                                <= 1.0
                        })
                    });
                    if let Some(len) = breach_len {
                        // Tonjolan silinder ke arah +X menembus kapsul.
                        let pr = 0.4 * r0;
                        if z.abs() <= pr {
                            let hw = (pr * pr - z * z).sqrt();
                            let prot = mask_from(size_px, s, |x, y| {
                                x >= cx && x <= cx + len && (y - cy).abs() <= hw
                            });
                            for (a, &b) in m.iter_mut().zip(prot.iter()) {
                                *a |= b;
                            }
                        }
                    }
                    m
                }
                // `h3d_imgproc::phantom` memusatkan bentuk di tengah citra
                // dengan konvensi pusat piksel yang sama, sehingga langsung
                // dipakai.
                PhantomShape::Torus { major, minor } => {
                    h3d_imgproc::phantom::slice_torus(major, minor, z, s, size_px)
                }
                PhantomShape::Branch {
                    trunk_len,
                    arm_len,
                    radius,
                    half_angle,
                } => h3d_imgproc::phantom::slice_branch(
                    trunk_len, arm_len, radius, half_angle, z, s, size_px,
                ),
            }
        };

        // Kapsul: kulit bola; tonjolan tumor melubanginya.
        if let (Some((t, _)), Some(ri)) = (spec.capsule, capsule_inner) {
            let ro = ri + t;
            let mut cap = mask_from(size_px, s, |x, y| {
                let d2 = (x - cx).powi(2) + (y - cy).powi(2) + z * z;
                d2 >= ri * ri && d2 <= ro * ro
            });
            for (c, &tv) in cap.iter_mut().zip(tumor.iter()) {
                if tv != 0 {
                    *c = 0;
                }
            }
            if any(&cap) {
                masks.push((LABEL_CAPSULE, cap));
            }
        }

        // Pembuluh: silinder sumbu z; tumor dikurangi pembuluh.
        let mut tumor = tumor;
        if let Some(rv) = spec.vessel {
            let vessel = mask_from(size_px, s, |x, y| {
                (x - cx).powi(2) + (y - cy).powi(2) <= rv * rv
            });
            for (t, &v) in tumor.iter_mut().zip(vessel.iter()) {
                if v != 0 {
                    *t = 0;
                }
            }
            masks.push((LABEL_VESSEL, vessel));
        }

        // Margin: bidang x ≥ rx + d.
        if let Some(d) = spec.margin_plane {
            let xm = cx + r0 + d;
            let m = mask_from(size_px, s, |x, _| x >= xm);
            if any(&m) {
                masks.push((LABEL_MARGIN, m));
            }
        }

        // Permukaan di z = 0.
        if spec.surface_at_z0.is_some() && z.abs() < 1e-9 {
            masks.push((
                LABEL_SURFACE,
                Array2::from_elem((size_px[1], size_px[0]), 1u8),
            ));
        }

        if any(&tumor) {
            masks.insert(0, (LABEL_TUMOR, tumor));
        }
        slices.push(PhantomSlice { z_um: z, masks });
    }

    // Kebenaran analitik.
    truth.volume_um3 = match spec.shape {
        PhantomShape::Ellipsoid { radii } => {
            if let Some(h) = spec.surface_at_z0 {
                std::f64::consts::PI * radii[0] * radii[1] * h
            } else if let Some((k, _)) = spec.foci {
                k as f64 * h3d_imgproc::phantom::ellipsoid_volume([r0, r0, r0])
            } else {
                h3d_imgproc::phantom::ellipsoid_volume(radii)
            }
        }
        PhantomShape::Torus { major, minor } => h3d_imgproc::phantom::torus_volume(major, minor),
        PhantomShape::Branch { .. } => f64::NAN,
    };
    truth.d_min = spec.margin_plane;
    truth.depth_um = spec.surface_at_z0;
    if let (Some(rv), PhantomShape::Ellipsoid { radii }) = (spec.vessel, spec.shape) {
        // Lingkaran r_v di dalam elips r_x·k, r_y·k: k ≥ r_v / min(rx, ry).
        let rmin = radii[0].min(radii[1]);
        let l = if rv < rmin {
            2.0 * radii[2] * (1.0 - (rv / rmin).powi(2)).sqrt()
        } else {
            0.0
        };
        truth.contact_area_mm2 = Some(2.0 * std::f64::consts::PI * rv * l / 1e6);
    }
    Phantom {
        spec: spec.clone(),
        slices,
        truth,
    }
}

/// [`LabelSampler`] atas masker `Array2<u8>` per label (transformasi
/// identitas, `u = (px + 0,5) × mpp`). Label dengan indeks lebih kecil dalam
/// daftar menang bila tumpang tindih.
pub struct ArrayLabelSampler {
    masks: Vec<(u8, Array2<u8>)>,
    bboxes: Vec<(u8, [f64; 4])>,
    mpp: f64,
}

impl ArrayLabelSampler {
    /// Dari daftar `(label, masker)` dengan resolusi `mpp` µm/px.
    pub fn new(masks: Vec<(u8, Array2<u8>)>, mpp: f64) -> Self {
        let mut bboxes = Vec::new();
        for (label, m) in &masks {
            let (h, w) = m.dim();
            let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
            let mut found = false;
            for y in 0..h {
                for x in 0..w {
                    if m[[y, x]] != 0 {
                        found = true;
                        x0 = x0.min(x);
                        y0 = y0.min(y);
                        x1 = x1.max(x);
                        y1 = y1.max(y);
                    }
                }
            }
            if found {
                bboxes.push((
                    *label,
                    [
                        x0 as f64 * mpp,
                        y0 as f64 * mpp,
                        (x1 + 1) as f64 * mpp,
                        (y1 + 1) as f64 * mpp,
                    ],
                ));
            }
        }
        ArrayLabelSampler { masks, bboxes, mpp }
    }

    /// Dari satu irisan fantom.
    pub fn from_slice(slice: &PhantomSlice, mpp: f64) -> Self {
        Self::new(slice.masks.clone(), mpp)
    }

    /// Label yang ada (`(label, interpolasi = false)`), untuk `SliceInput`.
    pub fn annotated_labels(&self) -> Vec<(u8, bool)> {
        self.bboxes.iter().map(|(l, _)| (*l, false)).collect()
    }
}

impl LabelSampler for ArrayLabelSampler {
    fn label_at_um(&self, u: [f64; 2]) -> u8 {
        if !(u[0] >= 0.0 && u[1] >= 0.0) {
            return 0;
        }
        let x = (u[0] / self.mpp).floor() as usize;
        let y = (u[1] / self.mpp).floor() as usize;
        for (label, m) in &self.masks {
            let (h, w) = m.dim();
            if x < w && y < h && m[[y, x]] != 0 {
                return *label;
            }
        }
        0
    }

    fn mpp_label(&self) -> f64 {
        self.mpp
    }

    fn bbox_um(&self, label: u8) -> Option<[f64; 4]> {
        self.bboxes
            .iter()
            .find(|(l, _)| *l == label)
            .map(|(_, b)| *b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ellipsoid_phantom_slices_and_truth() {
        let p = generate(&PhantomSpec::default());
        assert_eq!(p.slices.len(), 21);
        assert!((p.slices[10].z_um).abs() < 1e-9);
        assert!(p.slices[10].masks.iter().any(|(l, _)| *l == LABEL_TUMOR));
        let truth = 4.0 / 3.0 * std::f64::consts::PI * 1500.0 * 1200.0 * 1000.0;
        assert!((p.truth.volume_um3 - truth).abs() < 1e-3);
        let smp = ArrayLabelSampler::from_slice(&p.slices[10], 16.0);
        assert_eq!(smp.label_at_um([3200.0, 3200.0]), LABEL_TUMOR);
        assert_eq!(smp.label_at_um([10.0, 10.0]), 0);
        let bb = smp.bbox_um(LABEL_TUMOR).unwrap();
        assert!((bb[0] - 1700.0).abs() <= 16.0 && (bb[2] - 4700.0).abs() <= 16.0);
    }
}
