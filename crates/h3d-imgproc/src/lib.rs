//! `h3d-imgproc`: pemrosesan citra murni untuk HISTO3D (histo3d.md 8.3,
//! 8.12.5, 11.1). Tanpa I/O, tanpa GUI; bekerja pada `ndarray::Array2` dan
//! buffer `&[u8]`/`&[f32]` dengan dimensi eksplisit.
//!
//! # Konvensi
//! - `Array2<T>` berbentuk `(h, w)`: indeks `[y, x]`. Buffer RGBA row-major,
//!   4 byte/piksel. Ukuran keluaran `[w, h]`; dimensi 3D `[nx, ny, nz]`
//!   dengan indeks linear `x + nx·(y + ny·z)`.
//! - Pusat piksel di `+0,5` (histo3d.md 5.2) di semua konversi.
//! - `warp_affine_*` menerima `t_inv` **tujuan → sumber** (resampling mundur).
//! - EDT: jarak dari piksel NOL ke piksel SATU terdekat; SDF **negatif di
//!   dalam** masker.
//! - `phase_correlation` mengembalikan `(dx, dy)` sehingga `moving` yang
//!   digeser `(dx, dy)` berimpit dengan `fixed`.
//! - Semua hasil deterministik: reduksi paralel ditulis per baris/chunk ke
//!   lokasi tetap, tanpa penggabungan bergantung urutan thread.

pub mod color;
pub mod components;
pub mod edt;
pub mod error;
pub mod fft;
pub mod filters;
pub mod ihc;
pub mod metrics;
pub mod morphology;
pub mod nuclei;
pub mod phantom;
pub mod pyramid;
pub mod threshold;
pub mod tissue;
pub mod warp;

pub use color::{
    deconvolve, hematoxylin_channel, od_of, od_total, percentile, percentile_stretch, rgb_to_od,
    StainMatrix, DAB, EOSIN, HEMATOXYLIN,
};
pub use components::{
    component_areas, label_components, label_components_conn, largest_component,
    remove_small_components, Connectivity,
};
pub use edt::{
    edt3_anisotropic, edt3_anisotropic_with_progress, edt3_argmin, edt_2d_um, edt_squared_2d,
    signed_distance_2d, signed_distance_2d_um, NO_INDEX,
};
pub use error::ImgError;
pub use fft::{fft2_real, ifft2, phase_correlation};
pub use filters::{box_sum, gaussian_blur_f32, hann_window_2d, local_minima, log_filter};
pub use ihc::{
    allred_proportion_score, score_area_based, score_nuclear, tps_cps, AreaScore, IhcThresholds,
    NuclearScore,
};
pub use metrics::{dice, moments, mutual_information, ncc_masked, Moments};
pub use morphology::{close, dilate_disc, erode_disc, fill_holes, open};
pub use nuclei::{
    detect_nuclei, disc_mean, measure_chromogen, nuclei_density_per_mm2, NucleiParams, Nucleus,
};
pub use pyramid::{
    downsample2_attr_weighted, downsample2_f32, downsample2_label_mode, downsample2_rgba, half_size,
};
pub use threshold::{otsu, otsu_masked};
pub use tissue::tissue_mask;
pub use warp::{resize_f32, warp_affine_f32, warp_affine_rgba, warp_affine_u8, Interp};
