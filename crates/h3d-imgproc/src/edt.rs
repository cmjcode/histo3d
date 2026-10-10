//! Transformasi jarak Euclid eksak (Felzenszwalb & Huttenlocher) 2D dan 3D
//! anisotropik, serta SDF bertanda (histo3d.md 8.3, 8.6.2, 8.12.6).
//!
//! Konvensi: **jarak diukur dari tiap piksel/voxel bernilai NOL ke
//! piksel/voxel bernilai SATU terdekat**; piksel bernilai satu mendapat 0.
//! Bila tidak ada piksel bernilai satu sama sekali, semua jarak `+∞`.
//! Semua fungsi `*_squared*` mengembalikan jarak **kuadrat**.
//!
//! Bentuk `Array2`: `(h, w)` = `[y, x]`. Untuk 3D, `dims = [nx, ny, nz]` dan
//! indeks linear `i = x + nx·(y + ny·z)` (x tercepat).

use h3d_core::progress::{NoProgress, Progress};
use ndarray::Array2;
use rayon::prelude::*;

use crate::error::ImgError;

/// Indeks "tidak ada" untuk argmin.
pub const NO_INDEX: u32 = u32::MAX;

/// Transformasi jarak 1D: `out[q] = min_v f[v] + w·(q − v)²`.
/// `f[v] = +∞` berarti tidak ada parabola di `v`. Bila `arg_in`/`arg_out`
/// diberikan, `arg_out[q]` = `arg_in[v*]` untuk `v*` pemenang (seri → `v`
/// terkecil). `v` dan `z` adalah buffer kerja berukuran `≥ n` dan `≥ n + 1`.
#[allow(clippy::too_many_arguments)]
fn dt1d(
    f: &[f64],
    w: f64,
    out: &mut [f64],
    arg_in: Option<&[u32]>,
    arg_out: Option<&mut [u32]>,
    v: &mut [usize],
    z: &mut [f64],
) {
    let n = f.len();
    let mut k: usize = 0;
    let mut any = false;
    for q in 0..n {
        if f[q] == f64::INFINITY {
            continue;
        }
        if !any {
            any = true;
            v[0] = q;
            z[0] = f64::NEG_INFINITY;
            z[1] = f64::INFINITY;
            continue;
        }
        loop {
            let p = v[k];
            let s = ((f[q] + w * (q * q) as f64) - (f[p] + w * (p * p) as f64))
                / (2.0 * w * (q as f64 - p as f64));
            if s <= z[k] {
                if k == 0 {
                    // Parabola q mengalahkan semuanya.
                    v[0] = q;
                    z[0] = f64::NEG_INFINITY;
                    z[1] = f64::INFINITY;
                    break;
                }
                k -= 1;
            } else {
                k += 1;
                v[k] = q;
                z[k] = s;
                z[k + 1] = f64::INFINITY;
                break;
            }
        }
    }
    if !any {
        out.fill(f64::INFINITY);
        if let Some(ao) = arg_out {
            ao.fill(NO_INDEX);
        }
        return;
    }
    let mut k = 0usize;
    match (arg_in, arg_out) {
        (Some(ai), Some(ao)) => {
            for q in 0..n {
                while z[k + 1] < q as f64 {
                    k += 1;
                }
                let p = v[k];
                let d = q as f64 - p as f64;
                out[q] = f[p] + w * d * d;
                ao[q] = ai[p];
            }
        }
        _ => {
            for (q, o) in out.iter_mut().enumerate() {
                while z[k + 1] < q as f64 {
                    k += 1;
                }
                let p = v[k];
                let d = q as f64 - p as f64;
                *o = f[p] + w * d * d;
            }
        }
    }
}

/// Transpos buffer row-major `(rows, cols)` → `(cols, rows)`.
fn transpose(src: &[f64], rows: usize, cols: usize, dst: &mut [f64]) {
    dst.par_chunks_mut(rows)
        .enumerate()
        .for_each(|(c, out_row)| {
            for (r, o) in out_row.iter_mut().enumerate() {
                *o = src[r * cols + c];
            }
        });
}

/// EDT kuadrat 2D (piksel²) dalam `f64` pada buffer row-major `(h, w)`.
/// `f` masukan: 0 pada piksel satu, `+∞` pada piksel nol.
fn edt2_inplace(f: &mut [f64], w: usize, h: usize) {
    // Lintasan baris (sumbu x).
    f.par_chunks_mut(w).for_each(|row| {
        let input = row.to_vec();
        let mut v = vec![0usize; w];
        let mut z = vec![0.0f64; w + 1];
        dt1d(&input, 1.0, row, None, None, &mut v, &mut z);
    });
    // Lintasan kolom (sumbu y) lewat transpos agar tiap kolom kontigu.
    let mut t = vec![0.0f64; w * h];
    transpose(f, h, w, &mut t);
    t.par_chunks_mut(h).for_each(|col| {
        let input = col.to_vec();
        let mut v = vec![0usize; h];
        let mut z = vec![0.0f64; h + 1];
        dt1d(&input, 1.0, col, None, None, &mut v, &mut z);
    });
    transpose(&t, w, h, f);
}

/// EDT kuadrat 2D (satuan piksel²): untuk tiap piksel NOL, jarak kuadrat ke
/// piksel SATU terdekat; piksel SATU bernilai 0. Tanpa piksel satu → `+∞`.
pub fn edt_squared_2d(mask: &Array2<u8>) -> Array2<f32> {
    let (h, w) = mask.dim();
    if h == 0 || w == 0 {
        return Array2::zeros((h, w));
    }
    let mut f: Vec<f64> = mask
        .iter()
        .map(|&m| if m != 0 { 0.0 } else { f64::INFINITY })
        .collect();
    edt2_inplace(&mut f, w, h);
    let out: Vec<f32> = f.iter().map(|&d| d as f32).collect();
    Array2::from_shape_vec((h, w), out).unwrap_or_else(|_| Array2::zeros((h, w)))
}

/// EDT 2D dalam mikrometer: `sqrt(EDT²) × s_xy`.
pub fn edt_2d_um(mask: &Array2<u8>, s_xy: f64) -> Array2<f32> {
    let sq = edt_squared_2d(mask);
    sq.mapv(|d| ((d as f64).sqrt() * s_xy) as f32)
}

/// Hasil lintasan z satu chunk: kolom `(indeks xy, jarak per z)` dan argmin.
type ZColumns = (Vec<(usize, Vec<f32>)>, Vec<Vec<u32>>);

/// Inti bersama `edt3_anisotropic` / `edt3_argmin`.
fn edt3_core(
    mask: &[u8],
    dims: [usize; 3],
    spacing: [f64; 3],
    want_arg: bool,
    progress: &dyn Progress,
) -> Result<(Vec<f32>, Option<Vec<u32>>), ImgError> {
    let [nx, ny, nz] = dims;
    let n = nx
        .checked_mul(ny)
        .and_then(|a| a.checked_mul(nz))
        .ok_or_else(|| ImgError::InvalidParameter("dimensi volume meluap".to_string()))?;
    if n == 0 {
        return Err(ImgError::EmptyInput);
    }
    if mask.len() != n {
        return Err(ImgError::shape(&[n], &[mask.len()]));
    }
    if want_arg && n > NO_INDEX as usize {
        return Err(ImgError::InvalidParameter(
            "volume terlalu besar untuk indeks u32".to_string(),
        ));
    }
    for (k, s) in spacing.iter().enumerate() {
        if !(s.is_finite() && *s > 0.0) {
            return Err(ImgError::InvalidParameter(format!(
                "spacing sumbu {k} harus positif: {s}"
            )));
        }
    }
    let wx = spacing[0] * spacing[0];
    let wy = spacing[1] * spacing[1];
    let wz = spacing[2] * spacing[2];
    let slice = nx * ny;

    let mut dist = vec![0.0f32; n];
    let mut arg = if want_arg {
        Some(vec![NO_INDEX; n])
    } else {
        None
    };

    // Tahap 1: lintasan x lalu y per irisan z (independen, paralel per irisan).
    // Hasil antara disimpan sebagai f32 (memori O(n × 4)); eksak selama nilai
    // < 2^24 dan spacing diadik.
    {
        let arg_chunks: Vec<Option<&mut [u32]>> = match arg.as_mut() {
            Some(a) => a.chunks_mut(slice).map(Some).collect(),
            None => (0..nz).map(|_| None).collect(),
        };
        dist.par_chunks_mut(slice)
            .zip(arg_chunks.into_par_iter())
            .enumerate()
            .for_each(|(z, (dslice, aslice))| {
                let base = z * slice;
                let msl = &mask[base..base + slice];
                let mut f: Vec<f64> = msl
                    .iter()
                    .map(|&m| if m != 0 { 0.0 } else { f64::INFINITY })
                    .collect();
                let mut g = vec![0.0f64; slice];
                let mut v = vec![0usize; nx.max(ny)];
                let mut zb = vec![0.0f64; nx.max(ny) + 1];
                match aslice {
                    None => {
                        // x
                        for y in 0..ny {
                            let r = y * nx..(y + 1) * nx;
                            dt1d(&f[r.clone()], wx, &mut g[r], None, None, &mut v, &mut zb);
                        }
                        // y (kolom strided → kumpulkan)
                        let mut col = vec![0.0f64; ny];
                        let mut outc = vec![0.0f64; ny];
                        for x in 0..nx {
                            for y in 0..ny {
                                col[y] = g[y * nx + x];
                            }
                            dt1d(&col, wy, &mut outc, None, None, &mut v, &mut zb);
                            for y in 0..ny {
                                f[y * nx + x] = outc[y];
                            }
                        }
                        for (d, &val) in dslice.iter_mut().zip(f.iter()) {
                            *d = val as f32;
                        }
                    }
                    Some(aslice) => {
                        let mut a_in: Vec<u32> = (0..slice)
                            .map(|i| {
                                if msl[i] != 0 {
                                    (base + i) as u32
                                } else {
                                    NO_INDEX
                                }
                            })
                            .collect();
                        let mut a_g = vec![NO_INDEX; slice];
                        for y in 0..ny {
                            let r = y * nx..(y + 1) * nx;
                            dt1d(
                                &f[r.clone()],
                                wx,
                                &mut g[r.clone()],
                                Some(&a_in[r.clone()]),
                                Some(&mut a_g[r]),
                                &mut v,
                                &mut zb,
                            );
                        }
                        let mut col = vec![0.0f64; ny];
                        let mut outc = vec![0.0f64; ny];
                        let mut acol = vec![NO_INDEX; ny];
                        let mut aout = vec![NO_INDEX; ny];
                        for x in 0..nx {
                            for y in 0..ny {
                                col[y] = g[y * nx + x];
                                acol[y] = a_g[y * nx + x];
                            }
                            dt1d(
                                &col,
                                wy,
                                &mut outc,
                                Some(&acol),
                                Some(&mut aout),
                                &mut v,
                                &mut zb,
                            );
                            for y in 0..ny {
                                f[y * nx + x] = outc[y];
                                a_in[y * nx + x] = aout[y];
                            }
                        }
                        for (d, &val) in dslice.iter_mut().zip(f.iter()) {
                            *d = val as f32;
                        }
                        aslice.copy_from_slice(&a_in);
                    }
                }
            });
    }
    progress.set(0.5, "edt3: xy");
    if progress.is_cancelled() {
        return Err(ImgError::Cancelled);
    }

    // Tahap 2: lintasan z per chunk 64×64 kolom XY (seluruh kolom z dibaca).
    if nz > 1 {
        const CH: usize = 64;
        let n_cx = nx.div_ceil(CH);
        let n_cy = ny.div_ceil(CH);
        let chunks: Vec<(usize, usize)> = (0..n_cy)
            .flat_map(|cy| (0..n_cx).map(move |cx| (cx, cy)))
            .collect();
        // Tiap chunk menulis kolom-kolom miliknya sendiri; untuk menghindari
        // berbagi mutable, hitung hasil per chunk lalu tulis balik berurutan.
        let dist_ro = &dist;
        let arg_ro = arg.as_deref();
        let results: Vec<ZColumns> = chunks
            .par_iter()
            .map(|&(cx, cy)| {
                let x0 = cx * CH;
                let x1 = (x0 + CH).min(nx);
                let y0 = cy * CH;
                let y1 = (y0 + CH).min(ny);
                let mut v = vec![0usize; nz];
                let mut zb = vec![0.0f64; nz + 1];
                let mut col = vec![0.0f64; nz];
                let mut outc = vec![0.0f64; nz];
                let mut acol = vec![NO_INDEX; nz];
                let mut aout = vec![NO_INDEX; nz];
                let mut cols = Vec::with_capacity((x1 - x0) * (y1 - y0));
                let mut acols = Vec::new();
                for y in y0..y1 {
                    for x in x0..x1 {
                        let p = x + nx * y;
                        for z in 0..nz {
                            col[z] = dist_ro[p + z * slice] as f64;
                        }
                        if let Some(a) = arg_ro {
                            for z in 0..nz {
                                acol[z] = a[p + z * slice];
                            }
                            dt1d(
                                &col,
                                wz,
                                &mut outc,
                                Some(&acol),
                                Some(&mut aout),
                                &mut v,
                                &mut zb,
                            );
                            acols.push(aout.clone());
                        } else {
                            dt1d(&col, wz, &mut outc, None, None, &mut v, &mut zb);
                        }
                        cols.push((p, outc.iter().map(|&d| d as f32).collect::<Vec<f32>>()));
                    }
                }
                (cols, acols)
            })
            .collect();
        for (ci, (cols, acols)) in results.into_iter().enumerate() {
            for (k, (p, vals)) in cols.into_iter().enumerate() {
                for (z, val) in vals.into_iter().enumerate() {
                    dist[p + z * slice] = val;
                }
                if let Some(a) = arg.as_mut() {
                    for (z, &ai) in acols[k].iter().enumerate() {
                        a[p + z * slice] = ai;
                    }
                }
            }
            if ci % 16 == 15 && progress.is_cancelled() {
                return Err(ImgError::Cancelled);
            }
        }
    }
    progress.set(1.0, "edt3: selesai");
    Ok((dist, arg))
}

/// EDT kuadrat 3D anisotropik dalam **µm²**: untuk tiap voxel NOL, jarak
/// kuadrat ke voxel SATU terdekat dengan bobot `spacing[k]²` per sumbu.
/// `dims = [nx, ny, nz]`, indeks linear `x + nx·(y + ny·z)`. Sumbu z
/// diproses terakhir per chunk 64×64 kolom.
pub fn edt3_anisotropic(
    mask: &[u8],
    dims: [usize; 3],
    spacing: [f64; 3],
) -> Result<Vec<f32>, ImgError> {
    edt3_core(mask, dims, spacing, false, &NoProgress).map(|(d, _)| d)
}

/// Seperti [`edt3_anisotropic`] dengan laporan progres/pembatalan.
pub fn edt3_anisotropic_with_progress(
    mask: &[u8],
    dims: [usize; 3],
    spacing: [f64; 3],
    progress: &dyn Progress,
) -> Result<Vec<f32>, ImgError> {
    edt3_core(mask, dims, spacing, false, progress).map(|(d, _)| d)
}

/// Seperti [`edt3_anisotropic`] dan juga mengembalikan, untuk tiap voxel,
/// indeks linear voxel SATU terdekat (seri → indeks terkecil; `NO_INDEX`
/// bila tidak ada voxel satu).
pub fn edt3_argmin(
    mask: &[u8],
    dims: [usize; 3],
    spacing: [f64; 3],
) -> Result<(Vec<f32>, Vec<u32>), ImgError> {
    let (d, a) = edt3_core(mask, dims, spacing, true, &NoProgress)?;
    Ok((d, a.unwrap_or_default()))
}

/// SDF 2D dalam piksel: `sdf = EDT(luar) − EDT(dalam)` dengan koreksi
/// setengah piksel, **negatif di dalam** (`mask != 0`), dijepit ke
/// `±band_px`. Piksel tepat di tepi bernilai ±0,5. Masker penuh →
/// `−band_px` seluruhnya; masker kosong → `+band_px`.
pub fn signed_distance_2d(mask: &Array2<u8>, band_px: f32) -> Array2<f32> {
    signed_distance_impl(mask, 1.0, band_px)
}

/// SDF 2D dalam µm (`s_xy` = ukuran piksel), dijepit `±band_um`.
pub fn signed_distance_2d_um(mask: &Array2<u8>, s_xy: f64, band_um: f32) -> Array2<f32> {
    signed_distance_impl(mask, s_xy, band_um)
}

fn signed_distance_impl(mask: &Array2<u8>, scale: f64, band: f32) -> Array2<f32> {
    let (h, w) = mask.dim();
    if h == 0 || w == 0 {
        return Array2::zeros((h, w));
    }
    let band = band.abs() as f64;
    let inv: Array2<u8> = mask.mapv(|m| u8::from(m == 0));
    let d_out = edt_squared_2d(mask); // jarak piksel luar ke dalam
    let d_in = edt_squared_2d(&inv); // jarak piksel dalam ke luar
    let mut out = Array2::<f32>::zeros((h, w));
    for ((o, &m), (&eo, &ei)) in out
        .iter_mut()
        .zip(mask.iter())
        .zip(d_out.iter().zip(d_in.iter()))
    {
        let v = if m != 0 {
            -((ei as f64).sqrt() - 0.5) * scale
        } else {
            ((eo as f64).sqrt() - 0.5) * scale
        };
        *o = v.clamp(-band, band) as f32;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RNG kecil untuk tes (splitmix64).
    fn rng(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn brute_2d(mask: &Array2<u8>) -> Array2<f32> {
        let (h, w) = mask.dim();
        let ones: Vec<(usize, usize)> = (0..h)
            .flat_map(|y| (0..w).map(move |x| (y, x)))
            .filter(|&(y, x)| mask[[y, x]] != 0)
            .collect();
        Array2::from_shape_fn((h, w), |(y, x)| {
            let mut best = f64::INFINITY;
            for &(oy, ox) in &ones {
                let dy = y as f64 - oy as f64;
                let dx = x as f64 - ox as f64;
                best = best.min(dx * dx + dy * dy);
            }
            best as f32
        })
    }

    #[test]
    fn edt2_matches_brute_force() {
        let mut seed = 42u64;
        for trial in 0..60 {
            let h = 1 + (rng(&mut seed) % 12) as usize;
            let w = 1 + (rng(&mut seed) % 12) as usize;
            let density = (trial % 5) as u64; // 0 → kosong
            let mask = Array2::from_shape_fn((h, w), |_| u8::from(rng(&mut seed) % 10 < density));
            assert_eq!(edt_squared_2d(&mask), brute_2d(&mask), "trial {trial}");
        }
    }

    #[test]
    fn edt2_um_scales() {
        let mut m = Array2::<u8>::zeros((1, 4));
        m[[0, 0]] = 1;
        let d = edt_2d_um(&m, 2.5);
        assert!((d[[0, 3]] - 7.5).abs() < 1e-5);
    }

    fn brute_3d(mask: &[u8], dims: [usize; 3], sp: [f64; 3]) -> (Vec<f32>, Vec<u32>) {
        let [nx, ny, nz] = dims;
        let ones: Vec<usize> = (0..mask.len()).filter(|&i| mask[i] != 0).collect();
        let mut d = vec![f32::INFINITY; mask.len()];
        let mut a = vec![NO_INDEX; mask.len()];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let i = x + nx * (y + ny * z);
                    let mut best = f64::INFINITY;
                    let mut barg = NO_INDEX;
                    for &o in &ones {
                        let ox = o % nx;
                        let oy = (o / nx) % ny;
                        let oz = o / (nx * ny);
                        let dx = (x as f64 - ox as f64) * sp[0];
                        let dy = (y as f64 - oy as f64) * sp[1];
                        let dz = (z as f64 - oz as f64) * sp[2];
                        // Urutan penjumlahan sama dengan algoritma separabel.
                        let v = (dx * dx + dy * dy) + dz * dz;
                        if v < best {
                            best = v;
                            barg = o as u32;
                        }
                    }
                    d[i] = best as f32;
                    a[i] = barg;
                }
            }
        }
        (d, a)
    }

    #[test]
    fn edt3_matches_brute_force_anisotropic() {
        let mut seed = 7u64;
        let spacings = [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 3.0],
            [0.5, 0.5, 2.0],
            [2.0, 1.0, 4.0],
        ];
        for trial in 0..40 {
            let nx = 1 + (rng(&mut seed) % 7) as usize;
            let ny = 1 + (rng(&mut seed) % 7) as usize;
            let nz = 1 + (rng(&mut seed) % 6) as usize;
            let sp = spacings[trial % spacings.len()];
            let density = (trial % 4) as u64;
            let mask: Vec<u8> = (0..nx * ny * nz)
                .map(|_| u8::from(rng(&mut seed) % 8 < density))
                .collect();
            let (want_d, want_a) = brute_3d(&mask, [nx, ny, nz], sp);
            let got_d = edt3_anisotropic(&mask, [nx, ny, nz], sp).unwrap();
            assert_eq!(got_d, want_d, "trial {trial} dims {nx}x{ny}x{nz} sp {sp:?}");
            let (got_d2, got_a) = edt3_argmin(&mask, [nx, ny, nz], sp).unwrap();
            assert_eq!(got_d2, want_d);
            assert_eq!(got_a, want_a, "argmin trial {trial}");
        }
    }

    #[test]
    fn edt3_large_chunked_matches_brute_force() {
        // Melebihi satu chunk 64×64 agar jalur multi-chunk teruji.
        let mut seed = 99u64;
        let dims = [70, 66, 3];
        let n = dims[0] * dims[1] * dims[2];
        let mask: Vec<u8> = (0..n)
            .map(|_| u8::from(rng(&mut seed) % 400 == 0))
            .collect();
        let sp = [1.0, 1.0, 2.0];
        let (want_d, want_a) = brute_3d(&mask, dims, sp);
        let (got_d, got_a) = edt3_argmin(&mask, dims, sp).unwrap();
        assert_eq!(got_d, want_d);
        assert_eq!(got_a, want_a);
    }

    #[test]
    fn edt3_rejects_bad_input() {
        assert!(matches!(
            edt3_anisotropic(&[0u8; 5], [2, 2, 2], [1.0; 3]),
            Err(ImgError::ShapeMismatch { .. })
        ));
        assert!(matches!(
            edt3_anisotropic(&[0u8; 8], [2, 2, 2], [1.0, 0.0, 1.0]),
            Err(ImgError::InvalidParameter(_))
        ));
        assert!(matches!(
            edt3_anisotropic(&[], [0, 2, 2], [1.0; 3]),
            Err(ImgError::EmptyInput)
        ));
    }

    #[test]
    fn sdf_negative_inside_and_clamped() {
        let mut m = Array2::<u8>::zeros((21, 21));
        for y in 5..16 {
            for x in 5..16 {
                m[[y, x]] = 1;
            }
        }
        let band = 3.0;
        let s = signed_distance_2d(&m, band);
        assert!(s[[10, 10]] < 0.0);
        assert_eq!(s[[10, 10]], -band);
        assert_eq!(s[[0, 0]], band);
        // Tepi: piksel dalam terluar = −0,5, piksel luar pertama = +0,5.
        assert!((s[[10, 5]] + 0.5).abs() < 1e-6);
        assert!((s[[10, 4]] - 0.5).abs() < 1e-6);
        assert!((s[[10, 3]] - 1.5).abs() < 1e-6);
        for v in s.iter() {
            assert!(*v >= -band && *v <= band);
        }
        // Masker penuh / kosong.
        let full = Array2::<u8>::ones((3, 3));
        assert!(signed_distance_2d(&full, 2.0).iter().all(|v| *v == -2.0));
        let empty = Array2::<u8>::zeros((3, 3));
        assert!(signed_distance_2d(&empty, 2.0).iter().all(|v| *v == 2.0));
    }

    #[test]
    fn sdf_um_scales() {
        let mut m = Array2::<u8>::zeros((1, 6));
        m[[0, 0]] = 1;
        let s = signed_distance_2d_um(&m, 4.0, 100.0);
        assert!((s[[0, 2]] - 6.0).abs() < 1e-5); // (2 − 0,5) × 4
        assert!((s[[0, 0]] + 2.0).abs() < 1e-5); // −0,5 × 4
    }
}
