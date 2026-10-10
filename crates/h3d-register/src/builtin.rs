//! Backend bawaan (histo3d.md 8.4.2): R1 kandidat momen → R2 korelasi fase
//! → R3 penghalusan ECC berjenjang (atau Nelder–Mead atas informasi mutual).
//!
//! # Arah transformasi di modul ini
//! - `P_px`: piksel **moving → fixed** pada satu level piramida.
//! - `φ_px = P_px⁻¹`: piksel **fixed → moving**; inilah `t_inv` untuk
//!   `warp_affine_*` (resampling mundur: tiap piksel fixed mengambil nilai
//!   dari moving).
//! - `φ_n = N ∘ φ_px ∘ N⁻¹` dalam koordinat ternormalisasi (asal di pusat
//!   citra fixed, skala = setengah diagonal); parameter ECC hidup di sini.
//! - Keluaran [`RegResult::transform`] = `P_{moving→fixed}` µm → µm:
//!   `px_to_um_fixed ∘ P_px ∘ um_to_px_moving`.

use std::f64::consts::PI;

use h3d_core::{Affine2, Progress};
use h3d_imgproc::{
    dice, gaussian_blur_f32, hann_window_2d, moments, mutual_information, ncc_masked,
    phase_correlation, warp_affine_f32, warp_affine_u8, ImgError, Interp, Moments,
};
use ndarray::Array2;
use rayon::prelude::*;

use crate::error::RegError;
use crate::types::{RegImage, RegMetric, RegModel, RegOptions, RegResult, Registrar};

/// Rasio nilai eigen di bawahnya jaringan dianggap bundar (sumbu utama tidak
/// dipercaya) dan R1 memakai sapuan sudut.
pub const ROUND_EIGEN_RATIO: f64 = 1.2;
/// Langkah sapuan sudut untuk jaringan bundar (derajat).
pub const SWEEP_STEP_DEG: f64 = 10.0;
/// Jumlah kandidat sapuan yang diteruskan ke korelasi fase (R2).
const SWEEP_KEEP: usize = 8;
/// Sigma pemulusan Gauss (piksel) sebelum ECC/MI pada tiap level.
const SMOOTH_SIGMA_PX: f64 = 0.8;
/// Jumlah wadah histogram untuk informasi mutual.
const MI_BINS: usize = 32;
/// Jumlah piksel bermasker minimum per parameter agar ECC dijalankan.
const MIN_PIXELS_PER_PARAM: usize = 8;

/// Registrar bawaan (tanpa dependensi luar).
#[derive(Debug, Default, Clone, Copy)]
pub struct BuiltinRegistrar;

/// Kandidat awal dengan skornya (hasil R1+R2) pada level kasar.
#[derive(Clone, Copy, Debug)]
pub struct CoarseCandidate {
    /// `P_{moving→fixed}` µm → µm.
    pub p_um: Affine2,
    /// NCC bermasker (irisan masker) × Dice masker setelah geseran.
    pub score: f32,
}

fn map_img(e: ImgError, which: &str) -> RegError {
    match e {
        ImgError::EmptyInput => RegError::EmptyMask {
            which: which.to_string(),
        },
        other => other.into(),
    }
}

/// Validasi citra registrasi: bentuk cocok dan masker tidak kosong.
pub(crate) fn validate(img: &RegImage, which: &str) -> Result<()> {
    let (gh, gw) = img.gray.dim();
    let (mh, mw) = img.mask.dim();
    if (gh, gw) != (mh, mw) {
        return Err(RegError::ShapeMismatch {
            expected: [gh, gw],
            got: [mh, mw],
        });
    }
    if gh == 0 || gw == 0 || !img.has_tissue() {
        return Err(RegError::EmptyMask {
            which: which.to_string(),
        });
    }
    if !(img.mpp.is_finite() && img.mpp > 0.0) {
        return Err(RegError::InvalidInput(format!(
            "mpp {which} tidak sah: {}",
            img.mpp
        )));
    }
    Ok(())
}

/// Level piramida efektif: terurut kasar → halus, tanpa level lebih halus dari
/// resolusi sumber; minimal satu level (resolusi sumber).
pub(crate) fn effective_levels(opts: &RegOptions, source_mpp: f64) -> Vec<f64> {
    let mut lv: Vec<f64> = opts
        .levels_um
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0 && *v >= source_mpp * (1.0 - 1e-9))
        .collect();
    lv.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    lv.dedup_by(|a, b| (*a - *b).abs() <= 1e-9 * b.abs());
    if lv.is_empty() {
        lv.push(source_mpp);
    }
    lv
}

/// `P_px` (moving px → fixed px) dari `P_um` pada level `f`/`m`.
pub(crate) fn p_um_to_px(f: &RegImage, m: &RegImage, p_um: &Affine2) -> Affine2 {
    f.um_to_px_affine()
        .compose(p_um)
        .compose(&m.px_to_um_affine())
}

/// `P_um` (moving µm → fixed µm) dari `P_px` pada level `f`/`m`.
pub(crate) fn p_px_to_um(f: &RegImage, m: &RegImage, p_px: &Affine2) -> Affine2 {
    f.px_to_um_affine()
        .compose(p_px)
        .compose(&m.um_to_px_affine())
}

fn inverse_or_err(p: &Affine2) -> Result<Affine2> {
    p.inverse().ok_or(RegError::SingularTransform {
        det: p.determinant(),
    })
}

/// Warp citra abu-abu dan masker moving ke bingkai fixed dengan `P_px`
/// (moving px → fixed px). Mengembalikan `(abu-abu, masker)` berbentuk fixed.
fn warp_moving(f: &RegImage, m: &RegImage, p_px: &Affine2) -> Result<(Array2<f32>, Array2<u8>)> {
    let t_inv = inverse_or_err(p_px)?;
    let size = f.size_px();
    let g = warp_affine_f32(&m.gray, &t_inv, size, Interp::Bilinear, 0.0)?;
    let k = warp_affine_u8(&m.mask, &t_inv, size, Interp::Nearest, 0)?;
    Ok((g, k))
}

/// Skor dari moving yang sudah di-warp (`g`, `k` berbentuk fixed):
/// `(NCC pada irisan masker, Dice masker)`.
fn score_warped(f: &RegImage, g: &Array2<f32>, k: &Array2<u8>) -> Result<(f32, f32)> {
    let inter = Array2::from_shape_fn(f.mask.dim(), |ij| u8::from(f.mask[ij] != 0 && k[ij] != 0));
    let ncc = ncc_masked(&f.gray, g, Some(&inter))?;
    let d = dice(&f.mask, k)?;
    Ok((ncc, d))
}

/// Evaluasi `P_px` (moving px → fixed px): NCC pada irisan masker (keduanya
/// 1) dan Dice masker.
fn evaluate(f: &RegImage, m: &RegImage, p_px: &Affine2) -> Result<(f32, f32)> {
    let (g, k) = warp_moving(f, m, p_px)?;
    score_warped(f, &g, &k)
}

/// Skor kandidat = NCC pada irisan masker × Dice masker. NCC murni pada
/// irisan mengabaikan bentuk (tumpang-tindih kecil bisa bernilai tinggi);
/// faktor Dice menghukum ketidaksejajaran bentuk, sedangkan tekstur tetap
/// menentukan untuk bentuk simetris (cakram, cermin). Dipakai memilih
/// kandidat R2 dan sebagai kemiripan U1.
#[inline]
fn candidate_score(ncc: f32, dice: f32) -> f32 {
    ncc * dice
}

/// Kandidat rigid+cermin dari momen: `T(c_f) ∘ R(θ) ∘ [M] ∘ T(−c_m)`,
/// piksel moving → piksel fixed.
fn candidate_px(theta: f64, mirror: bool, c_f: [f64; 2], c_m: [f64; 2]) -> Affine2 {
    let mut t = Affine2::translation(c_f[0], c_f[1]).compose(&Affine2::rotation(theta));
    if mirror {
        t = t.compose(&Affine2::mirror_x());
    }
    t.compose(&Affine2::translation(-c_m[0], -c_m[1]))
}

/// R1: daftar kandidat `(P_px, mirror)`.
fn moment_candidates(mf: &Moments, mm: &Moments, allow_flip: bool) -> (Vec<Affine2>, bool) {
    let round = mf.eigen_ratio < ROUND_EIGEN_RATIO || mm.eigen_ratio < ROUND_EIGEN_RATIO;
    let mirrors: &[bool] = if allow_flip { &[false, true] } else { &[false] };
    let mut out = Vec::new();
    if round {
        let n = (360.0 / SWEEP_STEP_DEG).round() as usize;
        for &mirror in mirrors {
            for k in 0..n {
                let theta = k as f64 * SWEEP_STEP_DEG.to_radians();
                out.push(candidate_px(theta, mirror, mf.centroid, mm.centroid));
            }
        }
    } else {
        for &mirror in mirrors {
            // Cermin x membalik sudut sumbu utama: θ → −θ (mod π).
            let theta_m = if mirror { -mm.theta } else { mm.theta };
            let base = mf.theta - theta_m;
            out.push(candidate_px(base, mirror, mf.centroid, mm.centroid));
            out.push(candidate_px(base + PI, mirror, mf.centroid, mm.centroid));
        }
    }
    (out, round)
}

/// R2 untuk satu kandidat: korelasi fase berjendela Hann memberi geseran
/// `(dx, dy)` sehingga moving ter-warp yang digeser `(dx, dy)` berimpit dengan
/// fixed → `P' = T(dx, dy) ∘ P`. Dipilih yang lebih baik antara `P` dan `P'`.
fn refine_translation(
    f: &RegImage,
    f_win: &Array2<f32>,
    hann: &Array2<f32>,
    m: &RegImage,
    p_px: &Affine2,
) -> Result<(Affine2, f32)> {
    let (mw, mk) = warp_moving(f, m, p_px)?;
    let (n0, d0) = score_warped(f, &mw, &mk)?;
    let s0 = candidate_score(n0, d0);
    let mw_win = &mw * hann;
    let (dx, dy, _) = phase_correlation(f_win, &mw_win)?;
    if dx.abs() < 1e-9 && dy.abs() < 1e-9 {
        return Ok((*p_px, s0));
    }
    let shifted = Affine2::translation(dx, dy).compose(p_px);
    let (n1, d1) = evaluate(f, m, &shifted)?;
    let s1 = candidate_score(n1, d1);
    if s1 >= s0 {
        Ok((shifted, s1))
    } else {
        Ok((*p_px, s0))
    }
}

type Result<T> = std::result::Result<T, RegError>;

/// R1 + R2 pada level kasar (`f`, `m` sudah pada resolusi yang sama).
/// `init` = `P_{moving→fixed}` µm → µm; bila ada, menjadi satu-satunya
/// kandidat. Mengembalikan kandidat terbaik (skor NCC bermasker × Dice).
pub fn coarse_align(
    f: &RegImage,
    m: &RegImage,
    init: Option<Affine2>,
    allow_flip: bool,
) -> Result<CoarseCandidate> {
    validate(f, "fixed")?;
    validate(m, "moving")?;
    let (cands, round) = match init {
        Some(p_um) => (vec![p_um_to_px(f, m, &p_um)], false),
        None => {
            let mf = moments(&f.mask).map_err(|e| map_img(e, "fixed"))?;
            let mm = moments(&m.mask).map_err(|e| map_img(e, "moving"))?;
            moment_candidates(&mf, &mm, allow_flip)
        }
    };
    let [w, h] = f.size_px();
    let hann = hann_window_2d(w, h);
    let f_win = &f.gray * &hann;
    let score_all = |cs: &[Affine2]| -> Result<Vec<(Affine2, f32)>> {
        cs.par_iter()
            .map(|c| refine_translation(f, &f_win, &hann, m, c))
            .collect()
    };
    let pick = |scored: &[(Affine2, f32)]| -> (Affine2, f32) {
        let mut best = scored[0];
        for s in scored.iter().skip(1) {
            if s.1 > best.1 {
                best = *s;
            }
        }
        best
    };
    // Sapuan (jaringan bundar): pusat massa sudah berimpit, jadi kandidat
    // disaring dulu dengan skor tanpa korelasi fase; hanya yang terbaik yang
    // menjalani R2 penuh. Urutan stabil (seri → indeks kandidat terkecil).
    let cands = if round && cands.len() > SWEEP_KEEP {
        let pre: Vec<f32> = cands
            .par_iter()
            .map(|c| evaluate(f, m, c).map(|(n, d)| candidate_score(n, d)))
            .collect::<Result<Vec<f32>>>()?;
        let mut idx: Vec<usize> = (0..cands.len()).collect();
        idx.sort_by(|&a, &b| {
            pre[b]
                .partial_cmp(&pre[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        idx.truncate(SWEEP_KEEP);
        idx.sort_unstable();
        idx.into_iter().map(|i| cands[i]).collect()
    } else {
        cands
    };
    let scored = score_all(&cands)?;
    let mut best = pick(&scored);
    if round && init.is_none() {
        // Penghalusan sudut lokal di sekitar kandidat sapuan terbaik.
        let mf = moments(&f.mask).map_err(|e| map_img(e, "fixed"))?;
        let mm = moments(&m.mask).map_err(|e| map_img(e, "moving"))?;
        let mirror = best.0.determinant() < 0.0;
        let lin = best.0.linear_part();
        let theta0 = if mirror {
            (-lin[2]).atan2(-lin[0])
        } else {
            lin[2].atan2(lin[0])
        };
        let fine: Vec<Affine2> = [-5.0, -2.5, 2.5, 5.0]
            .iter()
            .map(|d: &f64| candidate_px(theta0 + d.to_radians(), mirror, mf.centroid, mm.centroid))
            .collect();
        let scored2 = score_all(&fine)?;
        let b2 = pick(&scored2);
        if b2.1 > best.1 {
            best = b2;
        }
    }
    Ok(CoarseCandidate {
        p_um: p_px_to_um(f, m, &best.0),
        score: best.1,
    })
}

// ---------------------------------------------------------------------------
// R3: ECC
// ---------------------------------------------------------------------------

/// Parameter warp `L(p)` dalam koordinat ternormalisasi (fixed_n → moving_n,
/// sebelum/ sesudah cermin tetap `M`: `φ_n = L(p) ∘ M`).
#[derive(Clone, Copy, Debug)]
struct Params {
    model: RegModel,
    p: [f64; 6],
}

impl Params {
    fn n(&self) -> usize {
        match self.model {
            RegModel::Rigid => 3,
            RegModel::Similarity => 4,
            RegModel::Affine => 6,
        }
    }

    /// Proyeksikan affine `l` (ternormalisasi) ke parameter model.
    fn from_linear(model: RegModel, l: &Affine2) -> Self {
        let [a, b, tx, c, d, ty] = l.m;
        let p = match model {
            RegModel::Rigid => [(c - b).atan2(a + d), tx, ty, 0.0, 0.0, 0.0],
            RegModel::Similarity => [(a + d) / 2.0 - 1.0, (c - b) / 2.0, tx, ty, 0.0, 0.0],
            RegModel::Affine => [a, b, tx, c, d, ty],
        };
        Self { model, p }
    }

    fn to_linear(self) -> Affine2 {
        let p = self.p;
        match self.model {
            RegModel::Rigid => {
                let (s, c) = p[0].sin_cos();
                Affine2::new([c, -s, p[1], s, c, p[2]])
            }
            RegModel::Similarity => Affine2::new([1.0 + p[0], -p[1], p[2], p[1], 1.0 + p[0], p[3]]),
            RegModel::Affine => Affine2::new(p),
        }
    }

    /// Jacobian `∂(x', y')/∂p` di titik ternormalisasi (sudah dicerminkan) `(x, y)`.
    #[inline]
    fn jacobian(&self, x: f64, y: f64, jx: &mut [f64; 6], jy: &mut [f64; 6]) {
        *jx = [0.0; 6];
        *jy = [0.0; 6];
        match self.model {
            RegModel::Rigid => {
                let (s, c) = self.p[0].sin_cos();
                jx[0] = -s * x - c * y;
                jy[0] = c * x - s * y;
                jx[1] = 1.0;
                jy[2] = 1.0;
            }
            RegModel::Similarity => {
                jx[0] = x;
                jy[0] = y;
                jx[1] = -y;
                jy[1] = x;
                jx[2] = 1.0;
                jy[3] = 1.0;
            }
            RegModel::Affine => {
                jx[0] = x;
                jx[1] = y;
                jx[2] = 1.0;
                jy[3] = x;
                jy[4] = y;
                jy[5] = 1.0;
            }
        }
    }

    fn add(&mut self, d: &[f64; 6]) {
        for (p, dp) in self.p.iter_mut().zip(d.iter()) {
            *p += dp;
        }
    }
}

/// Eliminasi Gauss berpivot parsial untuk sistem `n×n` (`n ≤ 6`).
fn solve_small(a: &[[f64; 6]; 6], b: &[f64; 6], n: usize) -> Option<[f64; 6]> {
    let mut m = *a;
    let mut r = *b;
    for col in 0..n {
        let mut piv = col;
        for row in (col + 1)..n {
            if m[row][col].abs() > m[piv][col].abs() {
                piv = row;
            }
        }
        if m[piv][col].abs() < 1e-300 {
            return None;
        }
        m.swap(col, piv);
        r.swap(col, piv);
        for row in (col + 1)..n {
            let f = m[row][col] / m[col][col];
            if f == 0.0 {
                continue;
            }
            let pivot_row = m[col];
            for (v, pv) in m[row].iter_mut().zip(pivot_row.iter()).take(n).skip(col) {
                *v -= f * pv;
            }
            r[row] -= f * r[col];
        }
    }
    let mut x = [0.0; 6];
    for i in (0..n).rev() {
        let mut s = r[i];
        for k in (i + 1)..n {
            s -= m[i][k] * x[k];
        }
        if m[i][i].abs() < 1e-300 {
            return None;
        }
        x[i] = s / m[i][i];
    }
    if x.iter().all(|v| v.is_finite()) {
        Some(x)
    } else {
        None
    }
}

/// Gradien pusat (per piksel), satu sisi di tepi.
fn gradient(img: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
    let (h, w) = img.dim();
    let gx = Array2::from_shape_fn((h, w), |(y, x)| {
        let x0 = x.saturating_sub(1);
        let x1 = (x + 1).min(w - 1);
        (img[[y, x1]] - img[[y, x0]]) / (x1 - x0).max(1) as f32
    });
    let gy = Array2::from_shape_fn((h, w), |(y, x)| {
        let y0 = y.saturating_sub(1);
        let y1 = (y + 1).min(h - 1);
        (img[[y1, x]] - img[[y0, x]]) / (y1 - y0).max(1) as f32
    });
    (gx, gy)
}

/// Kerangka normalisasi satu level: `N` (px fixed → ternormalisasi) dan
/// balikannya, pusat `c`, skala `s` (setengah diagonal).
struct Frame {
    c: [f64; 2],
    s: f64,
    n: Affine2,
    n_inv: Affine2,
}

impl Frame {
    fn of(f: &RegImage) -> Self {
        let c = f.center_px();
        let s = (c[0] * c[0] + c[1] * c[1]).sqrt().max(1.0);
        let n = Affine2::scale(1.0 / s, 1.0 / s).compose(&Affine2::translation(-c[0], -c[1]));
        let n_inv = Affine2::translation(c[0], c[1]).compose(&Affine2::scale(s, s));
        Self { c, s, n, n_inv }
    }

    /// `φ_n` (fixed_n → moving_n) dari `P_px` (moving px → fixed px).
    fn phi_n_from_p_px(&self, p_px: &Affine2) -> Result<Affine2> {
        let phi_px = inverse_or_err(p_px)?;
        Ok(self.n.compose(&phi_px).compose(&self.n_inv))
    }

    /// `P_px` (moving px → fixed px) dari `φ_n`.
    fn p_px_from_phi_n(&self, phi_n: &Affine2) -> Result<Affine2> {
        let phi_px = self.n_inv.compose(phi_n).compose(&self.n);
        inverse_or_err(&phi_px)
    }
}

/// Pisahkan `φ_n = L ∘ M`: `M` cermin tetap (untuk Rigid/Similarity dengan
/// determinan negatif), `L` diparameterkan.
fn split_mirror(model: RegModel, phi_n: &Affine2) -> (Params, Affine2) {
    let mirror = if model != RegModel::Affine && phi_n.determinant() < 0.0 {
        Affine2::mirror_x()
    } else {
        Affine2::IDENTITY
    };
    // M involutif: L = φ_n ∘ M⁻¹ = φ_n ∘ M.
    let l = phi_n.compose(&mirror);
    (Params::from_linear(model, &l), mirror)
}

/// Satu level ECC *forward additive* (Evangelidis & Psarakis 2008).
/// Masukan/keluaran `P_px` (moving px → fixed px) pada level ini.
/// Mengembalikan `(P_px terbaik, iterasi)`; bila korelasi tidak membaik,
/// `P_px` awal dikembalikan.
fn ecc_level(
    f: &RegImage,
    m: &RegImage,
    p_px0: &Affine2,
    opts: &RegOptions,
) -> Result<(Affine2, u32)> {
    let frame = Frame::of(f);
    let phi0 = frame.phi_n_from_p_px(p_px0)?;
    let (mut params, mirror) = split_mirror(opts.model, &phi0);
    let np = params.n();

    let fg = gaussian_blur_f32(&f.gray, SMOOTH_SIGMA_PX);
    let mg = gaussian_blur_f32(&m.gray, SMOOTH_SIGMA_PX);
    let (gx, gy) = gradient(&mg);
    let (h, w) = fg.dim();
    let size = [w, h];
    let fvals: Vec<f32> = fg.iter().copied().collect();
    let fmask: Vec<u8> = f.mask.iter().copied().collect();

    let mut best_rho = f64::NEG_INFINITY;
    let mut best_params = params;
    let mut iterations = 0u32;
    let mut jx = [0.0f64; 6];
    let mut jy = [0.0f64; 6];

    for it in 0..=opts.max_iterations {
        let phi_n = params.to_linear().compose(&mirror);
        let phi_px = frame.n_inv.compose(&phi_n).compose(&frame.n);
        if !phi_px.is_finite() || phi_px.determinant().abs() < 1e-9 {
            break;
        }
        let iw = warp_affine_f32(&mg, &phi_px, size, Interp::Bilinear, 0.0)?;
        let gxw = warp_affine_f32(&gx, &phi_px, size, Interp::Bilinear, 0.0)?;
        let gyw = warp_affine_f32(&gy, &phi_px, size, Interp::Bilinear, 0.0)?;
        let mkw = warp_affine_u8(&m.mask, &phi_px, size, Interp::Nearest, 0)?;

        // Piksel yang dihitung: masker fixed = 1 dan masker moving ter-warp = 1.
        let sel: Vec<usize> = (0..w * h)
            .filter(|&i| fmask[i] != 0 && mkw.as_slice().map(|s| s[i] != 0).unwrap_or(false))
            .collect();
        if sel.len() < np * MIN_PIXELS_PER_PARAM {
            break;
        }
        let iw_s = iw.as_slice().unwrap_or(&[]);
        let gx_s = gxw.as_slice().unwrap_or(&[]);
        let gy_s = gyw.as_slice().unwrap_or(&[]);
        let n = sel.len() as f64;
        let mean_r = sel.iter().map(|&i| fvals[i] as f64).sum::<f64>() / n;
        let mean_w = sel.iter().map(|&i| iw_s[i] as f64).sum::<f64>() / n;

        let mut hess = [[0.0f64; 6]; 6];
        let mut g_r = [0.0f64; 6];
        let mut g_w = [0.0f64; 6];
        let (mut rr, mut ww, mut rw) = (0.0f64, 0.0f64, 0.0f64);
        let mut row = [0.0f64; 6];
        for &i in &sel {
            let x = i % w;
            let y = i / w;
            let xn = (x as f64 + 0.5 - frame.c[0]) / frame.s;
            let yn = (y as f64 + 0.5 - frame.c[1]) / frame.s;
            let [xm, ym] = mirror.apply([xn, yn]);
            params.jacobian(xm, ym, &mut jx, &mut jy);
            let gxv = gx_s[i] as f64 * frame.s;
            let gyv = gy_s[i] as f64 * frame.s;
            for k in 0..np {
                row[k] = gxv * jx[k] + gyv * jy[k];
            }
            let ir = fvals[i] as f64 - mean_r;
            let iwv = iw_s[i] as f64 - mean_w;
            rr += ir * ir;
            ww += iwv * iwv;
            rw += ir * iwv;
            for a in 0..np {
                g_r[a] += row[a] * ir;
                g_w[a] += row[a] * iwv;
                for b in 0..np {
                    hess[a][b] += row[a] * row[b];
                }
            }
        }
        if rr <= 1e-20 || ww <= 1e-20 {
            break;
        }
        let rho = rw / (rr * ww).sqrt();
        if rho > best_rho {
            best_rho = rho;
            best_params = params;
        }
        if it == opts.max_iterations {
            break;
        }
        let Some(h_gw) = solve_small(&hess, &g_w, np) else {
            break;
        };
        let Some(h_gr) = solve_small(&hess, &g_r, np) else {
            break;
        };
        let gw_h_gw: f64 = (0..np).map(|k| g_w[k] * h_gw[k]).sum();
        let gr_h_gw: f64 = (0..np).map(|k| g_r[k] * h_gw[k]).sum();
        let lambda_n = ww - gw_h_gw;
        let lambda_d = rw - gr_h_gw;
        if lambda_d <= 0.0 {
            // Korelasi tidak bisa dinaikkan dengan langkah linear → berhenti.
            break;
        }
        let lambda = lambda_n / lambda_d;
        // Δp = H⁻¹ Gᵀ(λ·ir − iw) = λ·H⁻¹Gᵀir − H⁻¹Gᵀiw.
        let mut dp = [0.0f64; 6];
        let mut norm2 = 0.0;
        for k in 0..np {
            dp[k] = lambda * h_gr[k] - h_gw[k];
            norm2 += dp[k] * dp[k];
        }
        if !norm2.is_finite() {
            break;
        }
        params.add(&dp);
        iterations += 1;
        if norm2.sqrt() < opts.convergence_eps {
            // Evaluasi parameter akhir pada putaran berikutnya lalu berhenti.
            let phi_n = params.to_linear().compose(&mirror);
            if let Ok(p_px) = frame.p_px_from_phi_n(&phi_n) {
                if let Ok(rho_f) = masked_rho(&fg, &f.mask, &mg, &m.mask, &p_px) {
                    if rho_f > best_rho {
                        best_rho = rho_f;
                        best_params = params;
                    }
                }
            }
            break;
        }
    }
    if !best_rho.is_finite() {
        return Ok((*p_px0, iterations));
    }
    let phi_n = best_params.to_linear().compose(&mirror);
    let p_px = frame.p_px_from_phi_n(&phi_n)?;
    Ok((p_px, iterations))
}

/// Koefisien korelasi pada irisan masker untuk `P_px` (moving px → fixed px).
fn masked_rho(
    fg: &Array2<f32>,
    fmask: &Array2<u8>,
    mg: &Array2<f32>,
    mmask: &Array2<u8>,
    p_px: &Affine2,
) -> Result<f64> {
    let t_inv = inverse_or_err(p_px)?;
    let (h, w) = fg.dim();
    let g = warp_affine_f32(mg, &t_inv, [w, h], Interp::Bilinear, 0.0)?;
    let k = warp_affine_u8(mmask, &t_inv, [w, h], Interp::Nearest, 0)?;
    let inter = Array2::from_shape_fn((h, w), |ij| u8::from(fmask[ij] != 0 && k[ij] != 0));
    Ok(ncc_masked(fg, &g, Some(&inter))? as f64)
}

// ---------------------------------------------------------------------------
// R3 alternatif: Nelder–Mead atas informasi mutual
// ---------------------------------------------------------------------------

/// Nilai −MI untuk parameter (dalam kerangka ternormalisasi).
fn neg_mi(
    fg: &Array2<f32>,
    fmask: &Array2<u8>,
    mg: &Array2<f32>,
    mmask: &Array2<u8>,
    frame: &Frame,
    params: &Params,
    mirror: &Affine2,
) -> f64 {
    let phi_n = params.to_linear().compose(mirror);
    let phi_px = frame.n_inv.compose(&phi_n).compose(&frame.n);
    if !phi_px.is_finite() || phi_px.determinant().abs() < 1e-9 {
        return f64::INFINITY;
    }
    let (h, w) = fg.dim();
    let g = match warp_affine_f32(mg, &phi_px, [w, h], Interp::Bilinear, 0.0) {
        Ok(v) => v,
        Err(_) => return f64::INFINITY,
    };
    let k = match warp_affine_u8(mmask, &phi_px, [w, h], Interp::Nearest, 0) {
        Ok(v) => v,
        Err(_) => return f64::INFINITY,
    };
    let inter = Array2::from_shape_fn((h, w), |ij| u8::from(fmask[ij] != 0 && k[ij] != 0));
    let n = inter.iter().filter(|v| **v != 0).count();
    if n < MIN_PIXELS_PER_PARAM * params.n() {
        return f64::INFINITY;
    }
    match mutual_information(fg, &g, Some(&inter), MI_BINS) {
        Ok(mi) => -(mi as f64),
        Err(_) => f64::INFINITY,
    }
}

/// Satu level Nelder–Mead memaksimalkan informasi mutual. Masukan/keluaran
/// `P_px` (moving px → fixed px).
fn mi_level(
    f: &RegImage,
    m: &RegImage,
    p_px0: &Affine2,
    opts: &RegOptions,
) -> Result<(Affine2, u32)> {
    let frame = Frame::of(f);
    let phi0 = frame.phi_n_from_p_px(p_px0)?;
    let (p0, mirror) = split_mirror(opts.model, &phi0);
    let np = p0.n();
    let fg = gaussian_blur_f32(&f.gray, SMOOTH_SIGMA_PX);
    let mg = gaussian_blur_f32(&m.gray, SMOOTH_SIGMA_PX);
    let obj = |p: &Params| neg_mi(&fg, &f.mask, &mg, &m.mask, &frame, p, &mirror);

    // Simpleks awal: tiap parameter digeser 0,02 (ternormalisasi / radian).
    let mut simplex: Vec<(Params, f64)> = Vec::with_capacity(np + 1);
    simplex.push((p0, obj(&p0)));
    for k in 0..np {
        let mut q = p0;
        q.p[k] += 0.02;
        simplex.push((q, obj(&q)));
    }
    let mut iterations = 0u32;
    for _ in 0..opts.max_iterations {
        simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        // Ukuran simpleks.
        let spread = simplex
            .iter()
            .map(|(q, _)| {
                (0..np)
                    .map(|k| (q.p[k] - simplex[0].0.p[k]).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .fold(0.0f64, f64::max);
        if spread < opts.convergence_eps {
            break;
        }
        iterations += 1;
        let worst = simplex[np];
        let mut centroid = [0.0f64; 6];
        for (q, _) in simplex.iter().take(np) {
            for (c, v) in centroid.iter_mut().zip(q.p.iter()).take(np) {
                *c += v / np as f64;
            }
        }
        let point = |coef: f64| -> Params {
            let mut q = p0;
            for (k, c) in centroid.iter().enumerate().take(np) {
                q.p[k] = c + coef * (worst.0.p[k] - c);
            }
            q
        };
        let refl = point(-1.0);
        let fr = obj(&refl);
        if fr < simplex[0].1 {
            let exp = point(-2.0);
            let fe = obj(&exp);
            simplex[np] = if fe < fr { (exp, fe) } else { (refl, fr) };
        } else if fr < simplex[np - 1].1 {
            simplex[np] = (refl, fr);
        } else {
            let con = if fr < worst.1 {
                point(-0.5)
            } else {
                point(0.5)
            };
            let fc = obj(&con);
            if fc < fr.min(worst.1) {
                simplex[np] = (con, fc);
            } else {
                // Susut ke titik terbaik.
                let best = simplex[0].0;
                for (q, v) in simplex.iter_mut().skip(1) {
                    for k in 0..np {
                        q.p[k] = best.p[k] + 0.5 * (q.p[k] - best.p[k]);
                    }
                    *v = obj(q);
                }
            }
        }
    }
    simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let (best, fb) = simplex[0];
    if !fb.is_finite() || fb > obj(&p0) {
        return Ok((*p_px0, iterations));
    }
    let phi_n = best.to_linear().compose(&mirror);
    Ok((frame.p_px_from_phi_n(&phi_n)?, iterations))
}

impl Registrar for BuiltinRegistrar {
    fn name(&self) -> &'static str {
        "builtin-ecc"
    }

    fn register(
        &self,
        fixed: &RegImage,
        moving: &RegImage,
        init: Option<Affine2>,
        opts: &RegOptions,
        progress: &dyn Progress,
    ) -> Result<RegResult> {
        validate(fixed, "fixed")?;
        validate(moving, "moving")?;
        if let Some(p) = &init {
            if !p.is_finite() || p.determinant().abs() < h3d_core::affine::SINGULAR_EPS {
                return Err(RegError::SingularTransform {
                    det: p.determinant(),
                });
            }
        }
        let levels = effective_levels(opts, fixed.mpp.max(moving.mpp));
        let n_lv = levels.len() as f32;
        progress.set(0.0, "kandidat awal (momen + korelasi fase)");
        let f0 = fixed.downsample_to(levels[0]);
        let m0 = moving.downsample_to(levels[0]);
        let coarse = coarse_align(&f0, &m0, init, opts.allow_flip)?;
        if progress.is_cancelled() {
            return Err(RegError::Cancelled);
        }

        let mut p_um = coarse.p_um;
        let mut iterations = 0u32;
        let mut f_lv = f0;
        let mut m_lv = m0;
        for (li, &lvl) in levels.iter().enumerate() {
            if progress.is_cancelled() {
                return Err(RegError::Cancelled);
            }
            progress.set(
                (li as f32 + 0.5) / (n_lv + 0.5),
                &format!("penghalusan {lvl:.0} µm"),
            );
            if li > 0 {
                f_lv = fixed.downsample_to(lvl);
                m_lv = moving.downsample_to(lvl);
            }
            let p_px = p_um_to_px(&f_lv, &m_lv, &p_um);
            let (p_px_new, it) = match opts.metric {
                RegMetric::Ncc => ecc_level(&f_lv, &m_lv, &p_px, opts)?,
                RegMetric::MutualInformation => mi_level(&f_lv, &m_lv, &p_px, opts)?,
            };
            iterations += it;
            p_um = p_px_to_um(&f_lv, &m_lv, &p_px_new);
        }
        if progress.is_cancelled() {
            return Err(RegError::Cancelled);
        }

        // Pengaman: bila penghalusan tidak lebih baik dari hasil R2 pada level
        // terhalus, hasil R2 dikembalikan.
        let p_px_fine = p_um_to_px(&f_lv, &m_lv, &p_um);
        let (mut ncc, mut mask_dice) = evaluate(&f_lv, &m_lv, &p_px_fine)?;
        let p_px_coarse = p_um_to_px(&f_lv, &m_lv, &coarse.p_um);
        let (ncc_c, dice_c) = evaluate(&f_lv, &m_lv, &p_px_coarse)?;
        if ncc_c > ncc {
            log::debug!("ECC tidak membaik (ncc {ncc:.3} < {ncc_c:.3}); hasil R2 dipakai");
            p_um = coarse.p_um;
            ncc = ncc_c;
            mask_dice = dice_c;
        }
        progress.set(1.0, "selesai");
        Ok(RegResult {
            transform: p_um,
            ncc,
            mask_dice,
            area_ratio: p_um.determinant().abs() as f32,
            iterations,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_roundtrip_each_model() {
        for model in [RegModel::Rigid, RegModel::Similarity, RegModel::Affine] {
            let l = Affine2::rotation(0.3).compose(&Affine2::translation(0.1, -0.2));
            let p = Params::from_linear(model, &l);
            assert!(p.to_linear().approx_eq(&l, 1e-12), "{model:?}");
        }
    }

    #[test]
    fn solve_small_identity() {
        let mut a = [[0.0; 6]; 6];
        for (i, row) in a.iter_mut().enumerate() {
            row[i] = 2.0;
        }
        let b = [2.0, 4.0, 6.0, 0.0, 0.0, 0.0];
        let x = solve_small(&a, &b, 3).unwrap();
        assert_eq!(&x[..3], &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn effective_levels_filters_and_sorts() {
        let opts = RegOptions {
            levels_um: vec![16.0, 64.0, 32.0, -1.0],
            ..Default::default()
        };
        assert_eq!(effective_levels(&opts, 8.0), vec![64.0, 32.0, 16.0]);
        assert_eq!(effective_levels(&opts, 40.0), vec![64.0]);
        assert_eq!(effective_levels(&opts, 100.0), vec![100.0]);
    }
}
