//! Interpolasi bentuk antar irisan (histo3d.md 8.5.5) di ruang W memakai
//! SDF bertanda: `sdf_k = (1 − t)·sdf_a + t·sdf_b`, `masker_k = sdf_k < 0`.
//! Hasil dipetakan balik ke ruang U irisan `k` sebagai [`Proposal`] dengan
//! sumber `Interpolated { from: [a, b] }`.

use h3d_core::{AnnotationSource, LabelId, SlideId, SlideTransform};
use h3d_imgproc::edt::signed_distance_2d_um;
use ndarray::Array2;

use crate::error::SegError;
use crate::label_store::LabelStore;
use crate::proposal::Proposal;

/// Batas jumlah piksel kisi W agar masukan absurd tidak menghabiskan memori.
const MAX_GRID_PX: f64 = 400_000_000.0;

/// Blend linear dua SDF (bentuk sama) pada ketinggian `z_k`:
/// `t = (z_k − z_a)/(z_b − z_a)`. Galat bila bentuk berbeda atau `z_a == z_b`.
pub fn interpolate_between(
    sdf_a: &Array2<f32>,
    z_a: f64,
    sdf_b: &Array2<f32>,
    z_b: f64,
    z_k: f64,
) -> Result<Array2<f32>, SegError> {
    if sdf_a.dim() != sdf_b.dim() {
        return Err(SegError::InvalidParameter(format!(
            "bentuk SDF berbeda: {:?} vs {:?}",
            sdf_a.dim(),
            sdf_b.dim()
        )));
    }
    let dz = z_b - z_a;
    if !(dz.is_finite() && dz != 0.0 && z_k.is_finite()) {
        return Err(SegError::InvalidParameter(format!(
            "z tidak sah: z_a={z_a}, z_b={z_b}, z_k={z_k}"
        )));
    }
    let t = ((z_k - z_a) / dz) as f32;
    let mut out = sdf_a.clone();
    for (o, &b) in out.iter_mut().zip(sdf_b.iter()) {
        *o = (1.0 - t) * *o + t * b;
    }
    Ok(out)
}

/// Kotak pembatas µm `[x0, y0, x1, y1]` yang ditransformasi `T` (U → W),
/// memakai keempat sudut.
fn transform_bbox(bb: [f64; 4], t: &SlideTransform) -> [f64; 4] {
    let corners = [
        [bb[0], bb[1]],
        [bb[2], bb[1]],
        [bb[0], bb[3]],
        [bb[2], bb[3]],
    ];
    let mut out = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
    for c in corners {
        let w = t.apply(c);
        out[0] = out[0].min(w[0]);
        out[1] = out[1].min(w[1]);
        out[2] = out[2].max(w[0]);
        out[3] = out[3].max(w[1]);
    }
    out
}

fn union_bbox(a: Option<[f64; 4]>, b: Option<[f64; 4]>) -> Option<[f64; 4]> {
    match (a, b) {
        (None, None) => None,
        (Some(x), None) | (None, Some(x)) => Some(x),
        (Some(x), Some(y)) => Some([
            x[0].min(y[0]),
            x[1].min(y[1]),
            x[2].max(y[2]),
            x[3].max(y[3]),
        ]),
    }
}

/// Sampel `store` ke kisi W (`origin_w`, `s_xy`, `size = [nx, ny]`) secara
/// mundur: tiap pusat piksel W → `u = T⁻¹(w)` → `store.get_um(u) == label`.
fn sample_to_w_grid(
    store: &LabelStore,
    t: &SlideTransform,
    label: LabelId,
    origin_w: [f64; 2],
    s_xy: f64,
    size: [usize; 2],
) -> Result<Array2<u8>, SegError> {
    let [nx, ny] = size;
    let mut out = Array2::<u8>::zeros((ny, nx));
    for j in 0..ny {
        for i in 0..nx {
            let w = [
                origin_w[0] + (i as f64 + 0.5) * s_xy,
                origin_w[1] + (j as f64 + 0.5) * s_xy,
            ];
            let u = t
                .apply_inverse(w)
                .ok_or_else(|| SegError::InvalidParameter("transformasi singular".to_string()))?;
            if store.get_um(u) == label.0 {
                out[[j, i]] = 1;
            }
        }
    }
    Ok(out)
}

/// Interpolasi label `label` antara irisan `a` (`store_a`, `T_a`, `z_a`) dan
/// `b` menuju irisan `k` (`T_k`, `z_k`).
///
/// Semua `T_*` memetakan **U irisan masing-masing → W** (5.3). Kedua masker
/// disampel mundur ke kisi W bersama (`s_xy` µm/px; gabungan kotak pembatas
/// kedua label setelah transformasi + margin `band_um`), diubah ke SDF
/// (negatif di dalam, dijepit `±band_um`), diblend linear, lalu dipetakan
/// mundur ke U_k: tiap pusat piksel usulan → `w = T_k(u)` → sampel SDF
/// terdekat `< 0`. `ids = [a, b]` dicatat di `Interpolated { from }`.
#[allow(clippy::too_many_arguments)]
pub fn interpolate_label(
    store_a: &LabelStore,
    t_a: &SlideTransform,
    z_a: f64,
    store_b: &LabelStore,
    t_b: &SlideTransform,
    z_b: f64,
    t_k: &SlideTransform,
    z_k: f64,
    ids: [SlideId; 2],
    label: LabelId,
    s_xy: f64,
    band_um: f64,
) -> Result<Proposal, SegError> {
    if !(s_xy.is_finite() && s_xy > 0.0) {
        return Err(SegError::InvalidParameter(format!(
            "s_xy tidak sah: {s_xy}"
        )));
    }
    if !(band_um.is_finite() && band_um > 0.0) {
        return Err(SegError::InvalidParameter(format!(
            "band_um tidak sah: {band_um}"
        )));
    }
    let dz = z_b - z_a;
    if !(dz.is_finite() && dz != 0.0 && z_k.is_finite()) {
        return Err(SegError::InvalidParameter(format!(
            "z tidak sah: z_a={z_a}, z_b={z_b}, z_k={z_k}"
        )));
    }
    let bb_a = store_a.bbox_um(label).map(|bb| transform_bbox(bb, t_a));
    let bb_b = store_b.bbox_um(label).map(|bb| transform_bbox(bb, t_b));
    let bb_w = union_bbox(bb_a, bb_b).ok_or(SegError::LabelNotFound(label))?;
    let margin = band_um + 2.0 * s_xy;
    let origin_w = [bb_w[0] - margin, bb_w[1] - margin];
    let nx = ((bb_w[2] + margin - origin_w[0]) / s_xy).ceil();
    let ny = ((bb_w[3] + margin - origin_w[1]) / s_xy).ceil();
    if !(nx >= 1.0 && ny >= 1.0) || nx * ny > MAX_GRID_PX {
        return Err(SegError::InvalidParameter(format!(
            "kisi W {nx}×{ny} tidak sah / terlalu besar"
        )));
    }
    let size = [nx as usize, ny as usize];
    let mask_a = sample_to_w_grid(store_a, t_a, label, origin_w, s_xy, size)?;
    let mask_b = sample_to_w_grid(store_b, t_b, label, origin_w, s_xy, size)?;
    let sdf_a = signed_distance_2d_um(&mask_a, s_xy, band_um as f32);
    let sdf_b = signed_distance_2d_um(&mask_b, s_xy, band_um as f32);
    let sdf_k = interpolate_between(&sdf_a, z_a, &sdf_b, z_b, z_k)?;

    // Kotak kisi W → U_k (mundur lewat T_k⁻¹ pada keempat sudut).
    let w_bb = [
        origin_w[0],
        origin_w[1],
        origin_w[0] + size[0] as f64 * s_xy,
        origin_w[1] + size[1] as f64 * s_xy,
    ];
    let corners = [
        [w_bb[0], w_bb[1]],
        [w_bb[2], w_bb[1]],
        [w_bb[0], w_bb[3]],
        [w_bb[2], w_bb[3]],
    ];
    let mut u_bb = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
    for c in corners {
        let u = t_k
            .apply_inverse(c)
            .ok_or_else(|| SegError::InvalidParameter("T_k singular".to_string()))?;
        u_bb[0] = u_bb[0].min(u[0]);
        u_bb[1] = u_bb[1].min(u[1]);
        u_bb[2] = u_bb[2].max(u[0]);
        u_bb[3] = u_bb[3].max(u[1]);
    }
    // Asal usulan disejajarkan ke kelipatan s_xy agar deterministik dan rapi.
    let origin_u = [
        (u_bb[0] / s_xy).floor() * s_xy,
        (u_bb[1] / s_xy).floor() * s_xy,
    ];
    let ux = ((u_bb[2] - origin_u[0]) / s_xy).ceil();
    let uy = ((u_bb[3] - origin_u[1]) / s_xy).ceil();
    if !(ux >= 1.0 && uy >= 1.0) || ux * uy > MAX_GRID_PX {
        return Err(SegError::InvalidParameter(format!(
            "kisi U_k {ux}×{uy} tidak sah / terlalu besar"
        )));
    }
    let (ux, uy) = (ux as usize, uy as usize);
    let mut mask = Array2::<u8>::zeros((uy, ux));
    for j in 0..uy {
        for i in 0..ux {
            let u = [
                origin_u[0] + (i as f64 + 0.5) * s_xy,
                origin_u[1] + (j as f64 + 0.5) * s_xy,
            ];
            let w = t_k.apply(u);
            let wi = ((w[0] - origin_w[0]) / s_xy).floor();
            let wj = ((w[1] - origin_w[1]) / s_xy).floor();
            if wi < 0.0 || wj < 0.0 || wi >= size[0] as f64 || wj >= size[1] as f64 {
                continue;
            }
            if sdf_k[[wj as usize, wi as usize]] < 0.0 {
                mask[[j, i]] = 1;
            }
        }
    }
    Ok(Proposal {
        mask,
        mpp: s_xy,
        origin_um: origin_u,
        label,
        source: AnnotationSource::Interpolated { from: ids },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::label_store::PaintMode;
    use crate::proposal::accept_proposal;
    use h3d_core::Affine2;
    use h3d_imgproc::metrics::dice;
    use h3d_imgproc::phantom::slice_ellipsoid;

    #[test]
    fn interpolate_between_blends_and_validates() {
        let a = Array2::<f32>::from_elem((2, 2), -2.0);
        let b = Array2::<f32>::from_elem((2, 2), 2.0);
        let k = interpolate_between(&a, 0.0, &b, 10.0, 2.5).unwrap();
        assert!(k.iter().all(|v| (v + 1.0).abs() < 1e-6));
        assert!(interpolate_between(&a, 0.0, &b, 0.0, 0.0).is_err());
        let c = Array2::<f32>::zeros((3, 2));
        assert!(interpolate_between(&a, 0.0, &c, 1.0, 0.5).is_err());
    }

    #[test]
    fn ellipsoid_phantom_dice() {
        // Ellipsoid 800×600×400 µm; kisi 4 µm, 200×160 px; irisan tiap 10 µm.
        let radii = [400.0, 300.0, 200.0];
        let s_xy = 4.0;
        let size = [200usize, 160usize];
        let size_um = [size[0] as f64 * s_xy, size[1] as f64 * s_xy];
        let zs: Vec<f64> = (0..=36).map(|i| -180.0 + 10.0 * i as f64).collect();
        let label = LabelId(1);
        // Tiap irisan punya pergeseran U sendiri; T_i = translasi balik ke W.
        let shift = |i: usize| -> [f64; 2] { [(i % 3) as f64 * 20.0, (i % 2) as f64 * 12.0] };
        let transform = |i: usize| -> SlideTransform {
            let s = shift(i);
            SlideTransform::Affine(Affine2::translation(-s[0], -s[1]))
        };
        let make_store = |i: usize| -> LabelStore {
            let mut st = LabelStore::new(s_xy, [size_um[0] + 100.0, size_um[1] + 100.0]);
            let truth = slice_ellipsoid(radii, zs[i], s_xy, size);
            let s = shift(i);
            st.rasterize_mask(
                &truth,
                [(s[0] / s_xy) as i64, (s[1] / s_xy) as i64],
                label,
                PaintMode::Replace,
            );
            st
        };
        let mut dices = Vec::new();
        let step = 5usize;
        let mut a = 0usize;
        while a + step < zs.len() {
            let b = a + step;
            let sa = make_store(a);
            let sb = make_store(b);
            for k in a + 1..b {
                let p = interpolate_label(
                    &sa,
                    &transform(a),
                    zs[a],
                    &sb,
                    &transform(b),
                    zs[b],
                    &transform(k),
                    zs[k],
                    [SlideId(a as u32), SlideId(b as u32)],
                    label,
                    s_xy,
                    100.0,
                )
                .unwrap();
                assert!(matches!(
                    p.source,
                    AnnotationSource::Interpolated { from } if from == [SlideId(a as u32), SlideId(b as u32)]
                ));
                let mut sk = LabelStore::new(s_xy, [size_um[0] + 100.0, size_um[1] + 100.0]);
                accept_proposal(&mut sk, &p, PaintMode::Replace);
                let truth = make_store(k);
                let d = dice(&sk.to_array(label), &truth.to_array(label)).unwrap();
                dices.push(d);
            }
            a = b;
        }
        let min = dices.iter().cloned().fold(f32::INFINITY, f32::min);
        let mean = dices.iter().sum::<f32>() / dices.len() as f32;
        eprintln!(
            "Dice interpolasi: min {min:.4}, rata {mean:.4}, n {}",
            dices.len()
        );
        assert!(mean >= 0.95, "Dice rata-rata {mean}");
        assert!(min >= 0.9, "Dice minimum {min}");
    }

    #[test]
    fn missing_label_is_error() {
        let s = LabelStore::new(4.0, [100.0, 100.0]);
        let r = interpolate_label(
            &s,
            &SlideTransform::Identity,
            0.0,
            &s,
            &SlideTransform::Identity,
            10.0,
            &SlideTransform::Identity,
            5.0,
            [SlideId(0), SlideId(1)],
            LabelId(1),
            4.0,
            50.0,
        );
        assert!(matches!(r, Err(SegError::LabelNotFound(_))));
    }
}
