//! Morfologi biner lewat ambang EDT (isotropik, O(n)).

use ndarray::Array2;

use crate::components::{label_components_conn, Connectivity};
use crate::edt::edt_squared_2d;

/// Dilasi cakram: piksel menjadi 1 bila jarak pusatnya ke piksel 1 terdekat
/// `≤ radius_px`. `radius_px ≤ 0` → salinan masker.
pub fn dilate_disc(mask: &Array2<u8>, radius_px: f64) -> Array2<u8> {
    if radius_px.is_nan() || radius_px <= 0.0 {
        return mask.mapv(|m| u8::from(m != 0));
    }
    let r2 = (radius_px * radius_px) as f32;
    edt_squared_2d(mask).mapv(|d| u8::from(d <= r2))
}

/// Erosi cakram: piksel 1 bertahan bila jarak ke piksel 0 terdekat
/// `> radius_px`. `radius_px ≤ 0` → salinan masker.
pub fn erode_disc(mask: &Array2<u8>, radius_px: f64) -> Array2<u8> {
    if radius_px.is_nan() || radius_px <= 0.0 {
        return mask.mapv(|m| u8::from(m != 0));
    }
    let inv = mask.mapv(|m| u8::from(m == 0));
    let r2 = (radius_px * radius_px) as f32;
    edt_squared_2d(&inv).mapv(|d| u8::from(d > r2))
}

/// Pembukaan: erosi lalu dilasi.
pub fn open(mask: &Array2<u8>, radius_px: f64) -> Array2<u8> {
    dilate_disc(&erode_disc(mask, radius_px), radius_px)
}

/// Penutupan: dilasi lalu erosi.
pub fn close(mask: &Array2<u8>, radius_px: f64) -> Array2<u8> {
    erode_disc(&dilate_disc(mask, radius_px), radius_px)
}

/// Isi lubang: komponen latar (konektivitas 4) yang tidak menyentuh tepi
/// citra menjadi 1.
pub fn fill_holes(mask: &Array2<u8>) -> Array2<u8> {
    let (h, w) = mask.dim();
    if h == 0 || w == 0 {
        return mask.clone();
    }
    let bg = mask.mapv(|m| u8::from(m == 0));
    let (labels, n) = label_components_conn(&bg, Connectivity::Four);
    let mut touches = vec![false; n + 1];
    for x in 0..w {
        touches[labels[[0, x]] as usize] = true;
        touches[labels[[h - 1, x]] as usize] = true;
    }
    for y in 0..h {
        touches[labels[[y, 0]] as usize] = true;
        touches[labels[[y, w - 1]] as usize] = true;
    }
    let mut out = mask.mapv(|m| u8::from(m != 0));
    for (o, &l) in out.iter_mut().zip(labels.iter()) {
        if l != 0 && !touches[l as usize] {
            *o = 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn disc(n: usize, r: f64) -> Array2<u8> {
        let c = n as f64 / 2.0;
        Array2::from_shape_fn((n, n), |(y, x)| {
            let dx = x as f64 + 0.5 - c;
            let dy = y as f64 + 0.5 - c;
            u8::from(dx * dx + dy * dy <= r * r)
        })
    }

    fn area(m: &Array2<u8>) -> f64 {
        m.iter().filter(|v| **v != 0).count() as f64
    }

    #[test]
    fn dilate_erode_area_close_to_expected() {
        let m = disc(128, 20.0);
        let d = dilate_disc(&m, 10.0);
        let e = erode_disc(&m, 10.0);
        let want_d = PI * 30.0 * 30.0;
        let want_e = PI * 10.0 * 10.0;
        assert!(
            (area(&d) - want_d).abs() / want_d < 0.05,
            "dilasi {}",
            area(&d)
        );
        assert!(
            (area(&e) - want_e).abs() / want_e < 0.10,
            "erosi {}",
            area(&e)
        );
        // Open/close pada cakram mulus hampir identitas.
        assert!((area(&open(&m, 3.0)) - area(&m)).abs() < 20.0);
        assert!((area(&close(&m, 3.0)) - area(&m)).abs() < 20.0);
    }

    #[test]
    fn zero_radius_is_identity() {
        let m = disc(16, 5.0);
        assert_eq!(dilate_disc(&m, 0.0), m);
        assert_eq!(erode_disc(&m, 0.0), m);
    }

    #[test]
    fn fill_holes_fills_enclosed_only() {
        let mut m = Array2::<u8>::zeros((7, 7));
        for y in 1..6 {
            for x in 1..6 {
                m[[y, x]] = 1;
            }
        }
        m[[3, 3]] = 0; // lubang
        m[[0, 0]] = 0; // latar tepi
        let f = fill_holes(&m);
        assert_eq!(f[[3, 3]], 1);
        assert_eq!(f[[0, 0]], 0);
        assert_eq!(area(&f), 25.0);
    }

    #[test]
    fn fill_holes_diagonal_leak_is_not_hole() {
        // Latar yang hanya terhubung diagonal ke tepi tetap lubang
        // (konektivitas 4 untuk latar).
        let mut m = Array2::<u8>::ones((4, 4));
        m[[1, 1]] = 0;
        m[[0, 0]] = 0;
        let f = fill_holes(&m);
        assert_eq!(f[[1, 1]], 1);
        assert_eq!(f[[0, 0]], 0);
    }
}
