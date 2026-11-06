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

/// Isi lubang bermulut sempit: latar dianggap "luar" hanya bila terjangkau
/// dari tepi citra oleh cakram radius `mouth_radius_px` yang seluruhnya
/// berada di latar (pembukaan latar yang tersambung ke tepi). Rongga yang
/// hanya tersambung ke luar lewat celah lebih sempit dari `2 × radius`
/// ikut menjadi 1, seperti lubang tertutup pada [`fill_holes`]. Daerah di
/// luar citra dianggap latar. `radius ≤ 0` → sama dengan [`fill_holes`].
pub fn fill_narrow_mouthed_holes(mask: &Array2<u8>, mouth_radius_px: f64) -> Array2<u8> {
    let filled = fill_holes(mask);
    let (h, w) = filled.dim();
    if h == 0 || w == 0 || mouth_radius_px.is_nan() || mouth_radius_px <= 0.0 {
        return filled;
    }
    // Di luar citra dianggap latar: pad selebar radius + 1 agar latar sempit
    // antara jaringan dan tepi citra tetap terjangkau dari luar.
    let pad = mouth_radius_px.ceil() as usize + 1;
    let (ph, pw) = (h + 2 * pad, w + 2 * pad);
    let mut bg = Array2::<u8>::ones((ph, pw));
    for ((y, x), &m) in filled.indexed_iter() {
        bg[[y + pad, x + pad]] = u8::from(m == 0);
    }
    // Inti latar: titik yang berjarak > radius dari jaringan. Komponen inti
    // yang menyentuh tepi citra berpad adalah pusat cakram "luar".
    let core = erode_disc(&bg, mouth_radius_px);
    let (labels, n) = label_components_conn(&core, Connectivity::Four);
    let mut touches = vec![false; n + 1];
    for x in 0..pw {
        touches[labels[[0, x]] as usize] = true;
        touches[labels[[ph - 1, x]] as usize] = true;
    }
    for y in 0..ph {
        touches[labels[[y, 0]] as usize] = true;
        touches[labels[[y, pw - 1]] as usize] = true;
    }
    let outer_core = labels.mapv(|l| u8::from(l != 0 && touches[l as usize]));
    // Latar luar = sapuan cakram dari inti luar (masih di dalam latar).
    let outside = dilate_disc(&outer_core, mouth_radius_px);
    let mut out = filled;
    for ((y, x), o) in out.indexed_iter_mut() {
        if *o == 0 && outside[[y + pad, x + pad]] == 0 {
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

    #[test]
    fn narrow_mouthed_cavity_is_filled_outside_kept() {
        // Cincin jaringan 20..60 dengan rongga 26..54 yang terbuka ke kanan
        // lewat celah setinggi 4 px; latar luar tetap 0.
        let n = 80;
        let mut m = Array2::<u8>::zeros((n, n));
        for y in 20..60 {
            for x in 20..60 {
                let inner = (26..54).contains(&y) && (26..54).contains(&x);
                m[[y, x]] = u8::from(!inner);
            }
        }
        for y in 38..42 {
            for x in 54..60 {
                m[[y, x]] = 0;
            }
        }
        // Tanpa celah lebar, fill_holes biasa tidak mengisi rongga.
        assert_eq!(fill_holes(&m)[[40, 40]], 0);
        let f = fill_narrow_mouthed_holes(&m, 4.0);
        assert_eq!(f[[40, 40]], 1, "rongga bermulut sempit terisi");
        assert_eq!(f[[40, 57]], 1, "celah ikut terisi");
        assert_eq!(f[[5, 5]], 0, "latar luar tetap");
        assert_eq!(f[[40, 70]], 0, "latar luar di depan celah tetap");
        // Celah lebih lebar dari 2 × radius → rongga tetap latar.
        let wide = fill_narrow_mouthed_holes(&m, 1.0);
        assert_eq!(wide[[40, 40]], 0);
        // Masker tanpa jaringan tidak berubah.
        let empty = Array2::<u8>::zeros((10, 10));
        assert_eq!(fill_narrow_mouthed_holes(&empty, 3.0), empty);
    }
}
