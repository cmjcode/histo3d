//! Perasteran mungil di atas buffer RGBA untuk gambar laporan (8.12.8 butir
//! 3): garis, cakram, cincin, panah, dan pengodean PNG. Tanpa crate GUI.
//!
//! Koordinat kontinu dengan pusat piksel `(i, j)` di `(i + 0,5, j + 0,5)`
//! (aturan keras 5): titik `(x, y)` jatuh di piksel `(⌊x⌋, ⌊y⌋)`. Semua
//! operasi deterministik dan memotong diam-diam di luar kanvas.

use crate::error::EngineError;

/// Warna RGBA.
pub type Color = [u8; 4];

/// Warna penanda titik tumor terdekat.
pub const TUMOR_MARKER: Color = [214, 0, 110, 255];
/// Warna penanda titik tepi reseksi terdekat.
pub const MARGIN_MARKER: Color = [0, 120, 215, 255];
/// Warna panah tumor → tepi.
pub const ARROW: Color = [20, 20, 20, 255];
/// Warna lingkar terang di sekeliling penanda dan panah (kontras).
pub const HALO: Color = [255, 255, 255, 255];

/// Kanvas RGBA 8-bit, baris demi baris.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Canvas {
    w: u32,
    h: u32,
    rgba: Vec<u8>,
}

impl Canvas {
    /// Kanvas `w × h` berwarna `fill`.
    pub fn new(w: u32, h: u32, fill: Color) -> Self {
        let mut rgba = Vec::with_capacity(w as usize * h as usize * 4);
        for _ in 0..w as usize * h as usize {
            rgba.extend_from_slice(&fill);
        }
        Canvas { w, h, rgba }
    }

    /// Dari buffer RGBA; `None` bila panjangnya bukan `w × h × 4`.
    pub fn from_rgba(w: u32, h: u32, rgba: Vec<u8>) -> Option<Self> {
        (rgba.len() == w as usize * h as usize * 4).then_some(Canvas { w, h, rgba })
    }

    /// Lebar, piksel.
    pub fn width(&self) -> u32 {
        self.w
    }

    /// Tinggi, piksel.
    pub fn height(&self) -> u32 {
        self.h
    }

    /// Buffer RGBA.
    pub fn as_rgba(&self) -> &[u8] {
        &self.rgba
    }

    /// Ambil buffer RGBA.
    pub fn into_rgba(self) -> Vec<u8> {
        self.rgba
    }

    /// Warna piksel `(x, y)`; `None` di luar kanvas.
    pub fn get(&self, x: i64, y: i64) -> Option<Color> {
        if x < 0 || y < 0 || x >= i64::from(self.w) || y >= i64::from(self.h) {
            return None;
        }
        let i = (y as usize * self.w as usize + x as usize) * 4;
        Some([
            self.rgba[i],
            self.rgba[i + 1],
            self.rgba[i + 2],
            self.rgba[i + 3],
        ])
    }

    /// Timpa piksel `(x, y)` (tanpa pencampuran alfa); di luar kanvas diabaikan.
    pub fn put(&mut self, x: i64, y: i64, c: Color) {
        if x < 0 || y < 0 || x >= i64::from(self.w) || y >= i64::from(self.h) {
            return;
        }
        let i = (y as usize * self.w as usize + x as usize) * 4;
        self.rgba[i..i + 4].copy_from_slice(&c);
    }

    /// Cakram terisi berpusat `c` (koordinat kontinu) berjari-jari `r` piksel:
    /// piksel yang pusatnya berjarak ≤ `r` dari `c`. Piksel yang memuat `c`
    /// selalu ikut terisi.
    pub fn disc(&mut self, c: [f64; 2], r: f64, color: Color) {
        self.ring(c, 0.0, r, color);
        if c[0].is_finite() && c[1].is_finite() {
            self.put(c[0].floor() as i64, c[1].floor() as i64, color);
        }
    }

    /// Cincin: piksel yang pusatnya berjarak di `[r_in, r_out]` dari `c`.
    pub fn ring(&mut self, c: [f64; 2], r_in: f64, r_out: f64, color: Color) {
        if !(c[0].is_finite() && c[1].is_finite() && r_out.is_finite() && r_out >= 0.0) {
            return;
        }
        let x0 = (c[0] - r_out).floor().max(-1.0) as i64;
        let x1 = (c[0] + r_out).ceil().min(f64::from(self.w)) as i64;
        let y0 = (c[1] - r_out).floor().max(-1.0) as i64;
        let y1 = (c[1] + r_out).ceil().min(f64::from(self.h)) as i64;
        let (lo, hi) = (r_in.max(0.0).powi(2), r_out * r_out);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let dx = x as f64 + 0.5 - c[0];
                let dy = y as f64 + 0.5 - c[1];
                let d2 = dx * dx + dy * dy;
                if d2 >= lo && d2 <= hi {
                    self.put(x, y, color);
                }
            }
        }
    }

    /// Garis `a`–`b` setebal `width` piksel (cakram berdiameter `width`
    /// dicap tiap setengah piksel sepanjang garis).
    pub fn line(&mut self, a: [f64; 2], b: [f64; 2], width: f64, color: Color) {
        if !(a.iter().chain(b.iter()).all(|v| v.is_finite()) && width.is_finite()) {
            return;
        }
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let len = (dx * dx + dy * dy).sqrt();
        // Garis yang sangat panjang (titik jauh di luar kanvas) dipotong ke
        // jumlah langkah yang wajar; sisanya memang di luar kanvas.
        let steps = ((len * 2.0).ceil() as usize).clamp(1, 8 * (self.w as usize + self.h as usize));
        let r = (width / 2.0).max(0.0);
        for i in 0..=steps {
            let t = i as f64 / steps as f64;
            let p = [a[0] + dx * t, a[1] + dy * t];
            if r <= 0.5 {
                self.put(p[0].floor() as i64, p[1].floor() as i64, color);
            } else {
                self.disc(p, r, color);
            }
        }
    }

    /// Panah dari `from` ke `to`: batang setebal `width` dan dua sisi kepala
    /// sepanjang `head` piksel bersudut ±25° di ujung `to`. Panah yang lebih
    /// pendek dari kepalanya digambar sebagai batang saja.
    pub fn arrow(&mut self, from: [f64; 2], to: [f64; 2], width: f64, head: f64, color: Color) {
        self.line(from, to, width, color);
        let (dx, dy) = (to[0] - from[0], to[1] - from[1]);
        let len = (dx * dx + dy * dy).sqrt();
        if !(len.is_finite() && len > head && head > 0.0) {
            return;
        }
        let (ux, uy) = (dx / len, dy / len);
        let (s, c) = 25f64.to_radians().sin_cos();
        for sign in [1.0, -1.0] {
            // Arah mundur diputar ±25°.
            let bx = -(ux * c - sign * uy * s);
            let by = -(sign * ux * s + uy * c);
            self.line(to, [to[0] + bx * head, to[1] + by * head], width, color);
        }
    }

    /// Kode PNG (RGBA 8-bit). Deterministik untuk isi yang sama.
    pub fn encode_png(&self) -> Result<Vec<u8>, EngineError> {
        let mut out = Vec::new();
        let enc = image::codecs::png::PngEncoder::new(&mut out);
        image::ImageEncoder::write_image(
            enc,
            &self.rgba,
            self.w,
            self.h,
            image::ExtendedColorType::Rgba8,
        )
        .map_err(|e| EngineError::invalid(format!("gagal mengode PNG: {e}")))?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WHITE: Color = [255, 255, 255, 255];
    const RED: Color = [255, 0, 0, 255];

    #[test]
    fn primitives_follow_pixel_centre_convention_and_clip() {
        let mut c = Canvas::new(20, 10, WHITE);
        assert_eq!((c.width(), c.height()), (20, 10));
        c.put(-1, 0, RED);
        c.put(20, 0, RED);
        c.put(0, 10, RED);
        assert!(c.as_rgba().chunks_exact(4).all(|p| p == WHITE));
        assert_eq!(c.get(-1, 0), None);
        // Cakram r = 1,2 di (5,5; 5,5): pusat piksel (5,5) dan 4 tetangganya.
        c.disc([5.5, 5.5], 1.2, RED);
        for (x, y) in [(5, 5), (4, 5), (6, 5), (5, 4), (5, 6)] {
            assert_eq!(c.get(x, y), Some(RED), "({x},{y})");
        }
        assert_eq!(c.get(4, 4), Some(WHITE));
        // Cakram ber-jari-jari nol tetap menandai piksel yang memuat pusat.
        c.disc([12.9, 2.1], 0.0, RED);
        assert_eq!(c.get(12, 2), Some(RED));
        // Cincin tidak mengisi pusatnya.
        let mut r = Canvas::new(21, 21, WHITE);
        r.ring([10.5, 10.5], 6.0, 8.0, RED);
        assert_eq!(r.get(10, 10), Some(WHITE));
        assert_eq!(r.get(17, 10), Some(RED));
        assert_eq!(r.get(10, 3), Some(RED));
        // Sebagian di luar kanvas: tidak panik.
        r.disc([-3.0, 25.0], 6.0, RED);
        r.line([-50.0, -50.0], [500.0, 80.0], 3.0, RED);
        r.arrow([1e9, 1e9], [f64::NAN, 3.0], 2.0, 5.0, RED);
    }

    #[test]
    fn line_and_arrow_cover_their_endpoints() {
        let mut c = Canvas::new(40, 30, WHITE);
        c.line([2.5, 3.5], [30.5, 20.5], 1.0, RED);
        assert_eq!(c.get(2, 3), Some(RED));
        assert_eq!(c.get(30, 20), Some(RED));
        assert_eq!(c.get(16, 12), Some(RED), "titik tengah");
        assert_eq!(c.get(2, 20), Some(WHITE));
        let mut a = Canvas::new(40, 30, WHITE);
        a.arrow([5.5, 15.5], [30.5, 15.5], 2.0, 8.0, RED);
        assert_eq!(a.get(30, 15), Some(RED));
        // Sisi kepala panah: mundur 8 px pada ±25° → (23,2; 15,5 ± 3,4).
        assert_eq!(a.get(24, 18), Some(RED));
        assert_eq!(a.get(24, 12), Some(RED));
        assert_eq!(a.get(34, 15), Some(WHITE), "tidak melewati ujung");
    }

    #[test]
    fn png_roundtrip_and_determinism() {
        let mut c = Canvas::new(17, 9, [10, 20, 30, 255]);
        c.arrow([1.0, 1.0], [15.0, 7.0], 1.5, 4.0, RED);
        let a = c.encode_png().unwrap();
        assert_eq!(a, c.encode_png().unwrap());
        let img = image::load_from_memory(&a).unwrap().to_rgba8();
        assert_eq!((img.width(), img.height()), (17, 9));
        assert_eq!(img.as_raw().as_slice(), c.as_rgba());
        assert!(Canvas::from_rgba(2, 2, vec![0; 15]).is_none());
        assert_eq!(
            Canvas::from_rgba(1, 1, vec![1, 2, 3, 4])
                .unwrap()
                .into_rgba(),
            vec![1, 2, 3, 4]
        );
    }
}
