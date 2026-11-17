//! Kontur batas luar objek pada raster label (deteksi sel, histo3d.md
//! 8.12.5): mengubah wilayah hasil watershed menjadi poligon yang mengikuti
//! tepi inti sebenarnya, seperti tampilan segmentasi instance.
//!
//! # Konvensi
//! - Kontur ditelusuri di sepanjang **retakan** (sisi piksel antara isi dan
//!   latar, konektivitas 4). Titik sudut berada di koordinat sudut piksel:
//!   piksel `(x, y)` menempati `[x, x+1] × [y, y+1]`, sehingga pusatnya di
//!   `+0,5` (histo3d.md 5.2) dan luas poligon = jumlah piksel objek
//!   (tanpa lubang).
//! - Arah penelusuran searah jarum jam pada layar (y ke bawah), mulai dari
//!   sudut kiri-atas piksel pertama objek dalam urutan raster; deterministik.
//! - Hanya batas luar komponen-4 yang memuat piksel pertama; lubang dan
//!   bagian yang hanya tersambung diagonal diabaikan.

use ndarray::Array2;

use crate::components::{label_components_conn, Connectivity};

/// Arah gerak di kisi sudut: Timur, Selatan, Barat, Utara (searah jarum jam
/// pada layar y-ke-bawah).
const DIRS: [(i64, i64); 4] = [(1, 0), (0, 1), (-1, 0), (0, -1)];

/// Kontur batas luar setiap label `1..=n_labels` dalam koordinat sudut
/// piksel (satuan piksel). Indeks `0` dan label yang tidak muncul berisi
/// vektor kosong. Satu sapuan raster mencari piksel awal, lalu tiap objek
/// ditelusuri dengan biaya sebanding kelilingnya.
pub fn label_contours(labels: &Array2<u32>, n_labels: usize) -> Vec<Vec<[f32; 2]>> {
    let mut start: Vec<Option<(usize, usize)>> = vec![None; n_labels + 1];
    for ((y, x), &l) in labels.indexed_iter() {
        let i = l as usize;
        if l != 0 && i <= n_labels && start[i].is_none() {
            start[i] = Some((x, y));
        }
    }
    start
        .iter()
        .enumerate()
        .map(|(i, s)| match s {
            Some((x, y)) if i > 0 => trace_outer(labels, i as u32, *x, *y),
            _ => Vec::new(),
        })
        .collect()
}

/// Kontur lubang setiap label `1..=n_labels` (koordinat sudut piksel,
/// satuan piksel). Lubang = komponen latar (`labels == 0`, konektivitas 4)
/// yang tidak menyentuh tepi citra; pemiliknya label tepat di atas piksel
/// pertama komponen itu dalam urutan raster (piksel itu pasti di batas
/// luar lubang, sehingga objek kecil di dalam lubang, mis. serpihan di
/// lumen, tidak merebut lubang). Tiap kontur ditelusuri sebagai batas luar
/// komponen latar, searah jarum jam; urutan lubang per label = urutan
/// raster. Indeks `0` kosong.
pub fn label_hole_contours(labels: &Array2<u32>, n_labels: usize) -> Vec<Vec<Vec<[f32; 2]>>> {
    let mut out: Vec<Vec<Vec<[f32; 2]>>> = vec![Vec::new(); n_labels + 1];
    let (h, w) = labels.dim();
    if h < 3 || w < 3 {
        return out;
    }
    let bg = labels.mapv(|l| u8::from(l == 0));
    let (bl, nb) = label_components_conn(&bg, Connectivity::Four);
    let mut touches = vec![false; nb + 1];
    for x in 0..w {
        touches[bl[[0, x]] as usize] = true;
        touches[bl[[h - 1, x]] as usize] = true;
    }
    for y in 0..h {
        touches[bl[[y, 0]] as usize] = true;
        touches[bl[[y, w - 1]] as usize] = true;
    }
    // Piksel pertama tiap komponen latar dalam urutan raster.
    let mut first: Vec<Option<(usize, usize)>> = vec![None; nb + 1];
    for ((y, x), &b) in bl.indexed_iter() {
        let i = b as usize;
        if b != 0 && first[i].is_none() {
            first[i] = Some((x, y));
        }
    }
    for (b, f) in first.iter().enumerate().skip(1) {
        let Some((x, y)) = *f else {
            continue;
        };
        if touches[b] || y == 0 {
            continue;
        }
        let owner = labels[[y - 1, x]] as usize;
        if owner == 0 || owner > n_labels {
            continue;
        }
        let c = trace_outer(&bl, b as u32, x, y);
        if c.len() >= 4 {
            out[owner].push(c);
        }
    }
    out
}

/// Telusuri batas luar label `id` mulai dari piksel `(x0, y0)`, yang harus
/// piksel pertama label itu dalam urutan raster (tetangga atas dan kirinya
/// bukan `id`). Hanya sudut (titik belok) yang dikeluarkan.
pub fn trace_outer(labels: &Array2<u32>, id: u32, x0: usize, y0: usize) -> Vec<[f32; 2]> {
    let (h, w) = labels.dim();
    if x0 >= w || y0 >= h || labels[[y0, x0]] != id {
        return Vec::new();
    }
    let fg = |x: i64, y: i64| -> bool {
        x >= 0
            && y >= 0
            && (x as usize) < w
            && (y as usize) < h
            && labels[[y as usize, x as usize]] == id
    };
    // Piksel di depan-kanan dan depan-kiri sudut (vx, vy) untuk arah d.
    // Sekitar sudut: TL = (vx-1, vy-1), TR = (vx, vy-1), BL = (vx-1, vy),
    // BR = (vx, vy).
    let ahead = |vx: i64, vy: i64, d: usize| -> (bool, bool) {
        match d {
            0 => (fg(vx, vy), fg(vx, vy - 1)), // Timur: kanan BR, kiri TR
            1 => (fg(vx - 1, vy), fg(vx, vy)), // Selatan: kanan BL, kiri BR
            2 => (fg(vx - 1, vy - 1), fg(vx - 1, vy)), // Barat: kanan TL, kiri BL
            _ => (fg(vx, vy - 1), fg(vx - 1, vy - 1)), // Utara: kanan TR, kiri TL
        }
    };
    let start = (x0 as i64, y0 as i64);
    let mut out = vec![[x0 as f32, y0 as f32]];
    let mut d = 0usize;
    let mut p = (start.0 + 1, start.1);
    // Batas atas langkah: keliling objek tidak mungkin melebihi jumlah
    // retakan seluruh raster; penjaga terhadap loop tak berujung.
    let max_steps = 2 * (w + 1) * (h + 1) + 8;
    for _ in 0..max_steps {
        if p == start {
            return out;
        }
        let (right, left) = ahead(p.0, p.1, d);
        // Isi selalu di kanan: belok kanan bila depan-kanan latar, belok
        // kiri bila depan-kiri juga isi, selain itu lurus.
        let nd = if !right {
            (d + 1) % 4
        } else if left {
            (d + 3) % 4
        } else {
            d
        };
        if nd != d {
            out.push([p.0 as f32, p.1 as f32]);
        }
        d = nd;
        p = (p.0 + DIRS[d].0, p.1 + DIRS[d].1);
    }
    out
}

/// Sederhanakan poligon tertutup dengan Douglas–Peucker (toleransi `eps`
/// dalam satuan koordinat poligon). Titik pertama dan titik terjauh
/// darinya menjadi jangkar sehingga hasil tidak bergantung pada arah.
/// Poligon dengan < 4 titik atau `eps` ≤ 0 dikembalikan apa adanya.
pub fn simplify_closed(poly: &[[f32; 2]], eps: f32) -> Vec<[f32; 2]> {
    let n = poly.len();
    if n < 4 || !eps.is_finite() || eps <= 0.0 {
        return poly.to_vec();
    }
    let far = (1..n)
        .max_by(|&a, &b| {
            dist2(poly[0], poly[a])
                .total_cmp(&dist2(poly[0], poly[b]))
                .then(b.cmp(&a))
        })
        .unwrap_or(n / 2);
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[far] = true;
    dp_mark(poly, 0, far, eps, &mut keep);
    dp_mark_wrap(poly, far, n, eps, &mut keep);
    let out: Vec<[f32; 2]> = poly
        .iter()
        .zip(&keep)
        .filter_map(|(p, &k)| k.then_some(*p))
        .collect();
    if out.len() < 3 {
        poly.to_vec()
    } else {
        out
    }
}

/// Haluskan poligon tertutup dengan pemotongan sudut Chaikin (`iters`
/// kali; tiap iterasi menggandakan jumlah titik). Bentuk sedikit mengecil
/// di sudut tajam; dipakai hanya untuk tampilan, bukan pengukuran.
pub fn chaikin_closed(poly: &[[f32; 2]], iters: usize) -> Vec<[f32; 2]> {
    let mut cur = poly.to_vec();
    if cur.len() < 3 {
        return cur;
    }
    for _ in 0..iters {
        let n = cur.len();
        let mut next = Vec::with_capacity(2 * n);
        for i in 0..n {
            let a = cur[i];
            let b = cur[(i + 1) % n];
            next.push([0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]]);
            next.push([0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]]);
        }
        cur = next;
    }
    cur
}

/// Luas bertanda poligon (rumus tali sepatu); positif untuk arah searah
/// jarum jam pada layar y-ke-bawah.
pub fn signed_area(poly: &[[f32; 2]]) -> f64 {
    let n = poly.len();
    let mut s = 0.0f64;
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        s += f64::from(a[0]) * f64::from(b[1]) - f64::from(b[0]) * f64::from(a[1]);
    }
    0.5 * s
}

fn dist2(a: [f32; 2], b: [f32; 2]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

/// Jarak titik `p` ke segmen `a`–`b`.
fn seg_dist(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let (vx, vy) = (b[0] - a[0], b[1] - a[1]);
    let l2 = vx * vx + vy * vy;
    if l2 <= 0.0 {
        return dist2(p, a).sqrt();
    }
    let t = (((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / l2).clamp(0.0, 1.0);
    dist2(p, [a[0] + t * vx, a[1] + t * vy]).sqrt()
}

/// Douglas–Peucker iteratif pada indeks `i..=j` (tanpa rekursi dalam).
fn dp_mark(poly: &[[f32; 2]], i: usize, j: usize, eps: f32, keep: &mut [bool]) {
    let mut stack = vec![(i, j)];
    while let Some((a, b)) = stack.pop() {
        if b <= a + 1 {
            continue;
        }
        let (mut best, mut dmax) = (a, -1.0f32);
        for k in a + 1..b {
            let d = seg_dist(poly[k], poly[a], poly[b]);
            if d > dmax {
                dmax = d;
                best = k;
            }
        }
        if dmax > eps {
            keep[best] = true;
            stack.push((a, best));
            stack.push((best, b));
        }
    }
}

/// Douglas–Peucker untuk ruas penutup `far..n` lalu kembali ke indeks 0.
fn dp_mark_wrap(poly: &[[f32; 2]], far: usize, n: usize, eps: f32, keep: &mut [bool]) {
    let mut ext: Vec<[f32; 2]> = poly[far..n].to_vec();
    ext.push(poly[0]);
    let mut k2 = vec![false; ext.len()];
    dp_mark(&ext, 0, ext.len() - 1, eps, &mut k2);
    for (o, k) in k2.iter().enumerate() {
        if *k && far + o < n {
            keep[far + o] = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(h: usize, w: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> Array2<u32> {
        let mut a = Array2::<u32>::zeros((h, w));
        for y in y0..y1 {
            for x in x0..x1 {
                a[[y, x]] = 1;
            }
        }
        a
    }

    #[test]
    fn square_traces_four_corners_clockwise() {
        let a = rect(8, 8, 2, 3, 6, 5);
        let c = label_contours(&a, 1);
        assert!(c[0].is_empty());
        assert_eq!(c[1], vec![[2.0, 3.0], [6.0, 3.0], [6.0, 5.0], [2.0, 5.0]]);
        assert!((signed_area(&c[1]) - 8.0).abs() < 1e-9);
    }

    #[test]
    fn single_pixel_and_image_border() {
        let a = rect(3, 3, 0, 0, 1, 1);
        let c = label_contours(&a, 1);
        assert_eq!(c[1].len(), 4);
        assert!((signed_area(&c[1]) - 1.0).abs() < 1e-9);
        let full = rect(4, 5, 0, 0, 5, 4);
        assert!((signed_area(&label_contours(&full, 1)[1]) - 20.0).abs() < 1e-9);
    }

    #[test]
    fn concave_shape_area_matches_pixels() {
        // Bentuk L dan cincin (lubang diabaikan: luas = luar penuh).
        let mut a = rect(10, 10, 1, 1, 8, 3);
        for y in 3..8 {
            a[[y, 1]] = 1;
            a[[y, 2]] = 1;
        }
        let px = a.iter().filter(|&&v| v == 1).count() as f64;
        let c = label_contours(&a, 1);
        assert_eq!(c[1].len(), 6);
        assert!((signed_area(&c[1]) - px).abs() < 1e-9);
        let mut ring = rect(9, 9, 1, 1, 8, 8);
        ring[[4, 4]] = 0;
        assert!((signed_area(&label_contours(&ring, 1)[1]) - 49.0).abs() < 1e-9);
    }

    #[test]
    fn touching_labels_get_separate_contours() {
        let mut a = rect(6, 10, 1, 1, 5, 5);
        for y in 1..5 {
            for x in 5..9 {
                a[[y, x]] = 2;
            }
        }
        let c = label_contours(&a, 3);
        assert!((signed_area(&c[1]) - 16.0).abs() < 1e-9);
        assert!((signed_area(&c[2]) - 16.0).abs() < 1e-9);
        assert!(c[3].is_empty());
        // Sisi bersama x = 5 muncul di kedua kontur.
        assert!(c[1].iter().any(|p| p[0] == 5.0));
        assert!(c[2].iter().any(|p| p[0] == 5.0));
    }

    #[test]
    fn diagonal_only_part_is_ignored() {
        let mut a = Array2::<u32>::zeros((4, 4));
        a[[1, 1]] = 1;
        a[[2, 2]] = 1;
        let c = label_contours(&a, 1);
        assert!((signed_area(&c[1]) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn disc_simplify_and_smooth() {
        let (n, r) = (41usize, 15.0f64);
        let mut a = Array2::<u32>::zeros((n, n));
        for ((y, x), v) in a.indexed_iter_mut() {
            let dx = x as f64 + 0.5 - 20.5;
            let dy = y as f64 + 0.5 - 20.5;
            if dx * dx + dy * dy <= r * r {
                *v = 1;
            }
        }
        let px = a.iter().filter(|&&v| v == 1).count() as f64;
        let raw = &label_contours(&a, 1)[1];
        assert!((signed_area(raw) - px).abs() < 1e-9);
        let s = simplify_closed(raw, 0.75);
        assert!(s.len() < raw.len());
        assert!((signed_area(&s) - px).abs() / px < 0.05);
        let sm = chaikin_closed(&s, 2);
        assert_eq!(sm.len(), 4 * s.len());
        assert!((signed_area(&sm) - px).abs() / px < 0.05);
        // Deterministik.
        assert_eq!(simplify_closed(raw, 0.75), s);
    }

    #[test]
    fn ring_hole_belongs_to_ring_not_to_debris_inside() {
        // Cincin label 1 (luar 2..18, lumen 6..14) dengan serpihan label 2
        // di dalam lumen, plus latar tepi yang bukan lubang.
        let mut a = rect(20, 20, 2, 2, 18, 18);
        for y in 6..14 {
            for x in 6..14 {
                a[[y, x]] = 0;
            }
        }
        a[[9, 9]] = 2;
        let holes = label_hole_contours(&a, 2);
        assert!(holes[0].is_empty());
        assert_eq!(holes[1].len(), 1);
        assert!(holes[2].is_empty());
        // Luas lubang = 64 piksel (serpihan di dalamnya tidak mengurangi
        // batas luar lubang).
        assert!((signed_area(&holes[1][0]) - 64.0).abs() < 1e-9);
        // Luas objek = luar − lubang = jumlah piksel cincin.
        let ring_px = a.iter().filter(|&&v| v == 1).count() as f64;
        let outer = &label_contours(&a, 2)[1];
        assert!((signed_area(outer) - signed_area(&holes[1][0]) - ring_px).abs() < 1e-9);
        // Citra terlalu kecil → kosong.
        assert!(label_hole_contours(&rect(2, 2, 0, 0, 1, 1), 1)[1].is_empty());
    }

    #[test]
    fn degenerate_inputs() {
        let a = Array2::<u32>::zeros((0, 0));
        assert!(label_contours(&a, 2).iter().all(Vec::is_empty));
        assert!(trace_outer(&rect(3, 3, 0, 0, 1, 1), 2, 0, 0).is_empty());
        let tri = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        assert_eq!(simplify_closed(&tri, 1.0), tri.to_vec());
        assert_eq!(simplify_closed(&tri, f32::NAN), tri.to_vec());
        assert_eq!(chaikin_closed(&tri[..2], 3).len(), 2);
    }
}
