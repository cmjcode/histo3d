//! Penghitungan objek 2D pada satu penampang (histo3d.md 8.12.5): berapa
//! banyak objek tertutup (kelenjar, asinus, pembuluh, pulau tumor) yang ada
//! di sebuah masker biner, beserta ukuran tiap objek.
//!
//! # Alur
//! 1. Lubang di dalam objek (lumen kelenjar) diisi supaya cincin + lumen
//!    menjadi satu objek, bukan dua tepi (`fill_holes`).
//! 2. Pembukaan morfologi ringan memutus jembatan tipis antar objek.
//! 3. Komponen terhubung konektivitas 8; tiap komponen dipotong pada kotak
//!    pembatasnya lalu (bila `split_enabled`) dibelah dengan watershed
//!    berpenanda di atas EDT ([`split_touching`], h-maxima) supaya objek
//!    yang bersentuhan terhitung terpisah. Pemrosesan per kotak pembatas
//!    menjaga memori kecil pada slide besar.
//! 4. Wilayah hasil belah yang lebih kecil dari `merge_min_area_um2`
//!    digabung ke tetangga terluas; objek lebih kecil dari `min_area_um2`
//!    dibuang.
//! 5. Aturan tepi (bingkai hitung tak bias Gundersen): objek yang menyentuh
//!    tepi **kiri** atau **bawah** masker tidak dihitung tetapi tetap
//!    dilaporkan (`excluded`), sehingga jumlah per bingkai tidak bias
//!    terhadap ukuran objek. Tanpa ROI, tepi masker biasanya tidak disentuh
//!    jaringan sehingga aturan ini tidak berpengaruh.
//!
//! Objek dinomori 1..=n menurut `(y_min, x_min, luas)`; deterministik
//! (tanpa acak, urutan tetap). Semua ukuran dalam µm; sentroid memakai
//! pusat piksel `+0,5`.
//!
//! Keliling dihitung dari panjang "retakan" (sisi piksel antara isi dan
//! latar, 4-tetangga) dikali `π/4`: koreksi ini tepat untuk lingkaran besar
//! dan kurang-taksir ~21 % untuk persegi sejajar sumbu. Sirkularitas
//! `4πA/P²` dijepit ke `≤ 1`.

use ndarray::{s, Array2};
use serde::{Deserialize, Serialize};

use crate::components::{label_components, merge_small_regions};
use crate::error::ImgError;
use crate::morphology::{fill_holes, open};
use crate::watershed::split_touching;

/// Perlakuan objek yang menyentuh tepi masker.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum BorderRule {
    /// Semua objek dihitung.
    Include,
    /// Bingkai hitung tak bias: objek yang menyentuh tepi kiri atau bawah
    /// tidak dihitung (bawaan).
    #[default]
    CountingFrame,
    /// Objek yang menyentuh tepi mana pun tidak dihitung.
    ExcludeAll,
}

/// Parameter penghitungan objek (satuan µm).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CountParams {
    /// Isi lubang (lumen) sebelum menghitung (bawaan benar).
    pub fill_holes: bool,
    /// Radius pembukaan morfologi, µm; 0 = lewati (bawaan 0).
    pub open_radius_um: f64,
    /// Belah objek yang bersentuhan dengan watershed EDT (bawaan benar).
    pub split_enabled: bool,
    /// Jarak minimum antar puncak EDT, µm (bawaan 40).
    pub min_seed_distance_um: f64,
    /// Kedalaman lembah minimum agar dua puncak dianggap dua objek, µm
    /// (bawaan 8).
    pub h_um: f64,
    /// Wilayah hasil belah lebih kecil dari ini (µm²) digabung ke tetangga
    /// terluas (bawaan 200).
    pub merge_min_area_um2: f64,
    /// Objek lebih kecil dari ini (µm²) dibuang (bawaan 500).
    pub min_area_um2: f64,
    /// Aturan tepi.
    pub border: BorderRule,
}

impl Default for CountParams {
    fn default() -> Self {
        Self {
            fill_holes: true,
            open_radius_um: 0.0,
            split_enabled: true,
            min_seed_distance_um: 40.0,
            h_um: 8.0,
            merge_min_area_um2: 200.0,
            min_area_um2: 500.0,
            border: BorderRule::CountingFrame,
        }
    }
}

impl CountParams {
    /// Periksa nilai: semua jarak/luas harus terhingga dan `≥ 0`.
    pub fn validate(&self) -> Result<(), ImgError> {
        for (name, v) in [
            ("open_radius_um", self.open_radius_um),
            ("min_seed_distance_um", self.min_seed_distance_um),
            ("h_um", self.h_um),
            ("merge_min_area_um2", self.merge_min_area_um2),
            ("min_area_um2", self.min_area_um2),
        ] {
            if !(v.is_finite() && v >= 0.0) {
                return Err(ImgError::InvalidParameter(format!(
                    "{name} harus terhingga dan tidak negatif"
                )));
            }
        }
        Ok(())
    }

    /// Hash FNV-1a 64-bit stabil atas semua parameter; dicetak di laporan
    /// agar hasil dapat direproduksi (aturan keras 19).
    pub fn params_hash(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut feed = |bytes: &[u8]| {
            for &b in bytes {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        feed(b"h3d-count-v1");
        feed(&[u8::from(self.fill_holes), u8::from(self.split_enabled)]);
        for v in [
            self.open_radius_um,
            self.min_seed_distance_um,
            self.h_um,
            self.merge_min_area_um2,
            self.min_area_um2,
        ] {
            feed(&v.to_bits().to_le_bytes());
        }
        feed(&[match self.border {
            BorderRule::Include => 0u8,
            BorderRule::CountingFrame => 1,
            BorderRule::ExcludeAll => 2,
        }]);
        h
    }
}

/// Satu objek 2D. Koordinat dalam µm relatif pojok kiri-atas masker.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Object2D {
    /// Nomor objek 1..=n (urutan raster).
    pub id: u32,
    /// Luas, µm².
    pub area_um2: f64,
    /// Taksiran keliling, µm (lihat catatan modul).
    pub perimeter_um: f64,
    /// Diameter lingkaran berluas sama, µm.
    pub equiv_diameter_um: f64,
    /// Sirkularitas `4πA/P²` dalam `[0, 1]`.
    pub circularity: f64,
    /// Sentroid, µm (pusat piksel `+0,5`).
    pub centroid_um: [f64; 2],
    /// Kotak pembatas `[x0, y0, x1, y1]` dalam µm (`x1`, `y1` eksklusif).
    pub bbox_um: [f64; 4],
    /// Menyentuh tepi masker.
    pub touches_border: bool,
    /// Tidak dihitung karena aturan tepi.
    pub excluded: bool,
}

/// Hasil penghitungan satu masker.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObjectCount {
    /// Jumlah objek yang dihitung (tidak termasuk `n_excluded`).
    pub n: usize,
    /// Jumlah objek yang dikecualikan oleh aturan tepi.
    pub n_excluded: usize,
    /// Semua objek (dihitung maupun dikecualikan), urut `id`.
    pub objects: Vec<Object2D>,
    /// Ukuran piksel masker, µm.
    pub mpp: f64,
    /// Hash parameter.
    pub params_hash: u64,
}

impl ObjectCount {
    /// Luas total objek yang dihitung, µm².
    pub fn total_area_um2(&self) -> f64 {
        self.objects
            .iter()
            .filter(|o| !o.excluded)
            .map(|o| o.area_um2)
            .sum()
    }

    /// Median luas objek yang dihitung, µm² (0 bila tidak ada).
    pub fn median_area_um2(&self) -> f64 {
        let mut a: Vec<f64> = self
            .objects
            .iter()
            .filter(|o| !o.excluded)
            .map(|o| o.area_um2)
            .collect();
        if a.is_empty() {
            return 0.0;
        }
        a.sort_by(f64::total_cmp);
        let m = a.len() / 2;
        if a.len().is_multiple_of(2) {
            (a[m - 1] + a[m]) / 2.0
        } else {
            a[m]
        }
    }
}

/// Jumlah objek beserta rentang ketidakpastian dari kepekaan parameter
/// pemisah (aturan keras 18).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CountEstimate {
    /// Hasil dengan parameter yang diminta.
    pub count: ObjectCount,
    /// Jumlah terkecil di antara varian parameter.
    pub n_min: usize,
    /// Jumlah terbesar di antara varian parameter.
    pub n_max: usize,
}

/// Beri nomor objek pada `mask` (`(h, w)`, `≠ 0` = isi) dengan ukuran piksel
/// `mpp` (µm). Mengembalikan peta objek `1..=n` (0 = latar; objek yang
/// dikecualikan tetap bernomor) dan daftar objek.
pub fn label_objects(
    mask: &Array2<u8>,
    mpp: f64,
    params: &CountParams,
) -> Result<(Array2<u32>, Vec<Object2D>), ImgError> {
    params.validate()?;
    if !(mpp.is_finite() && mpp > 0.0) {
        return Err(ImgError::InvalidParameter(format!("mpp tidak sah: {mpp}")));
    }
    let (h, w) = mask.dim();
    let mut out = Array2::<u32>::zeros((h, w));
    if h == 0 || w == 0 {
        return Ok((out, Vec::new()));
    }
    let mut m = mask.mapv(|v| u8::from(v != 0));
    if params.fill_holes {
        m = fill_holes(&m);
    }
    if params.open_radius_um > 0.0 {
        m = open(&m, params.open_radius_um / mpp);
    }
    let (cc, n_cc) = label_components(&m);
    // Kotak pembatas tiap komponen kasar.
    let mut bbox = vec![[usize::MAX, usize::MAX, 0usize, 0usize]; n_cc + 1];
    for ((y, x), &c) in cc.indexed_iter() {
        if c == 0 {
            continue;
        }
        let b = &mut bbox[c as usize];
        b[0] = b[0].min(x);
        b[1] = b[1].min(y);
        b[2] = b[2].max(x + 1);
        b[3] = b[3].max(y + 1);
    }
    let merge_px = (params.merge_min_area_um2 / (mpp * mpp)).ceil() as usize;
    let min_px = (params.min_area_um2 / (mpp * mpp)).ceil() as usize;
    // Nomor sementara (urutan komponen kasar lalu sub-wilayah); dinomori
    // ulang menurut raster di akhir.
    let mut next = 0u32;
    for (c, &[x0, y0, x1, y1]) in bbox.iter().enumerate().skip(1) {
        let sub = cc
            .slice(s![y0..y1, x0..x1])
            .mapv(|v| u8::from(v as usize == c));
        let (lab, n) = if params.split_enabled {
            let (mut lab, n) = split_touching(&sub, mpp, params.min_seed_distance_um, params.h_um);
            let n = if merge_px > 1 {
                merge_small_regions(&mut lab, n, merge_px)
            } else {
                n
            };
            (lab, n)
        } else {
            (sub.mapv(u32::from), 1)
        };
        let base = next;
        next += n as u32;
        for ((y, x), &l) in lab.indexed_iter() {
            if l != 0 {
                out[[y0 + y, x0 + x]] = base + l;
            }
        }
    }
    let n_raw = next as usize;
    // Statistik per objek mentah.
    let mut area = vec![0usize; n_raw + 1];
    let mut sx = vec![0f64; n_raw + 1];
    let mut sy = vec![0f64; n_raw + 1];
    let mut crack = vec![0usize; n_raw + 1];
    let mut bb = vec![[usize::MAX, usize::MAX, 0usize, 0usize]; n_raw + 1];
    let mut touch_left_bottom = vec![false; n_raw + 1];
    let mut touch_any = vec![false; n_raw + 1];
    for ((y, x), &l) in out.indexed_iter() {
        if l == 0 {
            continue;
        }
        let i = l as usize;
        area[i] += 1;
        sx[i] += x as f64;
        sy[i] += y as f64;
        let b = &mut bb[i];
        b[0] = b[0].min(x);
        b[1] = b[1].min(y);
        b[2] = b[2].max(x + 1);
        b[3] = b[3].max(y + 1);
        // Sisi retakan: tetangga 4 yang bukan objek yang sama (tepi larik
        // dihitung sebagai latar).
        let mut edges = 0;
        if x == 0 || out[[y, x - 1]] != l {
            edges += 1;
        }
        if x + 1 == w || out[[y, x + 1]] != l {
            edges += 1;
        }
        if y == 0 || out[[y - 1, x]] != l {
            edges += 1;
        }
        if y + 1 == h || out[[y + 1, x]] != l {
            edges += 1;
        }
        crack[i] += edges;
        if x == 0 || y + 1 == h {
            touch_left_bottom[i] = true;
        }
        if x == 0 || y == 0 || x + 1 == w || y + 1 == h {
            touch_any[i] = true;
        }
    }
    // Buang objek kecil, lalu urutkan menurut raster.
    let mut keep: Vec<usize> = (1..=n_raw)
        .filter(|&i| area[i] > 0 && area[i] >= min_px.max(1))
        .collect();
    keep.sort_by_key(|&i| (bb[i][1], bb[i][0], area[i], i));
    let mut remap = vec![0u32; n_raw + 1];
    let mut objects = Vec::with_capacity(keep.len());
    for (k, &i) in keep.iter().enumerate() {
        let id = k as u32 + 1;
        remap[i] = id;
        let a_um2 = area[i] as f64 * mpp * mpp;
        let perim = crack[i] as f64 * mpp * std::f64::consts::FRAC_PI_4;
        let circ = if perim > 0.0 {
            (4.0 * std::f64::consts::PI * a_um2 / (perim * perim)).min(1.0)
        } else {
            0.0
        };
        let excluded = match params.border {
            BorderRule::Include => false,
            BorderRule::CountingFrame => touch_left_bottom[i],
            BorderRule::ExcludeAll => touch_any[i],
        };
        let a = area[i] as f64;
        objects.push(Object2D {
            id,
            area_um2: a_um2,
            perimeter_um: perim,
            equiv_diameter_um: 2.0 * (a_um2 / std::f64::consts::PI).sqrt(),
            circularity: circ,
            centroid_um: [(sx[i] / a + 0.5) * mpp, (sy[i] / a + 0.5) * mpp],
            bbox_um: [
                bb[i][0] as f64 * mpp,
                bb[i][1] as f64 * mpp,
                bb[i][2] as f64 * mpp,
                bb[i][3] as f64 * mpp,
            ],
            touches_border: touch_any[i],
            excluded,
        });
    }
    out.mapv_inplace(|l| remap[l as usize]);
    Ok((out, objects))
}

/// Hitung objek pada `mask` (lihat [`label_objects`]) tanpa menyimpan peta
/// objek.
pub fn count_objects(
    mask: &Array2<u8>,
    mpp: f64,
    params: &CountParams,
) -> Result<ObjectCount, ImgError> {
    let (_, objects) = label_objects(mask, mpp, params)?;
    let n_excluded = objects.iter().filter(|o| o.excluded).count();
    Ok(ObjectCount {
        n: objects.len() - n_excluded,
        n_excluded,
        objects,
        mpp,
        params_hash: params.params_hash(),
    })
}

/// Faktor varian parameter pemisah untuk rentang ketidakpastian:
/// `(pengali h_um, pengali min_seed_distance_um)`.
pub const UNCERTAINTY_VARIANTS: [(f64, f64); 4] =
    [(0.5, 1.0), (2.0, 1.0), (1.0, 0.75), (1.0, 1.25)];

/// Seperti [`count_objects`], ditambah rentang `n_min..=n_max` dari
/// menjalankan ulang dengan varian parameter pemisah
/// ([`UNCERTAINTY_VARIANTS`]). Tanpa pemisah, rentang sama dengan `n`.
pub fn count_objects_with_uncertainty(
    mask: &Array2<u8>,
    mpp: f64,
    params: &CountParams,
) -> Result<CountEstimate, ImgError> {
    let count = count_objects(mask, mpp, params)?;
    let (mut n_min, mut n_max) = (count.n, count.n);
    if params.split_enabled {
        for (fh, fs) in UNCERTAINTY_VARIANTS {
            let p = CountParams {
                h_um: params.h_um * fh,
                min_seed_distance_um: params.min_seed_distance_um * fs,
                ..params.clone()
            };
            let n = count_objects(mask, mpp, &p)?.n;
            n_min = n_min.min(n);
            n_max = n_max.max(n);
        }
    }
    Ok(CountEstimate {
        count,
        n_min,
        n_max,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cakram terisi berpusat `(cx, cy)` jari-jari `r` piksel.
    fn disc(m: &mut Array2<u8>, cx: f64, cy: f64, r: f64) {
        for ((y, x), v) in m.indexed_iter_mut() {
            let dx = x as f64 + 0.5 - cx;
            let dy = y as f64 + 0.5 - cy;
            if dx * dx + dy * dy <= r * r {
                *v = 1;
            }
        }
    }

    /// Cincin (kelenjar berlumen).
    fn ring(m: &mut Array2<u8>, cx: f64, cy: f64, r_out: f64, r_in: f64) {
        for ((y, x), v) in m.indexed_iter_mut() {
            let dx = x as f64 + 0.5 - cx;
            let dy = y as f64 + 0.5 - cy;
            let d2 = dx * dx + dy * dy;
            if d2 <= r_out * r_out && d2 > r_in * r_in {
                *v = 1;
            }
        }
    }

    fn params_px() -> CountParams {
        CountParams {
            min_seed_distance_um: 8.0,
            h_um: 2.0,
            merge_min_area_um2: 20.0,
            min_area_um2: 30.0,
            ..CountParams::default()
        }
    }

    #[test]
    fn ring_with_lumen_counts_once_and_is_round() {
        let mut m = Array2::<u8>::zeros((80, 80));
        ring(&mut m, 40.0, 40.0, 20.0, 10.0);
        let c = count_objects(&m, 1.0, &params_px()).unwrap();
        assert_eq!(c.n, 1);
        let o = &c.objects[0];
        let expect = std::f64::consts::PI * 400.0;
        assert!(
            (o.area_um2 - expect).abs() / expect < 0.05,
            "luas {}",
            o.area_um2
        );
        assert!(o.circularity > 0.9, "sirkularitas {}", o.circularity);
        assert!((o.centroid_um[0] - 40.0).abs() < 0.5 && (o.centroid_um[1] - 40.0).abs() < 0.5);
        assert!(!o.touches_border && !o.excluded);
        // Tanpa isi lubang, cincin tetap satu komponen tetapi luasnya kecil.
        let p2 = CountParams {
            fill_holes: false,
            ..params_px()
        };
        let c2 = count_objects(&m, 1.0, &p2).unwrap();
        assert_eq!(c2.n, 1);
        assert!(c2.objects[0].area_um2 < o.area_um2);
    }

    #[test]
    fn touching_discs_are_split() {
        let mut m = Array2::<u8>::zeros((60, 100));
        disc(&mut m, 30.0, 30.0, 15.0);
        disc(&mut m, 58.0, 30.0, 15.0);
        let no_split = CountParams {
            split_enabled: false,
            ..params_px()
        };
        assert_eq!(count_objects(&m, 1.0, &no_split).unwrap().n, 1);
        let c = count_objects(&m, 1.0, &params_px()).unwrap();
        assert_eq!(c.n, 2, "{:?}", c.objects);
        let xs: Vec<f64> = c.objects.iter().map(|o| o.centroid_um[0]).collect();
        assert!(xs[0] < 44.0 && xs[1] > 44.0, "{xs:?}");
    }

    #[test]
    fn counting_frame_excludes_left_and_bottom() {
        let mut m = Array2::<u8>::zeros((100, 100));
        disc(&mut m, 5.0, 50.0, 10.0); // menyentuh kiri
        disc(&mut m, 50.0, 95.0, 10.0); // menyentuh bawah
        disc(&mut m, 95.0, 50.0, 10.0); // menyentuh kanan
        disc(&mut m, 50.0, 50.0, 10.0); // di tengah
        let p = params_px();
        let c = count_objects(&m, 1.0, &p).unwrap();
        assert_eq!((c.n, c.n_excluded), (2, 2));
        assert_eq!(c.objects.len(), 4);
        let inc = CountParams {
            border: BorderRule::Include,
            ..p.clone()
        };
        assert_eq!(count_objects(&m, 1.0, &inc).unwrap().n, 4);
        let all = CountParams {
            border: BorderRule::ExcludeAll,
            ..p
        };
        assert_eq!(count_objects(&m, 1.0, &all).unwrap().n, 1);
    }

    #[test]
    fn small_objects_dropped_and_order_is_raster() {
        let mut m = Array2::<u8>::zeros((60, 60));
        disc(&mut m, 45.0, 15.0, 8.0);
        disc(&mut m, 15.0, 45.0, 8.0);
        m[[30, 30]] = 1; // 1 px
        let c = count_objects(&m, 2.0, &params_px()).unwrap();
        assert_eq!(c.n, 2);
        assert!(c.objects[0].centroid_um[1] < c.objects[1].centroid_um[1]);
        assert_eq!(c.objects[0].id, 1);
        assert!((c.objects[0].equiv_diameter_um - 32.0).abs() < 2.0);
    }

    #[test]
    fn deterministic_and_label_map_matches() {
        let mut m = Array2::<u8>::zeros((70, 70));
        disc(&mut m, 20.0, 20.0, 12.0);
        disc(&mut m, 42.0, 20.0, 12.0);
        ring(&mut m, 45.0, 50.0, 14.0, 6.0);
        let p = params_px();
        let a = label_objects(&m, 1.0, &p).unwrap();
        let b = label_objects(&m, 1.0, &p).unwrap();
        assert_eq!(a, b);
        let (labels, objs) = a;
        let max = labels.iter().copied().max().unwrap_or(0) as usize;
        assert_eq!(max, objs.len());
        for o in &objs {
            let n = labels.iter().filter(|&&l| l == o.id).count();
            assert_eq!(n as f64, o.area_um2);
        }
    }

    #[test]
    fn uncertainty_range_contains_n() {
        let mut m = Array2::<u8>::zeros((60, 100));
        disc(&mut m, 30.0, 30.0, 15.0);
        disc(&mut m, 58.0, 30.0, 15.0);
        let e = count_objects_with_uncertainty(&m, 1.0, &params_px()).unwrap();
        assert!(e.n_min <= e.count.n && e.count.n <= e.n_max);
        let off = CountParams {
            split_enabled: false,
            ..params_px()
        };
        let e2 = count_objects_with_uncertainty(&m, 1.0, &off).unwrap();
        assert_eq!((e2.n_min, e2.n_max), (1, 1));
    }

    #[test]
    fn invalid_inputs_are_errors_and_hash_is_sensitive() {
        let m = Array2::<u8>::zeros((4, 4));
        assert!(count_objects(&m, 0.0, &CountParams::default()).is_err());
        let bad = CountParams {
            h_um: -1.0,
            ..CountParams::default()
        };
        assert!(count_objects(&m, 1.0, &bad).is_err());
        let empty =
            count_objects(&Array2::<u8>::zeros((0, 0)), 1.0, &CountParams::default()).unwrap();
        assert_eq!(empty.n, 0);
        let a = CountParams::default().params_hash();
        let b = CountParams {
            h_um: 9.0,
            ..CountParams::default()
        }
        .params_hash();
        let c = CountParams {
            border: BorderRule::Include,
            ..CountParams::default()
        }
        .params_hash();
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_eq!(a, CountParams::default().params_hash());
    }
}
