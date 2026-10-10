//! Penyimpanan label raster per irisan di ruang U (histo3d.md 8.5.1).
//!
//! Label disimpan sebagai tile `tile_size × tile_size` byte pada resolusi
//! `mpp_label`; tile yang seluruhnya nol tidak disimpan. Semua konversi
//! µm ↔ indeks piksel memakai pusat piksel `+0,5`: piksel `x` meliputi
//! `[x·mpp, (x+1)·mpp)` sehingga indeks dari koordinat µm adalah
//! `floor(u / mpp)`, dan pusat piksel `x` berada di `(x + 0,5)·mpp`.

use std::collections::{BTreeSet, HashMap};

use h3d_core::{Affine2, LabelId};
use ndarray::Array2;

use crate::error::SegError;

/// Ukuran tile bawaan (piksel).
pub const DEFAULT_TILE_SIZE: u32 = 256;
/// Resolusi label bawaan (µm/piksel).
pub const DEFAULT_MPP_LABEL: f64 = 4.0;

/// Cara sebuah goresan menulis ke piksel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaintMode {
    /// Timpa apa pun yang ada.
    Replace,
    /// Hanya tulis pada piksel latar (0); label lain tidak ditimpa.
    OnlyBackground,
    /// Hapus: piksel yang bernilai `label` menjadi 0. Bila `label` adalah
    /// latar (0), semua label dihapus.
    Erase,
}

/// Kunci tile `(tx, ty)`.
pub type TileKey = (u32, u32);

/// Penyimpanan label satu irisan di ruang U.
#[derive(Debug, Clone)]
pub struct LabelStore {
    /// Ukuran piksel label (µm).
    pub mpp_label: f64,
    /// Ukuran slide pada `mpp_label` (piksel), `[lebar, tinggi]`.
    pub size_px: [u32; 2],
    /// Ukuran tile (piksel).
    pub tile_size: u32,
    /// Hanya tile tak kosong.
    tiles: HashMap<TileKey, Box<[u8]>>,
    /// Tile yang berubah sejak `clear_dirty`, terurut agar deterministik.
    dirty: BTreeSet<TileKey>,
    /// Jurnal undo: isi tile sebelum perubahan pertama selama perekaman.
    journal: Option<HashMap<TileKey, Option<Box<[u8]>>>>,
}

impl LabelStore {
    /// Penyimpanan kosong untuk slide berukuran `size_um` (µm) pada
    /// `mpp_label`. `mpp_label` yang tidak sah (≤ 0, NaN) diganti bawaan.
    pub fn new(mpp_label: f64, size_um: [f64; 2]) -> Self {
        let mpp = if mpp_label.is_finite() && mpp_label > 0.0 {
            mpp_label
        } else {
            DEFAULT_MPP_LABEL
        };
        let dim = |s: f64| -> u32 {
            if s.is_finite() && s > 0.0 {
                (s / mpp).ceil().min(u32::MAX as f64) as u32
            } else {
                0
            }
        };
        Self::with_size_px(mpp, [dim(size_um[0]), dim(size_um[1])], DEFAULT_TILE_SIZE)
    }

    /// Penyimpanan kosong dengan ukuran piksel dan tile eksplisit.
    pub fn with_size_px(mpp_label: f64, size_px: [u32; 2], tile_size: u32) -> Self {
        Self {
            mpp_label,
            size_px,
            tile_size: tile_size.max(1),
            tiles: HashMap::new(),
            dirty: BTreeSet::new(),
            journal: None,
        }
    }

    /// Ukuran slide dalam µm.
    pub fn size_um(&self) -> [f64; 2] {
        [
            self.size_px[0] as f64 * self.mpp_label,
            self.size_px[1] as f64 * self.mpp_label,
        ]
    }

    /// Jumlah tile per sumbu `[nx, ny]`.
    pub fn grid_size(&self) -> [u32; 2] {
        [
            self.size_px[0].div_ceil(self.tile_size),
            self.size_px[1].div_ceil(self.tile_size),
        ]
    }

    /// Jumlah elemen satu tile.
    fn tile_len(&self) -> usize {
        self.tile_size as usize * self.tile_size as usize
    }

    /// Benar bila tidak ada tile tersimpan.
    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }

    /// Jumlah tile tersimpan.
    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }

    /// Kunci semua tile tersimpan, terurut `(ty, tx)`.
    pub fn tile_keys(&self) -> Vec<TileKey> {
        let mut keys: Vec<TileKey> = self.tiles.keys().copied().collect();
        keys.sort_by_key(|&(tx, ty)| (ty, tx));
        keys
    }

    /// Isi tile `(tx, ty)` bila tersimpan.
    pub fn tile(&self, tx: u32, ty: u32) -> Option<&[u8]> {
        self.tiles.get(&(tx, ty)).map(|b| &b[..])
    }

    /// Pasang tile utuh. Tile yang seluruhnya nol dihapus. Galat bila ukuran
    /// data ≠ `tile_size²` atau tile di luar kisi.
    pub fn set_tile(&mut self, tx: u32, ty: u32, data: &[u8]) -> Result<(), SegError> {
        if data.len() != self.tile_len() {
            return Err(SegError::InvalidParameter(format!(
                "ukuran tile {} ≠ {}",
                data.len(),
                self.tile_len()
            )));
        }
        let [nx, ny] = self.grid_size();
        if tx >= nx || ty >= ny {
            return Err(SegError::InvalidParameter(format!(
                "tile ({tx}, {ty}) di luar kisi {nx}×{ny}"
            )));
        }
        let key = (tx, ty);
        self.journal_before(key);
        if data.iter().all(|&v| v == 0) {
            self.tiles.remove(&key);
        } else {
            self.tiles.insert(key, data.to_vec().into_boxed_slice());
        }
        self.dirty.insert(key);
        Ok(())
    }

    /// Hapus tile (menjadi latar).
    pub fn remove_tile(&mut self, tx: u32, ty: u32) {
        let key = (tx, ty);
        self.journal_before(key);
        if self.tiles.remove(&key).is_some() {
            self.dirty.insert(key);
        }
    }

    /// Tile yang berubah sejak `clear_dirty`, terurut.
    pub fn dirty_tiles(&self) -> Vec<TileKey> {
        self.dirty.iter().copied().collect()
    }

    /// Kosongkan daftar tile kotor (setelah disimpan).
    pub fn clear_dirty(&mut self) {
        self.dirty.clear();
    }

    /// Buang tile yang seluruhnya nol.
    pub fn prune_empty_tiles(&mut self) {
        let empty: Vec<TileKey> = self
            .tiles
            .iter()
            .filter(|(_, t)| t.iter().all(|&v| v == 0))
            .map(|(k, _)| *k)
            .collect();
        for k in empty {
            self.tiles.remove(&k);
        }
    }

    // ----------------------------------------------------------------------
    // Jurnal undo (dipakai `commands.rs`).
    // ----------------------------------------------------------------------

    /// Mulai merekam isi tile sebelum perubahan. Perekaman bersarang tidak
    /// didukung: panggilan kedua mengganti jurnal lama.
    pub(crate) fn begin_record(&mut self) {
        self.journal = Some(HashMap::new());
    }

    /// Akhiri perekaman; mengembalikan isi awal tiap tile yang tersentuh
    /// (`None` = tile belum ada).
    pub(crate) fn end_record(&mut self) -> Vec<(TileKey, Option<Box<[u8]>>)> {
        let j = self.journal.take().unwrap_or_default();
        let mut v: Vec<_> = j.into_iter().collect();
        v.sort_by_key(|&((tx, ty), _)| (ty, tx));
        v
    }

    fn journal_before(&mut self, key: TileKey) {
        if let Some(j) = self.journal.as_mut() {
            if let std::collections::hash_map::Entry::Vacant(e) = j.entry(key) {
                e.insert(self.tiles.get(&key).cloned());
            }
        }
    }

    // ----------------------------------------------------------------------
    // Baca.
    // ----------------------------------------------------------------------

    /// Nilai label di piksel `(x, y)`; 0 di luar slide.
    pub fn get(&self, x_px: u32, y_px: u32) -> u8 {
        if x_px >= self.size_px[0] || y_px >= self.size_px[1] {
            return 0;
        }
        let ts = self.tile_size;
        match self.tiles.get(&(x_px / ts, y_px / ts)) {
            Some(t) => t[((y_px % ts) * ts + (x_px % ts)) as usize],
            None => 0,
        }
    }

    /// Indeks piksel dari koordinat µm (`floor(u / mpp)`); `None` di luar slide.
    pub fn px_of_um(&self, u: [f64; 2]) -> Option<[u32; 2]> {
        let x = (u[0] / self.mpp_label).floor();
        let y = (u[1] / self.mpp_label).floor();
        if !(x >= 0.0 && y >= 0.0) || x >= self.size_px[0] as f64 || y >= self.size_px[1] as f64 {
            return None;
        }
        Some([x as u32, y as u32])
    }

    /// Pusat piksel `(x, y)` dalam µm.
    pub fn um_of_px(&self, x_px: u32, y_px: u32) -> [f64; 2] {
        [
            (x_px as f64 + 0.5) * self.mpp_label,
            (y_px as f64 + 0.5) * self.mpp_label,
        ]
    }

    /// Nilai label pada koordinat µm; 0 di luar slide.
    pub fn get_um(&self, u: [f64; 2]) -> u8 {
        match self.px_of_um(u) {
            Some([x, y]) => self.get(x, y),
            None => 0,
        }
    }

    /// Luas label dalam piksel.
    pub fn area_px(&self, label: LabelId) -> u64 {
        if label.is_background() {
            let total = self.size_px[0] as u64 * self.size_px[1] as u64;
            let non_bg: u64 = self
                .tiles
                .values()
                .map(|t| t.iter().filter(|&&v| v != 0).count() as u64)
                .sum();
            return total.saturating_sub(non_bg);
        }
        self.tiles
            .values()
            .map(|t| t.iter().filter(|&&v| v == label.0).count() as u64)
            .sum()
    }

    /// Luas label dalam µm².
    pub fn area_um2(&self, label: LabelId) -> f64 {
        self.area_px(label) as f64 * self.mpp_label * self.mpp_label
    }

    /// Kotak pembatas label `[x0, y0, x1, y1]` (piksel, `x1`/`y1` eksklusif);
    /// `None` bila label tidak ada.
    pub fn bbox_px(&self, label: LabelId) -> Option<[u32; 4]> {
        let ts = self.tile_size;
        let mut bb: Option<[u32; 4]> = None;
        for (&(tx, ty), t) in &self.tiles {
            for (i, &v) in t.iter().enumerate() {
                if v != label.0 {
                    continue;
                }
                let x = tx * ts + (i as u32 % ts);
                let y = ty * ts + (i as u32 / ts);
                bb = Some(match bb {
                    None => [x, y, x + 1, y + 1],
                    Some([x0, y0, x1, y1]) => [x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1)],
                });
            }
        }
        bb
    }

    /// Kotak pembatas label dalam µm `[x0, y0, x1, y1]`.
    pub fn bbox_um(&self, label: LabelId) -> Option<[f64; 4]> {
        self.bbox_px(label).map(|[x0, y0, x1, y1]| {
            [
                x0 as f64 * self.mpp_label,
                y0 as f64 * self.mpp_label,
                x1 as f64 * self.mpp_label,
                y1 as f64 * self.mpp_label,
            ]
        })
    }

    /// Label (bukan latar) yang muncul, terurut naik.
    pub fn labels_present(&self) -> Vec<LabelId> {
        let mut seen = [false; 256];
        for t in self.tiles.values() {
            for &v in t.iter() {
                seen[v as usize] = true;
            }
        }
        (1..=255u8)
            .filter(|&v| seen[v as usize])
            .map(LabelId)
            .collect()
    }

    /// Masker biner label pada wilayah `[x0, x0+w) × [y0, y0+h)` piksel,
    /// bentuk `(h, w)`; di luar slide bernilai 0.
    pub fn mask_region(&self, label: LabelId, x0: u32, y0: u32, w: u32, h: u32) -> Array2<u8> {
        let mut out = Array2::<u8>::zeros((h as usize, w as usize));
        if w == 0 || h == 0 {
            return out;
        }
        let ts = self.tile_size;
        let x1 = x0.saturating_add(w).min(self.size_px[0]);
        let y1 = y0.saturating_add(h).min(self.size_px[1]);
        if x0 >= x1 || y0 >= y1 {
            return out;
        }
        for ty in (y0 / ts)..=((y1 - 1) / ts) {
            for tx in (x0 / ts)..=((x1 - 1) / ts) {
                let Some(t) = self.tiles.get(&(tx, ty)) else {
                    continue;
                };
                let px0 = (tx * ts).max(x0);
                let px1 = ((tx + 1) * ts).min(x1);
                let py0 = (ty * ts).max(y0);
                let py1 = ((ty + 1) * ts).min(y1);
                for y in py0..py1 {
                    let row = ((y - ty * ts) * ts) as usize;
                    for x in px0..px1 {
                        if t[row + (x - tx * ts) as usize] == label.0 {
                            out[[(y - y0) as usize, (x - x0) as usize]] = 1;
                        }
                    }
                }
            }
        }
        out
    }

    /// Masker biner satu label untuk seluruh slide pada `mpp_label`
    /// (bentuk `(h, w)`). Untuk wilayah terbatas pakai `mask_region`.
    pub fn to_array(&self, label: LabelId) -> Array2<u8> {
        self.mask_region(label, 0, 0, self.size_px[0], self.size_px[1])
    }

    // ----------------------------------------------------------------------
    // Tulis.
    // ----------------------------------------------------------------------

    /// Primitif tulis: untuk tiap piksel di wilayah `[x0, x1) × [y0, y1)`
    /// (sudah dijepit ke slide) yang memenuhi `pred(x, y)`, terapkan `mode`.
    /// Bekerja per tile. Mengembalikan kunci tile yang benar-benar berubah.
    #[allow(clippy::too_many_arguments)]
    fn paint_region_with<F: Fn(u32, u32) -> bool>(
        &mut self,
        x0: u32,
        y0: u32,
        x1: u32,
        y1: u32,
        pred: F,
        label: LabelId,
        mode: PaintMode,
    ) -> Vec<TileKey> {
        let x1 = x1.min(self.size_px[0]);
        let y1 = y1.min(self.size_px[1]);
        if x0 >= x1 || y0 >= y1 {
            return Vec::new();
        }
        let ts = self.tile_size;
        let tile_len = self.tile_len();
        let mut changed_keys = Vec::new();
        for ty in (y0 / ts)..=((y1 - 1) / ts) {
            for tx in (x0 / ts)..=((x1 - 1) / ts) {
                let key = (tx, ty);
                let px0 = (tx * ts).max(x0);
                let px1 = ((tx + 1) * ts).min(x1);
                let py0 = (ty * ts).max(y0);
                let py1 = ((ty + 1) * ts).min(y1);
                let existed = self.tiles.contains_key(&key);
                // Mode selain Replace/OnlyBackground tidak bisa mengubah tile
                // kosong; hindari alokasi sia-sia.
                if !existed && mode == PaintMode::Erase {
                    continue;
                }
                let mut buf = match self.tiles.remove(&key) {
                    Some(b) => b,
                    None => vec![0u8; tile_len].into_boxed_slice(),
                };
                let mut changed = false;
                let mut any_nonzero = false;
                for y in py0..py1 {
                    let row = ((y - ty * ts) * ts) as usize;
                    for x in px0..px1 {
                        if !pred(x, y) {
                            continue;
                        }
                        let i = row + (x - tx * ts) as usize;
                        let cur = buf[i];
                        let new = match mode {
                            PaintMode::Replace => label.0,
                            PaintMode::OnlyBackground => {
                                if cur == 0 {
                                    label.0
                                } else {
                                    cur
                                }
                            }
                            PaintMode::Erase => {
                                if label.is_background() || cur == label.0 {
                                    0
                                } else {
                                    cur
                                }
                            }
                        };
                        if new != cur {
                            buf[i] = new;
                            changed = true;
                        }
                    }
                }
                if changed {
                    any_nonzero = buf.iter().any(|&v| v != 0);
                    self.dirty.insert(key);
                    changed_keys.push(key);
                } else if existed {
                    any_nonzero = true;
                }
                if any_nonzero {
                    self.tiles.insert(key, buf);
                }
            }
        }
        changed_keys.sort_by_key(|&(tx, ty)| (ty, tx));
        changed_keys
    }

    /// Rekam isi awal semua tile yang mungkin tersentuh wilayah ini (hanya
    /// saat perekaman undo aktif). Dipanggil sebelum menulis.
    fn snapshot_for_journal(&mut self, x0: u32, y0: u32, x1: u32, y1: u32) {
        if self.journal.is_none() {
            return;
        }
        let x1 = x1.min(self.size_px[0]);
        let y1 = y1.min(self.size_px[1]);
        if x0 >= x1 || y0 >= y1 {
            return;
        }
        let ts = self.tile_size;
        for ty in (y0 / ts)..=((y1 - 1) / ts) {
            for tx in (x0 / ts)..=((x1 - 1) / ts) {
                self.journal_before((tx, ty));
            }
        }
    }

    /// Cat cakram berpusat `center_um` berjari-jari `radius_um`. Piksel ikut
    /// bila pusatnya berjarak ≤ `radius_um`. Mengembalikan tile yang berubah.
    pub fn paint_disc(
        &mut self,
        center_um: [f64; 2],
        radius_um: f64,
        label: LabelId,
        mode: PaintMode,
    ) -> Vec<TileKey> {
        if !(radius_um.is_finite()
            && radius_um >= 0.0
            && center_um[0].is_finite()
            && center_um[1].is_finite())
        {
            return Vec::new();
        }
        let mpp = self.mpp_label;
        let (cx, cy) = (center_um[0] / mpp, center_um[1] / mpp);
        let r = radius_um / mpp;
        let x0 = (cx - r).floor().max(0.0);
        let y0 = (cy - r).floor().max(0.0);
        let x1 = (cx + r).ceil() + 1.0;
        let y1 = (cy + r).ceil() + 1.0;
        if x1 <= 0.0 || y1 <= 0.0 || x0 >= self.size_px[0] as f64 || y0 >= self.size_px[1] as f64 {
            return Vec::new();
        }
        let (x0, y0) = (x0 as u32, y0 as u32);
        let x1 = x1.min(self.size_px[0] as f64) as u32;
        let y1 = y1.min(self.size_px[1] as f64) as u32;
        let r2 = r * r;
        self.snapshot_for_journal(x0, y0, x1, y1);
        self.paint_region_with(
            x0,
            y0,
            x1,
            y1,
            |x, y| {
                let dx = x as f64 + 0.5 - cx;
                let dy = y as f64 + 0.5 - cy;
                dx * dx + dy * dy <= r2
            },
            label,
            mode,
        )
    }

    /// Goresan kuas: cakram di sepanjang polyline dengan jarak antar cakram
    /// ≤ `radius_um / 2`. Satu titik = satu cakram.
    pub fn paint_stroke(
        &mut self,
        points_um: &[[f64; 2]],
        radius_um: f64,
        label: LabelId,
        mode: PaintMode,
    ) -> Vec<TileKey> {
        let mut keys = BTreeSet::new();
        let Some(first) = points_um.first() else {
            return Vec::new();
        };
        keys.extend(self.paint_disc(*first, radius_um, label, mode));
        let step = (radius_um / 2.0).max(self.mpp_label * 0.5);
        for seg in points_um.windows(2) {
            let (a, b) = (seg[0], seg[1]);
            let dx = b[0] - a[0];
            let dy = b[1] - a[1];
            let len = (dx * dx + dy * dy).sqrt();
            if !len.is_finite() {
                continue;
            }
            let n = (len / step).ceil().max(1.0) as usize;
            for i in 1..=n {
                let t = i as f64 / n as f64;
                keys.extend(self.paint_disc(
                    [a[0] + dx * t, a[1] + dy * t],
                    radius_um,
                    label,
                    mode,
                ));
            }
        }
        let mut v: Vec<TileKey> = keys.into_iter().collect();
        v.sort_by_key(|&(tx, ty)| (ty, tx));
        v
    }

    /// Isi poligon tertutup (aturan genap-ganjil, pusat piksel). Titik dalam µm.
    pub fn fill_polygon(
        &mut self,
        points_um: &[[f64; 2]],
        label: LabelId,
        mode: PaintMode,
    ) -> Vec<TileKey> {
        self.fill_rings(std::slice::from_ref(&points_um), label, mode)
    }

    /// Isi beberapa cincin sekaligus dengan aturan genap-ganjil (cincin dalam
    /// menjadi lubang). Titik dalam µm.
    pub fn fill_rings(
        &mut self,
        rings: &[&[[f64; 2]]],
        label: LabelId,
        mode: PaintMode,
    ) -> Vec<TileKey> {
        let mpp = self.mpp_label;
        // Kumpulkan sisi dalam piksel.
        let mut edges: Vec<[f64; 4]> = Vec::new();
        let (mut minx, mut miny, mut maxx, mut maxy) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for ring in rings {
            let pts: Vec<[f64; 2]> = ring
                .iter()
                .filter(|p| p[0].is_finite() && p[1].is_finite())
                .map(|p| [p[0] / mpp, p[1] / mpp])
                .collect();
            if pts.len() < 3 {
                continue;
            }
            for i in 0..pts.len() {
                let a = pts[i];
                let b = pts[(i + 1) % pts.len()];
                minx = minx.min(a[0]);
                maxx = maxx.max(a[0]);
                miny = miny.min(a[1]);
                maxy = maxy.max(a[1]);
                if a[1] != b[1] {
                    edges.push([a[0], a[1], b[0], b[1]]);
                }
            }
        }
        if edges.is_empty() {
            return Vec::new();
        }
        let x0 = minx.floor().max(0.0);
        let y0 = miny.floor().max(0.0);
        let x1 = (maxx.ceil() + 1.0).min(self.size_px[0] as f64);
        let y1 = (maxy.ceil() + 1.0).min(self.size_px[1] as f64);
        if x0 >= x1 || y0 >= y1 {
            return Vec::new();
        }
        let (x0, y0, x1, y1) = (x0 as u32, y0 as u32, x1 as u32, y1 as u32);
        let w = (x1 - x0) as usize;
        let h = (y1 - y0) as usize;
        // Raster genap-ganjil per baris pada pusat baris `y + 0,5`.
        let mut mask = vec![0u8; w * h];
        let mut xs: Vec<f64> = Vec::new();
        for j in 0..h {
            let yc = (y0 + j as u32) as f64 + 0.5;
            xs.clear();
            for e in &edges {
                let (ax, ay, bx, by) = (e[0], e[1], e[2], e[3]);
                let (top, bot) = if ay < by { (ay, by) } else { (by, ay) };
                // Setengah terbuka [top, bot) agar simpul tidak dihitung dua kali.
                if yc >= top && yc < bot {
                    xs.push(ax + (yc - ay) * (bx - ax) / (by - ay));
                }
            }
            if xs.len() < 2 {
                continue;
            }
            xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let row = &mut mask[j * w..(j + 1) * w];
            for pair in xs.chunks_exact(2) {
                // Piksel ikut bila pusatnya x + 0,5 ∈ [xa, xb).
                let xa = (pair[0] - 0.5).ceil().max(x0 as f64);
                let xb = (pair[1] - 0.5).ceil().min(x1 as f64);
                if xa >= xb {
                    continue;
                }
                let ia = (xa as u32 - x0) as usize;
                let ib = (xb as u32 - x0) as usize;
                row[ia..ib].fill(1);
            }
        }
        self.snapshot_for_journal(x0, y0, x1, y1);
        self.paint_region_with(
            x0,
            y0,
            x1,
            y1,
            |x, y| mask[(y - y0) as usize * w + (x - x0) as usize] != 0,
            label,
            mode,
        )
    }

    /// Rasterkan masker biner (`!= 0`) yang pojok kiri-atasnya berada di
    /// piksel `origin_px` (boleh negatif; bagian di luar slide dipotong).
    pub fn rasterize_mask(
        &mut self,
        mask: &Array2<u8>,
        origin_px: [i64; 2],
        label: LabelId,
        mode: PaintMode,
    ) -> Vec<TileKey> {
        let (h, w) = mask.dim();
        if h == 0 || w == 0 {
            return Vec::new();
        }
        let x0 = origin_px[0].max(0);
        let y0 = origin_px[1].max(0);
        let x1 = (origin_px[0] + w as i64).min(self.size_px[0] as i64);
        let y1 = (origin_px[1] + h as i64).min(self.size_px[1] as i64);
        if x0 >= x1 || y0 >= y1 {
            return Vec::new();
        }
        let (ox, oy) = (origin_px[0], origin_px[1]);
        let (x0, y0, x1, y1) = (x0 as u32, y0 as u32, x1 as u32, y1 as u32);
        self.snapshot_for_journal(x0, y0, x1, y1);
        self.paint_region_with(
            x0,
            y0,
            x1,
            y1,
            |x, y| mask[[(y as i64 - oy) as usize, (x as i64 - ox) as usize]] != 0,
            label,
            mode,
        )
    }

    /// Isi lubang: komponen latar (0) yang tidak menyentuh tepi kotak
    /// pembatas label menjadi `label`. Piksel label lain tidak diubah.
    pub fn fill_holes(&mut self, label: LabelId) -> Vec<TileKey> {
        let Some([x0, y0, x1, y1]) = self.bbox_px(label) else {
            return Vec::new();
        };
        let region = self.mask_region(label, x0, y0, x1 - x0, y1 - y0);
        let filled = h3d_imgproc::morphology::fill_holes(&region);
        let mut holes = filled;
        for (hv, &rv) in holes.iter_mut().zip(region.iter()) {
            if rv != 0 {
                *hv = 0;
            }
        }
        self.rasterize_mask(
            &holes,
            [x0 as i64, y0 as i64],
            label,
            PaintMode::OnlyBackground,
        )
    }

    /// Salin label `labels` dari `other` ke penyimpanan ini ("Salin ke irisan
    /// berikut"). `other_to_self` memetakan **U_other (µm) → U_self (µm)**;
    /// resampling mundur: tiap pusat piksel tujuan dipetakan dengan
    /// `other_to_self⁻¹` lalu dibaca lewat `other.get_um`. Galat bila affine
    /// singular.
    pub fn copy_from(
        &mut self,
        other: &LabelStore,
        other_to_self: &Affine2,
        labels: &[LabelId],
        mode: PaintMode,
    ) -> Result<Vec<TileKey>, SegError> {
        let inv = other_to_self
            .inverse()
            .ok_or_else(|| SegError::InvalidParameter("affine singular".to_string()))?;
        let mut wanted = [false; 256];
        for l in labels {
            if !l.is_background() {
                wanted[l.0 as usize] = true;
            }
        }
        // Wilayah tujuan: gabungan kotak pembatas sumber yang ditransformasi.
        let mut dst_bb: Option<[f64; 4]> = None;
        for l in labels {
            let Some([ux0, uy0, ux1, uy1]) = other.bbox_um(*l) else {
                continue;
            };
            for c in [[ux0, uy0], [ux1, uy0], [ux0, uy1], [ux1, uy1]] {
                let p = other_to_self.apply(c);
                dst_bb = Some(match dst_bb {
                    None => [p[0], p[1], p[0], p[1]],
                    Some([a, b, c2, d]) => [a.min(p[0]), b.min(p[1]), c2.max(p[0]), d.max(p[1])],
                });
            }
        }
        let Some([bx0, by0, bx1, by1]) = dst_bb else {
            return Ok(Vec::new());
        };
        let mpp = self.mpp_label;
        let x0 = (bx0 / mpp).floor().max(0.0);
        let y0 = (by0 / mpp).floor().max(0.0);
        let x1 = ((bx1 / mpp).ceil() + 1.0).min(self.size_px[0] as f64);
        let y1 = ((by1 / mpp).ceil() + 1.0).min(self.size_px[1] as f64);
        if !(x0 < x1 && y0 < y1) {
            return Ok(Vec::new());
        }
        let (x0, y0, x1, y1) = (x0 as u32, y0 as u32, x1 as u32, y1 as u32);
        let w = (x1 - x0) as usize;
        let h = (y1 - y0) as usize;
        // Sampel nilai sumber untuk seluruh wilayah tujuan sekali.
        let mut src = vec![0u8; w * h];
        for j in 0..h {
            for i in 0..w {
                let u_self = [
                    (x0 as f64 + i as f64 + 0.5) * mpp,
                    (y0 as f64 + j as f64 + 0.5) * mpp,
                ];
                let v = other.get_um(inv.apply(u_self));
                if wanted[v as usize] {
                    src[j * w + i] = v;
                }
            }
        }
        let mut keys = BTreeSet::new();
        for l in labels {
            if l.is_background() {
                continue;
            }
            let lv = l.0;
            self.snapshot_for_journal(x0, y0, x1, y1);
            keys.extend(self.paint_region_with(
                x0,
                y0,
                x1,
                y1,
                |x, y| src[(y - y0) as usize * w + (x - x0) as usize] == lv,
                *l,
                mode,
            ));
        }
        let mut v: Vec<TileKey> = keys.into_iter().collect();
        v.sort_by_key(|&(tx, ty)| (ty, tx));
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> LabelStore {
        LabelStore::new(4.0, [4096.0, 2048.0])
    }

    #[test]
    fn new_sizes_px() {
        let s = store();
        assert_eq!(s.size_px, [1024, 512]);
        assert_eq!(s.grid_size(), [4, 2]);
        assert!(s.is_empty());
    }

    #[test]
    fn get_um_uses_floor() {
        let mut s = store();
        s.paint_disc([10.0, 10.0], 0.1, LabelId(1), PaintMode::Replace);
        // Pusat piksel (2,2) = (10, 10) µm.
        assert_eq!(s.get(2, 2), 1);
        assert_eq!(s.get_um([8.0, 8.0]), 1);
        assert_eq!(s.get_um([11.99, 11.99]), 1);
        assert_eq!(s.get_um([12.0, 12.0]), 0);
        assert_eq!(s.get_um([-1.0, 5.0]), 0);
    }

    #[test]
    fn disc_area_within_two_percent() {
        for r_px in [10.0f64, 17.5, 40.0, 100.0] {
            let mut s = LabelStore::new(1.0, [512.0, 512.0]);
            s.paint_disc([256.3, 255.7], r_px, LabelId(7), PaintMode::Replace);
            let area = s.area_px(LabelId(7)) as f64;
            let expect = std::f64::consts::PI * r_px * r_px;
            let err = (area - expect).abs() / expect;
            assert!(err < 0.02, "r={r_px}: luas {area} vs {expect} ({err:.4})");
        }
    }

    #[test]
    fn square_polygon_exact_area() {
        let mut s = store();
        // Persegi 100×50 piksel = 400×200 µm mulai dari piksel (10, 20).
        let pts = [[40.0, 80.0], [440.0, 80.0], [440.0, 280.0], [40.0, 280.0]];
        s.fill_polygon(&pts, LabelId(2), PaintMode::Replace);
        assert_eq!(s.area_px(LabelId(2)), 100 * 50);
        assert_eq!(s.bbox_px(LabelId(2)), Some([10, 20, 110, 70]));
        assert_eq!(s.get(10, 20), 2);
        assert_eq!(s.get(109, 69), 2);
        assert_eq!(s.get(110, 69), 0);
    }

    #[test]
    fn polygon_with_hole_even_odd() {
        let mut s = store();
        let outer: &[[f64; 2]] = &[[0.0, 0.0], [400.0, 0.0], [400.0, 400.0], [0.0, 400.0]];
        let inner: &[[f64; 2]] = &[
            [100.0, 100.0],
            [300.0, 100.0],
            [300.0, 300.0],
            [100.0, 300.0],
        ];
        s.fill_rings(&[outer, inner], LabelId(1), PaintMode::Replace);
        assert_eq!(s.area_px(LabelId(1)), 100 * 100 - 50 * 50);
        // Isi lubang mengembalikan luas penuh.
        s.fill_holes(LabelId(1));
        assert_eq!(s.area_px(LabelId(1)), 100 * 100);
    }

    #[test]
    fn modes_and_pruning() {
        let mut s = store();
        s.paint_disc([100.0, 100.0], 40.0, LabelId(1), PaintMode::Replace);
        let a1 = s.area_px(LabelId(1));
        s.paint_disc([100.0, 100.0], 40.0, LabelId(2), PaintMode::OnlyBackground);
        assert_eq!(s.area_px(LabelId(2)), 0);
        s.paint_disc([160.0, 100.0], 40.0, LabelId(2), PaintMode::OnlyBackground);
        assert_eq!(s.area_px(LabelId(1)), a1);
        assert!(s.area_px(LabelId(2)) > 0);
        assert_eq!(s.labels_present(), vec![LabelId(1), LabelId(2)]);
        // Hapus label 1 saja.
        s.paint_disc([130.0, 100.0], 200.0, LabelId(1), PaintMode::Erase);
        assert_eq!(s.area_px(LabelId(1)), 0);
        assert!(s.area_px(LabelId(2)) > 0);
        // Hapus semua → tile dibuang.
        s.paint_disc([130.0, 100.0], 200.0, LabelId::BACKGROUND, PaintMode::Erase);
        assert!(s.is_empty());
        assert!(!s.dirty_tiles().is_empty());
        s.clear_dirty();
        assert!(s.dirty_tiles().is_empty());
    }

    #[test]
    fn stroke_covers_segment() {
        let mut s = store();
        s.paint_stroke(
            &[[100.0, 100.0], [900.0, 100.0]],
            20.0,
            LabelId(1),
            PaintMode::Replace,
        );
        for x_um in (100..=900).step_by(50) {
            assert_eq!(s.get_um([x_um as f64, 100.0]), 1, "x={x_um}");
        }
        assert_eq!(s.get_um([500.0, 160.0]), 0);
    }

    #[test]
    fn copy_from_translation() {
        let mut a = store();
        a.fill_polygon(
            &[
                [400.0, 400.0],
                [800.0, 400.0],
                [800.0, 800.0],
                [400.0, 800.0],
            ],
            LabelId(3),
            PaintMode::Replace,
        );
        let mut b = store();
        // U_a → U_b: geser +200 µm di x, +100 µm di y.
        let t = Affine2::translation(200.0, 100.0);
        b.copy_from(&a, &t, &[LabelId(3)], PaintMode::Replace)
            .unwrap();
        assert_eq!(b.area_px(LabelId(3)), a.area_px(LabelId(3)));
        assert_eq!(b.bbox_um(LabelId(3)), Some([600.0, 500.0, 1000.0, 900.0]));
    }

    #[test]
    fn mask_region_and_rasterize_roundtrip() {
        let mut s = store();
        s.paint_disc([300.0, 300.0], 100.0, LabelId(5), PaintMode::Replace);
        let full = s.to_array(LabelId(5));
        let mut t = store();
        t.rasterize_mask(&full, [0, 0], LabelId(5), PaintMode::Replace);
        assert_eq!(t.to_array(LabelId(5)), full);
        let [x0, y0, x1, y1] = s.bbox_px(LabelId(5)).unwrap();
        let reg = s.mask_region(LabelId(5), x0, y0, x1 - x0, y1 - y0);
        assert_eq!(
            reg.iter().filter(|&&v| v != 0).count() as u64,
            s.area_px(LabelId(5))
        );
    }

    #[test]
    fn set_tile_validates() {
        let mut s = store();
        assert!(s.set_tile(0, 0, &[1u8; 10]).is_err());
        assert!(s.set_tile(9, 9, &vec![1u8; 256 * 256]).is_err());
        assert!(s.set_tile(1, 1, &vec![0u8; 256 * 256]).is_ok());
        assert!(s.is_empty());
        assert!(s.set_tile(1, 1, &vec![4u8; 256 * 256]).is_ok());
        assert_eq!(s.tile_count(), 1);
        assert_eq!(s.area_px(LabelId(4)), 256 * 256);
    }
}
