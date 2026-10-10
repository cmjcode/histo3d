//! Penyimpanan atribut skalar per irisan (histo3d.md 8.12.5) dan format
//! `.atr` (8.12.9).
//!
//! Nilai disimpan sebagai `u16` terkuantisasi: `nilai = raw × scale + offset`,
//! `raw ∈ 0..=65534`; `u16::MAX` = `NODATA`. Tile `64 × 64` pada `mpp_attr`
//! (bawaan 16 µm). Tile yang seluruhnya NODATA tidak disimpan.
//!
//! Format `.atr` (little-endian): header `.lbl` 28 byte dengan magic `"H3DA"`
//! (tile_size, mpp_attr, w, h, n_tile), lalu `scale: f32`, `offset: f32`,
//! `attr_id: u16` (total header 38 byte), lalu indeks `{tx, ty, offset, len}`
//! terurut `(ty, tx)`, lalu payload: tiap tile = lz4_flex (size-prepended)
//! dari `u16 × tile_size²` little-endian (`2·tile_size²` byte).

use std::collections::{BTreeSet, HashMap};

use ndarray::Array2;

use crate::error::SegError;
use crate::label_store::TileKey;
use crate::lbl_format::{
    decompress_tile, read_index, write_index_and_payload, Cursor, TileHeader, LBL_HEADER_LEN,
};

/// Penanda "tidak ada data".
pub const NODATA: u16 = u16::MAX;
/// Nilai raw maksimum yang sah.
pub const MAX_RAW: u16 = u16::MAX - 1;
/// Ukuran tile bawaan.
pub const DEFAULT_ATTR_TILE_SIZE: u32 = 64;
/// Resolusi atribut bawaan (µm/piksel).
pub const DEFAULT_MPP_ATTR: f64 = 16.0;
/// Magic berkas atribut.
pub const ATR_MAGIC: &[u8; 4] = b"H3DA";
/// Panjang header `.atr` (header `.lbl` + scale + offset + attr_id).
pub const ATR_HEADER_LEN: usize = LBL_HEADER_LEN + 4 + 4 + 2;

/// Penyimpanan atribut skalar terkuantisasi di ruang U.
#[derive(Debug, Clone)]
pub struct AttrStore {
    /// Ukuran piksel atribut (µm).
    pub mpp_attr: f64,
    /// Ukuran slide pada `mpp_attr` (piksel), `[lebar, tinggi]`.
    pub size_px: [u32; 2],
    /// Ukuran tile (piksel).
    pub tile_size: u32,
    /// Faktor kuantisasi.
    pub scale: f32,
    /// Ofset kuantisasi.
    pub offset: f32,
    tiles: HashMap<TileKey, Box<[u16]>>,
    /// Bobot (jumlah sel / piksel sumber) per tile, opsional; dipakai
    /// `downsample2`. Tidak ikut diserialisasi.
    weights: Option<HashMap<TileKey, Box<[u16]>>>,
    dirty: BTreeSet<TileKey>,
}

impl AttrStore {
    /// Penyimpanan kosong untuk slide `size_um` pada `mpp_attr` dengan
    /// kuantisasi `scale`/`offset`. `scale` ≤ 0 atau tidak hingga → 1.
    pub fn new(mpp_attr: f64, size_um: [f64; 2], scale: f32, offset: f32) -> Self {
        let mpp = if mpp_attr.is_finite() && mpp_attr > 0.0 {
            mpp_attr
        } else {
            DEFAULT_MPP_ATTR
        };
        let dim = |s: f64| -> u32 {
            if s.is_finite() && s > 0.0 {
                (s / mpp).ceil().min(u32::MAX as f64) as u32
            } else {
                0
            }
        };
        Self::with_size_px(
            mpp,
            [dim(size_um[0]), dim(size_um[1])],
            DEFAULT_ATTR_TILE_SIZE,
            scale,
            offset,
        )
    }

    /// Penyimpanan kosong dengan ukuran piksel dan tile eksplisit.
    pub fn with_size_px(
        mpp_attr: f64,
        size_px: [u32; 2],
        tile_size: u32,
        scale: f32,
        offset: f32,
    ) -> Self {
        let scale = if scale.is_finite() && scale > 0.0 {
            scale
        } else {
            1.0
        };
        let offset = if offset.is_finite() { offset } else { 0.0 };
        Self {
            mpp_attr,
            size_px,
            tile_size: tile_size.max(1),
            scale,
            offset,
            tiles: HashMap::new(),
            weights: None,
            dirty: BTreeSet::new(),
        }
    }

    /// Kuantisasi `scale`/`offset` yang meliputi rentang `[lo, hi]` penuh.
    pub fn quantization_for_range(lo: f32, hi: f32) -> (f32, f32) {
        let span = (hi - lo).max(f32::MIN_POSITIVE);
        (span / MAX_RAW as f32, lo)
    }

    fn tile_len(&self) -> usize {
        self.tile_size as usize * self.tile_size as usize
    }

    /// Jumlah tile per sumbu.
    pub fn grid_size(&self) -> [u32; 2] {
        [
            self.size_px[0].div_ceil(self.tile_size),
            self.size_px[1].div_ceil(self.tile_size),
        ]
    }

    /// Benar bila tidak ada tile.
    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }

    /// Kunci tile tersimpan, terurut `(ty, tx)`.
    pub fn tile_keys(&self) -> Vec<TileKey> {
        let mut k: Vec<TileKey> = self.tiles.keys().copied().collect();
        k.sort_by_key(|&(tx, ty)| (ty, tx));
        k
    }

    /// Isi tile raw bila ada.
    pub fn tile(&self, tx: u32, ty: u32) -> Option<&[u16]> {
        self.tiles.get(&(tx, ty)).map(|b| &b[..])
    }

    /// Bobot tile bila ada.
    pub fn tile_weights(&self, tx: u32, ty: u32) -> Option<&[u16]> {
        self.weights
            .as_ref()
            .and_then(|w| w.get(&(tx, ty)))
            .map(|b| &b[..])
    }

    /// Tile kotor sejak `clear_dirty`.
    pub fn dirty_tiles(&self) -> Vec<TileKey> {
        self.dirty.iter().copied().collect()
    }

    /// Kosongkan daftar kotor.
    pub fn clear_dirty(&mut self) {
        self.dirty.clear();
    }

    /// Kuantisasi nilai ke raw (dijepit `0..=65534`); NaN → NODATA.
    pub fn quantize(&self, value: f32) -> u16 {
        if !value.is_finite() {
            return NODATA;
        }
        let q = ((value - self.offset) / self.scale).round();
        q.clamp(0.0, MAX_RAW as f32) as u16
    }

    /// Dekuantisasi raw; `None` untuk NODATA.
    pub fn dequantize(&self, raw: u16) -> Option<f32> {
        if raw == NODATA {
            None
        } else {
            Some(raw as f32 * self.scale + self.offset)
        }
    }

    /// Nilai raw di `(x, y)`; NODATA di luar slide atau tile kosong.
    pub fn get_raw(&self, x_px: u32, y_px: u32) -> u16 {
        if x_px >= self.size_px[0] || y_px >= self.size_px[1] {
            return NODATA;
        }
        let ts = self.tile_size;
        match self.tiles.get(&(x_px / ts, y_px / ts)) {
            Some(t) => t[((y_px % ts) * ts + (x_px % ts)) as usize],
            None => NODATA,
        }
    }

    /// Nilai di `(x, y)`; `None` bila NODATA.
    pub fn get(&self, x_px: u32, y_px: u32) -> Option<f32> {
        self.dequantize(self.get_raw(x_px, y_px))
    }

    /// Bobot di `(x, y)` (0 bila tidak ada).
    pub fn weight(&self, x_px: u32, y_px: u32) -> u16 {
        if x_px >= self.size_px[0] || y_px >= self.size_px[1] {
            return 0;
        }
        let ts = self.tile_size;
        self.weights
            .as_ref()
            .and_then(|w| w.get(&(x_px / ts, y_px / ts)))
            .map(|t| t[((y_px % ts) * ts + (x_px % ts)) as usize])
            .unwrap_or(0)
    }

    /// Set nilai raw di `(x, y)`; di luar slide diabaikan.
    pub fn set_raw(&mut self, x_px: u32, y_px: u32, raw: u16) {
        if x_px >= self.size_px[0] || y_px >= self.size_px[1] {
            return;
        }
        let ts = self.tile_size;
        let key = (x_px / ts, y_px / ts);
        let i = ((y_px % ts) * ts + (x_px % ts)) as usize;
        let len = self.tile_len();
        let t = self
            .tiles
            .entry(key)
            .or_insert_with(|| vec![NODATA; len].into_boxed_slice());
        if t[i] != raw {
            t[i] = raw;
            self.dirty.insert(key);
        }
    }

    /// Set nilai (dikuantisasi) di `(x, y)`.
    pub fn set(&mut self, x_px: u32, y_px: u32, value: f32) {
        let raw = self.quantize(value);
        self.set_raw(x_px, y_px, raw);
    }

    /// Set nilai beserta bobotnya.
    pub fn set_with_weight(&mut self, x_px: u32, y_px: u32, value: f32, weight: u16) {
        self.set(x_px, y_px, value);
        if x_px >= self.size_px[0] || y_px >= self.size_px[1] {
            return;
        }
        let ts = self.tile_size;
        let key = (x_px / ts, y_px / ts);
        let i = ((y_px % ts) * ts + (x_px % ts)) as usize;
        let len = self.tile_len();
        let w = self.weights.get_or_insert_with(HashMap::new);
        w.entry(key)
            .or_insert_with(|| vec![0u16; len].into_boxed_slice())[i] = weight;
    }

    /// Pasang tile raw utuh. Galat bila ukuran ≠ `tile_size²` atau di luar kisi.
    pub fn set_tile_values(&mut self, tx: u32, ty: u32, raw: &[u16]) -> Result<(), SegError> {
        if raw.len() != self.tile_len() {
            return Err(SegError::InvalidParameter(format!(
                "ukuran tile {} ≠ {}",
                raw.len(),
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
        if raw.iter().all(|&v| v == NODATA) {
            self.tiles.remove(&key);
        } else {
            self.tiles.insert(key, raw.to_vec().into_boxed_slice());
        }
        self.dirty.insert(key);
        Ok(())
    }

    /// Buang tile yang seluruhnya NODATA.
    pub fn prune_empty_tiles(&mut self) {
        let empty: Vec<TileKey> = self
            .tiles
            .iter()
            .filter(|(_, t)| t.iter().all(|&v| v == NODATA))
            .map(|(k, _)| *k)
            .collect();
        for k in empty {
            self.tiles.remove(&k);
            if let Some(w) = self.weights.as_mut() {
                w.remove(&k);
            }
        }
    }

    /// Seluruh slide sebagai `Array2<f32>` bentuk `(h, w)`, NaN di NODATA.
    pub fn to_array(&self) -> Array2<f32> {
        let (w, h) = (self.size_px[0] as usize, self.size_px[1] as usize);
        let mut out = Array2::<f32>::from_elem((h, w), f32::NAN);
        let ts = self.tile_size as usize;
        for (&(tx, ty), t) in &self.tiles {
            let x0 = tx as usize * ts;
            let y0 = ty as usize * ts;
            for j in 0..ts.min(h.saturating_sub(y0)) {
                for i in 0..ts.min(w.saturating_sub(x0)) {
                    if let Some(v) = self.dequantize(t[j * ts + i]) {
                        out[[y0 + j, x0 + i]] = v;
                    }
                }
            }
        }
        out
    }

    /// Bobot seluruh slide sebagai `Array2<f32>`; tanpa bobot tersimpan,
    /// piksel berdata diberi bobot 1.
    pub fn weights_array(&self) -> Array2<f32> {
        let (w, h) = (self.size_px[0] as usize, self.size_px[1] as usize);
        let mut out = Array2::<f32>::zeros((h, w));
        let ts = self.tile_size as usize;
        for (&(tx, ty), t) in &self.tiles {
            let x0 = tx as usize * ts;
            let y0 = ty as usize * ts;
            let wt = self.weights.as_ref().and_then(|m| m.get(&(tx, ty)));
            for j in 0..ts.min(h.saturating_sub(y0)) {
                for i in 0..ts.min(w.saturating_sub(x0)) {
                    if t[j * ts + i] == NODATA {
                        continue;
                    }
                    out[[y0 + j, x0 + i]] = match wt {
                        Some(wt) => wt[j * ts + i] as f32,
                        None => 1.0,
                    };
                }
            }
        }
        out
    }

    /// Bangun penyimpanan dari larik nilai (NaN = NODATA) dan bobot opsional
    /// pada `mpp_attr`, memakai kuantisasi `scale`/`offset`.
    pub fn from_arrays(
        values: &Array2<f32>,
        weights: Option<&Array2<f32>>,
        mpp_attr: f64,
        tile_size: u32,
        scale: f32,
        offset: f32,
    ) -> Result<Self, SegError> {
        let (h, w) = values.dim();
        if let Some(wa) = weights {
            if wa.dim() != values.dim() {
                return Err(SegError::InvalidParameter(
                    "bentuk bobot ≠ bentuk nilai".to_string(),
                ));
            }
        }
        let mut s = Self::with_size_px(mpp_attr, [w as u32, h as u32], tile_size, scale, offset);
        for ((y, x), &v) in values.indexed_iter() {
            if v.is_nan() {
                continue;
            }
            match weights {
                Some(wa) => {
                    let wt = wa[[y, x]].round().clamp(0.0, u16::MAX as f32) as u16;
                    s.set_with_weight(x as u32, y as u32, v, wt);
                }
                None => s.set(x as u32, y as u32, v),
            }
        }
        s.clear_dirty();
        Ok(s)
    }

    /// Turunkan resolusi 2× (`mpp_attr × 2`) dengan rata-rata berbobot yang
    /// mengabaikan NODATA; bobot hasil = jumlah bobot. Tanpa bobot tersimpan,
    /// tiap piksel berdata berbobot 1.
    pub fn downsample2(&self) -> Result<AttrStore, SegError> {
        let v = self.to_array();
        let w = self.weights_array();
        let (v2, w2) = h3d_imgproc::pyramid::downsample2_attr_weighted(&v, &w, f32::NAN)?;
        Self::from_arrays(
            &v2,
            Some(&w2),
            self.mpp_attr * 2.0,
            self.tile_size,
            self.scale,
            self.offset,
        )
    }
}

/// Serialisasi `.atr` dengan `attr_id`.
pub fn encode_atr(store: &AttrStore, attr_id: u16) -> Vec<u8> {
    let keys = store.tile_keys();
    let tiles: Vec<((u32, u32), Vec<u8>)> = keys
        .iter()
        .filter_map(|&(tx, ty)| {
            store.tile(tx, ty).map(|t| {
                let mut bytes = Vec::with_capacity(t.len() * 2);
                for v in t {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                ((tx, ty), lz4_flex::compress_prepend_size(&bytes))
            })
        })
        .collect();
    let hdr = TileHeader {
        tile_size: store.tile_size.min(u16::MAX as u32) as u16,
        mpp: store.mpp_attr,
        w: store.size_px[0],
        h: store.size_px[1],
        n_tiles: tiles.len() as u32,
    };
    let mut out = Vec::new();
    hdr.write(ATR_MAGIC, &mut out);
    out.extend_from_slice(&store.scale.to_le_bytes());
    out.extend_from_slice(&store.offset.to_le_bytes());
    out.extend_from_slice(&attr_id.to_le_bytes());
    debug_assert_eq!(out.len(), ATR_HEADER_LEN);
    write_index_and_payload(&mut out, ATR_HEADER_LEN, &tiles);
    out
}

/// Baca `.atr` → `(AttrStore, attr_id)`. Pelanggaran format →
/// `SegError::InvalidFile`.
pub fn decode_atr(data: &[u8]) -> Result<(AttrStore, u16), SegError> {
    let mut cur = Cursor::new(data);
    let hdr = TileHeader::read(&mut cur, ATR_MAGIC)?;
    let scale = cur.f32()?;
    let offset = cur.f32()?;
    let attr_id = cur.u16()?;
    if !(scale.is_finite() && scale > 0.0) {
        return Err(SegError::invalid_file(format!("scale tidak sah: {scale}")));
    }
    if !offset.is_finite() {
        return Err(SegError::invalid_file(format!(
            "offset tidak sah: {offset}"
        )));
    }
    let entries = read_index(&mut cur, &hdr, data.len())?;
    let mut store =
        AttrStore::with_size_px(hdr.mpp, [hdr.w, hdr.h], hdr.tile_size as u32, scale, offset);
    let n = hdr.tile_size as usize * hdr.tile_size as usize;
    for e in &entries {
        let raw = decompress_tile(data, e, n * 2)?;
        let vals: Vec<u16> = raw
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        store.set_tile_values(e.tx, e.ty, &vals)?;
    }
    store.clear_dirty();
    Ok((store, attr_id))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_roundtrip_and_clamp() {
        let s = AttrStore::new(16.0, [1000.0, 1000.0], 0.01, -1.0);
        assert_eq!(s.quantize(-1.0), 0);
        assert_eq!(s.quantize(-5.0), 0);
        assert_eq!(s.quantize(1e9), MAX_RAW);
        assert_eq!(s.quantize(f32::NAN), NODATA);
        let v = s.dequantize(s.quantize(0.25)).unwrap();
        assert!((v - 0.25).abs() < 0.006);
    }

    #[test]
    fn set_get_nodata() {
        let mut s = AttrStore::new(16.0, [2048.0, 1024.0], 0.001, 0.0);
        assert_eq!(s.size_px, [128, 64]);
        assert_eq!(s.get(3, 3), None);
        s.set(3, 3, 0.5);
        assert!((s.get(3, 3).unwrap() - 0.5).abs() < 1e-3);
        assert_eq!(s.get_raw(999, 0), NODATA);
        s.set(999, 0, 1.0);
        assert_eq!(s.dirty_tiles(), vec![(0, 0)]);
        let arr = s.to_array();
        assert!(arr[[0, 0]].is_nan());
        assert!((arr[[3, 3]] - 0.5).abs() < 1e-3);
    }

    #[test]
    fn atr_roundtrip() {
        let mut s = AttrStore::new(16.0, [3000.0, 2000.0], 0.002, 0.0);
        for y in 0..20 {
            for x in 0..30 {
                s.set(x + 100, y + 60, (x + y) as f32 * 0.01);
            }
        }
        let bytes = encode_atr(&s, 7);
        assert_eq!(&bytes[0..4], b"H3DA");
        let (d, id) = decode_atr(&bytes).unwrap();
        assert_eq!(id, 7);
        assert_eq!(d.size_px, s.size_px);
        assert_eq!(d.scale, s.scale);
        assert_eq!(d.offset, s.offset);
        assert_eq!(d.tile_keys(), s.tile_keys());
        for (tx, ty) in s.tile_keys() {
            assert_eq!(d.tile(tx, ty), s.tile(tx, ty));
        }
        assert_eq!(encode_atr(&d, 7), bytes);
    }

    #[test]
    fn atr_corrupted() {
        let mut s = AttrStore::new(16.0, [1000.0, 1000.0], 1.0, 0.0);
        s.set(1, 1, 3.0);
        let good = encode_atr(&s, 1);
        let mut b = good.clone();
        b[0] = b'L';
        assert!(matches!(decode_atr(&b), Err(SegError::InvalidFile { .. })));
        // Magic .lbl pada pembaca .atr juga ditolak.
        let mut b = good.clone();
        b[3] = b'L';
        assert!(matches!(decode_atr(&b), Err(SegError::InvalidFile { .. })));
        assert!(matches!(
            decode_atr(&good[..good.len() - 3]),
            Err(SegError::InvalidFile { .. })
        ));
        assert!(matches!(
            decode_atr(&good[..30]),
            Err(SegError::InvalidFile { .. })
        ));
        // scale nol.
        let mut b = good.clone();
        b[28..32].copy_from_slice(&0.0f32.to_le_bytes());
        assert!(matches!(decode_atr(&b), Err(SegError::InvalidFile { .. })));
    }

    #[test]
    fn downsample_ignores_nodata_and_uses_weights() {
        let mut s = AttrStore::with_size_px(16.0, [4, 4], 64, 0.01, 0.0);
        // Blok (0,0): nilai 1.0 bobot 1 dan 3.0 bobot 3, dua NODATA → rata 2.5.
        s.set_with_weight(0, 0, 1.0, 1);
        s.set_with_weight(1, 0, 3.0, 3);
        // Blok (1,0): hanya NODATA → NODATA.
        // Blok (0,1): satu nilai 0.8 bobot 2.
        s.set_with_weight(0, 2, 0.8, 2);
        let d = s.downsample2().unwrap();
        assert_eq!(d.size_px, [2, 2]);
        assert_eq!(d.mpp_attr, 32.0);
        assert!((d.get(0, 0).unwrap() - 2.5).abs() < 0.02);
        assert_eq!(d.get(1, 0), None);
        assert!((d.get(0, 1).unwrap() - 0.8).abs() < 0.02);
        assert_eq!(d.weight(0, 0), 4);
        assert_eq!(d.weight(0, 1), 2);
        assert_eq!(d.weight(1, 1), 0);
    }
}
