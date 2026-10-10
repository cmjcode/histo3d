//! Format berkas `.lbl` (histo3d.md 8.7.2), little-endian.
//!
//! ```text
//! offset  ukuran  isi
//! 0       4       magic "H3DL"
//! 4       2       versi (u16) = 1
//! 6       2       tile_size (u16)
//! 8       8       mpp_label (f64)
//! 16      4       lebar_px (u32)
//! 20      4       tinggi_px (u32)
//! 24      4       n_tile (u32)
//! 28      20·n    indeks: { tx: u32, ty: u32, offset: u64, len: u32 }, terurut (ty, tx)
//! …       …       payload: tiap tile = lz4_flex (size-prepended) dari u8 × tile_size²
//! ```
//!
//! `offset` dihitung dari awal berkas. Pembaca memvalidasi semuanya dan
//! mengembalikan `SegError::InvalidFile`, tidak pernah panik.

use std::collections::HashSet;

use crate::error::SegError;
use crate::label_store::LabelStore;

/// Magic berkas label.
pub const LBL_MAGIC: &[u8; 4] = b"H3DL";
/// Versi format yang ditulis.
pub const LBL_VERSION: u16 = 1;
/// Panjang header tetap.
pub const LBL_HEADER_LEN: usize = 28;
/// Panjang satu entri indeks.
pub const INDEX_ENTRY_LEN: usize = 20;

/// Pembaca byte little-endian dengan pemeriksaan batas.
pub(crate) struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub(crate) fn position(&self) -> usize {
        self.pos
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], SegError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| SegError::invalid_file("offset meluap"))?;
        if end > self.data.len() {
            return Err(SegError::invalid_file(format!(
                "berkas terpotong: butuh {end} byte, ada {}",
                self.data.len()
            )));
        }
        let s = &self.data[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    pub(crate) fn bytes4(&mut self) -> Result<[u8; 4], SegError> {
        let s = self.take(4)?;
        Ok([s[0], s[1], s[2], s[3]])
    }

    pub(crate) fn u16(&mut self) -> Result<u16, SegError> {
        let s = self.take(2)?;
        Ok(u16::from_le_bytes([s[0], s[1]]))
    }

    pub(crate) fn u32(&mut self) -> Result<u32, SegError> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    pub(crate) fn u64(&mut self) -> Result<u64, SegError> {
        let s = self.take(8)?;
        let mut b = [0u8; 8];
        b.copy_from_slice(s);
        Ok(u64::from_le_bytes(b))
    }

    pub(crate) fn f32(&mut self) -> Result<f32, SegError> {
        Ok(f32::from_bits(self.u32()?))
    }

    pub(crate) fn f64(&mut self) -> Result<f64, SegError> {
        Ok(f64::from_bits(self.u64()?))
    }
}

/// Header bersama `.lbl`/`.atr` (28 byte pertama).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TileHeader {
    pub tile_size: u16,
    pub mpp: f64,
    pub w: u32,
    pub h: u32,
    pub n_tiles: u32,
}

impl TileHeader {
    pub(crate) fn write(&self, magic: &[u8; 4], out: &mut Vec<u8>) {
        out.extend_from_slice(magic);
        out.extend_from_slice(&LBL_VERSION.to_le_bytes());
        out.extend_from_slice(&self.tile_size.to_le_bytes());
        out.extend_from_slice(&self.mpp.to_le_bytes());
        out.extend_from_slice(&self.w.to_le_bytes());
        out.extend_from_slice(&self.h.to_le_bytes());
        out.extend_from_slice(&self.n_tiles.to_le_bytes());
    }

    /// Baca dan validasi header bersama; `magic` adalah magic yang diharapkan.
    pub(crate) fn read(cur: &mut Cursor<'_>, magic: &[u8; 4]) -> Result<Self, SegError> {
        let got = cur.bytes4()?;
        if &got != magic {
            return Err(SegError::invalid_file(format!(
                "magic salah: {:?} (diharapkan {:?})",
                String::from_utf8_lossy(&got),
                String::from_utf8_lossy(magic)
            )));
        }
        let version = cur.u16()?;
        if version != LBL_VERSION {
            return Err(SegError::invalid_file(format!(
                "versi {version} tidak didukung (didukung {LBL_VERSION})"
            )));
        }
        let tile_size = cur.u16()?;
        if tile_size == 0 {
            return Err(SegError::invalid_file("tile_size nol"));
        }
        let mpp = cur.f64()?;
        if !(mpp.is_finite() && mpp > 0.0) {
            return Err(SegError::invalid_file(format!("mpp tidak sah: {mpp}")));
        }
        let w = cur.u32()?;
        let h = cur.u32()?;
        let n_tiles = cur.u32()?;
        Ok(Self {
            tile_size,
            mpp,
            w,
            h,
            n_tiles,
        })
    }
}

/// Satu entri indeks.
#[derive(Debug, Clone, Copy)]
pub(crate) struct IndexEntry {
    pub tx: u32,
    pub ty: u32,
    pub offset: u64,
    pub len: u32,
}

/// Baca indeks dan validasi kisi, duplikat, serta batas payload. Mengembalikan
/// entri dalam urutan berkas.
pub(crate) fn read_index(
    cur: &mut Cursor<'_>,
    hdr: &TileHeader,
    file_len: usize,
) -> Result<Vec<IndexEntry>, SegError> {
    let n = hdr.n_tiles as usize;
    let index_bytes = n
        .checked_mul(INDEX_ENTRY_LEN)
        .ok_or_else(|| SegError::invalid_file("n_tile meluap"))?;
    if cur.position().saturating_add(index_bytes) > file_len {
        return Err(SegError::invalid_file(format!(
            "indeks {n} tile melampaui ukuran berkas"
        )));
    }
    let ts = hdr.tile_size as u32;
    let nx = hdr.w.div_ceil(ts);
    let ny = hdr.h.div_ceil(ts);
    let mut seen = HashSet::with_capacity(n);
    let mut entries = Vec::with_capacity(n);
    for _ in 0..n {
        let tx = cur.u32()?;
        let ty = cur.u32()?;
        let offset = cur.u64()?;
        let len = cur.u32()?;
        if tx >= nx || ty >= ny {
            return Err(SegError::invalid_file(format!(
                "tile ({tx}, {ty}) di luar kisi {nx}×{ny}"
            )));
        }
        if !seen.insert((tx, ty)) {
            return Err(SegError::invalid_file(format!("tile ({tx}, {ty}) ganda")));
        }
        let end = offset
            .checked_add(len as u64)
            .ok_or_else(|| SegError::invalid_file("offset + len meluap"))?;
        if end > file_len as u64 {
            return Err(SegError::invalid_file(format!(
                "tile ({tx}, {ty}): offset {offset} + len {len} melampaui ukuran berkas {file_len}"
            )));
        }
        entries.push(IndexEntry {
            tx,
            ty,
            offset,
            len,
        });
    }
    Ok(entries)
}

/// Dekompresi satu tile dan periksa ukurannya.
pub(crate) fn decompress_tile(
    data: &[u8],
    e: &IndexEntry,
    expected_len: usize,
) -> Result<Vec<u8>, SegError> {
    let start = e.offset as usize;
    let end = start + e.len as usize;
    let payload = &data[start..end];
    // lz4_flex mengalokasikan sesuai awalan ukuran; periksa dulu agar berkas
    // rusak tidak bisa meminta alokasi raksasa.
    let claimed = match payload {
        [a, b, c, d, ..] => u32::from_le_bytes([*a, *b, *c, *d]) as usize,
        _ => {
            return Err(SegError::invalid_file(format!(
                "tile ({}, {}): payload terlalu pendek",
                e.tx, e.ty
            )))
        }
    };
    if claimed != expected_len {
        return Err(SegError::invalid_file(format!(
            "tile ({}, {}): ukuran dekompresi {claimed} ≠ {expected_len}",
            e.tx, e.ty
        )));
    }
    let raw = lz4_flex::decompress_size_prepended(payload).map_err(|err| {
        SegError::invalid_file(format!("tile ({}, {}): lz4 gagal: {err}", e.tx, e.ty))
    })?;
    if raw.len() != expected_len {
        return Err(SegError::invalid_file(format!(
            "tile ({}, {}): ukuran dekompresi {} ≠ {expected_len}",
            e.tx,
            e.ty,
            raw.len()
        )));
    }
    Ok(raw)
}

/// Tulis indeks + payload untuk tile yang sudah dikompresi (urutan sesuai
/// `tiles`, yang harus sudah terurut `(ty, tx)`). `header_len` = panjang
/// header sebelum indeks.
pub(crate) fn write_index_and_payload(
    out: &mut Vec<u8>,
    header_len: usize,
    tiles: &[((u32, u32), Vec<u8>)],
) {
    let mut offset = (header_len + tiles.len() * INDEX_ENTRY_LEN) as u64;
    for ((tx, ty), comp) in tiles {
        out.extend_from_slice(&tx.to_le_bytes());
        out.extend_from_slice(&ty.to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
        out.extend_from_slice(&(comp.len() as u32).to_le_bytes());
        offset += comp.len() as u64;
    }
    for (_, comp) in tiles {
        out.extend_from_slice(comp);
    }
}

/// Serialisasi `LabelStore` ke byte `.lbl`. Deterministik: indeks terurut
/// `(ty, tx)`.
pub fn encode_lbl(store: &LabelStore) -> Vec<u8> {
    let keys = store.tile_keys();
    let tiles: Vec<((u32, u32), Vec<u8>)> = keys
        .iter()
        .filter_map(|&(tx, ty)| {
            store
                .tile(tx, ty)
                .map(|t| ((tx, ty), lz4_flex::compress_prepend_size(t)))
        })
        .collect();
    let hdr = TileHeader {
        tile_size: store.tile_size.min(u16::MAX as u32) as u16,
        mpp: store.mpp_label,
        w: store.size_px[0],
        h: store.size_px[1],
        n_tiles: tiles.len() as u32,
    };
    let mut out = Vec::with_capacity(
        LBL_HEADER_LEN
            + tiles.len() * INDEX_ENTRY_LEN
            + tiles.iter().map(|(_, c)| c.len()).sum::<usize>(),
    );
    hdr.write(LBL_MAGIC, &mut out);
    write_index_and_payload(&mut out, LBL_HEADER_LEN, &tiles);
    out
}

/// Baca byte `.lbl` menjadi `LabelStore`. Semua pelanggaran format →
/// `SegError::InvalidFile`.
pub fn decode_lbl(data: &[u8]) -> Result<LabelStore, SegError> {
    let mut cur = Cursor::new(data);
    let hdr = TileHeader::read(&mut cur, LBL_MAGIC)?;
    let entries = read_index(&mut cur, &hdr, data.len())?;
    let mut store = LabelStore::with_size_px(hdr.mpp, [hdr.w, hdr.h], hdr.tile_size as u32);
    let tile_len = hdr.tile_size as usize * hdr.tile_size as usize;
    for e in &entries {
        let raw = decompress_tile(data, e, tile_len)?;
        store.set_tile(e.tx, e.ty, &raw)?;
    }
    store.clear_dirty();
    Ok(store)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::label_store::PaintMode;
    use h3d_core::LabelId;
    use proptest::prelude::*;

    fn sample_store() -> LabelStore {
        let mut s = LabelStore::new(4.0, [3000.0, 2000.0]);
        s.paint_disc([500.0, 500.0], 300.0, LabelId(1), PaintMode::Replace);
        s.paint_disc([2500.0, 1500.0], 200.0, LabelId(9), PaintMode::Replace);
        s
    }

    #[test]
    fn roundtrip_basic() {
        let s = sample_store();
        let bytes = encode_lbl(&s);
        assert_eq!(&bytes[0..4], b"H3DL");
        let d = decode_lbl(&bytes).unwrap();
        assert_eq!(d.size_px, s.size_px);
        assert_eq!(d.mpp_label, s.mpp_label);
        assert_eq!(d.tile_keys(), s.tile_keys());
        for (tx, ty) in s.tile_keys() {
            assert_eq!(d.tile(tx, ty), s.tile(tx, ty));
        }
        assert!(d.dirty_tiles().is_empty());
        // Deterministik.
        assert_eq!(encode_lbl(&d), bytes);
    }

    #[test]
    fn empty_store_roundtrip() {
        let s = LabelStore::new(2.0, [100.0, 100.0]);
        let d = decode_lbl(&encode_lbl(&s)).unwrap();
        assert!(d.is_empty());
        assert_eq!(d.size_px, [50, 50]);
    }

    #[test]
    fn corrupted_inputs_are_errors() {
        let s = sample_store();
        let good = encode_lbl(&s);

        // Magic salah.
        let mut b = good.clone();
        b[0] = b'X';
        assert!(matches!(decode_lbl(&b), Err(SegError::InvalidFile { .. })));

        // Versi salah.
        let mut b = good.clone();
        b[4] = 9;
        assert!(matches!(decode_lbl(&b), Err(SegError::InvalidFile { .. })));

        // tile_size nol.
        let mut b = good.clone();
        b[6] = 0;
        b[7] = 0;
        assert!(matches!(decode_lbl(&b), Err(SegError::InvalidFile { .. })));

        // Offset di luar berkas (entri pertama).
        let mut b = good.clone();
        let off = LBL_HEADER_LEN + 8;
        b[off..off + 8].copy_from_slice(&(u64::MAX - 10).to_le_bytes());
        assert!(matches!(decode_lbl(&b), Err(SegError::InvalidFile { .. })));
        let mut b = good.clone();
        b[off..off + 8].copy_from_slice(&(good.len() as u64).to_le_bytes());
        assert!(matches!(decode_lbl(&b), Err(SegError::InvalidFile { .. })));

        // Payload terpotong.
        let b = &good[..good.len() - 5];
        assert!(matches!(decode_lbl(b), Err(SegError::InvalidFile { .. })));

        // Header terpotong.
        assert!(matches!(
            decode_lbl(&good[..10]),
            Err(SegError::InvalidFile { .. })
        ));
        assert!(matches!(decode_lbl(&[]), Err(SegError::InvalidFile { .. })));

        // Tile di luar kisi.
        let mut b = good.clone();
        b[LBL_HEADER_LEN..LBL_HEADER_LEN + 4].copy_from_slice(&999u32.to_le_bytes());
        assert!(matches!(decode_lbl(&b), Err(SegError::InvalidFile { .. })));

        // Ukuran dekompresi salah: ganti payload tile pertama dengan tile kecil.
        let mut st = LabelStore::with_size_px(4.0, [300, 300], 256);
        st.set_tile(0, 0, &vec![1u8; 256 * 256]).unwrap();
        let small = lz4_flex::compress_prepend_size(&[1u8; 100]);
        let mut b = Vec::new();
        TileHeader {
            tile_size: 256,
            mpp: 4.0,
            w: 300,
            h: 300,
            n_tiles: 1,
        }
        .write(LBL_MAGIC, &mut b);
        write_index_and_payload(&mut b, LBL_HEADER_LEN, &[((0, 0), small)]);
        assert!(matches!(decode_lbl(&b), Err(SegError::InvalidFile { .. })));

        // Tile ganda.
        let comp = lz4_flex::compress_prepend_size(&[2u8; 256 * 256]);
        let mut b = Vec::new();
        TileHeader {
            tile_size: 256,
            mpp: 4.0,
            w: 600,
            h: 300,
            n_tiles: 2,
        }
        .write(LBL_MAGIC, &mut b);
        write_index_and_payload(
            &mut b,
            LBL_HEADER_LEN,
            &[((0, 0), comp.clone()), ((0, 0), comp)],
        );
        assert!(matches!(decode_lbl(&b), Err(SegError::InvalidFile { .. })));

        // Sampah acak tidak panik.
        for seed in 0..64u8 {
            let junk: Vec<u8> = (0..200u32)
                .map(|i| (i as u8).wrapping_mul(seed.wrapping_add(3)))
                .collect();
            let _ = decode_lbl(&junk);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(24))]
        #[test]
        fn roundtrip_random_sparse_tiles(
            tiles in prop::collection::vec((0u32..5, 0u32..4, 0u8..4, 0u32..60_000), 0..6),
            w in 1u32..1200,
            h in 1u32..1000,
        ) {
            let mut s = LabelStore::with_size_px(4.0, [w, h], 256);
            let [nx, ny] = s.grid_size();
            for (tx, ty, val, pos) in tiles {
                if tx >= nx || ty >= ny { continue; }
                let mut t = vec![0u8; 256 * 256];
                // Pola jarang: beberapa piksel bernilai val.
                for k in 0..7usize {
                    t[(pos as usize + k * 977) % (256 * 256)] = val;
                }
                s.set_tile(tx, ty, &t).unwrap();
            }
            let bytes = encode_lbl(&s);
            let d = decode_lbl(&bytes).unwrap();
            prop_assert_eq!(d.size_px, s.size_px);
            prop_assert_eq!(d.tile_keys(), s.tile_keys());
            for (tx, ty) in s.tile_keys() {
                prop_assert_eq!(d.tile(tx, ty), s.tile(tx, ty));
            }
            prop_assert_eq!(encode_lbl(&d), bytes);
        }
    }
}
