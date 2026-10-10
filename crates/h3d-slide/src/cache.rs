//! Cache tile CPU: LRU berbatas byte (histo3d.md 8.2). Anggaran byte datang
//! dari `h3d_core::Budgets::tile_cache_cpu_bytes`; tidak ada angka tertanam
//! di sini (aturan keras 15).

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use h3d_core::SlideId;
use image::RgbaImage;

/// Kunci tile: `(slide, level, tx, ty)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileKey {
    /// Irisan.
    pub slide: SlideId,
    /// Level piramida.
    pub level: u32,
    /// Indeks tile horizontal.
    pub tx: u32,
    /// Indeks tile vertikal.
    pub ty: u32,
}

impl From<(SlideId, u32, u32, u32)> for TileKey {
    fn from((slide, level, tx, ty): (SlideId, u32, u32, u32)) -> Self {
        Self {
            slide,
            level,
            tx,
            ty,
        }
    }
}

struct Entry {
    tile: Arc<RgbaImage>,
    tick: u64,
}

/// Cache LRU tile RGBA8 berbatas byte. Pemanggil memegang `&mut`
/// (bungkus dengan `Mutex` bila dibagi antar thread).
pub struct TileCache {
    max_bytes: u64,
    bytes: u64,
    next_tick: u64,
    entries: HashMap<TileKey, Entry>,
    /// Urutan pemakaian: tick → kunci; tick terkecil = paling lama.
    lru: BTreeMap<u64, TileKey>,
}

/// Ukuran byte satu tile RGBA8.
pub fn tile_bytes(tile: &RgbaImage) -> u64 {
    u64::from(tile.width()) * u64::from(tile.height()) * 4
}

impl TileCache {
    /// Cache dengan batas `max_bytes` (dari `Budgets`).
    pub fn new(max_bytes: u64) -> Self {
        Self {
            max_bytes,
            bytes: 0,
            next_tick: 0,
            entries: HashMap::new(),
            lru: BTreeMap::new(),
        }
    }

    /// Batas byte saat ini.
    pub fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    /// Ubah batas byte dan buang entri bila perlu.
    pub fn set_max_bytes(&mut self, max_bytes: u64) {
        self.max_bytes = max_bytes;
        self.evict_to(max_bytes);
    }

    /// Byte yang sedang dipakai tile.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Jumlah tile tersimpan.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Benar bila kosong.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Ambil tile dan tandai sebagai baru dipakai.
    pub fn get(&mut self, key: &TileKey) -> Option<Arc<RgbaImage>> {
        let tick = self.bump();
        let e = self.entries.get_mut(key)?;
        self.lru.remove(&e.tick);
        e.tick = tick;
        self.lru.insert(tick, *key);
        Some(Arc::clone(&e.tile))
    }

    /// Benar bila `key` ada (tanpa mengubah urutan LRU).
    pub fn contains(&self, key: &TileKey) -> bool {
        self.entries.contains_key(key)
    }

    /// Simpan tile; mengembalikan `false` bila tile sendiri melebihi batas
    /// (tidak disimpan). Entri lama dengan kunci sama diganti.
    pub fn insert(&mut self, key: TileKey, tile: Arc<RgbaImage>) -> bool {
        let size = tile_bytes(&tile);
        if size > self.max_bytes {
            return false;
        }
        self.remove(&key);
        let room = self.max_bytes.saturating_sub(size);
        self.evict_to(room);
        let tick = self.bump();
        self.entries.insert(key, Entry { tile, tick });
        self.lru.insert(tick, key);
        self.bytes += size;
        true
    }

    /// Hapus satu tile.
    pub fn remove(&mut self, key: &TileKey) -> Option<Arc<RgbaImage>> {
        let e = self.entries.remove(key)?;
        self.lru.remove(&e.tick);
        self.bytes -= tile_bytes(&e.tile);
        Some(e.tile)
    }

    /// Hapus semua tile milik `slide`.
    pub fn remove_slide(&mut self, slide: SlideId) {
        let keys: Vec<TileKey> = self
            .entries
            .keys()
            .filter(|k| k.slide == slide)
            .copied()
            .collect();
        for k in keys {
            self.remove(&k);
        }
    }

    /// Buang tile paling lama sampai pemakaian ≤ `target_bytes`.
    pub fn evict_to(&mut self, target_bytes: u64) {
        while self.bytes > target_bytes {
            let Some((&tick, &key)) = self.lru.iter().next() else {
                break;
            };
            self.lru.remove(&tick);
            if let Some(e) = self.entries.remove(&key) {
                self.bytes -= tile_bytes(&e.tile);
            }
        }
    }

    /// Kosongkan cache.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.lru.clear();
        self.bytes = 0;
    }

    fn bump(&mut self) -> u64 {
        let t = self.next_tick;
        self.next_tick += 1;
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(i: u32) -> TileKey {
        TileKey::from((SlideId(1), 0, i, 0))
    }

    fn tile(side: u32) -> Arc<RgbaImage> {
        Arc::new(RgbaImage::new(side, side))
    }

    #[test]
    fn evicts_least_recently_used() {
        // Tiap tile 4x4 = 64 byte; batas 3 tile.
        let mut c = TileCache::new(64 * 3);
        for i in 0..3 {
            assert!(c.insert(key(i), tile(4)));
        }
        assert_eq!(c.bytes(), 192);
        // Sentuh tile 0 agar tile 1 jadi yang paling lama.
        assert!(c.get(&key(0)).is_some());
        assert!(c.insert(key(3), tile(4)));
        assert_eq!(c.len(), 3);
        assert!(c.contains(&key(0)));
        assert!(!c.contains(&key(1)));
        assert!(c.contains(&key(2)));
        assert!(c.contains(&key(3)));
        assert!(c.bytes() <= c.max_bytes());
    }

    #[test]
    fn rejects_oversize_and_evict_to_and_clear() {
        let mut c = TileCache::new(100);
        assert!(!c.insert(key(0), tile(8))); // 256 byte > 100
        assert!(c.is_empty());
        assert!(c.insert(key(1), tile(2))); // 16
        assert!(c.insert(key(2), tile(2)));
        assert!(c.insert(key(3), tile(2)));
        c.evict_to(32);
        assert_eq!(c.len(), 2);
        assert!(!c.contains(&key(1)));
        c.clear();
        assert_eq!((c.len(), c.bytes()), (0, 0));
    }

    #[test]
    fn remove_slide_only_removes_that_slide() {
        let mut c = TileCache::new(1 << 20);
        c.insert(TileKey::from((SlideId(1), 0, 0, 0)), tile(2));
        c.insert(TileKey::from((SlideId(2), 0, 0, 0)), tile(2));
        c.remove_slide(SlideId(1));
        assert_eq!(c.len(), 1);
        assert!(c.contains(&TileKey::from((SlideId(2), 0, 0, 0))));
        assert_eq!(c.bytes(), 16);
    }
}
