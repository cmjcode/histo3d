//! Command undo berbasis tile untuk `LabelStore` (histo3d.md 8.1, H4).
//!
//! Tiap command menyimpan salinan **terkompresi** (lz4) tile yang berubah,
//! sebelum dan sesudah, bukan seluruh label.

use h3d_core::Command;

use crate::label_store::{LabelStore, TileKey};

/// Potret satu tile sebelum/sesudah perubahan (lz4 size-prepended; `None`
/// = tile tidak ada / seluruhnya latar).
#[derive(Debug, Clone)]
pub struct TileSnapshot {
    /// Kunci tile.
    pub key: TileKey,
    /// Isi terkompresi sebelum perubahan.
    pub before: Option<Vec<u8>>,
    /// Isi terkompresi sesudah perubahan.
    pub after: Option<Vec<u8>>,
}

/// Suntingan label yang bisa dibatalkan.
#[derive(Debug, Clone)]
pub struct LabelEditCommand {
    name: String,
    snapshots: Vec<TileSnapshot>,
}

fn compress(t: Option<&[u8]>) -> Option<Vec<u8>> {
    t.map(lz4_flex::compress_prepend_size)
}

fn write_tile(store: &mut LabelStore, key: TileKey, comp: Option<&Vec<u8>>) {
    match comp {
        None => store.remove_tile(key.0, key.1),
        Some(c) => match lz4_flex::decompress_size_prepended(c) {
            // Data dibuat oleh kita sendiri; kegagalan hanya mungkin bila
            // memori rusak. Jangan panik: catat dan lewati.
            Ok(raw) => {
                if let Err(e) = store.set_tile(key.0, key.1, &raw) {
                    log::error!("undo: tile {key:?} ditolak: {e}");
                }
            }
            Err(e) => log::error!("undo: dekompresi tile {key:?} gagal: {e}"),
        },
    }
}

impl LabelEditCommand {
    /// Jalankan `mutate` pada `store` sambil merekam tile yang berubah;
    /// hasilnya command yang `apply`-nya menulis keadaan sesudah dan
    /// `revert`-nya mengembalikan keadaan sebelum. Mutasi sudah diterapkan
    /// saat fungsi ini kembali, sehingga `apply` pertama (mis. lewat
    /// `UndoStack::push`) hanya menulis ulang isi yang sama.
    pub fn capture<F: FnOnce(&mut LabelStore)>(
        store: &mut LabelStore,
        mutate: F,
        name: impl Into<String>,
    ) -> Self {
        store.begin_record();
        mutate(store);
        let before = store.end_record();
        let mut snapshots = Vec::with_capacity(before.len());
        for (key, old) in before {
            let now = store.tile(key.0, key.1);
            let old_slice = old.as_deref();
            if old_slice == now {
                continue;
            }
            snapshots.push(TileSnapshot {
                key,
                before: compress(old_slice),
                after: compress(now),
            });
        }
        Self {
            name: name.into(),
            snapshots,
        }
    }

    /// Benar bila tidak ada tile yang berubah.
    pub fn is_noop(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// Potret tile.
    pub fn snapshots(&self) -> &[TileSnapshot] {
        &self.snapshots
    }

    /// Ukuran memori terkompresi (byte).
    pub fn compressed_bytes(&self) -> usize {
        self.snapshots
            .iter()
            .map(|s| s.before.as_ref().map_or(0, Vec::len) + s.after.as_ref().map_or(0, Vec::len))
            .sum()
    }
}

impl Command<LabelStore> for LabelEditCommand {
    fn apply(&mut self, state: &mut LabelStore) {
        for s in &self.snapshots {
            write_tile(state, s.key, s.after.as_ref());
        }
    }

    fn revert(&mut self, state: &mut LabelStore) {
        for s in self.snapshots.iter().rev() {
            write_tile(state, s.key, s.before.as_ref());
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::label_store::PaintMode;
    use h3d_core::{LabelId, UndoStack};

    fn snapshot(s: &LabelStore) -> Vec<(TileKey, Vec<u8>)> {
        s.tile_keys()
            .into_iter()
            .map(|(tx, ty)| ((tx, ty), s.tile(tx, ty).unwrap().to_vec()))
            .collect()
    }

    #[test]
    fn undo_restores_exact_state_after_n_strokes() {
        let mut store = LabelStore::new(4.0, [4096.0, 4096.0]);
        store.paint_disc([300.0, 300.0], 150.0, LabelId(2), PaintMode::Replace);
        store.clear_dirty();
        let initial = snapshot(&store);
        let mut undo: UndoStack<LabelStore> = UndoStack::new(100);
        let mut states = vec![initial.clone()];
        for i in 0..12u32 {
            let cx = 200.0 + i as f64 * 300.0;
            let cmd = LabelEditCommand::capture(
                &mut store,
                |s| {
                    s.paint_stroke(
                        &[[cx, 200.0], [cx + 250.0, 3000.0]],
                        40.0,
                        LabelId(1 + (i % 3) as u8),
                        if i % 4 == 3 {
                            PaintMode::Erase
                        } else {
                            PaintMode::Replace
                        },
                    );
                },
                format!("goresan {i}"),
            );
            assert!(!cmd.is_noop() || i % 4 == 3);
            undo.push(Box::new(cmd), &mut store);
            states.push(snapshot(&store));
        }
        for k in (0..12).rev() {
            assert_eq!(
                undo.undo(&mut store).as_deref(),
                Some(format!("goresan {k}").as_str())
            );
            assert_eq!(snapshot(&store), states[k], "undo ke-{k}");
        }
        assert_eq!(snapshot(&store), initial);
        for k in 0..12 {
            undo.redo(&mut store);
            assert_eq!(snapshot(&store), states[k + 1], "redo ke-{k}");
        }
    }

    #[test]
    fn capture_noop_when_nothing_changes() {
        let mut store = LabelStore::new(4.0, [1000.0, 1000.0]);
        let cmd = LabelEditCommand::capture(
            &mut store,
            |s| {
                s.paint_disc([100.0, 100.0], 20.0, LabelId(1), PaintMode::Erase);
            },
            "kosong",
        );
        assert!(cmd.is_noop());
        assert_eq!(cmd.compressed_bytes(), 0);
    }

    #[test]
    fn revert_removes_created_tile() {
        let mut store = LabelStore::new(4.0, [1000.0, 1000.0]);
        let mut cmd = LabelEditCommand::capture(
            &mut store,
            |s| {
                s.paint_disc([100.0, 100.0], 20.0, LabelId(1), PaintMode::Replace);
            },
            "cakram",
        );
        assert_eq!(store.tile_count(), 1);
        cmd.revert(&mut store);
        assert!(store.is_empty());
        cmd.apply(&mut store);
        assert_eq!(store.tile_count(), 1);
        assert_eq!(store.get_um([100.0, 100.0]), 1);
    }
}
