//! Tumpukan undo berbasis command (8.1), pola `ducad-ink/src/commands.rs`.

/// Satu perubahan yang bisa diterapkan dan dibatalkan atas keadaan `S`.
pub trait Command<S>: Send {
    /// Terapkan perubahan.
    fn apply(&mut self, state: &mut S);
    /// Batalkan perubahan (keadaan harus kembali persis).
    fn revert(&mut self, state: &mut S);
    /// Nama untuk menu "Batalkan …".
    fn name(&self) -> &str;
}

/// Tumpukan undo/redo dengan batas jumlah entri.
pub struct UndoStack<S> {
    done: Vec<Box<dyn Command<S>>>,
    undone: Vec<Box<dyn Command<S>>>,
    limit: usize,
}

impl<S> Default for UndoStack<S> {
    fn default() -> Self {
        Self::new(200)
    }
}

impl<S> UndoStack<S> {
    /// Tumpukan baru dengan batas `limit` entri (≥ 1).
    pub fn new(limit: usize) -> Self {
        Self {
            done: Vec::new(),
            undone: Vec::new(),
            limit: limit.max(1),
        }
    }

    /// Terapkan command dan simpan; membuang riwayat redo.
    pub fn push(&mut self, mut cmd: Box<dyn Command<S>>, state: &mut S) {
        cmd.apply(state);
        self.undone.clear();
        self.done.push(cmd);
        if self.done.len() > self.limit {
            self.done.remove(0);
        }
    }

    /// Batalkan command terakhir; `Some(nama)` bila ada.
    pub fn undo(&mut self, state: &mut S) -> Option<String> {
        let mut cmd = self.done.pop()?;
        cmd.revert(state);
        let name = cmd.name().to_owned();
        self.undone.push(cmd);
        Some(name)
    }

    /// Ulangi command yang dibatalkan; `Some(nama)` bila ada.
    pub fn redo(&mut self, state: &mut S) -> Option<String> {
        let mut cmd = self.undone.pop()?;
        cmd.apply(state);
        let name = cmd.name().to_owned();
        self.done.push(cmd);
        Some(name)
    }

    /// Ada yang bisa dibatalkan.
    pub fn can_undo(&self) -> bool {
        !self.done.is_empty()
    }

    /// Ada yang bisa diulang.
    pub fn can_redo(&self) -> bool {
        !self.undone.is_empty()
    }

    /// Nama command yang akan dibatalkan berikutnya.
    pub fn undo_name(&self) -> Option<&str> {
        self.done.last().map(|c| c.name())
    }

    /// Kosongkan seluruh riwayat.
    pub fn clear(&mut self) {
        self.done.clear();
        self.undone.clear();
    }

    /// Jumlah entri undo.
    pub fn len(&self) -> usize {
        self.done.len()
    }

    /// Tidak ada entri undo.
    pub fn is_empty(&self) -> bool {
        self.done.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Add(i64);
    impl Command<i64> for Add {
        fn apply(&mut self, s: &mut i64) {
            *s += self.0;
        }
        fn revert(&mut self, s: &mut i64) {
            *s -= self.0;
        }
        fn name(&self) -> &str {
            "tambah"
        }
    }

    #[test]
    fn undo_redo_roundtrip() {
        let mut st = 0i64;
        let mut u = UndoStack::new(10);
        for i in 1..=5 {
            u.push(Box::new(Add(i)), &mut st);
        }
        assert_eq!(st, 15);
        assert_eq!(u.undo(&mut st).as_deref(), Some("tambah"));
        assert_eq!(st, 10);
        u.undo(&mut st);
        assert_eq!(st, 6);
        u.redo(&mut st);
        assert_eq!(st, 10);
        u.push(Box::new(Add(100)), &mut st);
        assert!(!u.can_redo());
        assert_eq!(st, 110);
    }

    #[test]
    fn limit_drops_oldest() {
        let mut st = 0i64;
        let mut u = UndoStack::new(2);
        u.push(Box::new(Add(1)), &mut st);
        u.push(Box::new(Add(2)), &mut st);
        u.push(Box::new(Add(3)), &mut st);
        assert_eq!(u.len(), 2);
        u.undo(&mut st);
        u.undo(&mut st);
        assert!(u.undo(&mut st).is_none());
        assert_eq!(st, 1);
    }
}
