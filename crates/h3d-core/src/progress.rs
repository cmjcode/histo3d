//! Laporan progres dan pembatalan (8.1).

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Mutex;

/// Diberikan ke semua fungsi yang bisa berjalan lebih dari ~100 ms. Pelaksana
/// memeriksa `is_cancelled()` minimal sekali per irisan atau per chunk.
pub trait Progress: Sync {
    /// `fraction` dalam 0..=1; `stage` teks tahap untuk ditampilkan.
    fn set(&self, fraction: f32, stage: &str);
    /// Benar bila pengguna meminta berhenti.
    fn is_cancelled(&self) -> bool;
}

/// Implementasi kosong untuk CLI dan tes.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoProgress;

impl Progress for NoProgress {
    fn set(&self, _fraction: f32, _stage: &str) {}
    fn is_cancelled(&self) -> bool {
        false
    }
}

/// Progres yang bisa dibagi antar thread: fraksi `AtomicU32` (bit `f32`),
/// tahap `Mutex<String>`, tanda batal `AtomicBool`. Dipakai `h3d-engine`.
#[derive(Debug, Default)]
pub struct SharedProgress {
    fraction_bits: AtomicU32,
    stage: Mutex<String>,
    cancelled: AtomicBool,
}

impl SharedProgress {
    /// Progres baru pada 0 %.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fraksi terakhir yang dilaporkan.
    pub fn fraction(&self) -> f32 {
        f32::from_bits(self.fraction_bits.load(Ordering::Relaxed))
    }

    /// Tahap terakhir yang dilaporkan.
    pub fn stage(&self) -> String {
        self.stage.lock().map(|s| s.clone()).unwrap_or_default()
    }

    /// Minta job berhenti pada titik periksa berikutnya.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }
}

impl Progress for SharedProgress {
    fn set(&self, fraction: f32, stage: &str) {
        self.fraction_bits
            .store(fraction.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
        if let Ok(mut s) = self.stage.lock() {
            if s.as_str() != stage {
                s.clear();
                s.push_str(stage);
            }
        }
    }

    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

/// Sub-rentang progres: memetakan 0..1 milik anak ke `[lo, hi]` milik induk.
pub struct SubProgress<'a> {
    parent: &'a dyn Progress,
    lo: f32,
    hi: f32,
}

impl<'a> SubProgress<'a> {
    /// Bagian `[lo, hi]` dari `parent`.
    pub fn new(parent: &'a dyn Progress, lo: f32, hi: f32) -> Self {
        Self { parent, lo, hi }
    }
}

impl Progress for SubProgress<'_> {
    fn set(&self, fraction: f32, stage: &str) {
        let f = self.lo + (self.hi - self.lo) * fraction.clamp(0.0, 1.0);
        self.parent.set(f, stage);
    }
    fn is_cancelled(&self) -> bool {
        self.parent.is_cancelled()
    }
}
