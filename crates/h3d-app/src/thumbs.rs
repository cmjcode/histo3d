//! Pembuat thumbnail di thread latar.
//!
//! GUI tidak pernah membaca slide di thread UI: ia meminta
//! [`h3d_engine::ThumbTask`] dari sesi (murah: sumber piksel bersama dan
//! sidik jari sudah di-cache), mengirimnya ke satu thread pekerja, lalu
//! mengambil hasilnya pada frame berikutnya. Tugas tidak memegang `Session`,
//! sehingga pekerja tidak pernah bersaing dengan UI maupun job atas kunci sesi.

use std::collections::{HashMap, HashSet};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::JoinHandle;

use h3d_core::SlideId;
use h3d_engine::{EngineError, Session, ThumbTask};
use image::RgbaImage;

/// Kunci satu thumbnail.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ThumbKey {
    /// Irisan.
    pub slide: SlideId,
    /// Sisi terpanjang (piksel).
    pub side: u32,
    /// Dengan masker jaringan di kanal alfa (tampilan 3D).
    pub tissue: bool,
}

type Done = (ThumbKey, Result<RgbaImage, EngineError>);

/// Antrean thumbnail dengan satu thread pekerja.
pub struct ThumbWorker {
    tx: Option<Sender<(ThumbKey, ThumbTask)>>,
    rx: Receiver<Done>,
    pending: HashSet<ThumbKey>,
    ready: HashMap<ThumbKey, Result<RgbaImage, EngineError>>,
    thread: Option<JoinHandle<()>>,
}

impl Default for ThumbWorker {
    fn default() -> Self {
        Self::new()
    }
}

impl ThumbWorker {
    /// Buat pekerja (thread dimulai sekarang; menganggur sampai ada tugas).
    pub fn new() -> Self {
        let (tx, task_rx) = mpsc::channel::<(ThumbKey, ThumbTask)>();
        let (done_tx, rx) = mpsc::channel::<Done>();
        let thread = std::thread::Builder::new()
            .name("h3d-thumb".into())
            .spawn(move || {
                for (key, task) in task_rx {
                    if done_tx.send((key, task.run())).is_err() {
                        return;
                    }
                }
            })
            .ok();
        ThumbWorker {
            tx: Some(tx),
            rx,
            pending: HashSet::new(),
            ready: HashMap::new(),
            thread,
        }
    }

    /// Awal frame: ambil hasil yang sudah selesai.
    pub fn poll(&mut self) {
        while let Ok((key, res)) = self.rx.try_recv() {
            self.pending.remove(&key);
            self.ready.insert(key, res);
        }
    }

    /// Pastikan `key` sedang dibuat atau sudah siap. Galat menyiapkan tugas
    /// (berkas hilang/berubah) langsung masuk hasil siap.
    pub fn ensure_requested(&mut self, session: &Session, key: ThumbKey) {
        if self.pending.contains(&key) || self.ready.contains_key(&key) {
            return;
        }
        match session.thumbnail_task(key.slide, key.side, key.tissue) {
            Ok(task) => {
                let sent = self
                    .tx
                    .as_ref()
                    .is_some_and(|tx| tx.send((key, task)).is_ok());
                if sent {
                    self.pending.insert(key);
                } else {
                    self.ready.insert(
                        key,
                        Err(EngineError::invalid("thread thumbnail tidak berjalan")),
                    );
                }
            }
            Err(e) => {
                self.ready.insert(key, Err(e));
            }
        }
    }

    /// Ambil hasil `key` bila sudah siap (sekali ambil).
    pub fn take(&mut self, key: &ThumbKey) -> Option<Result<RgbaImage, EngineError>> {
        self.ready.remove(key)
    }

    /// Minta dan, bila sudah siap, ambil.
    pub fn request(
        &mut self,
        session: &Session,
        key: ThumbKey,
    ) -> Option<Result<RgbaImage, EngineError>> {
        self.ensure_requested(session, key);
        self.take(&key)
    }

    /// Masih ada tugas yang belum selesai (perlu repaint).
    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty() || !self.ready.is_empty()
    }

    /// Lupakan semua hasil dan permintaan irisan (dihapus/berubah).
    pub fn forget(&mut self, slide: SlideId) {
        self.pending.retain(|k| k.slide != slide);
        self.ready.retain(|k, _| k.slide != slide);
    }

    /// Lupakan semua hasil siap (proyek ditutup/ganti).
    pub fn clear(&mut self) {
        self.pending.clear();
        self.ready.clear();
    }
}

impl Drop for ThumbWorker {
    fn drop(&mut self) {
        self.tx.take();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}
