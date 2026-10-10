//! Pemuat tile berprioritas dengan thread pekerja (histo3d.md 8.2).
//!
//! Permintaan masuk antrean prioritas (nilai prioritas lebih kecil = lebih
//! dulu; tile di tengah layar diberi nilai kecil). Tiap permintaan membawa
//! nomor generasi viewport; permintaan yang generasinya lebih tua dari
//! generasi saat ini dibuang, baik saat masuk antrean maupun saat diambil
//! pekerja. Hasil dikirim lewat `mpsc`. Hanya `std`, tanpa runtime async.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use h3d_core::SlideId;
use image::RgbaImage;

use crate::cache::TileKey;
use crate::error::SlideError;
use crate::source::SlideSource;

/// Permintaan satu tile.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileRequest {
    /// Irisan.
    pub slide: SlideId,
    /// Level piramida.
    pub level: u32,
    /// Indeks tile horizontal.
    pub tx: u32,
    /// Indeks tile vertikal.
    pub ty: u32,
    /// Generasi viewport saat permintaan dibuat.
    pub generation: u64,
    /// Prioritas: lebih kecil = lebih dulu (mis. jarak ke pusat layar).
    pub priority: f32,
}

impl TileRequest {
    /// Kunci cache untuk permintaan ini.
    pub fn key(&self) -> TileKey {
        TileKey {
            slide: self.slide,
            level: self.level,
            tx: self.tx,
            ty: self.ty,
        }
    }
}

/// Hasil pemuatan satu tile.
#[derive(Debug)]
pub struct TileResult {
    /// Permintaan asal.
    pub request: TileRequest,
    /// Tile `tile_size × tile_size` (tepi slide diisi putih), atau error.
    pub result: Result<RgbaImage, SlideError>,
}

/// Entri antrean; urutan: prioritas terkecil dulu, lalu urutan masuk.
struct Queued {
    req: TileRequest,
    seq: u64,
}

impl PartialEq for Queued {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Queued {}
impl PartialOrd for Queued {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Queued {
    fn cmp(&self, other: &Self) -> Ordering {
        // `BinaryHeap` adalah max-heap; balik urutan agar nilai kecil di puncak.
        other
            .req
            .priority
            .total_cmp(&self.req.priority)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

struct State {
    queue: BinaryHeap<Queued>,
    slides: HashMap<SlideId, Arc<dyn SlideSource>>,
    next_seq: u64,
    shutdown: bool,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    generation: AtomicU64,
    discarded: AtomicU64,
}

/// Pemuat tile dengan thread pekerja.
pub struct TileLoader {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

impl TileLoader {
    /// Buat pemuat dengan `threads` pekerja (minimal 1; ambil angkanya dari
    /// `Budgets::decode_threads`). Mengembalikan penerima hasil.
    pub fn new(threads: usize) -> (Self, Receiver<TileResult>) {
        let (tx, rx) = mpsc::channel();
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                queue: BinaryHeap::new(),
                slides: HashMap::new(),
                next_seq: 0,
                shutdown: false,
            }),
            wake: Condvar::new(),
            generation: AtomicU64::new(0),
            discarded: AtomicU64::new(0),
        });
        let workers = (0..threads.max(1))
            .map(|i| {
                let shared = Arc::clone(&shared);
                let tx = tx.clone();
                std::thread::Builder::new()
                    .name(format!("h3d-tile-{i}"))
                    .spawn(move || worker_loop(&shared, &tx))
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap_or_default();
        (Self { shared, workers }, rx)
    }

    /// Daftarkan sumber untuk `slide`; permintaan untuk slide yang tidak
    /// terdaftar menghasilkan `TileResult` berisi error.
    pub fn add_slide(&self, slide: SlideId, source: Arc<dyn SlideSource>) {
        if let Ok(mut st) = self.shared.state.lock() {
            st.slides.insert(slide, source);
        }
    }

    /// Cabut sumber `slide` dan buang permintaannya yang masih antre.
    pub fn remove_slide(&self, slide: SlideId) {
        if let Ok(mut st) = self.shared.state.lock() {
            st.slides.remove(&slide);
            let kept: Vec<Queued> = st.queue.drain().filter(|q| q.req.slide != slide).collect();
            st.queue.extend(kept);
        }
    }

    /// Generasi viewport saat ini.
    pub fn generation(&self) -> u64 {
        self.shared.generation.load(AtomicOrdering::SeqCst)
    }

    /// Tetapkan generasi viewport; permintaan dengan generasi lebih kecil
    /// akan dibuang.
    pub fn set_generation(&self, generation: u64) {
        self.shared
            .generation
            .store(generation, AtomicOrdering::SeqCst);
    }

    /// Masukkan permintaan. Mengembalikan `false` bila langsung dibuang
    /// karena generasinya sudah lewat.
    pub fn request(&self, req: TileRequest) -> bool {
        if req.generation < self.generation() {
            self.shared.discarded.fetch_add(1, AtomicOrdering::SeqCst);
            return false;
        }
        let Ok(mut st) = self.shared.state.lock() else {
            return false;
        };
        let seq = st.next_seq;
        st.next_seq += 1;
        st.queue.push(Queued { req, seq });
        drop(st);
        self.shared.wake.notify_one();
        true
    }

    /// Masukkan banyak permintaan; mengembalikan jumlah yang diterima.
    pub fn request_many<I: IntoIterator<Item = TileRequest>>(&self, reqs: I) -> usize {
        reqs.into_iter().filter(|r| self.request(*r)).count()
    }

    /// Buang semua permintaan yang masih antre.
    pub fn clear_pending(&self) {
        if let Ok(mut st) = self.shared.state.lock() {
            let n = st.queue.len() as u64;
            st.queue.clear();
            self.shared.discarded.fetch_add(n, AtomicOrdering::SeqCst);
        }
    }

    /// Jumlah permintaan yang masih antre (belum diambil pekerja).
    pub fn pending(&self) -> usize {
        self.shared
            .state
            .lock()
            .map(|st| st.queue.len())
            .unwrap_or(0)
    }

    /// Jumlah permintaan yang dibuang karena generasi lewat atau
    /// `clear_pending`.
    pub fn discarded(&self) -> u64 {
        self.shared.discarded.load(AtomicOrdering::SeqCst)
    }

    /// Jumlah thread pekerja yang hidup.
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }
}

impl Drop for TileLoader {
    fn drop(&mut self) {
        if let Ok(mut st) = self.shared.state.lock() {
            st.shutdown = true;
            st.queue.clear();
        }
        self.shared.wake.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

fn worker_loop(shared: &Shared, tx: &Sender<TileResult>) {
    loop {
        let (req, source) = {
            let Ok(mut st) = shared.state.lock() else {
                return;
            };
            loop {
                if st.shutdown {
                    return;
                }
                if let Some(q) = st.queue.pop() {
                    let source = st.slides.get(&q.req.slide).cloned();
                    break (q.req, source);
                }
                st = match shared.wake.wait(st) {
                    Ok(g) => g,
                    Err(_) => return,
                };
            }
        };

        if req.generation < shared.generation.load(AtomicOrdering::SeqCst) {
            shared.discarded.fetch_add(1, AtomicOrdering::SeqCst);
            continue;
        }

        let result = match source {
            Some(src) => read_tile(src.as_ref(), &req),
            None => Err(SlideError::InvalidArgument(format!(
                "slide {} tidak terdaftar di pemuat",
                req.slide
            ))),
        };
        if tx
            .send(TileResult {
                request: req,
                result,
            })
            .is_err()
        {
            // Penerima sudah dibuang: tidak ada gunanya melanjutkan.
            return;
        }
    }
}

/// Baca tile `(tx, ty)` pada `level`; tile tepi diisi putih oleh backend.
pub fn read_tile(src: &dyn SlideSource, req: &TileRequest) -> Result<RgbaImage, SlideError> {
    let info = src.info();
    let [cols, rows] = info.tile_grid(req.level)?;
    if req.tx >= cols || req.ty >= rows {
        return Err(SlideError::InvalidArgument(format!(
            "tile ({}, {}) di luar kisi {}x{} level {}",
            req.tx, req.ty, cols, rows, req.level
        )));
    }
    let ts = info.tile_size.max(1);
    src.read_region(
        req.level,
        i64::from(req.tx) * i64::from(ts),
        i64::from(req.ty) * i64::from(ts),
        ts,
        ts,
    )
}
