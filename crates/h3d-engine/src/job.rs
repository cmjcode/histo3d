//! Job latar (8.8).
//!
//! **Model thread yang dipilih.** [`crate::Session`] bersifat `Send`. GUI
//! menyimpannya sebagai [`SharedSession`] (`Arc<Mutex<Session>>`) dan
//! menjalankan operasi panjang lewat [`spawn_session_job`]: thread latar
//! mengunci sesi selama job, GUI memeriksa `result.try_recv()` tiap frame,
//! membaca `progress.fraction()`/`progress.stage()` untuk bilah progres, dan
//! memanggil `progress.cancel()` untuk membatalkan. Selama job berjalan GUI
//! memakai `try_lock()` (bukan `lock()`) agar thread UI tidak pernah
//! terblokir. Untuk pekerjaan yang tidak butuh sesi, pakai [`spawn_job`].

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;

use h3d_core::{Progress, SharedProgress};

use crate::error::EngineError;
use crate::session::Session;

/// Sesi yang dibagi antara thread UI dan thread job.
pub type SharedSession = Arc<Mutex<Session>>;

/// Pegangan job latar.
pub struct JobHandle<T> {
    /// Progres bersama: fraksi, tahap, dan tanda batal.
    pub progress: Arc<SharedProgress>,
    /// Hasil job; tepat satu pesan dikirim saat job selesai.
    pub result: mpsc::Receiver<Result<T, EngineError>>,
    thread: Option<JoinHandle<()>>,
}

impl<T> JobHandle<T> {
    /// Minta job berhenti pada titik periksa berikutnya.
    pub fn cancel(&self) {
        self.progress.cancel();
    }

    /// Benar bila thread job sudah berakhir.
    pub fn is_finished(&self) -> bool {
        self.thread.as_ref().is_none_or(|t| t.is_finished())
    }

    /// Tunggu sampai selesai dan ambil hasilnya (untuk CLI dan tes).
    pub fn wait(mut self) -> Result<T, EngineError> {
        let res = self
            .result
            .recv()
            .unwrap_or_else(|_| Err(EngineError::invalid("job berakhir tanpa hasil")));
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        res
    }
}

/// Jalankan `f` di thread latar. `f` menerima progres bersama dan wajib
/// memeriksa pembatalan lewat progres itu. Panik di dalam job ditangkap dan
/// dilaporkan sebagai galat, tidak menjatuhkan aplikasi.
pub fn spawn_job<T, F>(name: &str, f: F) -> JobHandle<T>
where
    T: Send + 'static,
    F: FnOnce(&SharedProgress) -> Result<T, EngineError> + Send + 'static,
{
    let progress = Arc::new(SharedProgress::new());
    let (tx, rx) = mpsc::channel();
    let p = Arc::clone(&progress);
    let job_name = name.to_owned();
    let spawn = std::thread::Builder::new()
        .name(format!("h3d-job-{name}"))
        .spawn(move || {
            let out = catch_unwind(AssertUnwindSafe(|| f(&p))).unwrap_or_else(|_| {
                Err(EngineError::invalid(format!(
                    "job \"{job_name}\" berhenti karena kesalahan internal"
                )))
            });
            // Penerima boleh sudah dibuang (GUI ditutup); itu bukan galat.
            let _ = tx.send(out);
        });
    match spawn {
        Ok(thread) => JobHandle {
            progress,
            result: rx,
            thread: Some(thread),
        },
        Err(e) => {
            // Thread tidak bisa dibuat: kirim galat lewat kanal cadangan.
            let (tx2, rx2) = mpsc::channel();
            let _ = tx2.send(Err(EngineError::invalid(format!(
                "tidak bisa membuat thread job: {e}"
            ))));
            JobHandle {
                progress,
                result: rx2,
                thread: None,
            }
        }
    }
}

/// Jalankan operasi sesi di thread latar. Sesi dikunci selama job.
pub fn spawn_session_job<T, F>(session: &SharedSession, name: &str, f: F) -> JobHandle<T>
where
    T: Send + 'static,
    F: FnOnce(&mut Session, &dyn Progress) -> Result<T, EngineError> + Send + 'static,
{
    let session = Arc::clone(session);
    spawn_job(name, move |progress| {
        let mut guard = session
            .lock()
            .map_err(|_| EngineError::invalid("sesi rusak akibat job sebelumnya yang gagal"))?;
        f(&mut guard, progress)
    })
}
