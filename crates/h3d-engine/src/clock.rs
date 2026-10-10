//! Jam yang bisa disuntik. `AnnotationRecord::at` dan `ReviewStatus::at`
//! adalah satu-satunya tempat mesin memakai jam; lewat trait ini tes dan
//! jalan emas CLI bisa menetapkannya (env `H3D_FIXED_TIME`) sehingga
//! keluaran tetap byte-identik (8.8 "Determinisme").

use std::sync::Arc;

/// Sumber waktu untuk catatan provenance.
pub trait Clock: Send + Sync {
    /// Waktu sekarang, ISO-8601 dengan zona lokal.
    fn now_iso8601(&self) -> String;
}

/// Jam sistem (zona lokal).
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_iso8601(&self) -> String {
        chrono::Local::now()
            .format("%Y-%m-%dT%H:%M:%S%:z")
            .to_string()
    }
}

/// Jam beku: selalu mengembalikan teks yang sama.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedClock(pub String);

impl Clock for FixedClock {
    fn now_iso8601(&self) -> String {
        self.0.clone()
    }
}

/// Nama variabel lingkungan untuk membekukan jam.
pub const FIXED_TIME_ENV: &str = "H3D_FIXED_TIME";

/// Jam dari lingkungan: [`FixedClock`] bila `H3D_FIXED_TIME` terisi, selain
/// itu [`SystemClock`]. Dipanggil CLI; mesin sendiri tidak membaca
/// lingkungan secara diam-diam.
pub fn clock_from_env() -> Arc<dyn Clock> {
    match std::env::var(FIXED_TIME_ENV) {
        Ok(v) if !v.trim().is_empty() => Arc::new(FixedClock(v.trim().to_owned())),
        _ => Arc::new(SystemClock),
    }
}
