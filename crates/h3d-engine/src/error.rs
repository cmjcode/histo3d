//! Galat `h3d-engine`.

use h3d_core::SlideId;
use thiserror::Error;

/// Galat lapisan headless. Membungkus galat crate di bawahnya; pembatalan
/// dari crate mana pun dinormalkan menjadi [`EngineError::Cancelled`].
#[derive(Debug, Error)]
pub enum EngineError {
    /// Slide tidak membawa metadata mpp dan pengguna tidak mengisinya.
    /// mpp tidak pernah ditebak (aturan keras 7).
    #[error(
        "mpp (mikrometer per piksel) tidak ada di metadata berkas {file}; \
         isi lewat --mpp atau di panel irisan"
    )]
    MppMissing {
        /// Nama berkas slide.
        file: String,
    },
    /// Job dibatalkan pengguna.
    #[error("dibatalkan")]
    Cancelled,
    /// Operasi tidak sah untuk keadaan proyek saat ini.
    #[error("{0}")]
    InvalidState(String),
    /// Irisan tidak ada di proyek.
    #[error("irisan {0} tidak ada di proyek")]
    SlideNotFound(SlideId),
    /// Berkas slide `External` tidak lagi sama dengan saat diimpor (ukuran
    /// atau sidik jari berbeda). Pikselnya tidak dibaca sampai pengguna
    /// menautkan ulang berkas aslinya (`Session::relink_slide`) atau menerima
    /// berkas baru (`Session::accept_changed_slide`).
    #[error(
        "berkas slide irisan {slide} ({file}) berubah sejak diimpor (ukuran atau sidik jari \
         berbeda); tautkan ulang berkas aslinya atau terima berkas baru sebelum membacanya"
    )]
    SlideChanged {
        /// Irisan.
        slide: SlideId,
        /// Nama berkas.
        file: String,
    },
    /// Label tidak ada di tabel label.
    #[error("label \"{0}\" tidak ada di tabel label")]
    LabelNotFound(String),
    /// Galat bundel/ekspor.
    #[error(transparent)]
    Io(#[from] h3d_io::IoError),
    /// Galat tipe inti.
    #[error(transparent)]
    Core(#[from] h3d_core::CoreError),
    /// Galat pembaca slide.
    #[error(transparent)]
    Slide(#[from] h3d_slide::SlideError),
    /// Galat pemrosesan citra.
    #[error("pemrosesan citra: {0}")]
    Img(h3d_imgproc::ImgError),
    /// Galat registrasi.
    #[error("registrasi: {0}")]
    Reg(h3d_register::RegError),
    /// Galat segmentasi/label.
    #[error("segmentasi: {0}")]
    Seg(h3d_segment::SegError),
    /// Galat rekonstruksi/analisis volume.
    #[error("rekonstruksi: {0}")]
    Vol(h3d_volume::VolError),
}

impl EngineError {
    /// Pembantu `InvalidState`.
    pub fn invalid(msg: impl Into<String>) -> Self {
        EngineError::InvalidState(msg.into())
    }
}

impl From<h3d_imgproc::ImgError> for EngineError {
    fn from(e: h3d_imgproc::ImgError) -> Self {
        match e {
            h3d_imgproc::ImgError::Cancelled => EngineError::Cancelled,
            other => EngineError::Img(other),
        }
    }
}

impl From<h3d_register::RegError> for EngineError {
    fn from(e: h3d_register::RegError) -> Self {
        match e {
            h3d_register::RegError::Cancelled => EngineError::Cancelled,
            h3d_register::RegError::Img(h3d_imgproc::ImgError::Cancelled) => EngineError::Cancelled,
            other => EngineError::Reg(other),
        }
    }
}

impl From<h3d_segment::SegError> for EngineError {
    fn from(e: h3d_segment::SegError) -> Self {
        match e {
            h3d_segment::SegError::Cancelled => EngineError::Cancelled,
            h3d_segment::SegError::ImgProc(h3d_imgproc::ImgError::Cancelled) => {
                EngineError::Cancelled
            }
            other => EngineError::Seg(other),
        }
    }
}

impl From<h3d_volume::VolError> for EngineError {
    fn from(e: h3d_volume::VolError) -> Self {
        match e {
            h3d_volume::VolError::Cancelled => EngineError::Cancelled,
            h3d_volume::VolError::Img(h3d_imgproc::ImgError::Cancelled) => EngineError::Cancelled,
            other => EngineError::Vol(other),
        }
    }
}
