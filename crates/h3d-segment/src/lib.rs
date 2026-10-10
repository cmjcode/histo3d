//! `h3d-segment`: penyimpanan label, alat manual, segmentasi klasik dan AI,
//! interpolasi bentuk, dan pertukaran GeoJSON (histo3d.md 8.5, 8.7.2, 8.12.5).
//!
//! Tanpa GUI dan tanpa I/O berkas proyek: format `.lbl`/`.atr` hanya
//! dikodekan/didekodekan ke byte (penulis bundel ada di `h3d-io`); satu-satunya
//! akses berkas adalah memuat model ONNX (`TractSegmenter::load`).
//!
//! # Konvensi
//! - Koordinat µm `f64` di ruang U (slide); pusat piksel di `+0,5`;
//!   indeks piksel dari µm = `floor(u / mpp)`.
//! - Hasil segmentasi otomatis dan interpolasi adalah [`Proposal`]; hanya
//!   [`accept_proposal`] yang menuliskannya ke [`LabelStore`].
//! - Transformasi: tiap fungsi menyebut arahnya di doc comment; resampling
//!   selalu mundur.
//! - Deterministik: iterasi tile selalu terurut `(ty, tx)`; inferensi paralel
//!   digabung dalam urutan tetap.

pub mod ai;
pub mod attr_store;
pub mod classic;
pub mod commands;
pub mod error;
pub mod geojson;
pub mod interpolate;
pub mod label_store;
pub mod lbl_format;
pub mod proposal;

pub use ai::{infer_slide, AiResult, ClassSpec, ModelSpec, Segmenter, TractSegmenter};
pub use attr_store::{decode_atr, encode_atr, AttrStore, NODATA};
pub use classic::{segment_classic, ClassicParams, StainChannel};
pub use commands::{LabelEditCommand, TileSnapshot};
pub use error::SegError;
pub use geojson::{export_geojson, import_geojson, ImportSummary};
pub use interpolate::{interpolate_between, interpolate_label};
pub use label_store::{LabelStore, PaintMode, TileKey};
pub use lbl_format::{decode_lbl, encode_lbl};
pub use proposal::{accept_proposal, Proposal};
