//! Berkas `out/tracks.json`: hasil penautan objek antar irisan
//! (`h3d_volume::track::TrackSet`) per label. Ditulis atomik lewat
//! [`ProjectBundle::save_tracks`](crate::ProjectBundle::save_tracks);
//! berkas rusak atau versi lebih baru → galat, bukan panik.

use std::path::Path;

use h3d_core::LabelId;
use h3d_volume::TrackSet;
use serde::{Deserialize, Serialize};

use crate::analysis_json::to_stable_json_of;
use crate::error::IoError;

/// Versi format `tracks.json` yang ditulis crate ini.
pub const TRACKS_FORMAT_VERSION: u32 = 1;
/// Nama berkas.
pub const TRACKS_FILE: &str = "tracks.json";

/// Isi `tracks.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TracksFile {
    /// Versi format.
    pub format_version: u32,
    /// Versi aplikasi penulis.
    #[serde(default)]
    pub app_version: String,
    /// Satu set per label, terurut id label.
    pub sets: Vec<TrackSet>,
}

impl TracksFile {
    /// Berkas kosong.
    pub fn new(app_version: &str) -> Self {
        TracksFile {
            format_version: TRACKS_FORMAT_VERSION,
            app_version: app_version.to_owned(),
            sets: Vec::new(),
        }
    }

    /// Set untuk `label`.
    pub fn set(&self, label: LabelId) -> Option<&TrackSet> {
        self.sets.iter().find(|s| s.label == label)
    }

    /// Ganti/tambah set `label`; urutan tetap menurut id label.
    pub fn put(&mut self, set: TrackSet) {
        self.sets.retain(|s| s.label != set.label);
        self.sets.push(set);
        self.sets.sort_by_key(|s| s.label.0);
    }
}

/// Serialisasi stabil (kunci terurut, format bilangan tetap) + baris baru.
pub fn encode_tracks(file: &TracksFile) -> Result<Vec<u8>, IoError> {
    let mut text = to_stable_json_of(file)?;
    text.push('\n');
    Ok(text.into_bytes())
}

/// Baca `tracks.json`; versi lebih baru ditolak dengan pesan jelas.
pub fn decode_tracks(bytes: &[u8], path: &Path) -> Result<TracksFile, IoError> {
    let file: TracksFile = serde_json::from_slice(bytes).map_err(|e| IoError::json(path, e))?;
    if file.format_version > TRACKS_FORMAT_VERSION {
        return Err(IoError::invalid(
            path,
            format!(
                "tracks.json versi {} lebih baru dari yang didukung ({})",
                file.format_version, TRACKS_FORMAT_VERSION
            ),
        ));
    }
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProjectBundle;
    use h3d_volume::{ObjectTrack, TrackMember, TrackParams};

    fn sample(label: u8) -> TrackSet {
        TrackSet {
            label: LabelId(label),
            params: TrackParams::default(),
            params_hash: 7,
            min_iou: 0.2,
            max_gap_um: 250.0,
            s_xy_um: 16.0,
            tracks: vec![ObjectTrack {
                instance_id: 1,
                label: LabelId(label),
                members: vec![TrackMember {
                    slide: h3d_core::SlideId(3),
                    component: 1,
                    area_um2: 256.0,
                    centroid_um: [10.5, 20.25],
                    z_um: -50.0,
                }],
                events: Vec::new(),
            }],
            warnings: Vec::new(),
        }
    }

    #[test]
    fn save_and_load_tracks_atomically() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("p.h3d");
        let b = ProjectBundle::create(&root).expect("bundle");
        assert!(b.load_tracks().expect("kosong").is_none());
        let mut f = TracksFile::new("0.0.0");
        f.put(sample(2));
        f.put(sample(1));
        b.save_tracks(&f).expect("tulis");
        assert!(!crate::atomic::tmp_path(&b.tracks_json()).exists());
        let back = b.load_tracks().expect("baca").expect("ada");
        assert_eq!(back, f);
        assert_eq!(
            back.sets.iter().map(|s| s.label.0).collect::<Vec<_>>(),
            [1, 2]
        );
        // Penulisan ulang identik byte demi byte.
        let a = encode_tracks(&f).expect("enc");
        let c = encode_tracks(&back).expect("enc");
        assert_eq!(a, c);
        // Versi lebih baru ditolak.
        let mut newer = f.clone();
        newer.format_version = TRACKS_FORMAT_VERSION + 1;
        let bytes = encode_tracks(&newer).expect("enc");
        assert!(decode_tracks(&bytes, Path::new("x")).is_err());
        assert!(decode_tracks(b"{", Path::new("x")).is_err());
    }
}
