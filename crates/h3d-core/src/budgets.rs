//! Anggaran memori dan thread (9.2). Satu-satunya tempat angka anggaran
//! boleh ditulis (aturan keras 15).

use serde::{Deserialize, Serialize};

/// Anggaran per platform, dipilih saat aplikasi mulai.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Budgets {
    /// Cache tile CPU (RGBA8) dalam byte.
    pub tile_cache_cpu_bytes: u64,
    /// Cache tile GPU dalam byte.
    pub tile_cache_gpu_bytes: u64,
    /// Set kerja rekonstruksi (SDF, masker voxel) dalam byte.
    pub recon_working_set_bytes: u64,
    /// Jumlah thread pendekode tile.
    pub decode_threads: usize,
    /// Cache tile internal `wsi-rs` **per slide terbuka** dalam byte. Cache
    /// utama ada di `h3d_slide::TileCache`; ini hanya penyangga dekode.
    pub wsi_internal_cache_bytes: u64,
    /// Jumlah slide yang boleh terbuka bersamaan di pemuat tile (tiap slide
    /// memegang handle berkas dan cache `wsi-rs`). Minimal 2.
    pub max_open_slides: usize,
    /// Jumlah pasangan registrasi per gelombang (membatasi berapa gambar
    /// registrasi berada di memori bersamaan). Minimal 1.
    pub register_batch: usize,
    /// Jumlah piramida bertile yang dibangun bersamaan oleh job "Menyiapkan
    /// piramida gambar" (tiap pembangunan memegang satu pendekode gambar
    /// yang dialirkan per pita). Minimal 1.
    pub pyramid_build_jobs: usize,
}

const MIB: u64 = 1024 * 1024;

impl Budgets {
    /// Anggaran desktop macOS (9.2) **[TARGET]**.
    pub const MACOS: Budgets = Budgets {
        tile_cache_cpu_bytes: 2048 * MIB,
        tile_cache_gpu_bytes: 1024 * MIB,
        recon_working_set_bytes: 4096 * MIB,
        decode_threads: 4,
        wsi_internal_cache_bytes: 16 * MIB,
        max_open_slides: 16,
        register_batch: 8,
        pyramid_build_jobs: 2,
    };

    /// Anggaran iPad (9.2) **[TARGET]**; harus diukur ulang di perangkat.
    pub const IPAD: Budgets = Budgets {
        tile_cache_cpu_bytes: 384 * MIB,
        tile_cache_gpu_bytes: 256 * MIB,
        recon_working_set_bytes: 1024 * MIB,
        decode_threads: 2,
        // Titik awal; ukur di perangkat (H0.7) sebelum diturunkan.
        wsi_internal_cache_bytes: 16 * MIB,
        max_open_slides: 4,
        register_batch: 2,
        pyramid_build_jobs: 1,
    };

    /// Anggaran bawaan menurut target kompilasi.
    pub fn for_platform() -> Budgets {
        if cfg!(target_os = "ios") {
            Budgets::IPAD
        } else {
            Budgets::MACOS
        }
    }

    /// Skala anggaran menurut memori fisik (byte); tidak pernah melebihi
    /// anggaran platform dan tidak kurang dari seperempatnya. Jumlah
    /// (`decode_threads`, `max_open_slides`, `register_batch`,
    /// `pyramid_build_jobs`) tidak diskalakan.
    pub fn scaled_to_physical_memory(self, physical_bytes: u64) -> Budgets {
        // Asumsi: aplikasi boleh memakai sekitar setengah memori fisik.
        let allowed = physical_bytes / 2;
        let planned =
            self.tile_cache_cpu_bytes + self.tile_cache_gpu_bytes + self.recon_working_set_bytes;
        if planned <= allowed || planned == 0 {
            return self;
        }
        let f = (allowed as f64 / planned as f64).max(0.25);
        let s = |b: u64| ((b as f64) * f) as u64;
        Budgets {
            tile_cache_cpu_bytes: s(self.tile_cache_cpu_bytes),
            tile_cache_gpu_bytes: s(self.tile_cache_gpu_bytes),
            recon_working_set_bytes: s(self.recon_working_set_bytes),
            decode_threads: self.decode_threads,
            wsi_internal_cache_bytes: s(self.wsi_internal_cache_bytes),
            max_open_slides: self.max_open_slides,
            register_batch: self.register_batch,
            pyramid_build_jobs: self.pyramid_build_jobs,
        }
    }
}

impl Default for Budgets {
    fn default() -> Self {
        Budgets::for_platform()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_stay_usable_after_scaling() {
        for base in [Budgets::MACOS, Budgets::IPAD] {
            assert!(base.max_open_slides >= 2);
            assert!(base.register_batch >= 1);
            assert!(base.pyramid_build_jobs >= 1);
            assert!(base.wsi_internal_cache_bytes > 0);
            for gib in [1u64, 2, 4, 8, 64] {
                let s = base.scaled_to_physical_memory(gib * 1024 * MIB);
                assert_eq!(s.max_open_slides, base.max_open_slides);
                assert_eq!(s.register_batch, base.register_batch);
                assert!(s.wsi_internal_cache_bytes > 0);
                assert!(s.wsi_internal_cache_bytes <= base.wsi_internal_cache_bytes);
            }
        }
    }
}
