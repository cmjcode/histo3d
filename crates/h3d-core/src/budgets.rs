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
}

const MIB: u64 = 1024 * 1024;

impl Budgets {
    /// Anggaran desktop macOS (9.2) **[TARGET]**.
    pub const MACOS: Budgets = Budgets {
        tile_cache_cpu_bytes: 2048 * MIB,
        tile_cache_gpu_bytes: 1024 * MIB,
        recon_working_set_bytes: 4096 * MIB,
        decode_threads: 4,
    };

    /// Anggaran iPad (9.2) **[TARGET]**; harus diukur ulang di perangkat.
    pub const IPAD: Budgets = Budgets {
        tile_cache_cpu_bytes: 384 * MIB,
        tile_cache_gpu_bytes: 256 * MIB,
        recon_working_set_bytes: 1024 * MIB,
        decode_threads: 2,
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
    /// anggaran platform dan tidak kurang dari seperempatnya.
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
        }
    }
}

impl Default for Budgets {
    fn default() -> Self {
        Budgets::for_platform()
    }
}
