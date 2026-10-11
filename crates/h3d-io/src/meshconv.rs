//! Konversi kerangka koordinat mesh untuk ekspor.
//!
//! Mesh disimpan di ruang W (µm): x ke kanan, y ke **bawah**, z arah
//! tumpukan. Kerangka itu tangan-kiri, jadi semua format ekspor memakai
//! konvensi render (histo3d.md 5.2, ruang R): y dibalik agar tangan-kanan
//! dengan Z ke atas. Pembalikan satu sumbu membalik orientasi segitiga,
//! sehingga urutan indeks ditukar agar normal tetap menghadap ke luar.

use h3d_volume::LabelMesh;

use crate::error::IoError;

/// µm per mm.
pub const UM_PER_MM: f64 = 1000.0;

/// W (µm) → milimeter tangan-kanan, Z ke atas: `(x, −y, z) / 1000`.
#[inline]
pub fn w_um_to_mm(p: [f64; 3]) -> [f64; 3] {
    [p[0] / UM_PER_MM, -p[1] / UM_PER_MM, p[2] / UM_PER_MM]
}

/// Normal W → kerangka mm tangan-kanan.
#[inline]
pub fn normal_to_mm_frame(n: [f32; 3]) -> [f32; 3] {
    [n[0], -n[1], n[2]]
}

/// W (µm) → meter glTF (Y ke atas, tangan-kanan): `(x, z, y) × 1e−6`.
/// Sama dengan kerangka mm di atas yang diputar agar Z-atas menjadi Y-atas.
#[inline]
pub fn w_um_to_gltf_m(p: [f64; 3]) -> [f32; 3] {
    [
        (p[0] * 1e-6) as f32,
        (p[2] * 1e-6) as f32,
        (p[1] * 1e-6) as f32,
    ]
}

/// Normal W → kerangka glTF.
#[inline]
pub fn normal_to_gltf(n: [f32; 3]) -> [f32; 3] {
    [n[0], n[2], n[1]]
}

/// Periksa konsistensi mesh sebelum ditulis: jumlah indeks kelipatan tiga
/// dan semua indeks menunjuk verteks yang ada.
pub fn validate_mesh(mesh: &LabelMesh) -> Result<(), IoError> {
    if !mesh.indices.len().is_multiple_of(3) {
        return Err(IoError::InvalidArgument(format!(
            "jumlah indeks mesh label {} bukan kelipatan tiga",
            mesh.label
        )));
    }
    let n = mesh.positions_um.len();
    if mesh.indices.iter().any(|&i| i as usize >= n) {
        return Err(IoError::InvalidArgument(format!(
            "indeks mesh label {} menunjuk verteks di luar rentang",
            mesh.label
        )));
    }
    if mesh
        .positions_um
        .iter()
        .any(|p| p.iter().any(|v| !v.is_finite()))
    {
        return Err(IoError::InvalidArgument(format!(
            "mesh label {} memuat koordinat tak hingga",
            mesh.label
        )));
    }
    Ok(())
}

/// Segitiga mesh dengan orientasi sudah ditukar untuk kerangka tangan-kanan:
/// `[a, c, b]`.
pub fn flipped_triangles(mesh: &LabelMesh) -> impl Iterator<Item = [u32; 3]> + '_ {
    mesh.indices.chunks_exact(3).map(|t| [t[0], t[2], t[1]])
}

/// Nama berkas aman dari nama label: huruf kecil ASCII, angka, `-`, `_`.
pub fn file_stem_of(name: &str, fallback: &str) -> String {
    let s: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    let t = s.trim_matches('_');
    if t.is_empty() {
        fallback.to_owned()
    } else {
        t.to_owned()
    }
}
