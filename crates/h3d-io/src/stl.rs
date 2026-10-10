//! Ekspor STL biner: satu label per berkas, satuan **mm** (8.7.3). STL tidak
//! menyimpan satuan, jadi satuannya ditulis di nama berkas dan di header.

use h3d_volume::LabelMesh;

use crate::error::IoError;
use crate::meshconv::{file_stem_of, flipped_triangles, validate_mesh, w_um_to_mm};

/// Panjang header STL biner.
pub const STL_HEADER_LEN: usize = 80;

/// Nama berkas STL untuk sebuah label: `<nama>_mm.stl`.
pub fn stl_file_name(label_name: &str, label_id: u8) -> String {
    format!(
        "{}_mm.stl",
        file_stem_of(label_name, &format!("label{label_id}"))
    )
}

/// Tulis STL biner (mm, tangan-kanan, Z ke atas). Normal per segitiga
/// dihitung dari verteks sehingga konsisten dengan orientasinya.
pub fn encode_stl(mesh: &LabelMesh) -> Result<Vec<u8>, IoError> {
    validate_mesh(mesh)?;
    let n_tri = u32::try_from(mesh.indices.len() / 3)
        .map_err(|_| IoError::InvalidArgument("terlalu banyak segitiga untuk STL".into()))?;
    let mut out = Vec::with_capacity(STL_HEADER_LEN + 4 + n_tri as usize * 50);
    let mut header = [b' '; STL_HEADER_LEN];
    // Header tidak boleh diawali "solid" (penanda STL ASCII).
    let text = format!("HISTO3D binary STL units=mm label={}", mesh.label);
    let n = text.len().min(STL_HEADER_LEN);
    header[..n].copy_from_slice(&text.as_bytes()[..n]);
    out.extend_from_slice(&header);
    out.extend_from_slice(&n_tri.to_le_bytes());
    for t in flipped_triangles(mesh) {
        let p: [[f64; 3]; 3] = [
            w_um_to_mm(mesh.positions_um[t[0] as usize]),
            w_um_to_mm(mesh.positions_um[t[1] as usize]),
            w_um_to_mm(mesh.positions_um[t[2] as usize]),
        ];
        let u = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
        let v = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
        let mut nrm = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        let len = (nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]).sqrt();
        if len > 0.0 {
            for c in &mut nrm {
                *c /= len;
            }
        }
        for c in nrm {
            out.extend_from_slice(&(c as f32).to_le_bytes());
        }
        for vert in p {
            for c in vert {
                out.extend_from_slice(&(c as f32).to_le_bytes());
            }
        }
        out.extend_from_slice(&0u16.to_le_bytes());
    }
    Ok(out)
}

/// Baca kembali STL biner: daftar segitiga `[[x, y, z]; 3]` (mm). Dipakai
/// tes; berkas terpotong → galat.
pub fn decode_stl(bytes: &[u8]) -> Result<Vec<[[f32; 3]; 3]>, IoError> {
    let p = std::path::Path::new("mesh.stl");
    if bytes.len() < STL_HEADER_LEN + 4 {
        return Err(IoError::invalid(p, "berkas terpotong"));
    }
    let n = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
    let expected = n
        .checked_mul(50)
        .and_then(|b| b.checked_add(STL_HEADER_LEN + 4))
        .ok_or_else(|| IoError::invalid(p, "jumlah segitiga tidak masuk akal"))?;
    if bytes.len() != expected {
        return Err(IoError::invalid(
            p,
            "panjang berkas tidak cocok dengan jumlah segitiga",
        ));
    }
    let f = |o: usize| f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
    let mut tris = Vec::with_capacity(n);
    for i in 0..n {
        let base = STL_HEADER_LEN + 4 + i * 50 + 12;
        let mut t = [[0.0f32; 3]; 3];
        for (v, vert) in t.iter_mut().enumerate() {
            for (c, comp) in vert.iter_mut().enumerate() {
                *comp = f(base + v * 12 + c * 4);
            }
        }
        tris.push(t);
    }
    Ok(tris)
}
