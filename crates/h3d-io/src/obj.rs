//! Ekspor Wavefront OBJ: semua label dalam satu berkas, satu grup per label,
//! satuan mm (8.7.3).

use std::fmt::Write as _;

use h3d_volume::LabelMesh;

use crate::error::IoError;
use crate::meshconv::{
    file_stem_of, flipped_triangles, normal_to_mm_frame, validate_mesh, w_um_to_mm,
};

/// Tulis OBJ. `meshes` = `(nama label, mesh)`. Format angka tetap (6 desimal)
/// agar keluaran deterministik.
pub fn encode_obj(meshes: &[(&str, &LabelMesh)]) -> Result<String, IoError> {
    let mut out = String::new();
    out.push_str("# HISTO3D\n");
    let _ = writeln!(out, "# {}", h3d_core::RESEARCH_ONLY_NOTICE);
    out.push_str("# units: mm\n");
    // Indeks OBJ mulai dari 1 dan bersifat global; `v` dan `vn` punya
    // pencacah sendiri karena mesh tanpa normal tidak menulis `vn`.
    let mut offset: u64 = 1;
    let mut n_offset: u64 = 1;
    for (name, mesh) in meshes {
        validate_mesh(mesh)?;
        if mesh.indices.is_empty() {
            continue;
        }
        let group = file_stem_of(name, &format!("label{}", mesh.label));
        let _ = writeln!(out, "o {group}");
        let _ = writeln!(out, "g {group}");
        for p in &mesh.positions_um {
            let q = w_um_to_mm(*p);
            let _ = writeln!(out, "v {:.6} {:.6} {:.6}", q[0], q[1], q[2]);
        }
        let has_normals = mesh.normals.len() == mesh.positions_um.len()
            && mesh.normals.iter().all(|n| n.iter().all(|v| v.is_finite()));
        if has_normals {
            for n in &mesh.normals {
                let q = normal_to_mm_frame(*n);
                let _ = writeln!(out, "vn {:.6} {:.6} {:.6}", q[0], q[1], q[2]);
            }
        }
        for t in flipped_triangles(mesh) {
            let a = u64::from(t[0]) + offset;
            let b = u64::from(t[1]) + offset;
            let c = u64::from(t[2]) + offset;
            if has_normals {
                let na = u64::from(t[0]) + n_offset;
                let nb = u64::from(t[1]) + n_offset;
                let nc = u64::from(t[2]) + n_offset;
                let _ = writeln!(out, "f {a}//{na} {b}//{nb} {c}//{nc}");
            } else {
                let _ = writeln!(out, "f {a} {b} {c}");
            }
        }
        offset += mesh.positions_um.len() as u64;
        if has_normals {
            n_offset += mesh.normals.len() as u64;
        }
    }
    Ok(out)
}
