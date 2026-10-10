//! Berkas cache `cache/vox/`: masker voxel per label dan mesh hasil
//! rekonstruksi (8.8). Cache boleh dihapus kapan saja; berkas rusak → galat
//! (pemanggil lalu membangun ulang).
//!
//! `.vox`: `"H3DV"`, versi u32, panjang JSON u32, JSON (kisi + asal lapis),
//! lalu data voxel lz4 (size-prepended).
//! `.mesh`: `"H3DM"`, versi u32, panjang JSON u32, JSON (label, jumlah,
//! daftar skalar), lalu payload lz4: posisi f64, normal f32, indeks u32,
//! nilai skalar f32 (semuanya little-endian).

use std::path::Path;

use h3d_core::{LabelId, SlideId};
use h3d_volume::{LabelMesh, MeshScalars, StackGrid, VoxelMask};
use serde::{Deserialize, Serialize};

use crate::error::IoError;

const VOX_MAGIC: &[u8; 4] = b"H3DV";
const MESH_MAGIC: &[u8; 4] = b"H3DM";
const CACHE_VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
struct VoxHeader {
    grid: StackGrid,
    annotated_layer: Vec<Option<(SlideId, bool)>>,
}

#[derive(Serialize, Deserialize)]
struct MeshHeader {
    label: u8,
    n_vertices: usize,
    n_normals: usize,
    n_indices: usize,
    scalars: Vec<(String, String)>,
}

fn frame(magic: &[u8; 4], header_json: &[u8], payload: &[u8]) -> Vec<u8> {
    let compressed = lz4_flex::compress_prepend_size(payload);
    let mut out = Vec::with_capacity(12 + header_json.len() + compressed.len());
    out.extend_from_slice(magic);
    out.extend_from_slice(&CACHE_VERSION.to_le_bytes());
    out.extend_from_slice(&(header_json.len() as u32).to_le_bytes());
    out.extend_from_slice(header_json);
    out.extend_from_slice(&compressed);
    out
}

fn unframe<'a>(
    magic: &[u8; 4],
    bytes: &'a [u8],
    path: &Path,
) -> Result<(&'a [u8], Vec<u8>), IoError> {
    if bytes.len() < 12 || &bytes[0..4] != magic {
        return Err(IoError::invalid(
            path,
            "magic cache salah atau berkas terpotong",
        ));
    }
    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if version != CACHE_VERSION {
        return Err(IoError::invalid(
            path,
            format!("versi cache {version} tidak dikenal"),
        ));
    }
    let json_len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    let json_end = 12usize
        .checked_add(json_len)
        .filter(|e| *e <= bytes.len())
        .ok_or_else(|| IoError::invalid(path, "header cache melewati akhir berkas"))?;
    let payload = lz4_flex::decompress_size_prepended(&bytes[json_end..])
        .map_err(|e| IoError::invalid(path, format!("payload cache rusak: {e}")))?;
    Ok((&bytes[12..json_end], payload))
}

/// Serialisasi masker voxel.
pub fn encode_vox(mask: &VoxelMask) -> Result<Vec<u8>, IoError> {
    let header = VoxHeader {
        grid: mask.grid.clone(),
        annotated_layer: mask.annotated_layer.clone(),
    };
    let json = serde_json::to_vec(&header).map_err(|e| IoError::json(Path::new("cache.vox"), e))?;
    Ok(frame(VOX_MAGIC, &json, &mask.data))
}

/// Baca masker voxel; ukuran data harus cocok dengan kisi.
pub fn decode_vox(bytes: &[u8], path: &Path) -> Result<VoxelMask, IoError> {
    let (json, data) = unframe(VOX_MAGIC, bytes, path)?;
    let header: VoxHeader = serde_json::from_slice(json).map_err(|e| IoError::json(path, e))?;
    let expected = (header.grid.size[0] as usize)
        .checked_mul(header.grid.size[1] as usize)
        .and_then(|n| n.checked_mul(header.grid.z_um.len()))
        .ok_or_else(|| IoError::invalid(path, "ukuran kisi tidak masuk akal"))?;
    if data.len() != expected || header.annotated_layer.len() != header.grid.z_um.len() {
        return Err(IoError::invalid(
            path,
            "ukuran data tidak cocok dengan kisi",
        ));
    }
    Ok(VoxelMask {
        grid: header.grid,
        data,
        annotated_layer: header.annotated_layer,
    })
}

/// Serialisasi mesh (termasuk skalar per verteks).
pub fn encode_mesh(mesh: &LabelMesh) -> Result<Vec<u8>, IoError> {
    let n = mesh.positions_um.len();
    let scalars: Vec<&MeshScalars> = mesh
        .scalars
        .iter()
        .filter(|s| s.values.len() == n)
        .collect();
    let header = MeshHeader {
        label: mesh.label.0,
        n_vertices: n,
        n_normals: mesh.normals.len(),
        n_indices: mesh.indices.len(),
        scalars: scalars
            .iter()
            .map(|s| (s.name.clone(), s.unit.clone()))
            .collect(),
    };
    let json =
        serde_json::to_vec(&header).map_err(|e| IoError::json(Path::new("cache.mesh"), e))?;
    let mut payload = Vec::with_capacity(
        n * 24 + mesh.normals.len() * 12 + mesh.indices.len() * 4 + scalars.len() * n * 4,
    );
    for p in &mesh.positions_um {
        for c in p {
            payload.extend_from_slice(&c.to_le_bytes());
        }
    }
    for nrm in &mesh.normals {
        for c in nrm {
            payload.extend_from_slice(&c.to_le_bytes());
        }
    }
    for i in &mesh.indices {
        payload.extend_from_slice(&i.to_le_bytes());
    }
    for s in &scalars {
        for v in &s.values {
            payload.extend_from_slice(&v.to_le_bytes());
        }
    }
    Ok(frame(MESH_MAGIC, &json, &payload))
}

/// Baca mesh dari cache dengan pemeriksaan batas penuh.
pub fn decode_mesh(bytes: &[u8], path: &Path) -> Result<LabelMesh, IoError> {
    let (json, payload) = unframe(MESH_MAGIC, bytes, path)?;
    let h: MeshHeader = serde_json::from_slice(json).map_err(|e| IoError::json(path, e))?;
    let bad = || IoError::invalid(path, "ukuran payload mesh tidak cocok dengan header");
    let need = h
        .n_vertices
        .checked_mul(24)
        .and_then(|a| h.n_normals.checked_mul(12).and_then(|b| a.checked_add(b)))
        .and_then(|a| h.n_indices.checked_mul(4).and_then(|b| a.checked_add(b)))
        .and_then(|a| {
            h.scalars
                .len()
                .checked_mul(h.n_vertices)
                .and_then(|s| s.checked_mul(4))
                .and_then(|b| a.checked_add(b))
        })
        .ok_or_else(bad)?;
    if payload.len() != need || h.n_indices % 3 != 0 {
        return Err(bad());
    }
    let mut o = 0usize;
    let f64_next = |o: &mut usize| {
        let mut b = [0u8; 8];
        b.copy_from_slice(&payload[*o..*o + 8]);
        *o += 8;
        f64::from_le_bytes(b)
    };
    let mut positions_um = Vec::with_capacity(h.n_vertices);
    for _ in 0..h.n_vertices {
        positions_um.push([f64_next(&mut o), f64_next(&mut o), f64_next(&mut o)]);
    }
    let f32_next = |o: &mut usize| {
        let mut b = [0u8; 4];
        b.copy_from_slice(&payload[*o..*o + 4]);
        *o += 4;
        f32::from_le_bytes(b)
    };
    let mut normals = Vec::with_capacity(h.n_normals);
    for _ in 0..h.n_normals {
        normals.push([f32_next(&mut o), f32_next(&mut o), f32_next(&mut o)]);
    }
    let mut indices = Vec::with_capacity(h.n_indices);
    for _ in 0..h.n_indices {
        let mut b = [0u8; 4];
        b.copy_from_slice(&payload[o..o + 4]);
        o += 4;
        let i = u32::from_le_bytes(b);
        if i as usize >= h.n_vertices {
            return Err(IoError::invalid(path, "indeks mesh di luar rentang"));
        }
        indices.push(i);
    }
    let mut scalars = Vec::with_capacity(h.scalars.len());
    for (name, unit) in h.scalars {
        let mut values = Vec::with_capacity(h.n_vertices);
        for _ in 0..h.n_vertices {
            values.push(f32_next(&mut o));
        }
        scalars.push(MeshScalars { name, unit, values });
    }
    Ok(LabelMesh {
        label: LabelId(h.label),
        positions_um,
        normals,
        indices,
        scalars,
    })
}
