//! Ekspor GLB (glTF 2.0 biner) tulis tangan: chunk JSON + chunk BIN
//! (histo3d.md 8.7.3, 8.12.9).
//!
//! - Satu mesh per label; warna label menjadi material.
//! - Satuan meter (standar glTF): µm × 1e−6; Y ke atas (lihat `meshconv`).
//! - Bila sebuah skalar dipilih: `COLOR_0` (RGBA u8 ternormalisasi) dari
//!   colormap, dan nilai mentahnya sebagai atribut verteks `_H3D_<NAMA>`
//!   (`f32`). glTF melarang NaN di accessor, jadi NODATA ditulis sebagai
//!   [`GLB_NODATA`] dan dicatat di `extras`.

use h3d_core::Colormap;
use h3d_volume::LabelMesh;
use serde_json::{json, Map, Value};

use crate::error::IoError;
use crate::meshconv::{flipped_triangles, normal_to_gltf, validate_mesh, w_um_to_gltf_m};

/// Magic "glTF".
pub const GLB_MAGIC: u32 = 0x4654_6C67;
/// Tipe chunk JSON.
pub const CHUNK_JSON: u32 = 0x4E4F_534A;
/// Tipe chunk BIN.
pub const CHUNK_BIN: u32 = 0x004E_4942;
/// Nilai pengganti NODATA (`NaN`) pada atribut `_H3D_*`.
pub const GLB_NODATA: f32 = f32::MIN;
/// Warna verteks untuk NODATA (abu-abu, 8.12.6).
pub const NODATA_COLOR: [u8; 4] = [128, 128, 128, 255];

const COMPONENT_U8: u32 = 5121;
const COMPONENT_U32: u32 = 5125;
const COMPONENT_F32: u32 = 5126;
const TARGET_ARRAY_BUFFER: u32 = 34962;
const TARGET_ELEMENT_ARRAY_BUFFER: u32 = 34963;

/// Satu mesh label untuk ditulis.
#[derive(Clone, Copy, Debug)]
pub struct GlbMesh<'a> {
    /// Nama label (nama node, mesh, dan material).
    pub name: &'a str,
    /// Warna label RGBA.
    pub color: [u8; 4],
    /// Mesh di ruang W (µm).
    pub mesh: &'a LabelMesh,
}

/// Skalar terpilih untuk `COLOR_0`.
#[derive(Clone, Copy, Debug)]
pub struct GlbScalar<'a> {
    /// Nama di `LabelMesh::scalars`.
    pub name: &'a str,
    /// Colormap.
    pub colormap: Colormap,
    /// Rentang tampilan; `None` = persentil 2–98 nilai finit semua mesh.
    pub range: Option<[f32; 2]>,
}

/// Nama atribut verteks glTF untuk skalar: `_H3D_<NAMA>` (huruf besar ASCII,
/// angka; karakter lain menjadi `_`).
pub fn scalar_attribute_name(name: &str) -> String {
    let body: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect();
    format!("_H3D_{body}")
}

struct BinBuilder {
    bin: Vec<u8>,
    views: Vec<Value>,
    accessors: Vec<Value>,
}

impl BinBuilder {
    fn new() -> Self {
        BinBuilder {
            bin: Vec::new(),
            views: Vec::new(),
            accessors: Vec::new(),
        }
    }

    /// Tambahkan bufferView rata 4 byte; kembalikan indeksnya.
    fn push_view(&mut self, bytes: &[u8], target: u32) -> usize {
        while !self.bin.len().is_multiple_of(4) {
            self.bin.push(0);
        }
        let offset = self.bin.len();
        self.bin.extend_from_slice(bytes);
        self.views.push(json!({
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": bytes.len(),
            "target": target,
        }));
        self.views.len() - 1
    }

    fn push_accessor(&mut self, mut acc: Map<String, Value>, view: usize) -> usize {
        acc.insert("bufferView".into(), json!(view));
        acc.insert("byteOffset".into(), json!(0));
        self.accessors.push(Value::Object(acc));
        self.accessors.len() - 1
    }
}

fn f32_bytes(values: impl Iterator<Item = f32>) -> Vec<u8> {
    let mut out = Vec::new();
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Persentil 2–98 nilai finit (indeks terdekat); `None` bila kosong.
fn auto_range(meshes: &[GlbMesh<'_>], name: &str) -> Option<[f32; 2]> {
    let mut vals: Vec<f32> = meshes
        .iter()
        .filter_map(|m| m.mesh.scalars.iter().find(|s| s.name == name))
        .flat_map(|s| s.values.iter().copied())
        .filter(|v| v.is_finite())
        .collect();
    if vals.is_empty() {
        return None;
    }
    vals.sort_by(|a, b| a.total_cmp(b));
    let at = |p: f64| vals[((vals.len() - 1) as f64 * p).round() as usize];
    Some([at(0.02), at(0.98)])
}

/// Tulis GLB. Mesh tanpa segitiga dilewati; bila tidak ada mesh berisi sama
/// sekali → galat. Keluaran deterministik untuk masukan sama.
pub fn encode_glb(
    meshes: &[GlbMesh<'_>],
    scalar: Option<&GlbScalar<'_>>,
) -> Result<Vec<u8>, IoError> {
    let mut b = BinBuilder::new();
    let mut gl_meshes: Vec<Value> = Vec::new();
    let mut materials: Vec<Value> = Vec::new();
    let mut nodes: Vec<Value> = Vec::new();

    let range = scalar.and_then(|s| s.range.or_else(|| auto_range(meshes, s.name)));

    for m in meshes {
        validate_mesh(m.mesh)?;
        if m.mesh.indices.is_empty() || m.mesh.positions_um.is_empty() {
            continue;
        }
        let n_vert = m.mesh.positions_um.len();

        // POSITION (+ min/max wajib).
        let pos: Vec<[f32; 3]> = m
            .mesh
            .positions_um
            .iter()
            .map(|p| w_um_to_gltf_m(*p))
            .collect();
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for p in &pos {
            for k in 0..3 {
                min[k] = min[k].min(p[k]);
                max[k] = max[k].max(p[k]);
            }
        }
        let view = b.push_view(
            &f32_bytes(pos.iter().flat_map(|p| p.iter().copied())),
            TARGET_ARRAY_BUFFER,
        );
        let mut acc = Map::new();
        acc.insert("componentType".into(), json!(COMPONENT_F32));
        acc.insert("count".into(), json!(n_vert));
        acc.insert("type".into(), json!("VEC3"));
        acc.insert(
            "min".into(),
            json!([min[0] as f64, min[1] as f64, min[2] as f64]),
        );
        acc.insert(
            "max".into(),
            json!([max[0] as f64, max[1] as f64, max[2] as f64]),
        );
        let a_pos = b.push_accessor(acc, view);

        let mut attributes = Map::new();
        attributes.insert("POSITION".into(), json!(a_pos));

        // NORMAL (hanya bila lengkap dan finit).
        if m.mesh.normals.len() == n_vert
            && m.mesh
                .normals
                .iter()
                .all(|n| n.iter().all(|v| v.is_finite()))
        {
            let view = b.push_view(
                &f32_bytes(
                    m.mesh
                        .normals
                        .iter()
                        .flat_map(|n| normal_to_gltf(*n).into_iter()),
                ),
                TARGET_ARRAY_BUFFER,
            );
            let mut acc = Map::new();
            acc.insert("componentType".into(), json!(COMPONENT_F32));
            acc.insert("count".into(), json!(n_vert));
            acc.insert("type".into(), json!("VEC3"));
            let a = b.push_accessor(acc, view);
            attributes.insert("NORMAL".into(), json!(a));
        }

        // Skalar terpilih: COLOR_0 + nilai mentah.
        let mut has_color = false;
        let mut extras = Map::new();
        if let Some(sc) = scalar {
            if let Some(s) = m
                .mesh
                .scalars
                .iter()
                .find(|s| s.name == sc.name && s.values.len() == n_vert)
            {
                let [lo, hi] = range.unwrap_or([0.0, 1.0]);
                let span = if hi > lo { hi - lo } else { 1.0 };
                let mut colors = Vec::with_capacity(n_vert * 4);
                for &v in &s.values {
                    if v.is_finite() {
                        let c = sc.colormap.sample((v - lo) / span);
                        colors.extend_from_slice(&[c[0], c[1], c[2], 255]);
                    } else {
                        colors.extend_from_slice(&NODATA_COLOR);
                    }
                }
                let view = b.push_view(&colors, TARGET_ARRAY_BUFFER);
                let mut acc = Map::new();
                acc.insert("componentType".into(), json!(COMPONENT_U8));
                acc.insert("normalized".into(), json!(true));
                acc.insert("count".into(), json!(n_vert));
                acc.insert("type".into(), json!("VEC4"));
                let a = b.push_accessor(acc, view);
                attributes.insert("COLOR_0".into(), json!(a));
                has_color = true;

                let view = b.push_view(
                    &f32_bytes(
                        s.values
                            .iter()
                            .map(|v| if v.is_finite() { *v } else { GLB_NODATA }),
                    ),
                    TARGET_ARRAY_BUFFER,
                );
                let mut acc = Map::new();
                acc.insert("componentType".into(), json!(COMPONENT_F32));
                acc.insert("count".into(), json!(n_vert));
                acc.insert("type".into(), json!("SCALAR"));
                let a = b.push_accessor(acc, view);
                let attr_name = scalar_attribute_name(&s.name);
                attributes.insert(attr_name.clone(), json!(a));
                extras.insert("h3d_scalar".into(), json!(s.name));
                extras.insert("h3d_scalar_attribute".into(), json!(attr_name));
                extras.insert("h3d_scalar_unit".into(), json!(s.unit));
                extras.insert("h3d_scalar_range".into(), json!([lo as f64, hi as f64]));
                extras.insert("h3d_scalar_nodata".into(), json!(GLB_NODATA as f64));
            }
        }

        // Indeks (orientasi ditukar untuk kerangka tangan-kanan).
        let mut idx = Vec::with_capacity(m.mesh.indices.len() * 4);
        for t in flipped_triangles(m.mesh) {
            for i in t {
                idx.extend_from_slice(&i.to_le_bytes());
            }
        }
        let view = b.push_view(&idx, TARGET_ELEMENT_ARRAY_BUFFER);
        let mut acc = Map::new();
        acc.insert("componentType".into(), json!(COMPONENT_U32));
        acc.insert("count".into(), json!(m.mesh.indices.len()));
        acc.insert("type".into(), json!("SCALAR"));
        let a_idx = b.push_accessor(acc, view);

        // Material: warna label; putih bila warna verteks dipakai.
        let base = if has_color {
            [1.0, 1.0, 1.0, 1.0]
        } else {
            [
                f64::from(m.color[0]) / 255.0,
                f64::from(m.color[1]) / 255.0,
                f64::from(m.color[2]) / 255.0,
                f64::from(m.color[3]) / 255.0,
            ]
        };
        let mut material = Map::new();
        material.insert("name".into(), json!(m.name));
        material.insert(
            "pbrMetallicRoughness".into(),
            json!({
                "baseColorFactor": base,
                "metallicFactor": 0.0,
                "roughnessFactor": 0.8,
            }),
        );
        material.insert("doubleSided".into(), json!(true));
        if base[3] < 1.0 {
            material.insert("alphaMode".into(), json!("BLEND"));
        }
        material.insert(
            "extras".into(),
            json!({
                "h3d_label": m.mesh.label.0,
                "h3d_label_color": [m.color[0], m.color[1], m.color[2], m.color[3]],
            }),
        );
        materials.push(Value::Object(material));

        let mut prim = Map::new();
        prim.insert("attributes".into(), Value::Object(attributes));
        prim.insert("indices".into(), json!(a_idx));
        prim.insert("material".into(), json!(materials.len() - 1));
        prim.insert("mode".into(), json!(4));
        let mut gm = Map::new();
        gm.insert("name".into(), json!(m.name));
        gm.insert("primitives".into(), json!([Value::Object(prim)]));
        if !extras.is_empty() {
            gm.insert("extras".into(), Value::Object(extras));
        }
        gl_meshes.push(Value::Object(gm));
        nodes.push(json!({ "name": m.name, "mesh": gl_meshes.len() - 1 }));
    }

    if gl_meshes.is_empty() {
        return Err(IoError::InvalidArgument(
            "tidak ada mesh berisi untuk ditulis ke GLB".into(),
        ));
    }
    while !b.bin.len().is_multiple_of(4) {
        b.bin.push(0);
    }
    if u32::try_from(b.bin.len()).is_err() {
        return Err(IoError::InvalidArgument(
            "mesh terlalu besar untuk satu berkas GLB (> 4 GiB)".into(),
        ));
    }

    let node_indices: Vec<usize> = (0..nodes.len()).collect();
    let doc = json!({
        "asset": {
            "version": "2.0",
            "generator": "HISTO3D",
            "extras": { "h3d_units": "metre", "h3d_notice": h3d_core::RESEARCH_ONLY_NOTICE },
        },
        "scene": 0,
        "scenes": [{ "name": "HISTO3D", "nodes": node_indices }],
        "nodes": nodes,
        "meshes": gl_meshes,
        "materials": materials,
        "accessors": b.accessors,
        "bufferViews": b.views,
        "buffers": [{ "byteLength": b.bin.len() }],
    });
    let mut json_bytes =
        serde_json::to_vec(&doc).map_err(|e| IoError::json(std::path::Path::new("mesh.glb"), e))?;
    while json_bytes.len() % 4 != 0 {
        json_bytes.push(b' ');
    }

    let total = 12 + 8 + json_bytes.len() + 8 + b.bin.len();
    let total_u32 = u32::try_from(total).map_err(|_| {
        IoError::InvalidArgument("mesh terlalu besar untuk satu berkas GLB (> 4 GiB)".into())
    })?;
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&GLB_MAGIC.to_le_bytes());
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&total_u32.to_le_bytes());
    out.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&CHUNK_JSON.to_le_bytes());
    out.extend_from_slice(&json_bytes);
    out.extend_from_slice(&(b.bin.len() as u32).to_le_bytes());
    out.extend_from_slice(&CHUNK_BIN.to_le_bytes());
    out.extend_from_slice(&b.bin);
    Ok(out)
}

/// Urai kontainer GLB menjadi `(JSON, BIN)` dengan validasi struktur
/// (magic, versi, panjang, perataan chunk). Dipakai tes dan pemeriksaan
/// mandiri; berkas rusak → galat.
pub fn parse_glb(bytes: &[u8]) -> Result<(Value, Vec<u8>), IoError> {
    let p = std::path::Path::new("mesh.glb");
    let u32_at = |o: usize| -> Result<u32, IoError> {
        bytes
            .get(o..o + 4)
            .and_then(|s| <[u8; 4]>::try_from(s).ok())
            .map(u32::from_le_bytes)
            .ok_or_else(|| IoError::invalid(p, "berkas terpotong"))
    };
    if u32_at(0)? != GLB_MAGIC {
        return Err(IoError::invalid(p, "magic bukan glTF"));
    }
    if u32_at(4)? != 2 {
        return Err(IoError::invalid(p, "versi kontainer bukan 2"));
    }
    if u32_at(8)? as usize != bytes.len() {
        return Err(IoError::invalid(p, "panjang total tidak cocok"));
    }
    let json_len = u32_at(12)? as usize;
    if u32_at(16)? != CHUNK_JSON || !json_len.is_multiple_of(4) {
        return Err(IoError::invalid(p, "chunk JSON tidak sah"));
    }
    let json_end = 20usize
        .checked_add(json_len)
        .filter(|e| *e <= bytes.len())
        .ok_or_else(|| IoError::invalid(p, "chunk JSON melewati akhir berkas"))?;
    let doc: Value =
        serde_json::from_slice(&bytes[20..json_end]).map_err(|e| IoError::json(p, e))?;
    if json_end == bytes.len() {
        return Ok((doc, Vec::new()));
    }
    let bin_len = u32_at(json_end)? as usize;
    if u32_at(json_end + 4)? != CHUNK_BIN || !bin_len.is_multiple_of(4) {
        return Err(IoError::invalid(p, "chunk BIN tidak sah"));
    }
    let bin_start = json_end + 8;
    let bin_end = bin_start
        .checked_add(bin_len)
        .filter(|e| *e == bytes.len())
        .ok_or_else(|| IoError::invalid(p, "chunk BIN tidak cocok dengan panjang berkas"))?;
    Ok((doc, bytes[bin_start..bin_end].to_vec()))
}
