//! Pertukaran anotasi GeoJSON gaya QuPath (histo3d.md 8.5.6).
//!
//! Koordinat GeoJSON adalah **piksel level 0** (ruang S0); konversi ke µm
//! memakai `mpp_l0 = [mpp_x, mpp_y]`. Ekspor menelusuri tepi tiap komponen
//! terhubung (konektivitas 8) sebagai poligon sudut piksel; lubang menjadi
//! cincin dalam. Impor mengisi poligon dengan aturan genap-ganjil.

use std::collections::BTreeMap;

use h3d_core::LabelId;
use h3d_imgproc::components::label_components;
use ndarray::Array2;
use serde_json::{json, Map, Value};

use crate::error::SegError;
use crate::label_store::{LabelStore, PaintMode, TileKey};

/// Ringkasan impor.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ImportSummary {
    /// Jumlah fitur dalam berkas.
    pub features: usize,
    /// Fitur yang dirasterkan.
    pub imported: usize,
    /// Fitur dilewati karena nama kelas tidak dikenal/kosong.
    pub skipped_unknown_label: usize,
    /// Fitur dilewati karena geometri tidak didukung/rusak.
    pub skipped_geometry: usize,
    /// Tile yang berubah.
    pub dirty_tiles: Vec<TileKey>,
}

fn classification_name(props: Option<&Value>) -> Option<String> {
    let props = props?.as_object()?;
    match props.get("classification") {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Object(o)) => o.get("name").and_then(Value::as_str).map(str::to_owned),
        _ => props.get("name").and_then(Value::as_str).map(str::to_owned),
    }
}

/// Baca satu cincin `[[x, y], …]` (piksel level 0) → µm.
fn ring_um(v: &Value, mpp: [f64; 2]) -> Option<Vec<[f64; 2]>> {
    let arr = v.as_array()?;
    let mut out = Vec::with_capacity(arr.len());
    for p in arr {
        let c = p.as_array()?;
        if c.len() < 2 {
            return None;
        }
        let x = c[0].as_f64()?;
        let y = c[1].as_f64()?;
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        out.push([x * mpp[0], y * mpp[1]]);
    }
    Some(out)
}

/// Kumpulan cincin dari geometri Polygon/MultiPolygon; `None` bila tidak
/// didukung/rusak.
fn geometry_rings(geom: &Value, mpp: [f64; 2]) -> Option<Vec<Vec<[f64; 2]>>> {
    let ty = geom.get("type")?.as_str()?;
    let coords = geom.get("coordinates")?;
    let mut rings = Vec::new();
    match ty {
        "Polygon" => {
            for r in coords.as_array()? {
                rings.push(ring_um(r, mpp)?);
            }
        }
        "MultiPolygon" => {
            for poly in coords.as_array()? {
                for r in poly.as_array()? {
                    rings.push(ring_um(r, mpp)?);
                }
            }
        }
        _ => return None,
    }
    Some(rings)
}

/// Impor GeoJSON (FeatureCollection, Feature, atau Geometry) ke `store`.
/// `label_by_name` memetakan `properties.classification.name` (atau string
/// `classification`, atau `properties.name`) ke `LabelId`; fitur tanpa
/// padanan dilewati. Pengisian memakai `PaintMode::Replace`.
pub fn import_geojson(
    text: &str,
    store: &mut LabelStore,
    mpp_l0: [f64; 2],
    label_by_name: &dyn Fn(&str) -> Option<LabelId>,
) -> Result<ImportSummary, SegError> {
    if !(mpp_l0[0].is_finite() && mpp_l0[0] > 0.0 && mpp_l0[1].is_finite() && mpp_l0[1] > 0.0) {
        return Err(SegError::InvalidParameter(format!(
            "mpp tidak sah: {mpp_l0:?}"
        )));
    }
    let root: Value = serde_json::from_str(text)
        .map_err(|e| SegError::invalid_file(format!("GeoJSON tidak sah: {e}")))?;
    let features: Vec<&Value> = match root.get("type").and_then(Value::as_str) {
        Some("FeatureCollection") => root
            .get("features")
            .and_then(Value::as_array)
            .ok_or_else(|| SegError::invalid_file("FeatureCollection tanpa features"))?
            .iter()
            .collect(),
        Some("Feature") => vec![&root],
        Some("Polygon") | Some("MultiPolygon") => vec![&root],
        other => {
            return Err(SegError::invalid_file(format!(
                "tipe GeoJSON tidak didukung: {other:?}"
            )))
        }
    };
    let mut summary = ImportSummary {
        features: features.len(),
        ..Default::default()
    };
    let mut dirty = std::collections::BTreeSet::new();
    for f in features {
        let (geom, props) = if f.get("type").and_then(Value::as_str) == Some("Feature") {
            (f.get("geometry"), f.get("properties"))
        } else {
            (Some(f), None)
        };
        let Some(geom) = geom else {
            summary.skipped_geometry += 1;
            continue;
        };
        let Some(rings) = geometry_rings(geom, mpp_l0) else {
            summary.skipped_geometry += 1;
            continue;
        };
        let label = classification_name(props).and_then(|n| label_by_name(&n));
        let Some(label) = label else {
            summary.skipped_unknown_label += 1;
            continue;
        };
        let refs: Vec<&[[f64; 2]]> = rings.iter().map(Vec::as_slice).collect();
        dirty.extend(store.fill_rings(&refs, label, PaintMode::Replace));
        summary.imported += 1;
    }
    let mut d: Vec<TileKey> = dirty.into_iter().collect();
    d.sort_by_key(|&(tx, ty)| (ty, tx));
    summary.dirty_tiles = d;
    Ok(summary)
}

// ---------------------------------------------------------------------------
// Ekspor: penelusuran tepi komponen.
// ---------------------------------------------------------------------------

/// Arah sisi: 0 = kanan, 1 = bawah, 2 = kiri, 3 = atas (y ke bawah).
type Edge = ((i64, i64), u8);

/// Kumpulkan sisi tepi berarah (interior di sisi kanan arah jalan = searah
/// jarum jam di layar) untuk piksel bernilai `id` dalam `labels`.
fn boundary_edges(labels: &Array2<u32>, id: u32, bbox: [usize; 4]) -> Vec<Edge> {
    let (h, w) = labels.dim();
    let [x0, y0, x1, y1] = bbox;
    let inside = |x: i64, y: i64| -> bool {
        x >= 0
            && y >= 0
            && (x as usize) < w
            && (y as usize) < h
            && labels[[y as usize, x as usize]] == id
    };
    let mut edges = Vec::new();
    for y in y0..y1 {
        for x in x0..x1 {
            if labels[[y, x]] != id {
                continue;
            }
            let (xi, yi) = (x as i64, y as i64);
            if !inside(xi, yi - 1) {
                edges.push(((xi, yi), 0u8)); // atas: (x,y) → (x+1,y)
            }
            if !inside(xi + 1, yi) {
                edges.push(((xi + 1, yi), 1u8)); // kanan: (x+1,y) → (x+1,y+1)
            }
            if !inside(xi, yi + 1) {
                edges.push(((xi + 1, yi + 1), 2u8)); // bawah: (x+1,y+1) → (x,y+1)
            }
            if !inside(xi - 1, yi) {
                edges.push(((xi, yi + 1), 3u8)); // kiri: (x,y+1) → (x,y)
            }
        }
    }
    edges
}

fn edge_end(e: Edge) -> (i64, i64) {
    let ((x, y), d) = e;
    match d {
        0 => (x + 1, y),
        1 => (x, y + 1),
        2 => (x - 1, y),
        _ => (x, y - 1),
    }
}

/// Rangkai sisi menjadi cincin tertutup. Pada simpul bercabang (dua piksel
/// bersentuhan diagonal) dipilih belokan kanan agar tiap cincin tetap
/// sederhana. Titik kolinear dibuang. Hasil deterministik (sisi diurutkan).
fn chain_rings(mut edges: Vec<Edge>) -> Vec<Vec<(i64, i64)>> {
    edges.sort_unstable();
    edges.dedup();
    let mut by_start: BTreeMap<(i64, i64), Vec<usize>> = BTreeMap::new();
    for (i, e) in edges.iter().enumerate() {
        by_start.entry(e.0).or_default().push(i);
    }
    let mut used = vec![false; edges.len()];
    let mut rings = Vec::new();
    for start in 0..edges.len() {
        if used[start] {
            continue;
        }
        let mut ring: Vec<(i64, i64)> = Vec::new();
        let mut cur = start;
        loop {
            used[cur] = true;
            ring.push(edges[cur].0);
            let end = edge_end(edges[cur]);
            let d_in = edges[cur].1;
            let Some(cands) = by_start.get(&end) else {
                break;
            };
            // Prioritas: belok kanan, lurus, belok kiri.
            let prefs = [(d_in + 1) % 4, d_in, (d_in + 3) % 4];
            let mut next = None;
            'outer: for p in prefs {
                for &c in cands {
                    if !used[c] && edges[c].1 == p {
                        next = Some(c);
                        break 'outer;
                    }
                }
            }
            match next {
                Some(n) => cur = n,
                None => break,
            }
        }
        // Buang titik kolinear.
        let n = ring.len();
        if n < 4 {
            continue;
        }
        let mut simp: Vec<(i64, i64)> = Vec::with_capacity(n);
        for i in 0..n {
            let p = ring[(i + n - 1) % n];
            let c = ring[i];
            let q = ring[(i + 1) % n];
            let collinear = (c.0 - p.0) * (q.1 - c.1) == (c.1 - p.1) * (q.0 - c.0);
            if !collinear {
                simp.push(c);
            }
        }
        if simp.len() >= 3 {
            rings.push(simp);
        }
    }
    rings
}

fn ring_area2(r: &[(i64, i64)]) -> i64 {
    let n = r.len();
    let mut a = 0i64;
    for i in 0..n {
        let p = r[i];
        let q = r[(i + 1) % n];
        a += p.0 * q.1 - q.0 * p.1;
    }
    a
}

/// Ekspor label `labels = [(id, nama)]` sebagai FeatureCollection GeoJSON.
/// Koordinat = sudut piksel label dalam piksel level 0
/// (`x_l0 = x_label × mpp_label / mpp_l0`). `properties.classification.name`
/// = nama; `properties.measurements.area_um2` = luas komponen; nilai objek dari
/// `measurements(label, indeks_komponen)` digabungkan ke `measurements`.
pub fn export_geojson(
    store: &LabelStore,
    labels: &[(LabelId, &str)],
    mpp_l0: [f64; 2],
    measurements: Option<&dyn Fn(LabelId, usize) -> Value>,
) -> String {
    let sx = store.mpp_label / mpp_l0[0];
    let sy = store.mpp_label / mpp_l0[1];
    let mut features = Vec::new();
    for &(label, name) in labels {
        let Some([bx0, by0, bx1, by1]) = store.bbox_px(label) else {
            continue;
        };
        // Bekerja pada wilayah bbox + tepi 1 piksel.
        let w = (bx1 - bx0) as usize;
        let h = (by1 - by0) as usize;
        let mask = store.mask_region(label, bx0, by0, w as u32, h as u32);
        let (comp, n) = label_components(&mask);
        for ci in 1..=n {
            let edges = boundary_edges(&comp, ci as u32, [0, 0, w, h]);
            let area_px = comp.iter().filter(|&&v| v == ci as u32).count();
            let mut rings = chain_rings(edges);
            if rings.is_empty() {
                continue;
            }
            // Cincin luar (|luas| terbesar) lebih dulu.
            rings.sort_by_key(|r| -ring_area2(r).abs());
            let coords: Vec<Value> = rings
                .iter()
                .map(|r| {
                    let mut pts: Vec<Value> = r
                        .iter()
                        .map(|&(x, y)| {
                            json!([(x + bx0 as i64) as f64 * sx, (y + by0 as i64) as f64 * sy])
                        })
                        .collect();
                    // GeoJSON: cincin ditutup.
                    if let Some(first) = pts.first().cloned() {
                        pts.push(first);
                    }
                    Value::Array(pts)
                })
                .collect();
            let mut meas = Map::new();
            meas.insert(
                "area_um2".to_string(),
                json!(area_px as f64 * store.mpp_label * store.mpp_label),
            );
            if let Some(f) = measurements {
                if let Value::Object(extra) = f(label, ci - 1) {
                    for (k, v) in extra {
                        meas.insert(k, v);
                    }
                }
            }
            features.push(json!({
                "type": "Feature",
                "geometry": { "type": "Polygon", "coordinates": coords },
                "properties": {
                    "objectType": "annotation",
                    "classification": { "name": name },
                    "measurements": Value::Object(meas),
                }
            }));
        }
    }
    let fc = json!({ "type": "FeatureCollection", "features": features });
    serde_json::to_string(&fc)
        .unwrap_or_else(|_| "{\"type\":\"FeatureCollection\",\"features\":[]}".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use h3d_imgproc::metrics::dice;

    fn by_name(n: &str) -> Option<LabelId> {
        match n {
            "Tumor" => Some(LabelId(1)),
            "Nekrosis" => Some(LabelId(2)),
            _ => None,
        }
    }

    #[test]
    fn export_import_roundtrip_dice() {
        let mpp_l0 = [0.5, 0.5];
        let mut s = LabelStore::new(4.0, [4000.0, 3000.0]);
        s.paint_disc([1200.0, 1500.0], 700.0, LabelId(1), PaintMode::Replace);
        s.paint_disc([1200.0, 1500.0], 200.0, LabelId(2), PaintMode::Replace); // lubang di Tumor
        s.paint_disc([3000.0, 1000.0], 400.0, LabelId(1), PaintMode::Replace);
        s.fill_polygon(
            &[[2600.0, 2000.0], [3600.0, 2100.0], [3300.0, 2800.0]],
            LabelId(2),
            PaintMode::Replace,
        );
        // Dua piksel bersentuhan diagonal (kasus pinch).
        s.rasterize_mask(
            &ndarray::arr2(&[[1u8, 0], [0, 1]]),
            [10, 10],
            LabelId(1),
            PaintMode::Replace,
        );
        let text = export_geojson(
            &s,
            &[(LabelId(1), "Tumor"), (LabelId(2), "Nekrosis")],
            mpp_l0,
            Some(&|l, i| json!({ "label": l.0, "idx": i })),
        );
        let v: Value = serde_json::from_str(&text).unwrap();
        let feats = v["features"].as_array().unwrap();
        assert!(feats.len() >= 4, "{} fitur", feats.len());
        assert_eq!(feats[0]["properties"]["classification"]["name"], "Tumor");
        assert!(
            feats[0]["properties"]["measurements"]["area_um2"]
                .as_f64()
                .unwrap()
                > 0.0
        );
        assert_eq!(feats[0]["properties"]["measurements"]["label"], 1);
        // Komponen Tumor besar punya cincin dalam (lubang Nekrosis).
        let big = feats
            .iter()
            .filter(|f| f["properties"]["classification"]["name"] == "Tumor")
            .max_by_key(|f| {
                f["properties"]["measurements"]["area_um2"]
                    .as_f64()
                    .unwrap() as i64
            })
            .unwrap();
        assert!(big["geometry"]["coordinates"].as_array().unwrap().len() >= 2);

        let mut t = LabelStore::new(4.0, [4000.0, 3000.0]);
        let sum = import_geojson(&text, &mut t, mpp_l0, &by_name).unwrap();
        assert_eq!(sum.imported, feats.len());
        assert_eq!(sum.skipped_unknown_label, 0);
        for l in [LabelId(1), LabelId(2)] {
            let d = dice(&s.to_array(l), &t.to_array(l)).unwrap();
            assert!(d >= 0.99, "label {l}: Dice {d}");
        }
        // Deterministik.
        assert_eq!(
            text,
            export_geojson(
                &s,
                &[(LabelId(1), "Tumor"), (LabelId(2), "Nekrosis")],
                mpp_l0,
                Some(&|l, i| json!({ "label": l.0, "idx": i })),
            )
        );
    }

    #[test]
    fn import_qupath_variants_and_errors() {
        let mut s = LabelStore::new(4.0, [1000.0, 1000.0]);
        let text = r#"{"type":"FeatureCollection","features":[
          {"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[0,0],[400,0],[400,400],[0,400],[0,0]]]},
           "properties":{"classification":{"name":"Tumor","color":[200,0,0]}}},
          {"type":"Feature","geometry":{"type":"MultiPolygon","coordinates":[[[[800,800],[1200,800],[1200,1200],[800,1200]]]]},
           "properties":{"classification":"Nekrosis"}},
          {"type":"Feature","geometry":{"type":"Point","coordinates":[1,1]},"properties":{"classification":"Tumor"}},
          {"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[0,0],[10,0],[10,10]]]},"properties":{}}
        ]}"#;
        let sum = import_geojson(text, &mut s, [0.5, 0.5], &by_name).unwrap();
        assert_eq!(sum.features, 4);
        assert_eq!(sum.imported, 2);
        assert_eq!(sum.skipped_geometry, 1);
        assert_eq!(sum.skipped_unknown_label, 1);
        // 400 px level-0 × 0,5 µm = 200 µm = 50 px label.
        assert_eq!(s.area_px(LabelId(1)), 50 * 50);
        assert_eq!(s.area_px(LabelId(2)), 50 * 50);
        assert!(import_geojson("bukan json", &mut s, [0.5, 0.5], &by_name).is_err());
        assert!(import_geojson(r#"{"type":"Point"}"#, &mut s, [0.5, 0.5], &by_name).is_err());
        assert!(import_geojson("{}", &mut s, [0.5, 0.5], &by_name).is_err());
        assert!(import_geojson(text, &mut s, [0.0, 0.5], &by_name).is_err());
    }

    #[test]
    fn export_empty_store() {
        let s = LabelStore::new(4.0, [100.0, 100.0]);
        let v: Value = serde_json::from_str(&export_geojson(
            &s,
            &[(LabelId(1), "Tumor")],
            [0.5, 0.5],
            None,
        ))
        .unwrap();
        assert_eq!(v["features"].as_array().unwrap().len(), 0);
    }
}
