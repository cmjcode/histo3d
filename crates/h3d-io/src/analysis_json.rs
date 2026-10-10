//! `analysis.json`: keluaran analisis spasial (8.12.6) yang deterministik.
//!
//! Penulis JSON sendiri dipakai agar keluaran byte-identik untuk masukan
//! sama: kunci objek selalu terurut, bilangan pecahan diformat tetap (enam
//! desimal, nol di belakang dipangkas), nilai tak-finit menjadi `null`.

use std::fmt::Write as _;
use std::path::Path;

use h3d_volume::SpatialAnalysis;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::IoError;

/// Penanda format berkas.
pub const ANALYSIS_FORMAT: &str = "h3d-analysis";
/// Versi format berkas analisis.
pub const ANALYSIS_VERSION: u32 = 1;
/// Jumlah desimal bilangan pecahan.
pub const FLOAT_DECIMALS: usize = 6;

/// Isi `analysis.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnalysisFile {
    /// Selalu [`ANALYSIS_FORMAT`].
    pub format: String,
    /// Versi format.
    pub version: u32,
    /// Versi aplikasi penulis.
    pub app_version: String,
    /// Kalimat "Hanya untuk riset".
    pub notice: String,
    /// Kunci masukan (hash label + transformasi + konfigurasi), heksadesimal.
    /// Dipakai untuk mengenali analisis yang sudah basi.
    pub input_key: String,
    /// Hasil analisis.
    pub analysis: SpatialAnalysis,
}

/// Format bilangan pecahan: enam desimal, nol di belakang dipangkas, minimal
/// satu desimal; `-0` dinormalkan.
pub fn format_float(v: f64) -> String {
    if !v.is_finite() {
        return "null".to_owned();
    }
    let mut s = format!("{:.*}", FLOAT_DECIMALS, v);
    while s.ends_with('0') && !s.ends_with(".0") {
        s.pop();
    }
    if s == "-0.0" {
        s = "0.0".to_owned();
    }
    s
}

fn write_string(out: &mut String, s: &str) {
    // serde_json menangani semua aturan escape JSON.
    match serde_json::to_string(s) {
        Ok(q) => out.push_str(&q),
        Err(_) => out.push_str("\"\""),
    }
}

fn write_value(out: &mut String, v: &Value, indent: usize) {
    const STEP: usize = 2;
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                let _ = write!(out, "{i}");
            } else if let Some(u) = n.as_u64() {
                let _ = write!(out, "{u}");
            } else {
                out.push_str(&format_float(n.as_f64().unwrap_or(f64::NAN)));
            }
        }
        Value::String(s) => write_string(out, s),
        Value::Array(a) => {
            if a.is_empty() {
                out.push_str("[]");
                return;
            }
            // Larik skalar pendek ditulis sebaris agar berkas tetap terbaca.
            let scalar = a.iter().all(|x| !x.is_array() && !x.is_object());
            if scalar && a.len() <= 8 {
                out.push('[');
                for (i, x) in a.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    write_value(out, x, indent);
                }
                out.push(']');
                return;
            }
            out.push_str("[\n");
            for (i, x) in a.iter().enumerate() {
                let _ = write!(out, "{:w$}", "", w = indent + STEP);
                write_value(out, x, indent + STEP);
                if i + 1 < a.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            let _ = write!(out, "{:w$}]", "", w = indent);
        }
        Value::Object(m) => {
            if m.is_empty() {
                out.push_str("{}");
                return;
            }
            // Urutkan kunci secara eksplisit: tidak bergantung pada fitur
            // `preserve_order` serde_json.
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            out.push_str("{\n");
            for (i, k) in keys.iter().enumerate() {
                let _ = write!(out, "{:w$}", "", w = indent + STEP);
                write_string(out, k);
                out.push_str(": ");
                write_value(out, &m[*k], indent + STEP);
                if i + 1 < keys.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            let _ = write!(out, "{:w$}}}", "", w = indent);
        }
    }
}

/// JSON stabil dari sebuah `Value` (kunci terurut, pecahan berformat tetap),
/// diakhiri baris baru.
pub fn to_stable_json(value: &Value) -> String {
    let mut out = String::new();
    write_value(&mut out, value, 0);
    out.push('\n');
    out
}

/// JSON stabil dari nilai serde apa pun.
pub fn to_stable_json_of<T: Serialize>(value: &T) -> Result<String, IoError> {
    let v = serde_json::to_value(value).map_err(|e| IoError::json(Path::new("json"), e))?;
    Ok(to_stable_json(&v))
}

/// Serialisasi analisis spasial ke teks `analysis.json`.
pub fn encode_analysis(
    analysis: &SpatialAnalysis,
    app_version: &str,
    input_key: u64,
) -> Result<String, IoError> {
    let file = AnalysisFile {
        format: ANALYSIS_FORMAT.to_owned(),
        version: ANALYSIS_VERSION,
        app_version: app_version.to_owned(),
        notice: h3d_core::RESEARCH_ONLY_NOTICE.to_owned(),
        input_key: format!("{input_key:016x}"),
        analysis: analysis.clone(),
    };
    to_stable_json_of(&file)
}

/// Baca `analysis.json`. Format atau versi tak dikenal → galat.
pub fn decode_analysis(text: &str, path: &Path) -> Result<AnalysisFile, IoError> {
    let file: AnalysisFile = serde_json::from_str(text).map_err(|e| IoError::json(path, e))?;
    if file.format != ANALYSIS_FORMAT {
        return Err(IoError::invalid(path, "bukan berkas analisis HISTO3D"));
    }
    if file.version > ANALYSIS_VERSION {
        return Err(IoError::invalid(
            path,
            format!(
                "versi berkas analisis {} lebih baru dari yang didukung",
                file.version
            ),
        ));
    }
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn floats_are_fixed_and_keys_sorted() {
        assert_eq!(format_float(1.0), "1.0");
        assert_eq!(format_float(0.1 + 0.2), "0.3");
        assert_eq!(format_float(-0.0), "0.0");
        assert_eq!(format_float(1234.56789012), "1234.56789");
        assert_eq!(format_float(f64::NAN), "null");
        let v = json!({"b": 1, "a": [1.5, 2], "c": {"z": null, "y": "x\"y"}});
        let s = to_stable_json(&v);
        assert!(s.find("\"a\"").unwrap() < s.find("\"b\"").unwrap());
        assert!(s.find("\"y\"").unwrap() < s.find("\"z\"").unwrap());
        let back: Value = serde_json::from_str(&s).unwrap();
        assert_eq!(back, v);
        // u64 besar tetap bilangan bulat persis.
        let big = json!({"h": u64::MAX});
        assert!(to_stable_json(&big).contains("18446744073709551615"));
    }
}
