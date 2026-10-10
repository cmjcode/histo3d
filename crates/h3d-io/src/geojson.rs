//! Ekspor GeoJSON kontur per irisan (8.7.3, 8.12.9), kompatibel QuPath.
//! Rasterisasi dan penelusuran kontur ada di `h3d-segment`; modul ini
//! menambahkan `properties.measurements` (luas + rata-rata atribut).

use h3d_core::LabelId;
use h3d_segment::LabelStore;
use serde_json::{Map, Value};

/// Rata-rata atribut per label: `(label, [(nama atribut, nilai)])`.
pub type LabelAttrMeans = Vec<(LabelId, Vec<(String, f64)>)>;

/// Ekspor label satu irisan sebagai FeatureCollection GeoJSON. Koordinat
/// dalam piksel level 0. `properties.classification.name` = nama label;
/// `properties.measurements` memuat `area_um2` (dari `h3d-segment`) dan
/// rata-rata tiap atribut pada label itu.
pub fn export_slide_geojson(
    store: &LabelStore,
    labels: &[(LabelId, &str)],
    mpp_l0: [f64; 2],
    attr_means: &LabelAttrMeans,
) -> String {
    let measure = |label: LabelId, _component: usize| -> Value {
        let mut m = Map::new();
        if let Some((_, list)) = attr_means.iter().find(|(l, _)| *l == label) {
            for (name, value) in list {
                if value.is_finite() {
                    m.insert(name.clone(), Value::from(*value));
                }
            }
        }
        Value::Object(m)
    };
    h3d_segment::export_geojson(store, labels, mpp_l0, Some(&measure))
}
