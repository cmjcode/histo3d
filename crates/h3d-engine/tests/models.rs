//! Model AI di bundel: daftar, pasang (salinan atomik), label kelas, dan
//! penolakan massal usulan. Model ONNX mungil (Conv 1×1) dibangun lewat tipe
//! protobuf tract agar tes tidak bergantung pada berkas biner di repo.

use std::fs;
use std::path::Path;

use h3d_core::LabelKind;
use h3d_engine::{ModelSpec, PaintMode, Session, MODEL_ONNX_FILE, MODEL_SPEC_FILE};
use prost::Message;
use tract_onnx::pb;

fn dim(n: i64) -> pb::tensor_shape_proto::Dimension {
    pb::tensor_shape_proto::Dimension {
        denotation: String::new(),
        value: Some(pb::tensor_shape_proto::dimension::Value::DimValue(n)),
    }
}

fn value_info(name: &str, dims: &[i64]) -> pb::ValueInfoProto {
    pb::ValueInfoProto {
        name: name.to_string(),
        r#type: Some(pb::TypeProto {
            denotation: String::new(),
            value: Some(pb::type_proto::Value::TensorType(pb::type_proto::Tensor {
                elem_type: pb::tensor_proto::DataType::Float as i32,
                shape: Some(pb::TensorShapeProto {
                    dim: dims.iter().map(|&d| dim(d)).collect(),
                }),
            })),
        }),
        doc_string: String::new(),
    }
}

fn float_tensor(name: &str, dims: &[i64], data: &[f32]) -> pb::TensorProto {
    pb::TensorProto {
        dims: dims.to_vec(),
        data_type: pb::tensor_proto::DataType::Float as i32,
        float_data: data.to_vec(),
        name: name.to_string(),
        ..Default::default()
    }
}

/// Conv 1×1 tiga kelas: `y = W·x + b` (opset 13).
fn tiny_onnx(s: i64) -> Vec<u8> {
    let conv = pb::NodeProto {
        input: vec!["input".into(), "w".into(), "b".into()],
        output: vec!["logits".into()],
        name: "conv".into(),
        op_type: "Conv".into(),
        domain: String::new(),
        attribute: vec![pb::AttributeProto {
            name: "kernel_shape".into(),
            r#type: pb::attribute_proto::AttributeType::Ints as i32,
            ints: vec![1, 1],
            ..Default::default()
        }],
        doc_string: String::new(),
    };
    let graph = pb::GraphProto {
        node: vec![conv],
        name: "tiny".into(),
        initializer: vec![
            float_tensor(
                "w",
                &[3, 3, 1, 1],
                &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            ),
            float_tensor("b", &[3], &[0.0, 0.0, 0.0]),
        ],
        input: vec![value_info("input", &[1, 3, s, s])],
        output: vec![value_info("logits", &[1, 3, s, s])],
        ..Default::default()
    };
    pb::ModelProto {
        ir_version: 8,
        producer_name: "h3d-engine-test".into(),
        graph: Some(graph),
        opset_import: vec![pb::OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
        ..Default::default()
    }
    .encode_to_vec()
}

const SPEC_JSON: &str = r#"{
  "name": "uji-model",
  "version": "0.1.0",
  "input_size": 16,
  "mpp": 2.0,
  "mean": [0.5, 0.5, 0.5],
  "std": [0.25, 0.25, 0.25],
  "classes": [
    {"index": 0, "label": null},
    {"index": 1, "label": "Jaringan"},
    {"index": 2, "label": "Tumor"}
  ],
  "task": "segmentation"
}"#;

/// Tulis folder model lengkap di `dir`.
fn write_model_dir(dir: &Path) {
    fs::create_dir_all(dir).unwrap();
    fs::write(dir.join(MODEL_SPEC_FILE), SPEC_JSON).unwrap();
    fs::write(dir.join(MODEL_ONNX_FILE), tiny_onnx(16)).unwrap();
}

fn new_session(root: &Path) -> Session {
    Session::create(&root.join("p.h3d")).expect("buat proyek")
}

#[test]
fn install_lists_and_replaces_model() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("luar").join("tumor-uji");
    write_model_dir(&src);
    let sess = new_session(tmp.path());
    assert!(sess.list_models().is_empty());

    let dest = sess.install_model(&src).expect("pasang model");
    assert_eq!(dest, sess.bundle().models_dir().join("tumor-uji"));
    assert!(dest.join(MODEL_SPEC_FILE).is_file());
    assert!(dest.join(MODEL_ONNX_FILE).is_file());
    assert!(
        !sess.bundle().models_dir().join("tumor-uji.tmp").exists(),
        "folder sementara harus hilang setelah rename"
    );

    let list = sess.list_models();
    assert_eq!(list.len(), 1);
    let entry = &list[0];
    assert_eq!(entry.folder, "tumor-uji");
    assert_eq!(entry.display_name(), "uji-model 0.1.0");
    assert_eq!(entry.class_names(), vec!["Jaringan", "Tumor"]);

    // Path `model.json` juga diterima, dan model bernama sama diganti.
    let dest2 = sess
        .install_model(&src.join(MODEL_SPEC_FILE))
        .expect("pasang ulang");
    assert_eq!(dest2, dest);
    assert_eq!(sess.list_models().len(), 1);

    // Memasang dari folder di dalam bundel sendiri tidak menyalin apa pun.
    assert_eq!(sess.install_model(&dest).unwrap(), dest);
}

#[test]
fn install_rejects_broken_or_incomplete_model() {
    let tmp = tempfile::tempdir().unwrap();
    let sess = new_session(tmp.path());

    // Tanpa model.onnx.
    let no_onnx = tmp.path().join("tanpa-onnx");
    fs::create_dir_all(&no_onnx).unwrap();
    fs::write(no_onnx.join(MODEL_SPEC_FILE), SPEC_JSON).unwrap();
    assert!(sess.install_model(&no_onnx).is_err());

    // model.json rusak.
    let bad_json = tmp.path().join("json-rusak");
    fs::create_dir_all(&bad_json).unwrap();
    fs::write(bad_json.join(MODEL_SPEC_FILE), "{ bukan json").unwrap();
    fs::write(bad_json.join(MODEL_ONNX_FILE), tiny_onnx(16)).unwrap();
    assert!(sess.install_model(&bad_json).is_err());

    // Folder tanpa model.json sama sekali.
    assert!(sess.install_model(tmp.path()).is_err());
    assert!(sess.list_models().is_empty());

    // Folder rusak di dalam bundel tetap terdaftar dengan galat, bukan panik.
    let inside = sess.bundle().models_dir().join("rusak");
    fs::create_dir_all(&inside).unwrap();
    fs::write(inside.join(MODEL_SPEC_FILE), SPEC_JSON).unwrap();
    let list = sess.list_models();
    assert_eq!(list.len(), 1);
    assert!(list[0].spec.is_err());
    assert_eq!(list[0].display_name(), "rusak");
}

#[test]
fn ensure_model_labels_creates_missing_only() {
    let tmp = tempfile::tempdir().unwrap();
    let mut sess = new_session(tmp.path());
    let spec = ModelSpec::from_json(SPEC_JSON).unwrap();

    // Proyek baru sudah punya "Tumor" bawaan; "Jaringan" belum.
    let before = sess.project().labels.len();
    let setup = sess.ensure_model_labels(&spec).unwrap();
    assert_eq!(setup.created, vec!["Jaringan"]);
    assert_eq!(setup.existing, vec!["Tumor"]);
    assert_eq!(sess.project().labels.len(), before + 1);
    let jaringan = sess
        .project()
        .label_by_name("jaringan")
        .expect("label baru");
    assert_eq!(jaringan.kind, LabelKind::Normal);
    let tumor = sess.project().label_by_name("Tumor").unwrap();
    assert_ne!(jaringan.color, tumor.color, "warna baru tidak boleh sama");

    // Idempoten.
    let again = sess.ensure_model_labels(&spec).unwrap();
    assert!(again.created.is_empty());
    assert_eq!(sess.project().labels.len(), before + 1);
}

#[test]
fn reject_all_pending_clears_queue_without_touching_labels() {
    let tmp = tempfile::tempdir().unwrap();
    let mut sess = new_session(tmp.path());
    assert_eq!(sess.reject_all_pending(), 0);
    assert!(sess.pending_slides().is_empty());
    assert_eq!(
        sess.accept_all_pending(PaintMode::OnlyBackground).unwrap(),
        0
    );
}
