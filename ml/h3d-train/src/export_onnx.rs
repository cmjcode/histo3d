//! Eksportir ONNX untuk [`UNet`]: membangun graf protobuf (`tract_onnx::pb`)
//! dari bobot burn. BatchNorm dilipat ke Conv (mode inferensi), sehingga
//! graf hanya memuat Conv, Relu, MaxPool, ConvTranspose, Concat (opset 13).
//! Juga menulis `model.json` sesuai kontrak `h3d_segment::ModelSpec`.

use std::fs;
use std::path::Path;

use burn::module::Module;
use burn::tensor::Tensor;
use prost::Message;
use tract_onnx::pb;

use h3d_segment::{ClassSpec, ModelSpec};

use crate::model::{DoubleConv, UNet};
use crate::train::TrainMeta;
use crate::TrainError;

/// Nama tensor masukan graf.
pub const INPUT_NAME: &str = "input";
/// Nama tensor keluaran graf.
pub const OUTPUT_NAME: &str = "logits";

fn to_vec<const D: usize>(t: Tensor<D>) -> Result<Vec<f32>, TrainError> {
    let d = t.into_data_as::<f32>();
    d.as_slice::<f32>()
        .map(|s| s.to_vec())
        .map_err(|e| TrainError::Model(format!("bobot bukan f32: {e:?}")))
}

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

fn float_tensor(name: &str, dims: &[i64], data: Vec<f32>) -> pb::TensorProto {
    pb::TensorProto {
        dims: dims.to_vec(),
        data_type: pb::tensor_proto::DataType::Float as i32,
        float_data: data,
        name: name.to_string(),
        ..Default::default()
    }
}

fn attr_ints(name: &str, v: &[i64]) -> pb::AttributeProto {
    pb::AttributeProto {
        name: name.into(),
        r#type: pb::attribute_proto::AttributeType::Ints as i32,
        ints: v.to_vec(),
        ..Default::default()
    }
}

fn attr_int(name: &str, v: i64) -> pb::AttributeProto {
    pb::AttributeProto {
        name: name.into(),
        r#type: pb::attribute_proto::AttributeType::Int as i32,
        i: v,
        ..Default::default()
    }
}

/// Pembangun graf berurutan.
struct Graph {
    nodes: Vec<pb::NodeProto>,
    inits: Vec<pb::TensorProto>,
    counter: usize,
}

impl Graph {
    fn fresh(&mut self, prefix: &str) -> String {
        self.counter += 1;
        format!("{prefix}_{}", self.counter)
    }

    fn node(&mut self, op: &str, inputs: Vec<String>, attrs: Vec<pb::AttributeProto>) -> String {
        let out = self.fresh(&op.to_lowercase());
        self.nodes.push(pb::NodeProto {
            input: inputs,
            output: vec![out.clone()],
            name: out.clone(),
            op_type: op.into(),
            domain: String::new(),
            attribute: attrs,
            doc_string: String::new(),
        });
        out
    }

    fn init(&mut self, prefix: &str, dims: &[i64], data: Vec<f32>) -> String {
        let name = self.fresh(prefix);
        self.inits.push(float_tensor(&name, dims, data));
        name
    }

    /// Conv dengan bobot `[co, ci, k, k]`, bias `[co]`.
    fn conv(&mut self, x: String, w: Vec<f32>, dims: [i64; 4], b: Vec<f32>, pad: i64) -> String {
        let k = dims[2];
        let wn = self.init("w", &dims, w);
        let bn = self.init("b", &[dims[0]], b);
        self.node(
            "Conv",
            vec![x, wn, bn],
            vec![
                attr_ints("kernel_shape", &[k, k]),
                attr_ints("pads", &[pad, pad, pad, pad]),
                attr_ints("strides", &[1, 1]),
            ],
        )
    }

    fn relu(&mut self, x: String) -> String {
        self.node("Relu", vec![x], vec![])
    }

    fn maxpool2(&mut self, x: String) -> String {
        self.node(
            "MaxPool",
            vec![x],
            vec![
                attr_ints("kernel_shape", &[2, 2]),
                attr_ints("strides", &[2, 2]),
            ],
        )
    }

    /// ConvTranspose 2×2 langkah 2, bobot `[ci, co, 2, 2]`.
    fn up2(&mut self, x: String, w: Vec<f32>, dims: [i64; 4], b: Vec<f32>) -> String {
        let wn = self.init("wt", &dims, w);
        let bn = self.init("bt", &[dims[1]], b);
        self.node(
            "ConvTranspose",
            vec![x, wn, bn],
            vec![
                attr_ints("kernel_shape", &[2, 2]),
                attr_ints("strides", &[2, 2]),
            ],
        )
    }

    fn concat(&mut self, a: String, b: String) -> String {
        self.node("Concat", vec![a, b], vec![attr_int("axis", 1)])
    }
}

/// Bobot Conv setelah BatchNorm dilipat: `w' = w·γ/√(σ²+ε)`,
/// `b' = (b − μ)·γ/√(σ²+ε) + β`.
pub fn fold_bn(
    w: &[f32],
    b: &[f32],
    gamma: &[f32],
    beta: &[f32],
    mean: &[f32],
    var: &[f32],
    eps: f64,
) -> Result<(Vec<f32>, Vec<f32>), TrainError> {
    let co = b.len();
    if co == 0
        || !w.len().is_multiple_of(co)
        || [gamma, beta, mean, var].iter().any(|v| v.len() != co)
    {
        return Err(TrainError::Model(
            "ukuran BatchNorm/Conv tidak konsisten".into(),
        ));
    }
    let per = w.len() / co;
    let scale: Vec<f32> = (0..co)
        .map(|o| gamma[o] / (var[o] as f64 + eps).sqrt() as f32)
        .collect();
    let w2 = w
        .iter()
        .enumerate()
        .map(|(i, &x)| x * scale[i / per])
        .collect();
    let b2 = (0..co)
        .map(|o| (b[o] - mean[o]) * scale[o] + beta[o])
        .collect();
    Ok((w2, b2))
}

fn conv_bn(
    g: &mut Graph,
    x: String,
    conv: &burn::nn::conv::Conv2d,
    bn: &burn::nn::BatchNorm,
) -> Result<String, TrainError> {
    let dims = conv.weight.val().dims();
    let dims_i = [
        dims[0] as i64,
        dims[1] as i64,
        dims[2] as i64,
        dims[3] as i64,
    ];
    let w = to_vec(conv.weight.val())?;
    let b = match &conv.bias {
        Some(p) => to_vec(p.val())?,
        None => vec![0.0; dims[0]],
    };
    let gamma = to_vec(bn.gamma.val())?;
    let beta = to_vec(bn.beta.val())?;
    let mean = to_vec(bn.running_mean.value())?;
    let var = to_vec(bn.running_var.value())?;
    let (w2, b2) = fold_bn(&w, &b, &gamma, &beta, &mean, &var, bn.epsilon)?;
    let pad = (dims[2] / 2) as i64;
    Ok(g.conv(x, w2, dims_i, b2, pad))
}

fn double_conv(g: &mut Graph, x: String, dc: &DoubleConv) -> Result<String, TrainError> {
    let x = conv_bn(g, x, &dc.conv1, &dc.bn1)?;
    let x = g.relu(x);
    let x = conv_bn(g, x, &dc.conv2, &dc.bn2)?;
    Ok(g.relu(x))
}

/// Bangun byte ONNX dari model untuk masukan `[1, in, s, s]`.
pub fn export_unet(model: &UNet, s: usize) -> Result<Vec<u8>, TrainError> {
    let depth = model.depth();
    if s == 0 || !s.is_multiple_of(1 << depth) {
        return Err(TrainError::InvalidParameter(format!(
            "input_size {s} harus kelipatan {}",
            1 << depth
        )));
    }
    let model = model.valid();
    let in_ch = model.enc[0].conv1.weight.val().dims()[1];
    let n_classes = model.head.weight.val().dims()[0];
    let mut g = Graph {
        nodes: Vec::new(),
        inits: Vec::new(),
        counter: 0,
    };
    let mut x = INPUT_NAME.to_string();
    let mut skips = Vec::with_capacity(depth);
    for e in &model.enc {
        x = double_conv(&mut g, x, e)?;
        skips.push(x.clone());
        x = g.maxpool2(x);
    }
    x = double_conv(&mut g, x, &model.bottleneck)?;
    for i in (0..depth).rev() {
        let up = &model.up[i];
        let dims = up.weight.val().dims();
        let w = to_vec(up.weight.val())?;
        let b = match &up.bias {
            Some(p) => to_vec(p.val())?,
            None => vec![0.0; dims[1]],
        };
        x = g.up2(
            x,
            w,
            [
                dims[0] as i64,
                dims[1] as i64,
                dims[2] as i64,
                dims[3] as i64,
            ],
            b,
        );
        x = g.concat(x, skips[i].clone());
        x = double_conv(&mut g, x, &model.dec[i])?;
    }
    // Head 1×1 tanpa BN.
    let hd = model.head.weight.val().dims();
    let hw = to_vec(model.head.weight.val())?;
    let hb = match &model.head.bias {
        Some(p) => to_vec(p.val())?,
        None => vec![0.0; hd[0]],
    };
    let out = g.conv(x, hw, [hd[0] as i64, hd[1] as i64, 1, 1], hb, 0);
    // Ganti nama keluaran terakhir.
    if let Some(last) = g.nodes.last_mut() {
        last.output = vec![OUTPUT_NAME.to_string()];
    }
    debug_assert!(out.starts_with("conv"));
    let graph = pb::GraphProto {
        node: g.nodes,
        name: "h3d-unet".into(),
        initializer: g.inits,
        input: vec![value_info(
            INPUT_NAME,
            &[1, in_ch as i64, s as i64, s as i64],
        )],
        output: vec![value_info(
            OUTPUT_NAME,
            &[1, n_classes as i64, s as i64, s as i64],
        )],
        ..Default::default()
    };
    let m = pb::ModelProto {
        ir_version: 8,
        producer_name: "h3d-train".into(),
        graph: Some(graph),
        opset_import: vec![pb::OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
        ..Default::default()
    };
    Ok(m.encode_to_vec())
}

/// `ModelSpec` untuk `model.json` dari metadata latih.
pub fn model_spec(meta: &TrainMeta, name: &str, version: &str, input_size: usize) -> ModelSpec {
    ModelSpec {
        name: name.to_string(),
        version: Some(version.to_string()),
        input_size,
        mpp: meta.mpp,
        mean: meta.mean,
        std: meta.std,
        classes: meta
            .classes
            .iter()
            .enumerate()
            .map(|(i, c)| ClassSpec {
                index: i,
                label: c.clone(),
            })
            .collect(),
        task: "segmentation".to_string(),
    }
}

/// Tulis `model.onnx` + `model.json` ke `out_dir` (atomik), lalu verifikasi
/// bahwa tract bisa memuatnya.
pub fn write_model_dir(
    model: &UNet,
    meta: &TrainMeta,
    name: &str,
    version: &str,
    input_size: usize,
    out_dir: &Path,
) -> Result<ModelSpec, TrainError> {
    fs::create_dir_all(out_dir).map_err(|e| TrainError::io(out_dir, e))?;
    let onnx = export_unet(model, input_size)?;
    let spec = model_spec(meta, name, version, input_size);
    spec.validate()?;
    // Verifikasi sebelum ditulis: berkas yang tidak bisa dimuat tidak pernah ada.
    h3d_segment::TractSegmenter::from_bytes(spec.clone(), &onnx)?;
    let onnx_path = out_dir.join("model.onnx");
    let tmp = out_dir.join("model.onnx.tmp");
    fs::write(&tmp, &onnx).map_err(|e| TrainError::io(&tmp, e))?;
    fs::rename(&tmp, &onnx_path).map_err(|e| TrainError::io(&onnx_path, e))?;
    let json_path = out_dir.join("model.json");
    let tmp = out_dir.join("model.json.tmp");
    fs::write(&tmp, serde_json::to_string_pretty(&spec)?).map_err(|e| TrainError::io(&tmp, e))?;
    fs::rename(&tmp, &json_path).map_err(|e| TrainError::io(&json_path, e))?;
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_bn_identity_when_bn_neutral() {
        let (w, b) = fold_bn(
            &[1.0, 2.0, 3.0, 4.0],
            &[0.5, -0.5],
            &[1.0, 1.0],
            &[0.0, 0.0],
            &[0.0, 0.0],
            &[1.0, 1.0],
            0.0,
        )
        .expect("ok");
        assert_eq!(w, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(b, vec![0.5, -0.5]);
    }

    #[test]
    fn fold_bn_scales_per_output_channel() {
        let (w, b) = fold_bn(
            &[1.0, 1.0, 1.0, 1.0],
            &[0.0, 0.0],
            &[2.0, 4.0],
            &[1.0, 1.0],
            &[0.0, 0.0],
            &[3.0, 15.0],
            1.0,
        )
        .expect("ok");
        // scale = γ/√(σ²+ε) = 2/2 = 1 dan 4/4 = 1.
        assert_eq!(w, vec![1.0, 1.0, 1.0, 1.0]);
        assert_eq!(b, vec![1.0, 1.0]);
        assert!(fold_bn(&[1.0], &[0.0], &[1.0, 1.0], &[0.0], &[0.0], &[1.0], 0.0).is_err());
    }
}
