#!/usr/bin/env python3
"""Ekspor model StarDist 2D (Keras) menjadi paket model HISTO3D:
`model.onnx` + `model.json` dengan `task = "nuclei"` (histo3d.md 8.5.4).

Dijalankan sekali di luar aplikasi (aplikasi tidak pernah mengunduh apa pun).
Butuh: stardist, tensorflow, tf2onnx, onnx (opsional onnxruntime untuk uji).

    pip install stardist tensorflow tf2onnx onnx onnxruntime
    python3 script/export_stardist_onnx.py --out models/stardist-he
    # model lain: --pretrained 2D_versatile_fluo, atau --model-dir <folder StarDist>

Lalu di aplikasi: laci "Label dan usulan" > "Hitung sel", atau
    h3d cells <proyek.h3d> --model models/stardist-he

Model `2D_versatile_he` (Schmidt dkk. 2018; lisensi BSD-3) dilatih pada citra
H&E ~0,5 µm/px; `--mpp` menentukan resolusi kerja yang dibaca dari slide.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", required=True, type=Path, help="folder keluaran (model.onnx + model.json)")
    ap.add_argument("--pretrained", default="2D_versatile_he", help="nama model pretrained StarDist")
    ap.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="folder model StarDist hasil latih sendiri (menggantikan --pretrained)",
    )
    ap.add_argument("--input-size", type=int, default=512, help="sisi tile (kelipatan 16)")
    ap.add_argument("--mpp", type=float, default=0.5, help="resolusi kerja, µm/px")
    ap.add_argument("--name", default=None, help="nama model (bawaan: nama pretrained/folder)")
    ap.add_argument("--version", default="1", help="versi model")
    ap.add_argument("--opset", type=int, default=13)
    args = ap.parse_args()

    if args.input_size % 16 != 0 or args.input_size <= 0:
        print("galat: --input-size harus kelipatan 16", file=sys.stderr)
        return 1

    try:
        import numpy as np
        import tensorflow as tf
        import tf2onnx
        from stardist.models import StarDist2D
    except ImportError as e:  # lingkungan pengguna
        print(
            f"galat: pustaka belum terpasang ({e}); pip install stardist tensorflow tf2onnx onnx",
            file=sys.stderr,
        )
        return 1

    if args.model_dir is not None:
        model = StarDist2D(None, name=args.model_dir.name, basedir=str(args.model_dir.parent))
        name = args.name or args.model_dir.name
    else:
        model = StarDist2D.from_pretrained(args.pretrained)
        name = args.name or args.pretrained

    cfg = model.config
    n_rays = int(cfg.n_rays)
    grid = [int(g) for g in cfg.grid]
    n_channel = int(cfg.n_channel_in)
    if n_channel != 3:
        print(f"galat: HISTO3D hanya mendukung masukan RGB (n_channel_in={n_channel})", file=sys.stderr)
        return 1
    thr = model.thresholds
    s = args.input_size

    # Keras StarDist: masukan NHWC [N, H, W, 3] ternormalisasi persentil 0..1;
    # keluaran [prob (1 kanal), dist (n_rays kanal)] pada H/grid × W/grid.
    keras_model = model.keras_model
    spec = (tf.TensorSpec((1, s, s, 3), tf.float32, name="input"),)
    onnx_model, _ = tf2onnx.convert.from_keras(keras_model, input_signature=spec, opset=args.opset)

    args.out.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out / "model.onnx"
    tmp = onnx_path.with_suffix(".onnx.tmp")
    tmp.write_bytes(onnx_model.SerializeToString())
    tmp.replace(onnx_path)

    # Uji cepat bentuk keluaran dengan onnxruntime bila ada.
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        x = np.random.default_rng(0).random((1, s, s, 3), dtype=np.float32)
        outs = sess.run(None, {sess.get_inputs()[0].name: x})
        shapes = [o.shape for o in outs]
        print(f"keluaran ONNX: {shapes}")
        chans = sorted(sh[-1] for sh in shapes)
        if chans != sorted([1, n_rays]):
            print(f"galat: kanal keluaran {chans} != [1, {n_rays}]", file=sys.stderr)
            return 1
    except ImportError:
        print("onnxruntime tidak ada; bentuk keluaran tidak diuji (tract memeriksanya saat dimuat)")

    spec_json = {
        "name": name,
        "version": str(args.version),
        "task": "nuclei",
        "input_size": s,
        "mpp": args.mpp,
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "classes": [],
        "nuclei": {
            "n_rays": n_rays,
            "grid": grid,
            "prob_threshold": float(thr.prob),
            "nms_threshold": float(thr.nms),
            "layout": "nhwc",
            "percentile_norm": [1.0, 99.8],
            "threshold_delta": 0.1,
            "margin_px": 32,
        },
    }
    json_path = args.out / "model.json"
    tmp = json_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(spec_json, indent=2) + "\n")
    tmp.replace(json_path)
    print(f"selesai: {onnx_path} dan {json_path}")
    print("Hanya untuk riset, bukan untuk diagnosis.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
