#!/usr/bin/env bash
# Ekspor checkpoint hasil pelatihan ML (`model.bpk` + `train_meta.json`)
# menjadi paket model inferensi (`model.onnx` + `model.json`) yang diverifikasi tract.
#
# Pakai:
#   script/export_model.sh [run_dir] [out_dir] [nama_model] [versi] [input_size]
#
# Contoh (bawaan):
#   script/export_model.sh
#   script/export_model.sh ml/runs/v1 models/tumor-he-v1 tumor-he-v1 0.1.0 512

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

RUN_ARG="${1:-$ROOT/ml/runs/v1}"
OUT_ARG="${2:-$ROOT/models/tumor-he-v1}"
NAME="${3:-tumor-he-v1}"
VERSION="${4:-0.1.0}"
INPUT_SIZE="${5:-512}"

# Ubah path relatif menjadi absolut terhadap direktori pemanggil
case "$RUN_ARG" in
  /*) RUN_DIR="$RUN_ARG" ;;
  *)  RUN_DIR="$(pwd)/$RUN_ARG" ;;
esac

case "$OUT_ARG" in
  /*) OUT_DIR="$OUT_ARG" ;;
  *)  OUT_DIR="$(pwd)/$OUT_ARG" ;;
esac

if [ ! -f "$RUN_DIR/model.bpk" ] || [ ! -f "$RUN_DIR/train_meta.json" ]; then
  echo "galat: '$RUN_DIR' tidak berisi model.bpk dan train_meta.json" >&2
  exit 1
fi

echo "==> Mengekspor model dari: $RUN_DIR"
echo "==> Tujuan keluaran      : $OUT_DIR"

cargo run --release --manifest-path "$ROOT/ml/Cargo.toml" -p h3d-train -- export \
  --run "$RUN_DIR" \
  --out "$OUT_DIR" \
  --name "$NAME" \
  --version "$VERSION" \
  --input-size "$INPUT_SIZE"

echo "==> Selesai: $OUT_DIR/model.onnx & $OUT_DIR/model.json"
