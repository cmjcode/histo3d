#!/usr/bin/env bash
# Pipeline ujung-ke-ujung HISTO3D pada tumpukan Tampere PTENX036-011
# (histo3d.md Fase 1/3/4/5). Masker tumor manual (TumorMasks_sub4) diimpor
# sebagai label "Tumor"; bila folder model ONNX diberikan, segmentasi AI
# dijalankan lebih dulu dan masker manual hanya mengisi piksel kosong.
#
# Pakai: script/run_tampere_pipeline.sh <proyek.h3d> [model_dir] [n_irisan] [full|sub4]
#   full : Images_FullRes (mpp 0,46; hanya im0–im7 yang terunduh)
#   sub4 : Images_sub4 (mpp 1,84; 32 irisan lengkap) — bawaan
# mpp 0,46 µm/px dan jarak 50 µm: lihat ml/README.md (sumber: BMC Cancer 2021).
set -euo pipefail
PROJ=${1:?folder proyek .h3d}
MODEL=${2:-}
N=${3:-32}
MODE=${4:-sub4}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DATA=$ROOT/images/tampere_mouse_prostate/PTENX036-011
H=$ROOT/target/release/h3d
MPP_SUB4=1.84
SPACING=50
case "$MODE" in
  full) IMG=$DATA/Images_FullRes; MPP0=0.46 ;;
  sub4) IMG=$DATA/Images_sub4;    MPP0=$MPP_SUB4 ;;
  *) echo "mode '$MODE' tidak dikenal (full|sub4)" >&2; exit 1 ;;
esac

[ -x "$H" ] || { echo "bangun dulu: cargo build --release -p h3d-cli" >&2; exit 1; }
slides=()
for ((i=0; i<N; i++)); do slides+=("$IMG/im$i.png"); done

"$H" new "$PROJ"
"$H" import "$PROJ" "${slides[@]}" --mpp $MPP0 --spacing-um $SPACING
"$H" register "$PROJ" --model affine --metric ncc
"$H" qc "$PROJ"
if [ -n "$MODEL" ]; then
  "$H" segment "$PROJ" --model "$MODEL"
fi
for ((i=0; i<N; i++)); do
  "$H" labels import-mask "$PROJ" --slide "$i" --mask "$DATA/TumorMasks_sub4/im$i.png" --mpp $MPP_SUB4 --label Tumor
done
"$H" track "$PROJ" --label Tumor
"$H" reconstruct "$PROJ" --per-object
"$H" measure "$PROJ"
"$H" export "$PROJ" --format glb --out "$PROJ/out/glb"
"$H" export "$PROJ" --format report --out "$PROJ/out/report"
echo "selesai: $PROJ/out"
