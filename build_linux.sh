#!/bin/bash
# HISTO3D — build Linux. Pemakaian: ./build_linux.sh [app|cli|all|check] [--debug]
# app   : binary dist/linux/histo3d + berkas desktop dist/linux/histo3d.desktop
# cli   : binary h3d ke dist/linux/h3d
# check : gerbang mutu lengkap (fmt, clippy, test, cek iOS)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"
APP_NAME="HISTO3D"; BIN_NAME="histo3d"
VERSION=$(grep -A4 '^\[workspace.package\]' Cargo.toml | grep '^version' | head -n1 | cut -d '"' -f2)
ACTION="${1:-all}"; shift || true
FLAG="--release"; SUB="release"
for a in "$@"; do [ "$a" = "--debug" ] && { FLAG=""; SUB="debug"; }; done
DIST="$SCRIPT_DIR/dist/linux"; mkdir -p "$DIST"

build_app() {
  cargo build $FLAG -p h3d-app -p h3d-updater
  cp "target/$SUB/$BIN_NAME" "$DIST/$BIN_NAME"
  # Pembaru (proses terpisah; h3d-app sendiri tanpa HTTP, aturan 11).
  cp "target/$SUB/histo3d-updater" "$DIST/histo3d-updater"
  chmod +x "$DIST/$BIN_NAME" "$DIST/histo3d-updater"
  cp crates/h3d-app/assets/histo3d-512.png "$DIST/$BIN_NAME.png"
  cat > "$DIST/$BIN_NAME.desktop" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=$APP_NAME
Comment=Rekonstruksi 3D tumor dari irisan histologi serial (hanya untuk riset, bukan untuk diagnosis)
Exec=$DIST/$BIN_NAME
Icon=$DIST/$BIN_NAME.png
Terminal=false
Categories=Science;MedicalSoftware;Graphics;
StartupWMClass=$APP_NAME
X-HISTO3D-Version=$VERSION
EOF
  echo "[OK] $DIST/$BIN_NAME"
}
build_cli() {
  cargo build $FLAG -p h3d-cli
  cp "target/$SUB/h3d" "$DIST/h3d"
  chmod +x "$DIST/h3d"
  echo "[OK] $DIST/h3d"
}
check_all() {
  cargo fmt --all -- --check
  cargo clippy --workspace --all-targets -- -D warnings -A clippy::chunks_exact_to_as_chunks -A clippy::needless_late_init
  cargo test --workspace
  if command -v xcrun >/dev/null 2>&1; then
    unset SDKROOT CFLAGS CXXFLAGS LDFLAGS CC CXX
    cargo check --target aarch64-apple-ios -p h3d-app
  fi
  echo "[OK] semua gerbang hijau"
}
case "$ACTION" in
  app) build_app;; cli) build_cli;; all) build_cli; build_app;; check) check_all;;
  *) echo "aksi tidak dikenal: $ACTION"; exit 1;;
esac

