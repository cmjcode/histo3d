#!/bin/bash
# HISTO3D — build macOS. Pemakaian: ./build_macos.sh [app|cli|all|check] [--debug]
# app   : bundel dist/macos/HISTO3D.app (binary h3d-app)
# cli   : binary h3d ke dist/macos/h3d
# check : gerbang mutu lengkap (fmt, clippy, test, cek iOS)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"
APP_NAME="HISTO3D"; BIN_NAME="histo3d"
VERSION=$(grep -A4 '^\[workspace.package\]' Cargo.toml | grep '^version' | head -n1 | cut -d '"' -f2)
BUILD_NUMBER="${BUILD_NUMBER:-$(echo "$VERSION" | tr -d '.' | sed 's/^0*//')}"; [ -z "$BUILD_NUMBER" ] && BUILD_NUMBER=1
APPLE_BUNDLE_ID="${APPLE_BUNDLE_ID:-id.jayuda.histo3d}"
ACTION="${1:-all}"; shift || true
FLAG="--release"; SUB="release"
for a in "$@"; do [ "$a" = "--debug" ] && { FLAG=""; SUB="debug"; }; done
DIST="$SCRIPT_DIR/dist/macos"; mkdir -p "$DIST"

build_app() {
  cargo build $FLAG -p h3d-app
  local b="$DIST/$APP_NAME.app"; rm -rf "$b"; mkdir -p "$b/Contents/MacOS" "$b/Contents/Resources"
  cp "target/$SUB/$BIN_NAME" "$b/Contents/MacOS/"
  sed -e "s/__BUNDLE_ID__/$APPLE_BUNDLE_ID/g" -e "s/__VERSION__/$VERSION/g" -e "s/__BUILD__/$BUILD_NUMBER/g" \
      -e "s/__BIN__/$BIN_NAME/g" -e "s/__APP__/$APP_NAME/g" apple/macos/Info.plist.template > "$b/Contents/Info.plist"
  codesign --force --sign "${APPLE_SIGN_IDENTITY:--}" "$b" 2>/dev/null || true
  echo "[OK] $b"
}
build_cli() { cargo build $FLAG -p h3d-cli; cp "target/$SUB/h3d" "$DIST/h3d"; echo "[OK] $DIST/h3d"; }
check_all() {
  cargo fmt --all -- --check
  cargo clippy --workspace --all-targets -- -D warnings
  cargo test --workspace
  unset SDKROOT CFLAGS CXXFLAGS LDFLAGS CC CXX
  cargo check --target aarch64-apple-ios -p h3d-app
  echo "[OK] semua gerbang hijau"
}
case "$ACTION" in
  app) build_app;; cli) build_cli;; all) build_cli; build_app;; check) check_all;;
  *) echo "aksi tidak dikenal: $ACTION"; exit 1;;
esac
