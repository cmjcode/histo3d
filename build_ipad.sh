#!/bin/bash
# ==============================================================================
# HISTO3D — build iPadOS (adaptasi dari DUCAD/build_ipad.sh, tanpa OCCT/CMake)
# ==============================================================================
# Pemakaian: ./build_ipad.sh [AKSI] [--debug] [--bundle-id ID] [--sim-name NAMA]
#                            [--identity ID] [--profile PATH]
# Aksi:
#   check          cargo check --target aarch64-apple-ios -p h3d-app (gerbang CI)
#   device         binary + bundel .app untuk iPad fisik (aarch64-apple-ios)
#   sim            binary + bundel .app untuk simulator (aarch64-apple-ios-sim)
#   run-sim        build sim, nyalakan simulator iPad, pasang dan jalankan
#   archive        .xcarchive dari bundel device
#   ipa            .ipa (Payload) dari bundel device
#   clean          hapus artefak iOS
#   install-deps   rustup target add aarch64-apple-ios aarch64-apple-ios-sim
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
[ -f .env ] && { set -a; source .env; set +a; }

APP_NAME="HISTO3D"
PKG_NAME="h3d-app"
BIN_NAME="histo3d"
IOS_ARM_TARGET="aarch64-apple-ios"
IOS_SIM_TARGET="aarch64-apple-ios-sim"
DIST_IOS_DIR="$SCRIPT_DIR/dist/ios"
VERSION=$(grep -A4 '^\[workspace.package\]' Cargo.toml | grep '^version' | head -n1 | cut -d '"' -f2)
BUILD_NUMBER="${BUILD_NUMBER:-$(echo "$VERSION" | tr -d '.' | sed 's/^0*//')}"
[ -z "$BUILD_NUMBER" ] && BUILD_NUMBER=1
APPLE_BUNDLE_ID="${APPLE_BUNDLE_ID:-id.jayuda.histo3d.ios}"
BUILD_MODE="release"; CARGO_BUILD_FLAG="--release"; BUILD_SUBDIR="release"
CUSTOM_SIM_NAME=""; SIGN_IDENTITY="${APPLE_SIGN_IDENTITY:-}"; PROVISIONING_PROFILE="${APPLE_PROVISIONING_PROFILE:-}"

ACTION="${1:-check}"; shift || true
while [ $# -gt 0 ]; do
  case "$1" in
    --debug) BUILD_MODE="debug"; CARGO_BUILD_FLAG=""; BUILD_SUBDIR="debug";;
    --bundle-id) APPLE_BUNDLE_ID="$2"; shift;;
    --sim-name) CUSTOM_SIM_NAME="$2"; shift;;
    --identity) SIGN_IDENTITY="$2"; shift;;
    --profile) PROVISIONING_PROFILE="$2"; shift;;
    -h|--help) sed -n '2,16p' "$0"; exit 0;;
    *) echo "[ERROR] opsi tidak dikenal: $1"; exit 1;;
  esac; shift
done

info() { echo -e "\033[0;34m[INFO]\033[0m $*"; }
ok()   { echo -e "\033[0;32m[OK]\033[0m $*"; }
err()  { echo -e "\033[0;31m[ERROR]\033[0m $*" >&2; }

setup_env() {
  # Variabel Xcode mengganggu kompilasi proc-macro host; bersihkan.
  unset SDKROOT CFLAGS CXXFLAGS LDFLAGS CC CXX
}

gen_plist() {
  local target_plist="$1" is_sim="$2"
  local platform="iPhoneOS" sdk="iphoneos"
  [ "$is_sim" = "true" ] && { platform="iPhoneSimulator"; sdk="iphonesimulator"; }
  sed -e "s/__BUNDLE_ID__/$APPLE_BUNDLE_ID/g" -e "s/__VERSION__/$VERSION/g" \
      -e "s/__BUILD__/$BUILD_NUMBER/g" -e "s/__PLATFORM__/$platform/g" \
      -e "s/__SDK__/$sdk/g" -e "s/__BIN__/$BIN_NAME/g" -e "s/__APP__/$APP_NAME/g" \
      apple/ios/Info.plist.template > "$target_plist"
}

sign_bundle() {
  local bundle="$1" is_sim="$2"
  command -v codesign >/dev/null || { info "codesign tidak ada; lewati."; return; }
  local id="$SIGN_IDENTITY"
  if [ "$is_sim" != "true" ] && [ -z "$id" ]; then
    id=$(security find-identity -v -p codesigning 2>/dev/null | grep -E "Apple (Development|Distribution)" | head -n1 | sed -E 's/.*"([^"]+)".*/\1/' || true)
  fi
  [ -z "$id" ] && id="-"
  local ent=""
  [ "$is_sim" != "true" ] && [ -f apple/ios/HISTO3D.entitlements ] && ent="--entitlements apple/ios/HISTO3D.entitlements"
  codesign --force --sign "$id" --timestamp=none $ent "$bundle" || codesign --force --sign "-" --timestamp=none "$bundle"
  ok "ditandatangani dengan: $id"
}

build_bundle() {
  local target="$1" is_sim="$2"
  local app_dir="$APP_NAME.app"; [ "$is_sim" = "true" ] && app_dir="$APP_NAME-Simulator.app"
  local bundle="$DIST_IOS_DIR/$app_dir"
  setup_env
  info "cargo build $CARGO_BUILD_FLAG -p $PKG_NAME --target $target"
  cargo build $CARGO_BUILD_FLAG -p "$PKG_NAME" --target "$target"
  local bin="target/$target/$BUILD_SUBDIR/$BIN_NAME"
  [ -f "$bin" ] || { err "binary tidak ditemukan: $bin"; exit 1; }
  rm -rf "$bundle"; mkdir -p "$bundle"
  cp "$bin" "$bundle/$BIN_NAME"; chmod +x "$bundle/$BIN_NAME"
  gen_plist "$bundle/Info.plist" "$is_sim"
  [ -d apple/Assets.xcassets ] && cp -R apple/Assets.xcassets "$bundle/" 2>/dev/null || true
  if [ -n "$PROVISIONING_PROFILE" ] && [ -f "$PROVISIONING_PROFILE" ]; then
    cp "$PROVISIONING_PROFILE" "$bundle/embedded.mobileprovision"
  fi
  xattr -cr "$bundle" 2>/dev/null || true
  sign_bundle "$bundle" "$is_sim"
  ok "bundel: $bundle"
}

run_sim() {
  build_bundle "$IOS_SIM_TARGET" true
  local bundle="$DIST_IOS_DIR/$APP_NAME-Simulator.app" sim_id=""
  if [ -n "$CUSTOM_SIM_NAME" ]; then
    sim_id=$(xcrun simctl list devices available | grep "$CUSTOM_SIM_NAME" | head -n1 | grep -oE '\([A-F0-9-]{36}\)' | tr -d '()')
  fi
  [ -z "$sim_id" ] && sim_id=$(xcrun simctl list devices | grep -i iPad | grep Booted | head -n1 | grep -oE '\([A-F0-9-]{36}\)' | tr -d '()')
  [ -z "$sim_id" ] && sim_id=$(xcrun simctl list devices available | grep -i iPad | head -n1 | grep -oE '\([A-F0-9-]{36}\)' | tr -d '()')
  [ -z "$sim_id" ] && { err "tidak ada simulator iPad"; exit 1; }
  xcrun simctl boot "$sim_id" 2>/dev/null || true
  open -a Simulator || true
  xcrun simctl install "$sim_id" "$bundle"
  xcrun simctl launch "$sim_id" "$APPLE_BUNDLE_ID"
  ok "berjalan di simulator $sim_id"
}

do_archive() {
  build_bundle "$IOS_ARM_TARGET" false
  local arch="$DIST_IOS_DIR/$APP_NAME.xcarchive"
  rm -rf "$arch"; mkdir -p "$arch/Products/Applications"
  cp -R "$DIST_IOS_DIR/$APP_NAME.app" "$arch/Products/Applications/"
  cat > "$arch/Info.plist" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>ApplicationProperties</key><dict>
    <key>ApplicationPath</key><string>Applications/$APP_NAME.app</string>
    <key>CFBundleIdentifier</key><string>$APPLE_BUNDLE_ID</string>
    <key>CFBundleShortVersionString</key><string>$VERSION</string>
    <key>CFBundleVersion</key><string>$BUILD_NUMBER</string>
  </dict>
  <key>ArchiveVersion</key><integer>2</integer>
  <key>Name</key><string>$APP_NAME</string>
  <key>SchemeName</key><string>$APP_NAME</string>
</dict></plist>
PL
  ok "arsip: $arch"
}

do_ipa() {
  build_bundle "$IOS_ARM_TARGET" false
  local payload="$DIST_IOS_DIR/Payload"
  rm -rf "$payload"; mkdir -p "$payload"
  cp -R "$DIST_IOS_DIR/$APP_NAME.app" "$payload/"
  (cd "$DIST_IOS_DIR" && rm -f "$APP_NAME-$VERSION.ipa" && zip -qr "$APP_NAME-$VERSION.ipa" Payload)
  rm -rf "$payload"
  ok "ipa: $DIST_IOS_DIR/$APP_NAME-$VERSION.ipa"
}

case "$ACTION" in
  check) setup_env; cargo check --target "$IOS_ARM_TARGET" -p "$PKG_NAME"; ok "cek iOS lulus";;
  device) build_bundle "$IOS_ARM_TARGET" false;;
  sim|simulator) build_bundle "$IOS_SIM_TARGET" true;;
  run-sim|launch) run_sim;;
  archive) do_archive;;
  ipa|package) do_ipa;;
  clean) rm -rf "$DIST_IOS_DIR" "target/$IOS_ARM_TARGET" "target/$IOS_SIM_TARGET"; ok "bersih";;
  install-deps) rustup target add "$IOS_ARM_TARGET" "$IOS_SIM_TARGET"; ok "target terpasang";;
  *) err "aksi tidak dikenal: $ACTION"; sed -n '2,16p' "$0"; exit 1;;
esac
