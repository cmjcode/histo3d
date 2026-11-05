#!/usr/bin/env bash
# HISTO3D — rilis lokal (pola engram/LLM-MEMORY):
#   gerbang mutu -> bump versi -> commit -> tag -> push
#   -> build macOS (.app + .dmg + CLI h3d) -> sign + notarisasi -> GitHub Release.
#
#   sh build.sh 0.1.1
#
# Kredensial notarisasi dibaca dari .env (sudah di .gitignore, JANGAN di-commit):
#   APPLE_ID="email@apple-id"
#   APPLE_PASSWORD="xxxx-xxxx-xxxx-xxxx"   # app-specific password (appleid.apple.com)
#   APPLE_SIGNING_IDENTITY / APPLE_TEAM_ID opsional (default di bawah).
#
# Variabel opsional: SKIP_CHECK=1 melewati gerbang mutu (fmt/clippy/test/iOS).
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
[ -f "$ROOT/.env" ] && { set -a; source "$ROOT/.env"; set +a; }

# ===== Kredensial notarisasi ================================================
export APPLE_SIGNING_IDENTITY="${APPLE_SIGNING_IDENTITY:-Developer ID Application: PT. VNEU TEKNOLOGI INDONESIA (YD4J5Z6A4G)}"
export APPLE_TEAM_ID="${APPLE_TEAM_ID:-YD4J5Z6A4G}"
export APPLE_ID="${APPLE_ID:-ISI_APPLE_ID}"
export APPLE_PASSWORD="${APPLE_PASSWORD:-xxxx-xxxx-xxxx-xxxx}"
# ============================================================================

APP_NAME="HISTO3D"
BIN_NAME="histo3d"
CLI_NAME="h3d"
MIN_FREE_GB=10
DISCLAIMER="Hanya untuk riset, bukan untuk diagnosis."

VERSION="${1:-}"
if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Pemakaian: sh build.sh <versi>   contoh: sh build.sh 0.1.1" >&2
  exit 1
fi
TAG="v$VERSION"

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mGagal: %s\033[0m\n' "$*" >&2; exit 1; }

# ---- Cek awal --------------------------------------------------------------
if [[ "$APPLE_ID" == ISI_* || "$APPLE_PASSWORD" == xxxx* ]]; then
  die "isi APPLE_ID & APPLE_PASSWORD di .env dulu"
fi
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "folder ini belum repo git"
git remote get-url origin >/dev/null 2>&1 || die "remote 'origin' belum diset"
gh auth status >/dev/null 2>&1 || die "gh belum login (jalankan: gh auth login)"
command -v xcrun >/dev/null || die "Xcode command line tools belum terpasang"
security find-identity -v -p codesigning | grep -qF "$APPLE_SIGNING_IDENTITY" \
  || die "identitas penandatangan tidak ada di keychain: $APPLE_SIGNING_IDENTITY"

TAG_EXISTS=0
if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
  TAG_EXISTS=1
  echo "Tag $TAG sudah ada -> lewati gerbang mutu, set versi, commit & tag; langsung build macOS."
fi

free_gb=$(df -g "$ROOT" | awk 'NR==2 {print $4}')
if (( free_gb < MIN_FREE_GB )); then
  echo "Peringatan: ruang disk tersisa ${free_gb} GB (disarankan >= ${MIN_FREE_GB} GB)."
  read -r -p "Lanjut juga? [y/N] " ans
  [[ "$ans" == [yY] ]] || exit 1
fi

if (( ! TAG_EXISTS )); then
if [[ -n "$(git status --porcelain)" ]]; then
  git status --short
  read -r -p "Working tree tidak bersih; perubahan di atas TIDAK ikut commit rilis. Lanjut? [y/N] " ans
  [[ "$ans" == [yY] ]] || exit 1
fi

# ---- 0. Gerbang mutu (CLAUDE.md) --------------------------------------------
if [[ "${SKIP_CHECK:-0}" != 1 ]]; then
  step "0. Gerbang mutu (fmt, clippy, test, cek iOS)"
  bash "$ROOT/build_macos.sh" check
fi

# ---- 1. Ubah versi ----------------------------------------------------------
step "1. Set versi $VERSION"
# [workspace.package] version
sed -i '' "/^\[workspace.package\]/,/^\[/ s/^version = \".*\"/version = \"$VERSION\"/" Cargo.toml
# versi dependensi path internal di [workspace.dependencies] (h3d-* = { path = "crates/...", version = "..." })
sed -i '' -E "/path = \"crates\//s/version = \"[^\"]*\"/version = \"$VERSION\"/" Cargo.toml
# Cargo.lock ikut diperbarui
cargo update --workspace --offline >/dev/null 2>&1 || cargo update --workspace
grep -n '^version' Cargo.toml | head -1

# ---- 2 & 3. Commit, tag, push ----------------------------------------------
step "2-3. Commit, tag $TAG, push"
git add Cargo.toml Cargo.lock
git commit -m "release $TAG"
git tag -a "$TAG" -m "release $TAG"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
git push origin "$BRANCH"
git push origin "$TAG"
else
  if ! git ls-remote --exit-code --tags origin "refs/tags/$TAG" >/dev/null 2>&1; then
    step "Push tag $TAG yang belum ada di remote"
    git push origin "$TAG"
  fi
fi

HOST_TARGET="$(rustc -vV | sed -n 's/^host: //p')"
case "$HOST_TARGET" in
  aarch64-*) ARCH=aarch64 ;;
  x86_64-*)  ARCH=x64 ;;
  *) die "arsitektur $HOST_TARGET tidak dikenal" ;;
esac
DIST="$ROOT/target/dist"
rm -rf "$DIST" && mkdir -p "$DIST"

# ---- 4. Build .app + sign (hardened runtime) + notarisasi --------------------
step "4. Build $APP_NAME.app, sign & notarisasi"
bash "$ROOT/build_macos.sh" app
APP="$ROOT/dist/macos/$APP_NAME.app"
[[ -d "$APP" ]] || die "bundel $APP tidak ditemukan"
xattr -cr "$APP"
codesign --force --options runtime --timestamp --sign "$APPLE_SIGNING_IDENTITY" "$APP/Contents/MacOS/$BIN_NAME"
codesign --force --options runtime --timestamp --sign "$APPLE_SIGNING_IDENTITY" "$APP"
codesign --verify --deep --strict --verbose=2 "$APP"

notarize() {
  xcrun notarytool submit "$1" \
    --apple-id "$APPLE_ID" --password "$APPLE_PASSWORD" --team-id "$APPLE_TEAM_ID" --wait
}
APP_ZIP="$DIST/$APP_NAME-notarize.zip"
ditto -c -k --keepParent "$APP" "$APP_ZIP"
notarize "$APP_ZIP"
rm "$APP_ZIP"
xcrun stapler staple "$APP"
spctl -a -t exec -vv "$APP"

# ---- 4a. DMG (+ pintasan /Applications), sign, notarisasi, staple -----------
step "4a. Buat DMG, sign & notarisasi"
DMG="$DIST/${APP_NAME}_${VERSION}_${ARCH}.dmg"
STAGE="$(mktemp -d)"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
hdiutil create -volname "$APP_NAME $VERSION" -srcfolder "$STAGE" -ov -format UDZO "$DMG"
rm -rf "$STAGE"
codesign --force --sign "$APPLE_SIGNING_IDENTITY" --timestamp "$DMG"
notarize "$DMG"
xcrun stapler staple "$DMG"
spctl -a -t open --context context:primary-signature -v "$DMG"

# ---- 4b. CLI h3d macOS + sign ----------------------------------------------
# Biner CLI polos tidak bisa di-staple; notarisasi lewat zip agar Gatekeeper
# mengenalinya saat daring.
step "4b. Build CLI '$CLI_NAME' macOS, sign & notarisasi"
cargo build --release --locked -p h3d-cli
cp "$ROOT/target/release/$CLI_NAME" "$DIST/$CLI_NAME"
codesign --force --options runtime --timestamp --sign "$APPLE_SIGNING_IDENTITY" "$DIST/$CLI_NAME"
CLI_ZIP="$DIST/$CLI_NAME-notarize.zip"
ditto -c -k "$DIST/$CLI_NAME" "$CLI_ZIP"
notarize "$CLI_ZIP"
rm "$CLI_ZIP"
CLI_TGZ="$DIST/$CLI_NAME-$VERSION-$HOST_TARGET.tar.gz"
tar -czf "$CLI_TGZ" -C "$DIST" "$CLI_NAME"
rm "$DIST/$CLI_NAME"
(cd "$DIST" && for f in *.tar.gz *.dmg; do shasum -a 256 "$f" > "$f.sha256"; done)
ls -la "$DIST"

# ---- 5. Upload ke GitHub Release -------------------------------------------
step "5. Upload ke GitHub Release $TAG"
gh release view "$TAG" >/dev/null 2>&1 \
  || gh release create "$TAG" --title "$APP_NAME $TAG" --notes "$DISCLAIMER" --generate-notes
for f in "$DIST"/*; do
  for attempt in 1 2 3 4 5; do
    gh release upload "$TAG" "$f" --clobber && break
    [[ $attempt -lt 5 ]] || die "gagal mengunggah $f ke GitHub Release"
    sleep $((attempt * 3))
  done
done

step "Selesai: $(gh release view "$TAG" --json url -q .url)"
