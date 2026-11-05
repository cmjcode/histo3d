//! Simpan target triple saat kompilasi: nama aset rilis memuatnya
//! (`histo3d-<versi>-<target>.*`, lihat `.github/workflows/release.yml`).

fn main() {
    let target = std::env::var("TARGET").unwrap_or_default();
    println!("cargo:rustc-env=H3D_TARGET={target}");
    println!("cargo:rerun-if-changed=build.rs");
}
