//! Tanam ikon aplikasi ke `histo3d.exe` (Windows). Di target lain
//! `embed_resource` langsung kembali tanpa melakukan apa-apa.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=histo3d.rc");
    println!("cargo:rerun-if-changed=assets/histo3d.ico");
    if let Err(e) = embed_resource::compile("histo3d.rc", embed_resource::NONE).manifest_optional()
    {
        // Ikon hilang tidak boleh menggagalkan build.
        println!("cargo:warning=ikon Windows tidak tertanam: {e}");
    }
}
