//! Tes arsitektur (7.3, aturan keras 2 dan 11): `h3d-engine` dan seluruh
//! pohon dependensinya tidak boleh memuat crate GUI, dialog berkas, HTTP,
//! maupun OCCT/ITK/VTK/OpenCV. Meniru tes bernama sama di DUCAD: membaca
//! `cargo metadata` lalu menelusuri graf resolve dari `h3d-engine`.

use std::collections::{HashMap, HashSet, VecDeque};
use std::process::Command;

const FORBIDDEN: &[&str] = &[
    // GUI / render / dialog
    "egui",
    "eframe",
    "egui-wgpu",
    "egui_wgpu",
    "wgpu",
    "winit",
    "rfd",
    "h3d-render",
    "h3d-ui",
    "h3d-app",
    // jaringan (K13)
    "ureq",
    "reqwest",
    "hyper",
    "tokio",
    // pustaka C++ yang dilarang (aturan keras 1)
    "opencascade",
    "opencascade-sys",
    "occt-sys",
    "opencv",
    "vtk-rs",
    "openslide-rs",
];

#[test]
fn engine_has_no_gui_dependency() {
    let manifest = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");
    let out = Command::new(env!("CARGO"))
        .args([
            "metadata",
            "--format-version",
            "1",
            "--manifest-path",
            manifest,
        ])
        .output()
        .expect("cargo metadata gagal dijalankan");
    assert!(out.status.success(), "cargo metadata keluar dengan error");
    let meta: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("keluaran cargo metadata bukan JSON");

    let packages = meta["packages"].as_array().expect("packages");
    let id_to_name: HashMap<&str, &str> = packages
        .iter()
        .map(|p| {
            (
                p["id"].as_str().unwrap_or(""),
                p["name"].as_str().unwrap_or(""),
            )
        })
        .collect();

    let resolve = meta["resolve"]["nodes"].as_array().expect("resolve.nodes");
    let mut deps_of: HashMap<&str, Vec<&str>> = HashMap::new();
    for node in resolve {
        let id = node["id"].as_str().unwrap_or("");
        // Hanya dependensi normal dan build; dev-dependencies tidak ikut biner.
        let deps: Vec<&str> = node["deps"]
            .as_array()
            .map(|d| {
                d.iter()
                    .filter(|dep| {
                        dep["dep_kinds"].as_array().is_none_or(|kinds| {
                            kinds
                                .iter()
                                .any(|k| k["kind"].is_null() || k["kind"] == "build")
                        })
                    })
                    .filter_map(|dep| dep["pkg"].as_str())
                    .collect()
            })
            .unwrap_or_default();
        deps_of.insert(id, deps);
    }

    let root = id_to_name
        .iter()
        .find(|(_, n)| **n == "h3d-engine")
        .map(|(id, _)| *id)
        .expect("h3d-engine tidak ada di metadata");

    let mut seen: HashSet<&str> = HashSet::new();
    let mut queue = VecDeque::from([root]);
    let mut offenders: Vec<String> = Vec::new();
    while let Some(id) = queue.pop_front() {
        if !seen.insert(id) {
            continue;
        }
        let name = id_to_name.get(id).copied().unwrap_or("");
        if FORBIDDEN.contains(&name) {
            offenders.push(name.to_owned());
        }
        for d in deps_of.get(id).into_iter().flatten() {
            queue.push_back(d);
        }
    }
    offenders.sort();
    offenders.dedup();
    assert!(
        offenders.is_empty(),
        "h3d-engine menarik crate terlarang: {offenders:?}"
    );
}
