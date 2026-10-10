# HISTO3D — panduan agent

Rekonstruksi 3D tumor dari irisan histologi serial. Rancangan lengkap ada di
`histo3d.md`; **baca bagian 5 (koordinat) sebelum menulis kode apa pun**.

## Gerbang mutu (wajib hijau sebelum menyatakan selesai)

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo check --target aarch64-apple-ios -p h3d-app
```

## Aturan keras (histo3d.md bagian 14)

1. **Tanpa OCCT, ITK, VTK, OpenCV.** Tidak ada crate yang menariknya.
2. **`h3d-engine` dan crate di bawahnya bebas GUI** (egui/eframe/wgpu/rfd);
   dijaga tes `engine_has_no_gui_dependency`.
3. **Satuan**: µm `f64` di data; mm `f32` hanya di `h3d-render`. Tidak ada
   koordinat piksel yang disimpan di proyek selain di GeoJSON.
4. **Arah transformasi** mengikuti 5.3: `T_i` memetakan ruang U (slide, µm)
   ke ruang W (tumpukan, µm). Tiap fungsi yang menerima transformasi
   menyebut arahnya di doc comment. Resampling selalu mundur (`T⁻¹`).
5. **Pusat piksel di +0,5** di semua konversi indeks ↔ koordinat.
6. **Tidak ada `unwrap()`/`expect()` pada masukan luar** (berkas, argumen
   CLI, model, metadata slide). Berkas rusak menghasilkan error, bukan panik.
7. **mpp tidak pernah ditebak.**
8. **Usulan tidak pernah menimpa label, urutan, atau orientasi irisan**
   tanpa "Terima" dari pengguna.
9. **Deterministik**: tanpa acak tak berbenih; hasil paralel digabung dalam
   urutan tetap.
10. **Tulis atomik** (`.tmp` lalu `rename`) untuk semua berkas proyek.
11. **Tanpa jaringan.** Tidak ada crate HTTP di pohon dependensi `h3d-app`.
12. **Komentar dan pesan pengguna berbahasa Indonesia; identifier Inggris.**
13. **Ikon dari `egui_icons::icons::ICON_*`**, bukan glyph Unicode/emoji;
    jalankan `cargo test -p h3d-ui --test ui_glyph_coverage` setelah menambah
    teks UI.
14. **Dependensi baru** harus lolos `cargo deny` dan
    `cargo check --target aarch64-apple-ios`.
15. **Angka anggaran** memori/thread hanya di struct `Budgets` (`h3d-core`).
16. **Semua laporan dan layar Tentang** memuat "Hanya untuk riset, bukan
    untuk diagnosis".
17. **Data klinis hanya di `clinical.json`**, pseudonim, tanggal bulan-tahun;
    tidak pernah masuk `project.json`, log, atau ekspor tanpa
    `--with-clinical` (dijaga tes properti).
18. **Setiap jarak/ukuran 3D dilaporkan dengan ketidakpastian**; tidak ada
    teks rekomendasi tindakan di kode, laporan, atau UI. `ClinicalMeta`
    tidak pernah menjadi masukan perhitungan.
19. **Analisis spasial dihitung pada masker voxel**, bukan mesh; parameter
    atribut tersimpan sebagai hash dan tercetak di laporan.

## Peta crate

Lihat histo3d.md 7.2. Panah ketergantungan:
core → slide, imgproc → register, segment, volume → io → engine → cli;
render, i18n, ui, app terpisah dan hanya `h3d-app` yang menyatukan semuanya.
