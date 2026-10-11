# Data uji

## `golden/` — berkas emas CLI (histo3d.md 11.2)

Keluaran `h3d measure --json` dan `out/analysis.json` dari dua fantom berbenih,
dibuat lewat biner `h3d` dengan jam beku (`H3D_FIXED_TIME`):

| Berkas | Fantom |
|---|---|
| `ellipsoid_margin.measure.json`, `ellipsoid_margin.analysis.json` | `h3d phantom --shape ellipsoid --slices 20 --seed 5 --size 128 --margin 240` |
| `two_foci_vessel.measure.json`, `two_foci_vessel.analysis.json` | `h3d phantom --shape ellipsoid --slices 20 --seed 6 --size 128 --foci 2 --gap 120 --vessel` |
| `PLATFORM` | Platform pembuat berkas (`arsitektur-OS`). |

Pipeline per fantom: `new` → `import` → `register --strict` (opsi bawaan) →
`labels import` tiap irisan → `reconstruct --smooth 10` → `analyze` →
`measure --json`.

Tes `golden_files_match_cli_output_of_seeded_phantoms`
(`crates/h3d-cli/tests/cli.rs`) menjalankan ulang pipeline itu dan
membandingkan hasilnya **byte demi byte** dengan berkas di sini.

**Selisih berarti perilaku berubah** — registrasi, rekonstruksi, pengukuran,
analisis spasial, atau format JSON-nya. Itu bukan tes yang "rewel": cari tahu
dulu perubahan mana yang menyebabkannya. Bila perubahan itu memang disengaja,
segarkan lalu tinjau diff-nya sebelum di-commit:

```bash
H3D_UPDATE_GOLDEN=1 cargo test -p h3d-cli golden
git diff testdata/golden
```

Catatan:

- Berkas memuat `app_version`; menaikkan versi workspace mengubah berkas emas.
- `input_key` di `analysis.json` adalah hash masukan (label, transformasi,
  konfigurasi); ia berubah bila salah satunya berubah walau angka lain sama.
- Perbandingan byte hanya dilakukan di platform yang tercatat di `PLATFORM`
  (sama dengan CI: `aarch64-macos`). Di platform lain FFT/libm bisa berbeda
  pada bit terakhir, jadi tes memeriksa struktur yang sama dan bilangan dalam
  0,1 %. Segarkan berkas emas hanya dari platform CI.
