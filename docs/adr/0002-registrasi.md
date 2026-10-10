# ADR 0002 — Backend registrasi bawaan

Status: diterima (2026-10-10).

## Konteks

histo3d.md K4 dan 8.4 mewajibkan registrasi di balik trait `Registrar`, dengan
`BuiltinRegistrar` (momen → korelasi fase → ECC) dan `sitk` sebagai backend
opsional yang hanya diaktifkan bila H0.3 membuktikannya setara SimpleITK.

## Temuan

- `sitk` di crates.io berada di versi 0.0.3 (Juli 2026), mengaku "pure-Rust
  port of SimpleITK". Tidak ada rilis stabil, tidak ada bukti kesetaraan, dan
  tidak ada data nyata untuk mengujinya (H0.1 belum ada).
- `BuiltinRegistrar` hanya butuh `rustfft`, `ndarray`, `rayon`; semuanya
  sudah lulus kompilasi iOS.

## Keputusan

1. **Bawaan: `BuiltinRegistrar`** (R1 momen, R2 korelasi fase, R3 ECC
   berjenjang 64→32→16 µm; MI Nelder–Mead untuk beda pewarnaan).
2. **`sitk` tidak ditambahkan** ke pohon dependensi. Fitur Cargo `sitk`
   dicadangkan namanya, tetapi kodenya tidak ditulis sampai ada data nyata
   dan versi `sitk` ≥ 0.1 yang bisa diuji terhadap SimpleITK.
3. Kriteria membuka kembali keputusan: TRE `BuiltinRegistrar` pada sampel
   nyata (H0.1) gagal mencapai ≤ 2 px pada resolusi registrasi, **dan**
   koreksi landmark manual tidak cukup dalam < 1 menit/irisan (H3).
4. Cadangan bila keduanya gagal: pra-proses desktop dengan elastix (di luar
   aplikasi), sesuai 10.H0.
