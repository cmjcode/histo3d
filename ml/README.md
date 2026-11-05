# `ml/` — pelatihan model segmentasi H&E (Rust murni)

Workspace Cargo **terpisah** dari workspace utama (histo3d.md Fase 2, versi
Rust). Alasan: `burn` + `wgpu` berat dikompilasi dan tidak boleh masuk pohon
dependensi `h3d-app` maupun target iOS. Aplikasi hanya memakai keluarannya:
`model.onnx` + `model.json` (kontrak `h3d_segment::ModelSpec`, histo3d.md 8.5.4).

## Alur

```bash
cd ml

# 1. Potong tile berlabel (kelas: 0 latar, 1 jaringan, 2 tumor) dari dataset sub4.
cargo run --release -p h3d-train -- tiles \
  --root ../images/tampere_mouse_prostate --cache cache \
  --mpp0 0.46 --mpp-source "Liimatainen dkk., BMC Cancer 2021, doi:10.1186/s12885-021-08542-9"

# 2. Latih U-Net (GPU Metal lewat wgpu bila ada; --device cpu untuk CPU murni).
cargo run --release -p h3d-train -- train \
  --cache cache --out runs/v1 --holdout PTENX036-011 --steps 3000

# 3. Ekspor ONNX (BatchNorm dilipat) + model.json; tract memverifikasi sebelum ditulis.
cargo run --release -p h3d-train -- export --run runs/v1 --out ../models/tumor-he-v1

# 4. Evaluasi lewat jalur inferensi aplikasi (h3d_segment::infer_slide) pada holdout.
cargo run --release -p h3d-train -- eval --model ../models/tumor-he-v1 \
  --root ../images/tampere_mouse_prostate --stack PTENX036-011 --mpp 1.84 --json
```

Dari aplikasi: `h3d segment <proyek.h3d> --model models/tumor-he-v1`.

## Tes

```bash
cargo test -p h3d-train          # termasuk tests/parity.rs: burn ↔ tract, loop latih sintetis
cargo deny check                 # ml/deny.toml
```

## mpp (aturan keras 7: tidak ditebak)

Dataset Tampere tidak mencantumkan µm/piksel. Nilai **0,46 µm/piksel** pada
resolusi penuh berasal dari makalah grup yang sama atas data Pten+/- yang
sama (irisan 5 µm, tiap ke-10 diwarnai H&E, jarak 50 µm):
Liimatainen K. dkk., "Virtual reality for 3D histology", BMC Cancer 2021,
doi:10.1186/s12885-021-08542-9, bagian Methods: "The pixel size of the
acquired WSIs was 0.46 μm". Folder `*_sub4` = 4× lebih kecil → 1,84 µm/piksel.
Nilai ini selalu diteruskan eksplisit lewat `--mpp0` dan dicatat beserta
sumbernya di `index.json` dan `train_meta.json`.

## Atribusi data (CC BY 4.0)

Liimatainen K., Latonen L., Ruusuvuori P. (2023). *3D histology data of mouse
prostates with manually annotated tumor hotspots.* Tampere University.
https://doi.org/10.23729/5c5e62ad-7831-439e-910c-9cbf74f526ec

## Catatan rancangan

- Model: U-Net 4 tingkat, 32–512 kanal, Conv3×3+BN+ReLU, ConvTranspose 2×2,
  hanya operator yang didukung tract 0.23. `input_size` ekspor 512, latih
  pada potongan 256 (model konvolusional penuh).
- Loss: entropi silang + (1 − Dice lunak kelas non-latar). Batch separuh
  tile tumor, separuh lainnya. Augmentasi: dihedral, jitter pewarna HED
  (dekonvolusi `h3d_imgproc::color`), blur ringan; semua berbenih.
- Validasi: leave-one-stack-out (`--holdout`), Dice global per kelas.
- Anotasi dataset adalah *hotspot* tumor (sekitar 0,2 % piksel), bukan
  seluruh tumor; tumor di luar hotspot ikut berlabel "jaringan". Batch
  diseimbangkan (`--tumor-min-frac`) dan hasil harus dibaca sebagai usulan.
- Hanya untuk riset, bukan untuk diagnosis.
