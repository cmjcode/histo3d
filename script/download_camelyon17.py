#!/usr/bin/env python3
"""
Script untuk mengunduh dataset CAMELYON17 dari mirror resmi AWS S3
(https://camelyon17.grand-challenge.org/Data/) serta membaca patch WSI
menggunakan multiresolutionimageinterface (ASAP).

Fitur:
- Tanpa perlu akun AWS / kredensial (mengakses bucket publik s3://camelyon-dataset/CAMELYON17/).
- Mendukung unduhan seluruh node dari 1 pasien (--patient) ke dalam folder `images/<patient_id>/`.
- Mendukung filter khusus slide yang positif kanker (--cancer-only: macro, micro, itc) berdasarkan stages.csv.
- Mendukung unduhan anotasi XML tumor & mask jaringan (--include-extras) ke folder pasien yang sama.
- Mendukung resume download (HTTP Range) untuk file .tif berukuran besar (1-4 GB per file).
"""

import argparse
import csv
import io
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

S3_BASE_URL = "https://camelyon-dataset.s3.us-west-2.amazonaws.com"
S3_PREFIX = "CAMELYON17"


def normalize_patient_id(patient_arg: str) -> str:
    """
    Menormalisasi input ID pasien menjadi format 'patient_XXX'.
    Contoh: '4' -> 'patient_004', '012' -> 'patient_012', 'patient_004' -> 'patient_004'.
    """
    cleaned = patient_arg.strip().lower()
    match = re.search(r"(\d+)", cleaned)
    if not match:
        raise ValueError(f"Format patient_id tidak valid: '{patient_arg}'. Gunakan contoh: 4, 004, atau patient_004")
    num = int(match.group(1))
    if not (0 <= num <= 199):
        raise ValueError(f"Nomor pasien harus antara 000 s/d 199 (diterima: {num})")
    return f"patient_{num:03d}"


def get_centre_for_patient(filename: str) -> str:
    """
    Menentukan folder centre_0 .. centre_4 berdasarkan nomor pasien CAMELYON17.
    Training: patient_000..099 (20 pasien per centre)
    Testing : patient_100..199 (20 pasien per centre)
    """
    match = re.search(r"patient_(\d{3})", filename)
    if not match:
        return "other"
    patient_id = int(match.group(1))
    centre_id = (patient_id % 100) // 20
    return f"centre_{centre_id}"


def fetch_stages_metadata() -> dict[str, dict[str, str]]:
    """
    Mengambil dan mem-parsing stages.csv langsung dari S3 CAMELYON17.
    Mengembalikan dict:
      {
        "patient_004": {
          "pN_stage": "pN0(i+)",
          "center": "0",
          "nodes": {
            "patient_004_node_0.tif": "negative",
            ...
            "patient_004_node_4.tif": "itc",
          }
        },
        ...
      }
    """
    url = f"{S3_BASE_URL}/{S3_PREFIX}/stages.csv"
    req = urllib.request.Request(url, headers={"User-Agent": "HISTO3D-Downloader/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        csv_text = resp.read().decode("utf-8")

    patients: dict[str, dict] = {}
    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        item = row["patient"].strip()
        stage = row["stage"].strip()
        center = row["center"].strip()

        match = re.match(r"(patient_\d{3})", item)
        if not match:
            continue
        pid = match.group(1)
        if pid not in patients:
            patients[pid] = {"pN_stage": "unknown", "center": center, "nodes": {}}

        if item.endswith(".zip"):
            patients[pid]["pN_stage"] = stage
        elif item.endswith(".tif"):
            patients[pid]["nodes"][item] = stage

    return patients


def list_s3_files(prefix: str = f"{S3_PREFIX}/images/") -> list[tuple[str, int]]:
    """Mengambil daftar file beserta ukurannya dari bucket S3 publik CAMELYON17."""
    files: list[tuple[str, int]] = []
    continuation_token = None
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

    while True:
        params = {"list-type": "2", "prefix": prefix}
        if continuation_token:
            params["continuation-token"] = continuation_token
        url = f"{S3_BASE_URL}/?{urllib.parse.urlencode(params)}"

        req = urllib.request.Request(url, headers={"User-Agent": "HISTO3D-Downloader/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            xml_data = resp.read()

        root = ET.fromstring(xml_data)
        for content in root.findall("s3:Contents", ns):
            key_elem = content.find("s3:Key", ns)
            size_elem = content.find("s3:Size", ns)
            if key_elem is not None and size_elem is not None:
                key = key_elem.text or ""
                size = int(size_elem.text or 0)
                if size > 0:
                    files.append((key, size))

        is_truncated = root.findtext("s3:IsTruncated", default="false", namespaces=ns)
        if is_truncated.lower() == "true":
            continuation_token = root.findtext("s3:NextContinuationToken", namespaces=ns)
        else:
            break

    return files


def format_size(num_bytes: float) -> str:
    """Format ukuran byte ke satuan yang mudah dibaca."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def download_file(s3_key: str, dest_path: Path, chunk_size: int = 8 * 1024 * 1024) -> Path:
    """
    Mengunduh file dari S3 dengan dukungan resume (HTTP Range) dan progress bar.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{S3_BASE_URL}/{s3_key}"

    head_req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "HISTO3D-Downloader/1.0"})
    try:
        with urllib.request.urlopen(head_req, timeout=30) as resp:
            total_size = int(resp.headers.get("Content-Length", 0))
    except urllib.error.HTTPError as e:
        raise FileNotFoundError(f"File tidak ditemukan di server: {url} (HTTP {e.code})") from e

    existing_size = dest_path.stat().st_size if dest_path.exists() else 0
    if total_size > 0 and existing_size == total_size:
        print(f"[SKIP] File sudah lengkap: {dest_path} ({format_size(total_size)})")
        return dest_path

    headers = {"User-Agent": "HISTO3D-Downloader/1.0"}
    mode = "wb"
    if 0 < existing_size < total_size:
        headers["Range"] = f"bytes={existing_size}-"
        mode = "ab"
        print(f"[RESUME] Melanjutkan unduhan dari {format_size(existing_size)} / {format_size(total_size)}")
    else:
        existing_size = 0
        print(f"[DOWNLOAD] {url} -> {dest_path} ({format_size(total_size)})")

    req = urllib.request.Request(url, headers=headers)
    downloaded = existing_size

    with urllib.request.urlopen(req, timeout=60) as resp, open(dest_path, mode) as f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = (downloaded / total_size) * 100
                sys.stdout.write(
                    f"\r  Progress: {pct:6.2f}% ({format_size(downloaded)} / {format_size(total_size)})"
                )
                sys.stdout.flush()

    print("\n[DONE] Tersimpan di:", dest_path)
    return dest_path


def download_patient_images(
    patient_arg: str,
    base_images_dir: Path = Path("images"),
    cancer_only: bool = False,
    include_extras: bool = False,
) -> list[Path]:
    """
    Mengunduh seluruh slide jaringan (node_0 .. node_4) dari satu pasien
    dan menyimpannya di dalam folder `images/<patient_id>/`.

    Jika `cancer_only=True`, hanya mengunduh slide yang berlabel kanker/metastasis
    ('macro', 'micro', atau 'itc') berdasarkan stages.csv.
    """
    patient_id = normalize_patient_id(patient_arg)
    patient_dir = base_images_dir / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)

    print(f"Mengecek informasi medis untuk {patient_id} dari stages.csv...")
    stages_info = fetch_stages_metadata().get(patient_id)

    if stages_info:
        print(
            f"Pasien: {patient_id} | Center: {stages_info['center']} | "
            f"Stadium pN: {stages_info['pN_stage']}"
        )
        node_stages = stages_info["nodes"]
    else:
        print(
            f"[INFO] {patient_id} berada di test set (patient_100..199), "
            "label per-node tidak dipublikasikan di stages.csv."
        )
        node_stages = {f"{patient_id}_node_{i}.tif": "unknown" for i in range(5)}

    downloaded_paths: list[Path] = []
    for node_idx in range(5):
        fname = f"{patient_id}_node_{node_idx}.tif"
        label = node_stages.get(fname, "unknown")

        if cancer_only and label in ("negative", "unknown"):
            print(f"  - [SKIP] {fname} (status: {label})")
            continue

        print(f"  - [TARGET] {fname} (status: {label})")
        s3_key = f"{S3_PREFIX}/images/{fname}"
        dest_path = patient_dir / fname
        downloaded_paths.append(download_file(s3_key, dest_path))

        if include_extras:
            stem = fname.replace(".tif", "")
            # Coba unduh anotasi XML dan mask jika tersedia di server
            for extra_key, extra_name in [
                (f"{S3_PREFIX}/annotations/{stem}.xml", f"{stem}.xml"),
                (f"{S3_PREFIX}/masks/{stem}_mask.tif", f"{stem}_mask.tif"),
            ]:
                try:
                    download_file(extra_key, patient_dir / extra_name)
                except FileNotFoundError:
                    pass

    if not downloaded_paths:
        print(
            f"\n[PERHATIAN] Tidak ada file yang diunduh untuk {patient_id}. "
            f"(Semua node berstatus 'negative' atau tidak memiliki label kanker). "
            f"Gunakan `--list-cancer` untuk melihat daftar pasien dengan sel kanker, "
            f"atau hilangkan `--cancer-only` untuk mengunduh kelima node pasien ini."
        )
    else:
        print(f"\n[SELESAI] {len(downloaded_paths)} file WSI tersimpan di folder: {patient_dir}/")

    return downloaded_paths


def read_patch_with_mir(
    tif_path: str | Path = "camelyon17/centre_0/patient_000_node_0.tif",
    x: int = 568,
    y: int = 732,
    width: int = 300,
    height: int = 200,
    level: int = 2,
):
    """
    Membaca image patch menggunakan multiresolutionimageinterface (ASAP)
    sesuai contoh dokumentasi CAMELYON17.
    """
    import multiresolutionimageinterface as mir

    reader = mir.MultiResolutionImageReader()
    mr_image = reader.open(str(tif_path))
    if mr_image is None:
        raise RuntimeError(f"Gagal membuka file WSI: {tif_path}")

    ds = mr_image.getLevelDownsample(level)
    image_patch = mr_image.getUCharPatch(int(x * ds), int(y * ds), width, height, level)
    print(
        f"[MIR] Berhasil membaca patch dari '{tif_path}' "
        f"(level={level}, downsample={ds}, shape={getattr(image_patch, 'shape', 'unknown')})"
    )
    return image_patch


def main():
    parser = argparse.ArgumentParser(
        description="Download dataset CAMELYON17 (https://camelyon17.grand-challenge.org/Data/)"
    )
    parser.add_argument(
        "--patient",
        type=str,
        default=None,
        help="Unduh keseluruhan jaringan dari 1 pasien ke folder images/<patient_id>/ (contoh: --patient patient_004 atau --patient 4)",
    )
    parser.add_argument(
        "--cancer-only",
        action="store_true",
        help="Jika digunakan bersama --patient, hanya unduh slide node yang positif kanker (macro, micro, itc)",
    )
    parser.add_argument(
        "--include-extras",
        action="store_true",
        help="Jika digunakan bersama --patient, ikut mengunduh file anotasi XML tumor & mask .tif bila tersedia",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("images"),
        help="Folder induk untuk mode --patient (default: ./images sehingga tersimpan di images/<patient_id>/)",
    )
    parser.add_argument(
        "--list-cancer",
        action="store_true",
        help="Tampilkan daftar pasien yang memiliki slide positif kanker (macro/micro/itc) dari stages.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("camelyon17"),
        help="Folder tujuan untuk mode --file / --centre (default: ./camelyon17)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default="patient_000_node_0.tif",
        help="Nama file spesifik yang ingin diunduh jika tanpa --patient (default: patient_000_node_0.tif)",
    )
    parser.add_argument(
        "--centre",
        type=int,
        choices=[0, 1, 2, 3, 4],
        default=None,
        help="Unduh seluruh file dari centre tertentu (0-4)",
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Unduh juga file metadata (README.md, stages.csv, checksums.md5, license.txt)",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Simpan langsung di <output-dir>/<filename> tanpa subfolder centre_X",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Tampilkan daftar file yang tersedia di server lalu keluar",
    )
    parser.add_argument(
        "--read-patch",
        action="store_true",
        help="Setelah selesai mengunduh, jalankan pembacaan patch dengan multiresolutionimageinterface",
    )

    args = parser.parse_args()

    if args.list_cancer:
        print("Mengambil daftar pasien dengan slide positif kanker dari stages.csv...")
        patients = fetch_stages_metadata()
        for pid, info in sorted(patients.items()):
            pos_nodes = [
                f"{fname.split('_')[-1].replace('.tif', '')}:{st}"
                for fname, st in info["nodes"].items()
                if st != "negative"
            ]
            if pos_nodes:
                print(
                    f"  {pid} (center_{info['center']}, {info['pN_stage']:<8}) -> "
                    + ", ".join(pos_nodes)
                )
        return

    if args.list:
        print("Mengambil daftar file dari S3 CAMELYON17...")
        files = list_s3_files(f"{S3_PREFIX}/images/")
        for key, size in files[:50]:
            fname = key.split("/")[-1]
            centre = get_centre_for_patient(fname)
            print(f"  {centre}/{fname:<25} ({format_size(size)})")
        if len(files) > 50:
            print(f"  ... dan {len(files) - 50} file lainnya (total {len(files)} WSI).")
        return

    if args.metadata:
        target_meta_dir = args.images_dir if args.patient is not None else args.output_dir
        for meta_file in ["README.md", "stages.csv", "checksums.md5", "license.txt"]:
            download_file(f"{S3_PREFIX}/{meta_file}", target_meta_dir / meta_file)

    # Mode unduh per-pasien ke dalam folder images/<patient_id>/
    if args.patient is not None:
        downloaded = download_patient_images(
            patient_arg=args.patient,
            base_images_dir=args.images_dir,
            cancer_only=args.cancer_only,
            include_extras=args.include_extras,
        )
        if args.read_patch and downloaded:
            read_patch_with_mir(downloaded[0])
        return

    if args.centre is not None:
        target_centre = f"centre_{args.centre}"
        print(f"Mencari semua file untuk {target_centre}...")
        files = list_s3_files(f"{S3_PREFIX}/images/")
        for key, _ in files:
            fname = key.split("/")[-1]
            if get_centre_for_patient(fname) == target_centre:
                dest = (
                    args.output_dir / fname
                    if args.flat
                    else args.output_dir / target_centre / fname
                )
                download_file(key, dest)
        return

    # Default: unduh satu file spesifik (misal patient_000_node_0.tif)
    fname = os.path.basename(args.file)
    centre = get_centre_for_patient(fname)
    dest_path = (
        args.output_dir / fname
        if args.flat
        else args.output_dir / centre / fname
    )
    s3_key = f"{S3_PREFIX}/images/{fname}"
    downloaded_path = download_file(s3_key, dest_path)

    if args.read_patch:
        read_patch_with_mir(downloaded_path)


if __name__ == "__main__":
    main()
