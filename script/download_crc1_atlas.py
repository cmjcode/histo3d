#!/usr/bin/env python3
"""
Mengunduh irisan serial tumor kolorektal manusia "CRC1" (spesimen WD-76845) dari
atlas 3D Lin dkk. (lab Sorger, Harvard Medical School):

  Lin J-R, dkk. "Multiplexed 3D atlas of state transitions and immune interaction
  in colorectal cancer." Cell (2023).
  Bucket publik : s3://lin-2021-crc-atlas (tanpa akun, HTTPS biasa)

Isi bucket untuk CRC1 (diperiksa 2026-11):
  - 22 irisan H&E   (WD-76845-001, -006, -013, ... -105), OME-TIFF RGB uint8,
    ~57.000 x 54.000 piksel, 0,325 µm/piksel, ~10-14 GB per irisan, ~253 GB total.
  - 25 irisan t-CyCIF (imunofluoresensi multipleks, puluhan kanal), ~75-95 GB per
    irisan, ~2 TB total.
  - metadata/WD-76845-NNN-metadata.csv : kanal & metode tiap irisan.
  - data/WD-76845-NNN-features.zip     : tabel fitur sel (hanya irisan CyCIF).
Nomor NNN = urutan irisan dalam blok (1..106). Jarak antar nomor TIDAK seragam.

Catatan untuk HISTO3D:
  - mpp dibaca dari header OME-XML (PhysicalSizeX) tiap berkas setelah diunduh,
    tidak ditebak (aturan 7). Hasilnya dicatat di dataset_info.json.
  - Tebal irisan / jarak z dalam µm TIDAK ada di bucket. Ambil dari makalah;
    dataset_info.json mencatat `section_thickness_um: null`.
  - Lisensi tidak tercantum di bucket. Cek pernyataan ketersediaan data di
    makalah sebelum memakai/mempublikasikan hasil turunan.

Fitur:
  - Resume sungguhan (HTTP Range) untuk berkas puluhan GB; tulis ke .part lalu rename.
  - Verifikasi ukuran + ETag S3 (MD5 multipart) setelah selesai.
  - Cek ruang disk sebelum mulai.

Contoh:
  python3 script/download_crc1_atlas.py --list
  python3 script/download_crc1_atlas.py --sections 1,6,13           # 3 irisan H&E (~37 GB)
  python3 script/download_crc1_atlas.py --sections 1-50             # semua H&E di 1..50
  python3 script/download_crc1_atlas.py --all                       # 22 irisan H&E (~253 GB)
  python3 script/download_crc1_atlas.py --sections 2 --modality cycif --features
"""

import argparse
import csv
import hashlib
import http.client
import io
import json
import os
import re
import shutil
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUCKET_URL = "https://lin-2021-crc-atlas.s3.amazonaws.com"
SPECIMEN = "WD-76845"
USER_AGENT = "HISTO3D-Downloader/1.0"
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "images" / "crc1_atlas"
CHUNK = 8 * 1024 * 1024

MODALITY_METHOD = {"he": "H&E", "cycif": "t-CyCIF"}

CITATION = (
    "Lin J-R, et al. Multiplexed 3D atlas of state transitions and immune interaction "
    "in colorectal cancer. Cell (2023). Data: s3://lin-2021-crc-atlas"
)


def open_url(url: str, headers: dict | None = None, timeout: int = 60):
    h = {"User-Agent": USER_AGENT}
    if headers:
        h.update(headers)
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout)


def list_bucket(prefix: str) -> dict[str, dict]:
    """Daftar objek S3 (key -> {size, etag}) dengan paginasi."""
    objects: dict[str, dict] = {}
    token = None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        with open_url(f"{BUCKET_URL}/?{urllib.parse.urlencode(params)}") as resp:
            root = ET.fromstring(resp.read())
        for c in root.findall("s3:Contents", S3_NS):
            key = c.findtext("s3:Key", default="", namespaces=S3_NS)
            size = int(c.findtext("s3:Size", default="0", namespaces=S3_NS))
            etag = c.findtext("s3:ETag", default="", namespaces=S3_NS).strip('"')
            if key and size > 0:
                objects[key] = {"size": size, "etag": etag}
        if root.findtext("s3:IsTruncated", default="false", namespaces=S3_NS).lower() == "true":
            token = root.findtext("s3:NextContinuationToken", namespaces=S3_NS)
        else:
            return objects


def fetch_method(section: int) -> str:
    """Metode irisan ('H&E' / 't-CyCIF') dari CSV metadata resmi; 'unknown' bila tak ada."""
    url = f"{BUCKET_URL}/metadata/{SPECIMEN}-{section:03d}-metadata.csv"
    try:
        with open_url(url, timeout=30) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError:
        return "unknown"
    methods = {row.get("Method", "").strip() for row in csv.DictReader(io.StringIO(text))}
    methods.discard("")
    return methods.pop() if len(methods) == 1 else "unknown"


def discover_sections() -> list[dict]:
    """Semua irisan CRC1 yang punya citra OME-TIFF, beserta metode dan ukurannya."""
    objects = list_bucket(f"data/{SPECIMEN}-")
    pattern = re.compile(rf"^data/{SPECIMEN}-(\d+)\.ome\.tif$")
    sections = []
    for key, obj in objects.items():
        m = pattern.match(key)
        if not m:
            continue
        n = int(m.group(1))
        feat_key = f"data/{SPECIMEN}-{n:03d}-features.zip"
        sections.append(
            {
                "n": n,
                "key": key,
                "size": obj["size"],
                "etag": obj["etag"],
                "features": {"key": feat_key, **objects[feat_key]} if feat_key in objects else None,
                "method": None,
            }
        )
    sections.sort(key=lambda s: s["n"])
    # 47 CSV kecil: ambil paralel; map() menjaga urutan hasil tetap.
    with ThreadPoolExecutor(max_workers=8) as pool:
        for s, method in zip(sections, pool.map(fetch_method, [s["n"] for s in sections])):
            s["method"] = method
    return sections


def parse_section_spec(spec: str) -> set[int]:
    """'1,6,13-20' -> {1, 6, 13, ..., 20}."""
    result: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            if lo > hi:
                raise ValueError(f"rentang terbalik: {part}")
            result.update(range(lo, hi + 1))
        else:
            result.add(int(part))
    return result


def format_size(num_bytes: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def s3_etag_matches(path: Path, etag: str) -> bool | None:
    """
    Cocokkan berkas lokal dengan ETag S3. ETag biasa = MD5; ETag multipart
    "<md5>-<N>" = MD5 dari gabungan MD5 tiap bagian. Ukuran bagian tidak
    disimpan S3, jadi dipakai ukuran umum terkecil yang menghasilkan tepat N
    bagian. Mengembalikan None bila tidak bisa diverifikasi.
    """
    size = path.stat().st_size
    if "-" not in etag:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(CHUNK), b""):
                h.update(chunk)
        return h.hexdigest() == etag

    digest, _, parts_str = etag.partition("-")
    try:
        parts = int(parts_str)
    except ValueError:
        return None
    mib = 1024 * 1024
    candidates = [p * mib for p in (5, 8, 16, 32, 64, 100, 128, 256, 512)]
    candidates = [p for p in candidates if -(-size // p) == parts]
    if not candidates:
        return None
    part_size = candidates[0]  # membaca ulang 10+ GB itu mahal; cukup satu percobaan
    md5s = []
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(part_size), b""):
            md5s.append(hashlib.md5(block).digest())
    return hashlib.md5(b"".join(md5s)).hexdigest() == digest


def download_resumable(key: str, size: int, dest: Path, retries: int = 5) -> str:
    """Unduh dengan resume HTTP Range ke `<dest>.part`, lalu rename atomik."""
    if dest.is_file() and dest.stat().st_size == size:
        return "lewati"

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    url = f"{BUCKET_URL}/{urllib.parse.quote(key)}"

    # Batas percobaan hanya menghitung putus koneksi TANPA progres; selama byte
    # bertambah, unduhan terus dilanjutkan (penting untuk berkas puluhan GB).
    stalled = 0
    while True:
        have = tmp.stat().st_size if tmp.exists() else 0
        if have > size:
            tmp.unlink()
            have = 0
        if have == size:
            break
        if stalled >= retries:
            raise RuntimeError(f"{key}: gagal setelah {retries} percobaan tanpa progres")
        start = have
        try:
            headers = {"Range": f"bytes={have}-"} if have else {}
            with open_url(url, headers=headers, timeout=120) as resp:
                if have and resp.status != 206:
                    have = 0  # server mengabaikan Range: mulai ulang
                mode = "ab" if have else "wb"
                with open(tmp, mode) as out:
                    t0, done0, last = time.time(), have, 0.0
                    for chunk in iter(lambda: resp.read(CHUNK), b""):
                        out.write(chunk)
                        have += len(chunk)
                        now = time.time()
                        if now - last >= 1.0:
                            speed = (have - done0) / max(now - t0, 1e-6)
                            eta = (size - have) / speed if speed > 0 else 0
                            print(
                                f"\r    {100 * have / size:5.1f}%  {format_size(have)} / {format_size(size)}"
                                f"  {format_size(speed)}/s  sisa ~{eta / 60:.0f} mnt   ",
                                end="",
                                flush=True,
                            )
                            last = now
            print()
        except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
            # HTTPException mencakup IncompleteRead (koneksi putus di tengah respons).
            print(f"\n    terputus: {e!r}; melanjutkan ...", file=sys.stderr)
        current = tmp.stat().st_size if tmp.exists() else 0
        if current > start:
            stalled = 0
        else:
            stalled += 1
            time.sleep(min(30, 3 * stalled))

    if tmp.stat().st_size != size:
        raise RuntimeError(f"{key}: ukuran {tmp.stat().st_size} != {size}")
    os.replace(tmp, dest)
    return "unduh"


def read_ome_info(path: Path) -> dict:
    """
    Membaca OME-XML dari ImageDescription IFD pertama (BigTIFF/TIFF klasik).
    Mengembalikan mpp (µm/piksel) dan ukuran piksel; nilai None bila tak ada.
    Berkas rusak menghasilkan dict berisi 'error', bukan panik.
    """
    info: dict = {"mpp_um": None, "size_px": None}
    try:
        with open(path, "rb") as f:
            head = f.read(16)
            bo = {b"II": "<", b"MM": ">"}.get(head[:2])
            if bo is None:
                return {**info, "error": "bukan TIFF"}
            version = struct.unpack(bo + "H", head[2:4])[0]
            if version == 43:
                f.seek(struct.unpack(bo + "Q", head[8:16])[0])
                (count,) = struct.unpack(bo + "Q", f.read(8))
                entry_fmt, entry_len = bo + "HHQQ", 20
            elif version == 42:
                f.seek(struct.unpack(bo + "I", head[4:8])[0])
                (count,) = struct.unpack(bo + "H", f.read(2))
                entry_fmt, entry_len = bo + "HHII", 12
            else:
                return {**info, "error": f"versi TIFF {version}"}
            if count > 10000:
                return {**info, "error": "IFD tidak wajar"}
            entries = f.read(count * entry_len)
            for i in range(count):
                tag, _typ, cnt, val = struct.unpack(entry_fmt, entries[i * entry_len : (i + 1) * entry_len])
                if tag == 270 and cnt < 50_000_000:
                    f.seek(val)
                    xml = f.read(cnt).rstrip(b"\0").decode("utf-8", errors="replace")
                    pixels = ET.fromstring(xml).find(".//{*}Pixels")
                    if pixels is None:
                        return {**info, "error": "OME Pixels tidak ada"}
                    unit = pixels.get("PhysicalSizeXUnit", "µm")
                    px = pixels.get("PhysicalSizeX")
                    if px is not None and unit in ("µm", "um"):
                        info["mpp_um"] = float(px)
                    sx, sy = pixels.get("SizeX"), pixels.get("SizeY")
                    if sx and sy:
                        info["size_px"] = [int(sx), int(sy)]
                    info["channels"] = int(pixels.get("SizeC", "0") or 0)
                    return info
    except (OSError, struct.error, ET.ParseError, ValueError) as e:
        return {**info, "error": str(e)}
    return {**info, "error": "ImageDescription tidak ada"}


def write_dataset_info(out_dir: Path, sections: list[dict]) -> None:
    """Catat irisan yang sudah lengkap di disk ke dataset_info.json (tulis atomik)."""
    target = out_dir / "dataset_info.json"
    info = {
        "dataset": "CRC1 (WD-76845), Multiplexed 3D atlas of colorectal cancer",
        "citation": CITATION,
        "source": f"{BUCKET_URL}/data/",
        "license": "belum diverifikasi - cek makalah",
        # Tidak ada di bucket; ambil dari makalah, jangan ditebak.
        "section_thickness_um": None,
        "section_order": "nomor NNN pada WD-76845-NNN (1..106), jarak antar nomor tidak seragam",
        "sections": {},
    }
    for s in sections:
        path = out_dir / f"{s['n']:03d}" / Path(s["key"]).name
        if not path.is_file() or path.stat().st_size != s["size"]:
            continue
        info["sections"][f"{s['n']:03d}"] = {
            "file": str(path.relative_to(out_dir)),
            "method": s["method"],
            **read_ome_info(path),
        }
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, target)


def print_listing(sections: list[dict]) -> None:
    print(f"Spesimen {SPECIMEN} (CRC1): {len(sections)} irisan dengan citra\n")
    print("  No.  Metode     Ukuran      Fitur")
    for s in sections:
        feat = format_size(s["features"]["size"]) if s["features"] else "-"
        print(f"  {s['n']:3d}  {s['method']:9s}  {format_size(s['size']):>10s}  {feat}")
    for label in ("H&E", "t-CyCIF"):
        sel = [s for s in sections if s["method"] == label]
        print(f"\nTotal {label}: {len(sel)} irisan, {format_size(sum(s['size'] for s in sel))}", end="")
        if sel:
            print(f"  (nomor: {','.join(str(s['n']) for s in sel)})", end="")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Unduh irisan serial CRC1 (atlas 3D kolorektal, Lin dkk.).")
    parser.add_argument("--list", action="store_true", help="Tampilkan semua irisan lalu keluar.")
    parser.add_argument("--sections", help="Nomor irisan, mis. '1,6,13' atau '1-50'.")
    parser.add_argument("--all", action="store_true", help="Semua irisan dengan modalitas terpilih.")
    parser.add_argument(
        "--modality",
        choices=["he", "cycif", "all"],
        default="he",
        help="he (default), cycif (sangat besar, ~80 GB/irisan), atau all.",
    )
    parser.add_argument("--features", action="store_true", help="Ikut unduh features.zip (irisan CyCIF).")
    parser.add_argument("--no-verify", action="store_true", help="Lewati verifikasi ETag (membaca ulang berkas).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"Folder tujuan (default: {DEFAULT_OUT})")
    parser.add_argument("--dry-run", action="store_true", help="Tampilkan rencana tanpa mengunduh.")
    args = parser.parse_args()

    wanted = None
    if args.sections:
        try:
            wanted = parse_section_spec(args.sections)
        except ValueError as e:
            print(f"--sections tidak valid: {e}", file=sys.stderr)
            return 2

    print("Mengambil daftar irisan dari s3://lin-2021-crc-atlas ...")
    try:
        sections = discover_sections()
    except (urllib.error.URLError, OSError, ET.ParseError) as e:
        print(f"Gagal mengambil daftar: {e}", file=sys.stderr)
        return 1
    if not sections:
        print("Tidak ada irisan ditemukan di bucket.", file=sys.stderr)
        return 1

    if args.list or (wanted is None and not args.all):
        print_listing(sections)
        if not args.list:
            print("\nPilih --sections <nomor> atau --all. Contoh: --sections 1,6,13")
        return 0

    methods = set(MODALITY_METHOD.values()) if args.modality == "all" else {MODALITY_METHOD[args.modality]}
    selected = [s for s in sections if s["method"] in methods and (wanted is None or s["n"] in wanted)]
    if wanted is not None:
        skipped = [s["n"] for s in sections if s["n"] in wanted and s["method"] not in methods]
        if skipped:
            print(f"Catatan: dilewati karena modalitas bukan '{args.modality}': {', '.join(map(str, skipped))}")
    if not selected:
        print("Tidak ada irisan yang cocok dengan pilihan (lihat --list).", file=sys.stderr)
        return 2

    # Tiap irisan: <out>/<NNN>/WD-76845-NNN.ome.tif (+ metadata.csv, features.zip)
    jobs = []
    for s in selected:
        folder = args.out / f"{s['n']:03d}"
        jobs.append((s["key"], s["size"], s["etag"], folder / Path(s["key"]).name))
        meta_key = f"metadata/{SPECIMEN}-{s['n']:03d}-metadata.csv"
        jobs.append((meta_key, None, None, folder / Path(meta_key).name))
        if args.features and s["features"]:
            f = s["features"]
            jobs.append((f["key"], f["size"], f["etag"], folder / Path(f["key"]).name))

    def remaining(job) -> int:
        _key, size, _etag, dest = job
        if size is None or (dest.is_file() and dest.stat().st_size == size):
            return 0
        part = dest.with_name(dest.name + ".part")
        return size - (part.stat().st_size if part.exists() else 0)

    total = sum(j[1] or 0 for j in jobs)
    need = sum(remaining(j) for j in jobs)
    # Cek ruang di folder induk terdekat yang sudah ada (dry-run tidak membuat folder).
    probe = args.out
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    print(f"Irisan   : {', '.join(str(s['n']) for s in selected)} ({len(selected)})")
    print(f"Total    : {format_size(total)}; masih perlu diunduh {format_size(need)}")
    print(f"Ruang    : {format_size(free)} kosong di {args.out}")
    if need > free:
        print("Ruang disk tidak cukup. Kurangi --sections atau pakai --out ke disk lain.", file=sys.stderr)
        return 1
    if args.dry_run:
        return 0

    failures = []
    for i, (key, size, etag, dest) in enumerate(jobs, start=1):
        print(f"[{i}/{len(jobs)}] {key}")
        try:
            if size is None:
                # CSV metadata kecil: unduh utuh, tulis atomik.
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open_url(f"{BUCKET_URL}/{urllib.parse.quote(key)}", timeout=30) as resp:
                    data = resp.read()
                tmp = dest.with_name(dest.name + ".tmp")
                tmp.write_bytes(data)
                os.replace(tmp, dest)
                continue
            status = download_resumable(key, size, dest)
            print(f"    {status}")
            if status == "unduh" and not args.no_verify and etag:
                print("    verifikasi ETag ...", flush=True)
                ok = s3_etag_matches(dest, etag)
                if ok is False:
                    bad = dest.with_name(dest.name + ".corrupt")
                    os.replace(dest, bad)
                    raise RuntimeError(f"ETag tidak cocok; berkas dipindah ke {bad.name}, jalankan ulang")
                print("    ETag cocok" if ok else "    ETag tidak bisa diverifikasi (ukuran cocok)")
        except (RuntimeError, urllib.error.URLError, OSError, http.client.HTTPException) as e:
            failures.append(f"{key}: {e}")
            print(f"    GAGAL: {e}", file=sys.stderr)

    write_dataset_info(args.out, sections)
    print(f"\nMetadata: {args.out / 'dataset_info.json'} (mpp dari header OME, tebal irisan = null)")
    print(f"Sitasi  : {CITATION}")
    if failures:
        print(f"\n{len(failures)} berkas gagal (jalankan ulang untuk melanjutkan):", file=sys.stderr)
        for msg in failures:
            print(f"  {msg}", file=sys.stderr)
        return 1
    print("Selesai.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
