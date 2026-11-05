#!/usr/bin/env python3
"""
Mengunduh dataset "3D histology data of mouse prostates with manually annotated
tumor hotspots" (Liimatainen, Latonen, Ruusuvuori; Tampere University, 2023).

  DOI     : https://doi.org/10.23729/5c5e62ad-7831-439e-910c-9cbf74f526ec
  Etsin   : https://etsin.fairdata.fi/dataset/9d4b6847-b3e7-41d5-bcd1-da077b201c67
  Lisensi : CC BY 4.0 (wajib mencantumkan atribusi bila dipakai/dipublikasikan)

Isi dataset: 6 tumpukan irisan serial H&E prostat tikus Pten+/- (25-32 irisan per
tumpukan, 171 irisan total). Irisan setebal 5 µm, tiap irisan ke-10 diwarnai H&E,
sehingga jarak antar irisan yang tersedia 50 µm. Tumpukan SUDAH diregistrasi oleh
pembuatnya (plugin Elastic Stack Alignment, Fiji) — cocok sebagai pembanding.

Struktur di server (per tumpukan, mis. PTENX036-011):
  Images_FullRes/   PNG H&E resolusi penuh (10x)           ~10,1 GB total
  Images_sub4/      PNG H&E diperkecil 4x                   ~0,85 GB total
  Masks_sub4/       masker jaringan (ukuran = Images_sub4)  kecil
  TumorMasks_sub4/  masker tumor anotasi manual (16-bit)    kecil
  Images_ihc_sub2/  IHC sintetis (Cycle-GAN)                ~3,6 GB total
  Images_pas_sub2/  PAS sintetis (Cycle-GAN)                ~3,7 GB total
  Images_orto_sub2/ translasi eksperimental "ortho"         ~6,0 GB total
  nucNumber_100/, nucNumber_400/  peta jumlah inti          kecil
Nama berkas im0.png .. imN.png = urutan irisan.

PENTING (aturan 7 histo3d: mpp tidak pernah ditebak): metadata dataset tidak
menyebut ukuran piksel (µm/piksel). Skrip ini menulis `mpp_um: null` di
dataset_info.json; isi nilainya hanya dari sumber resmi (makalah/penulis).

Unduhan lewat Fairdata Download Service (tanpa akun). Server tidak mendukung
HTTP Range, jadi "resume" bekerja per berkas: berkas yang sudah ada dan lolos
verifikasi SHA-256 dilewati; berkas setengah jadi (.part) diunduh ulang.

Contoh:
  python3 script/download_tampere_prostate3d.py --list
  python3 script/download_tampere_prostate3d.py --stack PTENX036-32
  python3 script/download_tampere_prostate3d.py --all --types Images_sub4,TumorMasks_sub4
  python3 script/download_tampere_prostate3d.py --stack PTENX036-011 --types all
"""

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DATASET_ID = "9d4b6847-b3e7-41d5-bcd1-da077b201c67"
DATASET_DOI = "10.23729/5c5e62ad-7831-439e-910c-9cbf74f526ec"
METAX_FILES_URL = (
    f"https://metax.fairdata.fi/rest/v2/datasets/{DATASET_ID}/files"
    "?file_fields=file_path,byte_size,checksum"
)
AUTHORIZE_URL = "https://etsin.fairdata.fi/api/download/authorize"
USER_AGENT = "HISTO3D-Downloader/1.0"

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "images" / "tampere_mouse_prostate"
# Default: cukup untuk rekonstruksi 3D + acuan tumor, ~0,9 GB untuk semua tumpukan.
DEFAULT_TYPES = ["Images_sub4", "Masks_sub4", "TumorMasks_sub4"]

CITATION = (
    "Liimatainen K., Latonen L., Ruusuvuori P. (2023). 3D histology data of mouse "
    "prostates with manually annotated tumor hotspots. Tampere University. "
    f"https://doi.org/{DATASET_DOI}"
)


def http_request(url: str, data: bytes | None = None, timeout: int = 60):
    headers = {"User-Agent": USER_AGENT}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    return urllib.request.urlopen(req, timeout=timeout)


def fetch_file_list() -> list[dict]:
    """Mengambil daftar berkas (path, ukuran, SHA-256) dari API Metax Fairdata."""
    with http_request(METAX_FILES_URL) as resp:
        files = json.load(resp)
    if not isinstance(files, list):
        raise RuntimeError("Format respons Metax tidak dikenali (bukan list).")

    result = []
    for item in files:
        path = item.get("file_path", "")
        parts = path.strip("/").split("/")
        # Bentuk yang diharapkan: Histology3D_VR/<stack>/<tipe>/<nama>.png
        if len(parts) != 4 or any(p in ("", ".", "..") for p in parts):
            print(f"  [peringatan] path tak terduga dilewati: {path}", file=sys.stderr)
            continue
        checksum = item.get("checksum") or {}
        sha256 = checksum.get("value") if checksum.get("algorithm") == "SHA-256" else None
        result.append(
            {
                "path": path,
                "stack": parts[1],
                "type": parts[2],
                "name": parts[3],
                "size": int(item.get("byte_size", 0)),
                "sha256": sha256,
            }
        )
    return result


def section_index(name: str) -> int:
    """'im12.png' -> 12. Urutan irisan mengikuti angka, bukan urutan abjad."""
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else -1


def format_size(num_bytes: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def format_eta(seconds: float) -> str:
    if seconds <= 0 or seconds != seconds or seconds == float("inf"):
        return "--:--"
    sec = int(seconds + 0.5)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def make_bar(done: int, total: int, width: int = 24) -> str:
    ratio = min(max(done / total, 0.0), 1.0) if total > 0 else 1.0
    filled = int(round(ratio * width))
    return "█" * filled + "░" * (width - filled)


class DownloadProgress:
    """Progress bar thread-safe untuk unduhan paralel."""

    def __init__(self, total_files: int, total_bytes: int) -> None:
        self.total_files = total_files
        self.total_bytes = total_bytes
        self.done_files = 0
        self.done_bytes = 0
        self.net_bytes = 0
        self.t0 = time.monotonic()
        self.last_draw = 0.0
        self.lock = threading.Lock()
        self.is_tty = sys.stdout.isatty()

    def _bar_line(self) -> str:
        elapsed = max(time.monotonic() - self.t0, 1e-6)
        speed = max(self.net_bytes, 0) / elapsed
        remaining = max(self.total_bytes - self.done_bytes, 0)
        eta = remaining / speed if speed > 0 else float("inf")
        pct = 100.0 * self.done_bytes / self.total_bytes if self.total_bytes else 100.0
        bar = make_bar(self.done_bytes, self.total_bytes)
        return (
            f"  [{bar}] {pct:5.1f}%  {format_size(self.done_bytes)}/{format_size(self.total_bytes)}"
            f"  {format_size(speed)}/s  ETA {format_eta(eta)}"
            f"  ({self.done_files}/{self.total_files})"
        )

    def add_chunk(self, delta: int, counted_for_speed: bool = True) -> None:
        with self.lock:
            self.done_bytes = max(0, self.done_bytes + delta)
            if counted_for_speed:
                self.net_bytes = max(0, self.net_bytes + delta)
            now = time.monotonic()
            if self.is_tty and (now - self.last_draw >= 0.08):
                sys.stdout.write("\r\033[K" + self._bar_line())
                sys.stdout.flush()
                self.last_draw = now

    def file_finished(self, entry: dict, status: str) -> None:
        with self.lock:
            self.done_files += 1
            pct = 100.0 * self.done_bytes / self.total_bytes if self.total_bytes else 100.0
            msg = (
                f"  [{self.done_files:4d}/{self.total_files}] {pct:5.1f}% {status:6s} "
                f"{entry['stack']}/{entry['type']}/{entry['name']} ({format_size(entry['size'])})"
            )
            if self.is_tty:
                sys.stdout.write("\r\033[K" + msg + "\n" + self._bar_line())
                sys.stdout.flush()
                self.last_draw = time.monotonic()
            else:
                print(msg, flush=True)

    def finish(self) -> None:
        with self.lock:
            if self.is_tty:
                sys.stdout.write("\r\033[K" + self._bar_line() + "\n")
                sys.stdout.flush()
            else:
                print(self._bar_line(), flush=True)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_valid(path: Path, entry: dict) -> bool:
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    return entry["sha256"] is None or sha256_of(path) == entry["sha256"]


def authorize(file_path: str) -> str:
    """Meminta URL unduhan bertoken (berlaku sementara) untuk satu berkas."""
    body = json.dumps({"cr_id": DATASET_ID, "file": file_path}).encode("utf-8")
    with http_request(AUTHORIZE_URL, data=body, timeout=30) as resp:
        payload = json.load(resp)
    url = payload.get("url") if isinstance(payload, dict) else None
    if not url:
        raise RuntimeError(f"Otorisasi gagal untuk {file_path}: {payload}")
    return url


def download_one(
    entry: dict,
    dest: Path,
    retries: int = 3,
    on_chunk: Callable[[int, bool], None] | None = None,
) -> str:
    """Unduh satu berkas secara atomik (.part lalu rename) dan verifikasi SHA-256."""
    if is_valid(dest, entry):
        if on_chunk:
            on_chunk(entry["size"], False)
        return "lewati"

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        attempt_bytes = 0
        try:
            url = authorize(entry["path"])
            h = hashlib.sha256()
            with http_request(url, timeout=120) as resp, open(tmp, "wb") as out:
                for chunk in iter(lambda: resp.read(256 * 1024), b""):
                    out.write(chunk)
                    h.update(chunk)
                    attempt_bytes += len(chunk)
                    if on_chunk:
                        on_chunk(len(chunk), True)
            size = tmp.stat().st_size
            if size != entry["size"]:
                raise RuntimeError(f"ukuran {size} != {entry['size']}")
            if entry["sha256"] and h.hexdigest() != entry["sha256"]:
                raise RuntimeError("SHA-256 tidak cocok")
            os.replace(tmp, dest)
            return "unduh"
        except (urllib.error.URLError, OSError, RuntimeError, ValueError) as e:
            last_error = e
            if attempt_bytes and on_chunk:
                on_chunk(-attempt_bytes, True)
            if tmp.exists():
                tmp.unlink()
            if attempt < retries:
                time.sleep(2 * attempt)

    raise RuntimeError(f"{entry['path']}: gagal setelah {retries} percobaan ({last_error})")


def print_listing(files: list[dict]) -> None:
    stacks = sorted({f["stack"] for f in files})
    types = sorted({f["type"] for f in files})

    print(f"Dataset: {CITATION}\n")
    print("Per tumpukan (jumlah irisan = jumlah berkas Images_sub4):")
    for s in stacks:
        n = sum(1 for f in files if f["stack"] == s and f["type"] == "Images_sub4")
        default_size = sum(f["size"] for f in files if f["stack"] == s and f["type"] in DEFAULT_TYPES)
        all_size = sum(f["size"] for f in files if f["stack"] == s)
        print(
            f"  {s:14s} {n:3d} irisan | default: {format_size(default_size):>10s}"
            f" | semua tipe: {format_size(all_size):>10s}"
        )

    print("\nPer tipe (semua tumpukan):")
    for t in types:
        sel = [f for f in files if f["type"] == t]
        mark = " (default)" if t in DEFAULT_TYPES else ""
        print(f"  {t:18s} {len(sel):4d} berkas {format_size(sum(f['size'] for f in sel)):>10s}{mark}")


def write_dataset_info(out_dir: Path, stacks: list[str], files: list[dict]) -> None:
    """Catatan metadata untuk impor ke HISTO3D. Ditulis atomik."""
    info = {
        "dataset": "3D histology data of mouse prostates with manually annotated tumor hotspots",
        "doi": f"https://doi.org/{DATASET_DOI}",
        "etsin": f"https://etsin.fairdata.fi/dataset/{DATASET_ID}",
        "license": "CC BY 4.0",
        "citation": CITATION,
        "species": "mouse (Pten+/-, 11 bulan)",
        "stain": "H&E",
        "section_thickness_um": 5.0,
        "section_spacing_um": 50.0,
        "objective": "10x",
        # Tidak disebut di metadata resmi; JANGAN ditebak (aturan 7).
        "mpp_um": None,
        "pre_registered": "Elastic Stack Alignment (Fiji) oleh pembuat dataset",
        "section_order": "im0.png, im1.png, ... (urut angka)",
        "stacks": {
            s: sum(1 for f in files if f["stack"] == s and f["type"] == "Images_sub4") for s in stacks
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "dataset_info.json"
    # Pertahankan nilai mpp (dan sumbernya) yang sudah diisi tangan dari sumber
    # resmi pada unduhan sebelumnya; skrip ini sendiri tidak pernah menebaknya.
    if target.exists():
        try:
            old = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            old = {}
        for key in ("mpp_um", "mpp_um_source", "sub4_mpp_um", "spacing_um"):
            if old.get(key) is not None:
                info[key] = old[key]
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, target)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unduh dataset histologi 3D prostat tikus (Tampere University, Fairdata)."
    )
    parser.add_argument("--list", action="store_true", help="Tampilkan tumpukan & ukuran, lalu keluar.")
    parser.add_argument(
        "--stack",
        action="append",
        default=[],
        help="ID tumpukan (mis. PTENX036-32). Boleh diulang atau dipisah koma.",
    )
    parser.add_argument("--all", action="store_true", help="Unduh semua 6 tumpukan.")
    parser.add_argument(
        "--types",
        default=",".join(DEFAULT_TYPES),
        help=f"Tipe folder, dipisah koma, atau 'all'. Default: {','.join(DEFAULT_TYPES)}",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"Folder tujuan (default: {DEFAULT_OUT})")
    parser.add_argument("--jobs", type=int, default=4, help="Jumlah unduhan paralel (default 4).")
    parser.add_argument("--dry-run", action="store_true", help="Tampilkan rencana tanpa mengunduh.")
    args = parser.parse_args()

    print("Mengambil daftar berkas dari Fairdata Metax ...")
    try:
        files = fetch_file_list()
    except (urllib.error.URLError, OSError, ValueError, RuntimeError) as e:
        print(f"Gagal mengambil daftar berkas: {e}", file=sys.stderr)
        return 1

    all_stacks = sorted({f["stack"] for f in files})
    all_types = sorted({f["type"] for f in files})

    if args.list or (not args.stack and not args.all):
        print_listing(files)
        if not args.list:
            print("\nPilih --stack <ID> atau --all untuk mengunduh. Contoh: --stack PTENX036-32")
        return 0

    stacks = all_stacks if args.all else sorted({s.strip() for a in args.stack for s in a.split(",") if s.strip()})
    unknown = [s for s in stacks if s not in all_stacks]
    if unknown:
        print(f"Tumpukan tidak dikenal: {', '.join(unknown)}. Pilihan: {', '.join(all_stacks)}", file=sys.stderr)
        return 2

    types = all_types if args.types.strip().lower() == "all" else [t.strip() for t in args.types.split(",") if t.strip()]
    unknown = [t for t in types if t not in all_types]
    if unknown:
        print(f"Tipe tidak dikenal: {', '.join(unknown)}. Pilihan: {', '.join(all_types)}", file=sys.stderr)
        return 2

    if args.jobs < 1:
        print("--jobs minimal 1.", file=sys.stderr)
        return 2

    selected = sorted(
        (f for f in files if f["stack"] in stacks and f["type"] in types),
        key=lambda f: (f["stack"], f["type"], section_index(f["name"])),
    )
    total = sum(f["size"] for f in selected)
    print(f"Tumpukan : {', '.join(stacks)}")
    print(f"Tipe     : {', '.join(types)}")
    print(f"Berkas   : {len(selected)} ({format_size(total)})")
    print(f"Tujuan   : {args.out}")
    if args.dry_run:
        return 0

    # Hasil: <out>/<stack>/<tipe>/imN.png
    def dest_of(f: dict) -> Path:
        return args.out / f["stack"] / f["type"] / f["name"]

    failures: list[str] = []
    counts = {"unduh": 0, "lewati": 0}
    progress = DownloadProgress(total_files=len(selected), total_bytes=total)
    progress.add_chunk(0, False)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(download_one, f, dest_of(f), on_chunk=progress.add_chunk): f
            for f in selected
        }
        for fut in as_completed(futures):
            f = futures[fut]
            try:
                status = fut.result()
                counts[status] += 1
            except RuntimeError as e:
                failures.append(str(e))
                status = "GAGAL"
            progress.file_finished(f, status)
    progress.finish()

    write_dataset_info(args.out, stacks, files)

    print(f"\nSelesai: {counts['unduh']} diunduh, {counts['lewati']} sudah ada, {len(failures)} gagal.")
    print(f"Metadata: {args.out / 'dataset_info.json'} (mpp_um tidak ditebak; nilai yang sudah diisi dipertahankan)")
    print(f"Atribusi (CC BY 4.0): {CITATION}")
    if failures:
        print("\nBerkas gagal (jalankan ulang perintah yang sama untuk mencoba lagi):", file=sys.stderr)
        for msg in failures:
            print(f"  {msg}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
