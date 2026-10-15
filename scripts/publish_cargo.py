#!/usr/bin/env python3
"""Automated standalone Crates.io publisher.

Allows publishing the Rust crate to crates.io using pure standard library Python,
even on machines without cargo or rustc installed.
Reads CARGO_TOKEN from .env.
"""

import io
import re
import json
import tarfile
import struct
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUST_DIR = ROOT / "rust"
ENV_FILE = ROOT / ".env"

token = None
if ENV_FILE.exists():
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        m = re.match(r'^\s*CARGO_TOKEN\s*=\s*["\']?([^"\']+)["\']?', line)
        if m:
            token = m.group(1).strip()
            break

if not token:
    raise SystemExit("Error: CARGO_TOKEN not found in .env file.")

name = "iaexplore-llm-profiler"
version = "1.0.0"
prefix = f"{name}-{version}"

print(f"[*] Packaging {prefix} for Crates.io...")
tar_buf = io.BytesIO()
with tarfile.open(fileobj=tar_buf, mode="w:gz") as tar:
    for rel_path in ["Cargo.toml", "README.md", "src/lib.rs", "src/main.rs"]:
        file_path = RUST_DIR / rel_path
        with open(file_path, "rb") as f:
            data = f.read()
        ti = tarfile.TarInfo(name=f"{prefix}/{rel_path}")
        ti.size = len(data)
        ti.mtime = 1710000000
        tar.addfile(ti, io.BytesIO(data))

tar_bytes = tar_buf.getvalue()

metadata = {
    "name": name,
    "vers": version,
    "deps": [],
    "features": {},
    "authors": ["IAExplore AI Infrastructure Research <engineering@iaexplore.com>"],
    "description": "Deterministic high-precision VRAM, GQA KV-Cache, and MoE memory bandwidth inference profiler for 2026 AI models and silicon.",
    "homepage": "https://www.iaexplore.com/html/en/blog/ai/npu-tpu-gpu-ai-hardware-guide.html",
    "documentation": "https://www.iaexplore.com/html/en/blog/comparisons/best-computers-which-pc-to-buy-based-on-your-needs-and-budget.html",
    "readme": (RUST_DIR / "README.md").read_text(encoding="utf-8"),
    "readme_file": "README.md",
    "keywords": ["llm", "vram", "kv-cache", "gqa", "llama-cpp"],
    "categories": ["command-line-utilities", "science", "hardware-support"],
    "license": "MIT",
    "license_file": None,
    "repository": "https://github.com/iaexplore/iaexplore-llm-profiler",
    "links": None,
}

json_bytes = json.dumps(metadata).encode("utf-8")
payload = struct.pack("<I", len(json_bytes)) + json_bytes + struct.pack("<I", len(tar_bytes)) + tar_bytes

print("[*] Submitting to Crates.io API (https://crates.io/api/v1/crates/new)...")
req = urllib.request.Request(
    "https://crates.io/api/v1/crates/new",
    data=payload,
    headers={
        "Authorization": token,
        "Content-Type": "application/octet-stream",
        "User-Agent": "cargo 1.80.0 (iaexplore-llm-profiler)",
    },
    method="PUT",
)

try:
    with urllib.request.urlopen(req) as resp:
        print("[OK] Successfully published to Crates.io:", resp.status, resp.read().decode())
except urllib.error.HTTPError as e:
    err_body = e.read().decode()
    print(f"[!] Crates.io API returned HTTP {e.code}: {err_body}")
except Exception as e:
    print("[!] Upload error:", e)
