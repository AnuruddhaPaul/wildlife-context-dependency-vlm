"""Build the ONE release zip: a snapshot of every git-tracked file (so .gitignore rules, e.g. no model weights, apply).

Usage (after `git add -A`, before committing the zip):  python tools/make_release_zip.py
"""
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAME = "wildlife_context_paper_package.zip"
out = ROOT / NAME

# files git would commit: tracked + staged (not ignored); the zip itself is excluded
files = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8").split("\0")
files = sorted(f for f in files if f and f != NAME and (ROOT / f).is_file())
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for f in files:
        z.write(ROOT / f, f"wildlife_context_paper_package/{f}")
bad = zipfile.ZipFile(out).testzip()
print(f"{NAME}: {len(files)} files, {out.stat().st_size / 1048576:.1f} MiB, integrity {'OK' if bad is None else 'CORRUPT ' + bad}")
weights = [f for f in files if f.lower().endswith((".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".onnx", ".h5", ".npz"))]
print("model-weight-like files inside the zip:", weights or "none")
