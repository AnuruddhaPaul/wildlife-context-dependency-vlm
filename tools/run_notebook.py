"""Execute a notebook top to bottom with the project kernel and save it with outputs.  Usage: python run_notebook.py in.ipynb out.ipynb"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

src, dst = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
nb = nbformat.read(src, as_version=4)
client = NotebookClient(nb, timeout=5400, kernel_name="wildlife-cdr", resources={"metadata": {"path": str(src.parent)}},
                        allow_errors=False, record_timing=True)
t0 = time.time()
try:
    client.execute()
    status = "OK"
except Exception as e:                              # keep the partial notebook (with the failing cell's traceback) for debugging
    status = f"FAILED: {type(e).__name__}: {str(e)[:1500]}"
nbformat.write(nb, dst)
n_code = sum(c.cell_type == "code" for c in nb.cells)
n_done = sum(1 for c in nb.cells if c.cell_type == "code" and c.get("execution_count"))
print(f"{status} | executed {n_done}/{n_code} code cells in {time.time() - t0:.0f}s -> {dst}")
sys.exit(0 if status == "OK" else 1)
