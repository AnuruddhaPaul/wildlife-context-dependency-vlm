"""Register a Jupyter kernel named 'wildlife-cdr' that launches THIS venv's python (user-level kernelspec)."""
import json
import sys
import tempfile
from pathlib import Path

from jupyter_client.kernelspec import KernelSpecManager

spec = {"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Python (wildlife_context_repro)", "language": "python", "metadata": {"debugger": True}}
with tempfile.TemporaryDirectory() as d:
    (Path(d) / "kernel.json").write_text(json.dumps(spec, indent=1))
    dest = KernelSpecManager().install_kernel_spec(d, kernel_name="wildlife-cdr", user=True)
print("installed ->", dest)
print(json.loads((Path(dest) / "kernel.json").read_text())["argv"][:3])
