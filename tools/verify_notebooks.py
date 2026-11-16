"""Cross-check notebook results against the earlier script run and count outputs/images inside the executed notebooks."""
from pathlib import Path

import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
a = pd.read_csv(ROOT / "results" / "notebook_animal" / "all_predictions.csv")
b = pd.read_csv(ROOT / "results" / "animal_only" / "all_predictions.csv")
key = ["model", "activation", "experiment", "condition", "prompt", "sample_id"]
a, b = a[a.activation.isin(["quickgelu", "gelu"])], b[b.activation.isin(["quickgelu", "gelu"])]
m = a.merge(b, on=key, suffixes=("_nb", "_script"))
print(f"rows notebook: {len(a):,} | rows script (same activations): {len(b):,} | merged: {len(m):,}")
print("identical predictions:", bool((m.prediction_nb == m.prediction_script).all()),
      "| max |confidence diff|:", float((m.confidence_nb - m.confidence_script).abs().max()))

for nbp in sorted((ROOT / "notebooks").glob("0*.ipynb")):
    nb = nbformat.read(nbp, as_version=4)
    code = [c for c in nb.cells if c.cell_type == "code"]
    imgs = sum(1 for c in code for o in c.outputs if o.get("output_type") in ("display_data", "execute_result") and "image/png" in o.get("data", {}))
    errs = sum(1 for c in code for o in c.outputs if o.get("output_type") == "error")
    print(f"{nbp.name}: {len(nb.cells)} cells ({len(code)} code), executed {sum(1 for c in code if c.execution_count)}, "
          f"embedded PNG figures {imgs}, error outputs {errs}, size {nbp.stat().st_size / 1e6:.1f} MB")
