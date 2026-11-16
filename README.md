# Visual Context Dependency in Wildlife Classification

A systematic study of how much zero-shot vision-language models (**CLIP ViT-B/32, CLIP ViT-L/14, BioCLIP**) rely on the habitat around an animal versus
the animal itself, on **Caltech Camera Traps** (16 animal species, 1,472 bounding-box samples). Four controlled experiments and three metrics:

| Experiment | What is changed | Metric |
|---|---|---|
| 1. Gradual context removal | crop around the box with 100 / 50 / 25 / 5 / 0 % margin | **CDS** - Context Dependency Score |
| 2. Context mismatch | animal pasted on a wrong-habitat synthetic background | **CR** - Context Robustness |
| 3. Prompt manipulation | four text templates | **PS** - Prompt Sensitivity |
| 4. Background only | animal blurred, habitat kept | accuracy vs. 6.25 % chance |

Every model is evaluated with **QuickGELU** (primary) and **GELU** activations (see `models/README.md` for why).

## Headline results (16 animal classes, QuickGELU)
| Model | Acc full | Acc ultra-tight | CDS | CR | PS | BG-only |
|---|---|---|---|---|---|---|
| ViT-B/32 | 36.8 | 38.2 | -3.9 (95% CI -11.4 to 3.0) | 110.8 | 61.5 | 10.0 |
| ViT-L/14 | 59.2 | 53.9 | 9.1 (4.8 to 13.0) | 84.9 | 25.8 | 11.2 |
| BioCLIP | 45.0 | 31.0 | 31.2 (26.0 to 36.3) | 43.6 | 49.0 | 13.8 |

BioCLIP is the most context dependent and least robust model; ViT-B/32 shows no measurable benefit from context. Further findings (activation-function
sensitivity, sign change of context dependence between day and infrared night frames, failure of the `car` control class) are in the paper.

## Repository layout
```
paper/        IEEE (IEEEtran) LaTeX source: main.tex, auto-generated numbers.tex, figures/, tables/, tables_csv/
notebooks/    01_animal_only_quickgelu_gelu.ipynb  - EDA, sample points, the full experiment (QuickGELU + GELU), results   [executed]
              02_paper_results_and_benchmark.ipynb - all paper tables/figures, benchmark vs. the earlier 17-class paper, claim checks, results EDA [executed]
scripts/      pipeline modules (imaging, models, habitats, metrics) and command-line runners
tools/        notebook builders/runner, paper-asset generator, LaTeX static checker, Windows helper scripts
results/      CSV/JSON results and per-image predictions (notebook_animal/, notebook_report/, animal_only/, plain_gelu/, ...)
figures/      all figures produced by the runs
data/         sample.csv = the evaluation sample list (images and raw metadata are downloaded, see below)
models/       README only - NO weights are stored here; it lists the three models and where to download them
wildlife_context_paper_package.zip   single-file snapshot of this repository (everything above, without .git)
```

## Reproduce
```bash
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128   # CUDA 12.8 build
pip install -r requirements.txt
python scripts/00_download.py caltech_bboxes_20200316.json      # LILA BC bbox metadata (38 MB)
python scripts/06_build_sample.py                               # 1,572-box sample (16 animals after dropping 'car'), seed 42
python scripts/07_fetch_images.py                               # ~1.5k full-resolution images from LILA's per-image copy (~650 MB)
jupyter lab notebooks/01_animal_only_quickgelu_gelu.ipynb       # ~10 min on an RTX 5070 Ti; then notebooks/02_...
python tools/make_paper_assets.py                               # regenerate paper figures, tables and numbers.tex
```
Model weights download automatically on first use (about 2.8 GB, see `models/README.md`). `NB_SMOKE=1` runs notebook 01 on a 3-species smoke subset.
Compile the paper with pdfLaTeX (Overleaf: upload the contents of `paper/`; main document `main.tex`).

## Notes and deviations (all deliberate; nothing was tuned to match earlier numbers)
- `caltech_images_20210113.json` is not published on LILA; the bbox file the dataset page lists, `caltech_bboxes_20200316.json`, is used.
- Sampling is **per bounding box** (badger 28 / mountain lion 44 are box counts); the `eccv_18_all_images_sm` subset lacks mountain-lion and lizard images, so
  the needed full-resolution originals are fetched individually.
- The Experiment-2 species-to-habitat assignment (`results/notebook_animal/habitat_mapping.csv`) is a reconstruction.
- The `car` control class is excluded from the main analysis: its boxes fill the frame, so CDS is zero by construction. The 17-class runs are kept for comparison.
- `paper/main.tex` was statically checked (`tools/check_tex.py`) but **not compiled** where it was produced (no LaTeX engine available); please compile once.
  Author block, affiliation and references need your review (references were written from memory).
- Package installation on the machine that produced the results used a PowerShell/.NET wheel unpacker (`tools/install_wheels.ps1`) because pip's own extraction hung; a normal `pip install` works elsewhere.

## License
No license has been chosen yet; add one before making the repository public. Data: Caltech Camera Traps via LILA BC; please follow its terms.
