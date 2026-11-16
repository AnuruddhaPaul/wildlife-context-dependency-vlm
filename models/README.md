# Models

**No model weights are stored in this repository.** The experiments use three pretrained, zero-shot vision-language models.
They are downloaded automatically by [OpenCLIP](https://github.com/mlfoundations/open_clip) (v4.2.2) from the Hugging Face Hub the first
time a notebook or script creates them, and are cached in the Hugging Face cache (default `~/.cache/huggingface/hub`).

| Name in the paper | OpenCLIP call (see `scripts/models.py`) | Hugging Face repo | File(s) the loader reads | Size | Parameters | Patch |
|---|---|---|---|---|---|---|
| CLIP ViT-B/32 | `create_model_and_transforms("ViT-B-32", pretrained="openai")` | [`timm/vit_base_patch32_clip_224.openai`](https://huggingface.co/timm/vit_base_patch32_clip_224.openai) | `open_clip_model.safetensors` | ~577 MB | 151.3 M | 32 |
| CLIP ViT-L/14 | `create_model_and_transforms("ViT-L-14", pretrained="openai")` | [`timm/vit_large_patch14_clip_224.openai`](https://huggingface.co/timm/vit_large_patch14_clip_224.openai) | `open_clip_model.safetensors` | ~1.6 GB | 427.6 M | 14 |
| BioCLIP | `create_model_and_transforms("hf-hub:imageomics/bioclip")` | [`imageomics/bioclip`](https://huggingface.co/imageomics/bioclip) | `open_clip_pytorch_model.bin`, `open_clip_config.json` | ~571 MB | 149.6 M | 16 (ViT-B/16) |

Total download: about 2.8 GB. All three are used in evaluation mode only (no training, no fine-tuning).

## Download them yourself (optional, otherwise automatic on first use)
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download timm/vit_base_patch32_clip_224.openai
huggingface-cli download timm/vit_large_patch14_clip_224.openai
huggingface-cli download imageomics/bioclip
```

## Using an existing local cache
`scripts/common.py` points `HF_HUB_CACHE` to `D:/HUG_MODELs` (and sets `HF_HUB_OFFLINE=1`) **only if that folder exists** on the machine that
produced the results. On any other machine it does nothing; to use your own cache folder, set `HF_HUB_CACHE` before starting Python/Jupyter, or edit
`HF_LOCAL_CACHE` in `scripts/common.py`.

## Activation function note
The OpenAI weights were trained with QuickGELU. OpenCLIP 4.x builds `ViT-B-32` / `ViT-L-14` with plain GELU unless `force_quick_gelu=True`
(it only prints a warning). The experiments therefore evaluate every model with both QuickGELU and GELU (`SwitchAct` in notebook 01,
`scripts/run_sweep.py`). BioCLIP's own configuration uses GELU. See the paper (Sec. III-C) for details.

Please check each model card for its license and usage terms before redistributing weights.
