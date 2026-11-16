"""Zero-shot CLIP-family wrappers: image embeddings + class-prompt text embeddings (open_clip)."""
import numpy as np
import open_clip
import torch

from common import SPECIES, class_name

# OpenAI weights were trained with QuickGELU; open_clip >= 3 only warns (and builds plain GELU) unless forced.
MODEL_SPECS = {
    "ViT-B/32": dict(name="ViT-B-32", pretrained="openai", quick_gelu=True),
    "ViT-L/14": dict(name="ViT-L-14", pretrained="openai", quick_gelu=True),
    "BioCLIP": dict(name="hf-hub:imageomics/bioclip", pretrained=None, quick_gelu=False),
}
PROMPTS = {
    "neutral": "a photo of a {}",
    "habitat": "a photo of a {} in its natural habitat",
    "wildlife": "a wildlife camera trap image of a {}",
    "scientific": "a {} specimen",
}


class ZeroShotModel:
    def __init__(self, label: str, device: str, force_quick_gelu: bool = True):
        """force_quick_gelu=False reproduces the literal create_model_and_transforms(name, pretrained='openai') call
        of open_clip 4.x (plain GELU + a warning); True builds the architecture the OpenAI weights were trained with."""
        spec = MODEL_SPECS[label]
        self.label, self.device = label, device
        kw = {"pretrained": spec["pretrained"]} if spec["pretrained"] else {}
        qg = spec["quick_gelu"] and force_quick_gelu
        if qg:
            kw["force_quick_gelu"] = True
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(spec["name"], **kw)
        self.tokenizer = open_clip.get_tokenizer(spec["name"])
        self.model.to(device).eval()
        self.logit_scale = float(self.model.logit_scale.exp().item())
        self.info = dict(
            open_clip_name=spec["name"], pretrained=spec["pretrained"], quick_gelu_forced=qg,
            patch_size=list(self.model.visual.conv1.kernel_size), image_size=self.model.visual.image_size,
            params_total_M=round(sum(p.numel() for p in self.model.parameters()) / 1e6, 1),
            params_visual_M=round(sum(p.numel() for p in self.model.visual.parameters()) / 1e6, 1))
        self._text: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def text_embeddings(self, prompt_key: str) -> torch.Tensor:
        """Cached [17, D] L2-normalised text embeddings for one prompt template."""
        if prompt_key not in self._text:
            texts = [PROMPTS[prompt_key].format(class_name(s)) for s in SPECIES]
            f = self.model.encode_text(self.tokenizer(texts).to(self.device))
            self._text[prompt_key] = f / f.norm(dim=-1, keepdim=True)
        return self._text[prompt_key]

    @torch.no_grad()
    def encode_images(self, batch: torch.Tensor) -> np.ndarray:
        f = self.model.encode_image(batch.to(self.device, non_blocking=True))
        return (f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy()

    @torch.no_grad()
    def classify(self, feats: np.ndarray, prompt_key: str) -> tuple[np.ndarray, np.ndarray]:
        """argmax cosine similarity; confidence = softmax(logit_scale * sims)[pred]."""
        sims = torch.from_numpy(feats).to(self.device) @ self.text_embeddings(prompt_key).T
        probs = torch.softmax(self.logit_scale * sims, dim=-1)
        conf, pred = probs.max(dim=-1)
        return pred.cpu().numpy(), conf.cpu().numpy()
