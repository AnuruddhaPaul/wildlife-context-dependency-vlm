"""Verify the PyTorch / CUDA install (RTX 5070 Ti, Blackwell sm_120) with a REAL GPU matmul. No hiding failures."""
import json
import platform
import sys

import torch

from common import RESULTS, set_seed


def check() -> dict:
    set_seed()
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "device_used": "cpu", "gpu_matmul_ok": False, "notes": []}
    try:
        import open_clip
        info["open_clip"] = open_clip.__version__
    except Exception as e:  # pragma: no cover
        info["notes"].append(f"open_clip import failed: {e!r}")
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_capability"] = "sm_%d%d" % torch.cuda.get_device_capability(0)
        info["gpu_arch_list_in_build"] = torch.cuda.get_arch_list()
        info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
        try:
            g = torch.Generator().manual_seed(0)
            a, b = torch.randn(2048, 2048, generator=g), torch.randn(2048, 2048, generator=g)
            out = (a.cuda() @ b.cuda()).cpu()
            torch.cuda.synchronize()
            err = (out - a @ b).abs().max().item()
            info["gpu_matmul_max_abs_err_vs_cpu"] = err
            info["gpu_matmul_ok"] = bool(err < 1e-2)
            if info["gpu_matmul_ok"]:
                info["device_used"] = "cuda"
            else:
                info["notes"].append(f"GPU matmul result disagrees with CPU (max abs err {err}); using CPU")
        except Exception as e:
            info["notes"].append(f"CUDA KERNELS FAILED: {e!r} -> falling back to CPU")
    else:
        info["notes"].append("torch.cuda.is_available() is False -> CPU")
    return info


if __name__ == "__main__":
    info = check()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "env_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
