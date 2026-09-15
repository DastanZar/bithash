"""Demo: codebook x scale-policy grid, error + bytes receipt + chart."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bithash import quantize, reconstruction_error  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "receipts")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(11)
w = (torch.randn(256, 2048) * 0.05).to(torch.float16)

grid = {}
for cb in ("uniform16", "odd16"):
    for pol in ("absmax", "ls"):
        grid[f"{cb}/{pol}"] = round(reconstruction_error(w, quantize(w, codebook=cb, scale_policy=pol)), 4)

packed_bytes = w.numel() // 2 + 256 * 4
receipt = {
    "shape": list(w.shape),
    "bits_per_weight_effective": round(packed_bytes * 8 / w.numel(), 3),
    "compression_vs_fp16": round((w.numel() * 2) / packed_bytes, 2),
    "reconstruction_rel_err_grid": grid,
    "findings": [
        "LS scale always beats absmax on identical codes (by construction)",
        "uniform16 (with zero) beats odd16 (no-zero) on dense rows: near-zero weights are the mode",
    ],
}
with open(os.path.join(OUT, "receipt.json"), "w") as f:
    json.dump(receipt, f, indent=2)
print(json.dumps(receipt, indent=2))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(grid)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colors = ["#a55" if "absmax" in n else "#5a7" for n in names]
    ax.bar(names, [grid[n] for n in names], color=colors)
    ax.set_ylabel("reconstruction rel. error")
    ax.set_title("4-bit codebook x scale policy (dense Gaussian rows)")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "scales.png"), dpi=140)
    print("chart -> receipts/scales.png")
except Exception as e:  # pragma: no cover
    print("chart skipped:", e)

assert grid["uniform16/ls"] < grid["uniform16/absmax"]
assert grid["odd16/ls"] < grid["odd16/absmax"]
assert grid["uniform16/ls"] < grid["odd16/ls"]
print("DEMO_OK")
