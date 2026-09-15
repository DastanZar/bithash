"""bithash figures: the measurement apparatus drawn; and the sparse-regime bar (the finding)."""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bithash import dequantize, quantize, reconstruction_error  # noqa: E402
from bithash.quant import CODEBOOKS  # noqa: E402

FIG = os.path.join(ROOT, "assets")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 140, "font.size": 9})

# Fig 1: the two grids on real weights, with the zero-zone blow-up
torch.manual_seed(1)
w = torch.randn(4096) * 0.05
grid = w.quantile(torch.linspace(0, 1, 500))
fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [2, 1]})
t = torch.linspace(-0.16, 0.16, 2)
for name, col in (("uniform16", "#4472c4"), ("odd16", "#c0504d")):
    cb = CODEBOOKS[name] / 15.0 * 0.16
    for c in cb:
        axA.axvline(c.item(), color=col, alpha=.35, lw=.7)
hist = axA.hist(w, bins=120, range=(-0.17, 0.17), color="#ddd", edgecolor="#999",
                label="Gaussian weights")
axA.axvspan(-0.02, 0.02, color="#ffd", zorder=0)
axA.text(0, hist[0].max() * .92, "the zero zone:\nuniform codes → 0 exactly,\nodd codes → ±1·scale",
         ha="center", fontsize=7.2, color="#665")
axA.set_xlabel("weight value"); axA.set_ylabel("count"); axA.legend(fontsize=7)
axA.set_title("one shared 16-code grid per tensor — where the codes land", fontsize=9.5)

# per-weight error histogram for both codebooks at identical absmax scale
errs = {}
for name in ("uniform16", "odd16"):
    wf = w.view(1, -1)
    p = quantize(wf, codebook=name, scale_policy="absmax")
    e = (dequantize(p) - wf).abs().flatten()
    errs[name] = e
axB.hist(errs["uniform16"].tolist(), bins=60, range=(0, .02), color="#4472c4", alpha=.6, label="uniform")
axB.hist(errs["odd16"].tolist(), bins=60, range=(0, .02), color="#c0504d", alpha=.6, label="no-zero")
axB.set_title("|w − Q(w)| per weight\n(same scale, only the grid differs)", fontsize=9)
axB.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIG}/codebook_grids.png"); plt.close(fig)

# Fig 2: THE finding — dense and sparse side by side, log scale
fig, ax = plt.subplots(figsize=(6.4, 4.0))
dense = torch.randn(256, 2048) * 0.05
sp = torch.zeros(64, 2048)
idx = torch.randint(0, 64 * 2048, (512,))
flat = sp.flatten(); flat[idx] = torch.randn(512) * 15.0; sp = flat.reshape(64, 2048)
rows = {
    "uniform16": (reconstruction_error(dense, quantize(dense, "uniform16")),
                  reconstruction_error(sp, quantize(sp, "uniform16"))),
    "odd16": (reconstruction_error(dense, quantize(dense, "odd16")),
              reconstruction_error(sp, quantize(sp, "odd16"))),
}
x = np.arange(2)
ax.bar(x - .18, [rows["uniform16"][i] for i in (0, 1)], .36, color="#4472c4", label="uniform16 (INT4 grid)")
ax.bar(x + .18, [rows["odd16"][i] for i in (0, 1)], .36, color="#c0504d", label="odd16 (no zero)")
for i, (lbl, (d, s)) in enumerate(rows.items()):
    ax.text(i - .18, d * 1.08, f"{d:.3f}", ha="center", fontsize=8)
    ax.text(i + .18, s * 1.08, f"{s:.3f}", ha="center", fontsize=8, color="#a33")
ax.set_xticks(x, ["dense Gaussian rows", "99.25% sparse rows"])
ax.set_yscale("log"); ax.set_ylabel("reconstruction rel. error")
ax.set_title("the folklore loses in BOTH regimes —\nuniform wins, and by 10× on sparse", fontsize=9.5)
ax.legend(fontsize=8); ax.grid(axis="y", alpha=.2, which="both")
fig.tight_layout(); fig.savefig(f"{FIG}/falsified_folklore.png"); plt.close(fig)

# Fig 3: the LS-vs-absmax scale fit, visualized on one row
fig, ax = plt.subplots(figsize=(6.6, 3.4))
w1 = (torch.randn(2048) * 0.05)
p_am = quantize(w1.unsqueeze(0), "uniform16", "absmax")
p_ls = quantize(w1.unsqueeze(0), "uniform16", "ls")
srt = w1.sort()
ax.plot(srt.values.tolist(), label="true weights (sorted)", color="#333", lw=1)
ax.plot(dequantize(p_am)[0].sort().values.tolist(), "--", color="#c0504d", lw=1, label="absmax scale")
ax.plot(dequantize(p_ls)[0].sort().values.tolist(), ":", color="#4472c4", lw=1.6, label="LS scale")
ax.legend(fontsize=8); ax.set_xlabel("rank"); ax.set_ylabel("value")
ax.set_title("same 4-bit codes; the LS fit (closed form) hugs the true distribution tighter", fontsize=9)
ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(f"{FIG}/scale_fit.png"); plt.close(fig)
print("bithash figures ->", sorted(os.listdir(FIG)))
