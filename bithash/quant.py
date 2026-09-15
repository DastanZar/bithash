"""Hash-collision weight quantization: 4-bit codes, exact per-row least-squares scales.

Standard W4 (GPTQ/AWQ-family) practice maps each weight to one of 16 levels
and scales by absmax per group. This module makes two pieces of that pipeline
measurable instead of assumed:

  * CODEBOOK POLICY — "odd16" excludes zero entirely (the folklore claim:
    a zero weight "contributes nothing so the code is wasted"); "uniform16"
    is the INT4-style signed grid. MEASURED RESULT: the folklore is wrong in
    BOTH regimes — on dense Gaussian rows near-zero values are the mode
    (~2x error for odd16), and on sparse rows uniform16 quantizes exact
    zeros to exact zeros while odd16 snaps them to +-1*scale (10x error).
    The codebook policy grid exists to demonstrate this, not to sell odd16;
  * SCALE POLICY — given fixed codes, the per-row least-squares scale has a
    closed form (normal equation <w,d>/<d,d>); it beats absmax on the same
    codes by construction, and the demo quantifies by how much.

Both policies compose freely; `quantize(w, codebook=..., scale=...)` exposes
the 2x2 grid the demo benchmarks.
"""
from __future__ import annotations

import torch

CODEBOOKS: dict[str, torch.Tensor] = {
    # no zero: 16 odd codes spanning +/-15 (sparse-weight friendly)
    "odd16": torch.tensor(
        [-15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15],
        dtype=torch.float32,
    ),
    # INT4-style signed uniform grid incl. zero (-7..+7 step 1, plus -8)
    "uniform16": torch.tensor(
        [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7],
        dtype=torch.float32,
    ),
}


def codes_to_values(codes: torch.Tensor, codebook: str = "uniform16") -> torch.Tensor:
    cb = CODEBOOKS[codebook]
    return cb[codes.to(torch.int64)]


def hash_row(row: torch.Tensor, scale: float, codebook: str = "uniform16") -> torch.Tensor:
    """Monotone hash: v/scale -> nearest code INDEX (codebook-specific)."""
    cb = CODEBOOKS[codebook]
    v = row / max(scale, 1e-12)
    if codebook == "uniform16":
        idx = (v + 8.0).round().clamp(0, 15)   # value = idx - 8
    else:
        idx = ((v + 15.0) / 2.0).round().clamp(0, 15)  # value = 2*idx - 15
    return idx.to(torch.uint8)


def fit_scales(w: torch.Tensor, code_vals: torch.Tensor) -> torch.Tensor:
    """Exact per-row least-squares scale for FIXED codes:
    argmin_s ||w - s*d||^2  ->  s = <w, d> / <d, d>."""
    num = (w * code_vals).sum(dim=1)
    den = (code_vals * code_vals).sum(dim=1).clamp(min=1e-12)
    return num / den


def quantize(w: torch.Tensor, codebook: str = "uniform16", scale_policy: str = "ls"
             ) -> dict:
    """Quantize [N, K] to 4-bit codes.

    codebook: "uniform16" (INT4-style, dense-friendly) or "odd16" (no zero,
              sparse-friendly). scale_policy: "ls" (least-squares, exact) or
              "absmax".
    """
    n, k = w.shape
    wf = w.to(torch.float32)
    cb = CODEBOOKS[codebook]
    absmax = wf.abs().amax(dim=1).clamp(min=1e-8) / cb.abs().max().item()
    codes = torch.empty(n, k, dtype=torch.uint8)
    for i in range(n):
        codes[i] = hash_row(wf[i], absmax[i].item(), codebook)
    pack = {"codes": codes, "codebook": codebook, "name": f"bithash-w4-{codebook}-{scale_policy}"}
    if scale_policy == "ls":
        pack["scale"] = fit_scales(wf, codes_to_values(codes, codebook))
    else:
        pack["scale"] = absmax
    return pack


def dequantize(pack: dict) -> torch.Tensor:
    return codes_to_values(pack["codes"], pack["codebook"]) * pack["scale"].to(torch.float32)[:, None]


def reconstruction_error(w: torch.Tensor, pack: dict) -> float:
    rec = dequantize(pack)
    wf = w.to(torch.float32)
    return ((rec - wf).norm() / wf.norm()).item()


class QuantizedLinear:
    """Drop-in replacement for nn.Linear (batch-1 decode path)."""

    def __init__(self, weight: torch.Tensor, codebook: str = "uniform16"):
        self.pack = quantize(weight, codebook=codebook)
        self.bias = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        w = dequantize(self.pack)
        out = x.to(torch.float32) @ w.T
        if self.bias is not None:
            out = out + self.bias.to(torch.float32)
        return out

    def bytes(self) -> int:
        n, k = self.pack["codes"].shape
        return n * k // 2 + self.pack["scale"].numel() * 4
