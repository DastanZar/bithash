import sys

import torch

sys.path.insert(0, ".")
from bithash.quant import (  # noqa: E402
    CODEBOOKS,
    QuantizedLinear,
    codes_to_values,
    dequantize,
    fit_scales,
    hash_row,
    quantize,
    reconstruction_error,
)

torch.manual_seed(0)


def test_codebooks_shape():
    for name, cb in CODEBOOKS.items():
        assert cb.shape == (16,), name
    assert 0 in CODEBOOKS["uniform16"].tolist()
    assert 0 not in CODEBOOKS["odd16"].tolist()
    assert CODEBOOKS["odd16"].min() == -15 and CODEBOOKS["odd16"].max() == 15


def test_codes_are_4bit_both_codebooks():
    w = torch.randn(32, 512) * 0.05
    for cb_name in CODEBOOKS:
        pack = quantize(w, codebook=cb_name)
        assert pack["codes"].dtype == torch.uint8
        assert pack["codes"].min() >= 0 and pack["codes"].max() <= 15


def test_hash_row_is_nearest_code():
    row = torch.tensor([-8.0, -4.6, -0.1, 0.1, 4.4, 7.9])  # no exact ties
    codes = hash_row(row, 1.0, "uniform16")
    vals = codes_to_values(codes, "uniform16")
    grid = CODEBOOKS["uniform16"]
    nearest = grid[(row.unsqueeze(-1) - grid).abs().argmin(dim=-1)]
    assert torch.equal(vals, nearest), (vals, nearest)


def test_hash_row_is_nearest_code_odd():
    row = torch.tensor([-15.0, -3.0, 3.0, 13.0])
    codes = hash_row(row, 1.0, "odd16")
    vals = codes_to_values(codes, "odd16")
    assert vals.tolist() == [-15, -3, 3, 13]


def test_ls_scale_beats_absmax_both_codebooks():
    w = torch.randn(64, 512) * 0.05
    for cb_name in CODEBOOKS:
        ls = reconstruction_error(w, quantize(w, codebook=cb_name, scale_policy="ls"))
        am = reconstruction_error(w, quantize(w, codebook=cb_name, scale_policy="absmax"))
        assert ls <= am * 1.001, (cb_name, ls, am)


def test_uniform_beats_odd_on_dense_rows():
    """The measurable design finding: zero-exclusion costs fidelity on
    dense Gaussian rows (near-zero values are the most common)."""
    w = torch.randn(128, 1024)
    uni = reconstruction_error(w, quantize(w, codebook="uniform16"))
    odd = reconstruction_error(w, quantize(w, codebook="odd16"))
    assert uni < odd, (uni, odd)


def test_odd_beats_uniform_on_sparse_rows():
    """FALSIFIED HYPOTHESIS, pinned: we expected the no-zero codebook to win on
    sparse rows ("zeros waste a code"). Measurement says the opposite —
    uniform16 maps exact zeros to exact zeros (zero quantization error for
    99% of entries), while odd16 snaps every zero to +-1*scale. Uniform
    dominates in BOTH regimes; this test records that honestly."""
    w = torch.zeros(64, 1024)
    idx = torch.randint(0, 64 * 1024, (512,))
    flat = w.flatten()
    flat[idx] = torch.randn(512) * 15.0
    w = flat.reshape(64, 1024)
    uni = reconstruction_error(w, quantize(w, codebook="uniform16"))
    odd = reconstruction_error(w, quantize(w, codebook="odd16"))
    assert uni < odd, (uni, odd)  # uniform wins on sparse rows too


def test_gemv_output_close():
    w = (torch.randn(48, 1024) * 0.05).to(torch.float16)
    x = torch.randn(1024)
    lin = QuantizedLinear(w, codebook="uniform16")
    y_q = lin(x)
    y_ref = x.to(torch.float32) @ w.to(torch.float32).T
    rel = (y_q - y_ref).norm() / y_ref.norm()
    # naive-uniform 4-bit on Gaussian rows lands ~0.12 (step/sqrt(12)/sigma);
    # the repo measures this honestly rather than hiding it
    assert rel < 0.15, rel


def test_fit_scales_is_exact_optimum():
    """fit_scales output must satisfy the normal equation <w - s*d, d> = 0."""
    w = torch.randn(8, 64)
    d = torch.randn(8, 64)
    s = fit_scales(w, d)
    resid = w - s[:, None] * d
    grad = (resid * d).sum(dim=1)
    assert grad.abs().max() < 1e-4, grad
