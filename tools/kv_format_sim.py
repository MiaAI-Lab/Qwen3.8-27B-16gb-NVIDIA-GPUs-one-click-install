#!/usr/bin/env python3
"""KV-cache format simulator (CPU, numpy only): stock ExLlamaV3 int8/int4 vs the
fp8 / nvfp4 lanes, on synthetic K/V with Qwen-like statistics.

This is the *format* comparison, independent of the Triton kernels: it re-implements
  - stock int{2..8}: Hadamard-32 rotation, absmax/32-group fp16 scale, midpoint grid
    (q_cache_kernels.cuh, compand_a = 0)
  - fp8: raw E4M3 per element, no scales (cache/fp8.py)
  - nvfp4: E2M1 per element, E4M3 scale per 16 = amax/6 (cache/nvfp4.py)
and measures dequantisation error plus the error it causes in attention outputs.
The kernel-level parity tests (in-kernel decode == this math) run on the GPU via
tools/kv_cache_tests.py.

    python tools/kv_format_sim.py            # default: 8 kv heads x 128 dims, 4096 tokens
"""
from __future__ import annotations

import argparse
import numpy as np

# ----------------------------------------------------------------- formats ----

def e4m3_round(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest to the FP8 E4M3 (fn) grid: 3 mantissa bits, exponents
    2^-6 .. 2^8, subnormal step 2^-9, max 448, no inf."""
    x = x.astype(np.float64)
    a = np.abs(x)
    a = np.minimum(a, 448.0)
    e = np.floor(np.log2(np.maximum(a, 2.0 ** -6)))       # exponent of the binade
    step = 2.0 ** (e - 3)                                   # 3 mantissa bits
    step = np.where(a < 2.0 ** -6, 2.0 ** -9, step)         # subnormals
    q = np.round(a / step) * step
    q = np.minimum(q, 448.0)
    return (np.sign(x) * q).astype(np.float32)


_E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_E2M1_BOUNDS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def nvfp4(x: np.ndarray) -> np.ndarray:
    """cache/nvfp4.py: blocks of 16 along the last dim, scale = amax/6 in E4M3,
    values -> nearest E2M1 level of |x|/scale (clamped to 6)."""
    g = x.shape[-1]
    xg = x.astype(np.float32).reshape(*x.shape[:-1], g // 16, 16)
    amax = np.abs(xg).max(axis=-1, keepdims=True)
    scale = e4m3_round(np.minimum(amax / 6.0, 448.0))
    sc = np.where(scale == 0, 1.0, scale)
    t = np.minimum(np.abs(xg) / sc, 6.0)
    idx = (t[..., None] > _E2M1_BOUNDS).sum(axis=-1)
    mag = _E2M1[idx]
    return (np.sign(xg) * mag * scale).reshape(x.shape).astype(np.float32)


def fp8(x: np.ndarray) -> np.ndarray:
    return e4m3_round(x)


def _hadamard(n: int) -> np.ndarray:
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


_H32 = _hadamard(32) / np.sqrt(32.0)


def stock_int(x: np.ndarray, bits: int) -> np.ndarray:
    """q_cache_kernels.cuh quant_block_x4 / dequant, compand_a = 0:
    group of 32 -> H32 rotate (1/sqrt(32)) -> absmax scale (fp16) -> midpoint grid."""
    g = x.shape[-1]
    xg = x.astype(np.float32).reshape(*x.shape[:-1], g // 32, 32)
    r = xg @ _H32.T
    s = np.abs(r).max(axis=-1, keepdims=True) + 1e-10
    s = s.astype(np.float16).astype(np.float32)                   # fp16 scale
    m = float(1 << (bits - 1))
    qmax = (1 << bits) - 1
    q = np.floor(r / s * m + m)
    q = np.clip(q, 0, qmax)
    deq = ((2 * q + 1) / (1 << bits) - 1.0) * s                   # midpoint centroids
    back = deq @ _H32                                               # H is symmetric/orthonormal
    return back.reshape(x.shape).astype(np.float32)


FORMATS = {
    "fp16":   lambda x: x.astype(np.float16).astype(np.float32),
    "int8":   lambda x: stock_int(x, 8),
    "fp8":    fp8,
    "int6":   lambda x: stock_int(x, 6),
    "int4":   lambda x: stock_int(x, 4),
    "nvfp4":  nvfp4,
}
BITS = {"fp16": 16, "int8": 8.5, "fp8": 8, "int6": 6.5, "int4": 4.5, "nvfp4": 4.5}


# ------------------------------------------------------------ synthetic KV ----

def make_kv(tokens: int, heads: int, dim: int, seed: int, profile: str):
    """Qwen-like statistics. K is post-RMSNorm (unit RMS per head) with a learned
    per-channel gain that has a few large channels (the classic 'outlier dims'),
    then RoPE-like mixing. V is un-normalised, small, heavier-tailed."""
    rng = np.random.default_rng(seed)
    if profile == "gaussian":
        k = rng.standard_normal((tokens, heads, dim)).astype(np.float32)
        v = (rng.standard_normal((tokens, heads, dim)) * 0.1).astype(np.float32)
        return k, v
    k = rng.standard_normal((tokens, heads, dim)).astype(np.float32)
    k /= np.sqrt((k ** 2).mean(axis=-1, keepdims=True))
    gain = np.exp(rng.normal(0.0, 0.35, size=(heads, dim))).astype(np.float32)
    n_out = max(1, dim // 32)
    for h in range(heads):
        idx = rng.choice(dim, n_out, replace=False)
        gain[h, idx] *= rng.uniform(4.0, 10.0, size=n_out)
    k = k * gain
    # RoPE: rotate pairs by position-dependent angles (mixes channel pairs)
    pos = np.arange(tokens)[:, None]
    inv = 1.0 / (10_000_000 ** (np.arange(0, dim, 2) / dim))
    ang = pos * inv
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    k1, k2 = k[..., 0::2], k[..., 1::2]
    k = np.empty_like(k)
    k[..., 0::2] = k1 * c - k2 * s
    k[..., 1::2] = k1 * s + k2 * c
    v = rng.standard_t(df=4, size=(tokens, heads, dim)).astype(np.float32)
    vgain = np.exp(rng.normal(np.log(0.08), 0.5, size=(heads, dim))).astype(np.float32)
    v = v * vgain
    return k.astype(np.float32), v.astype(np.float32)


# ----------------------------------------------------------------- metrics ----

def rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).mean()) / (np.sqrt((a ** 2).mean()) + 1e-12))


def attention(q, k, v):
    """q: (nq, heads, dim); k, v: (tokens, heads, dim). Softmax over tokens."""
    d = q.shape[-1]
    logits = np.einsum("qhd,thd->hqt", q, k) / np.sqrt(d)
    logits -= logits.max(axis=-1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=-1, keepdims=True)
    out = np.einsum("hqt,thd->qhd", p, v)
    return out, p


def kl(p, q):
    return float((p * (np.log(p + 1e-30) - np.log(q + 1e-30))).sum(axis=-1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--queries", type=int, default=64)
    ap.add_argument("--profile", choices=["qwen", "gaussian"], default="qwen")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    k, v = make_kv(a.tokens, a.heads, a.dim, a.seed, a.profile)
    rng = np.random.default_rng(a.seed + 1)
    # queries that actually attend sharply: mix of real keys + noise, like a trained model
    pick = rng.choice(a.tokens, a.queries)
    q = (k[pick] * 0.6 + rng.standard_normal((a.queries, a.heads, a.dim)) * 0.4).astype(np.float32)
    q *= 1.4
    out_ref, p_ref = attention(q, k, v)
    ent = float(-(p_ref * np.log(p_ref + 1e-30)).sum(axis=-1).mean())

    print(f"profile={a.profile}  tokens={a.tokens}  kv_heads={a.heads}  head_dim={a.dim}  "
          f"queries={a.queries}  mean attention entropy={ent:.2f} nats")
    print(f"K stats: rms={np.sqrt((k**2).mean()):.3f} absmax={np.abs(k).max():.2f}   "
          f"V stats: rms={np.sqrt((v**2).mean()):.4f} absmax={np.abs(v).max():.3f} "
          f"|v|<2^-6 share={(np.abs(v) < 2**-6).mean()*100:.0f}%")
    print()
    hdr = f"{'format':7s} {'bits':>5s} | {'K relRMS':>9s} {'V relRMS':>9s} | {'attn-out relRMS':>15s} {'KL(p_ref||p)':>13s} {'top1 agree':>10s}"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for name, fn in FORMATS.items():
        kq, vq = fn(k), fn(v)
        out, p = attention(q, kq, vq)
        top1 = float((p.argmax(-1) == p_ref.argmax(-1)).mean())
        rows.append((name, BITS[name], rel_rms(k, kq), rel_rms(v, vq), rel_rms(out_ref, out), kl(p_ref, p), top1))
    for name, bits, ek, ev, eo, kld, top1 in rows:
        print(f"{name:7s} {bits:5.1f} | {ek:9.4f} {ev:9.4f} | {eo:15.4f} {kld:13.2e} {top1:10.3f}")
    print()
    # mixed formats the kit actually uses: K at 8 bits, V at 4
    for kn, vn in (("int8", "int4"), ("fp8", "nvfp4"), ("int8", "nvfp4")):
        kq, vq = FORMATS[kn](k), FORMATS[vn](v)
        out, p = attention(q, kq, vq)
        top1 = float((p.argmax(-1) == p_ref.argmax(-1)).mean())
        print(f"K={kn:5s} V={vn:5s} | attn-out relRMS {rel_rms(out_ref, out):.4f}  KL {kl(p_ref, p):.2e}  top1 {top1:.3f}")


if __name__ == "__main__":
    main()
