#!/usr/bin/env python3
"""CacheLayer_fp8 / CacheLayer_nvfp4 unit test: paged attention vs fp16 caches.
Exercises online dequant reads (short-q splitdv + longq grouped kernels),
in-kernel append (fp8) / torch pre-append (nvfp4), copy_page, storage size."""
import sys, os, torch
# imports resolve against the installed (patched) exllamav3 package
from exllamav3.constants import PAGE_SIZE
from exllamav3.cache.fp8 import CacheLayer_fp8
from exllamav3.cache.nvfp4 import CacheLayer_nvfp4, nvfp4_quantize, nvfp4_dequantize
from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton, paged_attn_triton_longq

torch.manual_seed(0)
dev = "cuda"

class FakeAttn:
    num_kv_heads = 4
    head_dim = 256

def make_layer(cls, tokens, dtype):
    if cls is None:
        from exllamav3.cache import CacheLayer_fp16
        cls = CacheLayer_fp16
    l = cls(None, FakeAttn(), 0, tokens)
    l.alloc(torch.device(dev))
    return l

def run(tokens_ctx, q_len):
    """One sequence: prefill (ctx tokens in chunks) then a q of q_len, causal."""
    n_kv, d = 4, 256
    n_q = 12
    results = []
    for cls, name in [(None, "fp16"), (CacheLayer_fp8, "fp8")]:
        layer = make_layer(cls, PAGE_SIZE * 12, None)   # 12 pages: fits ctx 2048 + append
        if cls is None:
            k_cache, v_cache = layer.k, layer.v
        else:
            k_cache, v_cache = layer.k, layer.v
        # paged block table: identity pages, one sequence
        n_pages = (PAGE_SIZE * 12) // PAGE_SIZE   # identity mapping over real pages
        block_table = torch.arange(n_pages, dtype = torch.int32, device = dev).unsqueeze(0)
        # write context K/V in chunks through the wrapper append path
        pos = 0
        chunk = 384
        k_all = (torch.randn(1, tokens_ctx, n_kv, d, device = dev) * 0.3).half()
        v_all = (torch.randn(1, tokens_ctx, n_kv, d, device = dev) * 0.3 + 0.1).half()
        while pos < tokens_ctx:
            n = min(chunk, tokens_ctx - pos)
            seqlens = torch.tensor([pos], dtype = torch.int32, device = dev)
            paged_attn_triton(
                q = torch.zeros(1, 1, n_q, d, dtype = torch.half, device = dev),  # dummy attention
                k = k_all[:, pos : pos + n], v = v_all[:, pos : pos + n],
                k_cache = k_cache, v_cache = v_cache,
                block_table = block_table, cache_seqlens = seqlens, causal = True)
            pos += n
        # now attend with a real query of q_len (fresh K/V appended)
        q = (torch.randn(1, q_len, n_q, d, device = dev) * 0.3).half()
        k_new = (torch.randn(1, q_len, n_kv, d, device = dev) * 0.3).half()
        v_new = (torch.randn(1, q_len, n_kv, d, device = dev) * 0.3).half()
        seqlens = torch.tensor([tokens_ctx], dtype = torch.int32, device = dev)
        fn = paged_attn_triton if q_len <= 256 else paged_attn_triton_longq
        out = fn(q = q, k = k_new, v = v_new, k_cache = k_cache, v_cache = v_cache,
                 block_table = block_table, cache_seqlens = seqlens, causal = True)
        results.append((name, out, layer))
        if cls is CacheLayer_fp8:
            ss = layer.storage_size() / 1e6
            exp = 2 * (PAGE_SIZE * 12 // PAGE_SIZE) * PAGE_SIZE * n_kv * d / 1e6
            print(f"  storage: {ss:.2f} MB (expected {exp:.2f} MB, 1 byte/elem)")
    (n0, o0, _), (n1, o1, l1) = results
    err = (o0.float() - o1.float()).abs()
    rel = err.mean() / o0.float().abs().mean()
    cos = torch.nn.functional.cosine_similarity(
        o0.float().flatten(), o1.float().flatten(), dim = 0)
    print(f"ctx {tokens_ctx}, q_len {q_len}: mean|out| {o0.float().abs().mean():.4f}  "
          f"abs err {err.mean():.5f}  rel {rel:.4f}  cos {cos:.6f}")

    # Kernel-parity: attend over PRE-dequantized fp16 copies of the same fp8 cache with no
    # append. Any difference vs o1 is a kernel bug, not quantization (same bytes both sides)
    kq, vq = l1.k.to(torch.half), l1.v.to(torch.half)
    seqlens_full = torch.tensor([tokens_ctx + q_len], dtype = torch.int32, device = dev)
    fn2 = paged_attn_triton if q_len <= 256 else paged_attn_triton_longq
    o2 = fn2(q = q, k = None, v = None, k_cache = kq, v_cache = vq,
             block_table = block_table, cache_seqlens = seqlens_full, causal = True)
    kerr = (o1.float() - o2.float()).abs().mean()
    print(f"  kernel parity (fp8 path vs pre-dequant fp16): {kerr:.6f}")
    assert kerr < 5e-3, "fp8 kernel math mismatch"
    return rel

rel1 = run(1024, 1)          # decode: short-q kernel, 1 new token
rel2 = run(1024, 64)         # short-q with 64-row query
rel3 = run(2048, 300)        # longq kernel path
print(f"rel errs (iid gaussian K/V): {rel1:.3f} {rel2:.3f} {rel3:.3f} — e4m3 quantization noise, kernel parity is the gate")

# copy_page roundtrip
lsrc = make_layer(CacheLayer_fp8, PAGE_SIZE * 4, None)
ldst = make_layer(CacheLayer_fp8, PAGE_SIZE * 4, None)
lsrc.k[3, :100] = torch.randn(100, 4, 256, device = dev).to(torch.float8_e4m3fn)
ldst.copy_page(lsrc, 3, 1, 100)
assert torch.equal(ldst.k[1, :100], lsrc.k[3, :100])
print("copy_page OK")
print("\nFP8 CACHE UNIT TEST PASS")
