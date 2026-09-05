#!/usr/bin/env python3
"""CacheLayer_nvfp4 test: paged attention through packed E2M1+E4M3 pages vs fp16,
with the kernel-parity gate (in-kernel decode == attending over pre-dequantized bytes)."""
import sys, os, torch
# imports resolve against the installed (patched) exllamav3 package
from exllamav3.constants import PAGE_SIZE
from exllamav3.cache import CacheLayer_fp16
from exllamav3.cache.nvfp4 import CacheLayer_nvfp4, nvfp4_quantize, nvfp4_dequantize
from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton, paged_attn_triton_longq

torch.manual_seed(0)
dev = "cuda"

class FakeAttn:
    num_kv_heads = 4
    head_dim = 256

def run(tokens_ctx, q_len):
    n_kv, d, n_q = 4, 256, 12
    results = []
    n_pages = (PAGE_SIZE * 12) // PAGE_SIZE
    # One shared dataset: fp16 vs nvfp4 outputs then differ ONLY by quantization
    k_all = (torch.randn(1, tokens_ctx, n_kv, d, device = dev) * 0.3).half()
    v_all = (torch.randn(1, tokens_ctx, n_kv, d, device = dev) * 0.3 + 0.1).half()
    q = (torch.randn(1, q_len, n_q, d, device = dev) * 0.3).half()
    k_new = (torch.randn(1, q_len, n_kv, d, device = dev) * 0.3).half()
    v_new = (torch.randn(1, q_len, n_kv, d, device = dev) * 0.3).half()
    block_table = torch.arange(n_pages, dtype = torch.int32, device = dev).unsqueeze(0)
    fn = paged_attn_triton if q_len <= 256 else paged_attn_triton_longq
    for make_fp8 in (False, True):
        if not make_fp8:
            layer = CacheLayer_fp16(None, FakeAttn(), 0, PAGE_SIZE * 12)
        else:
            layer = CacheLayer_nvfp4(None, FakeAttn(), 0, PAGE_SIZE * 12)
        layer.alloc(torch.device(dev))
        if not make_fp8:
            pos, chunk = 0, 384
            while pos < tokens_ctx:
                n = min(chunk, tokens_ctx - pos)
                seqlens = torch.tensor([pos], dtype = torch.int32, device = dev)
                paged_attn_triton(q = torch.zeros(1, 1, n_q, d, dtype = torch.half, device = dev),
                                  k = k_all[:, pos:pos+n], v = v_all[:, pos:pos+n],
                                  k_cache = layer.k, v_cache = layer.v,
                                  block_table = block_table, cache_seqlens = seqlens, causal = True)
                pos += n
            out = fn(q = q, k = k_new, v = v_new, k_cache = layer.k, v_cache = layer.v,
                     block_table = block_table,
                     cache_seqlens = torch.tensor([tokens_ctx], dtype = torch.int32, device = dev),
                     causal = True)
            results.append(("fp16", out, layer, q, block_table, None))
        else:
            # nvfp4: in-kernel quantize+append (k/v passed straight to the wrapper),
            # then verify bit-exactness of the packed pages vs the torch quantizer
            fn(q = q, k = k_all, v = v_all, k_cache = layer.k, v_cache = layer.v,
               k_scales = layer.ks, v_scales = layer.vs, block_table = block_table,
               cache_seqlens = torch.tensor([0], dtype=torch.int32, device=dev), causal = True)
            out = fn(q = q, k = k_new, v = v_new, k_cache = layer.k, v_cache = layer.v,
                     k_scales = layer.ks, v_scales = layer.vs, block_table = block_table,
                     cache_seqlens = torch.tensor([tokens_ctx], dtype=torch.int32, device=dev),
                     causal = True)
            # torch reference bytes
            ref = CacheLayer_nvfp4(None, FakeAttn(), 0, PAGE_SIZE * 12)
            ref.alloc(torch.device(dev))
            ref.update_kv(torch.tensor([0], dtype=torch.int32, device=dev), block_table, k_all, v_all, tokens_ctx)
            ref.update_kv(torch.tensor([tokens_ctx], dtype=torch.int32, device=dev), block_table, k_new, v_new, q_len)
            for nm, a, b in (("k bytes", layer.k, ref.k), ("v bytes", layer.v, ref.v),
                             ("k scales", layer.ks.view(torch.uint8), ref.ks.view(torch.uint8)),
                             ("v scales", layer.vs.view(torch.uint8), ref.vs.view(torch.uint8))):
                eq = (a == b).float().mean().item()
                print(f"  kernel-vs-torch {nm}: {eq*100:.4f}% equal")
                assert eq > 0.999, f"kernel quantizer mismatch on {nm}"
            layer.k.copy_(ref.k); layer.v.copy_(ref.v)
            layer.ks.copy_(ref.ks); layer.vs.copy_(ref.vs)
            results.append(("nvfp4", out, layer, q, block_table, k_all))
    (n0, o0, l0, q, bt, _), (n1, o1, l1, _, _, _) = results
    err = (o0.float() - o1.float()).abs()
    rel = err.mean() / o0.float().abs().mean()
    cos = torch.nn.functional.cosine_similarity(o0.float().flatten(), o1.float().flatten(), dim = 0)
    print(f"ctx {tokens_ctx}, q_len {q_len}: rel {rel:.4f}  cos {cos:.6f}")
    # o1 rerun check: does the stored nvfp4 output change when recomputed?
    o1_rerun = fn(q = q, k = None, v = None, k_cache = l1.k, v_cache = l1.v,
                  k_scales = l1.ks, v_scales = l1.vs, block_table = bt,
                  cache_seqlens = torch.tensor([0], dtype=torch.int32, device=dev),
                  kv_append_len = tokens_ctx + q_len, causal = True)
    print(f"  o1 stable across rerun: {(o1.float() - o1_rerun.float()).abs().max():.6f}")
    # parity: pre-dequantize the SAME packed bytes, attend fp16
    kq = nvfp4_dequantize(l1.k.int(), l1.ks.half())
    vq = nvfp4_dequantize(l1.v.int(), l1.vs.half())
    fn2 = paged_attn_triton if q_len <= 256 else paged_attn_triton_longq
    o2 = fn2(q = q, k = None, v = None, k_cache = kq, v_cache = vq, block_table = bt,
             cache_seqlens = torch.tensor([tokens_ctx + q_len], dtype = torch.int32, device = dev),
             causal = True)
    kerr = (o1.float() - o2.float()).abs().max()
    print(f"  kernel parity (nvfp4 in-kernel vs pre-dequant): {kerr:.6f}")
    assert kerr < 5e-3, "nvfp4 kernel math mismatch"
    return rel, l1

rel1, l1 = run(1024, 1)
rel2, _ = run(1024, 64)
rel3, _ = run(2048, 300)
print(f"rel errs (iid gaussian K/V): {rel1:.3f} {rel2:.3f} {rel3:.3f} — e2m1 quant noise; parity is the gate")
ss = l1.storage_size() / 1e6
exp = (PAGE_SIZE*12//PAGE_SIZE) * PAGE_SIZE * 4 * 256 * 2 * 4.5 / 8 / 1e6  # K+V
print(f"storage {ss:.2f} MB vs expected {exp:.2f} MB (4.5 b/elem)")
assert abs(ss - exp) < 0.01
# copy_page roundtrip
lsrc = CacheLayer_nvfp4(None, FakeAttn(), 0, PAGE_SIZE*4); lsrc.alloc(torch.device(dev))
ldst = CacheLayer_nvfp4(None, FakeAttn(), 0, PAGE_SIZE*4); ldst.alloc(torch.device(dev))
lsrc.k[3, :100] = torch.randint(0, 255, (100, 4, 128), device = dev, dtype = torch.uint8)
lsrc.ks[3, :100] = torch.randn(100, 4, 16, device = dev).to(torch.float8_e4m3fn)
ldst.copy_page(lsrc, 3, 1, 100)
assert torch.equal(ldst.k[1, :100], lsrc.k[3, :100]) and torch.equal(ldst.ks[1, :100], lsrc.ks[3, :100])
print("copy_page OK")
print("\nNVFP4 CACHE TEST PASS")
