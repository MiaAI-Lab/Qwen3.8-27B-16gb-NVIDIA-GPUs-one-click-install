from __future__ import annotations
from typing_extensions import override
import torch
from ..constants import PAGE_SIZE
from .cache import CacheLayer
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..modules import Attention
    from ..model import Model, Config
import numpy as np

class CacheLayer_fp8(CacheLayer):
    """KV cache stored as raw FP8 E4M3 (the NVFP4 recipe's KV format), no scales.

    K/V are bounded post-projection for the architectures this targets (k post-k_norm,
    v post-hidden_norm etc.), so no per-group scales are needed; E4M3 covers the range
    directly. Reads take the Triton online-dequant path (k/v tiles are cast to fp16
    inside the attention kernels); the generic get_kv returns dequantized fp16 tensors
    for non-Triton backends and diagnostics.
    """

    def __init__(
        self,
        config: Config | None,
        attention: Attention,
        cache_id: int,
        max_num_tokens: int,
        **kwargs,   # accept k_bits/v_bits/compand_a for Cache(**layer_kwargs) compatibility
    ):
        super().__init__(config, attention, cache_id, max_num_tokens)

        assert max_num_tokens % PAGE_SIZE == 0, \
            f"max_num_tokens must be a multiple of {PAGE_SIZE}."

        self.shape = (
            (max_num_tokens // PAGE_SIZE, PAGE_SIZE, attention.num_kv_heads, attention.head_dim)
            if attention else None
        )
        self.k = None
        self.v = None
        self.device = None


    @override
    def alloc(self, device: torch.device):
        self.device = device
        self.k = torch.zeros(self.shape, dtype = torch.float8_e4m3fn, device = device) if self.shape else None
        self.v = torch.zeros(self.shape, dtype = torch.float8_e4m3fn, device = device) if self.shape else None


    @override
    def free(self):
        self.device = None
        self.k = None
        self.v = None


    @override
    def get_qkv(self):
        # Not a packed-int quant layer; the fp8 fast path is selected by layer type in the
        # attention dispatch, not through this interface
        return None


    def get_paged(self) -> tuple:
        """Raw paged fp8 tensors for the Triton fp8 attention path."""
        return self.k, self.v


    @override
    def get_kv(self, cache_seqlens: torch.Tensor, block_table: torch.Tensor, sliding_window: int = -1) -> tuple:
        return self.k, self.v


    @override
    def update_kv(
        self,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        length: int
    ):
        # Torch scatter-cast write. Used by the dispatch post-hook on the fallback path
        # (non-Triton backends attend over dequantized copies and never touch the fp8
        # pages); on the Triton fast path new K/V are appended in-kernel instead and this
        # is not called
        bsz = k.shape[0]
        for b in range(bsz):
            pos = int(cache_seqlens[b])
            rows = torch.arange(pos, pos + length, device = k.device)
            page_idx = rows // PAGE_SIZE
            page_off = rows - page_idx * PAGE_SIZE
            phys = block_table[b].index_select(0, page_idx)
            self.k[phys, page_off] = k[b, : length].to(torch.float8_e4m3fn)
            self.v[phys, page_off] = v[b, : length].to(torch.float8_e4m3fn)

    @override
    def update_kv_direct(
        self,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        length: int
    ):
        self.update_kv(cache_seqlens, block_table, k, v, length)


    @override
    def copy_page(self, source: CacheLayer_fp8, from_page: int, to_page: int, num_tokens: int):
        assert self.shape == source.shape
        self.k[to_page, :num_tokens, :, :].copy_(source.k[from_page, :num_tokens, :, :], non_blocking = True)
        self.v[to_page, :num_tokens, :, :].copy_(source.v[from_page, :num_tokens, :, :], non_blocking = True)


    @override
    def get_tensors(self):
        return [self.k, self.v]


    @override
    def storage_size(self):
        return 2 * np.prod(self.shape) * 1


    @override
    def overhead_size(self):
        return 0


    @override
    def tp_export(self, plan):
        return {
            "cls": CacheLayer_fp8,
            "args": {
                "cache_id": self.cache_id,
                "max_num_tokens": self.max_num_tokens
            }
        }
