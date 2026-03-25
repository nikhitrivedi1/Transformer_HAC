from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CustomEncoder(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
            is_causal=is_causal,
        )[0]
        return self.dropout1(x)


def _rotate_half(x: Tensor) -> Tensor:
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
    """Apply RoPE to q, k. Shapes: q,k (B, H, N, D); cos,sin (1, 1, N, D//2)."""
    cos = torch.repeat_interleave(cos, 2, dim=-1)
    sin = torch.repeat_interleave(sin, 2, dim=-1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """RoPE frequencies for one attention head (head_dim must be even)."""

    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even, got {head_dim}")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        freqs = freqs.to(dtype)
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]
        return cos, sin


class RoPEMultiheadSelfAttention(nn.Module):
    """Multi-head self-attention with rotary position embeddings (batch_first)."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        batch_first: bool = True,
        max_seq_len: int = 2048,
        bias: bool = True,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} not divisible by num_heads {num_heads}")
        head_dim = embed_dim // num_heads
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires even head_dim, got {head_dim}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.batch_first = batch_first
        self.dropout = float(dropout)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.rope = RotaryEmbedding(head_dim, max_seq_len=max_seq_len)
        self.dropout_layer = nn.Dropout(dropout)
        # nn.TransformerEncoder probes these on encoder_layer.self_attn (nested tensor path)
        self._qkv_same_embed_dim = True
        self.in_proj_bias = self.q_proj.bias

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        average_attn_weights: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if not self.batch_first:
            raise ValueError("RoPEMultiheadSelfAttention only supports batch_first=True")
        if attn_mask is not None or is_causal:
            raise NotImplementedError("attn_mask / is_causal not supported for RoPE attention in PatchTST")

        x = query
        bsz, tgt_len, _ = x.shape

        q = self.q_proj(query).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(tgt_len, x.device, x.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if key_padding_mask is not None:
            # True = ignore key position (PyTorch MHA convention)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
        attn_out = self.out_proj(attn_out)

        out_weights: Optional[Tensor] = None
        if need_weights:
            out_weights = attn_weights.mean(dim=1) if average_attn_weights else attn_weights

        return attn_out, out_weights


class RoPE_TransformerEncoderLayer(nn.TransformerEncoderLayer):
    """TransformerEncoderLayer with RoPE inside self-attention; max_seq_len passed via keyword only."""

    def __init__(self, *args, max_seq_len: int = 512, **kwargs):
        super().__init__(*args, **kwargs)
        mha = self.self_attn
        self.self_attn = RoPEMultiheadSelfAttention(
            embed_dim=mha.embed_dim,
            num_heads=mha.num_heads,
            dropout=self.dropout1.p,
            batch_first=mha.batch_first,
            max_seq_len=max_seq_len,
            bias=getattr(mha, "in_proj_bias", None) is not None,
        )

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
            attn_mask=attn_mask,
            is_causal=is_causal,
        )[0]
        return self.dropout1(x)
