# Adapted from ohayonguy/PMRF and crowsonkb/k-diffusion under the MIT License.
# See the project LICENSE file for the corresponding third-party sections.
# Source: https://github.com/ohayonguy/PMRF/blob/main/arch/hourglass/image_transformer_v2.py
# Upstream source: https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/models/image_transformer_v2.py
# Adjusted for this project needs.

import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from functools import lru_cache, reduce

from models.HDiT.create_arch import create_arch
from models.HDiT.hourglass.axial_rope import make_axial_pos
from models.HDiT.hourglass.image_transformer_v2 import AdaRMSNorm, apply_wd, AxialRoPE, zero_init, Linear, \
    scale_for_cosine_sim, TokenMerge, FourierFeatures, MappingNetwork, tag_module, \
    GlobalAttentionSpec, GlobalTransformerLayer, NeighborhoodAttentionSpec, \
    ShiftedWindowAttentionSpec, NoAttentionSpec, NoAttentionTransformerLayer, \
    NeighborhoodTransformerLayer, ShiftedWindowTransformerLayer, Level, TokenSplit, RMSNorm, \
    TokenSplitWithoutSkip, filter_params, downscale_pos, checkpoint  # , apply_rotary_emb_
from models.HDiT.hourglass.image_transformer_v2 import NeighborhoodSelfAttentionBlock as OriginalNeighborhoodSelfAttentionBlock
from models.HDiT.hourglass.image_transformer_v2 import SelfAttentionBlock as OriginalSelfAttentionBlock
from models.HDiT.hourglass import image_transformer_v2 as itv2
from models.HDiT.hourglass import flops
from jvp_utils.functional import sdpa_jvp
from torch.autograd import forward_ad



try:
    import natten
except ImportError:
    natten = None


# Rotary position embeddings

# @flags.compile_wrap
def apply_rotary_emb(x, theta, conj=False):
    out_dtype = x.dtype
    dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
    d = theta.shape[-1]
    assert d * 2 <= x.shape[-1]
    x1, x2, x3 = x[..., :d], x[..., d : d * 2], x[..., d * 2 :]
    x1, x2, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conj else sin
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    y1, y2 = y1.to(out_dtype), y2.to(out_dtype)
    return torch.cat((y1, y2, x3), dim=-1)


# @flags.compile_wrap
def _apply_rotary_emb_inplace(x, theta, conj):
    dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
    d = theta.shape[-1]
    assert d * 2 <= x.shape[-1]
    x1, x2 = x[..., :d], x[..., d : d * 2]
    x1_, x2_, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conj else sin
    y1 = x1_ * cos - x2_ * sin
    y2 = x2_ * cos + x1_ * sin
    x1.copy_(y1)
    x2.copy_(y2)


class ApplyRotaryEmbeddingInplace(torch.autograd.Function):
    @staticmethod
    def forward(x, theta, conj):
        _apply_rotary_emb_inplace(x, theta, conj=conj)
        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, theta, conj = inputs
        ctx.save_for_backward(theta)
        ctx.conj = conj

        # For forward-mode JVP (reliable):
        ctx.theta = theta  # stash directly (works in jvp)

    @staticmethod
    def backward(ctx, grad_output):
        theta, = ctx.saved_tensors
        _apply_rotary_emb_inplace(grad_output, theta, conj=not ctx.conj)
        return grad_output, None, None

    @staticmethod
    def jvp(ctx, t_x, t_theta, t_conj):
        if t_x is None:
            return None


        out_t = t_x.clone()
        theta = ctx.theta
        _apply_rotary_emb_inplace(out_t, theta, conj=ctx.conj)
        return out_t


def apply_rotary_emb_(x, theta):
    return ApplyRotaryEmbeddingInplace.apply(x, theta, False)



## Natten neighborhood attention with JVP-capable SDPA
def _unfold_2d_kv(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    x: (B*NH, D, H, W)
    returns: (B*NH, HW, K2, D) where K2 = kernel_size*kernel_size
    """
    pad = kernel_size // 2
    cols = F.unfold(
        x,
        kernel_size=(kernel_size, kernel_size),
        padding=pad,
        stride=1,
    )  # (B*NH, D*K2, HW)
    bnh, d_k2, hw = cols.shape
    d = x.shape[1]
    k2 = kernel_size * kernel_size
    cols = (
        cols.view(bnh, d, k2, hw)
        .permute(0, 3, 2, 1)
        .contiguous()
    )  # (B*NH, HW, K2, D)
    return cols


def na2d_sdpa_jvp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: int,
) -> torch.Tensor:
    """
    JVP-capable replacement for natten.functional.na2d.

    Shapes (matching your fused NA path):
      q,k,v: (B, H, W, NH, D)
    Returns:
      out:   (B, H, W, NH, D)

    NOTE: sdpa_jvp internally uses a softmax scale of 1/sqrt(D),
    while natten.na2d(..., scale=1.0) does not. To match natten's
    behavior, we pre-scale Q by sqrt(D).
    """
    B, H, W, NH, D = q.shape
    HW = H * W
    K2 = kernel_size * kernel_size

    # (B*NH, D, H, W)
    q_ = q.permute(0, 3, 4, 1, 2).reshape(B * NH, D, H, W)
    k_ = k.permute(0, 3, 4, 1, 2).reshape(B * NH, D, H, W)
    v_ = v.permute(0, 3, 4, 1, 2).reshape(B * NH, D, H, W)

    # (B*NH, HW, K2, D)
    k_win = _unfold_2d_kv(k_, kernel_size)
    v_win = _unfold_2d_kv(v_, kernel_size)

    # (B*NH, HW, 1, D)
    q_pos = (
        q_.view(B * NH, D, HW)
        .permute(0, 2, 1)
        .contiguous()
        .unsqueeze(2)
    )

    # SDPA shapes: (B', H', S, D) with H'=1
    # q: (B*NH*HW, 1, 1, D)
    # k,v: (B*NH*HW, 1, K2, D)
    q_sdpa = q_pos.view(B * NH * HW, 1, 1, D)
    k_sdpa = k_win.view(B * NH * HW, 1, K2, D)
    v_sdpa = v_win.view(B * NH * HW, 1, K2, D)

    # compensate for sdpa_jvp's implicit 1/sqrt(D) scale
    q_sdpa = q_sdpa * math.sqrt(D)

    if not torch.is_autocast_enabled():
        org_dtype = q_sdpa.dtype
        q_sdpa = q_sdpa.to(torch.bfloat16)
        k_sdpa = k_sdpa.to(torch.bfloat16)
        v_sdpa = v_sdpa.to(torch.bfloat16)
    out_sdpa = sdpa_jvp(q_sdpa, k_sdpa, v_sdpa)  # (B*NH*HW, 1, 1, D)
    if not torch.is_autocast_enabled():
        out_sdpa = out_sdpa.to(org_dtype)

        # reshape back to (B, H, W, NH, D)
    out = out_sdpa.view(B * NH, HW, D)
    out = out.permute(0, 2, 1).contiguous().view(B * NH, D, H, W)
    out = out.view(B, NH, D, H, W).permute(0, 3, 4, 1, 2).contiguous()
    return out



def na2d_sdpa(q, k, v, kernel_size: int):
    """
    Safe drop-in replacement:
      - Normal forward: uses torch SDPA (correct)
      - Forward-mode AD (JVP): uses sdpa_jvp (your Triton JVP kernel)
    Shapes:
      q,k,v: (B, H, W, NH, D)
    """
    B, H, W, NH, D = q.shape
    HW = H * W
    K2 = kernel_size * kernel_size

    # (B*NH, D, H, W)
    q_ = q.permute(0, 3, 4, 1, 2).reshape(B * NH, D, H, W)
    k_ = k.permute(0, 3, 4, 1, 2).reshape(B * NH, D, H, W)
    v_ = v.permute(0, 3, 4, 1, 2).reshape(B * NH, D, H, W)

    k_win = _unfold_2d_kv(k_, kernel_size)  # (B*NH, HW, K2, D)
    v_win = _unfold_2d_kv(v_, kernel_size)

    q_pos = q_.view(B * NH, D, HW).permute(0, 2, 1).contiguous().unsqueeze(2)  # (B*NH, HW, 1, D)

    # to (B', 1, S, D)
    q_sdpa = q_pos.view(B * NH * HW, 1, 1, D)
    k_sdpa = k_win.view(B * NH * HW, 1, K2, D)
    v_sdpa = v_win.view(B * NH * HW, 1, K2, D)

    if forward_ad._is_fwd_grad_enabled():
        # JVP mode: use your Triton JVP SDPA
        q_sdpa = q_sdpa * math.sqrt(D)   # cancel implicit 1/sqrt(D)
        if not torch.is_autocast_enabled():
            dtype = torch.bfloat16
            q_sdpa = q_sdpa.to(dtype)
            k_sdpa = k_sdpa.to(dtype)
            v_sdpa = v_sdpa.to(dtype)
        out = sdpa_jvp(q_sdpa, k_sdpa, v_sdpa)  # (B', 1, 1, D)
        if not torch.is_autocast_enabled():
            out = out.to(torch.float32)
    else:
        # Normal forward mode: use correct SDPA
        # torch SDPA uses 1/sqrt(D) internally; cancel it to match natten scale=1.0
        out = F.scaled_dot_product_attention(
            q_sdpa * math.sqrt(D),  # cancel 1/sqrt(D)
            k_sdpa,
            v_sdpa,
            dropout_p=0.0,
            is_causal=False,
        )  # (B', 1, 1, D)

    out = out.view(B * NH, HW, D)
    out = out.permute(0, 2, 1).contiguous().view(B * NH, D, H, W)
    out = out.view(B, NH, D, H, W).permute(0, 3, 4, 1, 2).contiguous()
    return out

class NeighborhoodSelfAttentionBlockJVPSupport(nn.Module):
    def __init__(
        self,
        d_model,
        d_head,
        cond_features,
        kernel_size,
        dropout=0.0,
    ):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.kernel_size = kernel_size

        self.norm = AdaRMSNorm(d_model, cond_features)
        self.qkv_proj = apply_wd(Linear(d_model, d_model * 3, bias=False))
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.pos_emb = AxialRoPE(d_head // 2, self.n_heads)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = apply_wd(zero_init(Linear(d_model, d_model, bias=False)))

    def extra_repr(self):
        return (
            f"d_head={self.d_head}, kernel_size={self.kernel_size}, "
            f"use_jvp_sdpa={self.use_jvp_sdpa}"
        )

    def forward(self, x, pos, cond, use_jvp_sdpa=False):
        if natten is None:
            raise ModuleNotFoundError("natten is required for neighborhood attention")

        skip = x
        x = self.norm(x, cond)
        qkv = self.qkv_proj(x)

        # Always use fused-layout branch, since natten.functional.na2d works
        q, k, v = rearrange(
            qkv,
            "n h w (t nh e) -> t n h w nh e",
            t=3,
            e=self.d_head,
        )

        # cosine-style scaling
        q, k = scale_for_cosine_sim(q, k, self.scale[:, None], 1e-6)

        # RoPE
        theta = self.pos_emb(pos)
        q = apply_rotary_emb_(q, theta)
        k = apply_rotary_emb_(k, theta)

        flops.op(flops.op_natten, q.shape, k.shape, v.shape, self.kernel_size)

        if use_jvp_sdpa:
            x = na2d_sdpa(q, k, v, kernel_size=self.kernel_size)
        else:
            try:
                x = natten.functional.na2d(q, k, v, self.kernel_size, scale=1.0)
            except RuntimeError as exc:
                # Some GPU / NATten builds do not ship a compatible fused FNA kernel.
                # Newer Torch/NATTEN combinations can also fail on storage-less
                # wrapper tensors. Fall back to SDPA for those runtime issues.
                msg = str(exc)
                if (
                    "compatible fused neighborhood attention kernel" not in msg
                    and "Cannot access data pointer of Tensor that doesn't have storage" not in msg
                ):
                    raise
                x = na2d_sdpa(q, k, v, kernel_size=self.kernel_size)

        x = rearrange(x, "n h w nh e -> n h w (nh e)")

        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip

def _sdpa_scale1_jvp(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Effective attention scale=1.0.
    - Normal forward: torch SDPA(scale=1.0)
    - JVP trace (DualTensor): sdpa_jvp with q pre-scaled by sqrt(D) to cancel implicit 1/sqrt(D).
    Shapes: (B, H, S, D)
    """
    D = q.shape[-1]

    # Normal training does not need the custom JVP kernel. Using torch SDPA here
    # keeps gradients in the standard autograd path and avoids NATTEN backward.
    if not forward_ad._is_fwd_grad_enabled():
        return F.scaled_dot_product_attention(
            q * math.sqrt(D),
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
        )

    q_sdpa = q * math.sqrt(D)
    if not torch.is_autocast_enabled():
        org_dtype = q_sdpa.dtype
        dtype = torch.bfloat16
        q_sdpa = q_sdpa.to(dtype)
        k = k.to(dtype)
        v = v.to(dtype)
    out = sdpa_jvp(q_sdpa, k, v)
    if not torch.is_autocast_enabled():
        out = out.to(org_dtype)
    return out

class SelfAttentionBlockJVP(OriginalSelfAttentionBlock):
    """
    Inherits your original SelfAttentionBlock and adds JVP SDPA support.

    Usage:
      block = SelfAttentionBlockJVP(...)
      y = block(x, pos, cond)  # normal forward => identical to original (non-flash path)
      y, dy = torch.func.jvp(lambda x: block(x, pos, cond, use_sdpa_jvp=True), (x,), (dx,))

    Notes:
      - When use_sdpa_jvp=True, we always take the non-flash path (flash packed QKV has no JVP support here).
      - In normal forward, we still compute correct outputs using torch SDPA(scale=1.0).
      - In JVP trace, we route to sdpa_jvp.
    """

    def forward(self, x, pos, cond, use_sdpa_jvp: bool = False):
        skip = x
        x = self.norm(x, cond)
        qkv = self.qkv_proj(x)

        pos = rearrange(pos, "... h w e -> ... (h w) e").to(qkv.dtype)
        theta = self.pos_emb(pos)

        # If not using JVP-SDPA, fall back to the original behavior (including flash path)
        if not use_sdpa_jvp:
            return super().forward(x=skip, pos=rearrange(pos, "... (h w) e -> ... h w e", h=skip.shape[-3], w=skip.shape[-2]), cond=cond)  # type: ignore

        # JVP-SDPA path: force non-flash branch with separate q,k,v
        q, k, v = rearrange(qkv, "n h w (t nh e) -> t n nh (h w) e", t=3, e=self.d_head)

        # cosine attention preprocessing
        q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None], 1e-6)

        # RoPE
        theta_m = theta.movedim(-2, -3)
        q = apply_rotary_emb_(q, theta_m)
        k = apply_rotary_emb_(k, theta_m)

        # JVP-aware SDPA (scale=1.0 effective)
        x = _sdpa_scale1_jvp(q, k, v)

        x = rearrange(
            x, "n nh (h w) e -> n h w (nh e)", h=skip.shape[-3], w=skip.shape[-2]
        )

        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip

# Tests
def test_neighborhood_attention_forward_close():
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float32

    B, H, W = 2, 16, 16
    d_model = 128
    d_head = 32
    cond_features = 64
    kernel_size = 7

    x = torch.randn(B, H, W, d_model, device=device, dtype=dtype)
    pos = torch.randn(B, H, W, 2, device=device, dtype=dtype)
    cond = torch.randn(B, cond_features, device=device, dtype=dtype)

    ref = OriginalNeighborhoodSelfAttentionBlock(
        d_model,
        d_head,
        cond_features,
        kernel_size,
    ).to(device, dtype)

    jvp = NeighborhoodSelfAttentionBlockJVPSupport(
        d_model,
        d_head,
        cond_features,
        kernel_size,
    ).to(device, dtype)

    ref.load_state_dict(jvp.state_dict())

    with torch.no_grad():
        y_ref = ref(x, pos, cond)
        y_jvp = jvp(x, pos, cond, use_jvp_sdpa=False)

    max_err = (y_ref - y_jvp).abs().max().item()
    mean_err = (y_ref - y_jvp).abs().mean().item()

    print("max error :", max_err)
    print("mean error:", mean_err)

    torch.testing.assert_close(
        y_ref,
        y_jvp,
        rtol=5e-3,
        atol=5e-3,
    )


def test_neighborhood_attention_jvp_runs():
    from torch.func import jvp

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float32

    B, H, W = 1, 8, 8
    d_model = 64
    d_head = 32
    cond_features = 32
    kernel_size = 5

    model = NeighborhoodSelfAttentionBlockJVPSupport(
        d_model,
        d_head,
        cond_features,
        kernel_size,
    ).to(device, dtype)

    x = torch.randn(B, H, W, d_model, device=device, dtype=dtype)
    pos = torch.randn(B, H, W, 2, device=device, dtype=dtype)
    cond = torch.randn(B, cond_features, device=device, dtype=dtype)

    dx = torch.randn_like(x)

    def f(x):
        return model(x, pos, cond, use_jvp_sdpa=True)

    y, dy = jvp(f, (x,), (dx,))

    assert torch.isfinite(y).all()
    assert torch.isfinite(dy).all()
    assert dy.norm() > 0

    print("✅ JVP through NeighborhoodSelfAttentionBlock works")


def test_neighborhood_attention_forward_close_v2():
    from torch.func import jvp

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float32

    B, H, W = 2, 16, 16
    d_model = 128
    d_head = 32
    cond_features = 64
    kernel_size = 7

    x = torch.randn(B, H, W, d_model, device=device, dtype=dtype)
    pos = torch.randn(B, H, W, 2, device=device, dtype=dtype)
    cond = torch.randn(B, cond_features, device=device, dtype=dtype)

    ref = OriginalNeighborhoodSelfAttentionBlock(
        d_model,
        d_head,
        cond_features,
        kernel_size,
    ).to(device, dtype)

    jvp_model = NeighborhoodSelfAttentionBlockJVPSupport(
        d_model,
        d_head,
        cond_features,
        kernel_size,
    ).to(device, dtype)

    ref.load_state_dict(jvp_model.state_dict())

    dx = torch.randn_like(x)

    def f(x):
        return jvp_model(x, pos, cond, use_jvp_sdpa=True)



    with torch.no_grad():
        y_ref = ref(x, pos, cond)
        y_jvp, dy = jvp(f, (x,), (dx,))

    max_err = (y_ref - y_jvp).abs().max().item()
    mean_err = (y_ref - y_jvp).abs().mean().item()

    print("max error :", max_err)
    print("mean error:", mean_err)

    torch.testing.assert_close(
        y_ref,
        y_jvp,
        rtol=5e-3,
        atol=5e-3,
    )


def test_self_attention_forward_close():
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float32

    B, H, W = 2, 16, 16
    d_model = 128
    d_head = 32
    cond_features = 64

    x = torch.randn(B, H, W, d_model, device=device, dtype=dtype)
    pos = torch.randn(B, H, W, 2, device=device, dtype=dtype)
    cond = torch.randn(B, cond_features, device=device, dtype=dtype)

    ref = OriginalSelfAttentionBlock(
        d_model,
        d_head,
        cond_features,
    ).to(device, dtype)

    jvp_model = SelfAttentionBlockJVP(
        d_model,
        d_head,
        cond_features,
    ).to(device, dtype)

    # Ensure identical params
    ref.load_state_dict(jvp_model.state_dict())

    with torch.no_grad():
        y_ref = ref(x, pos, cond)
        y_jvp = jvp_model(x, pos, cond, use_sdpa_jvp=False)


    max_err = (y_ref - y_jvp).abs().max().item()
    mean_err = (y_ref - y_jvp).abs().mean().item()

    print("max error :", max_err)
    print("mean error:", mean_err)

    torch.testing.assert_close(
        y_ref,
        y_jvp,
        rtol=5e-3,
        atol=5e-3,
    )


def test_self_attention_jvp_runs():
    from torch.func import jvp

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float32

    B, H, W = 1, 8, 8
    d_model = 64
    d_head = 32
    cond_features = 32

    model = SelfAttentionBlockJVP(
        d_model,
        d_head,
        cond_features,
    ).to(device, dtype)

    x = torch.randn(B, H, W, d_model, device=device, dtype=dtype)
    pos = torch.randn(B, H, W, 2, device=device, dtype=dtype)
    cond = torch.randn(B, cond_features, device=device, dtype=dtype)

    dx = torch.randn_like(x)

    def f(x):
        return model(x, pos, cond, use_sdpa_jvp=True)

    y, dy = jvp(f, (x,), (dx,))

    assert torch.isfinite(y).all()
    assert torch.isfinite(dy).all()
    assert dy.norm() > 0

    print("✅ JVP through SelfAttentionBlockJVP works")


def test_self_attention_forward_close_v2():
    from torch.func import jvp

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float32

    B, H, W = 2, 16, 16
    d_model = 128
    d_head = 32
    cond_features = 64

    x = torch.randn(B, H, W, d_model, device=device, dtype=dtype)
    pos = torch.randn(B, H, W, 2, device=device, dtype=dtype)
    cond = torch.randn(B, cond_features, device=device, dtype=dtype)

    ref = OriginalSelfAttentionBlock(
        d_model,
        d_head,
        cond_features,
    ).to(device, dtype)

    jvp_model = SelfAttentionBlockJVP(
        d_model,
        d_head,
        cond_features,
    ).to(device, dtype)

    ref.load_state_dict(jvp_model.state_dict())

    dx = torch.randn_like(x)

    def f(x):
        return jvp_model(x, pos, cond, use_sdpa_jvp=True)

    with torch.no_grad():
        y_ref = ref(x, pos, cond)
        y_jvp, dy = jvp(f, (x,), (dx,))

    max_err = (y_ref - y_jvp).abs().max().item()
    mean_err = (y_ref - y_jvp).abs().mean().item()

    print("max error :", max_err)
    print("mean error:", mean_err)

    torch.testing.assert_close(
        y_ref,
        y_jvp,
        rtol=5e-3,
        atol=5e-3,
    )


# JVP Models

class NeighborhoodTransformerLayerJVP(NeighborhoodTransformerLayer):
    def __init__(self, d_model, d_ff, d_head, cond_features, kernel_size, dropout=0.0):
        super().__init__(d_model, d_ff, d_head, cond_features, kernel_size, dropout=dropout)
        self.self_attn = NeighborhoodSelfAttentionBlockJVPSupport(d_model, d_head,
                                                                  cond_features, kernel_size, dropout=dropout)

    def forward(self, x, pos, cond, use_jvp_sdpa=False):
        x = checkpoint(self.self_attn, x, pos, cond, use_jvp_sdpa)
        x = checkpoint(self.ff, x, cond)
        return x

class GlobalTransformerLayerJVP(GlobalTransformerLayer):
    def __init__(self, d_model, d_ff, d_head, cond_features, dropout=0.0):
        super().__init__(d_model, d_ff, d_head, cond_features, dropout=dropout)
        self.self_attn = SelfAttentionBlockJVP(d_model, d_head, cond_features, dropout=dropout)


    def forward(self, x, pos, cond, use_jvp_sdpa=False):
        x = checkpoint(self.self_attn, x, pos, cond, use_jvp_sdpa)
        x = checkpoint(self.ff, x, cond)
        return x


class ImageTransformerDenoiserModelV2WithJVP(nn.Module):
    def __init__(self, levels, mapping, in_channels, out_channels, patch_size, num_classes=0, mapping_cond_dim=0, degradation_params_dim=None):
        super().__init__()
        self.num_classes = num_classes
        self.patch_in = TokenMerge(in_channels, levels[0].width, patch_size)
        self.mapping_width = mapping.width
        self.time_t_emb = FourierFeatures(1, mapping.width)
        self.time_t_in_proj = Linear(mapping.width, mapping.width, bias=False)
        self.time_s_emb = FourierFeatures(1, mapping.width)
        self.time_s_in_proj = Linear(mapping.width, mapping.width, bias=False)
        self.degradation_proj = Linear(degradation_params_dim, mapping.width, bias=False) if degradation_params_dim else None
        self.class_emb = nn.Embedding(num_classes, mapping.width) if num_classes else None
        self.mapping_cond_in_proj = Linear(mapping_cond_dim, mapping.width, bias=False) if mapping_cond_dim else None
        self.mapping = tag_module(MappingNetwork(mapping.depth, mapping.width, mapping.d_ff, dropout=mapping.dropout), "mapping")

        self.down_levels, self.up_levels = nn.ModuleList(), nn.ModuleList()
        for i, spec in enumerate(levels):
            if isinstance(spec.self_attn, GlobalAttentionSpec):
                layer_factory = lambda _: GlobalTransformerLayerJVP(spec.width, spec.d_ff, spec.self_attn.d_head, mapping.width, dropout=spec.dropout)
            elif isinstance(spec.self_attn, NeighborhoodAttentionSpec):
                layer_factory = lambda _: NeighborhoodTransformerLayerJVP(spec.width, spec.d_ff, spec.self_attn.d_head, mapping.width, spec.self_attn.kernel_size, dropout=spec.dropout)
            elif isinstance(spec.self_attn, ShiftedWindowAttentionSpec):
                layer_factory = lambda i: ShiftedWindowTransformerLayer(spec.width, spec.d_ff, spec.self_attn.d_head, mapping.width, spec.self_attn.window_size, i, dropout=spec.dropout)
            elif isinstance(spec.self_attn, NoAttentionSpec):
                layer_factory = lambda _: NoAttentionTransformerLayer(spec.width, spec.d_ff, mapping.width, dropout=spec.dropout)
            else:
                raise ValueError(f"unsupported self attention spec {spec.self_attn}")

            if i < len(levels) - 1:
                self.down_levels.append(Level([layer_factory(i) for i in range(spec.depth)]))
                self.up_levels.append(Level([layer_factory(i + spec.depth) for i in range(spec.depth)]))
            else:
                self.mid_level = Level([layer_factory(i) for i in range(spec.depth)])

        self.merges = nn.ModuleList([TokenMerge(spec_1.width, spec_2.width) for spec_1, spec_2 in zip(levels[:-1], levels[1:])])
        self.splits = nn.ModuleList([TokenSplit(spec_2.width, spec_1.width) for spec_1, spec_2 in zip(levels[:-1], levels[1:])])

        self.out_norm = RMSNorm(levels[0].width)
        self.patch_out = TokenSplitWithoutSkip(levels[0].width, out_channels, patch_size)
        nn.init.zeros_(self.patch_out.proj.weight)

    def param_groups(self, base_lr=5e-4, mapping_lr_scale=1 / 3):
        wd = filter_params(lambda tags: "wd" in tags and "mapping" not in tags, self)
        no_wd = filter_params(lambda tags: "wd" not in tags and "mapping" not in tags, self)
        mapping_wd = filter_params(lambda tags: "wd" in tags and "mapping" in tags, self)
        mapping_no_wd = filter_params(lambda tags: "wd" not in tags and "mapping" in tags, self)
        groups = [
            {"params": list(wd), "lr": base_lr},
            {"params": list(no_wd), "lr": base_lr, "weight_decay": 0.0},
            {"params": list(mapping_wd), "lr": base_lr * mapping_lr_scale},
            {"params": list(mapping_no_wd), "lr": base_lr * mapping_lr_scale, "weight_decay": 0.0}
        ]
        return groups

    def forward(self, x, t=None, s=None, class_cond=None, mapping_cond=None, degradation_params=None, use_sdpa_jvp=False):
        # Patching
        x = x.movedim(-3, -1)
        x = self.patch_in(x)
        pos = make_axial_pos(x.shape[-3], x.shape[-2], device=x.device).view(x.shape[-3], x.shape[-2], 2)

        # Mapping network
        if class_cond is None and self.class_emb is not None:
            raise ValueError("class_cond must be specified if num_classes > 0")
        if mapping_cond is None and self.mapping_cond_in_proj is not None:
            raise ValueError("mapping_cond must be specified if mapping_cond_dim > 0")

        if t is not None:
            t_emb = self.time_t_in_proj(self.time_t_emb(t[..., None]))
        else:
            t_emb = self.time_t_in_proj(torch.ones(1, 1, device=x.device, dtype=x.dtype).expand(x.shape[0], self.mapping_width))

        s = x.new_zeros([x.shape[0], 1]) if s is None else s
        s_emb = self.time_s_in_proj(self.time_s_emb(s[..., None]))
        class_emb = self.class_emb(class_cond) if self.class_emb is not None else 0
        mapping_emb = self.mapping_cond_in_proj(mapping_cond) if self.mapping_cond_in_proj is not None else 0
        degradation_emb = self.degradation_proj(degradation_params) if degradation_params is not None else 0
        cond = self.mapping(t_emb + s_emb + class_emb + mapping_emb + degradation_emb)

        # Hourglass transformer
        skips, poses = [], []
        for down_level, merge in zip(self.down_levels, self.merges):
            x = down_level(x, pos, cond, use_jvp_sdpa=use_sdpa_jvp)
            skips.append(x)
            poses.append(pos)
            x = merge(x)
            pos = downscale_pos(pos)

        x = self.mid_level(x, pos, cond, use_jvp_sdpa=use_sdpa_jvp)

        for up_level, split, skip, pos in reversed(list(zip(self.up_levels, self.splits, skips, poses))):
            x = split(x, skip)
            x = up_level(x, pos, cond, use_jvp_sdpa=use_sdpa_jvp)

        # Unpatching
        x = self.out_norm(x)
        x = self.patch_out(x)
        x = x.movedim(-1, -3)

        return x


def create_hdit_jvp_model(widths,
                      depths,
                      self_attns,
                      dropout_rate,
                      mapping_depth,
                      mapping_width,
                      mapping_dropout_rate,
                      in_channels,
                      out_channels,
                      patch_size
                      ):
    assert len(widths) == len(depths)
    assert len(widths) == len(self_attns)
    assert len(widths) == len(dropout_rate)
    mapping_d_ff = mapping_width * 3
    d_ffs = []
    for width in widths:
        d_ffs.append(width * 3)

    levels = []
    for depth, width, d_ff, self_attn, dropout in zip(depths, widths, d_ffs, self_attns, dropout_rate):
        if self_attn['type'] == 'global':
            self_attn = itv2.GlobalAttentionSpec(self_attn.get('d_head', 64))
        elif self_attn['type'] == 'neighborhood':
            self_attn = itv2.NeighborhoodAttentionSpec(self_attn.get('d_head', 64), self_attn.get('kernel_size', 7))
        elif self_attn['type'] == 'shifted-window':
            self_attn = itv2.ShiftedWindowAttentionSpec(self_attn.get('d_head', 64), self_attn['window_size'])
        elif self_attn['type'] == 'none':
            self_attn = itv2.NoAttentionSpec()
        else:
            raise ValueError(f'unsupported self attention type {self_attn["type"]}')
        levels.append(itv2.LevelSpec(depth, width, d_ff, self_attn, dropout))
    mapping = itv2.MappingSpec(mapping_depth, mapping_width, mapping_d_ff, mapping_dropout_rate)
    model = ImageTransformerDenoiserModelV2WithJVP(
        levels=levels,
        mapping=mapping,
        in_channels=in_channels,
        out_channels=out_channels,
        patch_size=patch_size,
        num_classes=0,
        mapping_cond_dim=0,
    )

    return model


def _hdit_jvp_config(
        in_channels,
        out_channels,
        widths,
        depths,
        mapping_width,
        mapping_depth=2,
        d_head=64,
        self_attns=None,
):
    if self_attns is None:
        self_attns = [
            {"type": "neighborhood", "d_head": d_head, "kernel_size": 7},
            {"type": "global", "d_head": d_head},
        ]
    return {
        'in_channels': in_channels,
        'out_channels': out_channels,
        'widths': widths,
        'depths': depths,
        'patch_size': [2, 2],
        'self_attns': self_attns,
        'mapping_depth': mapping_depth,
        'mapping_width': mapping_width,
        'dropout_rate': [0] * len(widths),
        'mapping_dropout_rate': 0.0
    }


def _hdit_jvp_family(in_channels, out_channels, suffix=""):
    return {
        f"XL2lat{suffix}": _hdit_jvp_config(in_channels, out_channels, [384, 768], [2, 11], 768),
        f"M2lat{suffix}": _hdit_jvp_config(in_channels, out_channels, [192, 384], [2, 6], 384),
        f"S2lat{suffix}": _hdit_jvp_config(in_channels, out_channels, [128, 256], [2, 4], 256),
    }

arch_configs = {
    'hdit-jvp': {
        **_hdit_jvp_family(16, 16, suffix="16"),
    }
}
arch_name_to_object = {
    'hdit-jvp': create_hdit_jvp_model,
}

def create_jvp_arch(arch, condition_channels=0):
    # arch should be, e.g., hdit-jvp_XL2lat16
    arch_name, arch_size = arch.split('_')
    arch_config = arch_configs[arch_name][arch_size].copy()
    arch_config['in_channels'] += condition_channels
    return arch_name_to_object[arch_name](**arch_config)


def test_hdit_model_forward_close():
    """Original vs JVP-enabled model: plain forward outputs should match (non-JVP path)."""
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32


    in_channels = 16


    # Build models (use your factory or direct constructors if different)
    model_ref = create_arch("hdit16_XL2lat").to(device, dtype)


    model_jvp = create_jvp_arch("hdit-jvp_XL2lat16").to(device, dtype)
    model_jvp.load_state_dict(model_ref.state_dict(), strict=False)

    # tiny image input
    B, H, W = 2, 32, 32
    x = torch.randn(B, in_channels, H, W, device=device, dtype=dtype)
    pos = torch.randn(B, device=device, dtype=dtype)


    # Force both into non-flash (so they follow the same numeric path)

    with torch.no_grad():
        y_ref = model_ref(x, pos)
        # call the JVP-enabled model with default/no JVP flags so plain forward is used
        y_jvp = model_jvp(x, pos)


    assert torch.isfinite(y_ref).all(), "reference has NaNs/Infs"
    assert torch.isfinite(y_jvp).all(), "jvp-enabled forward has NaNs/Infs"

    max_err = (y_ref - y_jvp).abs().max().item()
    mean_err = (y_ref - y_jvp).abs().mean().item()
    print("max error :", max_err)
    print("mean error:", mean_err)

    # assert_close with reasonable tolerances
    torch.testing.assert_close(y_ref, y_jvp, rtol=5e-3, atol=5e-3)


def test_hdit_model_jvp_runs():
    from torch.func import jvp
    """Check that forward-mode JVP runs on the JVP-enabled model end-to-end."""
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    in_channels = 16

    # Build JVP-enabled model and load weights from reference architecture
    model_ref = create_arch("hdit16_XL2lat").to(device, dtype)
    model = create_jvp_arch("hdit-jvp_XL2lat16").to(device, dtype)
    model.load_state_dict(model_ref.state_dict(), strict=False)

    B, H, W = 1, 16, 16
    x = torch.randn(B, in_channels, H, W, device=device, dtype=dtype)
    pos = torch.randn(B, device=device, dtype=dtype)
    dx = torch.randn_like(x)

    def f(x_):
        return model(x_, pos, use_jvp_sdpa=True)

    y, dy = jvp(f, (x,), (dx,))

    assert torch.isfinite(y).all(), "primal contains NaNs/Infs"
    assert torch.isfinite(dy).all(), "tangent contains NaNs/Infs"
    assert dy.norm().item() > 0, "JVP tangent is zero (unexpected for random dx)"

    print("✅ JVP through the full model works; dy norm:", dy.norm().item())


def test_hdit_model_forward_close_v2():
    from torch.func import jvp
    """Run the JVP-enabled model under jvp and compare primal to the original model forward."""
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    in_channels = 16

    model_ref = create_arch("hdit16_XL2lat").to(device, dtype)
    model_jvp = create_jvp_arch("hdit-jvp_XL2lat16").to(device, dtype)
    model_jvp.load_state_dict(model_ref.state_dict(), strict=False)

    B, H, W = 2, 32, 32
    x = torch.randn(B, in_channels, H, W, device=device, dtype=dtype)
    pos = torch.randn(B, device=device, dtype=dtype)
    dx = torch.randn_like(x)

    with torch.no_grad():
        y_ref = model_ref(x, pos)

    def f(x_):
        return model_jvp(x_, pos, use_jvp_sdpa=True)

    y_jvp, dy = jvp(f, (x,), (dx,))

    assert torch.isfinite(y_ref).all(), "reference has NaNs/Infs"
    assert torch.isfinite(y_jvp).all(), "jvp-enabled primal has NaNs/Infs"

    max_err = (y_ref - y_jvp).abs().max().item()
    mean_err = (y_ref - y_jvp).abs().mean().item()
    print("max error :", max_err)
    print("mean error:", mean_err)

    torch.testing.assert_close(y_ref, y_jvp, rtol=5e-3, atol=5e-3)



if __name__ == "__main__":


    # Test HDIT model JVP vs original
    test_hdit_model_forward_close()
    test_hdit_model_jvp_runs()
    test_hdit_model_forward_close_v2()

    quit()
