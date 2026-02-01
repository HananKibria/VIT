"""
DROID WORLD MODEL - Action-Conditioned Video Prediction for Robot Manipulation
================================================================================

Adapted from the Surgical World Model to work with the DROID dataset
(Distributed Robot Interaction Dataset) — 76k manipulation trajectories,
350 hours, Franka Panda 7-DOF, 15 Hz control, 3 stereo camera views.

Key Changes from Surgical Model:
- action_dim=7: 6D cartesian target (x,y,z,rx,ry,rz) + 1 gripper target
- state_dim=14: 7 joint positions + 6 cartesian position + 1 gripper position
- Temporal scale 1.0 (manipulation has faster, more varied motion than surgery)
- Task success / grasp classifier replaces sterility classifier
- Language-conditioned task head for DROID's natural language instructions

Architecture (unchanged core):
- Tubelet size (2, 16, 16): Groups 2 temporal frames into each tubelet patch
- 3D RoPE: Separate rotations for temporal (256 dims), height (384), width (384)
- Hybrid predictor: ViT spatial attention + Gated DeltaNet temporal memory
- Action/state interpolation: Linear interpolation to match tubelet temporal resolution
"""

import math
from typing import Optional, Tuple, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============================================================================
# HELPER FUNCTIONS FOR ROPE
# ============================================================================

def rotate_half(x):
    """Rotate half the hidden dims of the input for RoPE application."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    """Apply rotary position embedding with proper rotation."""
    return (x * cos) + (rotate_half(x) * sin)


# ============================================================================
# VIDEO EMBEDDING: 3D TUBELET PATCHIFICATION
# ============================================================================

class TubeletEmbedding(nn.Module):
    """
    Extract 3D volumetric patches (tubelets) from video.

    Tubelet size (2, 16, 16) means 2 temporal frames are grouped spatially.
    For 16 frames at 224x224: produces 8 temporal x 196 spatial = 1,568 tokens.
    """

    def __init__(self, img_size=224, num_frames=16, tubelet_size=(2, 16, 16),
                 in_channels=3, embed_dim=1024):
        super().__init__()
        t, h, w = tubelet_size
        self.tubelet_size = tubelet_size
        self.img_size = img_size
        self.num_frames = num_frames

        self.projection = nn.Conv3d(
            in_channels=in_channels, out_channels=embed_dim,
            kernel_size=tubelet_size, stride=tubelet_size, padding=0
        )

        self.num_temporal_patches = num_frames // t
        self.num_spatial_patches_h = img_size // h
        self.num_spatial_patches_w = img_size // w
        self.num_patches = (self.num_temporal_patches *
                            self.num_spatial_patches_h *
                            self.num_spatial_patches_w)

    def forward(self, video):
        """(B, C, T, H, W) -> (B, num_patches, embed_dim)"""
        x = self.projection(video)
        x = x.flatten(2).transpose(1, 2)
        return x

    def inflate_weights_from_2d(self, pretrained_2d_weights):
        """Inflate 2D conv weights to 3D using central frame initialization."""
        if pretrained_2d_weights.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {pretrained_2d_weights.dim()}D")
        out_ch, in_ch, h, w = pretrained_2d_weights.shape
        t = self.tubelet_size[0]
        center_idx = t // 2
        expected_h, expected_w = self.tubelet_size[1], self.tubelet_size[2]
        if h != expected_h or w != expected_w:
            raise ValueError(
                f"Spatial dimensions mismatch: expected ({expected_h},{expected_w}), got ({h},{w})")
        weights_3d = torch.zeros(out_ch, in_ch, t, h, w,
                                 device=pretrained_2d_weights.device,
                                 dtype=pretrained_2d_weights.dtype)
        weights_3d[:, :, center_idx, :, :] = pretrained_2d_weights
        self.projection.weight.data = weights_3d
        print(f"  Inflated 2D weights ({pretrained_2d_weights.shape}) -> 3D ({weights_3d.shape})")


# ============================================================================
# COMPLETE 3D ROPE IMPLEMENTATION
# ============================================================================

class VideoRoPE3D(nn.Module):
    """
    3D Rotary Position Embeddings for video transformers.

    For DROID manipulation data at 15 Hz we use temporal_scale=1.0 to
    capture faster, more varied motion patterns compared to surgical
    procedures (which used 0.6).
    """

    def __init__(self, embed_dim, temporal_dims=256, spatial_dims_h=384,
                 spatial_dims_w=384, base=10000, temporal_scale=1.0):
        super().__init__()
        self.d_t = temporal_dims
        self.d_h = spatial_dims_h
        self.d_w = spatial_dims_w
        self.temporal_scale = temporal_scale
        assert self.d_t + self.d_h + self.d_w == embed_dim

        inv_freq_t = base ** (-torch.arange(0, self.d_t, 2).float() / self.d_t)
        inv_freq_h = base ** (-torch.arange(0, self.d_h, 2).float() / self.d_h)
        inv_freq_w = base ** (-torch.arange(0, self.d_w, 2).float() / self.d_w)
        self.register_buffer('inv_freq_t', inv_freq_t)
        self.register_buffer('inv_freq_h', inv_freq_h)
        self.register_buffer('inv_freq_w', inv_freq_w)

    def forward(self, seq_len_t, seq_len_h, seq_len_w):
        device = self.inv_freq_t.device
        t_pos = torch.arange(seq_len_t, device=device).float() * self.temporal_scale
        h_pos = torch.arange(seq_len_h, device=device).float()
        w_pos = torch.arange(seq_len_w, device=device).float()

        freqs_t = torch.cat([torch.outer(t_pos, self.inv_freq_t)] * 2, dim=-1)
        freqs_h = torch.cat([torch.outer(h_pos, self.inv_freq_h)] * 2, dim=-1)
        freqs_w = torch.cat([torch.outer(w_pos, self.inv_freq_w)] * 2, dim=-1)

        return ((freqs_t.cos(), freqs_t.sin()),
                (freqs_h.cos(), freqs_h.sin()),
                (freqs_w.cos(), freqs_w.sin()))


class RoPEMultiheadAttention(nn.Module):
    """Multi-head attention with 3D RoPE applied BEFORE head splitting."""

    def __init__(self, embed_dim, num_heads, rope_dims=None, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        if rope_dims is None:
            self.rope_dims = {
                't_dim': embed_dim // 4,
                'h_dim': (embed_dim * 3) // 8,
                'w_dim': embed_dim - (embed_dim // 4) - ((embed_dim * 3) // 8)
            }
        else:
            self.rope_dims = rope_dims

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def apply_rope_to_qk(self, qk, rope_cache, token_pos):
        if rope_cache is None or token_pos is None:
            return qk
        (cos_t, sin_t), (cos_h, sin_h), (cos_w, sin_w) = rope_cache
        t_idx, h_idx, w_idx = token_pos['t_idx'], token_pos['h_idx'], token_pos['w_idx']
        d_t = self.rope_dims['t_dim']; d_h = self.rope_dims['h_dim']
        qk_t = apply_rotary_pos_emb(qk[..., :d_t], cos_t[t_idx], sin_t[t_idx])
        qk_h = apply_rotary_pos_emb(qk[..., d_t:d_t+d_h], cos_h[h_idx], sin_h[h_idx])
        qk_w = apply_rotary_pos_emb(qk[..., d_t+d_h:], cos_w[w_idx], sin_w[w_idx])
        return torch.cat([qk_t, qk_h, qk_w], dim=-1)

    def forward(self, x, rope_cache=None, token_positions=None, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.embed_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        if rope_cache is not None:
            q = self.apply_rope_to_qk(q, rope_cache, token_positions)
            k = self.apply_rope_to_qk(k, rope_cache, token_positions)
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn = attn.masked_fill(attn_mask, float('-inf'))
        attn = self.dropout(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)


# ============================================================================
# GATED DELTA NETWORK LAYER
# ============================================================================

class GatedDeltaOperator:
    @staticmethod
    def identity(B, H, D, device, dtype):
        I = torch.eye(D, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, H, D, D)
        Z = torch.zeros(B, H, D, D, device=device, dtype=dtype)
        return (I, Z)

    @staticmethod
    def combine(elem1, elem2):
        A1, B1 = elem1; A2, B2 = elem2
        return (torch.matmul(A2, A1), torch.matmul(A2, B1) + B2)


def parallel_associative_scan_gated_delta(elements, initial_state, device, dtype):
    n = len(elements)
    if n == 0:
        return []
    B, H, D, _ = elements[0][0].shape
    if n == 1:
        A, B_mat = elements[0]
        return [torch.matmul(initial_state, A) + B_mat]

    elements = list(elements)
    tree_depth = int(np.ceil(np.log2(n)))
    for d in range(tree_depth):
        step = 2 ** (d + 1)
        for i in range(0, n, step):
            r = i + step - 1; l = i + step // 2 - 1
            if r < n and l >= 0:
                elements[r] = GatedDeltaOperator.combine(elements[l], elements[r])

    identity = GatedDeltaOperator.identity(B, H, D, device, dtype)
    temp = [None] * n; temp[-1] = identity
    for d in range(tree_depth - 1, -1, -1):
        step = 2 ** (d + 1)
        for i in range(0, n, step):
            r = i + step - 1; l = i + step // 2 - 1
            if r < n and l >= 0:
                if temp[l] is None:
                    t_ = elements[l]; temp[l] = temp[r]
                    temp[r] = GatedDeltaOperator.combine(temp[r], t_)
                else:
                    t_ = elements[l]; elements[l] = temp[r]
                    temp[r] = GatedDeltaOperator.combine(temp[r], t_)

    states = []; cs = initial_state
    for A, B_mat in elements:
        cs = torch.matmul(cs, A) + B_mat; states.append(cs)
    return states


class GatedDeltaLayer(nn.Module):
    """Gated Delta Network layer with parallel, sequential, and chunkwise modes."""

    def __init__(self, hidden_size, num_heads=4, head_dim=128, chunk_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        total_dim = num_heads * head_dim

        self.q_proj = nn.Linear(hidden_size, total_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, total_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, total_dim * 2, bias=False)
        self.a_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.q_conv1d = nn.Conv1d(total_dim, total_dim, 4, padding=3, groups=total_dim)
        self.k_conv1d = nn.Conv1d(total_dim, total_dim, 4, padding=3, groups=total_dim)
        self.v_conv1d = nn.Conv1d(total_dim*2, total_dim*2, 4, padding=3, groups=total_dim*2)
        self.g_proj = nn.Linear(hidden_size, total_dim, bias=False)
        self.o_norm = nn.LayerNorm(total_dim, eps=1e-5)
        self.o_proj = nn.Linear(total_dim, hidden_size, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x, state=None, use_parallel=None):
        B, L, D = x.shape
        if use_parallel is None:
            use_parallel = self.training and L > self.chunk_size
        q = self.q_proj(x); k = self.k_proj(x); v = self.v_proj(x)
        q = self.q_conv1d(q.transpose(1,2))[...,:L].transpose(1,2)
        k = self.k_conv1d(k.transpose(1,2))[...,:L].transpose(1,2)
        v = self.v_conv1d(v.transpose(1,2))[...,:L].transpose(1,2)
        q = F.normalize(self.silu(q), p=2, dim=-1)
        k = F.normalize(self.silu(k), p=2, dim=-1)
        v = self.silu(v)
        alpha = torch.sigmoid(self.a_proj(x)); beta = torch.sigmoid(self.b_proj(x))
        q = q.view(B, L, self.num_heads, self.head_dim)
        k = k.view(B, L, self.num_heads, self.head_dim)
        v, v_gate = v.chunk(2, dim=-1)
        v = v.view(B, L, self.num_heads, self.head_dim)
        v_gate = v_gate.view(B, L, self.num_heads, self.head_dim)
        if use_parallel:
            output, new_state = self._parallel(q, k, v, alpha, beta, state)
        else:
            output, new_state = self._sequential(q, k, v, alpha, beta, state)
        output = (output * v_gate).reshape(B, L, -1)
        g = self.silu(self.g_proj(x))
        return self.o_proj(self.o_norm(output * g)), new_state

    def _sequential(self, q, k, v, alpha, beta, state):
        B, L, H, D = q.shape
        if state is None:
            state = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
        outputs = []
        I = torch.eye(D, device=q.device, dtype=q.dtype).unsqueeze(0).unsqueeze(0)
        for t in range(L):
            a_t = alpha[:,t].unsqueeze(-1).unsqueeze(-1)
            b_t = beta[:,t].unsqueeze(-1).unsqueeze(-1)
            k_outer = k[:,t].unsqueeze(-1) @ k[:,t].unsqueeze(-2)
            state = state @ (a_t * (I - b_t * k_outer)) + b_t * (v[:,t].unsqueeze(-1) @ k[:,t].unsqueeze(-2))
            outputs.append(torch.matmul(state, q[:,t].unsqueeze(-1)).squeeze(-1))
        return torch.stack(outputs, dim=1), state

    def _parallel(self, q, k, v, alpha, beta, state):
        B, L, H, D = q.shape
        if state is None:
            state = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
        if L > self.chunk_size * 4:
            return self._chunkwise(q, k, v, alpha, beta, state)
        I = torch.eye(D, device=q.device, dtype=q.dtype).unsqueeze(0).unsqueeze(0)
        elements = []
        for t in range(L):
            a_t = alpha[:,t].unsqueeze(-1).unsqueeze(-1)
            b_t = beta[:,t].unsqueeze(-1).unsqueeze(-1)
            k_outer = k[:,t].unsqueeze(-1) @ k[:,t].unsqueeze(-2)
            elements.append((a_t*(I - b_t*k_outer), b_t*(v[:,t].unsqueeze(-1) @ k[:,t].unsqueeze(-2))))
        states = parallel_associative_scan_gated_delta(elements, state, q.device, q.dtype)
        outputs = [torch.matmul(states[t], q[:,t].unsqueeze(-1)).squeeze(-1) for t in range(L)]
        return torch.stack(outputs, dim=1), states[-1]

    def _chunkwise(self, q, k, v, alpha, beta, state):
        B, L, H, D = q.shape; C = self.chunk_size
        pad = (C - L % C) % C
        if pad > 0:
            q = F.pad(q,(0,0,0,0,0,pad)); k = F.pad(k,(0,0,0,0,0,pad))
            v = F.pad(v,(0,0,0,0,0,pad))
            alpha = F.pad(alpha,(0,0,0,pad),value=1.0); beta = F.pad(beta,(0,0,0,pad),value=0.0)
        I = torch.eye(D, device=q.device, dtype=q.dtype).unsqueeze(0).unsqueeze(0)
        outputs = []; cs = state
        for ci in range(q.shape[1] // C):
            s, e = ci*C, (ci+1)*C
            elems = []
            for t in range(C):
                a_t = alpha[:,s+t].unsqueeze(-1).unsqueeze(-1)
                b_t = beta[:,s+t].unsqueeze(-1).unsqueeze(-1)
                ko = k[:,s+t].unsqueeze(-1) @ k[:,s+t].unsqueeze(-2)
                elems.append((a_t*(I-b_t*ko), b_t*(v[:,s+t].unsqueeze(-1)@k[:,s+t].unsqueeze(-2))))
            sts = parallel_associative_scan_gated_delta(elems, cs, q.device, q.dtype)
            for t in range(C):
                outputs.append(torch.matmul(sts[t], q[:,s+t].unsqueeze(-1)).squeeze(-1))
            cs = sts[-1]
        out = torch.stack(outputs, dim=1)
        return (out[:,:L] if pad else out), cs


# ============================================================================
# VISION TRANSFORMER WITH 3D RoPE
# ============================================================================

class VideoViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, rope_dims=None, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = RoPEMultiheadAttention(embed_dim, num_heads, rope_dims, dropout)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        h = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, h), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(h, embed_dim), nn.Dropout(dropout))

    def forward(self, x, rope_cache=None, token_positions=None, attn_mask=None):
        x = x + self.attn(self.norm1(x), rope_cache, token_positions, attn_mask)
        return x + self.mlp(self.norm2(x))


class VideoViTEncoder(nn.Module):
    """Video ViT encoder with tubelet embedding and 3D RoPE."""

    def __init__(self, img_size=224, num_frames=16, tubelet_size=(2,16,16),
                 in_channels=3, embed_dim=1024, depth=12, num_heads=12,
                 temporal_scale=1.0, use_grad_checkpoint=False):
        super().__init__()
        self.tubelet_embed = TubeletEmbedding(img_size, num_frames, tubelet_size, in_channels, embed_dim)
        rope_config = {'t_dim': 256, 'h_dim': 384, 'w_dim': 384}
        self.rope = VideoRoPE3D(embed_dim, 256, 384, 384, temporal_scale=temporal_scale)
        self.blocks = nn.ModuleList([
            VideoViTBlock(embed_dim, num_heads, rope_config, 4.0, 0.1)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.use_grad_checkpoint = use_grad_checkpoint

    def _create_token_positions(self, T, H, W, device):
        return {
            't_idx': torch.arange(T, device=device).repeat_interleave(H * W),
            'h_idx': torch.arange(H, device=device).repeat_interleave(W).repeat(T),
            'w_idx': torch.arange(W, device=device).repeat(T * H),
        }

    def forward(self, video):
        x = self.tubelet_embed(video)
        te = self.tubelet_embed
        rope_cache = self.rope(te.num_temporal_patches, te.num_spatial_patches_h, te.num_spatial_patches_w)
        tp = self._create_token_positions(te.num_temporal_patches, te.num_spatial_patches_h, te.num_spatial_patches_w, x.device)
        for blk in self.blocks:
            if self.use_grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, rope_cache, tp, use_reentrant=False)
            else:
                x = blk(x, rope_cache=rope_cache, token_positions=tp)
        return self.norm(x)


# ============================================================================
# ACTION AND STATE CONDITIONING WITH INTERPOLATION
# ============================================================================

class ActionStateEmbedding(nn.Module):
    """
    Embed action/state tokens with temporal RoPE and interpolation.

    DROID dimensions:
      action_dim=7  : cartesian target (6D) + gripper target (1D)
      state_dim=14  : joint_position (7) + cartesian_position (6) + gripper_position (1)
    """

    def __init__(self, action_dim=7, state_dim=14, hidden_dim=1024, temporal_rope_base=10000):
        super().__init__()
        self.action_dim = action_dim; self.state_dim = state_dim
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.d_t = hidden_dim // 4
        inv_freq_t = temporal_rope_base ** (-torch.arange(0, self.d_t, 2).float() / self.d_t)
        self.register_buffer('inv_freq_t', inv_freq_t)

    def interpolate_to_tubelet_resolution(self, x, target_length):
        if x.shape[1] == target_length:
            return x
        return F.interpolate(x.transpose(1,2), size=target_length, mode='linear', align_corners=True).transpose(1,2)

    def create_temporal_rope(self, seq_len, device, temporal_scale=1.0):
        t_pos = torch.arange(seq_len, device=device).float() * temporal_scale
        freqs = torch.cat([torch.outer(t_pos, self.inv_freq_t)]*2, dim=-1)
        return freqs.cos(), freqs.sin()

    def apply_temporal_rope(self, x, indices, cos_t, sin_t):
        squeezed = x.dim() == 2
        if squeezed: x = x.unsqueeze(1)
        x_t = apply_rotary_pos_emb(x[..., :self.d_t], cos_t[indices], sin_t[indices])
        result = torch.cat([x_t, x[..., self.d_t:]], dim=-1)
        return result.squeeze(1) if squeezed else result

    def forward(self, actions, states, target_temporal_length=None, timestep_indices=None):
        B, T, _ = actions.shape
        if target_temporal_length is not None and target_temporal_length != T:
            actions = self.interpolate_to_tubelet_resolution(actions, target_temporal_length)
            states = self.interpolate_to_tubelet_resolution(states, target_temporal_length)
            T = target_temporal_length
        action_emb = self.action_proj(actions); state_emb = self.state_proj(states)
        if timestep_indices is None:
            timestep_indices = torch.arange(T, device=actions.device)
        cos_t, sin_t = self.create_temporal_rope(T, actions.device)
        return (self.apply_temporal_rope(action_emb, timestep_indices, cos_t, sin_t),
                self.apply_temporal_rope(state_emb, timestep_indices, cos_t, sin_t))


# ============================================================================
# HYBRID PREDICTOR WITH BLOCK-CAUSAL ATTENTION
# ============================================================================

class HybridPredictorBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, use_deltanet=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.spatial_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.use_deltanet = use_deltanet
        if use_deltanet:
            self.temporal_layer = GatedDeltaLayer(hidden_dim, num_heads, hidden_dim // num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*4), nn.GELU(), nn.Linear(hidden_dim*4, hidden_dim))

    def forward(self, x, attn_mask=None, state=None):
        normed = self.norm1(x)
        attn_out, _ = self.spatial_attn(normed, normed, normed, attn_mask=attn_mask)
        x = x + attn_out
        new_state = state
        if self.use_deltanet:
            temp_out, new_state = self.temporal_layer(x, state)
            x = x + temp_out
        return x + self.mlp(self.norm2(x)), new_state


class BlockCausalMask:
    @staticmethod
    def create_mask(seq_len, tokens_per_timestep, device):
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        for t in range(seq_len // tokens_per_timestep):
            s = t * tokens_per_timestep; e = (t+1) * tokens_per_timestep
            mask[s:e, :e] = False
        return mask


# ============================================================================
# SPATIAL ATTENTION POOLING & AUTOREGRESSIVE PREDICTOR
# ============================================================================

class SpatialAttentionPooling(nn.Module):
    def __init__(self, hidden_dim, num_queries=4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim))
        self.attention = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, patch_tokens):
        B = patch_tokens.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        pooled, _ = self.attention(q, patch_tokens, patch_tokens)
        return self.norm(pooled).reshape(B, -1)


class AutoregressivePredictor(nn.Module):
    """Multi-step prediction with uncertainty for DROID action/state dims."""

    def __init__(self, predictor_dim, encoder_dim, num_patch_tokens,
                 action_dim=7, state_dim=14):
        super().__init__()
        self.predictor_dim = predictor_dim; self.encoder_dim = encoder_dim
        self.num_patch_tokens = num_patch_tokens
        self.action_dim = action_dim; self.state_dim = state_dim

        self.spatial_pooling = SpatialAttentionPooling(predictor_dim, 4)
        self.spatial_predictor = nn.Sequential(
            nn.LayerNorm(predictor_dim), nn.Linear(predictor_dim, predictor_dim*2),
            nn.GELU(), nn.Dropout(0.1), nn.Linear(predictor_dim*2, predictor_dim))
        self.to_encoder_features = nn.Sequential(
            nn.Linear(predictor_dim, encoder_dim*2), nn.GELU(), nn.Linear(encoder_dim*2, encoder_dim))

        pooled_dim = predictor_dim * 4
        self.action_predictor = nn.Sequential(
            nn.Linear(pooled_dim, predictor_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(predictor_dim, predictor_dim//2), nn.GELU(), nn.Linear(predictor_dim//2, action_dim))
        self.state_predictor = nn.Sequential(
            nn.Linear(pooled_dim, predictor_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(predictor_dim, predictor_dim//2), nn.GELU(), nn.Linear(predictor_dim//2, state_dim))
        self.action_uncertainty = nn.Sequential(
            nn.Linear(pooled_dim, predictor_dim//4), nn.GELU(), nn.Linear(predictor_dim//4, action_dim), nn.Softplus())
        self.state_uncertainty = nn.Sequential(
            nn.Linear(pooled_dim, predictor_dim//4), nn.GELU(), nn.Linear(predictor_dim//4, state_dim), nn.Softplus())

    def forward(self, x, tokens_per_timestep, predict_horizon=1,
                future_actions=None, future_states=None,
                action_state_embed=None, predictor_blocks=None, attn_mask=None):
        B = x.shape[0]; predictions = []; current_sequence = x
        block_states = [None]*len(predictor_blocks) if predictor_blocks else None

        for h in range(predict_horizon):
            patch_tokens = current_sequence[:, -tokens_per_timestep:][:, :self.num_patch_tokens]
            predicted_patches = self.spatial_predictor(patch_tokens)
            predicted_encoder_features = self.to_encoder_features(predicted_patches.mean(dim=1))
            spatial_context = self.spatial_pooling(patch_tokens)
            pred_action = self.action_predictor(spatial_context)
            pred_state = self.state_predictor(spatial_context)
            predictions.append({
                'features': predicted_encoder_features,
                'action': pred_action, 'state': pred_state,
                'action_uncertainty': self.action_uncertainty(spatial_context),
                'state_uncertainty': self.state_uncertainty(spatial_context),
            })

            if h < predict_horizon - 1:
                na = future_actions[:,h] if future_actions is not None and h < future_actions.shape[1] else pred_action
                ns = future_states[:,h] if future_states is not None and h < future_states.shape[1] else pred_state
                if action_state_embed is not None:
                    ae, se = action_state_embed(na.unsqueeze(1), ns.unsqueeze(1))
                    ae = ae.squeeze(1); se = se.squeeze(1)
                else:
                    ae = torch.zeros(B, self.predictor_dim, device=x.device)
                    se = torch.zeros(B, self.predictor_dim, device=x.device)
                new_ts = torch.cat([predicted_patches, ae.unsqueeze(1), se.unsqueeze(1)], dim=1)
                if predictor_blocks is not None:
                    old = current_sequence.shape[1]; new_len = old + tokens_per_timestep
                    if attn_mask is not None:
                        em = torch.ones(new_len, new_len, dtype=torch.bool, device=x.device)
                        em[:old,:old] = attn_mask; em[old:,:new_len] = False; attn_mask = em
                    current_sequence = torch.cat([current_sequence, new_ts], dim=1)
                    for i, blk in enumerate(predictor_blocks):
                        current_sequence, block_states[i] = blk(current_sequence, attn_mask, block_states[i])
                else:
                    current_sequence = torch.cat([current_sequence, new_ts], dim=1)
        return predictions


# ============================================================================
# DUAL TASK CLASSIFIER (replaces sterility classifier)
# ============================================================================

class DualTaskClassifier(nn.Module):
    """
    Dual task classification for DROID manipulation:
      1. Current observation classifier - is manipulation progressing well?
      2. Future prediction classifier  - will upcoming steps succeed?
    """

    def __init__(self, encoder_dim, predictor_dim, num_classes=2):
        super().__init__()
        self.current_classifier = nn.Sequential(
            nn.LayerNorm(encoder_dim), nn.Linear(encoder_dim, encoder_dim//2),
            nn.GELU(), nn.Dropout(0.5), nn.Linear(encoder_dim//2, num_classes))
        self.future_classifier = nn.Sequential(
            nn.LayerNorm(predictor_dim), nn.Linear(predictor_dim, predictor_dim//2),
            nn.GELU(), nn.Dropout(0.5), nn.Linear(predictor_dim//2, num_classes))

    def forward(self, current_features, future_features=None):
        cur = self.current_classifier(current_features)
        fut = self.future_classifier(future_features) if future_features is not None else None
        return cur, fut


# ============================================================================
# COMPLETE DROID WORLD MODEL
# ============================================================================

class DROIDActionConditionedWorldModel(nn.Module):
    """
    DROID World Model — Action-Conditioned Video Prediction for Manipulation.

    Franka Panda 7-DOF, Robotiq gripper, 3 cameras, 15 Hz, 76k trajectories.
    action_dim=7  (cartesian_target 6 + gripper 1)
    state_dim=14  (joint_position 7 + cartesian_position 6 + gripper_position 1)
    """

    def __init__(self, img_size=224, num_frames=16, tubelet_size=(2,16,16),
                 in_channels=3, encoder_dim=1024, encoder_depth=12, encoder_heads=12,
                 predictor_dim=1024, predictor_depth=12, predictor_heads=8,
                 action_dim=7, state_dim=14, num_task_classes=2,
                 temporal_scale=1.0, use_grad_checkpoint=False, pretrained_2d_weights=None):
        super().__init__()

        self.encoder = VideoViTEncoder(
            img_size, num_frames, tubelet_size, in_channels, encoder_dim,
            encoder_depth, encoder_heads, temporal_scale, use_grad_checkpoint)
        if pretrained_2d_weights is not None:
            self.encoder.tubelet_embed.inflate_weights_from_2d(pretrained_2d_weights)

        self.video_proj = nn.Linear(encoder_dim, predictor_dim)
        self.action_state_embed = ActionStateEmbedding(action_dim, state_dim, predictor_dim)

        self.predictor_blocks = nn.ModuleList([
            HybridPredictorBlock(predictor_dim, predictor_heads,
                                 use_deltanet=(i < predictor_depth*2//3))
            for i in range(predictor_depth)])
        self.predictor_norm = nn.LayerNorm(predictor_dim, eps=1e-6)
        self.predictor_to_encoder = nn.Linear(predictor_dim, encoder_dim)

        self.task_classifier = DualTaskClassifier(encoder_dim, predictor_dim, num_task_classes)

        self.num_frames = num_frames
        self.encoder_dim = encoder_dim; self.predictor_dim = predictor_dim
        self.action_dim = action_dim; self.state_dim = state_dim
        self.tokens_per_frame = self.encoder.tubelet_embed.num_patches // self.encoder.tubelet_embed.num_temporal_patches
        self.use_grad_checkpoint = use_grad_checkpoint

        self.autoregressive_predictor = AutoregressivePredictor(
            predictor_dim, encoder_dim, self.tokens_per_frame, action_dim, state_dim)

    def encode_video(self, video):
        return self.encoder(video)

    def interleave_tokens(self, video_features, action_emb, state_emb):
        B, T, N, D = video_features.shape
        tokens = []
        for t in range(T):
            tokens.append(video_features[:, t])
            tokens.append(action_emb[:, t].unsqueeze(1))
            tokens.append(state_emb[:, t].unsqueeze(1))
        return torch.cat(tokens, dim=1)

    def forward(self, video, actions, states, predict_horizon=1, encoder_frozen=True):
        """
        Args:
            video:   (B, C, T, H, W)
            actions: (B, T, action_dim)  — cartesian(6) + gripper(1)
            states:  (B, T, state_dim)   — joint(7) + cartesian(6) + gripper(1)
        """
        B = video.shape[0]
        if encoder_frozen:
            with torch.no_grad(): encoded = self.encoder(video)
        else:
            encoded = self.encoder(video)
        current_enc_feat = encoded.mean(dim=1)

        encoded = self.video_proj(encoded)
        ntp = self.encoder.tubelet_embed.num_temporal_patches
        tpf = self.tokens_per_frame
        encoded = encoded.view(B, ntp, tpf, -1)

        ti = torch.arange(ntp, device=actions.device)
        action_emb, state_emb = self.action_state_embed(
            actions, states, target_temporal_length=ntp, timestep_indices=ti)
        interleaved = self.interleave_tokens(encoded, action_emb, state_emb)

        tpt = tpf + 2
        mask = BlockCausalMask.create_mask(interleaved.shape[1], tpt, interleaved.device)
        x = interleaved; sl = [None]*len(self.predictor_blocks)
        for i, blk in enumerate(self.predictor_blocks):
            if self.use_grad_checkpoint and self.training:
                x, sl[i] = torch.utils.checkpoint.checkpoint(blk, x, mask, sl[i], use_reentrant=False)
            else:
                x, sl[i] = blk(x, attn_mask=mask, state=sl[i])
        x = self.predictor_norm(x)

        tf = []
        for t in range(ntp):
            s = t*tpt; tf.append(x[:, s:s+tpf].mean(dim=1))
        tf = torch.stack(tf, dim=1)
        tw = torch.softmax(torch.arange(ntp, device=x.device, dtype=torch.float32)/2.0, dim=0).view(1,-1,1)
        future_feat = (tf * tw).sum(dim=1)

        if predict_horizon > 0:
            fa = actions[:, ntp:ntp+predict_horizon] if actions.shape[1] > ntp else None
            fs = states[:, ntp:ntp+predict_horizon] if states.shape[1] > ntp else None
            predictions = self.autoregressive_predictor(
                x, tpt, predict_horizon, fa, fs, self.action_state_embed, self.predictor_blocks, mask)
        else:
            predictions = []

        cur_logits, fut_logits = self.task_classifier(current_enc_feat, future_feat)
        return predictions, cur_logits, fut_logits


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("DROID WORLD MODEL — Action-Conditioned Video Prediction")
    print("="*70)

    model = DROIDActionConditionedWorldModel(
        img_size=224, num_frames=16, encoder_dim=1024, encoder_depth=12,
        encoder_heads=12, predictor_dim=1024, predictor_depth=12, predictor_heads=8,
        action_dim=7, state_dim=14, num_task_classes=2, temporal_scale=1.0)

    print(f"\n  Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    B = 2
    vid = torch.randn(B,3,16,224,224)
    act = torch.randn(B,16,7)
    st  = torch.randn(B,16,14)
    with torch.no_grad():
        preds, cur, fut = model(vid, act, st, predict_horizon=4)
    print(f"  Predictions: {len(preds)} steps")
    print(f"  Current task logits: {cur.shape}")
    print(f"  Future task logits:  {fut.shape}")
    print(f"  Predicted action:    {preds[0]['action'].shape}")   # (B,7)
    print(f"  Predicted state:     {preds[0]['state'].shape}")    # (B,14)
    print(f"  Action uncertainty:  {preds[0]['action_uncertainty'].shape}")
    print("\n  ALL TESTS PASSED\n" + "="*70)
