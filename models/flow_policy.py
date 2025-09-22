import torch
import torch.nn as nn
import torch.nn.functional as F

class ActionEmbed(nn.Module):
    def __init__(self, d_action:int=2, d_model:int=512, tau_dim:int=32, max_H:int=1024):
        super().__init__()
        self.pos = nn.Embedding(max_H, d_model)
        self.tau = nn.Sequential(nn.Linear(1, tau_dim), nn.SiLU(), nn.Linear(tau_dim, tau_dim))
        self.mlp = nn.Sequential(
            nn.Linear(d_action + tau_dim, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
    def forward(self, a_tau:torch.Tensor, tau:torch.Tensor, start_idx:int=0):
        # a_tau: [B,H,2], tau: [B,1] or [B,H,1]
        if tau.dim()==2:
            tau = tau.unsqueeze(1).expand(a_tau.size(0), a_tau.size(1), 1)
        t = self.tau(tau)
        h = self.mlp(torch.cat([a_tau, t], dim=-1))
        B, H, _ = h.shape
        idx = torch.arange(start_idx, start_idx+H, device=h.device).clamp_max(self.pos.num_embeddings-1)
        return h + self.pos(idx)[None, :, :]

def build_block_mask(Lctx:int, H:int, device=None):
    """
    Prefix-style causal mask:
      - context rows (0..Lctx-1) cannot look at future action columns (Lctx..Lctx+H-1)
      - action rows can look at all (ctx + previous actions), we keep it dense here.
    """
    L = Lctx + H
    M = torch.zeros(L, L, device=device)
    NEG = float("-inf")
    M[:Lctx, Lctx:] = NEG
    return M

class DualExpertBlock(nn.Module):
    """Shared self-attn; FFN split into Context-Expert and Action-Expert."""
    def __init__(self, d_ctx:int, d_act:int, n_head:int=8, ffn_mult:int=4, p:float=0.1):
        super().__init__()
        d = max(d_ctx, d_act)
        self.map_ctx = nn.Linear(d_ctx, d) if d_ctx!=d else nn.Identity()
        self.map_act = nn.Linear(d_act, d) if d_act!=d else nn.Identity()
        self.unmap_ctx = nn.Linear(d, d_ctx) if d_ctx!=d else nn.Identity()
        self.unmap_act = nn.Linear(d, d_act) if d_act!=d else nn.Identity()
        self.ln_attn = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_head, dropout=p, batch_first=True)
        self.drop = nn.Dropout(p)
        # Experts
        self.ln_ctx = nn.LayerNorm(d_ctx)
        self.ffn_ctx = nn.Sequential(nn.Linear(d_ctx, ffn_mult*d_ctx), nn.GELU(), nn.Linear(ffn_mult*d_ctx, d_ctx))
        self.ln_act = nn.LayerNorm(d_act)
        self.ffn_act = nn.Sequential(nn.Linear(d_act, ffn_mult*d_act), nn.GELU(), nn.Linear(ffn_mult*d_act, d_act))

    def forward(self, ctx, act, attn_mask=None, padmask=None):
        x = torch.cat([ self.map_ctx(ctx), self.map_act(act) ], dim=1)
        h = self.ln_attn(x)
        y, attn_w = self.attn(
            h, h, h,
            attn_mask=attn_mask,
            key_padding_mask=padmask,
            need_weights=True,
            average_attn_weights=False  # ← 헤드별 가중치
        )
        self.last_self_attn = attn_w.detach()  # [B, num_heads, L_tgt, L_src] # log용
        x = x + self.drop(y)
        Lctx = ctx.size(1)
        x_ctx, x_act = x[:, :Lctx, :], x[:, Lctx:, :]
        x_ctx = self.unmap_ctx(x_ctx) + self.drop(self.ffn_ctx(self.ln_ctx(self.unmap_ctx(x_ctx))))
        x_act = self.unmap_act(x_act) + self.drop(self.ffn_act(self.ln_act(self.unmap_act(x_act))))
        return x_ctx, x_act

class FlowTransformer(nn.Module):
    def __init__(self, d_ctx:int=512, d_act:int=512, n_layers:int=6, n_head:int=8, ffn_mult:int=4, p:float=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([DualExpertBlock(d_ctx, d_act, n_head, ffn_mult, p) for _ in range(n_layers)])
        self.ln_ctx = nn.LayerNorm(d_ctx)
        self.ln_act = nn.LayerNorm(d_act)
    def forward(self, ctx_tokens, act_tokens, attn_mask=None, padmask=None):
        c, a = ctx_tokens, act_tokens
        for blk in self.blocks:
            c, a = blk(c, a, attn_mask, padmask)
        return self.ln_ctx(c), self.ln_act(a)

class ActContentXAttn(nn.Module):
    """Q=Action, K/V=Content cross-attn with pre/post LN + residual."""
    def __init__(self, d_act:int=512, d_ctx:int=512, n_head:int=1, p:float=0.0):
        super().__init__()
        self.q_ln = nn.LayerNorm(d_act)
        self.kv_ln = nn.LayerNorm(d_ctx)
        self.attn = nn.MultiheadAttention(d_act, n_head, dropout=p, batch_first=True)
        self.drop = nn.Dropout(p)
    def forward(self, act_tokens, content_tokens):
        q = self.q_ln(act_tokens)
        kv = self.kv_ln(content_tokens)
        out, _ = self.attn(q, kv, kv)
        out, attn_w = self.attn(
            q, kv, kv,
            need_weights=True,
            average_attn_weights=False
        )
        self.last_xattn = attn_w.detach()  # [B, num_heads, L_act, L_content] # log용
        return act_tokens + self.drop(out)

class FlowPolicy(nn.Module):
    """
    If content_tokens is None => 'prefix' mode (no cross-attn)
    Else => 'xattn' mode (action->content cross-attn applied)
    """
    def __init__(self, d_ctx=512, d_act=512, n_layers=6, n_head=8, ffn_mult=4, p=0.1, out_pen=3):
        super().__init__()
        self.tr = FlowTransformer(d_ctx, d_act, n_layers, n_head, ffn_mult, p)
        self.xattn = ActContentXAttn(d_act=d_act, d_ctx=d_ctx, n_head=1, p=0.0)
        self.flow_head = nn.Linear(d_act, 2)
        self.pen_head  = nn.Linear(d_act, out_pen)

    def forward(self, ctx_tokens, act_tokens, content_tokens, attn_mask, padmask=None):
        _, a = self.tr(ctx_tokens, act_tokens, attn_mask, padmask)
        if content_tokens is not None:   # xattn mode
            a = self.xattn(a, content_tokens)
        v = self.flow_head(a)
        pen = self.pen_head(a)
        return v, pen
