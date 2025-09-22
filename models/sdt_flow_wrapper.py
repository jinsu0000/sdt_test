
import torch, torch.nn as nn, torch.nn.functional as F
from models.flow_policy import FlowPolicy, ActionEmbed, build_block_mask

class SDT_FlowWrapper(nn.Module):
    """
    Wrap SDT_Generator:
      - Writer/Glyph: Context prefix (freeze, NCE memory preserved)
      - Content: either prefix with [W,G,Cx4] (condition_mode='prefix'),
                 or separate tokens used by Action->Content cross-attn (condition_mode='xattn')
      - Flow Matching with sliding windows + padding mask + sigma-matched noise + smoothing
    """
    def __init__(self, sdt_model, H:int=64, n_layers:int=6, n_head:int=8, ffn_mult:int=4,
                 p:float=0.1, condition_mode:str="prefix"):
        super().__init__()
        assert condition_mode in ("prefix", "xattn")
        self.sdt = sdt_model
        self.H = H
        self.d_model = 512
        self.condition_mode = condition_mode

        self.action_embed = ActionEmbed(d_action=2, d_model=self.d_model)
        self.policy = FlowPolicy(d_ctx=self.d_model, d_act=self.d_model,
                                 n_layers=n_layers, n_head=n_head, ffn_mult=ffn_mult, p=p)

        # style projections (freeze to keep NCE memory intact)
        self.proj_style = nn.Linear(self.d_model, self.d_model, bias=False)
        self.proj_glyph = nn.Linear(self.d_model, self.d_model, bias=False)
        self.proj_style.weight.requires_grad_(False)
        self.proj_glyph.weight.requires_grad_(False)

        # content projection (trainable; routes to prefix or xattn depending on mode)
        self.proj_content = nn.Linear(self.d_model, self.d_model, bias=False)

        self.ctx_type_embed = nn.Embedding(4, self.d_model)  # 0:Writer 1:Glyph 2:Content 3:Action
        self.lambda_smooth = 0.1

    # ---- Encoders ----
    def _encode_style_ctx(self, style_imgs, keep_tokens: int = 0):
        """
        style_imgs: [B, 2N, 1, H, W]
        반환: [B, Lctx, 512]  (Lctx = writer_tokens + glyph_tokens)
        keep_tokens=0이면 4*N 전부 사용, >0이면 그 길이로 다운샘플(AdaptiveAvgPool1d)
        """
        device = next(self.parameters()).device
        style_imgs = style_imgs.to(device, non_blocking=True)
        with torch.no_grad():
            B, num_imgs, C, H, W = style_imgs.shape
            x = style_imgs.view(-1, C, H, W)                         # [B*2N, 1, H, W]
            feat = self.sdt.Feat_Encoder(x)                          # [B*2N, 512, 2, 2]
            feat = feat.view(B*num_imgs, 512, -1).permute(2, 0, 1)   # [4, B*2N, 512]
            feat = self.sdt.add_position(feat)
            mem  = self.sdt.base_encoder(feat)                       # [4, B*2N, 512]
            wmem = self.sdt.writer_head(mem)                         # [4, B*2N, 512]
            gmem = self.sdt.glyph_head(mem)                          # [4, B*2N, 512]
            # [4, B*2N, 512] -> [4, 2, B, N, 512] -> [4*N, 2, B, 512]
            N = num_imgs // 2
            wmem = wmem.view(4, B, 2, N, 512).permute(0, 3, 2, 1, 4).reshape(4*N, 2, B, 512)
            gmem = gmem.view(4, B, 2, N, 512).permute(0, 3, 2, 1, 4).reshape(4*N, 2, B, 512)
            # 앵커/포지티브 2개 축을 평균하여 [4*N, B, 512]
            wtok = wmem.mean(dim=2)  # [4*N, B, 512]
            gtok = gmem.mean(dim=2)  # [4*N, B, 512]
            # [4*N, B, 512] -> [B, 4*N, 512]
            wtok = wtok.permute(1, 0, 2).contiguous()
            gtok = gtok.permute(1, 0, 2).contiguous()

        # (선택) 길이 다운샘플
        if keep_tokens and wtok.size(1) != keep_tokens:
            # AdaptiveAvgPool1d는 [B, C, L]을 받으므로 전치해서 처리
            wtok_ = torch.nn.functional.adaptive_avg_pool1d(wtok.transpose(1, 2), keep_tokens).transpose(1, 2)
            gtok_ = torch.nn.functional.adaptive_avg_pool1d(gtok.transpose(1, 2), keep_tokens).transpose(1, 2)
            wtok, gtok = wtok_, gtok_

        # 프로젝션(동결 유지) + 타입 임베딩
        wtok = self.proj_style(wtok) + self.ctx_type_embed.weight[0][None, None, :]  # 0:Writer
        gtok = self.proj_glyph(gtok) + self.ctx_type_embed.weight[1][None, None, :]  # 1:Glyph
        ctx  = torch.cat([wtok, gtok], dim=1)  # [B, Lctx, 512]
        return ctx


    def _encode_content_tokens(self, char_img):
        """
        SDT Content_TR 출력: [4, B, 512]
        → 우리가 쓰려는 콘텐츠 토큰: [B, 4, 512]
        """
        device = next(self.parameters()).device
        char_img = char_img.to(device, non_blocking=True)

        with torch.no_grad():
            cont = self.sdt.content_encoder(char_img)  # [4, B, 512] (로그 기준)
            if cont.dim() != 3 or cont.shape[0] != 4:
                # 방어적 처리: 다른 변형이 들어와도 안전하게 4토큰 뽑기
                # [B, C, H, W] → 2x2 풀링 → [B, 4, C]; [B, S, C] → 1D 풀링 → [B, 4, C]
                if cont.dim() == 4:  # [B, C, H, W]
                    B, C, H, W = cont.shape
                    cont = torch.nn.functional.adaptive_avg_pool2d(cont, (2, 2))   # [B, C, 2, 2]
                    cont = cont.permute(0, 2, 3, 1).reshape(B, 4, C)               # [B, 4, C]
                elif cont.dim() == 3:
                    # [S, B, C] 또는 [B, S, C] 케이스 통합
                    if cont.shape[0] != char_img.shape[0] and cont.shape[1] == char_img.shape[0]:
                        cont = cont.permute(1, 0, 2)  # [B, S, C]
                    # [B, S, C] → 1D 풀링으로 4 토큰
                    B, S, C = cont.shape
                    if S == 4:
                        pass  # [B, 4, C]
                    else:
                        cont_ = cont.transpose(1, 2)                               # [B, C, S]
                        cont_ = torch.nn.functional.adaptive_avg_pool1d(cont_, 4)  # [B, C, 4]
                        cont  = cont_.transpose(1, 2)                               # [B, 4, C]
                else:  # [B, C] 등
                    B, C = cont.shape
                    cont = cont.unsqueeze(1).repeat(1, 4, 1)  # [B, 4, C]

                # 최종 채널을 512로 보정
                if cont.shape[-1] != 512:
                    proj_to_512 = torch.nn.Linear(cont.shape[-1], 512, bias=False).to(device)
                    cont = proj_to_512(cont)
            else:
                # 정상 경로: [4, B, 512] → [B, 4, 512]
                cont = cont.permute(1, 0, 2).contiguous()

        # [B, 4, 512] → proj + type embedding
        ctok = self.proj_content(cont)                                 # [B, 4, 512]
        ctok = ctok + self.ctx_type_embed.weight[2][None, None, :]     # Content type (idx=2)
        return ctok

    # ---- Context builders for two modes ----
    def _make_prefix_ctx(self, style_imgs, char_img):
        """[W,G,C1..C4] as prefix context tokens (SDT-like)."""
        ctx_s = self._encode_style_ctx(style_imgs, keep_tokens=keep_k)   # 예: keep_k=32 또는 60
        ctok  = self._encode_content_tokens(char_img)                    # [B, 4, 512]
        ctx   = torch.cat([ctx_s, ctok], dim=1)                          # [B, (2*keep_k or 120)+4, 512]
        Lctx  = ctx.size(1)

        return ctx, Lctx, None

    def _make_xattn_ctx(self, style_imgs, char_img):
        """[W,G] as prefix; content tokens for cross-attn."""
        ctx   = self._encode_style_ctx(style_imgs, keep_tokens=keep_k)   # [B, 2*keep_k, 512]
        Lctx  = ctx.size(1)
        ctok  = self._encode_content_tokens(char_img)                    # [B, 4, 512]  (xattn의 KV로 사용)

        return ctx, Lctx, ctok

    # ---- Flow Matching loss ----
    def flow_match_loss(self, style_imgs, coords, char_img, tau_beta=(2.0,4.0), stride=None):
        device = coords.device
        B, T, C = coords.shape
        H = self.H
        stride = stride or max(1, H//2)

        if self.condition_mode == "prefix":
            ctx, Lctx, content_tok = self._make_prefix_ctx(style_imgs, char_img)
        else:
            ctx, Lctx, content_tok = self._make_xattn_ctx(style_imgs, char_img)

        loss_flow = coords.new_tensor(0.0)
        loss_pen  = coords.new_tensor(0.0)
        n = 0

        for start in range(0, T, stride):
            end = min(T, start+H)
            A_gt = coords[:, start:end, :2]                      # [B,L<=H,2]
            Lcur = A_gt.size(1)
            pad = 0
            if Lcur < H:
                pad = H - Lcur
                A_gt = torch.cat([A_gt, torch.zeros(B, pad, 2, device=device)], dim=1)

            # valid mask
            mask = torch.ones(B, H, device=device)
            if pad > 0:
                mask[:, -pad:] = 0.0

            # sigma-matched noise & tau
            std = A_gt.reshape(-1, 2).std(dim=0, unbiased=False).clamp_min(1e-3).view(1,1,2)
            a,b = tau_beta
            tau = torch.distributions.Beta(a,b).sample((B,1)).to(device)
            eps = torch.randn_like(A_gt) * std
            A_tau = tau.view(B,1,1)*A_gt + (1-tau.view(B,1,1))*eps

            # action tokens (with type embedding)
            act_emb = self.action_embed(A_tau, tau, start_idx=start)
            act_emb = act_emb + self.ctx_type_embed.weight[3][None, None, :]

            # mask and forward
            attn_mask = build_block_mask(Lctx, H, device=device)
            v, pen_logits = self.policy(ctx, act_emb, content_tok, attn_mask)  # content_tok=None => prefix, else xattn

            # log용 - self-attn 가중치는 첫 블록에 있으니 policy.tr.blocks[0].last_self_attn에서 꺼냄
            blk0 = self.policy.tr.blocks[0]
            self._last_style_attn, self._last_content_attn = self._attn_stats(
                getattr(blk0, "last_self_attn", None),
                Lctx=Lctx, H=H,
                prefix_has_content=(self.condition_mode=="prefix")
            )

            # log용 - xattn 모드일 때는 cross-attn도 저장
            if self.condition_mode == "xattn":
                self._last_xattn_mean = getattr(self.policy.xattn, "last_xattn", None)
                if self._last_xattn_mean is not None:
                    self._last_xattn_mean = self._last_xattn_mean.mean().item()
            else:
                self._last_xattn_mean = None


            # flow loss (masked)
            u_star = (A_gt - eps)
            diff = (v - u_star).pow(2).sum(-1) * mask
            lf = diff.sum() / mask.sum().clamp_min(1.0)

            # smoothing loss (masked finite-diff)
            if H > 1:
                dv = (v[:,1:,:] - v[:,:-1,:]) * mask[:,1:].unsqueeze(-1)
                du = (u_star[:,1:,:] - u_star[:,:-1,:]) * mask[:,1:].unsqueeze(-1)
                ls = torch.mean((dv - du).pow(2))
            else:
                ls = torch.zeros((), device=device)

            loss_flow = loss_flow + (lf + self.lambda_smooth * ls)

            # pen CE (masked; assume label 0 is padding)
            if coords.size(-1) >= 5:
                lbl = coords[:, start:end, 2:].argmax(-1)
                if pad > 0:
                    lbl = torch.cat([lbl, torch.zeros(B, pad, dtype=lbl.dtype, device=device)], dim=1)
                lp = F.cross_entropy(pen_logits.reshape(-1, pen_logits.size(-1)), lbl.reshape(-1), ignore_index=0)
                loss_pen = loss_pen + lp
            n += 1

        loss_flow = loss_flow / max(n,1)
        loss_pen  = loss_pen  / max(n,1)
        return loss_flow, loss_pen

    # ---- Inference ----
    @torch.no_grad()
    def flow_infer(self, style_imgs, char_img, T:int, steps:int=10, stride:int=None):
        device = next(self.parameters()).device
        stride = stride or self.H

        if self.condition_mode == "prefix":
            ctx, Lctx, content_tok = self._make_prefix_ctx(style_imgs, char_img)
        else:
            ctx, Lctx, content_tok = self._make_xattn_ctx(style_imgs, char_img)

        B = ctx.size(0)
        out = []
        cur = 0
        while cur < T:
            H = min(self.H, T-cur)
            a = torch.randn(B, H, 2, device=device)
            attn_mask = build_block_mask(Lctx, H, device=device)
            for k in range(steps):
                tau_t = torch.full((B,1), float(k)/steps, device=device)
                emb = self.action_embed(a, tau_t, start_idx=cur)
                emb = emb + self.ctx_type_embed.weight[3][None, None, :]
                v, _ = self.policy(ctx, emb, content_tok, attn_mask)
                a = a + (1.0/steps) * v
            out.append(a)
            cur += stride
        return torch.cat(out, dim=1)[:, :T]


    # log용
    def _attn_stats(self, attn_w, Lctx:int, H:int, prefix_has_content: bool):
        """
        attn_w: [B, heads, L_tgt, L_src]  (self-attn 가중치)
        Lctx:   prefix 길이 (Writer/Glyph [+ Content×4 in prefix 모드])
        H:      action 윈도 길이
        prefix_has_content: prefix 모드일 때 True (컨텐트가 Lctx의 끝 4칸)
        반환: (style_attn_mean, content_attn_mean or None)
        """
        if attn_w is None:
            return None, None
        B, Hn, L_tgt, L_src = attn_w.shape
        # 액션 행: Lctx .. Lctx+H-1
        action_rows = attn_w[..., Lctx:Lctx+H, :]
        # 스타일 열: 0 .. (Lctx-1) 중 컨텐트 제외
        if prefix_has_content and Lctx >= 5:
            style_cols = action_rows[..., :Lctx-4]
            content_cols = action_rows[..., Lctx-4:Lctx]  # 뒤 4칸이 content
            style_mean = style_cols.mean().item() if style_cols.numel() else 0.0
            content_mean = content_cols.mean().item()
        else:
            style_mean = action_rows[..., :Lctx].mean().item()
            content_mean = None
        return style_mean, content_mean

