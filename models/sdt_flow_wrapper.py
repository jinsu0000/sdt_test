import torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange
from models.flow_policy import FlowPolicy, ActionEmbed, build_block_mask

class SDT_FlowWrapper(nn.Module):
    """
    SDT를 Flow Matching + Action Chunking로 감싼 래퍼.
    - 스타일 토큰을 Content로 2단계 cross-attn 요약해 cond(1토큰) 생성
    - 학습: Δx,Δy (MSE) + pen (CrossEntropy)  [옵션B]
    - 추론: 청크(H) 단위 오일러 적분 + Temporal Ensembling + R-step 재계획
    - 출력 포맷: [B, T, 5] (dx, dy, one-hot(3))  ← test.py/evaluate.py와 100% 호환
    """
    def __init__(self, sdt_model,
                 H:int=6,          # action chunk length
                 n_layers:int=6, n_head:int=8, ffn_mult:int=4, p:float=0.1,
                 stride_default:int=3):
        super().__init__()
        self.sdt = sdt_model
        self.H = H
        self.stride_default = stride_default
        self.d_model = 512

        # 정책 네트워크 (ctx: [B,Lctx,512], act_emb: [B,H,512])
        self.action_embed = ActionEmbed(d_action=2, d_model=self.d_model)
        self.policy = FlowPolicy(d_ctx=self.d_model, d_act=self.d_model,
                                 n_layers=n_layers, n_head=n_head, ffn_mult=ffn_mult, p=p)
        # cond 요약용 cross-attn (content -> writer -> glyph)
        self.mha_w = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=8, batch_first=True)
        self.mha_g = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=8, batch_first=True)

        # 로그/디버그용
        self._last_style_attn = None
        self._last_content_attn = None

    # ---------- 스타일/컨텐트 인코딩 ----------
    @torch.no_grad()
    def _style_tokens(self, style_imgs):
        """
        SDT 인코더로 스타일 토큰 추출.
        return:
          wtok: [B, Lw(=4*N), 512]
          gtok: [B, Lg(=4*N), 512]
        """
        dev = style_imgs.device
        B, N, C, H, W = style_imgs.shape
        x = style_imgs.view(-1, C, H, W)                              # [B*N,1,H,W]
        feat = self.sdt.Feat_Encoder(x)                               # [B*N, 512, 2, 2]
        feat = feat.reshape(B*N, 512, -1).permute(2, 0, 1)            # [4, B*N, 512]
        feat = self.sdt.add_position(feat)
        mem  = self.sdt.base_encoder(feat)                            # [4, B*N, 512]
        wmem = self.sdt.writer_head(mem)                              # [4, B*N, 512]
        gmem = self.sdt.glyph_head(mem)                               # [4, B*N, 512]
        # → [4*N, B, 512] → [B, 4*N, 512]
        wtok = rearrange(wmem, 't (b n) c -> b (t n) c', b=B)         # [B, 4*N, 512]
        gtok = rearrange(gmem, 't (b n) c -> b (t n) c', b=B)         # [B, 4*N, 512]
        return wtok.contiguous(), gtok.contiguous()

    @torch.no_grad()
    def _content_token(self, char_img):
        """
        SDT Content_TR 출력 평균으로 1토큰 생성.
        return: ctok [B, 1, 512]
        """
        cont = self.sdt.content_encoder(char_img)   # [4, B, 512]
        cont = cont.mean(0, keepdim=False)          # [B, 512]
        return cont.unsqueeze(1)                    # [B, 1, 512]

    @torch.no_grad()
    def _make_cond_ctx(self, style_imgs, char_img):
        """
        Content(1) -> Writer -> Glyph 순으로 2단계 cross-attn하여 cond(1토큰) 생성.
        return:
          ctx  : [B, 1, 512]   (cond)
          Lctx : 1
        """
        wtok, gtok = self._style_tokens(style_imgs)     # [B,4N,512] each
        ctok = self._content_token(char_img)            # [B,1,512]
        # content -> writer
        mid, _ = self.mha_w(query=ctok, key=wtok, value=wtok, need_weights=False)
        # mid -> glyph
        cond, _ = self.mha_g(query=mid, key=gtok, value=gtok, need_weights=False)
        return cond.contiguous(), 1

    # ---------- 학습 로스 ----------
    def flow_match_loss(self, style_imgs, coords, char_img,
                        tau_beta=(2.0,4.0), stride:int=None):
        """
        coords: [B, T, 5]  (dx,dy, one-hot(3))
        Δx,Δy : MSE on windows(H), pen : CrossEntropy
        """
        device = coords.device
        B, T, C = coords.shape
        H = self.H
        stride = stride or self.stride_default

        ctx, Lctx = self._make_cond_ctx(style_imgs, char_img)   # [B,1,512], 1

        loss_flow = coords.new_tensor(0.0)
        loss_pen  = coords.new_tensor(0.0)
        n_blocks  = 0

        for start in range(0, T, stride):
            end = min(T, start + H)
            A_gt = coords[:, start:end, :2]              # [B, L<=H, 2]
            P_gt = coords[:, start:end, 2:]              # [B, L<=H, 3]
            Lcur = A_gt.size(1)

            # padding to H
            pad = H - Lcur
            if pad > 0:
                A_gt = torch.cat([A_gt, torch.zeros(B, pad, 2, device=device)], dim=1)
                P_gt = torch.cat([P_gt, torch.zeros(B, pad, 3, device=device)], dim=1)

            # 유효마스크
            mask = torch.ones(B, H, device=device)
            if pad > 0: mask[:, -pad:] = 0.0

            # τ 샘플 + 중간점 생성
            std = A_gt.reshape(-1, 2).std(dim=0, unbiased=False).clamp_min(1e-3).view(1,1,2)
            a,b = tau_beta
            tau = torch.distributions.Beta(a,b).sample((B,1)).to(device)
            eps = torch.randn_like(A_gt) * std
            A_tau = tau.view(B,1,1) * A_gt + (1 - tau.view(B,1,1)) * eps  # [B,H,2]

            # 액션 임베딩 + 블록 마스크
            act_emb = self.action_embed(A_tau, tau, start_idx=start)  # [B,H,512]
            attn_mask = build_block_mask(Lctx, H, device=device)      # [ (Lctx+H) x (Lctx+H) ]

            # 정책: v(Δx,Δy), pen_logits
            v, pen_logits = self.policy(ctx, act_emb, None, attn_mask)   # v:[B,H,2], pen_logits:[B,H,3]

            # FM 회귀 타깃
            u_star = (A_gt - eps)                                      # [B,H,2]

            # MSE(Δ) + time-smoothness
            diff = (v - u_star).pow(2).sum(-1) * mask                   # [B,H]
            lf = diff.sum() / mask.sum().clamp_min(1.0)

            if H > 1:
                dv = (v[:,1:,:] - v[:,:-1,:]) * mask[:,1:].unsqueeze(-1)
                du = (u_star[:,1:,:] - u_star[:,:-1,:]) * mask[:,1:].unsqueeze(-1)
                ls = torch.mean((dv - du).pow(2))
            else:
                ls = torch.zeros((), device=device)

            # pen CE
            tgt_lbl = P_gt.argmax(-1)                                  # [B,H]
            if pad > 0:
                # 패딩은 무시
                ignore = torch.full((B, pad), -100, device=device, dtype=torch.long)
                tgt_lbl = torch.cat([tgt_lbl[:, :Lcur], ignore], dim=1)
            lp = F.cross_entropy(pen_logits.reshape(-1,3), tgt_lbl.reshape(-1), ignore_index=-100)

            loss_flow += (lf + 0.1 * ls)
            loss_pen  += lp
            n_blocks  += 1

        loss_flow /= max(n_blocks, 1)
        loss_pen  /= max(n_blocks, 1)
        return loss_flow, loss_pen

    # ---------- 추론: 청크 + 앙상블 ----------
    @torch.no_grad()
    def flow_infer(self, style_imgs, char_img, T:int,
                   steps:int=20, stride:int=None, replan:int=None):
        """
        오일러 적분으로 Δx,Δy 생성 + pen 확률 → one-hot.
        Temporal ensembling(삼각가중), R-step 재계획.
        return: [B, T, 5]
        """
        device = next(self.parameters()).device
        B = style_imgs.size(0)
        H = self.H
        stride = stride if stride is not None else self.stride_default
        replan = replan if replan is not None else stride

        ctx, Lctx = self._make_cond_ctx(style_imgs, char_img)           # [B,1,512], 1

        # 누적 버퍼
        acc_xy  = torch.zeros(B, T, 2, device=device)
        acc_pen = torch.zeros(B, T, 3, device=device)
        acc_w   = torch.zeros(B, T, 1, device=device)

        def w_tri(h):  # 삼각 가중
            return torch.linspace(h, 1, steps=h, device=device) / max(h,1)

        t = 0
        executed = []                          # 확정 스텝 저장(리스트로 모았다가 마지막에 cat)
        eos_t = torch.full((B,), -1, device=device, dtype=torch.long)

        while t < T:
            h = min(H, T - t)

            # 청크 초기값(노이즈)
            a = torch.randn(B, h, 2, device=device)
            attn_mask = build_block_mask(Lctx, h, device=device)

            # 오일러 적분 (0→1)
            for k in range(steps):
                tau = torch.full((B,1), float(k)/steps, device=device)
                emb = self.action_embed(a, tau, start_idx=t)           # [B,h,512]
                v, _ = self.policy(ctx, emb, None, attn_mask)          # v:[B,h,2]
                a = a + (1.0/steps) * v

            # pen은 마지막 상태로 예측
            tau1 = torch.ones(B,1, device=device)
            emb1 = self.action_embed(a, tau1, start_idx=t)
            _, pen_logits = self.policy(ctx, emb1, None, attn_mask)    # [B,h,3]
            pen_prob = pen_logits.softmax(-1)                          # [B,h,3]

            # 시간축 앙상블 누적
            w = w_tri(h).view(1,h,1)                                   # [1,h,1]
            acc_xy[:,  t:t+h, :] += w * a
            acc_pen[:, t:t+h, :] += w * pen_prob
            acc_w[:,   t:t+h, :] += w

            # 앞 R 스텝 확정
            R = min(replan, T - t)
            wsum = acc_w[:, t:t+R, :].clamp_min(1e-8)
            xy   = acc_xy[:, t:t+R, :] / wsum                           # [B,R,2]
            pen  = acc_pen[:, t:t+R, :] / wsum                          # [B,R,3]
            pen_oh = F.one_hot(pen.argmax(-1), num_classes=3).to(pen).float()

            step_chunk = torch.cat([xy, pen_oh], dim=-1)                # [B,R,5]

            # EOS 처리: 이번 구간에 처음으로 EOS가 생기면 그 이후 0패딩 + EOS 유지
            for b in range(B):
                if eos_t[b] >= 0:
                    # 이미 EOS를 만난 샘플은 전부 0/EOS
                    step_chunk[b, :, :2] = 0.0
                    step_chunk[b, :, 2:] = torch.tensor([0.,0.,1.], device=device)
                else:
                    # 새 EOS가 R범위 안에 생겼으면 거기서 잘라 패딩
                    idx = (pen_oh[b].argmax(-1) == 2).nonzero(as_tuple=False)
                    if idx.numel() > 0:
                        first = int(idx[0].item())
                        eos_t[b] = t + first
                        if first + 1 < R:
                            step_chunk[b, first+1:, :2] = 0.0
                            step_chunk[b, first+1:, 2:] = torch.tensor([0.,0.,1.], device=device)

            executed.append(step_chunk)
            t += R

        out = torch.cat(executed, dim=1)[:, :T, :]                     # [B,T,5]

        # 최종 EOS 보강: 절대 없으면 마지막 프레임 EOS 강제
        pen_idx = out[..., 2:].argmax(-1)                               # [B,T]
        for b in range(B):
            if not (pen_idx[b] == 2).any():
                out[b, -1, :2] = 0.0
                out[b, -1, 2:] = torch.tensor([0.,0.,1.], device=device)

        return out.to(torch.float32)
