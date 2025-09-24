import torch, torch.nn as nn, torch.nn.functional as F
from models.flow_policy import FlowPolicy, ActionEmbed, build_block_mask
from models.loss import SupConLoss  # 기존 구현 사용 (temperature는 init에서 설정)

from einops import rearrange

class SDT_FlowWrapper(nn.Module):
    """
    Wrap SDT_Generator:
      - Writer/Glyph: 스타일 프리픽스 토큰 (keep_k로 길이 제어)
      - Content: prefix ('prefix') 또는 Action->Content cross-attn ('xattn')
      - Flow Matching + Action Chunking (슬라이딩/패딩/σ-매칭/스무딩)
      - SupCon NCE(Writer/Glyph) 학습 포함 + 텐보드 로깅용 diag
    """
    def __init__(self, sdt_model, H:int=64, n_layers:int=6, n_head:int=8, ffn_mult:int=4,
                 p:float=0.1, condition_mode:str="prefix", keep_k:int=0, nce_temperature:float=0.07):
        super().__init__()
        assert condition_mode in ("prefix", "xattn")
        self.sdt = sdt_model
        self.H = H
        self.d_model = 512
        self.condition_mode = condition_mode
        self.keep_k = keep_k

        self.action_embed = ActionEmbed(d_action=2, d_model=self.d_model)
        self.policy = FlowPolicy(d_ctx=self.d_model, d_act=self.d_model,
                                 n_layers=n_layers, n_head=n_head, ffn_mult=ffn_mult, p=p)

        # style projections (동결: 스타일 메모리 drift 방지)
        self.proj_style = nn.Linear(self.d_model, self.d_model, bias=False)
        self.proj_glyph = nn.Linear(self.d_model, self.d_model, bias=False)
        self.proj_style.weight.requires_grad_(False)
        self.proj_glyph.weight.requires_grad_(False)

        # content projection (trainable)
        self.proj_content = nn.Linear(self.d_model, self.d_model, bias=False)

        self.ctx_type_embed = nn.Embedding(4, self.d_model)  # 0:Writer 1:Glyph 2:Content 3:Action
        self.lambda_smooth = 0.1

        # SupCon (NCE) — 기존 구현 사용
        self.supcon = SupConLoss(temperature=nce_temperature)

        # last diag holders
        self._last_style_attn = None
        self._last_content_attn = None
        self._last_xattn_mean = None

    # ---- Encoders ----
    @torch.no_grad()
    def _encode_style_ctx(self, style_imgs, keep_tokens=None):
        """
        return:
        ctx: [B, Ls, 512]  (style context tokens)
        Ls : int           (length of ctx)
        """
        device = style_imgs.device
        B, N, C, H, W = style_imgs.shape

        # --- SDT 스타일 인코딩 파이프라인 (원본 inference와 동일한 전개) ---
        style = style_imgs.view(-1, C, H, W)                        # [B*N, 1, H, W]
        style_embe = self.sdt.Feat_Encoder(style)                   # [B*N, 512, 2, 2]
        FEAT_ST = style_embe.reshape(B*N, 512, -1).permute(2, 0, 1) # [4, B*N, 512]
        FEAT_ST_ENC = self.sdt.add_position(FEAT_ST)
        memory = self.sdt.base_encoder(FEAT_ST_ENC)                 # [4, B*N, 512]

        wmem = self.sdt.writer_head(memory)                         # [4, B*N, 512]
        gmem = self.sdt.glyph_head(memory)                          # [4, B*N, 512]

        # [4, B*N, 512] -> [4*N, B, 512] -> [B, 4*N, 512]
        wmem = rearrange(wmem, 't (b n) c -> (t n) b c', b=B).transpose(0, 1)
        gmem = rearrange(gmem, 't (b n) c -> (t n) b c', b=B).transpose(0, 1)

        # 둘 중 하나만 쓰거나 평균(선호하는 쪽 택하세요)
        ctx = 0.5 * (wmem + gmem)                                   # [B, 4*N, 512]
        if keep_tokens is not None:
            Ls = min(ctx.size(1), keep_tokens)
            ctx = ctx[:, :Ls, :]
        else:
            Ls = ctx.size(1)

        return ctx, Ls

    @torch.no_grad()
    def _encode_content_tokens(self, char_img):
        """
        SDT Content_TR 출력: [4, B, 512] → [B, 4, 512] 로 표준화
        """
        device = next(self.parameters()).device
        char_img = char_img.to(device, non_blocking=True)
        cont = self.sdt.content_encoder(char_img)  # 기대: [4, B, 512]
        if cont.dim()==3 and cont.shape[0]==4:  # 정상 경로
            cont = cont.permute(1, 0, 2).contiguous()  # [B,4,512]
        else:
            # 방어: 다른 모양도 4 토큰으로 압축
            if cont.dim()==4:  # [B,C,H,W] -> 2x2
                B, C, H, W = cont.shape
                cont = F.adaptive_avg_pool2d(cont, (2,2)).permute(0,2,3,1).reshape(B, 4, C)
            elif cont.dim()==3:
                if cont.shape[1] != 4:  # [B,S,C] -> 1D pool to 4
                    cont = F.adaptive_avg_pool1d(cont.transpose(1,2), 4).transpose(1,2)
            elif cont.dim()==2:
                cont = cont.unsqueeze(1).repeat(1, 4, 1)
            else:
                raise RuntimeError(f"Unsupported content shape: {tuple(cont.shape)}")
            if cont.shape[-1] != 512:
                proj = nn.Linear(cont.shape[-1], 512, bias=False).to(device)
                cont = proj(cont)
        ctok = self.proj_content(cont)  # [B,4,512]
        ctok = ctok + self.ctx_type_embed.weight[2][None, None, :]  # Content type
        return ctok

    # ---- Context builders for two modes ----
    @torch.no_grad()
    def _make_prefix_ctx(self, style_imgs, char_img, keep_k=None):
        """
        return:
        ctx:      [B, Lctx, 512]  = [B, 1 + Ls, 512] (content 먼저)
        Lctx:     int
        content:  [B, 1, 512]
        """
        # 1) 스타일 컨텍스트
        ctx_s, Ls = self._encode_style_ctx(style_imgs, keep_tokens=keep_k)  # [B, Ls, 512], int

        # 2) 컨텐츠 토큰(맨 앞에 1개)
        content = self.sdt.content_encoder(char_img).mean(0, keepdim=True)  # [1, B, 512]
        content = content.transpose(0, 1)                                    # [B, 1, 512]

        # 3) content 먼저 + style context 이어붙이기 (원본 SDT와 동일한 순서)
        ctx = torch.cat([content, ctx_s], dim=1)  # [B, 1+Ls, 512]
        Lctx = 1 + Ls
        return ctx, Lctx, content

    def _make_xattn_ctx(self, style_imgs, char_img):
        keep_k = getattr(self, "keep_k", 0)
        ctx   = self._encode_style_ctx(style_imgs, keep_tokens=keep_k)  # [B,Ls,512]
        Lctx  = ctx.size(1)
        ctok  = self._encode_content_tokens(char_img)                   # [B,4,512]
        return ctx, Lctx, ctok

    # ---- Flow Matching loss ----
    def _attn_stats(self, attn_w, Lctx:int, H:int, prefix_has_content: bool):
        if attn_w is None:
            return None, None
        B, Hd, L_tgt, L_src = attn_w.shape
        action_rows = attn_w[..., Lctx:Lctx+H, :]
        if prefix_has_content and Lctx >= 5:
            style_cols = action_rows[..., :Lctx-4]
            content_cols = action_rows[..., Lctx-4:Lctx]
            style_mean = style_cols.mean().item() if style_cols.numel() else 0.0
            content_mean = content_cols.mean().item()
        else:
            style_mean = action_rows[..., :Lctx].mean().item()
            content_mean = None
        return style_mean, content_mean

    def get_diag_attn(self):
        return {
            "style": getattr(self, "_last_style_attn", None),
            "content": getattr(self, "_last_content_attn", None),
            "xattn": getattr(self, "_last_xattn_mean", None),
        }

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

            mask = torch.ones(B, H, device=device)
            if pad > 0: mask[:, -pad:] = 0.0

            std = A_gt.reshape(-1, 2).std(dim=0, unbiased=False).clamp_min(1e-3).view(1,1,2)
            a,b = tau_beta
            tau = torch.distributions.Beta(a,b).sample((B,1)).to(device)
            eps = torch.randn_like(A_gt) * std
            A_tau = tau.view(B,1,1)*A_gt + (1-tau.view(B,1,1))*eps

            act_emb = self.action_embed(A_tau, tau, start_idx=start)
            act_emb = act_emb + self.ctx_type_embed.weight[3][None, None, :]

            attn_mask = build_block_mask(Lctx, H, device=device)
            v, pen_logits = self.policy(ctx, act_emb, content_tok, attn_mask)  # content_tok=None => prefix

            u_star = (A_gt - eps)
            diff = (v - u_star).pow(2).sum(-1) * mask
            lf = diff.sum() / mask.sum().clamp_min(1.0)

            if H > 1:
                dv = (v[:,1:,:] - v[:,:-1,:]) * mask[:,1:].unsqueeze(-1)
                du = (u_star[:,1:,:] - u_star[:,:-1,:]) * mask[:,1:].unsqueeze(-1)
                ls = torch.mean((dv - du).pow(2))
            else:
                ls = torch.zeros((), device=device)

            loss_flow = loss_flow + (lf + self.lambda_smooth * ls)

            if coords.size(-1) >= 5:
                lbl = coords[:, start:end, 2:].argmax(-1)
                if pad > 0:
                    lbl = torch.cat([lbl, torch.zeros(B, pad, dtype=lbl.dtype, device=device)], dim=1)
                lp = F.cross_entropy(pen_logits.reshape(-1, pen_logits.size(-1)),
                                     lbl.reshape(-1), ignore_index=0)
                loss_pen = loss_pen + lp
            n += 1

            # ---- diagnostics (첫 블록 self-attn, xattn) ----
            blk0 = self.policy.tr.blocks[0]
            self._last_style_attn, self._last_content_attn = self._attn_stats(
                getattr(blk0, "last_self_attn", None), Lctx=Lctx, H=H,
                prefix_has_content=(self.condition_mode=="prefix")
            )
            if self.condition_mode == "xattn":
                attn = getattr(self.policy.xattn, "last_xattn", None)
                self._last_xattn_mean = attn.mean().item() if attn is not None else None
            else:
                self._last_xattn_mean = None

        loss_flow = loss_flow / max(n,1)
        loss_pen  = loss_pen  / max(n,1)
        return loss_flow, loss_pen

    # ---- SupCon NCE losses (학습 포함) ----
    @torch.enable_grad()
    def nce_losses_supcon(self, style_imgs):
        """
        기존 SDT 경로 그대로 따라 writer/glyph 임베딩을 만들고,
        두 뷰를 SupConLoss에 넣어 학습 손실로 사용.
        반환: (writer_supcon, glyph_supcon)
        """
        dev = next(self.parameters()).device
        style_imgs = style_imgs.to(dev, non_blocking=True)
        B, num_imgs, C, H, W = style_imgs.shape  # [B, 2N, 1, H, W]
        x = style_imgs.view(-1, C, H, W)         # [B*2N, 1, H, W]

        feat = self.sdt.Feat_Encoder(x)                          # [B*2N, 512, 2, 2]
        feat = feat.view(B*num_imgs, 512, -1).permute(2, 0, 1)   # [4, B*2N, 512]
        feat = self.sdt.add_position(feat)
        mem  = self.sdt.base_encoder(feat)                       # [4, B*2N, 512]
        wmem = self.sdt.writer_head(mem)                         # [4, B*2N, 512]
        gmem = self.sdt.glyph_head(mem)                          # [4, B*2N, 512]
        N = num_imgs // 2

        # Writer
        writer_memory = wmem.view(4, B, 2, N, 512).permute(0, 3, 2, 1, 4).reshape(4*N, 2*B, 512)
        writer_compact = writer_memory.mean(0)                   # [2B, 512]
        emb_w = self.sdt.pro_mlp_writer(writer_compact)          # [2B, D]
        q_w, p_w = emb_w[:B], emb_w[B:]                          # [B, D], [B, D]
        q_w = F.normalize(q_w, dim=-1); p_w = F.normalize(p_w, dim=-1)
        feat_w = torch.stack([q_w, p_w], dim=1)                  # [B, 2, D]
        writer_supcon = self.supcon(feat_w)

        # Glyph
        patch = gmem.view(4, B, 2, N, 512)[:, :, 0]              # [4, B, N, 512]
        try:
            anc, pos = self.sdt.random_double_sampling(patch)    # [B, N, 1, 512] 유사
            anc = anc.reshape(B, -1, 512).mean(1)
            pos = pos.reshape(B, -1, 512).mean(1)
        except Exception:
            gm = patch.permute(1, 2, 0, 3)                       # [B, N, 4, 512]
            anc = gm.mean(dim=(1, 2))
            pos = gm[:, 0].mean(1)
        q_g = self.sdt.pro_mlp_character(anc.unsqueeze(1)).squeeze(1)   # [B, D]
        p_g = self.sdt.pro_mlp_character(pos.unsqueeze(1)).squeeze(1)   # [B, D]
        q_g = F.normalize(q_g, dim=-1); p_g = F.normalize(p_g, dim=-1)
        feat_g = torch.stack([q_g, p_g], dim=1)                  # [B, 2, D]
        glyph_supcon = self.supcon(feat_g)

        return writer_supcon, glyph_supcon

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


    @torch.no_grad()
    def inference_chunk_ensemble(self,
                                style_imgs,
                                char_img,
                                total_len: int,
                                chunk_size: int = 4,   # H
                                replan: int = 1,       # R
                                age_decay: float = 1.0,
                                use_sigma_sampling: bool = True):
        device = style_imgs.device
        B, N_imgs, C_in, H_in, W_in = style_imgs.shape

        # --- style memory (원본과 동일) ---
        style = style_imgs.view(-1, C_in, H_in, W_in)
        style_embe = self.Feat_Encoder(style)
        FEAT_ST = style_embe.reshape(B*N_imgs, 512, -1).permute(2, 0, 1)
        memory = self.base_encoder(self.add_position(FEAT_ST))
        mem_w  = rearrange(self.writer_head(memory), 't (b n) c -> (t n) b c', b=B)
        mem_g  = rearrange(self.glyph_head(memory),  't (b n) c -> (t n) b c', b=B)
        d_model = mem_w.size(-1)

        # --- content token (맨 앞에 1개, 출력엔 포함 안됨) ---
        char_emb = self.content_encoder(char_img).mean(0)  # [B,512]

        # --- 누적 버퍼 ---
        acc_xy = torch.zeros(B, total_len, 2, device=device)
        acc_w  = torch.zeros(B, total_len, 1, device=device)

        # --- 실행된 전체 시퀀스(완전-AR 프리픽스) ---
        executed = torch.zeros(B, 0, 5, device=device)
        # 샘플별 EOS 시점(-1이면 아직)
        eos_t = torch.full((B,), fill_value=-1, device=device, dtype=torch.long)

        t = 0
        while t < total_len:
            # 아직 살아있는(=EOS 안난) 마스크
            alive = (eos_t < 0)
            if not alive.any():
                break

            H = min(chunk_size, total_len - t)

            # 기존 계획 감쇠
            if age_decay < 1.0:
                acc_xy[:, t:, :] *= age_decay
                acc_w[:,  t:, :] *= age_decay

            # --- 완전-AR 프리픽스 src 구성: [content] + [executed] + [H칸] ---
            Lhist = executed.size(1)
            src = torch.zeros(1 + Lhist + H, B, d_model, device=device)
            src[0] = char_emb
            if Lhist > 0:
                src[1:1+Lhist] = self.SeqtoEmb(executed).transpose(0, 1)
                for j in range(Lhist):
                    src[1+j] = self.add_position(src[1+j], step=j)

            tgt_mask = generate_square_subsequent_mask(sz=1 + Lhist + H).to(device)

            # --- 내부 AR 롤아웃(H step) ---
            # k=0(이번 계획의 가장 앞 step)의 “샘플”을 pen 기준으로 사용
            k0_step = None
            for i in range(H):
                cur_idx = 1 + Lhist + i
                step_glob = t + i
                src[cur_idx] = self.add_position(src[cur_idx], step=step_glob)

                wri_hs = self.wri_decoder(src, mem_w, tgt_mask=tgt_mask)
                hs     = self.gly_decoder(wri_hs[-1], mem_g, tgt_mask=tgt_mask)
                h_i    = hs[-1][cur_idx]
                gmm_i  = self.EmbtoSeq(h_i)

                # 샘플링(원본과 동일한 성질 유지)
                step_i = get_seq_from_gmm(gmm_i) if use_sigma_sampling else get_seq_from_gmm(gmm_i)

                # k=0 결과 저장(시간 t의 pen 사용)
                if i == 0:
                    k0_step = step_i.clone()  # [B,5]

                # 다음 입력(teacher forcing 없음)
                if i < H - 1:
                    src[cur_idx + 1] = self.SeqtoEmb(step_i)

                # (Δx,Δy)는 ensembling 대상 → 누적(살아있는 샘플만)
                dxdy_i = step_i[..., :2]                    # [B,2]
                w_i = 1.0 - (i / H)                        # 삼각가중 (최근일수록 큼)
                w_i = torch.tensor(w_i, device=device).view(1,1)
                mask_alive = alive.view(B, 1)              # [B,1]
                acc_xy[:, t+i, :] += mask_alive * w_i * dxdy_i
                acc_w[:,  t+i, :] += mask_alive * w_i

                # i 시점에서 EOS가 나온 샘플은 eos_t 설정 (가장 이른 시점 고정)
                pen_i = step_i[..., 2:]                    # [B,3] one-hot
                eos_now = (pen_i.argmax(-1) == 2) & (eos_t < 0)
                eos_t[eos_now] = t + i

            # --- 앞 R 스텝 확정 실행(= 완료된 out에 append) ---
            R = min(replan, total_len - t)
            wsum = acc_w[:, t:t+R, :].clamp_min(1e-8)      # [B,R,1]
            xy   = acc_xy[:, t:t+R, :] / wsum              # [B,R,2]

            # pen은 평균/argmax 하지 말고 k=0 샘플을 사용 (원본과 성질 맞춤)
            pen0 = k0_step[..., 2:].unsqueeze(1).expand(B, R, 3)  # [B,R,3]

            steps = torch.cat([xy, pen0], dim=-1)           # [B,R,5]

            # EOS 넘어가면 전부 0으로
            for b in range(B):
                if eos_t[b] >= 0:
                    # 이번에 실행하는 구간에 EOS가 포함되면 거기서 끊고 뒤는 0패딩
                    cut = eos_t[b].item() - t + 1  # eos 포함 위치(상대 인덱스)
                    if 0 <= cut < R:
                        steps[b, cut:, :2] = 0.0
                        steps[b, cut:, 2:] = torch.tensor([0.,0.,1.], device=device)

            executed = torch.cat([executed, steps], dim=1)
            t += R

        # --- 최종 out: total_len 길이로 맞추고, 각 샘플 EOS 이후 0패딩 보장 ---
        out = torch.zeros(B, total_len, 5, device=device)
        Lout = executed.size(1)
        out[:, :Lout, :] = executed[:, :total_len, :]

        for b in range(B):
            et = eos_t[b].item()
            if et < 0:
                # EOS가 한 번도 없으면 마지막 프레임을 EOS로
                out[b, -1, :2] = 0.0
                out[b, -1, 2:] = torch.tensor([0.,0.,1.], device=device)
            else:
                if et+1 < total_len:
                    out[b, et+1:, :2] = 0.0
                    out[b, et+1:, 2:] = torch.tensor([0.,0.,1.], device=device)

        return out.to(torch.float32)

