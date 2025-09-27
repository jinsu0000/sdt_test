import torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange
from models.flow_policy import FlowPolicy, ActionEmbed, build_block_mask
from utils.logger import print_once
from models.loss import SupConLoss

class SDT_FlowWrapper(nn.Module):
    """
    SDT를 Flow Matching + Action Chunking로 감싼 래퍼.
    - 스타일 토큰을 Content로 2단계 cross-attn 요약해 cond(1토큰) 생성
    - 학습: Δx,Δy (MSE) + pen (CrossEntropy)
    - 추론: 청크(H) 단위 오일러 적분 + Temporal Ensembling + R-step 재계획
    - 출력 포맷: [B, T, 5] (dx, dy, one-hot(3))  ← test.py/evaluate.py와 100% 호환
    """
    def __init__(self, sdt_model,
                 H:int=6,          # action chunk length
                 n_layers:int=6, n_head:int=8, ffn_mult:int=4, p:float=0.1,
                 stride_default:int=3,
                 nce_temperature: float = 0.07):
        super().__init__()
        self.sdt = sdt_model
        self.H = H
        self.stride_default = stride_default
        self.d_model = 512
        self.need_weights = True

        print_once(f"SDT_FlowWrapper:: __init__ H: {H}, stride_default: {stride_default}, n_layers: {n_layers}, n_head: {n_head}, ffn_mult: {ffn_mult}, p: {p}")

        # 정책 네트워크 (ctx: [B,Lctx,512], act_emb: [B,H,512])
        self.action_embed = ActionEmbed(d_action=2, d_model=self.d_model)
        self.policy = FlowPolicy(d_ctx=self.d_model, d_act=self.d_model,
                                 n_layers=n_layers, n_head=n_head, ffn_mult=ffn_mult, p=p)
        # cond 요약용 cross-attn (content -> writer -> glyph)
        self.mha_w = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=8, batch_first=True)
        self.mha_g = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=8, batch_first=True)

        self._last_style_attn = None
        self._last_content_attn = None
        
        # SupCon (NCE) — 기존 구현 사용
        self.supcon = SupConLoss(temperature=nce_temperature)

    # ---------- 스타일/컨텐트 인코딩 ----------
    @torch.no_grad()
    def _style_tokens(self, style_imgs):
        """
        SDT 인코더로 스타일 토큰 추출.
        return:
          wtok: [B, Lw(=4*N), 512]
          gtok: [B, Lg(=4*N), 512]
        """
        B, N, C, H, W = style_imgs.shape
        print_once(f"SDT_FlowWrapper::_style_tokens style_imgs: {style_imgs.shape}, {style_imgs.dtype}")
        x = style_imgs.view(-1, C, H, W)                              # [B*N,1,H,W]
        feat = self.sdt.Feat_Encoder(x)                               # [B*N, 512, 2, 2]
        feat = feat.reshape(B*N, 512, -1).permute(2, 0, 1)            # [4, B*N, 512]
        feat = self.sdt.add_position(feat)
        print_once(f"SDT_FlowWrapper::_style_tokens [DEBUG] feat: {feat.shape}, {feat.dtype}")
        mem  = self.sdt.base_encoder(feat)                            # [4, B*N, 512]
        print_once(f"SDT_FlowWrapper::_style_tokens [DEBUG] mem: {mem.shape}, {mem.dtype}")
        wmem = self.sdt.writer_head(mem)                              # [4, B*N, 512]
        gmem = self.sdt.glyph_head(mem)                               # [4, B*N, 512]
        print_once(f"SDT_FlowWrapper::_style_tokens [DEBUG] wmem: {wmem.shape}, {wmem.dtype}, gmem: {gmem.shape}, {gmem.dtype} ")
        # → [4*N, B, 512] → [B, 4*N, 512]
        wtok = rearrange(wmem, 't (b n) c -> b (t n) c', b=B)         # [B, 4*N, 512]
        gtok = rearrange(gmem, 't (b n) c -> b (t n) c', b=B)         # [B, 4*N, 512]
        print_once(f"SDT_FlowWrapper::_style_tokens [DEBUG] wtok: {wtok.shape}, {wtok.dtype}, gtok: {gtok.shape}, {gtok.dtype} ")
        return wtok.contiguous(), gtok.contiguous()

    @torch.no_grad()
    def _content_token(self, char_img):
        """
        SDT Content_TR 출력 평균으로 1토큰 생성.
        return: ctok [B, 1, 512]
        """
        cont = self.sdt.content_encoder(char_img)   # [4, B, 512]
        cont = cont.mean(0, keepdim=False)          # [B, 512]
        print_once(f"SDT_FlowWrapper::_content_token char_img: {char_img.shape}, {char_img.dtype} -> cont: {cont.shape}, {cont.dtype}")
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
        mid, _ = self.mha_w(query=ctok, key=wtok, value=wtok, need_weights=self.need_weights)
        print_once(f"SDT_FlowWrapper::_make_cond_ctx [DEBUG] mid: {mid.shape}, {mid.dtype}")
        # mid -> glyph
        cond, _ = self.mha_g(query=mid, key=gtok, value=gtok, need_weights=self.need_weights)
        print_once(f"SDT_FlowWrapper::_make_cond_ctx [DEBUG] cond: {cond.shape}, {cond.dtype}")

        if self.need_weights:
            print_once(f"SDT_FlowWrapper::_make_cond_ctx [DEBUG] style_attn: {_.shape}, {_.dtype}")
            # content -> writer
            mid, attn_w_w = self.mha_w(query=ctok, key=wtok, value=wtok, need_weights=True)  # [B,1,4N]
            print_once(f"[CTX] wtok L={wtok.size(1)}, attn_w_w mean={attn_w_w.mean().item():.4f}, max={attn_w_w.max().item():.4f}")

            # mid -> glyph
            cond, attn_w_g = self.mha_g(query=mid, key=gtok, value=gtok, need_weights=True)  # [B,1,4N]
            print_once(f"[CTX] gtok L={gtok.size(1)}, attn_w_g mean={attn_w_g.mean().item():.4f}, max={attn_w_g.max().item():.4f}")
            self.need_weights = False

        return cond.contiguous(), 1

    @torch.enable_grad()
    def nce_losses_supcon(self, style_imgs):
        """
        style_imgs: [B, 2*N, 1, H, W]  (훈련에서만 호출)
        원본 SDT forward()의 NCE 경로를 1:1로 재현.
        반환: (writer_supcon, glyph_supcon)
        """
        import torch
        import torch.nn.functional as F

        dev = next(self.parameters()).device
        B, num_imgs, C, H, W = style_imgs.shape

        # 두 뷰가 필요하므로 2의 배수 확인 (홀수면 마지막 1장 드롭)
        if num_imgs < 2:
            zero = torch.tensor(0.0, device=dev)
            return zero, zero
        if num_imgs % 2 == 1:
            num_imgs = num_imgs - 1
            style_imgs = style_imgs[:, :num_imgs]

        anchor_num = num_imgs // 2  # N
        x = style_imgs.view(-1, C, H, W).to(dev, non_blocking=True)  # [B*2N,1,H,W]

        # ===== 원본 SDT 인코더 경로 복제 =====
        feat = self.sdt.Feat_Encoder(x)                                # [B*2N, 512, 2, 2]
        feat = feat.view(B*num_imgs, 512, -1).permute(2, 0, 1)         # [4, B*2N, 512]
        feat = self.sdt.add_position(feat)
        mem  = self.sdt.base_encoder(feat)                             # [4, B*2N, 512]

        wmem = self.sdt.writer_head(mem)                               # [4, B*2N, 512]
        gmem = self.sdt.glyph_head(mem)                                # [4, B*2N, 512]

        # ===== Writer NCE =====
        # [4, B*2N, 512] -> [4, 2B, N, 512]
        writer_memory = rearrange(wmem, 't (b p n) c -> t (p b) n c', b=B, p=2, n=anchor_num)
        # [4, 2B, N, 512] -> [4N, 2B, 512] -> 평균 -> [2B, 512]
        memory_fea = rearrange(writer_memory, 't b n c -> (t n) b c')  # [4*N, 2B, 512]
        compact_fea = memory_fea.mean(dim=0)                           # [2B, 512]
        pro_w = self.sdt.pro_mlp_writer(compact_fea)                   # [2B, 256]
        # 앞 B = query, 뒤 B = pos
        q_w, p_w = pro_w[:B], pro_w[B:]
        q_w = F.normalize(q_w, dim=-1); p_w = F.normalize(p_w, dim=-1)
        feats_w = torch.stack([q_w, p_w], dim=1)              # [B,2,256]
        writer_supcon = self.supcon(feats_w)

        # ===== Glyph NCE =====
        # gmem: [4, B*2N, 512] -> 앞 B만 사용해서 [4, B, N, 512]
        # gmem: [4, B*2N, 512] -> [4, 2, B, N, 512] 로 '뷰(p)' 차원을 분리
        glyph_memory = rearrange(
            gmem, 't (b p n) c -> t p b n c', b=B, p=2, n=anchor_num
        )  # [4, 2, B, N, 512]

        # 첫 번째 뷰(p=0)만 사용 -> [4, B, N, 512]
        patch_emb = glyph_memory[:, 0]  # [4, B, N, 512]

        # 같은 글자 내 토큰에서 anchor/positive 샘플 (원본 함수와 동일 요구형태)
        anc, pos = self.sdt.random_double_sampling(patch_emb)  # OK: [L=4, B, N, D=512]

        D = anc.shape[-1]
        anc = anc.reshape(B, -1, D).mean(dim=1, keepdim=True)  # [B, 1, 512]
        pos = pos.reshape(B, -1, D).mean(dim=1, keepdim=True)  # [B, 1, 512]

        anc = self.sdt.pro_mlp_character(anc).squeeze(1)  # [B, 256]
        pos = self.sdt.pro_mlp_character(pos).squeeze(1)  # [B, 256]
        anc = torch.nn.functional.normalize(anc, dim=-1)
        pos = torch.nn.functional.normalize(pos, dim=-1)

        glyph_feats = torch.stack([anc, pos], dim=1)  # [B, 2, 256]
        glyph_supcon = self.supcon(glyph_feats)

        return writer_supcon, glyph_supcon
    
    def get_diag_attn(self):
        return {
            "style": getattr(self, "_last_style_attn", None),
            "content": getattr(self, "_last_content_attn", None),
            "xattn": getattr(self, "_last_xattn_mean", None),
        }
    
    def _update_diag_attn(self, Lctx: int, H: int, prefix_has_content: bool):
        # self-attn(행: target, 열: source) 가중치가 저장돼 있다면 읽기
        try:
            blk0 = self.policy.tr.blocks[0]
            attn = getattr(blk0, "last_self_attn", None)  # [B, Hd, L_tgt, L_src]
        except Exception:
            attn = None

        if attn is not None and attn.ndim == 4:
            # 액션 토큰 행만 잘라 평균
            action_rows = attn[..., Lctx:Lctx+H, :]  # [B, Hd, H, Lctx+H]
            # 컨텍스트(왼쪽 Lctx 열)과 나머지 분리
            style_part = action_rows[..., :Lctx]     # [B, Hd, H, Lctx]
            action_part = action_rows[..., Lctx:]    # [B, Hd, H, H]
            self._last_style_attn = style_part.mean().item()
            # content 분리가 필요하면 여기서 content 범위를 나눠 평균(우린 cond=1토큰이라 None)
            self._last_content_attn = None
        else:
            self._last_style_attn = None
            self._last_content_attn = None

        # (옵션) cross-attn 기록
        xa = getattr(getattr(self.policy, "xattn", None), "last_xattn", None)
        self._last_xattn_mean = xa.mean().item() if xa is not None else None

    # ---------- 학습 로스 ----------
    def flow_match_loss(self, style_imgs, coords, char_img,
                        tau_beta=(2.0,4.0), stride:int=None):
        """
        coords: [B, T, 5]  (dx,dy, one-hot(3))
        - Δx,Δy: MSE는 EOS '직전까지만' 유효
        - pen  : CE는 EOS '포함'까지 유효
        - EOS 이후/패딩은 전부 무시
        """
        device = coords.device
        B, T, C = coords.shape
        H = self.H
        stride = stride or self.stride_default
        print_once(f"SDT_FlowWrapper::flow_match_loss style_imgs: {style_imgs.shape}, {style_imgs.dtype}, B: {B}, char_img: {char_img.shape}, {char_img.dtype}")

        # cond context
        ctx, Lctx = self._make_cond_ctx(style_imgs, char_img)   # [B,1,512], 1

        # ---- 시퀀스 전체에서 EOS 위치 산출 ----
        pen_1hot  = coords[..., 2:]                  # [B,T,3]
        eos_full  = (pen_1hot[..., 2] == 1)         # [B,T]  (EOS는 class=2)
        has_eos   = eos_full.any(dim=1)             # [B]
        # 첫 EOS 인덱스 (없으면 T로 설정)
        first_eos = torch.where(
            has_eos, eos_full.float().argmax(dim=1), torch.full((B,), T, device=device, dtype=torch.long)
        ) # [B]
        print_once(f"SDT_FlowWrapper::flow_match_loss B={B} T={T} H={H} stride={stride} | EOS%={(has_eos.float().mean()*100):.1f} | first_eos(mean)={first_eos.float().mean().item():.1f}")


        # Δ 유효:  t < first_eos
        idxs = torch.arange(T, device=device)[None, :].expand(B, T)
        valid_delta_full = (idxs < first_eos[:, None])           # [B,T]
        # pen 유효: t <= first_eos (EOS 프레임 포함)
        valid_pen_full   = (idxs <= first_eos[:, None])          # [B,T]

        loss_flow = coords.new_tensor(0.0)
        loss_pen  = coords.new_tensor(0.0)
        n_blocks  = 0

        for start in range(0, T, stride):
            end = min(T, start + H)
            A_gt = coords[:, start:end, :2]                  # [B,L<=H,2]
            P_gt = coords[:, start:end, 2:]                  # [B,L<=H,3]
            Lcur = A_gt.size(1)

            # 패딩
            pad = H - Lcur
            if pad > 0:
                A_gt = torch.cat([A_gt, torch.zeros(B, pad, 2, device=device)], dim=1)
                P_gt = torch.cat([P_gt, torch.zeros(B, pad, 3, device=device)], dim=1)

            # 유효 마스크(패딩 제외)
            pad_mask = torch.ones(B, H, device=device)
            if pad > 0: pad_mask[:, -pad:] = 0.0

            # ---- EOS 마스크를 윈도우에 맞게 슬라이스 ----
            valid_delta = valid_delta_full[:, start:end]      # [B,L<=H]
            valid_pen   = valid_pen_full[:,   start:end]      # [B,L<=H]
            if pad > 0:
                valid_delta = torch.cat([valid_delta, torch.zeros(B, pad, device=device, dtype=torch.bool)], dim=1)
                valid_pen   = torch.cat([valid_pen,   torch.zeros(B, pad, device=device, dtype=torch.bool)], dim=1)

            # 윈도우가 아예 전부 무효(=EOS 이후/패딩뿐)면 스킵
            if (valid_delta.sum() == 0) and (valid_pen.sum() == 0):
                continue

            # τ 샘플 + 중간점 생성 (Δ만 사용하므로 Δ마스크 기준 통계)
            # 통계는 유효 Δ에서만 계산 (없으면 ε-σ 기본값)
            if valid_delta.any():
                std = A_gt[valid_delta].reshape(-1, 2).std(dim=0, unbiased=False).clamp_min(1e-3).view(1,1,2)
            else:
                std = torch.ones(1,1,2, device=device) * 1e-3
            a,b = tau_beta
            tau = torch.distributions.Beta(a,b).sample((B,1)).to(device)
            eps = torch.randn_like(A_gt) * std
            A_tau = tau.view(B,1,1) * A_gt + (1 - tau.view(B,1,1)) * eps  # [B,H,2]

            if start == 0:
                print_once(f"[FM-win0] Lcur={Lcur}, pad={pad}, validΔ={int(valid_delta.sum())}, validPen={int(valid_pen.sum())}, "
                        f"std={std.view(-1).tolist()}")

            # 정책
            act_emb   = self.action_embed(A_tau, tau, start_idx=start)      # [B,H,512]
            attn_mask = build_block_mask(Lctx, H, device=device)
            v, pen_logits = self.policy(ctx, act_emb, None, attn_mask)      # v:[B,H,2], pen_logits:[B,H,3]

            _check_finite(v, "policy output v")
            _check_finite(pen_logits, "policy output pen_logits")
            
            self._update_diag_attn(Lctx=Lctx, H=H, prefix_has_content=True)

            # ---- Δ 손실 (EOS 이전만) ----
            u_star = (A_gt - eps)                                           # [B,H,2]
            m_delta = valid_delta.float() * pad_mask                        # [B,H]
            if m_delta.sum() > 0:
                diff = (v - u_star).pow(2).sum(-1) * m_delta               # [B,H]
                lf   = diff.sum() / m_delta.sum().clamp_min(1.0)

                # time-smoothness: 인접 두 지점 모두 유효한 곳만
                if H > 1:
                    m2 = (m_delta[:, 1:] * m_delta[:, :-1]).unsqueeze(-1)
                    dv = (v[:,1:,:] - v[:,:-1,:]) * m2
                    du = (u_star[:,1:,:] - u_star[:,:-1,:]) * m2
                    ls = torch.mean((dv - du).pow(2)) if m2.sum() > 0 else torch.zeros((), device=device)
                else:
                    ls = torch.zeros((), device=device)
                loss_flow += (lf + 0.1 * ls)

            # ---- pen CE (EOS 포함까지만) ----
            # 무시할 위치를 -100으로 라벨링
            tgt_lbl = P_gt.argmax(-1)                                       # [B,H] (0/1/2)
            ignore_mask = (~valid_pen) | (pad_mask == 0)                    # [B,H]  (무효 or 패딩)
            tgt_lbl = tgt_lbl.masked_fill(ignore_mask, -100)
            lp = F.cross_entropy(pen_logits.reshape(-1,3),
                                tgt_lbl.reshape(-1),
                                ignore_index=-100)
            loss_pen += lp


            if start == 0:
                # lf/ls/lp는 위에서 계산된 텐서들 (없으면 0으로 대체)
                lf_val = float(lf.item()) if 'lf' in locals() else 0.0
                ls_val = float(ls.item()) if 'ls' in locals() else 0.0
                lp_val = float(lp.item()) if 'lp' in locals() else 0.0
                print_once(f"[FM-win0] losses: ΔMSE={lf_val:.4f} smooth={ls_val:.4f} penCE={lp_val:.4f}")


            n_blocks += 1

        # 평균
        loss_flow = loss_flow / max(n_blocks, 1)
        loss_pen  = loss_pen  / max(n_blocks, 1)
        return loss_flow, loss_pen


    # ---------- 추론: 청크 + 앙상블 ----------
    @torch.no_grad()
    def flow_infer(self, style_imgs, char_img, T:int,
                steps:int=20, stride:int=None, replan:int=None,
                solver:str="euler",               # NEW: "euler" | "rk4"
                micro_pen_ensemble:bool=True,     # NEW: 내부 스텝 pen 앙상블
                micro_pen_weight:str="linear"     # NEW: "linear" | "uniform"
                ):
        """
        오일러/ RK4 적분으로 Δx,Δy 생성 + pen 확률 → one-hot.
        Temporal ensembling(삼각가중), R-step 재계획.
        return: [B, T, 5]
        """
        device = next(self.parameters()).device
        B = style_imgs.size(0)
        H = self.H
        stride = stride if stride is not None else self.stride_default
        replan = replan if replan is not None else stride

        print_once(f"SDT_FlowWrapper::flow_infer style_imgs: {style_imgs.shape}, {style_imgs.dtype}, B: {B}, char_img: {char_img.shape}, {char_img.dtype}, T: {T}, steps: {steps}, stride: {stride}, replan: {replan}, solver: {solver}, micro_pen_ensemble: {micro_pen_ensemble}, micro_pen_weight: {micro_pen_weight}")

        ctx, Lctx = self._make_cond_ctx(style_imgs, char_img)  # [B,1,512], 1

        # 누적 버퍼
        acc_xy  = torch.zeros(B, T, 2, device=device)
        acc_pen = torch.zeros(B, T, 3, device=device)
        acc_w   = torch.zeros(B, T, 1, device=device)

        def w_tri(h: int):  # 청크 내부 삼각 가중 (offset 최근일수록 가중↑)
            return torch.linspace(1, h, steps=h, device=device) / max(h, 1)

        t = 0
        executed = []
        eos_t = torch.full((B,), -1, device=device, dtype=torch.long)

        while t < T:
            h = min(H, T - t)

            # 청크 초기값(노이즈): 학습과 동일 σ를 쓰는 게 중요 (옵션 B라면 self.sigma_eps 사용)
            a = torch.randn(B, h, 2, device=device)
            if hasattr(self, "sigma_eps"):  # 고정 σ0를 쓰는 세팅이라면
                a = a * self.sigma_eps      # [2] -> 브로드캐스트

            attn_mask = build_block_mask(Lctx, h, device=device)

            # ---------- ODE 적분 ----------
            dt = 1.0 / max(steps, 1)
            tau = torch.zeros(B, 1, device=device)

            # pen micro-ensemble 버퍼
            if micro_pen_ensemble:
                pen_micro_acc = torch.zeros(B, h, 3, device=device)
                pen_micro_w   = torch.zeros(B, h, 1, device=device)

                def micro_w(s_idx: int):
                    if micro_pen_weight == "linear":
                        return float(s_idx + 1) / steps
                    else:  # "uniform"
                        return 1.0

            # f(x, τ) = v_θ(x, τ, ctx)
            def f(x, tau_s):
                emb = self.action_embed(x, tau_s, start_idx=t)          # [B,h,512]
                v, _ = self.policy(ctx, emb, None, attn_mask)           # [B,h,2]
                return v

            for s in range(steps):
                if solver == "rk4":
                    # classical RK4
                    k1 = f(a,               tau)
                    k2 = f(a + 0.5*dt*k1,   tau + 0.5*dt)
                    k3 = f(a + 0.5*dt*k2,   tau + 0.5*dt)
                    k4 = f(a + dt*k3,       tau + dt)
                    a  = a + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
                    tau = tau + dt
                else:
                    # Euler
                    v = f(a, tau)
                    a = a + dt * v
                    tau = tau + dt

                # 내부 스텝마다 pen 예측을 누적(옵션)
                if micro_pen_ensemble:
                    emb_pen = self.action_embed(a, tau, start_idx=t)
                    _, pen_logits_s = self.policy(ctx, emb_pen, None, attn_mask)  # [B,h,3]
                    pen_prob_s = pen_logits_s.softmax(-1)
                    w_s = micro_w(s)
                    w_s = torch.tensor(w_s, device=device).view(1, 1, 1)
                    pen_micro_acc += w_s * pen_prob_s
                    pen_micro_w   += w_s

            # 청크 최종 pen (마이크로 앙상블 사용/미사용 분기)
            if micro_pen_ensemble:
                pen_prob = pen_micro_acc / pen_micro_w.clamp_min(1e-8)       # [B,h,3]
            else:
                tau1 = torch.ones(B, 1, device=device)
                emb1 = self.action_embed(a, tau1, start_idx=t)
                _, pen_logits = self.policy(ctx, emb1, None, attn_mask)
                pen_prob = pen_logits.softmax(-1)                             # [B,h,3]

            if t == 0:
                pen_summary = pen_prob.mean(dim=(0,1)).tolist()  # [3]
                print_once(f"[INF-chunk0] h={h}, a.std={a.std().item():.4f}, pen_prob(mean)={pen_summary}, "
                        f"acc_w.sum(firstR)={acc_w[:, t:t+min(replan,h), :].sum().item():.2f}")


            # ---------- 시간축 앙상블(청크 오프셋) ----------
            w = w_tri(h).view(1, h, 1)                                        # [1,h,1]
            acc_xy[:,  t:t+h, :] += w * a
            acc_pen[:, t:t+h, :] += w * pen_prob
            acc_w[:,   t:t+h, :] += w

            # ---------- 앞 R 스텝 확정 ----------
            R = min(replan, T - t)
            wsum = acc_w[:, t:t+R, :].clamp_min(1e-8)
            xy   = acc_xy[:, t:t+R, :] / wsum                                  # [B,R,2]
            pen  = acc_pen[:, t:t+R, :] / wsum                                  # [B,R,3]
            pen_oh = torch.nn.functional.one_hot(pen.argmax(-1), num_classes=3).to(pen).float()
            step_chunk = torch.cat([xy, pen_oh], dim=-1)                        # [B,R,5]

            # EOS 처리
            for b in range(B):
                if eos_t[b] >= 0:
                    step_chunk[b, :, :2] = 0.0
                    step_chunk[b, :, 2:] = torch.tensor([0., 0., 1.], device=device)
                else:
                    idx = (pen_oh[b].argmax(-1) == 2).nonzero(as_tuple=False)
                    if idx.numel() > 0:
                        first = int(idx[0].item())
                        eos_t[b] = t + first
                        if first + 1 < R:
                            step_chunk[b, first+1:, :2] = 0.0
                            step_chunk[b, first+1:, 2:] = torch.tensor([0., 0., 1.], device=device)

            executed.append(step_chunk)
            t += R
            
            if (eos_t >= 0).all():
                break

        out = torch.cat(executed, dim=1)[:, :T, :]                            # [B,T,5]

        # 최종 EOS 보강
        pen_idx = out[..., 2:].argmax(-1)
        for b in range(B):
            if not (pen_idx[b] == 2).any():
                out[b, -1, :2] = 0.0
                out[b, -1, 2:] = torch.tensor([0., 0., 1.], device=device)

        return out.to(torch.float32)


def _check_finite(x, name):
    if not torch.isfinite(x).all():
        print_once(f"[WARN] {name} contains non-finite values! "
                f"mean={x.nanmean().item() if torch.isfinite(x).any() else float('nan')}")
