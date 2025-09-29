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
                 nce_temperature: float = 0.07
                 ):
        super().__init__()
        self.sdt = sdt_model
        self.H = H
        self.stride_default = stride_default
        self.d_model = 512

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
        self._last_xattn_mean = None
        
        # SupCon (NCE) — 기존 구현 사용
        self.supcon = SupConLoss(temperature=nce_temperature)
    
    
    # --------- 한 번만 인코딩 ---------
    def _encode_style_once(self, style_imgs):
        """
        return dict:
          wtok:[B,4N,512], gtok:[B,4N,512],
          feats_w:[B,2,256], feats_g:[B,2,256], anchor_num:int, num_imgs:int
        """
        device = style_imgs.device
        B, num_imgs, C, H, W = style_imgs.shape
        x = style_imgs.reshape(-1, C, H, W)                        # [B*2N,1,H,W] (train) / [B*N,1,H,W] (val)

        feat = self.sdt.Feat_Encoder(x)                            # [B*?,512,2,2]
        feat = feat.view(B*num_imgs, 512, -1).permute(2,0,1)       # [4, B*?, 512]
        feat = self.sdt.add_position(feat)
        mem  = self.sdt.base_encoder(feat)                         # [4, B*?, 512]
        wmem = self.sdt.writer_head(mem)                           # [4, B*?, 512]
        gmem = self.sdt.glyph_head(mem)                            # [4, B*?, 512]

        # cond 토큰 (모든 N을 그대로 사용)
        wtok = rearrange(wmem, 't (b n) c -> b (t n) c', b=B)      # [B,4N,512]
        gtok = rearrange(gmem, 't (b n) c -> b (t n) c', b=B)      # [B,4N,512]

        # --- NCE 임베딩 (학습시에만 2뷰 가정) ---
        if num_imgs >= 2:
            anchor_num = num_imgs // 2
            p2 = 2 * anchor_num  # 사용할 총 이미지 수 (짝수)

            # Writer SupCon
            wmem_nce = wmem[:, :B * p2]  # [4, 2B*N, 512]
            writer_memory = rearrange(wmem_nce, 't (b p n) c -> t (p b) n c', b=B, p=2, n=anchor_num)  # [4,2B,N,512]
            memory_fea = rearrange(writer_memory, 't b n c -> (t n) b c')                          # [4N,2B,512]
            compact = memory_fea.mean(0)                                                           # [2B,512]
            pro_w = self.sdt.pro_mlp_writer(compact)                                              # [2B,256]
            q_w, p_w = pro_w[:B], pro_w[B:]
            q_w = F.normalize(q_w, dim=-1); p_w = F.normalize(p_w, dim=-1)
            feats_w = torch.stack([q_w, p_w], dim=1)                                              # [B,2,256]

            # Glyph SupCon (첫 번째 뷰만)
            gmem_nce = gmem[:, :B * p2]
            glyph_memory = rearrange(gmem_nce, 't (b p n) c -> t p b n c', b=B, p=2, n=anchor_num)     # [4,2,B,N,512]
            patch = glyph_memory[:, 0]                                                             # [4,B,N,512]
            anc, pos = self.sdt.random_double_sampling(patch)                                      # [L=4,B,N,512]
            D = anc.shape[-1]
            anc = anc.reshape(B, -1, D).mean(1, keepdim=True)                                      # [B,1,512]
            pos = pos.reshape(B, -1, D).mean(1, keepdim=True)                                      # [B,1,512]
            anc = self.sdt.pro_mlp_character(anc).squeeze(1)                                       # [B,256]
            pos = self.sdt.pro_mlp_character(pos).squeeze(1)                                       # [B,256]
            anc = F.normalize(anc, dim=-1); pos = F.normalize(pos, dim=-1)
            feats_g = torch.stack([anc, pos], dim=1)                                               # [B,2,256]
        else:
            anchor_num = 0
            feats_w = torch.zeros(B,2,256, device=device)
            feats_g = torch.zeros(B,2,256, device=device)

        return {
            "wtok": wtok.contiguous(), "gtok": gtok.contiguous(),
            "feats_w": feats_w, "feats_g": feats_g,
            "anchor_num": anchor_num, "num_imgs": num_imgs
        }
    
    def _content_token(self, char_img):
        """
        SDT Content_TR 출력 평균으로 1토큰 생성.
        return: ctok [B, 1, 512]
        """
        cont = self.sdt.content_encoder(char_img)   # [4, B, 512]
        cont = cont.mean(0, keepdim=False)          # [B, 512]
        print_once(f"SDT_FlowWrapper::_content_token char_img: {char_img.shape}, {char_img.dtype} -> cont: {cont.shape}, {cont.dtype}")
        return cont.unsqueeze(1)                    # [B, 1, 512]
    
    def _make_cond_ctx(self, wtok, gtok, char_img):
        # 1) encoders (grad ON)
        ctok = self._content_token(char_img)             # [B, 1, 512]

        # 2) content -> writer -> glyph 로 cond 2개 생성
        # wtok [B, Lw, 512], gtok [B, Lg, 512]
        cond_w, _ = self.mha_w(query=ctok, key=wtok, value=wtok, need_weights=False)  # [B,1,512]
        cond_g, _ = self.mha_g(query=cond_w, key=gtok, value=gtok, need_weights=False) # [B,1,512]

        # 3) prefix = [ content | cond_w | cond_g ]  => Lctx = 3
        ctx = torch.cat([ctok, cond_w, cond_g], dim=1)   # [B, 3, 512]
        return ctx.contiguous(), 3

    # --------- Losses (precomputed 사용) ---------
    def nce_losses_supcon(self, pre):
        writer_supcon = self.supcon(pre["feats_w"])
        glyph_supcon  = self.supcon(pre["feats_g"])
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
    def flow_match_loss(self, coords, cond_ctx, Lctx:int,
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
        print_once(f"SDT_FlowWrapper::flow_match_loss B: {B}, T: {T}, C: {C}, H: {H}, stride: {stride}, cond_ctx: {cond_ctx.shape}, {cond_ctx.dtype}")

        loss_flow = coords.new_tensor(0.0)
        loss_pen  = coords.new_tensor(0.0)
        n_blocks  = 0

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

            # if start == 0:
            #     print_once(f"[FM-win0] Lcur={Lcur}, pad={pad}, validΔ={int(valid_delta.sum())}, validPen={int(valid_pen.sum())}, "
            #             f"std={std.view(-1).tolist()}")

            # 정책
            act_emb   = self.action_embed(A_tau, tau, start_idx=start)      # [B,H,512]
            attn_mask = build_block_mask(Lctx, H, device=device)
            v, _ = self.policy(cond_ctx, act_emb, None, attn_mask)          # v:[B,H,2]
            #v, pen_logits = self.policy(cond_ctx, act_emb, None, attn_mask)      # v:[B,H,2], pen_logits:[B,H,3]

            # ✅ pen은 τ=1에서 따로 예측 (라벨과 시계열 정합)
            tau1 = torch.ones(B, 1, device=device)
            # - A_gt 기준으로 pen 예측 (패딩/마스크는 아래에서 무시됨)
            act_pen = self.action_embed(A_gt, tau1, start_idx=start)        # [B,H,512]
            _, pen_logits = self.policy(cond_ctx, act_pen, None, attn_mask) # pen_logits:[B,H,3]


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

            # ✅ EOS 클래스 가중치(예: 2.0)로 희소 보정
            class_w = torch.tensor([1.0, 1.0, 2.0], device=device)          # 필요시 1.5~3.0 튜닝
            lp = F.cross_entropy(
                pen_logits.reshape(-1, 3),
                tgt_lbl.reshape(-1),
                ignore_index=-100,
                weight=class_w,                                             # ✅
            )
            loss_pen += lp
            n_blocks += 1

        # 평균
        loss_flow = loss_flow / max(n_blocks, 1)
        loss_pen  = loss_pen  / max(n_blocks, 1)
        return loss_flow, loss_pen

    def forward(self, style_imgs, coords, char_img, return_nce: bool=True):
        pre = self._encode_style_once(style_imgs)                     # wtok/gtok + NCE feats
        cond_ctx, Lctx = self._make_cond_ctx(pre["wtok"], pre["gtok"], char_img)
        loss_flow, loss_pen = self.flow_match_loss(coords, cond_ctx, Lctx)
        if return_nce and pre["num_imgs"] >= 2:
            writer_supcon, glyph_supcon = self.nce_losses_supcon(pre)
        else:
            z = (loss_flow.detach()*0)
            writer_supcon, glyph_supcon = z, z
        return {
            "loss_flow": loss_flow,
            "loss_pen":  loss_pen,
            "nce_w":     writer_supcon,
            "nce_g":     glyph_supcon,
        }

    # ---------- 추론: 청크 + 앙상블 ----------
    @torch.no_grad()
    def flow_infer(self, style_imgs, char_img, T:int,
                steps:int=20, stride:int=None, replan:int=None,
                solver:str="euler",               # NEW: "euler" | "rk4"
                micro_pen_ensemble:bool=False,     # NEW: 내부 스텝 pen 앙상블
                micro_pen_weight:str="linear",     # NEW: "linear" | "uniform"
                temporal_ensemble:bool=False        # NEW: 시간축 앙상블 on/off
                ):
        """
        오일러/ RK4 적분으로 Δx,Δy 생성 + pen 확률 → one-hot.
        Temporal ensembling(삼각가중), R-step 재계획.
        return: [B, T, 5]
        """
        device = style_imgs.device
        B = style_imgs.size(0)
        H = self.H
        stride = stride if stride is not None else self.stride_default
        replan = replan if replan is not None else stride

        print_once(f"SDT_FlowWrapper::flow_infer style_imgs: {style_imgs.shape}, {style_imgs.dtype}, B: {B}, char_img: {char_img.shape}, {char_img.dtype}, T: {T}, steps: {steps}, stride: {stride}, replan: {replan}, solver: {solver}, micro_pen_ensemble: {micro_pen_ensemble}, micro_pen_weight: {micro_pen_weight}")

        pre = self._encode_style_once(style_imgs)               # wtok/gtok 생성
        ctx, Lctx = self._make_cond_ctx(pre["wtok"], pre["gtok"], char_img)

        if temporal_ensemble:
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
                print_once(f"[INF-chunk0] h={h}, a.std={a.std().item():.4f}, pen_prob(mean)={pen_summary}")


            R = min(replan, T - t)
            if temporal_ensemble:
                # ---------- 시간축 앙상블(청크 오프셋) ----------
                w = w_tri(h).view(1, h, 1)                                        # [1,h,1]
                acc_xy[:,  t:t+h, :] += w * a
                acc_pen[:, t:t+h, :] += w * pen_prob
                acc_w[:,   t:t+h, :] += w

                # ---------- 앞 R 스텝 확정 ----------
                wsum = acc_w[:, t:t+R, :].clamp_min(1e-8)
                xy   = acc_xy[:, t:t+R, :] / wsum                                  # [B,R,2]
                pen  = acc_pen[:, t:t+R, :] / wsum                                  # [B,R,3]
            else:
                # NEW: 앙상블 OFF — 현재 청크 결과만 사용
                # replan 유지: h개 중 앞 R개만 확정(commit)
                xy  = a[:, :R, :]                                      # [B,R,2]
                pen = pen_prob[:, :R, :]                               # [B,R,3]

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
