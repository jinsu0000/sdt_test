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

        # --- NEW: Pen/EOC & DTW 하이퍼 (필요 시 실험에서 조정 가능) ---
        self.eoc_tail: int   = 0  # EOC tail disabled for fixed-length training     # EOC 뒤 추가 감독 스텝(흡수 꼬리)
        self.focal_gamma: float = 2.0
        self.pen_class_w = nn.Parameter(torch.tensor([1.0, 1.0, 2.0]), requires_grad=False)
        self.dtw_weight: float = 0.10    # soft-DTW 보조손실 가중치(0이면 꺼짐)
        self.dtw_gamma:  float = 0.05    # soft-min 온도(작을수록 DTW에 가까움)
 
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

    
    # ---------- focal CE (ignore_index 지원) ----------
    def _focal_ce(self, logits: torch.Tensor, target: torch.Tensor,
                  ignore_index: int = -100, class_w: torch.Tensor = None, gamma: float = 2.0):
        """
        logits: [N, C], target: [N]
        """
        C = logits.size(-1)
        valid = (target != ignore_index)
        if not valid.any():
            return logits.new_tensor(0.0)
        # 기본 CE(가중치 없이): -log p_y
        logp = torch.log_softmax(logits[valid], dim=-1)         # [Nv, C]
        tgtv = target[valid]
        ce_unred = torch.nn.NLLLoss(reduction="none")(logp, tgtv)  # [Nv]
        # p_y
        p_t = torch.gather(torch.softmax(logits[valid], dim=-1), 1, tgtv.view(-1,1)).squeeze(1)  # [Nv]
        mod = (1.0 - p_t).pow(gamma)
        loss = mod * ce_unred
        if class_w is not None:
            w = class_w.to(logits)[tgtv]
            loss = loss * w
        return loss.mean()

    # ---------- soft-DTW (소형 H에서 배치 루프) ----------
    def _soft_dtw_single(self, X: torch.Tensor, Y: torch.Tensor, gamma: float):
        """
        X: [m,2], Y: [n,2]
        반환: soft-DTW 스칼라
        """
        m, n = X.size(0), Y.size(0)
        if m == 0 or n == 0:
            return X.new_tensor(0.0)
        # 제곱거리 행렬
        D = torch.cdist(X, Y, p=2).pow(2)  # [m,n]
        INF = 1e6
        R = X.new_full((m+1, n+1), INF)
        R[0,0] = 0.0
        for i in range(1, m+1):
            # 벡터화된 j loop도 가능하지만 H가 작으므로 명시 루프로 안정성 우선
            for j in range(1, n+1):
                r0 = R[i-1, j]
                r1 = R[i,   j-1]
                r2 = R[i-1, j-1]
                sm = -gamma * torch.logsumexp(torch.stack([ -r0/gamma, -r1/gamma, -r2/gamma ]), dim=0)
                R[i, j] = D[i-1, j-1] + sm
        return R[m, n]    

    # ---------- 학습 로스 ----------
    def flow_match_loss(self, coords, cond_ctx, Lctx:int,
                        tau_beta=(2.0,4.0), stride:int=None):
        """
        coords: [B, T, 5]  (dx,dy, one-hot(3))
        - Δx,Δy: MSE는 EOC '직전까지만' 유효
        - pen  : CE는 EOC '포함'까지 유효
        - EOC 이후/패딩은 전부 무시
        """
        device = coords.device
        B, T, C = coords.shape
        H = self.H
        stride = stride or self.stride_default
        print_once(f"SDT_FlowWrapper::flow_match_loss B: {B}, T: {T}, C: {C}, H: {H}, stride: {stride}, cond_ctx: {cond_ctx.shape}, {cond_ctx.dtype}")

        loss_flow = coords.new_tensor(0.0)
        loss_pen  = coords.new_tensor(0.0)
        n_blocks  = 0
        # ✅ pen accuracy 집계
        pen_correct = coords.new_tensor(0.0)
        pen_count   = coords.new_tensor(0.0)

        # ---- 시퀀스 전체에서 EOC 위치 산출 ----
        pen_1hot  = coords[..., 2:]                  # [B,T,3]
        eoc_full  = (pen_1hot[..., 2] == 1)         # [B,T]  (EOC는 class=2)
        has_eoc   = eoc_full.any(dim=1)             # [B]
        # 첫 EOC 인덱스 (없으면 T로 설정)
        first_eoc = torch.where(
            has_eoc, eoc_full.float().argmax(dim=1), torch.full((B,), T, device=device, dtype=torch.long)
        ) # [B]
        print_once(f"SDT_FlowWrapper::flow_match_loss B={B} T={T} H={H} stride={stride} | EOC%={(has_eoc.float().mean()*100):.1f} | first_eoc(mean)={first_eoc.float().mean().item():.1f}")

        # Δ 유효:  t < first_eoc
        idxs = torch.arange(T, device=device)[None, :].expand(B, T)
        valid_delta_full = (idxs < first_eoc[:, None])           # [B,T]

        # pen 유효: t <= first_eoc (EOC 프레임 포함) + ✅ absorbing tail (EOC 뒤 K스텝 추가 감독)
        valid_pen_full   = (idxs <= first_eoc[:, None])  # [B,T]
        if getattr(self, "eoc_tail", 0) > 0:
            tail = (idxs > first_eoc[:, None]) & (idxs <= first_eoc[:, None] + self.eoc_tail)
            valid_pen_full = valid_pen_full | tail

        class_w = self.pen_class_w.to(device)  # [3]  # EOC 가중(필요시 튜닝)
        focal_gamma = float(getattr(self, "focal_gamma", 2.0))
        use_tail = int(getattr(self, "eoc_tail", 0))
        use_dtw  = float(getattr(self, "dtw_weight", 0.0)) > 0.0
        dtw_w    = float(getattr(self, "dtw_weight", 0.0))
        dtw_tau  = float(getattr(self, "dtw_gamma", 0.05))

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

            # ---- EOC 마스크를 윈도우에 맞게 슬라이스 ----
            valid_delta = valid_delta_full[:, start:end]      # [B,L<=H]
            valid_pen   = valid_pen_full[:,   start:end]      # [B,L<=H]
            if pad > 0:
                valid_delta = torch.cat([valid_delta, torch.zeros(B, pad, device=device, dtype=torch.bool)], dim=1)
                valid_pen   = torch.cat([valid_pen,   torch.zeros(B, pad, device=device, dtype=torch.bool)], dim=1)

            # 윈도우가 아예 전부 무효(=EOC 이후/패딩뿐)면 스킵
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

            # 정책
            act_emb   = self.action_embed(A_tau, tau, start_idx=start)      # [B,H,512]
            attn_mask = build_block_mask(Lctx, H, device=device)
            v, _ = self.policy(cond_ctx, act_emb, None, attn_mask)          # v:[B,H,2]

            # ✅ pen은 τ=1에서 따로 예측 (라벨과 시계열 정합)
            tau1 = torch.ones(B, 1, device=device)
            act_pen = self.action_embed(A_gt, tau1, start_idx=start)        # [B,H,512]
            v1, pen_logits = self.policy(cond_ctx, act_pen, None, attn_mask) # v1:[B,H,2], pen_logits:[B,H,3]

            _check_finite(v, "policy output v")
            _check_finite(pen_logits, "policy output pen_logits")
            
            self._update_diag_attn(Lctx=Lctx, H=H, prefix_has_content=True)

            # ---- Δ 손실 (EOC 이전만) ----
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

            
            # ---- pen CE (2-class Move/Up; EOC 프레임은 로스에서 제외) ----
            # GT 2-class 라벨 구성
            is_up = (P_gt[..., 1] == 1)                                    # Up=1 위치
            tgt2 = torch.where(is_up, torch.ones_like(is_up, dtype=torch.long), torch.zeros_like(is_up, dtype=torch.long))  # [B,H] in {0,1}
            ignore2 = (~valid_pen) | (pad_mask == 0) | (P_gt[..., 2] == 1)  # pad/EOC 제외
            tgt2 = tgt2.masked_fill(ignore2, -100)
            # 3-class 로짓에서 앞의 2채널만 사용
            lp = self._focal_ce(
                pen_logits[...,:2].reshape(-1, 2),
                tgt2.reshape(-1),
                ignore_index=-100,
                class_w=torch.tensor([1.0, 1.0], device=device),
                gamma=focal_gamma
            )
            loss_pen += lp


            # 유효 프레임 마스크
            m_pen = (~ignore_mask)  # [B,H]

            # 확률과 EOC 채널
            p = pen_logits.softmax(-1)                  # [B,H,3]
            p_eoc = p[..., 2]                           # [B,H]

            # 유효 프레임에서만 정규화(soft-argmax 가중)
            pe = p_eoc * m_pen.float()                  # [B,H]
            Z  = pe.sum(dim=1, keepdim=True).clamp_min(1e-8)
            w_pos = pe / Z                              # [B,H]

            # 예측 EOC 인덱스(연속값)
            idx = torch.arange(H, device=device, dtype=w_pos.dtype).view(1, H)  # [1,H]
            pred_eoc_idx = (w_pos * idx).sum(dim=1)          # [B]

            # GT EOC 인덱스(윈도우 좌표) — 없으면 유효 프레임 마지막으로 대체
            gt_eoc = (P_gt[..., 2] == 1) & m_pen              # [B,H]
            valid_rows = (m_pen.float().sum(dim=1) > 0)       # [B]

            # 각 배치에서 유효 프레임 길이(윈도우 내)
            valid_len = m_pen.float().sum(dim=1)              # [B]
            fallback_idx = (valid_len - 1).clamp_min(0).long()

            # 첫 EOC 위치(없으면 fallback)
            has_eoc = gt_eoc.any(dim=1)
            first_eoc_idx = gt_eoc.float().argmax(dim=1)      # [B] (없으면 0)
            gt_idx_local = torch.where(has_eoc, first_eoc_idx, fallback_idx).float()  # [B]

            # L1 위치 손실(유효한 행만)
            if valid_rows.any():
                loc_loss = F.l1_loss(pred_eoc_idx[valid_rows], gt_idx_local[valid_rows])
                loss_pen += 0.1 * loc_loss   # <- 가중치(0.05~0.2 권장)

            # ✅ pen acc 집계
            with torch.no_grad():
                valid = (tgt_lbl != -100)                                   # [B,H]
                if valid.any():
                    pred = pen_logits.argmax(-1)                            # [B,H]
                    pen_correct += (pred[valid] == tgt_lbl[valid]).float().sum()
                    pen_count   += valid.float().sum()

            # ---- soft-DTW 보조 손실 (좌표 정렬 압력; 유효 Δ만 사용) ----
            if use_dtw and (m_delta.sum() > 1):
                # 배치별 유효 길이만큼 잘라 누적 위치(절대좌표) 생성
                dtw_sum = coords.new_tensor(0.0)
                dtw_cnt = 0
                for b in range(B):
                    Lb = int(m_delta[b].sum().item())
                    if Lb > 1:
                        # 예측 경로: v1(τ=1) 누적합 / GT 경로: A_gt 누적합
                        pred_xy = torch.cumsum(v1[b, :Lb, :], dim=0)   # [Lb,2]
                        gt_xy   = torch.cumsum(A_gt[b, :Lb, :], dim=0) # [Lb,2]
                        dtw_sum = dtw_sum + self._soft_dtw_single(pred_xy, gt_xy, gamma=dtw_tau)
                        dtw_cnt += 1
                if dtw_cnt > 0:
                    loss_flow = loss_flow + dtw_w * (dtw_sum / float(dtw_cnt))

            n_blocks += 1

        # 평균
        loss_flow = loss_flow / max(n_blocks, 1)
        loss_pen  = loss_pen  / max(n_blocks, 1)
        pen_acc   = (pen_correct / pen_count.clamp_min(1.0)).detach()
        return loss_flow, loss_pen, pen_acc

    def forward(self, style_imgs, coords, char_img, return_nce: bool=True):
        pre = self._encode_style_once(style_imgs)                     # wtok/gtok + NCE feats
        cond_ctx, Lctx = self._make_cond_ctx(pre["wtok"], pre["gtok"], char_img)
        loss_flow, loss_pen, pen_acc = self.flow_match_loss(coords, cond_ctx, Lctx)
        if return_nce and pre["num_imgs"] >= 2:
            writer_supcon, glyph_supcon = self.nce_losses_supcon(pre)
        else:
            z = (loss_flow.detach()*0)
            writer_supcon, glyph_supcon = z, z
        return {
            "loss_flow": loss_flow,
            "loss_pen":  loss_pen,
            "pen_acc":   pen_acc,    # ✅ 추가
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
        eoc_t = torch.full((B,), -1, device=device, dtype=torch.long)

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

            # EOC 처리
            for b in range(B):
                if eoc_t[b] >= 0:
                    step_chunk[b, :, :2] = 0.0
                    step_chunk[b, :, 2:] = torch.tensor([0., 0., 1.], device=device)
                else:
                    idx = (pen_oh[b].argmax(-1) == 2).nonzero(as_tuple=False)
                    if idx.numel() > 0:
                        first = int(idx[0].item())
                        eoc_t[b] = t + first
                        if first + 1 < R:
                            step_chunk[b, first+1:, :2] = 0.0
                            step_chunk[b, first+1:, 2:] = torch.tensor([0., 0., 1.], device=device)

            executed.append(step_chunk)
            t += R
            
            if (eoc_t >= 0).all():
                break

        out = torch.cat(executed, dim=1)[:, :T, :]                            # [B,T,5]

        # 최종 EOC 보강
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
