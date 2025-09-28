import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizerEMA(nn.Module):
    def __init__(self, n_embed: int, embed_dim: int, decay: float = 0.99, eps: float = 1e-5, commitment_cost: float = 0.25):
        super().__init__()
        self.n_embed = n_embed           # 코드북 크기 K
        self.embed_dim = embed_dim       # 코드 벡터 차원 D
        self.decay = decay               # EMA 지수이동 평균 감쇠
        self.eps = eps                   # 수치 안정성용 작은값
        self.commitment_cost = commitment_cost  # β (z가 코드에 '붙도록' 하는 비용)

        embed = torch.randn(embed_dim, n_embed)   # [D, K] 초기 코드북
        self.register_buffer("embedding", embed)  # [D, K] 학습 파라미터가 아니라 버퍼로 저장(EMA로 갱신)
        self.register_buffer("cluster_size", torch.zeros(n_embed))  # [K] 각 코드가 몇 번 선택됐는지 카운트
        self.register_buffer("embed_avg", embed.clone())            # [D, K] 코드 벡터의 EMA 누적합

    @torch.no_grad()
    def _ema_update(self, z_flat: torch.Tensor, codes: torch.Tensor):
        K = self.n_embed
        onehot = F.one_hot(codes, K).type_as(z_flat)  # [N, K] 각 샘플이 고른 코드의 원-핫
        cluster_size = onehot.sum(0)                  # [K] 코드별 선택 횟수
        embed_sum = z_flat.t() @ onehot               # [D, K] 코드별로 해당 z의 합(평균의 분자 역할)
        
        # EMA 누적
        self.cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(embed_sum,    alpha=1 - self.decay)
        
        # 정규화(“빈 코드” 방지 + 안정화)
        n = self.cluster_size.sum()
        cluster_size = (self.cluster_size + self.eps) / (n + K * self.eps) * n
        
        # 새 코드 = (코드가 선택된 z들의 평균)
        embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)  # [D, K]
        self.embedding.copy_(embed_normalized)

    def forward(self, z: torch.Tensor):
        orig_shape = z.shape
        D = self.embed_dim
        assert orig_shape[-1] == D, f"Last dim must be {D}, got {orig_shape[-1]}"
        z_flat = z.reshape(-1, D)      # [N, D] (배치/시간/공간 축을 모두 펼침)

        # 거리 계산: ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z·e
        z_sq = (z_flat ** 2).sum(1, keepdim=True)                      # [N,1]
        e_sq = (self.embedding ** 2).sum(0, keepdim=True)              # [1,K]
        ze = z_flat @ self.embedding                                   # [N,K]
        dist = z_sq + e_sq - 2 * ze# [N,K]

        codes = torch.argmin(dist, dim=1)                              # [N] 최단거리 코드 인덱스
        z_q = F.embedding(codes, self.embedding.t()).view(orig_shape)  # [...,D]

        # EMA 업데이트 (학습시에만) self.embedding이 [D, K]라서 .t()로 [K, D]로 바꿔서 코드 인덱스 → 임베딩으로 뽑음.
        if self.training:
            self._ema_update(z_flat.detach(), codes.detach())

        # Straight-Through (detach: 코드북 갱신은 그라디언트 없이(EMA 규칙대로) 하려고)
        z_st = z_q + (z - z_q).detach()
        loss = self.commitment_cost * F.mse_loss(z.detach(), z_q) # 순전파는 z_q(양자화된 값)를 쓰지만, 역전파는 z로 흘러가게 만드는 트릭
        #commitment loss(β‖z - sg(z_q)‖²): 인코더 출력 z가 선택한 코드에 ‘붙도록’ 유도

        # loss = F.mse_loss(z, z_q)  # (실험용) commitment_cost 없이 그냥 MSE
        with torch.no_grad():
            K = self.n_embed
            avg_probs = F.one_hot(codes, K).float().mean(0)    # [K] 각 코드가 선택될 확률
            perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum()) # 코드북의 퍼플렉시티(=효과적으로 사용된 코드 개수) 다양성 척도
            usage = (avg_probs > 0).float().sum()             # 실제로 쓰인 코드 개수 (“0회 미사용 코드”가 많은지)

        info = {"indices": codes.view(*orig_shape[:-1]).detach(),
                "perplexity": perplexity.detach(),
                "code_usage": usage.detach()}
        # 출력: 양자화된 특징(STE), VQ 손실(커밋먼트), 로그용 정보.
        return z_st, loss, info


class ResidualVectorQuantizerEMA(nn.Module):
    def __init__(self, n_embed: int, embed_dim: int, n_levels: int = 2, decay: float = 0.99, eps: float = 1e-5, commitment_cost: float = 0.25):
        super().__init__()
        self.levels = nn.ModuleList([
            VectorQuantizerEMA(n_embed, embed_dim, decay=decay, eps=eps, commitment_cost=commitment_cost)
            for _ in range(n_levels)
        ]) # 여러 층의 VQ를 연쇄로 둠(1층이 큰 덩어리, 2층이 남은 잔차, …)

    def forward(self, z: torch.Tensor):
        residual = z
        losses, infos = [], []
        for vq in self.levels:
            z_q, l, info = vq(residual)     # 현재 잔차를 양자화
            losses.append(l)
            infos.append(info)
            residual = residual - z_q.detach()  # 잔차 갱신(다음 레벨이 더 미세하게)
        
        out = z - residual          # 합성된 양자화 결과(STE 적용 후)
        loss = sum(losses)         # 레벨별 커밋먼트 손실 합
        px = torch.stack([i["perplexity"] for i in infos]).mean()
        usage = torch.stack([i["code_usage"] for i in infos]).mean()
        #info = {"perplexity": px, "code_usage": usage, "levels": len(self.levels)}
        info = {
            "perplexity": px,          # Tensor
            "code_usage": usage,       # Tensor
            # "levels": len(self.levels)        # ❌ DP에 실지 말기
            # "levels_info": level_infos        # ❌ DP에 실지 말기
        }
        return out, loss, info  # 모든 레벨 합성 결과(STE는 내부 VectorQuantizerEMA에서 이미 적용됨)


class VQAdapter(nn.Module):
    """
    d_model -> vq_dim -> (VQ/RVQ) -> d_model
    seq_first=True면 [S,B,D], False면 [B,S,D] 입력을 처리합니다.
    """
    def __init__(self, d_model: int, vq_dim: int = 256, n_embed: int = 512, n_levels: int = 1,
                 decay: float = 0.99, commitment_cost: float = 0.25, seq_first: bool = True):
        super().__init__()
        self.seq_first = seq_first
        self.in_proj  = nn.Linear(d_model, vq_dim)
        self.vq = (VectorQuantizerEMA(n_embed, vq_dim, decay=decay, commitment_cost=commitment_cost)
                   if n_levels == 1 else
                   ResidualVectorQuantizerEMA(n_embed, vq_dim, n_levels=n_levels, decay=decay, commitment_cost=commitment_cost))
        self.out_proj = nn.Linear(vq_dim, d_model)

    def forward(self, x):
        if self.seq_first:
            S,B,D = x.shape
            x_ = x.permute(1,0,2)              # [B,S,D]
        else:
            x_ = x                              # [B,S,D]
        y = self.in_proj(x_)            # [B,S,vq_dim]
        y_q, vq_loss, info = self.vq(y) # 양자화(+STE)
        y_out = self.out_proj(y_q)      # [B,S,D] 다시 원 차원으로
        if self.seq_first:
            y_out = y_out.permute(1,0,2)  # 원래대로 [S,B,D]
        return y_out, vq_loss, info
