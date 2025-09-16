# models/vq.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizerEMA(nn.Module):
    """
    EMA 기반 VQ (dead-code 방지용 안정판)
    - K: 코드북 크기, D: 코드 차원
    - beta: commitment 가중치, decay: EMA 계수
    반환: z_q_st, vq_loss, codes  (codes: [*,] 인덱스 텐서)
    """
    def __init__(self, K: int, D: int, beta: float=0.25, decay: float=0.99, eps: float=1e-5):
        super().__init__()
        self.K, self.D, self.beta, self.decay, self.eps = K, D, beta, decay, eps
        self.register_buffer("embedding", torch.randn(K, D))
        self.register_buffer("cluster_size", torch.zeros(K))
        self.register_buffer("embed_avg", torch.randn(K, D))
        with torch.no_grad():
            self.embedding = F.normalize(self.embedding, dim=1)

    def forward(self, z_e):  # [B,T,D] or [N,D]
        orig_shape = z_e.shape
        if z_e.dim() == 3:
            B, T, D = orig_shape
            z = z_e.reshape(-1, D)  # [N,D]
        else:
            z = z_e  # [N,D]

        # 거리 계산 (||z||^2 - 2 z e^T + ||e||^2)
        z_sq = (z**2).sum(1, keepdim=True)                # [N,1]
        e_sq = (self.embedding**2).sum(1, keepdim=True).T # [1,K]
        dist = z_sq - 2 * z @ self.embedding.T + e_sq     # [N,K]

        codes = torch.argmin(dist, dim=1)                 # [N]
        z_q = self.embedding.index_select(0, codes)       # [N,D]
        z_q = z_q.view(*orig_shape)

        # EMA 업데이트
        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(codes, self.K).type_as(z)  # [N,K]
                cluster_size = onehot.sum(0)                  # [K]
                embed_sum = onehot.T @ z                      # [K,D]

                self.cluster_size.mul_(self.decay).add_(cluster_size*(1-self.decay))
                self.embed_avg.mul_(self.decay).add_(embed_sum*(1-self.decay))

                n = self.cluster_size.sum()
                cluster_size = (self.cluster_size + self.eps) / (n + self.K*self.eps) * n
                embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
                self.embedding.copy_(embed_normalized)

        # VQ 손실 (EMA엔 codebook 항 생략해도 되지만 commit 모니터링 위해 유지)
        commit_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = self.beta * commit_loss

        # Straight-Through
        z_q_st = z_e + (z_q - z_e).detach()
        return z_q_st, vq_loss, codes.view(orig_shape[:-1])
