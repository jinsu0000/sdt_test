import torch
from utils.logger import print_once

### split final output of our model into Mixture Density Network (MDN) parameters and pen state
def get_mixture_coef(output):
    z = output
    z_pen_logits = z[:, 0:3]  # pen state

    # MDN parameters are used to predict the pen moving
    z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = torch.split(z[:, 3:], 20, 1) # 6개의 20차 벡터로 concat된 z를 나눔눔

    # softmax pi weights:
    z_pi = torch.softmax(z_pi, -1) # GMM의 mixture 확률값

    # exponentiate the sigmas and also make corr between -1 and 1.
    z_sigma1 = torch.minimum(torch.exp(z_sigma1), torch.Tensor([500.0]).cuda())
    z_sigma2 = torch.minimum(torch.exp(z_sigma2), torch.Tensor([500.0]).cuda())
    z_corr = torch.tanh(z_corr) #상관계수로 -1~1
    result = [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen_logits]
    return result

### generate the pen moving and state from the predict output
def get_seq_from_gmm(gmm_pred):
    gmm_pred = gmm_pred.reshape(-1, 123)
    [pi, mu1, mu2, sigma1, sigma2, corr, pen_logits] = get_mixture_coef(gmm_pred)
    max_mixture_idx = torch.stack([torch.arange(pi.shape[0], dtype=torch.int64).cuda(), torch.argmax(pi, 1)], 1)
    next_x1 = mu1[list(max_mixture_idx.T)]
    next_x2 = mu2[list(max_mixture_idx.T)]
    pen_state = torch.argmax(gmm_pred[:, :3], dim=-1)
    pen_state = torch.nn.functional.one_hot(pen_state, num_classes=3).to(gmm_pred)
    seq_pred = torch.cat([next_x1.unsqueeze(1), next_x2.unsqueeze(1), pen_state],-1)
    return seq_pred


def get_seq_from_gmm_with_sigma(gmm_pred, noise_scale=0.3):
    gmm_pred = gmm_pred.reshape(-1, 123)
    [pi, mu1, mu2, sigma1, sigma2, corr, pen_logits] = get_mixture_coef(gmm_pred)

    # 가장 확률이 높은 mixture 선택
    max_mixture_idx = torch.stack([torch.arange(pi.shape[0], dtype=torch.int64).cuda(), torch.argmax(pi, 1)], 1)

    # 선택된 mixture의 mu, sigma 가져오기
    selected_mu1 = mu1[list(max_mixture_idx.T)]
    selected_mu2 = mu2[list(max_mixture_idx.T)]
    selected_sigma1 = sigma1[list(max_mixture_idx.T)]
    selected_sigma2 = sigma2[list(max_mixture_idx.T)]

    # ✨ 약간의 noise 추가
    eps1 = torch.randn_like(selected_mu1)
    eps2 = torch.randn_like(selected_mu2)

    # sigma를 아주 작게 반영
    next_x1 = selected_mu1 + eps1 * selected_sigma1 * noise_scale
    next_x2 = selected_mu2 + eps2 * selected_sigma2 * noise_scale

    # pen state는 그대로
    pen_state = torch.argmax(gmm_pred[:, :3], dim=-1)
    pen_state = torch.nn.functional.one_hot(pen_state, num_classes=3).to(gmm_pred)

    # 최종 시퀀스
    seq_pred = torch.cat([next_x1.unsqueeze(1), next_x2.unsqueeze(1), pen_state], -1)

    return seq_pred