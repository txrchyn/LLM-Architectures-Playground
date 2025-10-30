import math
import torch
import torch.nn as nn
from .moe_config import MoEConfig
from .moe_expert import FeedForwardExpert
from .moe_router import TopKRouter

class MoEFeedForward(nn.Module):
    """
    MoE-FFN: K-лучших экспертов на токен, емкости per-expert, scatter/gather.
    Однопроходная реализация с явным циклом по экспертам для читаемости.
    """
    def __init__(self, config: MoEConfig):
        super().__init__()
        assert config.num_experts >= config.top_k >= 1
        self.config = config
        self.router = TopKRouter(config)
        self.experts = nn.ModuleList([FeedForwardExpert(config) for _ in range(config.num_experts)])

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, C)
        Возвращает:
          y: (B, T, C)
          aux_loss: скалярная потеря балансировки
        """
        B, T, C = x.shape
        N = B * T
        x_flat = x.reshape(N, C)

        dispatch, aux_loss = self.router(x_flat)  # topk_idx, topk_scores
        topk_idx = dispatch["topk_idx"]       # (N, K)
        topk_scores = dispatch["topk_scores"] # (N, K)

        E = self.config.num_experts
        K = self.config.top_k
        # емкость на эксперта
        capacity = math.ceil(self.config.capacity_factor * (N * K) / E)

        # Подготовка выходного тензора
        y_flat = x_flat.new_zeros(N, C)

        # Для каждого эксперта собираем токены
        for e in range(E):
            # какие позиции (токены) направлены в эксперта e?
            # mask_e: (N, K) -> выбираем где topk_idx == e
            mask_e = (topk_idx == e)  # bool
            if not mask_e.any():
                continue

            # индексы всех вхождений для эксперта e
            pos_e = mask_e.nonzero(as_tuple=False)  # (M_e, 2) каждая строка: [n_idx, k_slot]
            if pos_e.numel() == 0:
                continue

            # ограничение capacity: берём первые capacity токенов
            if pos_e.size(0) > capacity:
                pos_e = pos_e[:capacity]

            n_idx = pos_e[:, 0]  # позиции токенов в [0..N)
            k_slot = pos_e[:, 1] # 0..K-1, для коэффициента

            # входы эксперту
            x_e = x_flat.index_select(0, n_idx)  # (Me, C)
            out_e = self.experts[e](x_e)         # (Me, C)

            # взвешивание выходов по соответствующему gate-score
            gate_e = topk_scores[n_idx, k_slot].unsqueeze(-1)  # (Me, 1)
            out_e = out_e * gate_e

            # scatter-add в y_flat
            y_flat.index_add_(0, n_idx, out_e)

        y = y_flat.view(B, T, C)
        return y, aux_loss
