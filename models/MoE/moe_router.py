import torch
import torch.nn as nn
import torch.nn.functional as F
from .moe_config import MoEConfig

class TopKRouter(nn.Module):
    """
    Простой top-k роутер с softmax-гейтингом и опциональным шумом.
    Возвращает:
      - dispatch_info: словарь с индексами/масками для диспетчеризации
      - aux_loss: вспомогательная потеря балансировки
    """
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.jitter = config.router_jitter
        self.proj = nn.Linear(config.n_embd, self.num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        """
        x: (N, C) — батч токенов после LN внутри блока
        """
        logits = self.proj(x)  # (N, E)
        if self.jitter > 0:
            logits = logits + self.jitter * torch.randn_like(logits)

        # softmax для вероятностей выбора экспертов
        scores = F.softmax(logits, dim=-1)  # (N, E)
        # top-k
        topk_scores, topk_idx = torch.topk(scores, k=self.top_k, dim=-1)  # (N, K), (N, K)

        # вспомогательная потеря балансировки (Switch-style)
        # encourage uniform expert usage
        me = scores.mean(0)             # (E,)
        ce = (scores > 0).float().mean(0)  # приближенная активность, тут это mean prob; упростим как mean(scores>0)
        # Но корректнее — как в Switch: importance = sum(scores), load = sum(assignments).
        importance = scores.sum(0) / scores.sum()      # (E,)
        # загрузка как доля маршрутизированных токенов на эксперт
        one_hot_assign = torch.zeros_like(scores)
        one_hot_assign.scatter_(1, topk_idx, 1.0)
        load = one_hot_assign.sum(0) / one_hot_assign.sum()  # (E,)
        aux_loss = (importance * load).sum() * self.num_experts  # чем ближе к 1, тем лучше; берём как цель → минимизируем (importance*load).sum()

        dispatch_info = {
            "topk_idx": topk_idx,         # (N, K)
            "topk_scores": topk_scores,   # (N, K)
        }
        return dispatch_info, aux_loss
