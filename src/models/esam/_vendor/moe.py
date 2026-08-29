# Vendored from https://github.com/Asphyxiate-Rye/E-SAM, model/MoE.py.
# Kept only what Sam_my uses (Expert, ExpertChoiceTokenNoisyTopkRouter,
# ExpertChoiceTokenSparseMoE); dropped unused variants, unused imports, and a
# broken `__main__` demo.
#
# Deviation: upstream baked a fixed `top_k: int` sized for one specific batch
# size. This framework's DataLoaders use drop_last=False, so the last batch
# of an epoch is smaller and a fixed top_k can exceed the available token
# count (RuntimeError from `.topk()`). Replaced with `top_k_ratio: float`,
# resolved against the actual token count on every forward call.
#
# Deviation: routing noise is now training-only. Upstream sampled noise
# unconditionally, so validation/evaluation results depended on random
# expert selection. In eval mode the router uses the clean logits.

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """An MLP is a simple linear layer followed by a non-linearity, i.e. each Expert."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.net(x)


class ExpertChoiceTokenNoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k_ratio):
        super().__init__()
        if not 0.0 < top_k_ratio <= 1.0:
            raise ValueError("top_k_ratio must be in (0, 1].")
        self.top_k_ratio = top_k_ratio
        self.num_experts = num_experts
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)

    def forward(self, mh_output):
        total_tokens = mh_output.shape[0] * mh_output.shape[1]
        top_k = max(1, round(total_tokens * self.top_k_ratio))

        logits = self.topkroute_linear(mh_output).reshape(-1, self.num_experts).T
        if self.training:
            noise_logits = self.noise_linear(mh_output).reshape(-1, self.num_experts).T
            noise = torch.randn_like(logits) * F.softplus(noise_logits)
            logits = logits + noise

        top_k_logits, indices = logits.topk(top_k, dim=-1)
        zeros = torch.full_like(logits, float("-inf"))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices


class ExpertChoiceTokenSparseMoE(nn.Module):
    def __init__(self, n_embed, num_experts, top_k_ratio):
        super().__init__()
        self.router = ExpertChoiceTokenNoisyTopkRouter(n_embed, num_experts, top_k_ratio)
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])

    def forward(self, x):
        bs, seq_len, dim = x.size()
        gating_output, indices = self.router(x)
        flat_x = x.view(-1, x.size(-1))
        final_output = torch.zeros_like(flat_x)

        for i, expert in enumerate(self.experts):
            x_ = flat_x[indices[i]]
            expert_output = expert(x_)

            gating_scores = gating_output[i, indices[i]].unsqueeze(1)
            weighted_output = expert_output * gating_scores

            final_output[indices[i]] += weighted_output.squeeze(1)

        return final_output.reshape(bs, seq_len, dim) + x, indices
