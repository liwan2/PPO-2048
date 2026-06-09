import torch
import torch.nn as nn

from config import N, STATE_DIM, ACTION_DIM


class ActorCritic(nn.Module):
    """归一化 log2 棋盘 + MLP Actor-Critic 模型。"""

    def __init__(self, n=N, state_dim=STATE_DIM, action_dim=ACTION_DIM):
        super().__init__()
        self.n = n
        self.state_dim = state_dim
        self.action_dim = action_dim
        in_features = n * n

        self.backbone = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
        )

        self.actor_head = nn.Linear(128, action_dim)
        self.critic_head = nn.Linear(128, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("tanh"))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def encode(self, x):
        batch = x.size(0)
        normed = x.view(batch, -1).float() / float(self.state_dim - 1)
        return self.backbone(normed)

    def forward(self, x, action_mask=None):
        feat = self.encode(x)
        logits = self.actor_head(feat)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask <= 0, -1e8)
        value = self.critic_head(feat).squeeze(-1)
        return logits, value


class DQN(ActorCritic):
    pass
