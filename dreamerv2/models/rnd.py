import torch
import torch.nn as nn


class RNDModel(nn.Module):
    """
    Random Network Distillation for intrinsic curiosity.
    Operates on the RSSM model state (concat of deterministic + stochastic),
    so it measures novelty in latent space rather than pixel space.
    """
    def __init__(self, input_size, hidden_size=256, output_size=128):
        super().__init__()

        # Fixed random target
        self.target = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, output_size),
        )

        # Predictor trained to match the target
        self.predictor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, output_size),
        )

        # Freeze target permanently
        for p in self.target.parameters():
            p.requires_grad = False

        # Running normalization stats
        self.register_buffer("reward_running_mean", torch.zeros(1))
        self.register_buffer("reward_running_var", torch.ones(1))
        self.update_count = 0

    def forward(self, latent_state):
        """
        latent_state: [..., input_size] — detached model states
        Returns: intrinsic_reward [...], predictor_features, target_features
        """
        target_feat = self.target(latent_state)
        pred_feat   = self.predictor(latent_state)
        # Per-sample MSE across feature dim = novelty signal
        intrinsic_reward = ((pred_feat - target_feat.detach()) ** 2).mean(dim=-1)
        return intrinsic_reward, pred_feat, target_feat

    def normalize(self, reward):
        """Running mean/variance normalization to keep intrinsic reward scaled sensibly."""
        self.update_count += 1
        batch_mean = reward.mean().detach()
        batch_var  = reward.var().detach().clamp(min=1e-8)
        # Exponential moving average
        self.reward_running_mean = 0.99 * self.reward_running_mean + 0.01 * batch_mean
        self.reward_running_var  = 0.99 * self.reward_running_var  + 0.01 * batch_var
        return (reward - self.reward_running_mean) / (self.reward_running_var.sqrt() + 1e-8)