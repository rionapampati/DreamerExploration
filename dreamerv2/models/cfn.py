import torch
import torch.nn as nn

class CoinFlipNetwork(nn.Module):
    """
    Predicts inverse count sqrt(1/N(s)) via Rademacher regression.
    Architecture matches RND's prediction network.
    """
    def __init__(self, input_size, hidden_size=400, d=20):
        super().__init__()
        self.d = d
        # Trainable network
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ELU(),
            nn.Linear(hidden_size, hidden_size), nn.ELU(),
            nn.Linear(hidden_size, d)
        )
        # Frozen random prior for optimistic init
        self.prior = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ELU(),
            nn.Linear(hidden_size, hidden_size), nn.ELU(),
            nn.Linear(hidden_size, d)
        )
        for p in self.prior.parameters():
            p.requires_grad = False

        # Running stats for prior normalization
        self.register_buffer('prior_mean', torch.zeros(d))
        self.register_buffer('prior_var', torch.ones(d))
        self.register_buffer('prior_count', torch.tensor(0.0))

    def update_prior_stats(self, prior_out):
        """Online mean/var update for prior normalization."""
        batch_mean = prior_out.mean(0)
        batch_var = prior_out.var(0)
        self.prior_count += 1
        alpha = 1.0 / self.prior_count
        self.prior_mean = (1 - alpha) * self.prior_mean + alpha * batch_mean
        self.prior_var = (1 - alpha) * self.prior_var + alpha * batch_var

    def normalized_prior(self, x):
        raw = self.prior(x)
        self.update_prior_stats(raw.detach())
        return (raw - self.prior_mean) / (self.prior_var.sqrt() + 1e-8)

    def forward(self, x):
        # x: latent state embedding, shape [B, input_size]
        f_hat = self.net(x)
        f_prior = self.normalized_prior(x)
        f = f_hat + f_prior
        # Exploration bonus: sqrt(1/d * ||f||^2)
        bonus = (f.pow(2).mean(dim=-1)).sqrt()  # [B]
        return f, bonus

    def loss(self, f, coin_flips):
        """MSE against Rademacher samples."""
        return (f - coin_flips).pow(2).mean()