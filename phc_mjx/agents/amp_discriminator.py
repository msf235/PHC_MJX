import torch
import torch.nn as nn
import torch.nn.functional as F


class AMPDiscriminator(nn.Module):
    def __init__(self, obs_dim: int, hidden_dims=(1024, 512)):
        super().__init__()
        layers = []
        input_dim = obs_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(input_dim, hdim))
            layers.append(nn.ReLU())
            input_dim = hdim
        layers.append(nn.Linear(input_dim, 1))  # Output is a single logit
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        logits = self.net(obs)
        prob = torch.sigmoid(logits)
        return prob

    def compute_loss(self, real_obs, fake_obs):
        # Discriminator loss
        real_pred = self.forward(real_obs)
        fake_pred = self.forward(fake_obs)

        real_loss = F.binary_cross_entropy(real_pred, torch.ones_like(real_pred))
        fake_loss = F.binary_cross_entropy(fake_pred, torch.zeros_like(fake_pred))

        loss = real_loss + fake_loss
        return loss, real_pred.mean().item(), fake_pred.mean().item()

    def compute_reward(self, fake_obs):
        # Return the log-based reward for the policy
        with torch.no_grad():
            d = self.forward(fake_obs)
            reward = -torch.log(1 - d + 1e-8)
        return reward
