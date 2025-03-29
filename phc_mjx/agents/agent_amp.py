import torch
import torch.nn as nn
import torch.optim as optim
from phc_mjx.agents.agent_im import AgentIm  # reuse imitation agent structure
from phc_mjx.models.amp_discriminator import AMPDiscriminator


class AgentAMP(AgentIm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        self.amp_obs_dim = self._get_amp_obs_dim()
        self.discriminator = AMPDiscriminator(self.amp_obs_dim).to(self.device)
        self.optimizer_disc = optim.Adam(
            self.discriminator.parameters(), lr=cfg.learning.disc_lr
        )

        self.amp_update_every = cfg.learning.amp_update_every
        self.amp_batch_size = cfg.learning.amp_batch_size

        self.amp_real_buffer = []
        self.amp_fake_buffer = []

    def _get_amp_obs_dim(self):
        # You might need to align this with your env's AMP observation logic
        sample_obs = self.env.compute_amp_obs()
        return sample_obs.shape[-1]

    def compute_amp_reward(self, obs):
        amp_obs = self.env.compute_amp_obs(obs)
        return self.discriminator.compute_reward(amp_obs)

    def store_amp_data(self, real_obs, fake_obs):
        self.amp_real_buffer.append(real_obs.detach().cpu())
        self.amp_fake_buffer.append(fake_obs.detach().cpu())

    def update_discriminator(self):
        if len(self.amp_real_buffer) < self.amp_batch_size:
            return {}

        real_batch = torch.cat(self.amp_real_buffer[: self.amp_batch_size]).to(
            self.device
        )
        fake_batch = torch.cat(self.amp_fake_buffer[: self.amp_batch_size]).to(
            self.device
        )

        self.optimizer_disc.zero_grad()
        loss, real_score, fake_score = self.discriminator.compute_loss(
            real_batch, fake_batch
        )
        loss.backward()
        self.optimizer_disc.step()

        self.amp_real_buffer = self.amp_real_buffer[self.amp_batch_size :]
        self.amp_fake_buffer = self.amp_fake_buffer[self.amp_batch_size :]

        return {
            "disc_loss": loss.item(),
            "disc_real_score": real_score,
            "disc_fake_score": fake_score,
        }

    def compute_reward(self, obs, actions):
        return self.compute_amp_reward(obs)

    def update(self, batch):
        # Call base update (e.g., PPO step)
        stats = super().update(batch)

        # Update AMP discriminator
        disc_stats = self.update_discriminator()
        stats.update(disc_stats)

        return stats
