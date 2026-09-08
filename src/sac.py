import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from src.models import GaussianPolicy, QNetwork

class SACAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        seq_len=1,
        gamma=0.99,
        tau=0.005,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        target_entropy=None,
        device="cpu",
        max_episodes=1000
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        
        # Policy
        self.policy = GaussianPolicy(state_dim, action_dim, seq_len).to(self.device)
        
        # Critics
        self.q1 = QNetwork(state_dim, action_dim, seq_len).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim, seq_len).to(self.device)
        self.q1_target = QNetwork(state_dim, action_dim, seq_len).to(self.device)
        self.q2_target = QNetwork(state_dim, action_dim, seq_len).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Optimizers
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=actor_lr)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=critic_lr)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=critic_lr)

        # Learning Rate Schedulers
        # Note: Aggressive LR scheduling (like CosineAnnealing to 1e-5) often causes 
        # catastrophic forgetting in SAC because the agent loses plasticity.
        # Switched to ConstantLR to maintain stability while keeping the DL practice hooks.
        self.policy_scheduler = optim.lr_scheduler.ConstantLR(self.policy_opt, factor=1.0)
        self.q1_scheduler = optim.lr_scheduler.ConstantLR(self.q1_opt, factor=1.0)
        self.q2_scheduler = optim.lr_scheduler.ConstantLR(self.q2_opt, factor=1.0)

        # Entropy tuning
        self.target_entropy = -float(action_dim) if target_entropy is None else target_entropy
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=alpha_lr)
        
    def step_schedulers(self):
        """Advances the learning rate schedulers. Call at end of each episode."""
        self.policy_scheduler.step()
        self.q1_scheduler.step()
        self.q2_scheduler.step()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, evaluate=False):
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if evaluate:
                _, _, action = self.policy.sample(state_t)
            else:
                action, _, _ = self.policy.sample(state_t)
        return action.cpu().numpy()[0]

    def train_step(self, replay, batch_size=256):
        if len(replay) < batch_size:
            return None

        s, a, r, s2, d = replay.sample(batch_size)
        s = s.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        s2 = s2.to(self.device)
        d = d.to(self.device)

        with torch.no_grad():
            a2, logp2, _ = self.policy.sample(s2)
            q_next = torch.min(
                self.q1_target(s2, a2),
                self.q2_target(s2, a2)
            ) - self.alpha.detach() * logp2
            y = r + self.gamma * (1.0 - d) * q_next

        # Critic updates
        q1_loss = F.mse_loss(self.q1(s, a), y)
        q2_loss = F.mse_loss(self.q2(s, a), y)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
        self.q2_opt.step()

        # Actor update
        new_a, logp, _ = self.policy.sample(s)
        q_new = torch.min(self.q1(s, new_a), self.q2(s, new_a))
        policy_loss = (self.alpha.detach() * logp - q_new).mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_opt.step()

        # Alpha update
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # Target updates
        self.soft_update(self.q1_target, self.q1)
        self.soft_update(self.q2_target, self.q2)

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item())
        }

    def soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    def save(self, folder="sac_model"):
        os.makedirs(folder, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(folder, "policy.pth"))
        torch.save(self.q1.state_dict(), os.path.join(folder, "q1.pth"))
        torch.save(self.q2.state_dict(), os.path.join(folder, "q2.pth"))
        torch.save(self.log_alpha.detach().cpu(), os.path.join(folder, "log_alpha.pt"))
