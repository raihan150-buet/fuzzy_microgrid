import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AttentionEncoder(nn.Module):
    """
    Novelty: Attention-based Temporal State Encoder.
    Processes a sequence of states (history) to extract robust temporal features.
    """
    def __init__(self, state_dim, seq_len, embed_dim=64, num_heads=4):
        super().__init__()
        self.state_proj = nn.Linear(state_dim, embed_dim)
        # Positional encoding is critical for sequence ordering
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        
        # RL requires stable transformers: dropout=0.0 and norm_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 2, 
            dropout=0.0, 
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.output_dim = embed_dim

    def forward(self, state_seq):
        # state_seq: (B, seq_len, state_dim)
        x = self.state_proj(state_seq) + self.pos_embed
        x = self.transformer(x)
        # Global average pooling over the sequence length
        return x.mean(dim=1)

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, seq_len=1, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.seq_len = seq_len
        self.state_dim = state_dim

        if seq_len > 1:
            self.encoder = AttentionEncoder(state_dim, seq_len)
            encoded_dim = self.encoder.output_dim
        else:
            self.encoder = nn.Identity()
            encoded_dim = state_dim

        self.backbone = nn.Sequential(
            nn.Linear(encoded_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        if self.seq_len > 1:
            state = state.view(-1, self.seq_len, self.state_dim)
        x = self.encoder(state)
        x = self.backbone(x)
        mean = self.mean_layer(x)
        log_std = torch.clamp(self.log_std_layer(x), self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        x_t = normal.rsample()
        action = torch.tanh(x_t)

        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob, torch.tanh(mean)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, seq_len=1, hidden_dim=256):
        super().__init__()
        self.seq_len = seq_len
        self.state_dim = state_dim

        if seq_len > 1:
            self.encoder = AttentionEncoder(state_dim, seq_len)
            encoded_dim = self.encoder.output_dim
        else:
            self.encoder = nn.Identity()
            encoded_dim = state_dim

        self.q = nn.Sequential(
            nn.Linear(encoded_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        if self.seq_len > 1:
            state = state.view(-1, self.seq_len, self.state_dim)
        feat = self.encoder(state)
        return self.q(torch.cat([feat, action], dim=1))
