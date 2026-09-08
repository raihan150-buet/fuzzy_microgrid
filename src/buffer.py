import random
import torch
import numpy as np
from collections import deque

class ReplayBuffer:
    """
    Standard replay buffer for off-policy algorithms.
    """
    def __init__(self, capacity=500000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))

        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.float32),
            torch.tensor(r, dtype=torch.float32).unsqueeze(1),
            torch.tensor(s2, dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)
