# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Experience replay buffer for on-policy GQE training (self-contained)."""

from __future__ import annotations

import pickle
import sys
from collections import deque

from torch.utils.data import Dataset


class ExperienceBuffer:
    """Ring buffer of ``(sequence, energy, behavior_log_prob)`` tuples."""

    def __init__(self, size=sys.maxsize, capacity=1_000_000):
        self.size = size
        self.store = deque(maxlen=capacity)

    def add(self, seq, energy, behavior_log_prob):
        self.store.append((seq, energy, behavior_log_prob))
        if len(self.store) > self.size:
            self.store.popleft()

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.store, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.store = pickle.load(f)

    def __getitem__(self, idx):
        seq, energy, blp = self.store[idx]
        return {"idx": seq, "energy": energy, "old_log_probs": blp}

    def __len__(self):
        return len(self.store)


class ExperienceDataset(Dataset):
    """Repeats the buffer ``repeat`` times to give several updates per rollout."""

    def __init__(self, buffer: ExperienceBuffer, repeat: int):
        self.buffer = buffer
        self.repeat = repeat

    def __getitem__(self, idx):
        return self.buffer[idx % len(self.buffer)]

    def __len__(self):
        return len(self.buffer) * self.repeat
