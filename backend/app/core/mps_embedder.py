"""Apple MPS-compatible MobileNetV2 appearance embedder."""

from __future__ import annotations

import numpy as np
import torch
from deep_sort_realtime.embedder.embedder_pytorch import MobileNetv2_Embedder


class MPSMobileNetV2Embedder:
    def __init__(self, max_batch_size: int = 16, bgr: bool = True):
        self.base = MobileNetv2_Embedder(
            half=False,
            max_batch_size=max_batch_size,
            bgr=bgr,
            gpu=False,
        )

        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.base.model.to(self.device)
        self.max_batch_size = max_batch_size

    def predict(self, images: list[np.ndarray]) -> list[np.ndarray]:
        if not images:
            return []

        prepared = [self.base.preprocess(image) for image in images]
        features = []

        for start in range(0, len(prepared), self.max_batch_size):
            batch = torch.cat(
                prepared[start:start + self.max_batch_size],
                dim=0,
            ).to(self.device)

            with torch.inference_mode():
                output = self.base.model(batch)

            features.extend(output.detach().cpu().numpy())

        return features
