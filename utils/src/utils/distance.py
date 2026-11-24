from math import ceil

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

__all__ = [
    "DistanceProbe",
    "DistanceProbeDataset",
    "LazyDistanceProbeDataset",
    "distance_collate",
    "distance_loss",
]


class DistanceProbe(nn.Module):
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.proj = nn.Linear(dim, rank, bias=False)
        nn.init.uniform_(self.proj.weight, -0.05, 0.05)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        p = self.proj(h)
        sq = (p**2).sum(-1)
        d = sq[:, :, None] + sq[:, None, :] - 2 * p @ p.transpose(1, 2)
        return d.clamp(min=0)


class DistanceProbeDataset(Dataset):
    def __init__(self, hf_dataset, layer: int):
        ds = hf_dataset.with_format(
            "numpy", columns=["hidden_representation", "distance"]
        )
        self.hs, self.ds = [], []
        total = ceil(len(ds) / 256)
        for batch in tqdm(ds.iter(batch_size=256), desc="Loading", total=total):
            for h, d in zip(batch["hidden_representation"], batch["distance"]):
                self.hs.append(
                    torch.from_numpy(np.ascontiguousarray(h[:, layer]))
                )  # fp16, (n, D)
                self.ds.append(torch.from_numpy(np.stack(d)))  # int16, (n, n)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.hs[i], self.ds[i]


class LazyDistanceProbeDataset(Dataset):
    def __init__(self, hf_dataset, layer: int | None = None):
        self.layer = layer

        self.ds = hf_dataset.with_format(
            "numpy", columns=["hidden_representation", "distance"]
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        if self.layer is None:
            raise ValueError("Layer not set")

        hs = torch.from_numpy(
            np.ascontiguousarray(self.ds["hidden_representation"][i][:, self.layer, :])
        )
        ds = torch.from_numpy(np.stack(self.ds["distance"][i]))
        return hs, ds


def distance_collate(batch):
    hs, ds = zip(*batch)
    B, D = len(hs), hs[0].shape[1]
    n_max = max(h.shape[0] for h in hs)

    H = torch.zeros(B, n_max, D)
    Dist = torch.zeros(B, n_max, n_max)
    mask = torch.zeros(B, n_max, dtype=torch.bool)
    for i, (h, d) in enumerate(zip(hs, ds)):
        n = h.shape[0]
        H[i, :n], Dist[i, :n, :n], mask[i, :n] = h, d, True
    return H, Dist, mask


def distance_loss(predicted, actual, mask, reduce: str | None = "mean"):
    pair_mask = mask[:, :, None] & mask[:, None, :]
    abs_diff = (predicted - actual).abs() * pair_mask

    lengths = mask.sum(dim=1).float().clamp(min=1)
    per_sentence = abs_diff.sum(dim=(1, 2)) / lengths**2
    if reduce == "mean":
        return per_sentence.mean()
    elif reduce == "sum":
        return per_sentence.sum()
    elif reduce is None:
        return per_sentence
    else:
        raise ValueError("Invalid reduction strategy")
