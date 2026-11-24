from math import ceil

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

__all__ = [
    "DepthProbe",
    "DepthProbeDataset",
    "LazyDepthProbeDataset",
    "depth_collate",
    "depth_loss",
]


class DepthProbe(nn.Module):
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.proj = nn.Linear(dim, rank, bias=False)
        nn.init.uniform_(self.proj.weight, -0.05, 0.05)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return (self.proj(h) ** 2).sum(-1)


class DepthProbeDataset(Dataset):
    def __init__(self, hf_dataset, layer: int):
        ds = hf_dataset.with_format("numpy", columns=["hidden_representation", "depth"])
        self.ds, self.hs = [], []
        total = ceil(len(ds) / 256)
        for batch in tqdm(ds.iter(batch_size=256), desc="Loading", total=total):
            for h, d in zip(batch["hidden_representation"], batch["depth"]):
                self.hs.append(
                    torch.from_numpy(np.ascontiguousarray(h[:, layer]))
                )  # fp16, (n, D)
                self.ds.append(torch.from_numpy(np.stack(d)))  # int16, (n,)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.hs[i], self.ds[i]


class LazyDepthProbeDataset(Dataset):
    def __init__(self, hf_dataset, layer: int | None = None):
        self.layer = layer

        self.ds = hf_dataset.with_format(
            "numpy", columns=["hidden_representation", "depth"]
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        if self.layer is None:
            raise ValueError("Layer not set")

        hs = torch.from_numpy(
            np.ascontiguousarray(self.ds["hidden_representation"][i][:, self.layer, :])
        )
        ds = torch.from_numpy(np.stack(self.ds["depth"][i]))
        return hs, ds


def depth_collate(batch):
    hs, ds = zip(*batch)
    B, D = len(hs), hs[0].shape[1]
    n_max = max(h.shape[0] for h in hs)

    H = torch.zeros(B, n_max, D)
    depth = torch.zeros(B, n_max)
    mask = torch.zeros(B, n_max, dtype=torch.bool)
    for i, (h, d) in enumerate(zip(hs, ds)):
        n = h.shape[0]
        H[i, :n], depth[i, :n], mask[i, :n] = h, d, True
    return H, depth, mask


def depth_loss(predicted, actual, mask, reduce="mean"):
    abs_diff = (predicted - actual).abs() * mask
    lengths = mask.sum(dim=1)

    per_sentence = abs_diff.sum(dim=1) / lengths

    if reduce == "mean":
        return per_sentence.mean()
    elif reduce == "sum":
        return per_sentence.sum()
    else:
        raise ValueError("Invalid reduction strategy")
