import json
from dataclasses import dataclass
from itertools import batched

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Array3D, Dataset, Features, Sequence, Value
from transformers import AutoModel, AutoTokenizer

from data import Cst

MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B"
PADDING = 400
BATCH_SIZE = 100


# FIXME: This is probably un-necessary
@dataclass
class Item:
    distance: torch.Tensor
    depth: torch.Tensor
    mask: torch.Tensor
    tokens: list[str]
    hidden_representation: torch.Tensor
    language: str


def pad_distance(
    distance: torch.Tensor, pad_to: int = PADDING
) -> tuple[torch.Tensor, torch.Tensor]:
    N, _ = distance.shape

    mask = torch.full((pad_to,), 0, dtype=torch.bool)
    output = torch.full((pad_to, pad_to), 0, dtype=distance.dtype)

    output[:N, :N] = distance
    mask[:N] = 1

    return output, mask


def pad_depth(
    depth: torch.Tensor, pad_to: int = PADDING
) -> tuple[torch.Tensor, torch.Tensor]:
    (N,) = depth.shape

    mask = torch.full((pad_to,), 0, dtype=torch.bool)
    output = torch.full((pad_to,), 0, dtype=depth.dtype)

    output[:N] = depth
    mask[:N] = 1

    return output, mask.to(torch.bool)


def process_batch(batch, tokenizer, model, device):
    batch_size = len(batch)

    distances = []
    depths = []
    masks = []
    fragments = []
    languages = []
    for item in batch:
        if item.num_leaves > PADDING:
            batch_size -= 1
            continue

        distance, mask = pad_distance(torch.tensor(item.distance))
        depth, _ = pad_depth(torch.tensor(item.depth[item.leaf_indicies]))

        distances.append(distance)
        depths.append(depth)
        masks.append(mask)
        fragments.append([leaf.text for leaf in item.leaves])
        languages.append(item.language)

    if not distances:
        return []

    distances = torch.stack(distances)
    depths = torch.stack(depths)
    masks = torch.stack(masks)

    enc = tokenizer(
        fragments,
        is_split_into_words=True,
        max_length=PADDING,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output = model(**enc, output_hidden_states=True)

    # I won't lie, claude wrote this. My implementation took ~4s/batch,
    # this one takes significantly less.
    H = torch.stack(output.hidden_states)  # (L=25, B, T, D)
    word_ids = torch.tensor(
        [
            [wid if wid is not None else -1 for wid in enc.word_ids(i)]
            for i in range(batch_size)
        ]
    ).to(device)  # (B, T), -1 = special/pad token

    # A[b, t, w] = 1/count(w) if token t belongs to word w
    A = F.one_hot(word_ids.clamp(min=0), num_classes=PADDING).to(H.dtype)  # (B, T, W)
    A = A * (word_ids >= 0).unsqueeze(-1)  # drop special/pad tokens
    A = A / A.sum(dim=1, keepdim=True).clamp(min=1)  # mean over subwords

    pooled = torch.einsum("btw,lbtd->lbwd", A, H)  # (L, B, W, D)
    pooled = pooled.cpu()

    items = [
        Item(
            distance=distances[b],
            depth=depths[b],
            mask=masks[b],
            tokens=fragments[b],
            hidden_representation=pooled[:, b],
            language=languages[b],
        )
        for b in range(batch_size)
        if not max(w for w in enc.word_ids(b) if w is not None) + 1 < len(fragments[b])
    ]
    return items


def dataset_generator(data, tokenizer, model, device):
    for batch in batched(data, BATCH_SIZE):
        processed = process_batch(batch, tokenizer)

        for item in processed:
            n = len(item.tokens)
            yield {
                "tokens": item.tokens,
                "distance": item.distance[:n, :n].cpu().numpy().astype(np.int16),
                "depth": item.depth[:n].cpu().numpy().astype(np.int16),
                "hidden_representation": item.hidden_representation[:, :n]
                .permute(1, 0, 2)
                .to(torch.float16)
                .cpu()
                .numpy(),
                "language": item.language,
            }


def main():
    device = torch.device("mps")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)

    with open("../trees.json", "r") as file:
        data = [Cst.from_dict(sample) for sample in json.load(file)]

    features = Features(
        {
            "tokens": Sequence(Value("string")),
            "distance": Sequence(Sequence(Value("int16"))),
            "depth": Sequence(Value("int16")),
            "hidden_representation": Array3D(
                shape=(
                    None,
                    model.config.num_hidden_layers + 1,
                    model.config.hidden_size,
                ),
                dtype="float16",
            ),
            "language": Value("string"),
        }
    )
    ds = Dataset.from_generator(
        dataset_generator,
        gen_kwargs={
            "data": data,
            "tokenizer": tokenizer,
            "model": model,
            "device": device,
        },
        features=features,
        fingerprint=f"{MODEL_NAME.replace('/', '_')}-pad{PADDING}-v1",
    ).train_test_split()

    ds.save_to_disk(f"../dataset/{MODEL_NAME}")


if __name__ == "__main__":
    main()
