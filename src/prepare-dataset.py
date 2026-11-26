import json
from itertools import batched

import numpy as np
import torch
from datasets import Array3D, Dataset, Features, Sequence, Value
from transformers import AutoModel, AutoTokenizer
from utils.data import Cst

MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B"
MAX_LEAVES = 350
MAX_TOKENS = 1024
BATCH_SIZE = 128

kept = 0


def process_batch(batch: list[Cst], tokenizer, model, device) -> list[dict]:
    batch = [cst for cst in batch if cst.num_leaves <= MAX_LEAVES]
    if not batch:
        return []

    enc = tokenizer(
        [cst.leaf_chunks for cst in batch],
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding="longest",
        truncation=True,
        max_length=MAX_TOKENS,
        return_tensors="pt",
    )
    B, T = enc["input_ids"].shape

    # pool[b, t, w] = 1 if token t contains part of leaf w's text.
    # Whitespace-only tokens are fed to the model but not pooled.
    pool = torch.zeros(B, T, MAX_LEAVES)
    for b, cst in enumerate(batch):
        for t, w in enumerate(enc.word_ids(b)):
            if w is None:
                continue

            leaf_start = len(cst.leaf_chunks[w]) - len(cst.leaves[w].text)
            token_end = enc["offset_mapping"][b, t, 1].item()
            if token_end > leaf_start:
                pool[b, t, w] = 1

    sink = torch.full((B, 1), tokenizer.convert_tokens_to_ids("<|endoftext|>"))
    input_ids = torch.cat([sink, enc["input_ids"]], dim=1)
    attention_mask = torch.cat([torch.ones_like(sink), enc["attention_mask"]], dim=1)
    pool = torch.cat([torch.zeros(B, 1, MAX_LEAVES), pool], dim=1)

    with torch.no_grad():
        output = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            output_hidden_states=True,
        )

    # Mean over each leaf's tokens, for every layer
    tokens_per_leaf = pool.sum(dim=1)  # (B, W)
    pool = pool / tokens_per_leaf.clamp(min=1).unsqueeze(1)
    pool = pool.to(device=device, dtype=model.dtype)
    pooled = torch.stack(
        [torch.einsum("btw,btd->bwd", pool, h) for h in output.hidden_states],
        dim=2,
    ).cpu()  # (B, W, L, D)

    samples = []
    for b, cst in enumerate(batch):
        n = cst.num_leaves
        # Skip samples where a leaf got no tokens (truncated, or a zero-width leaf)
        if (tokens_per_leaf[b, :n] == 0).any():
            continue

        samples.append(
            {
                "tokens": [leaf.text for leaf in cst.leaves],
                "distance": cst.distance.astype(np.int16),
                "depth": cst.depth[cst.leaf_indicies].astype(np.int16),
                "hidden_representation": pooled[b, :n].to(torch.float16).numpy(),
                "language": cst.language,
            }
        )
    return samples


def dataset_generator(data, tokenizer, model, device):
    global kept
    for batch in batched(data, BATCH_SIZE):
        samples = process_batch(batch, tokenizer, model, device)
        kept += len(samples)
        yield from samples


def main():
    device = torch.device("mps")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)

    with open("./trees.json", "r") as file:
        data = [Cst.from_dict(sample) for sample in json.load(file)]

    # Batch similar lengths together to minimise padding (train_test_split shuffles)
    data.sort(key=lambda cst: len(cst.source))

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
        writer_batch_size=250,
        fingerprint=f"{MODEL_NAME.replace('/', '_')}-v2",
    ).train_test_split()

    ds.save_to_disk(f"./dataset/{MODEL_NAME}")


if __name__ == "__main__":
    main()
