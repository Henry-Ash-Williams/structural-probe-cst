import argparse
import os
import sys
from pathlib import Path

import datasets
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.distance import (
    DistanceProbe,
    DistanceProbeDataset,
    distance_collate,
    distance_loss,
)


def main(args):
    device = torch.device("mps")

    dataset = datasets.load_from_disk(args.dataset_path).with_format("numpy")

    train_distance = DistanceProbeDataset(dataset["train"], layer=args.layer)
    train_loader = DataLoader(
        train_distance,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=distance_collate,
    )
    test_distance = DistanceProbeDataset(dataset["test"], layer=args.layer)
    test_loader = DataLoader(
        test_distance,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=distance_collate,
    )

    probe = DistanceProbe(896, args.rank).to(device)
    optim = torch.optim.AdamW(probe.parameters())

    num_train_batches = len(train_loader)
    num_test_batches = len(test_loader)
    num_batches = num_train_batches + num_test_batches
    test_loss = []

    output_dir = Path(args.output_dir) / f"distance/l{args.layer}-r{args.rank}/"

    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        progress_bar = tqdm(
            total=num_batches,
            desc="Training",
            postfix=f"Epoch {epoch + 1}/{args.epochs}",
            leave=False,
        )
        probe.train()
        for hidden, distance, mask in train_loader:
            hidden = hidden.to(device)
            distance = distance.to(device)
            mask = mask.to(device)

            optim.zero_grad()

            predicted_distance = probe(hidden)

            loss = distance_loss(predicted_distance, distance, mask)
            loss.backward()

            optim.step()
            progress_bar.update(1)

        probe.eval()
        progress_bar.set_description_str("Evaluating")
        total_test_loss = 0.0
        for hidden, distance, mask in test_loader:
            hidden = hidden.to(device)
            distance = distance.to(device)
            mask = mask.to(device)

            with torch.no_grad():
                predicted_distance = probe(hidden)

            loss = distance_loss(predicted_distance, distance, mask)
            total_test_loss += loss
            progress_bar.update(1)

        progress_bar.close()
        avg_test_loss = (total_test_loss / num_test_batches).item()
        test_loss.append(avg_test_loss)

        output_file = output_dir / f"{epoch:04d}.pt"
        torch.save(probe.state_dict(), output_file)
        new_best_probe = min(test_loss) == avg_test_loss

        print(
            f"Avg test loss for epoch {epoch + 1:04d}/{args.epochs:04d}: {avg_test_loss:.2f}",
            end=", new best probe!\n" if new_best_probe else "\n",
        )

        if min(test_loss) == avg_test_loss:
            best_probe = output_dir / "best.pt"
            Path.unlink(best_probe, missing_ok=True)
            best_probe.symlink_to(output_file.absolute())


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        prog="train-distance",
        description="Trains a distance probe",
    )

    arg_parser.add_argument(
        "-l",
        "--layer",
        type=int,
        help="which layer the hidden representations are taken from",
    )
    arg_parser.add_argument(
        "-d",
        "--dataset-path",
        type=str,
        help="path to the dataset",
    )
    arg_parser.add_argument(
        "-r",
        "--rank",
        type=int,
        help="rank to which the hidden representations are projected to",
    )
    arg_parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        help="batch size used during training",
    )
    arg_parser.add_argument(
        "-e", "--epochs", type=int, help="number of epochs to train for"
    )
    arg_parser.add_argument(
        "-o", "--output-dir", type=str, help="trained probe save location"
    )
    args = arg_parser.parse_args(sys.argv[1:])
    main(args)
