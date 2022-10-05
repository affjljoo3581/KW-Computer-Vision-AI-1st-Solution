from __future__ import annotations

import argparse
import glob
import os
import string

import albumentations as A
import albumentations.pytorch as AP
import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader

from data import ImageDataset


def prepare_dataloader(args: argparse) -> tuple[DataLoader, pd.DataFrame]:
    filenames = sorted(glob.glob(os.path.join(args.imagedir, "*.png")))
    indices = [int(os.path.splitext(os.path.basename(x))[0]) for x in filenames]
    labels = pd.DataFrame(indices)

    labels[list(string.ascii_lowercase)] = 0
    labels = labels.set_index(0)
    labels.index.name = "index"

    transform = [
        A.Resize(args.image_size, args.image_size),
        A.Normalize(mean=0.5, std=0.5),
        AP.ToTensorV2(),
    ]
    dataset = ImageDataset(
        filenames=filenames, labels=labels, transform=A.Compose(transform)
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=os.cpu_count(),
        pin_memory=True,
        persistent_workers=True,
    )
    return dataloader, labels


@torch.inference_mode()
def main(args: argparse.Namespace):
    dataloader, labels = prepare_dataloader(args)
    model = torch.load(args.model).cuda().half().eval()

    total_probs = []
    for images, _ in tqdm.tqdm(dataloader):
        images, probs = images.cuda().half(), []
        for i in range(4 if args.use_tta else 1):
            probs.append(model(images.rot90(i, (2, 3))).sigmoid())
        total_probs += (sum(probs) / len(probs)).tolist()

    labels.iloc[:, :] = total_probs
    if not args.return_probs:
        labels = (labels > 0.5).astype(int)
    labels.to_csv(f"{os.path.splitext(os.path.basename(args.model))[0]}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("imagedir")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--use-tta", action="store_true", default=False)
    parser.add_argument("--return-probs", action="store_true", default=False)
    main(parser.parse_args())
