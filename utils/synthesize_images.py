from __future__ import annotations

import argparse
import os
import random
import string

import cv2
import numpy as np
import pandas as pd
import tqdm


def transform_letter_image(image: np.ndarray) -> tuple[np.ndarray, float]:
    scale = np.random.uniform(1, 2)
    rotation = np.random.uniform(0, 360)

    center = (image.shape[0] / 2, image.shape[1] / 2)
    transform = cv2.getRotationMatrix2D(center, rotation, scale)

    height, width = np.abs(transform[:, :2]) @ image.shape
    transform[:, 2] += [width / 2 - center[1], height / 2 - center[0]]

    image = cv2.warpAffine(image, transform, (int(width), int(height)))
    if scale > 2.0:
        kernel = np.ones((int(scale), int(scale)))
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return image, scale


def apply_random_effects(image: np.ndarray, scale: float) -> np.ndarray:
    if np.random.random() < 0.3:
        image = (image * np.random.uniform(0.3, 0.7)).astype(np.uint8)
    if np.random.random() < 0.1:
        thickness = np.random.randint(int(scale), int(scale) * 2)
        image = cv2.dilate(image, np.ones((thickness, thickness)))
    return image


def generate_image(letters: dict[str, pd.DataFrame]) -> tuple[np.ndarray, list[str]]:
    canvas = np.random.randint(0, 0xFF, (256, 256))
    labels = sorted(random.sample(string.ascii_lowercase, np.random.randint(10, 15)))

    for label in labels:
        image = letters[label.upper()].sample(1).iloc[0, 1:].to_numpy()
        image = np.clip(image.reshape(28, 28) * 2, 0, 0xFF).astype(np.uint8)

        image, scale = transform_letter_image(image)
        image = apply_random_effects(image, scale)

        x = np.random.randint(0, 256 - image.shape[1])
        y = np.random.randint(0, 256 - image.shape[0])
        canvas[y : y + image.shape[0], x : x + image.shape[1]] += image

    return np.clip(canvas, 0, 0xFF).astype(np.uint8), labels


def main(args: argparse.Namespace):
    os.makedirs(args.output_dir, exist_ok=True)
    letters = dict(list(pd.read_csv(args.mnist_data).iloc[:, 1:].groupby("letter")))

    labels_list = []
    for i in tqdm.trange(args.index_offset, args.index_offset + args.num_images):
        image, labels = generate_image(letters)
        cv2.imwrite(os.path.join(args.output_dir, "{:06d}.png".format(i)), image)

        labels = {x: 1 if x in labels else 0 for x in string.ascii_lowercase}
        labels_list.append({"index": i, **labels})
    pd.DataFrame(labels_list).to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-data", default="resources/mnist_data/test.csv")
    parser.add_argument("--output-dir", default="val_dirty_mnist_2nd")
    parser.add_argument("--output-csv", default="val_dirty_mnist_2nd_answer.csv")
    parser.add_argument("--num-images", type=int, default=5000)
    parser.add_argument("--index-offset", type=int, default=100000)
    main(parser.parse_args())
