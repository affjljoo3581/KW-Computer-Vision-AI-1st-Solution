from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class ImageDataset(Dataset):
    filenames: list[str]
    labels: pd.DataFrame
    transform: Callable

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = cv2.imread(self.filenames[index], cv2.IMREAD_GRAYSCALE)
        image = self.transform(image=image)["image"]

        label_id = int(os.path.splitext(os.path.basename(self.filenames[index]))[0])
        labels = torch.tensor(self.labels.loc[label_id], dtype=torch.int64)
        return image, labels
