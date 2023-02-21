# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from flash import DataKeys
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from src.logger import get_logger

_logger = get_logger(__name__)


class DropletsClassificationDataset(Dataset):
    def __init__(
        self,
        inputs: List[Path],
        targets: List[int],
        transforms: Optional[Callable[[Tensor], Dict[str, Tensor]]] = None,
    ):
        self.inputs = inputs
        self.targets = targets
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Dict[DataKeys, Tensor]:
        input = Image.open(self.inputs[idx]).convert("RGB")
        input = torch.from_numpy(input)

        target = torch.from_numpy(self.targets[idx])

        if self.transforms:
            input = self.transforms(input)

        return {DataKeys.INPUT: input, DataKeys.TARGET: target}
