# -*- coding: utf-8 -*-
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.image import imread
from pandas import DataFrame, Series
from tqdm import tqdm

from src import consts

tqdm.pandas()
logging.basicConfig(level=logging.INFO)

SEED = 42


def extract_sample_description(filepath: Path) -> Series:

    sample_info = filepath.parent.name
    sample_info_parts = sample_info.split("_")

    name = sample_info_parts[0]
    concentration = float(sample_info_parts[1].replace("mgml", ""))

    return pd.Series([name, concentration])


def display_samples(df: DataFrame, n_samples: int, random_state: int = SEED) -> None:

    _df = df.sample(n=n_samples, random_state=random_state)
    filepaths = _df["filepath"]
    names = _df["name"]
    concentrations = _df["concentration"]

    fig, axs = plt.subplots(n_samples, 1)

    for ax, fp, n, c in zip(axs, filepaths, names, concentrations):
        image = imread(fp)
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(f"{n}: {c}mg/l")

    plt.tight_layout()
    plt.show()


def calculate_sample_statistics(filepath: Path) -> Series:
    image = imread(filepath)
    mean = np.mean(image)
    std = np.std(image)

    return Series([mean, std])


def main():
    absolute_filepaths = sorted(list(consts.DATA_DIR.glob("*/*.jpg")))
    df = pd.DataFrame(data=absolute_filepaths, columns=["filepath"])
    df[["name", "concentration"]] = df["filepath"].progress_apply(lambda x: extract_sample_description(filepath=x))
    df[["mean", "std"]] = df["filepath"].progress_apply(lambda x: calculate_sample_statistics(filepath=x))
    display_samples(df=df, n_samples=4)

    df["filepath"] = df["filepath"].progress_apply(lambda x: x.relative_to(consts.DATA_DIR).as_posix())

    out_fp = consts.DATA_DIR / f"dataset_{datetime.now().date().isoformat()}.feather"
    df.to_feather(path=out_fp)


if __name__ == "__main__":
    main()
