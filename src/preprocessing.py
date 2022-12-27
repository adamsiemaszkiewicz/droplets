# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from pandas import DataFrame, Series

from src import consts

logging.basicConfig(level=logging.INFO)

SEED = 42


def extract_sample_description(filepath: Path) -> Series:
    parts = filepath.parts
    sample_info = parts[0]
    sample_info_parts = sample_info.split("_")

    name = sample_info_parts[0]
    concentration = float(sample_info_parts[1].replace("mgml", ""))

    return pd.Series([name, concentration])


def display_samples(df: DataFrame, n_samples: int, random_state: int = SEED):

    _df = df.sample(n=n_samples, random_state=random_state)
    filepaths = _df["filepath"]
    names = _df["name"]
    concentrations = _df["concentration"]

    fig, axs = plt.subplots(n_samples, 1)

    for ax, fp, n, c in zip(axs, filepaths, names, concentrations):
        image = imread(consts.DATA_DIR / fp)
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(f"{n}: {c}mg/l")

    plt.tight_layout()
    plt.show()


def main():
    absolute_filepaths = sorted(list(consts.DATA_DIR.glob("*/*.jpg")))
    relative_filepaths = [p.relative_to(consts.DATA_DIR) for p in absolute_filepaths]
    df = pd.DataFrame(data=relative_filepaths, columns=["filepath"])
    df[["name", "concentration"]] = df["filepath"].apply(lambda x: extract_sample_description(x))

    display_samples(df=df, n_samples=4)


if __name__ == "__main__":
    main()
