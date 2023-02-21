# -*- coding: utf-8 -*-
from flash import Trainer
from flash.image import ImageClassifier

from src import consts
from src.datamodule import DropletsClassificationDataModule


def main():
    dm = DropletsClassificationDataModule(
        data_dir=consts.DATA_DIR,
        dataset_path=consts.DATA_DIR / "dataset_2022-12-27_split.feather",
        img_size=(256, 256),
        batch_size=8,
        num_workers=6,
    )

    model = ImageClassifier(num_classes=dm.num_classes, labels=dm.classes)

    trainer = Trainer(max_epochs=3)
    trainer.finetune(model, datamodule=dm, strategy="freeze")


if __name__ == "__main__":
    main()
