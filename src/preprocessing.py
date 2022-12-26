# -*- coding: utf-8 -*-
import logging

from src import consts

logging.basicConfig(level=logging.INFO)


def main():
    files = sorted(list(consts.DATA_DIR.glob("*")))
    logging.info(files)


if __name__ == "__main__":
    main()
