"""
python -m das_conformer.main fit --model.nb_freq 128 --model.nb_classes 2 --data.data_dir ../../birds/conformer/dat.ng.npy
"""
# main.py

from lightning.pytorch.cli import LightningCLI

# simple demo classes for your convenience
from . import model
from .data_npy import NPYDataModule


def cli_main():
    cli = LightningCLI(model.ConformerModel, NPYDataModule)
    # note: don't call fit!!


if __name__ == "__main__":
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block
