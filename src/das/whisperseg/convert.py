"""Convert a whisperseg-aer .pt bundle to the DAS checkpoint format."""

import argparse

from .model import convert_aer_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Converted whisperseg-aer .pt bundle")
    parser.add_argument("destination", help="DAS .ckpt output path")
    args = parser.parse_args()
    print(convert_aer_bundle(args.source, args.destination))


if __name__ == "__main__":
    main()
