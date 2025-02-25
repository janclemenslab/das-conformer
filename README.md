# DAS Conformer
Prototype of the next version of DAS for segmenting audio using conformer networks. See the [conformer paper](https://arxiv.org/abs/2005.08100).


## Installation

macOS (arm only):
```shell
conda env create -f https://raw.githubusercontent.com/janclemenslab/das-conformer/refs/heads/main/envs/env_macos.yaml
```

windows with cuda (should works for linux, too):
```shell
conda env create -f https://raw.githubusercontent.com/janclemenslab/das-conformer/refs/heads/main/envs/env_other.yaml
```

This will create a conda environment named `das-conformer`.

## Usage
See `docs/zebra_train.ipynb`, `docs/zebra_predict.ipynb`.
