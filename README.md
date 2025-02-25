# DAS Conformer
Prototype of the next version of DAS for segmenting audio using conformer networks. See the [conformer paper](https://arxiv.org/abs/2005.08100).


## Installation


### 1. Clone the repo
```shell
git clone https://github.com/janclemenslab/das_conformer
cd das_conformer
```

### 2. Create environment

macOS (arm only):
```shell
conda create -n das-conformer pytorch=2.5 torchvision torchaudio torchinfo h5py matplotlib ipykernel flammkuchen librosa rich lightning=2.5 ffmpeg=6 ipykernel pandas pysoundfile numba xarray pydantic -c conda-forge
```

windows with cuda (should works for linux, too):
```shell
conda create -n das-conformer pytorch=2.5 torchvision torchaudio pytorch-cuda=12.4 torchinfo h5py matplotlib ipykernel flammkuchen librosa rich lightning=2.5 ffmpeg=6 ipykernel pandas pysoundfile numba xarray pydantic -c conda-forge -c nvidia
```

### 3. Install additional dependencies and das-conformer
```shell
conda activate das-conformer
pip install "lightning[pytorch-extra]"==2.5 nnAudio pytorch-tcn vocalpy crowsetta
pip install -e . --no-deps --force --upgrade
```


## Usage
See `docs/zebra_train.ipynb`, `docs/zebra_predict.ipynb`.
