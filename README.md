# DAS Conformer
Prototype of the next version of DAS for segmenting audio using conformer networks [paper](https://arxiv.org/abs/2005.08100).


## Installation

```shell
git clone https://github.com/janclemenslab/das_conformer
cd das_conformer
```


macos Mx
```shell
conda create -n das-conformer pytorch=2.5 torchvision torchaudio torchinfo h5py matplotlib ipykernel flammkuchen librosa rich lightning=2.5 ffmpeg=6 ipykernel pandas pysoundfile numba xarray pydantic -c conda-forge
```

on windows with cuda (haven't tested this recently):
```shell
conda create -n das-conformer pytorch=2.5 torchvision torchaudio pytorch-cuda=12.4 torchinfo h5py matplotlib ipykernel flammkuchen librosa rich lightning=2.5 ffmpeg=6 ipykernel pandas pysoundfile numba xarray pydantic -c conda-forge -c nvidia
```

then
```shell
conda activate das-conformer
pip install "lightning[pytorch-extra]"==2.5 nnAudio pytorch-tcn vocalpy crowsetta
pip install -e . --no-deps --force --upgrade
```


## Usage
See `docs/zebra_train.ipynb`, `docs/zebra_predict.ipynb`.
