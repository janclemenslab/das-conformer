# Models and data

## Model files

- Conformer, TCN, and TweetyNet use DAS `.ckpt` files.
- Legacy DAS `*_model.h5` plus `*_params.yaml` models still load directly for prediction. `das convert-legacy` can convert them to `.ckpt`.
- WhisperSeg uses DAS `.ckpt` files for training and prediction. Original WhisperSeg checkpoints and Hugging Face model IDs are not loaded at runtime.

The converted `whisperseg-aer/v1` `.pt` bundles can be repacked once, without retraining:

```shell
python -m das.whisperseg.convert /path/to/whisperseg-aer.pt /path/to/whisperseg-aer.ckpt
```

The `.pt` and `.ckpt` extensions are conventions; their contents determine whether DAS can load them. The repacking step adds DAS checkpoint metadata while preserving the model weights, configuration, and tokenizer.

## Training data

DAS trains directly from annotated WAV folders and continues to accept prebuilt `.npy`, H5, and Zarr DAS datasets. The existing dataset-generation path remains for other supported audio containers. Single- and multi-channel audio remain supported.

Embeddings, generic transfer learning, self-supervised training, automatic threshold grid search, and binary app downloads are outside this pre-release.
