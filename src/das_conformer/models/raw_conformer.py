import torch
from torch import optim, nn
import lightning as L
import torchaudio
import torchmetrics
from nnAudio.features.mel import MelSpectrogram
from typing import Optional


def syllcount_loss(output_labels, target_labels):
    """
    Calculate the syllable count loss between predicted and target labels.

    This function computes a loss based on the character error rate between
    the predicted and target syllable onset sequences. It identifies syllable
    onsets in both the predicted and target labels, converts them to strings,
    and then calculates the character error rate.

    Args:
        output_labels (torch.Tensor): Predicted labels from the model.
        target_labels (torch.Tensor): Ground truth labels.

    Returns:
        torch.Tensor: The computed loss value. Returns 0 if the result is NaN,
                      or 1000 if the result is infinity.

    Note:
        - The function assumes that syllable onsets are represented by
          transitions where the label value increases by 1 or more.
        - The character error rate is used as the metric for comparing
          the predicted and target onset sequences.
    """
    # count onsets
    # nb_syll_true = torch.sum(torch.diff(target_labels) >= 1).float()
    # nb_syll_pred = torch.sum(torch.diff(output_labels) >= 1).float()

    # FIXME: using boolean indices fails because of size mismatch...
    true_onsets = torch.where(torch.diff(target_labels) >= 1)
    pred_onsets = torch.where(torch.diff(output_labels) >= 1)
    # true_onsets[0] = 1
    # pred_onsets[0] += 1
    # prepend = torch.zeros((target_labels.shape[0], 1, target_labels.shape[-3]))
    # true_onsets = torch.diff(target_labels, prepend=prepend) >= 1
    # pred_onsets = torch.diff(output_labels, prepend=prepend) >= 1

    true_labels = target_labels[true_onsets]
    pred_labels = target_labels[pred_onsets]

    # true_labels_str = ''.join([str(int(label)) for label in true_labels])
    # pred_labels_str = ''.join([str(int(label)) for label in pred_labels])
    true_labels_str = "".join(map(str, true_labels))
    pred_labels_str = "".join(map(str, pred_labels))

    loss = torchmetrics.functional.text.char_error_rate(preds=pred_labels_str, target=true_labels_str)

    if torch.isnan(loss):
        loss = 0
    elif torch.isinf(loss):
        loss = 1_000

    return loss


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels, bias=bias, padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class RawConformer(L.LightningModule):
    """
    RawConformer: A PyTorch Lightning module implementing a Conformer-based model for audio processing.

    This class combines a Mel spectrogram layer with a Conformer architecture for processing raw audio input.
    It's designed for tasks such as speech recognition or audio classification.

    Attributes:
        nb_classes (int): Number of output classes.
        sr (float): Sampling rate of the input audio.
        nb_freq (int): Number of mel frequency bins (default: 128).
        n_fft (int): FFT size for spectrogram computation (default: 1024).
        hop_length (int): Hop length for spectrogram computation (default: 512).
        trainable_mel (bool): Whether the mel filterbank is trainable (default: True).
        trainable_STFT (bool): Whether the STFT parameters are trainable (default: False).
        spec_fmin (float): Minimum frequency for mel spectrogram (default: 100 Hz).
        spec_fmax (Optional[float]): Maximum frequency for mel spectrogram (default: None, set to 3/4 of sr).
        num_heads (int): Number of attention heads in the Conformer (default: 4).
        ffn_dim (int): Dimension of the feed-forward network in the Conformer (default: 128).
        num_layers (int): Number of Conformer layers (default: 2).
        depthwise_conv_kernel_size (int): Kernel size for depthwise convolution in Conformer (default: 31).
        loss_weight_xent (float): Weight for cross-entropy loss (default: 0.9).
        lr (float): Learning rate for optimization (default: 1e-4).

    The model first converts the input audio to a mel spectrogram, then processes it through
    a Conformer network. It's suitable for various audio processing tasks and can be fine-tuned
    for specific applications.
    """

    def __init__(
        self,
        nb_classes: int,
        sr: float,
        nb_freq: int = 128,
        n_fft: int = 1024,
        hop_length: int = 512,
        trainable_mel: bool = True,
        trainable_STFT: bool = False,
        spec_fmin: float = 100,
        spec_fmax: Optional[float] = None,
        num_heads: int = 4,
        ffn_dim: int = 128,
        num_layers: int = 2,
        depthwise_conv_kernel_size: int = 31,
        loss_weight_xent: float = 0.9,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.nb_freq = nb_freq
        self.nb_classes = nb_classes
        self.lr = lr

        if spec_fmax is None:
            spec_fmax = (sr * 3) // 4

        self.spec_layer = MelSpectrogram(
            n_fft=n_fft,
            n_mels=nb_freq,
            hop_length=hop_length,
            verbose=0,
            window="hann",
            center=True,
            pad_mode="constant",
            fmin=spec_fmin,
            fmax=spec_fmax,
            sr=sr,
            trainable_mel=trainable_mel,
            trainable_STFT=trainable_STFT,
        )  # Initializing the model

        self.conformer = torchaudio.models.Conformer(
            input_dim=nb_freq,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
        )
        self.loss_weight_xent = loss_weight_xent

        # LINEAR
        self.decoder = nn.Linear(self.nb_freq, self.nb_classes, bias=False)

        # LSTM
        # lstm_hidden_size = 64
        # self.decoder_lstm = nn.LSTM(input_size=self.nb_freq, hidden_size=lstm_hidden_size, batch_first=True, bidirectional=True)
        # self.decoder = nn.Linear(2 * lstm_hidden_size, self.nb_classes, bias=False)

        # CONVOLUTIONAL
        # self.decoder_conv = SeparableConv2d(in_channels=ffn_dim, out_channels=, kernel_size=8)
        # self.decoder_conv = nn.Conv2D(in_channels=self.nb_freq, out_channels=self.nb_classes, kernel_size=(8, self.nb_freq))

        # self.criterion = nn.NLLLoss(reduce=True, reduction="mean")
        self.criterion = nn.CrossEntropyLoss(reduce=True, reduction="mean")
        # self.criterion = nn.BCELoss(reduce=True, reduction="mean")

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.nb_classes, average="macro")
        self.f1score = torchmetrics.F1Score(task="multiclass", num_classes=self.nb_classes, average="macro")
        self.precision = torchmetrics.Precision(task="multiclass", num_classes=self.nb_classes, average="macro")
        self.recall = torchmetrics.Recall(task="multiclass", num_classes=self.nb_classes, average="macro")

    def forward(self, inputs, input_lengths):
        specs = self.spec_layer(inputs).transpose(1, 2)
        input_lengths = torch.full((specs.shape[0],), fill_value=specs.shape[1]).to(self.device)
        output, output_lengths = self.conformer(specs, input_lengths)
        outputs = self.decoder(output)

        # outputs, _ = self.decoder_lstm(output)
        # outputs = self.decoder(outputs)

        # outputs = nn.functional.log_softmax(outputs, dim=-1)

        return outputs, output_lengths

    def step(self, batch):
        inputs, input_lengths, targets, target_lengths = batch
        outputs, output_lengths = self.forward(inputs, input_lengths)
        self.loss = self.criterion(outputs, targets)
        return self.loss

    def training_step(self, batch, batch_idx):
        inputs, input_lengths, targets, target_lengths = batch
        outputs, output_lengths = self.forward(inputs, input_lengths)
        output_labels, target_labels = torch.argmax(outputs, axis=-1), torch.argmax(targets, axis=-1)

        loss_xent = self.criterion(outputs, targets)
        loss_nbsyll = syllcount_loss(output_labels, target_labels)
        loss = self.loss_weight_xent * loss_xent + (1 - self.loss_weight_xent) * loss_nbsyll
        self.log_dict({"train_loss": loss, "crossentropy": loss_xent, "syllcount": loss_nbsyll})
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, input_lengths, targets, target_lengths = batch
        outputs, output_lengths = self.forward(inputs, input_lengths)
        output_labels, target_labels = torch.argmax(outputs, axis=-1), torch.argmax(targets, axis=-1)

        val_xent = self.criterion(outputs, targets)
        val_nbsyll = syllcount_loss(output_labels, target_labels)
        val_loss = self.loss_weight_xent * val_xent + (1 - self.loss_weight_xent) * val_nbsyll
        val_acc = self.accuracy(output_labels, target_labels)
        lr = self.trainer.lr_scheduler_configs[0].scheduler.optimizer.param_groups[0]["lr"]

        self.log_dict(
            {
                "val_loss": val_loss,
                "val_acc": val_acc,
                "crossentropy": val_xent,
                "syllcount": val_nbsyll,
                "lr": lr,
            },
            prog_bar=True,
        )
        return val_loss

    def predict_step(self, batch, batch_idx):
        inputs, input_lengths, targets, target_lengths = batch
        outputs, output_lengths = self.forward(inputs, input_lengths)
        return outputs

    def test_step(self, batch, batch_idx):
        inputs, input_lengths, targets, target_lengths = batch
        outputs, output_lengths = self.forward(inputs, input_lengths)

        output_labels, target_labels = torch.argmax(outputs, axis=-1), torch.argmax(targets, axis=-1)
        test_loss = self.criterion(outputs, targets)
        test_acc = self.accuracy(output_labels, target_labels)
        test_f1 = self.f1score(output_labels, target_labels)
        test_precision = self.precision(output_labels, target_labels)
        test_recall = self.recall(output_labels, target_labels)
        self.log_dict(
            {
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_f1": test_f1,
                "test_precision": test_precision,
                "test_recall": test_recall,
            },
            prog_bar=True,
        )
        return test_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=5,
            cooldown=2,
            min_lr=1e-8,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
            },
        }
