def syllcount_loss(output_labels, target_labels):
    # count onsets
    nb_syll_true = torch.sum(torch.diff(target_labels) == 1).float()
    nb_syll_pred = torch.sum(torch.diff(output_labels) == 1).float()
    loss = torch.sum(torch.square(nb_syll_true - nb_syll_pred), dim=-1)
    return loss


class ConformerModel(pl.LightningModule):
    def __init__(
        self,
        nb_freq: int,
        nb_classes: int,
        num_heads: int = 4,
        ffn_dim: int = 128,
        num_layers: int = 2,
        depthwise_conv_kernel_size: int = 31,
        loss_weight_xent: float = 0.9,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.nb_freq = nb_freq
        self.nb_classes = nb_classes
        self.lr = lr
        self.conformer = torchaudio.models.Conformer(
            input_dim=nb_freq,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
        )
        self.loss_weight_xent = loss_weight_xent
        self.decoder = nn.Linear(self.nb_freq, self.nb_classes, bias=False)
        # self.decoder = nn.LSTM(input_size=self.nb_freq, hidden_size=self.nb_classes, batch_first=True)

        self.criterion = nn.CrossEntropyLoss(reduce=True, reduction="mean")
        # self.criterion = nn.BCELoss(reduce=True, reduction="mean")

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.nb_classes, average="macro")
        self.f1score = torchmetrics.F1Score(task="multiclass", num_classes=self.nb_classes, average="macro")
        self.precision = torchmetrics.Precision(task="multiclass", num_classes=self.nb_classes, average="macro")
        self.recall = torchmetrics.Recall(task="multiclass", num_classes=self.nb_classes, average="macro")

    def forward(self, inputs, input_lengths):
        output_frames, output_lengths = self.conformer(inputs, input_lengths)
        dec = self.decoder(output_frames)
        outputs = nn.functional.log_softmax(dec, dim=-1)
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


conformer = ConformerModel(
    nb_freq=params["n_mels"], nb_classes=params["nb_classes"], num_layers=params["num_layers"], lr=params["lr"]
)
torchinfo.summary(conformer)

torchinfo.summary(conformer, input_data=batch[:2])
