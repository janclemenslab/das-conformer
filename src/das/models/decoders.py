from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import nn


@dataclass
class DecoderConfig:
    """Decoder options.

    Args:
        type: Prediction head architecture to use. WhisperSeg is a backend sentinel, not a native decoder module.
        hidden_size: Hidden size for the LSTM decoder.
        kernel_size: Kernel size for the convolutional decoder.
        num_heads: Number of attention heads for the attention or timestamp decoder.
        num_layers: Number of attention layers in the attention or timestamp decoder.
        dropout: Dropout rate for the attention, timestamp, or WhisperSeg decoder.
        max_length: Maximum timestamp or WhisperSeg decoder token length during training.
        generation_max_length: Maximum WhisperSeg decoder token length during prediction.
        num_trials: Number of WhisperSeg prediction trials.
        num_beams: Number of WhisperSeg generation beams.
        top_k: WhisperSeg generation top-k value.
        top_p: WhisperSeg generation top-p value.
        length_penalty: WhisperSeg generation length penalty.
    """

    type: Literal["linear", "lstm", "conv", "attention", "timestamp", "legacy_linear_upsample", "whisperseg"] = "linear"
    hidden_size: int = 64
    kernel_size: int = 8
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    max_length: int = 100
    generation_max_length: int = 448
    num_trials: int = 1
    num_beams: int = 4
    top_k: int = 1
    top_p: float = 1.0
    length_penalty: float = 1.0


@dataclass
class LinearDecoderConfig:
    type: Literal["linear"] = "linear"


@dataclass
class LSTMDecoderConfig:
    type: Literal["lstm"] = "lstm"
    hidden_size: int = 64


@dataclass
class ConvDecoderConfig:
    type: Literal["conv"] = "conv"
    kernel_size: int = 8


@dataclass
class AttentionDecoderConfig:
    type: Literal["attention"] = "attention"
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1


@dataclass
class TimestampDecoderConfig:
    type: Literal["timestamp"] = "timestamp"
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    max_length: int = 100


@dataclass
class LegacyLinearUpsampleDecoderConfig:
    type: Literal["legacy_linear_upsample"] = "legacy_linear_upsample"
    upsample_factor: int = 1


@dataclass
class WhisperSegDecoderConfig:
    type: Literal["whisperseg"] = "whisperseg"
    dropout: float = 0.1
    max_length: int = 100
    generation_max_length: int = 448
    num_trials: int = 1
    num_beams: int = 4
    top_k: int = 1
    top_p: float = 1.0
    length_penalty: float = 1.0


ResolvedDecoderConfig = (
    LinearDecoderConfig
    | LSTMDecoderConfig
    | ConvDecoderConfig
    | AttentionDecoderConfig
    | TimestampDecoderConfig
    | LegacyLinearUpsampleDecoderConfig
    | WhisperSegDecoderConfig
)


class LinearDecoder(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.decoder = nn.Linear(input_dim, num_classes, bias=False)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.decoder(encoded)


class LSTMDecoder(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, decoder_lstm_hidden_size: int = 64):
        super().__init__()
        self.decoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=decoder_lstm_hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.projection = nn.Linear(2 * decoder_lstm_hidden_size, num_classes, bias=False)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        decoded, _ = self.decoder(encoded)
        return self.projection(decoded)


class ConvDecoder(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, decoder_conv_kernel_size: int = 8):
        super().__init__()
        self.decoder = nn.Conv1d(
            in_channels=input_dim,
            out_channels=num_classes,
            kernel_size=decoder_conv_kernel_size,
            padding="same",
        )

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.decoder(encoded.transpose(1, 2)).transpose(1, 2)


class AttentionDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        decoder_num_heads: int = 4,
        decoder_num_layers: int = 2,
        decoder_dropout: float = 0.1,
    ):
        super().__init__()
        if input_dim % decoder_num_heads != 0:
            raise ValueError(
                f"Attention decoder input_dim={input_dim} must be divisible by decoder_num_heads={decoder_num_heads}."
            )

        self.attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=input_dim,
                    num_heads=decoder_num_heads,
                    dropout=decoder_dropout,
                    batch_first=True,
                )
                for _ in range(decoder_num_layers)
            ]
        )
        self.layer_norms = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(decoder_num_layers)])
        self.dropout = nn.Dropout(decoder_dropout) if decoder_dropout > 0 else None
        self.projection = nn.Linear(input_dim, num_classes, bias=False)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        decoded = encoded
        for attention, layer_norm in zip(self.attention_layers, self.layer_norms, strict=True):
            attention_output, _ = attention(decoded, decoded, decoded)
            decoded = layer_norm(decoded + attention_output)
            if self.dropout is not None:
                decoded = self.dropout(decoded)
        return self.projection(decoded)


class TimestampDecoder(nn.Module):
    PAD = 0
    BOS = 1
    EOS = 2

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        max_time_index: int,
        class_types: list[str] | None = None,
        decoder_num_heads: int = 4,
        decoder_num_layers: int = 2,
        decoder_dropout: float = 0.1,
        max_length: int = 100,
    ):
        super().__init__()
        if max_length < 5:
            raise ValueError("Timestamp decoder max_length must be at least 5.")
        if input_dim % decoder_num_heads != 0:
            raise ValueError(
                f"Timestamp decoder input_dim={input_dim} must be divisible by decoder_num_heads={decoder_num_heads}."
            )
        if num_classes < 2:
            raise ValueError("Timestamp decoder requires at least one non-noise class.")

        self.num_classes = int(num_classes)
        self.max_time_index = int(max_time_index)
        self.max_length = int(max_length)
        self.max_annotations = (self.max_length - 2) // 3
        self.time_token_start = 3
        self.class_token_start = self.time_token_start + self.max_time_index + 1
        self.vocab_size = self.class_token_start + self.num_classes - 1
        normalized_types = list(class_types or ["segment"] * self.num_classes)
        if len(normalized_types) != self.num_classes:
            normalized_types = ["segment"] * self.num_classes
        self.class_types = normalized_types

        self.token_embedding = nn.Embedding(self.vocab_size, input_dim, padding_idx=self.PAD)
        nn.init.normal_(self.token_embedding.weight, std=input_dim**-0.5)
        with torch.no_grad():
            self.token_embedding.weight[self.PAD].zero_()
        self.target_positions = nn.Embedding(self.max_length, input_dim)
        self.memory_positions = nn.Embedding(self.max_time_index + 1, input_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=input_dim,
            nhead=decoder_num_heads,
            dim_feedforward=4 * input_dim,
            dropout=decoder_dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            layer,
            num_layers=decoder_num_layers,
            norm=nn.LayerNorm(input_dim),
        )
        self.activity_head = nn.Linear(input_dim, 1)

    def forward(
        self,
        encoded: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        memory_lengths: torch.Tensor,
    ) -> torch.Tensor:
        target_length = decoder_input_ids.shape[1]
        memory_length = encoded.shape[1]
        if target_length > self.max_length:
            raise ValueError(f"Timestamp decoder input length {target_length} exceeds max_length={self.max_length}.")
        if memory_length > self.memory_positions.num_embeddings:
            raise ValueError(
                f"Encoder produced {memory_length} frames, exceeding timestamp decoder capacity "
                f"{self.memory_positions.num_embeddings}."
            )

        target_positions = torch.arange(target_length, device=encoded.device)
        memory_positions = torch.arange(memory_length, device=encoded.device)
        target = self.token_embedding(decoder_input_ids) + self.target_positions(target_positions)
        memory = encoded + self.memory_positions(memory_positions)
        causal_mask = torch.triu(
            torch.ones((target_length, target_length), dtype=torch.bool, device=encoded.device),
            diagonal=1,
        )
        memory_padding_mask = memory_positions.unsqueeze(0) >= memory_lengths.unsqueeze(1)
        decoded = self.transformer(
            target,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=decoder_input_ids.eq(self.PAD),
            memory_key_padding_mask=memory_padding_mask,
        )
        return torch.nn.functional.linear(decoded, self.token_embedding.weight)

    def activity_logits(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.activity_head(encoded).squeeze(-1)

    def triples_to_activity(self, triples: torch.Tensor, lengths: torch.Tensor, num_frames: int) -> torch.Tensor:
        frames = torch.arange(num_frames, device=triples.device).view(1, 1, -1)
        active = torch.arange(triples.shape[1], device=triples.device).view(1, -1, 1) < lengths.view(-1, 1, 1)
        onsets = triples[..., 0].long().unsqueeze(-1)
        offsets = triples[..., 2].long().unsqueeze(-1)
        event_classes = torch.tensor(
            [class_type == "event" for class_type in self.class_types],
            device=triples.device,
        )
        events = event_classes[triples[..., 1].long()].unsqueeze(-1)
        event_frames = frames.eq(onsets.clamp_max(max(num_frames - 1, 0)))
        segment_frames = frames.ge(onsets) & frames.lt(offsets)
        return (active & torch.where(events, event_frames, segment_frames)).any(dim=1).float()

    def triples_to_tokens(self, triples: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        lengths = lengths.to(device=triples.device, dtype=torch.long)
        tokens = torch.full(
            (triples.shape[0], self.max_length),
            self.PAD,
            dtype=torch.long,
            device=triples.device,
        )
        tokens[:, 0] = self.BOS
        for triple_index in range(self.max_annotations):
            active = lengths > triple_index
            if not torch.any(active):
                break
            offset = 1 + 3 * triple_index
            tokens[active, offset] = self.time_token_start + triples[active, triple_index, 0].long()
            tokens[active, offset + 1] = self.class_token_start + triples[active, triple_index, 1].long() - 1
            tokens[active, offset + 2] = self.time_token_start + triples[active, triple_index, 2].long()
        eos_positions = 1 + 3 * lengths
        tokens[torch.arange(tokens.shape[0], device=tokens.device), eos_positions] = self.EOS
        return tokens

    def tokens_to_triples(self, tokens: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        triples = torch.zeros(
            (tokens.shape[0], self.max_annotations, 3),
            dtype=torch.long,
            device=tokens.device,
        )
        triple_lengths = torch.zeros(tokens.shape[0], dtype=torch.long, device=tokens.device)
        for batch_index in range(tokens.shape[0]):
            token_length = int(lengths[batch_index])
            triple_count = min(max((token_length - 2) // 3, 0), self.max_annotations)
            if triple_count == 0:
                continue
            values = tokens[batch_index, 1 : 1 + 3 * triple_count].reshape(triple_count, 3)
            triples[batch_index, :triple_count, 0] = values[:, 0] - self.time_token_start
            triples[batch_index, :triple_count, 1] = values[:, 1] - self.class_token_start + 1
            triples[batch_index, :triple_count, 2] = values[:, 2] - self.time_token_start
            triple_lengths[batch_index] = triple_count
        return triples, triple_lengths

    def generate(self, encoded: torch.Tensor, memory_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = encoded.shape[0]
        activity_logits = self.activity_logits(encoded)
        memory_positions = torch.arange(encoded.shape[1], device=encoded.device)
        has_activity = activity_logits.masked_fill(
            memory_positions.unsqueeze(0) >= memory_lengths.unsqueeze(1),
            float("-inf"),
        ).amax(dim=1) >= 0
        tokens = torch.full(
            (batch_size, self.max_length),
            self.PAD,
            dtype=torch.long,
            device=encoded.device,
        )
        tokens[:, 0] = self.BOS
        lengths = torch.full((batch_size,), self.max_length, dtype=torch.long, device=encoded.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=encoded.device)

        for token_index in range(1, self.max_length):
            logits = self.forward(encoded, tokens[:, :token_index], memory_lengths)[:, -1]
            allowed = torch.zeros_like(logits, dtype=torch.bool)
            slot = (token_index - 1) % 3
            triple_count = (token_index - 1) // 3
            if slot == 0:
                activity_length = min(activity_logits.shape[1], self.max_time_index + 1)
                logits[:, self.time_token_start : self.time_token_start + activity_length] += activity_logits[
                    :, :activity_length
                ]

            for batch_index in range(batch_size):
                if finished[batch_index]:
                    allowed[batch_index, self.PAD] = True
                    continue
                max_boundary = min(int(memory_lengths[batch_index]), self.max_time_index)
                if slot == 0:
                    allowed[batch_index, self.EOS] = triple_count > 0 or not bool(has_activity[batch_index])
                    if triple_count < self.max_annotations:
                        previous_onset = (
                            0
                            if triple_count == 0
                            else int(tokens[batch_index, 1 + 3 * (triple_count - 1)]) - self.time_token_start
                        )
                        if any(class_type == "event" for class_type in self.class_types[1:]):
                            max_onset = max_boundary
                        else:
                            max_onset = max_boundary - 1
                        if max_onset >= previous_onset:
                            allowed[
                                batch_index,
                                self.time_token_start + previous_onset : self.time_token_start + max_onset + 1,
                            ] = True
                elif slot == 1:
                    onset = int(tokens[batch_index, token_index - 1]) - self.time_token_start
                    for class_index in range(1, self.num_classes):
                        if self.class_types[class_index] == "event" or onset < max_boundary:
                            allowed[batch_index, self.class_token_start + class_index - 1] = True
                else:
                    onset = int(tokens[batch_index, token_index - 2]) - self.time_token_start
                    class_index = int(tokens[batch_index, token_index - 1]) - self.class_token_start + 1
                    if self.class_types[class_index] == "event":
                        allowed[batch_index, self.time_token_start + onset] = True
                    elif max_boundary > onset:
                        allowed[
                            batch_index,
                            self.time_token_start + onset + 1 : self.time_token_start + max_boundary + 1,
                        ] = True

            next_tokens = logits.masked_fill(~allowed, float("-inf")).argmax(dim=-1)
            tokens[:, token_index] = next_tokens
            just_finished = next_tokens.eq(self.EOS) & ~finished
            lengths[just_finished] = token_index + 1
            finished |= just_finished
            if torch.all(finished):
                break
        return tokens, lengths


class LegacyLinearUpsampleDecoder(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, upsample_factor: int = 1):
        super().__init__()
        self.upsample_factor = int(upsample_factor)
        self.decoder = nn.Linear(input_dim, num_classes, bias=True)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        logits = self.decoder(encoded)
        if self.upsample_factor > 1:
            logits = logits.repeat_interleave(self.upsample_factor, dim=1)
        return logits


def normalize_decoder_config(config: DecoderConfig | ResolvedDecoderConfig | Mapping[str, object]) -> ResolvedDecoderConfig:
    if isinstance(
        config,
        (
            LinearDecoderConfig,
            LSTMDecoderConfig,
            ConvDecoderConfig,
            AttentionDecoderConfig,
            TimestampDecoderConfig,
            LegacyLinearUpsampleDecoderConfig,
            WhisperSegDecoderConfig,
        ),
    ):
        return config
    if isinstance(config, Mapping) and str(config.get("type")) == "whisperseg":
        return WhisperSegDecoderConfig(
            dropout=float(config.get("dropout", 0.1)),
            max_length=int(config.get("max_length", 100)),
            generation_max_length=int(config.get("generation_max_length", 448)),
            num_trials=int(config.get("num_trials", 1)),
            num_beams=int(config.get("num_beams", 4)),
            top_k=int(config.get("top_k", 1)),
            top_p=float(config.get("top_p", 1.0)),
            length_penalty=float(config.get("length_penalty", 1.0)),
        )
    if isinstance(config, Mapping) and str(config.get("type")) == "legacy_linear_upsample":
        return LegacyLinearUpsampleDecoderConfig(upsample_factor=int(config.get("upsample_factor", 1)))
    if not isinstance(config, DecoderConfig):
        config = DecoderConfig(**dict(config))

    if config.type == "linear":
        return LinearDecoderConfig()
    if config.type == "lstm":
        return LSTMDecoderConfig(hidden_size=int(config.hidden_size))
    if config.type == "conv":
        return ConvDecoderConfig(kernel_size=int(config.kernel_size))
    if config.type == "attention":
        return AttentionDecoderConfig(
            num_heads=int(config.num_heads),
            num_layers=int(config.num_layers),
            dropout=float(config.dropout),
        )
    if config.type == "timestamp":
        return TimestampDecoderConfig(
            num_heads=int(config.num_heads),
            num_layers=int(config.num_layers),
            dropout=float(config.dropout),
            max_length=int(config.max_length),
        )
    if config.type == "whisperseg":
        return WhisperSegDecoderConfig(
            dropout=float(config.dropout),
            max_length=int(config.max_length),
            generation_max_length=int(config.generation_max_length),
            num_trials=int(config.num_trials),
            num_beams=int(config.num_beams),
            top_k=int(config.top_k),
            top_p=float(config.top_p),
            length_penalty=float(config.length_penalty),
        )
    raise ValueError(f"Unknown decoder config '{config.type}'.")


def serialize_decoder_config(config: DecoderConfig | ResolvedDecoderConfig | Mapping[str, object]) -> dict[str, object]:
    return asdict(normalize_decoder_config(config))


def build_decoder(
    config: DecoderConfig | ResolvedDecoderConfig | Mapping[str, object],
    *,
    input_dim: int,
    num_classes: int,
    max_time_index: int | None = None,
    class_types: list[str] | None = None,
) -> nn.Module:
    decoder_config = normalize_decoder_config(config)
    if isinstance(decoder_config, LinearDecoderConfig):
        return LinearDecoder(input_dim=input_dim, num_classes=num_classes)
    if isinstance(decoder_config, LSTMDecoderConfig):
        return LSTMDecoder(
            input_dim=input_dim,
            num_classes=num_classes,
            decoder_lstm_hidden_size=decoder_config.hidden_size,
        )
    if isinstance(decoder_config, ConvDecoderConfig):
        return ConvDecoder(
            input_dim=input_dim,
            num_classes=num_classes,
            decoder_conv_kernel_size=decoder_config.kernel_size,
        )
    if isinstance(decoder_config, AttentionDecoderConfig):
        return AttentionDecoder(
            input_dim=input_dim,
            num_classes=num_classes,
            decoder_num_heads=decoder_config.num_heads,
            decoder_num_layers=decoder_config.num_layers,
            decoder_dropout=decoder_config.dropout,
        )
    if isinstance(decoder_config, TimestampDecoderConfig):
        if max_time_index is None:
            raise ValueError("Timestamp decoder requires a known num_time_steps.")
        return TimestampDecoder(
            input_dim=input_dim,
            num_classes=num_classes,
            max_time_index=max_time_index,
            class_types=class_types,
            decoder_num_heads=decoder_config.num_heads,
            decoder_num_layers=decoder_config.num_layers,
            decoder_dropout=decoder_config.dropout,
            max_length=decoder_config.max_length,
        )
    if isinstance(decoder_config, LegacyLinearUpsampleDecoderConfig):
        return LegacyLinearUpsampleDecoder(
            input_dim=input_dim,
            num_classes=num_classes,
            upsample_factor=decoder_config.upsample_factor,
        )
    if isinstance(decoder_config, WhisperSegDecoderConfig):
        raise ValueError("decoder_type=whisperseg uses the WhisperSeg backend and cannot be built as a native decoder.")
    raise TypeError(f"Unsupported decoder config '{type(decoder_config).__name__}'.")
