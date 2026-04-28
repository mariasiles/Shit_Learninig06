"""CNN encoder + LSTM decoder for image captioning (yunjey-style, modernized)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    """ResNet-50 encoder. Outputs an embedding of size `embed_size`."""

    def __init__(self, embed_size: int = 256, backbone: str = "resnet50"):
        super().__init__()
        if backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            net = models.resnet50(weights=weights)
        elif backbone == "resnet152":
            weights = models.ResNet152_Weights.IMAGENET1K_V2
            net = models.resnet152(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        modules = list(net.children())[:-1]  # drop final fc
        self.cnn = nn.Sequential(*modules)
        self.linear = nn.Linear(net.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

        # Freeze CNN backbone (only fine-tune linear + bn)
        for p in self.cnn.parameters():
            p.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.cnn(images)            # [B, C, 1, 1]
        features = features.flatten(1)             # [B, C]
        features = self.bn(self.linear(features))  # [B, embed_size]
        return features


class DecoderRNN(nn.Module):
    """LSTM decoder conditioned on image features (fed once as the first input)."""

    def __init__(self, embed_size: int, hidden_size: int, vocab_size: int,
                 num_layers: int = 1, max_seq_length: int = 20):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seq_length = max_seq_length

    def forward(self, features: torch.Tensor, captions: torch.Tensor, lengths: list[int]):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        # captions already include <start>; lengths are full caption lengths
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens.data)
        return outputs

    @torch.no_grad()
    def sample(self, features: torch.Tensor, states=None) -> torch.Tensor:
        """Greedy generation. Returns [B, max_seq_length] token ids."""
        sampled = []
        inputs = features.unsqueeze(1)
        for _ in range(self.max_seq_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled.append(predicted)
            inputs = self.embed(predicted).unsqueeze(1)
        return torch.stack(sampled, dim=1)
