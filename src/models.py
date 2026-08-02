"""
models.py

KWS model architectures.

CRNN follows Arık et al. [2017] "Convolutional Recurrent Neural Networks
for Small-Footprint Keyword Spotting": a small convolutional front end for
local time-frequency feature extraction, followed by bidirectional GRU
layers that capture the full temporal extent of the keyword window, and a
final linear classifier over the 12-class label set.

Reference configuration from the paper: 229K parameters, 30M FLOPs,
97.71% Top-One accuracy on PCEN-normalised mel spectrogram features.

Input shape convention (matches data.py):
    (batch, n_mels, time) — e.g. (B, 40, 101) for 1-second clips.
"""

import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, n_mels=40, n_classes=12, conv_channels=32,
                 gru_hidden=64, gru_layers=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(conv_channels, conv_channels, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )

        reduced_freq = n_mels // 4
        gru_input_size = conv_channels * reduced_freq

        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Linear(2 * gru_hidden, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)

        batch, channels, freq, time = x.shape
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch, time, channels * freq)

        x, _ = self.gru(x)
        x = x.mean(dim=1)

        logits = self.classifier(x)
        return logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = CRNN(n_mels=40, n_classes=12)
    n_params = count_parameters(model)
    print(f"CRNN parameter count: {n_params:,}")
    print(f"Reference (Arık et al. 2017): 229,000 params")

    dummy = torch.randn(8, 40, 101)
    out = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}  (expected: (8, 12))")
    assert out.shape == (8, 12), "Output shape mismatch"
    print("Forward pass OK")
