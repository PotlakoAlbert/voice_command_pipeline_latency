"""
models.py

KWS model architectures.

CRNN follows Arik et al. [2017] "Convolutional Recurrent Neural Networks
for Small-Footprint Keyword Spotting": a small convolutional front end for
local time-frequency feature extraction, followed by bidirectional GRU
layers that capture the full temporal extent of the keyword window, and a
final linear classifier over the 12-class label set.

Reference configuration from the paper: 229K parameters, 30M FLOPs,
97.71% Top-One accuracy on PCEN-normalised mel spectrogram features.

Dilated ResNet (res8-narrow) follows Tang and Lin [2018] "Deep Residual
Learning for Small-Footprint Keyword Spotting": exponentially dilated
residual blocks replace the GRU, giving a fully parallelisable
computation graph. Reference: 19.9K params, 5.65M MACs, 90.1% accuracy.

Input shape convention (matches data.py):
    (batch, n_mels, time) — e.g. (B, 40, 101) for 1-second clips.
"""

import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, n_mels=40, n_classes=12, conv_channels=37,
                 gru_hidden=72, gru_layers=1):
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
    def forward_frames(self, x):
        """Per-timestep posteriors for Jose et al. latency-control loss.
        Returns (batch, time, n_classes) instead of pooling over time."""
        x = x.unsqueeze(1)
        x = self.conv(x)
        batch, channels, freq, time = x.shape
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch, time, channels * freq)
        x, _ = self.gru(x)
        logits = self.classifier(x)  # (batch, time, n_classes)
        return logits


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3,
            padding=(1, dilation), dilation=(1, dilation), bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3,
            padding=(1, dilation), dilation=(1, dilation), bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.relu(out)


class DilatedResNet(nn.Module):
    """res8-narrow. channels=13 tuned to hit paper's ~19.9K param budget."""
    def __init__(self, n_mels=40, n_classes=12, channels=13, n_blocks=6):
        super().__init__()

        self.input_conv = nn.Conv2d(1, channels, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

        dilations = [2 ** i for i in range(n_blocks)]
        self.blocks = nn.ModuleList([
            DilatedResidualBlock(channels, dilation=d) for d in dilations
        ])

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.input_bn(self.input_conv(x)))

        for block in self.blocks:
            x = block(x)

        x = self.global_pool(x)
        x = x.flatten(1)
        logits = self.classifier(x)
        return logits
    def forward_frames(self, x):
        """Per-timestep posteriors: pool over frequency only, keep time."""
        x = x.unsqueeze(1)
        x = self.relu(self.input_bn(self.input_conv(x)))
        for block in self.blocks:
            x = block(x)
        # x: (batch, channels, freq, time) -> pool freq only
        x = x.mean(dim=2)          # (batch, channels, time)
        x = x.permute(0, 2, 1)     # (batch, time, channels)
        logits = self.classifier(x)  # (batch, time, n_classes)
        return logits



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_crnn():
    model = CRNN(n_mels=40, n_classes=12)
    n_params = count_parameters(model)
    print(f"CRNN parameter count: {n_params:,}")
    print("Reference (Arik et al. 2017): 229,000 params (tuned config: conv_channels=37, gru_hidden=72)")

    dummy = torch.randn(8, 40, 101)
    out = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}  (expected: (8, 12))")
    assert out.shape == (8, 12), "Output shape mismatch"
    print("Forward pass OK")


def test_dilated_resnet():
    model = DilatedResNet(n_mels=40, n_classes=12)
    n_params = count_parameters(model)
    print(f"Dilated ResNet (res8-narrow) parameter count: {n_params:,}")
    print("Reference (Tang and Lin 2018): 19,900 params")

    dummy = torch.randn(8, 40, 101)
    out = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}  (expected: (8, 12))")
    assert out.shape == (8, 12), "Output shape mismatch"
    print("Forward pass OK")


if __name__ == "__main__":
    print("--- CRNN ---")
    test_crnn()
    print()
    print("--- Dilated ResNet ---")
    test_dilated_resnet()
