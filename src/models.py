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
import torch.nn.functional as F


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

class RepConvBlock(nn.Module):
    """Re-parameterizable Convolutional Block (Kundu et al. 2024).
    Training: n parallel depthwise conv+BN branches (kernel k) + one
    depthwise 1x1 conv+BN branch (skip), summed, then ReLU.
    Inference: all branches algebraically fused into one depthwise
    conv (same kernel size k), then a clip activation.
    """
    def __init__(self, channels, kernel_size, n_branches=2):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_branches = n_branches
        self.deployed = False

        pad = kernel_size // 2
        self.branches = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, padding=pad, groups=channels)
            for _ in range(n_branches)
        ])
        self.branch_bns = nn.ModuleList([nn.BatchNorm1d(channels) for _ in range(n_branches)])
        self.skip_conv = nn.Conv1d(channels, channels, 1, groups=channels)
        self.skip_bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

        # Populated by switch_to_deploy()
        self.fused_conv = None

    def _fuse_conv_bn(self, conv, bn, target_kernel_size):
        # BN-fusion identity: W_k = (alpha/sigma) * w_k, W_b = beta - (alpha*mu/sigma)
        std = (bn.running_var + bn.eps).sqrt()
        scale = bn.weight / std
        fused_weight = conv.weight * scale.reshape(-1, 1, 1)  # depthwise: (C,1,k)
        fused_bias = bn.bias - (bn.running_mean - conv.bias) * scale
        if conv.kernel_size[0] != target_kernel_size:
            # zero-pad the 1x1 skip kernel to target size, weight at center tap
            pad_total = target_kernel_size - conv.kernel_size[0]
            left = pad_total // 2
            right = pad_total - left
            fused_weight = F.pad(fused_weight, (left, right))
        return fused_weight, fused_bias

    def switch_to_deploy(self, use_clip=False):
        self.use_clip = use_clip
        k = self.kernel_size
        weight_sum = 0.0
        bias_sum = 0.0
        for conv, bn in zip(self.branches, self.branch_bns):
            w, b = self._fuse_conv_bn(conv, bn, k)
            weight_sum = weight_sum + w
            bias_sum = bias_sum + b
        w_skip, b_skip = self._fuse_conv_bn(self.skip_conv, self.skip_bn, k)
        weight_sum = weight_sum + w_skip
        bias_sum = bias_sum + b_skip

        device = self.skip_conv.weight.device
        self.fused_conv = nn.Conv1d(self.channels, self.channels, k,
                                     padding=k // 2, groups=self.channels).to(device)
        with torch.no_grad():
            self.fused_conv.weight.copy_(weight_sum)
            self.fused_conv.bias.copy_(bias_sum)
        self.deployed = True
        del self.branches
        del self.branch_bns
        del self.skip_conv
        del self.skip_bn

    def forward(self, x):
        if self.deployed:
            out = self.fused_conv(x)
            if self.use_clip:
                # Paper's described inference-time swap (ReLU -> bounded clip),
                # applied without retraining. At this param budget/data scale
                # this causes severe detection collapse (see results write-up) —
                # kept here only to reproduce that documented finding, not used
                # as the primary deploy path.
                return torch.clamp(out, min=0.0, max=6.0)
            return self.relu(out)
        out = self.skip_bn(self.skip_conv(x))
        for conv, bn in zip(self.branches, self.branch_bns):
            out = out + bn(conv(x))
        return self.relu(out)


class RepCNN(nn.Module):
    """RepCNN (Kundu et al. 2024) sized to ~19.7K fused params — same
    budget as DilatedResNet (19.9K) for fair 3-way comparison. Exact
    Kundu et al. hyperparameters (channel widths) are not published
    (no code release, no param-count table in the paper); kernel
    sizes, block structure, and branch count (n=2) ARE taken directly
    from the paper (Sec 3.3, Table 3 ablation).
    """
    def __init__(self, n_mels=40, n_classes=12, c0=48, c1=56, c2=64, n_branches=2):
        super().__init__()
        self.stem = nn.Conv1d(n_mels, c0, kernel_size=5, stride=2, padding=2)
        self.stem_bn = nn.BatchNorm1d(c0)
        self.relu = nn.ReLU()

        self.block1a = RepConvBlock(c0, kernel_size=5, n_branches=n_branches)
        self.block1b = RepConvBlock(c0, kernel_size=7, n_branches=n_branches)
        self.pw1 = nn.Conv1d(c0, c1, kernel_size=1)
        self.pw1_bn = nn.BatchNorm1d(c1)

        self.block2a = RepConvBlock(c1, kernel_size=11, n_branches=n_branches)
        self.block2b = RepConvBlock(c1, kernel_size=13, n_branches=n_branches)
        self.pw2 = nn.Conv1d(c1, c2, kernel_size=1)
        self.pw2_bn = nn.BatchNorm1d(c2)

        self.classifier = nn.Linear(c2, n_classes)

    def _features(self, x):
        # x: (batch, n_mels, time)
        x = self.relu(self.stem_bn(self.stem(x)))
        x = self.block1a(x)
        x = self.block1b(x)
        x = self.relu(self.pw1_bn(self.pw1(x)))
        x = self.block2a(x)
        x = self.block2b(x)
        x = self.relu(self.pw2_bn(self.pw2(x)))
        return x  # (batch, c2, time')

    def forward(self, x):
        feats = self._features(x)
        pooled = feats.mean(dim=2)  # (batch, c2)
        return self.classifier(pooled)

    def forward_frames(self, x):
        feats = self._features(x)  # (batch, c2, time')
        feats = feats.transpose(1, 2)  # (batch, time', c2)
        return self.classifier(feats)  # (batch, time', n_classes)

    def switch_to_deploy(self, use_clip=False):
        for m in [self.block1a, self.block1b, self.block2a, self.block2b]:
            m.switch_to_deploy(use_clip=use_clip)
