"""
data.py

Speech Commands v2 loading and preprocessing for KWS training.
"""

import os
import argparse

import torch
import torchaudio
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS

TARGET_LABELS = [
    "yes", "no", "up", "down", "left", "right",
    "on", "off", "stop", "go",
]
SILENCE_LABEL = "_silence_"
UNKNOWN_LABEL = "_unknown_"
LABEL_LIST = TARGET_LABELS + [SILENCE_LABEL, UNKNOWN_LABEL]
LABEL_TO_IDX = {label: i for i, label in enumerate(LABEL_LIST)}

SAMPLE_RATE = 16000
FIXED_LENGTH = SAMPLE_RATE


class SpeechCommandsDataset(torch.utils.data.Dataset):
    def __init__(self, root, subset, n_mels=40, n_fft=480, hop_length=160):
        self._base = SPEECHCOMMANDS(root=root, download=False, subset=subset)
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

    def __len__(self):
        return len(self._base)

    def _pcen(self, mel_spec):
        return torchaudio.functional.pcen(
            mel_spec,
            smooth=0.025,
            gain=0.98,
            bias=2.0,
            power=0.5,
            eps=1e-6,
        )

    def _fix_length(self, waveform):
        length = waveform.shape[-1]
        if length < FIXED_LENGTH:
            pad = FIXED_LENGTH - length
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        elif length > FIXED_LENGTH:
            waveform = waveform[..., :FIXED_LENGTH]
        return waveform

    def _remap_label(self, word):
        if word in TARGET_LABELS:
            return LABEL_TO_IDX[word]
        return LABEL_TO_IDX[UNKNOWN_LABEL]

    def __getitem__(self, idx):
        waveform, sample_rate, label, speaker_id, utterance_number = self._base[idx]
        assert sample_rate == SAMPLE_RATE, (
            f"Expected {SAMPLE_RATE}Hz, got {sample_rate}Hz"
        )
        waveform = self._fix_length(waveform)
        mel = self.mel_spec(waveform)
        feat = self._pcen(mel)
        feat = feat.squeeze(0)
        label_idx = self._remap_label(label)
        return feat, label_idx


def get_dataloaders(root, batch_size=128, num_workers=2, n_mels=40):
    root = os.path.expanduser(root)
    train_ds = SpeechCommandsDataset(root, subset="training", n_mels=n_mels)
    val_ds = SpeechCommandsDataset(root, subset="validation", n_mels=n_mels)
    test_ds = SpeechCommandsDataset(root, subset="testing", n_mels=n_mels)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader, LABEL_LIST


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="~/voice_command_pipeline_latency/data")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    train_loader, val_loader, test_loader, labels = get_dataloaders(
        root=args.data, batch_size=args.batch_size,
    )

    print(f"Classes ({len(labels)}): {labels}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    feat, label = next(iter(train_loader))
    print(f"Feature batch shape: {feat.shape}  (batch, n_mels, time)")
    print(f"Label batch shape:   {label.shape}")
    print(f"Sample labels: {label[:8].tolist()}")
