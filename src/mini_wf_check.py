import sys
sys.path.insert(0, "src")
import numpy as np
import torch
from mini_data import MiniData, WindowFeats, window_logits, window_logits_frames, SR, SEG, N_WIN
from models import CRNN, DilatedResNet, RepCNN

torch.manual_seed(0)
m = MiniData("val")
rng = np.random.default_rng(0)
wav = torch.from_numpy(np.stack([m.make_segment(rng, True)[0] for _ in range(4)]))
wav2 = wav.clone()
wav2[:, 3 * SR:] = torch.randn(4, SEG - 3 * SR) * 0.1
feats = WindowFeats()
models = {
    "crnn": CRNN(n_mels=40, n_classes=2, conv_channels=37, gru_hidden=72),
    "dilated_resnet": DilatedResNet(n_mels=40, n_classes=2),
    "repcnn": RepCNN(n_mels=40, n_classes=2, c0=48, c1=56, c2=64, n_branches=2),
}
for name, mod in models.items():
    mod.eval()
    with torch.no_grad():
        a = window_logits_frames(mod, feats, wav)
        b = window_logits_frames(mod, feats, wav2)
        p = window_logits(mod, feats, wav)
    assert tuple(a.shape) == (4, N_WIN, 2), a.shape
    early = (a[:, :11] - b[:, :11]).abs().max().item()
    late = (a[:, 11:] - b[:, 11:]).abs().max().item()
    diff = (a - p).abs().max().item()
    assert early < 1e-5, early
    assert late > 1e-4, late
    assert diff > 1e-4, diff
    print("PASS %s shape %s | early-window change %.2e | late-window change %.2e | differs from pooled readout by %.2e" % (name, tuple(a.shape), early, late, diff))
print("WF CHECK PASSED")
