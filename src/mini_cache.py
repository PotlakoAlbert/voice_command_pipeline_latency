import os, numpy as np, torch, torchaudio
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS

ROOT = "data"
OUT = os.path.join(ROOT, "mini_cache")
os.makedirs(OUT, exist_ok=True)
SR = 16000


class Wrap(torch.utils.data.Dataset):
    def __init__(self, subset):
        self.b = SPEECHCOMMANDS(root=ROOT, download=False, subset=subset)

    def __len__(self):
        return len(self.b)

    def __getitem__(self, i):
        w, sr, label, spk, n = self.b[i]
        assert sr == SR, sr
        w = w.mean(0) if w.shape[0] > 1 else w[0]
        L = min(w.shape[0], SR)
        w = torch.nn.functional.pad(w, (0, SR - w.shape[0])) if w.shape[0] < SR else w[:SR]
        return (w.clamp(-1, 1) * 32767).round().to(torch.int16), label, L


for split, subset in [("train", "training"), ("val", "validation"), ("test", "testing")]:
    dl = DataLoader(Wrap(subset), batch_size=512, num_workers=4)
    wavs, labels, lens = [], [], []
    for w, lab, L in dl:
        wavs.append(w.numpy()); labels += list(lab); lens.append(L.numpy())
    wavs = np.concatenate(wavs); lens = np.concatenate(lens); labels = np.array(labels)
    np.save(os.path.join(OUT, f"{split}_wav.npy"), wavs)
    np.save(os.path.join(OUT, f"{split}_label.npy"), labels)
    np.save(os.path.join(OUT, f"{split}_len.npy"), lens)
    print(split, wavs.shape, "marvin:", int((labels == "marvin").sum()), flush=True)

nd = os.path.join(ROOT, "SpeechCommands", "speech_commands_v0.02", "_background_noise_")
parts = {"train": {}, "val": {}, "test": {}}
for f in sorted(os.listdir(nd)):
    if not f.endswith(".wav"):
        continue
    w, sr = torchaudio.load(os.path.join(nd, f))
    if sr != SR:
        w = torchaudio.transforms.Resample(sr, SR)(w)
    w = w.mean(0).numpy().astype(np.float32)
    n = len(w)
    a, b = int(0.70 * n), int(0.85 * n)
    parts["train"][f[:-4]] = w[:a]; parts["val"][f[:-4]] = w[a:b]; parts["test"][f[:-4]] = w[b:]
for s, d in parts.items():
    np.savez(os.path.join(OUT, f"noise_{s}.npz"), **d)
    print("noise", s, {k: round(len(v) / SR, 1) for k, v in d.items()}, flush=True)
print("DONE")
