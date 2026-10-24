"""Mini Jose-style binary environment: window-level posteriors on 5 s tracks."""
import os
import numpy as np
import torch
import torchaudio

SR = 16000
SEG = 5 * SR            # 5 s track; first 1 s is PCEN warm-up only
WARM = SR
WIN_FRAMES = 98         # 1 s window with center=False, n_fft=480, hop=160
STRIDE_FRAMES = 10      # 100 ms window stride
N_WIN = 31              # window k ends at 2.0 + 0.1*k s; last ends at 5.0 s
CACHE = "data/mini_cache"
KEYWORD = "marvin"


def speech_bounds(wavs, rel_db=-35.0):
    """Energy VAD on int16 clips -> (start, end) sample indices."""
    starts, ends = [], []
    for i in range(0, len(wavs), 1000):
        x = np.asarray(wavs[i:i + 1000], dtype=np.float64) / 32768.0
        cs = np.concatenate([np.zeros((len(x), 1)), np.cumsum(x * x, axis=1)], axis=1)
        idx = np.arange(0, x.shape[1] - 480 + 1, 160)
        db = 10.0 * np.log10((cs[:, idx + 480] - cs[:, idx]) / 480.0 + 1e-12)
        act = db >= (db.max(axis=1, keepdims=True) + rel_db)
        first = act.argmax(axis=1)
        last = act.shape[1] - 1 - act[:, ::-1].argmax(axis=1)
        starts.append(first * 160)
        ends.append(np.minimum(last * 160 + 480, x.shape[1]))
    return np.concatenate(starts), np.concatenate(ends)


class MiniData:
    def __init__(self, split):
        self.wav = np.load(f"{CACHE}/{split}_wav.npy", mmap_mode="r")
        lab = np.load(f"{CACHE}/{split}_label.npy")
        is_kw = lab == KEYWORD
        self.kw_idx = np.nonzero(is_kw)[0]
        self.neg_idx = np.nonzero(~is_kw)[0]
        bp = f"{CACHE}/{split}_bounds.npy"
        if os.path.exists(bp):
            b = np.load(bp)
        else:
            s, e = speech_bounds(self.wav)
            b = np.stack([s, e])
            np.save(bp, b)
        self.s, self.e = b[0], b[1]
        nz = np.load(f"{CACHE}/noise_{split}.npz")
        self.noise = [nz[k] for k in nz.files]

    def word(self, rng, kw):
        i = rng.choice(self.kw_idx if kw else self.neg_idx)
        return np.asarray(self.wav[i, self.s[i]:self.e[i]], dtype=np.float32) / 32768.0

    def fill(self, rng, track, lo, hi):
        """Fill [lo, hi) with non-keyword words separated by gaps."""
        t = lo + int(rng.uniform(0.0, 0.3) * SR)
        for _ in range(40):
            gap = int(rng.uniform(0.05, 0.5) * SR)
            if rng.random() < 0.2:
                t += gap + int(0.5 * SR)
                continue
            for _try in range(3):
                w = self.word(rng, False)
                if t + len(w) <= hi:
                    break
            else:
                break
            track[t:t + len(w)] += w * 10 ** (rng.uniform(-6.0, 3.0) / 20)
            t += len(w) + gap

    def make_segment(self, rng, positive, trail=True, mirror=False):
        track = np.zeros(SEG, dtype=np.float32)
        if positive:
            kw = self.word(rng, True) * 10 ** (rng.uniform(-3.0, 3.0) / 20)
            ks = int(rng.uniform(1.5, 2.5) * SR)
            ke = ks + len(kw)
            track[ks:ke] += kw
            self.fill(rng, track, 0, ks - int(0.1 * SR))
            if trail:
                self.fill(rng, track, ke + int(0.1 * SR), SEG)
        elif mirror:
            w = self.word(rng, False) * 10 ** (rng.uniform(-3.0, 3.0) / 20)
            ws = int(rng.uniform(1.5, 2.5) * SR)
            track[ws:ws + len(w)] += w
            self.fill(rng, track, 0, ws - int(0.1 * SR))
            ks = ke = -1
        else:
            ks = ke = -1
            self.fill(rng, track, 0, SEG)
        rms = max(float(np.sqrt(np.mean(track ** 2))), 0.02)
        n = self.noise[int(rng.integers(len(self.noise)))]
        off = int(rng.integers(0, len(n) - SEG))
        nz = n[off:off + SEG].astype(np.float32)
        nz = nz / (float(np.sqrt(np.mean(nz ** 2))) + 1e-8)
        snr = rng.uniform(5.0, 30.0)
        track = np.clip(track + nz * rms / 10 ** (snr / 20), -1.0, 1.0)
        return track.astype(np.float32), ks, ke


class MiniSegments(torch.utils.data.Dataset):
    def __init__(self, split, n=20000, seed=0, pos_frac=0.5, trail=True, mirror=False):
        self.m = MiniData(split)
        self.trail = trail
        self.mirror = mirror
        self.n, self.seed, self.pos_frac, self.epoch = n, seed, pos_frac, 0

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, self.epoch, i])
        positive = rng.random() < self.pos_frac
        wav, ks, ke = self.m.make_segment(rng, positive, self.trail, self.mirror)
        return torch.from_numpy(wav), int(positive), ks, ke


class WindowFeats(torch.nn.Module):
    """Causal log-mel+PCEN over the whole track, sliced into 1 s windows."""

    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=480, hop_length=160, n_mels=40, center=False)

    @staticmethod
    def pcen(mel, smooth=0.025, gain=0.98, bias=2.0, power=0.5, eps=1e-6):
        mel = mel + eps
        m = mel[..., 0]
        out = [m]
        for t in range(1, mel.shape[-1]):
            m = (1 - smooth) * m + smooth * mel[..., t]
            out.append(m)
        sm = torch.stack(out, dim=-1)
        return (mel / (sm + eps) ** gain + bias) ** power - bias ** power

    @torch.no_grad()
    def forward(self, wav):                       # (B, SEG) -> (B, N_WIN, 40, 98)
        f = self.pcen(self.mel(wav))[..., WARM // 160:]
        return f.unfold(2, WIN_FRAMES, STRIDE_FRAMES).permute(0, 2, 1, 3)


def window_logits(model, feats, wav):
    f = feats(wav)
    B, N = f.shape[:2]
    return model(f.reshape(B * N, *f.shape[2:])).reshape(B, N, -1)
