"""Original-style (1 s clips, frame-level max-pool loss) training on the mini binary task,
evaluated on the same 5 s streams as the window arm (max over frames per 1 s window)."""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
sys.path.insert(0, "src")
from latency_loss import latency_controlled_loss
from mini_data import MiniData, MiniSegments, WindowFeats, SR, WARM, N_WIN, STRIDE_FRAMES
from mini_train import build, auc


class MiniClips(torch.utils.data.Dataset):
    def __init__(self, split, n, seed, pos_frac=0.5):
        self.m, self.n, self.seed, self.pos_frac, self.epoch = MiniData(split), n, seed, pos_frac, 0

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, self.epoch, i])
        m = self.m
        if rng.random() < self.pos_frac:
            j = rng.choice(m.kw_idx)
            x = np.asarray(m.wav[j], dtype=np.float32) / 32768.0 * 10 ** (rng.uniform(-3, 3) / 20)
            label = 1
        elif rng.random() < 0.2:
            x, label = np.zeros(SR, np.float32), 0
        else:
            j = rng.choice(m.neg_idx)
            x = np.asarray(m.wav[j], dtype=np.float32) / 32768.0 * 10 ** (rng.uniform(-6, 3) / 20)
            label = 0
        rms = max(float(np.sqrt(np.mean(x ** 2))), 0.02)
        n = m.noise[int(rng.integers(len(m.noise)))]
        off = int(rng.integers(0, len(n) - SR))
        nz = n[off:off + SR].astype(np.float32)
        nz = nz / (float(np.sqrt(np.mean(nz ** 2))) + 1e-8)
        x = np.clip(x + nz * rms / 10 ** (rng.uniform(5, 30) / 20), -1.0, 1.0)
        return torch.from_numpy(x.astype(np.float32)), label


@torch.no_grad()
def collect(model, mel, ds, dev, bs, nw, pooled):
    model.eval()
    P, L, KS, KE = [], [], [], []
    for wav, lab, ks, ke in DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw):
        wav = wav.to(dev)
        w = wav[:, WARM:].unfold(1, SR, STRIDE_FRAMES * 160)       # (B, 31, 16000)
        B = w.shape[0]
        assert w.shape[1] == N_WIN, w.shape
        f = WindowFeats.pcen(mel(w.reshape(B * N_WIN, SR)))        # per-window features, PCEN reset per window
        if pooled:
            lg = model(f)
        else:
            lg, _ = model.forward_frames(f).max(dim=1)             # max-pool logits over frames
        P.append(torch.softmax(lg, -1)[:, 1].reshape(B, N_WIN).cpu().numpy())
        L.append(lab.numpy()); KS.append(ks.numpy()); KE.append(ke.numpy())
    return np.concatenate(P), np.concatenate(L), np.concatenate(KS), np.concatenate(KE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["crnn", "dilated_resnet", "repcnn"])
    ap.add_argument("--b", type=float, default=0.0)
    ap.add_argument("--pooled-ce", action="store_true", help="b=0 only: pooled forward() + cross-entropy (original sweep's b=0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--n-pos", type=int, default=2000)
    ap.add_argument("--n-neg", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default="results/mini_frame")
    ap.add_argument("--ckpt-dir", default="checkpoints/mini_frame")
    a = ap.parse_args()
    if a.pooled_ce and a.b != 0.0:
        ap.error("--pooled-ce requires --b 0.0")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    tag = f"{a.arch}_b{a.b}_s{a.seed}"
    print(f"{tag} | pooled_ce={a.pooled_ce} | device {dev} | torch {torch.__version__}", flush=True)
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=480, hop_length=160, n_mels=40).to(dev)
    model = build(a.arch).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print("params:", n_params, flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    train_ds = MiniClips("train", n=a.n_train, seed=a.seed)
    val_ds = MiniSegments("val", n=a.n_val, seed=1000)
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.ckpt_dir, exist_ok=True)
    ck = os.path.join(a.ckpt_dir, tag + "_best.pt")

    best, bad, hist, t0 = -1.0, 0, [], time.time()
    for ep in range(1, a.epochs + 1):
        train_ds.set_epoch(ep)
        model.train()
        tl, nb = 0.0, 0
        for wav, lab in DataLoader(train_ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers):
            wav, lab = wav.to(dev), lab.to(dev)
            f = WindowFeats.pcen(mel(wav))                          # (B, 40, 101)
            if a.pooled_ce:
                loss = F.cross_entropy(model(f), lab)
            else:
                loss = latency_controlled_loss(model.forward_frames(f), lab, b=a.b, negative_indices=(0,))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tl += loss.item(); nb += 1
        P, L, _, _ = collect(model, mel, val_ds, dev, 64, a.num_workers, a.pooled_ce)
        v = auc(P[L == 1].max(1), P[L == 0].max(1))
        hist.append({"epoch": ep, "train_loss": tl / nb, "val_auc": v})
        print(f"epoch {ep} train_loss {tl / nb:.4f} val_auc {v:.5f} ({time.time() - t0:.0f}s)", flush=True)
        if v > best:
            best, bad = v, 0
            torch.save({"model_state_dict": model.state_dict(), "arch": a.arch, "b": a.b,
                        "pooled_ce": a.pooled_ce, "seed": a.seed, "epoch": ep, "val_auc": v}, ck)
        else:
            bad += 1
            if bad >= a.patience:
                print("early stop", flush=True)
                break

    model.load_state_dict(torch.load(ck, map_location=dev)["model_state_dict"])
    pos_p, _, ks, ke = collect(model, mel, MiniSegments("test", n=a.n_pos, seed=2000, pos_frac=1.0),
                               dev, 64, a.num_workers, a.pooled_ce)
    neg_p, _, _, _ = collect(model, mel, MiniSegments("test", n=a.n_neg, seed=3000, pos_frac=0.0),
                             dev, 64, a.num_workers, a.pooled_ce)
    np.savez_compressed(os.path.join(a.out, tag + ".npz"), pos_p=pos_p, pos_ks=ks, pos_ke=ke, neg_p=neg_p)
    with open(os.path.join(a.out, tag + ".json"), "w") as fo:
        json.dump({"arch": a.arch, "b": a.b, "pooled_ce": a.pooled_ce, "seed": a.seed, "n_params": n_params,
                   "epochs_run": len(hist), "best_val_auc": best, "history": hist,
                   "train_seconds": time.time() - t0}, fo, indent=2)
    print("DONE", tag, flush=True)


if __name__ == "__main__":
    main()
