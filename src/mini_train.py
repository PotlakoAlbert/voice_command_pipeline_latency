"""Train + test one (arch, b, seed) on the mini Jose-style binary window task."""
import argparse, json, os, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, "src")
from latency_loss import latency_controlled_loss
from mini_data import MiniSegments, WindowFeats, window_logits, window_logits_frames

WL = window_logits
from models import CRNN, DilatedResNet, RepCNN


def build(arch):
    if arch == "crnn":
        return CRNN(n_mels=40, n_classes=2, conv_channels=37, gru_hidden=72)
    if arch == "dilated_resnet":
        return DilatedResNet(n_mels=40, n_classes=2)
    return RepCNN(n_mels=40, n_classes=2, c0=48, c1=56, c2=64, n_branches=2)


def auc(pos, neg):
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort() + 1.0
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


@torch.no_grad()
def collect(model, feats, ds, dev, bs, nw):
    model.eval()
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw)
    P, L, KS, KE = [], [], [], []
    for wav, lab, ks, ke in dl:
        lg = WL(model, feats, wav.to(dev))
        P.append(torch.softmax(lg, -1)[..., 1].cpu().numpy())
        L.append(lab.numpy()); KS.append(ks.numpy()); KE.append(ke.numpy())
    return np.concatenate(P), np.concatenate(L), np.concatenate(KS), np.concatenate(KE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["crnn", "dilated_resnet", "repcnn"])
    ap.add_argument("--b", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--n-pos", type=int, default=2000)
    ap.add_argument("--n-neg", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default="results/mini")
    ap.add_argument("--ckpt-dir", default="checkpoints/mini")
    ap.add_argument("--readout", choices=["pooled", "frames"], default="pooled", help="window score: model forward() (default) or max over forward_frames() logits")
    ap.add_argument("--no-trail", action="store_true", help="positives have no speech after the keyword (ablation)")
    ap.add_argument("--mirror-neg", action="store_true", help="with --no-trail: negatives also end in noise-only tails (no shortcut)")
    a = ap.parse_args()
    global WL
    WL = window_logits_frames if a.readout == "frames" else window_logits

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    tag = f"{a.arch}_b{a.b}_s{a.seed}"
    print(f"{tag} | device {dev} | torch {torch.__version__}", flush=True)

    feats = WindowFeats().to(dev)
    model = build(a.arch).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print("params:", n_params, flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    train_ds = MiniSegments("train", n=a.n_train, seed=a.seed, trail=not a.no_trail, mirror=a.mirror_neg)
    val_ds = MiniSegments("val", n=a.n_val, seed=1000, trail=not a.no_trail, mirror=a.mirror_neg)
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.ckpt_dir, exist_ok=True)
    ck = os.path.join(a.ckpt_dir, tag + "_best.pt")

    best, bad, hist, t0 = -1.0, 0, [], time.time()
    for ep in range(1, a.epochs + 1):
        train_ds.set_epoch(ep)
        model.train()
        tl, nb = 0.0, 0
        for wav, lab, ks, ke in DataLoader(train_ds, batch_size=a.batch_size, shuffle=False,
                                           num_workers=a.num_workers):
            wav, lab = wav.to(dev), lab.to(dev)
            lg = WL(model, feats, wav)
            loss = latency_controlled_loss(lg, lab, b=a.b, negative_indices=(0,))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tl += loss.item(); nb += 1
        P, L, _, _ = collect(model, feats, val_ds, dev, 64, a.num_workers)
        v = auc(P[L == 1].max(1), P[L == 0].max(1))
        hist.append({"epoch": ep, "train_loss": tl / nb, "val_auc": v})
        print(f"epoch {ep} train_loss {tl / nb:.4f} val_auc {v:.5f} ({time.time() - t0:.0f}s)", flush=True)
        if v > best:
            best, bad = v, 0
            torch.save({"model_state_dict": model.state_dict(), "arch": a.arch, "b": a.b,
                        "seed": a.seed, "epoch": ep, "val_auc": v}, ck)
        else:
            bad += 1
            if bad >= a.patience:
                print("early stop", flush=True)
                break

    model.load_state_dict(torch.load(ck, map_location=dev)["model_state_dict"])
    pos_p, _, ks, ke = collect(model, feats, MiniSegments("test", n=a.n_pos, seed=2000, pos_frac=1.0, trail=not a.no_trail, mirror=a.mirror_neg),
                               dev, 64, a.num_workers)
    neg_p, _, _, _ = collect(model, feats, MiniSegments("test", n=a.n_neg, seed=3000, pos_frac=0.0, mirror=a.mirror_neg),
                             dev, 64, a.num_workers)
    np.savez_compressed(os.path.join(a.out, tag + ".npz"), pos_p=pos_p, pos_ks=ks, pos_ke=ke, neg_p=neg_p)
    with open(os.path.join(a.out, tag + ".json"), "w") as f:
        json.dump({"arch": a.arch, "b": a.b, "seed": a.seed, "n_params": n_params,
                   "epochs_run": len(hist), "best_val_auc": best, "history": hist,
                   "train_seconds": time.time() - t0}, f, indent=2)
    print("DONE", tag, flush=True)


if __name__ == "__main__":
    main()
