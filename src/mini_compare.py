import glob, json, os, re
import numpy as np

SR, N_WIN = 16000, 31
TEND = 2.0 + 0.1 * np.arange(N_WIN)
FRRS = [0.02, 0.05, 0.10]
HOURS = 4000 * 4.0 / 3600.0
ARCHS = ["crnn", "dilated_resnet", "repcnn"]
BS = [0.0, 0.1, 0.2, 0.33, 0.5, 1.0]


def analyze(path):
    d = np.load(path)
    P, N = d["pos_p"], d["neg_p"]
    ks, ke = d["pos_ks"] / SR, d["pos_ke"] / SR
    Pv = np.where(TEND[None, :] >= ks[:, None], P, -1.0)
    ps, ns = Pv.max(1), N.max(1)
    out = {}
    for f in FRRS:
        thr = float(np.sort(ps)[int(np.floor(f * len(ps)))])
        det = ps >= thr
        first = (Pv >= thr).argmax(1)
        lat = (TEND[first] - ke)[det] * 1000.0
        fa = int((ns >= thr).sum())
        out[f] = (float(lat.mean()), fa / HOURS)
    return out


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]) if np.std(ry) > 0 and np.std(rx) > 0 else float("nan")


arms = {"window": "results/mini", "frame": "results/mini_frame"}
R = {}
for name, d in arms.items():
    for arch in ARCHS:
        for b in BS:
            p = f"{d}/{arch}_b{b}_s0.npz"
            if os.path.exists(p):
                R[(name, arch, b)] = analyze(p)
CE = {a: analyze(f"results/mini_frame_ce/{a}_b0.0_s0.npz") for a in ARCHS
      if os.path.exists(f"results/mini_frame_ce/{a}_b0.0_s0.npz")}

for f in FRRS:
    print(f"\n===== FRR target {int(f * 100)}% : mean latency ms from keyword end | FA per hour =====")
    for arch in ARCHS:
        print(f"{arch}:   b      window_lat  frame_lat   window_FA  frame_FA")
        for b in BS:
            w, fr = R.get(("window", arch, b)), R.get(("frame", arch, b))
            print(f"        {b:<5}  {w[f][0]:9.0f}  {fr[f][0]:9.0f}   {w[f][1]:9.1f}  {fr[f][1]:8.1f}")
        if arch in CE:
            print(f"        pooled-CE b=0 (original recipe): lat {CE[arch][f][0]:.0f} ms | FA/hr {CE[arch][f][1]:.1f}")

print("\n===== Spearman(b, mean latency) / Spearman(b, FA/hr) =====")
for arch in ARCHS:
    for name in arms:
        s = []
        for f in FRRS:
            lat = [R[(name, arch, b)][f][0] for b in BS]
            fa = [R[(name, arch, b)][f][1] for b in BS]
            s.append(f"{int(f*100)}%: lat {spearman(BS, lat):+.2f} FA {spearman(BS, fa):+.2f}")
        print(f"{arch:15s} {name:7s} " + " | ".join(s))
