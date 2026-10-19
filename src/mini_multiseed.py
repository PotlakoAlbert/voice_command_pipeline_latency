import json, os
import numpy as np

SR, N_WIN = 16000, 31
TEND = 2.0 + 0.1 * np.arange(N_WIN)
FRRS = [0.02, 0.05, 0.10]
HOURS = 4000 * 4.0 / 3600.0
ARCHS = ["crnn", "dilated_resnet", "repcnn"]
BS = [0.0, 0.1, 0.2, 0.33, 0.5, 1.0]
SEEDS = [0, 1, 2]
ARMS = {"window": "results/mini", "frame": "results/mini_frame"}
DETAIL = 0.05


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
        out[f] = (float(lat.mean()), int((ns >= thr).sum()) / HOURS)
    return out


def rank(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    s = x[order]
    r = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and s[j + 1] == s[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


R = {}
for arm, d in ARMS.items():
    for arch in ARCHS:
        for b in BS:
            for s in SEEDS:
                p = f"{d}/{arch}_b{b}_s{s}.npz"
                assert os.path.exists(p), p
                R[(arm, arch, b, s)] = analyze(p)

summary = {}
cells = {arm: {"rho": [], "delta": []} for arm in ARMS}
for f in FRRS:
    print(f"\n################ FRR target {int(f * 100)}% ################")
    for arm in ARMS:
        for arch in ARCHS:
            lat = np.array([[R[(arm, arch, b, s)][f][0] for b in BS] for s in SEEDS])   # seed x b
            fa = np.array([[R[(arm, arch, b, s)][f][1] for b in BS] for s in SEEDS])
            rho = [spearman(BS, lat[i]) for i in range(len(SEEDS))]
            rho_fa = [spearman(BS, fa[i]) for i in range(len(SEEDS))]
            delta = lat[:, -1] - lat[:, 0]
            rho_mean = spearman(BS, lat.mean(0))
            cells[arm]["rho"] += rho
            cells[arm]["delta"] += list(delta)
            summary[f"{arm}_{arch}_frr{f}"] = {"lat_mean": lat.mean(0).tolist(), "lat_std": lat.std(0, ddof=1).tolist(),
                                              "rho_per_seed": rho, "rho_of_mean": rho_mean, "delta_b1_minus_b0": delta.tolist(),
                                              "fa_mean": fa.mean(0).tolist(), "rho_fa_per_seed": rho_fa}
            print(f"[{arm:6s}] {arch:15s} rho(b,lat) per seed {np.round(rho, 2).tolist()}  seed-mean rho {rho_mean:+.2f}  "
                  f"d(b1-b0) ms {np.round(delta).astype(int).tolist()}  rho(b,FA) {np.round(rho_fa, 2).tolist()}")
            if abs(f - DETAIL) < 1e-9:
                print("          b      lat_mean+-std (ms)     FA/hr mean+-std")
                for j, b in enumerate(BS):
                    print(f"          {b:<5}  {lat[:, j].mean():7.0f} +- {lat[:, j].std(ddof=1):4.0f}     {fa[:, j].mean():7.1f} +- {fa[:, j].std(ddof=1):5.1f}")

print("\n################ SUMMARY over 27 cells per arm (3 archs x 3 FRR x 3 seeds) ################")
for arm in ARMS:
    r = np.array(cells[arm]["rho"]); dl = np.array(cells[arm]["delta"])
    print(f"[{arm}] rho(b,lat): median {np.median(r):+.2f}  min {r.min():+.2f}  max {r.max():+.2f} | "
          f"rho<=-0.9: {(r <= -0.9).sum()}/27  rho<=-0.6: {(r <= -0.6).sum()}/27  rho>-0.3: {(r > -0.3).sum()}/27 | "
          f"d(b1-b0) ms: median {np.median(dl):.0f}  min {dl.min():.0f}  max {dl.max():.0f}")
json.dump(summary, open("results/mini_multiseed_summary.json", "w"), indent=2)
print("SAVED results/mini_multiseed_summary.json")
