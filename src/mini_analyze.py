import glob, json, os, re
import numpy as np

SR, N_WIN = 16000, 31
TEND = 2.0 + 0.1 * np.arange(N_WIN)      # window k ends at 2.0 + 0.1k s
FRRS = [0.02, 0.05, 0.10]
NEG_SEC = 4.0                            # scored audio per negative segment (windows cover 1.0-5.0 s)


def analyze(d):
    P, N = d["pos_p"], d["neg_p"]
    ks, ke = d["pos_ks"] / SR, d["pos_ke"] / SR
    Pv = np.where(TEND[None, :] >= ks[:, None], P, -1.0)   # only windows ending after keyword start
    ps, ns = Pv.max(1), N.max(1)
    hours = len(N) * NEG_SEC / 3600.0
    out = {}
    for f in FRRS:
        thr = float(np.sort(ps)[int(np.floor(f * len(ps)))])
        det = ps >= thr
        first = (Pv >= thr).argmax(1)
        lat = (TEND[first] - ke)[det] * 1000.0
        fa = int((ns >= thr).sum())
        out[f] = {"thr": thr, "frr": float(1 - det.mean()), "lat_mean": float(lat.mean()),
                  "lat_med": float(np.median(lat)), "fa_cnt": fa, "fa_hr": fa / hours}
    return out


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]) if np.std(ry) > 0 else float("nan")


res = {}
for npz in sorted(glob.glob("results/mini/*_s*.npz")):
    tag = os.path.basename(npz)[:-4]
    m = re.match(r"(.+)_b([0-9.]+)_s(\d+)$", tag)
    arch, b, seed = m.group(1), float(m.group(2)), int(m.group(3))
    meta = json.load(open(npz[:-4] + ".json"))
    res[(arch, b, seed)] = (analyze(np.load(npz)), meta["best_val_auc"], meta["epochs_run"])

summary = {}
for f in FRRS:
    print(f"\n===== false-reject target {int(f * 100)}% =====")
    for arch in ["crnn", "dilated_resnet", "repcnn"]:
        keys = sorted(k for k in res if k[0] == arch)
        if not keys:
            continue
        base = res[(arch, 0.0, 0)][0][f] if (arch, 0.0, 0) in res else None
        print(f"{arch}:  b    valAUC  ep  FRR%   lat_mean  lat_med  FA(cnt)  FA/hr  dLat_vs_b0  FA_x_vs_b0")
        bs, lm, fh = [], [], []
        for k in keys:
            r, auc, ep = res[k]; r = r[f]
            dl = r["lat_mean"] - base["lat_mean"] if base else float("nan")
            fx = r["fa_hr"] / base["fa_hr"] if base and base["fa_hr"] > 0 else float("nan")
            print(f"        {k[1]:<5} {auc:.4f} {ep:3d} {r['frr']*100:5.1f} {r['lat_mean']:9.0f} {r['lat_med']:8.0f} {r['fa_cnt']:8d} {r['fa_hr']:6.2f} {dl:10.0f} {fx:10.2f}")
            bs.append(k[1]); lm.append(r["lat_mean"]); fh.append(r["fa_hr"])
            summary[f"{arch}_b{k[1]}_frr{f}"] = r
        print(f"        Spearman(b, mean latency) = {spearman(bs, lm):+.2f}   Spearman(b, FA/hr) = {spearman(bs, fh):+.2f}")
json.dump(summary, open("results/mini_summary.json", "w"), indent=2)
print("\nSAVED results/mini_summary.json")
