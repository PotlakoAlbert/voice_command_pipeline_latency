import numpy as np

SR, N_WIN = 16000, 31
TEND = 2.0 + 0.1 * np.arange(N_WIN)
FRRS = [0.02, 0.05, 0.10]
ARCHS = ["crnn", "dilated_resnet", "repcnn"]
BS = [0.0, 0.1, 0.2, 0.33, 0.5, 1.0]
SEEDS = [0, 1, 2]
ARMS = {
    "window (trailing speech)": "results/mini",
    "frame (1-frame shift)": "results/mini_frame",
    "frame (100 ms shift)": "results/mini_frame_s100",
}


def analyze(path):
    d = np.load(path)
    P, N = d["pos_p"], d["neg_p"]
    ks, ke = d["pos_ks"] / SR, d["pos_ke"] / SR
    Pv = np.where(TEND[None, :] >= ks[:, None], P, -1.0)
    ps = Pv.max(1)
    out = {}
    for f in FRRS:
        thr = float(np.sort(ps)[int(np.floor(f * len(ps)))])
        det = ps >= thr
        first = (Pv >= thr).argmax(1)
        out[f] = float(((TEND[first] - ke)[det] * 1000.0).mean())
    return out


def rank(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    s = x[o]
    r = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and s[j + 1] == s[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = rank(x), rank(y)
    return float(np.corrcoef(rx, ry)[0, 1]) if rx.std() > 0 and ry.std() > 0 else float("nan")


R = {}
for a, d in ARMS.items():
    for ar in ARCHS:
        for b in BS:
            for s in SEEDS:
                R[(a, ar, b, s)] = analyze(d + "/" + ar + "_b" + str(b) + "_s" + str(s) + ".npz")

print("===== mean latency ms from keyword end, 5% FRR (mean over 3 seeds) =====")
print("%-26s %-15s " % ("arm", "arch") + " ".join("b=%-5s" % b for b in BS))
for a in ARMS:
    for ar in ARCHS:
        m = [np.mean([R[(a, ar, b, s)][0.05] for s in SEEDS]) for b in BS]
        print("%-26s %-15s " % (a, ar) + " ".join("%7.0f" % v for v in m))

print("\n===== SUMMARY over 27 cells per arm (3 archs x 3 FRR x 3 seeds) =====")
for a in ARMS:
    rho, dl = [], []
    for f in FRRS:
        for ar in ARCHS:
            for s in SEEDS:
                lat = [R[(a, ar, b, s)][f] for b in BS]
                rho.append(spearman(BS, lat))
                dl.append(lat[-1] - lat[0])
    rho, dl = np.array(rho), np.array(dl)
    print("[%s] rho(b,lat): median %+.2f min %+.2f max %+.2f | rho<=-0.9: %d/27  rho<=-0.6: %d/27  rho>-0.3: %d/27 | d(b1-b0) ms: median %.0f min %.0f max %.0f"
          % (a, np.median(rho), rho.min(), rho.max(), (rho <= -0.9).sum(), (rho <= -0.6).sum(), (rho > -0.3).sum(),
             np.median(dl), dl.min(), dl.max()))

print("\nPre-set rule: shift granularity explains the non-monotonicity if the 100 ms-shift frame arm has median rho <= -0.9 and median b1-b0 change of ~150 ms or more;")
print("it does not if the arm stays near the original frame arm (median rho ~ -0.4, tens of ms); in between = partial.")
