import numpy as np

SR, N_WIN = 16000, 31
TEND = 2.0 + 0.1 * np.arange(N_WIN)
FRRS = [0.02, 0.05, 0.10]
HOURS = 4000 * 4.0 / 3600.0
ARCHS = ["crnn", "dilated_resnet", "repcnn"]
BS = [0.0, 0.1, 0.2, 0.33, 0.5, 1.0]
SEEDS = [0, 1, 2]
ARMS = {"window (trailing speech)": "results/mini", "window NO trailing speech": "results/mini_notrail",
        "frame (original structure)": "results/mini_frame"}


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
    o = np.argsort(x, kind="mergesort"); s = x[o]; r = np.empty(len(x)); i = 0
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


R = {(a, ar, b, s): analyze(f"{d}/{ar}_b{b}_s{s}.npz")
     for a, d in ARMS.items() for ar in ARCHS for b in BS for s in SEEDS}

print("===== mean latency ms from keyword end, 5% FRR (mean over 3 seeds) =====")
print(f"{'arm':28s} {'arch':15s} " + " ".join(f"b={b:<5}" for b in BS))
for a in ARMS:
    for ar in ARCHS:
        m = [np.mean([R[(a, ar, b, s)][0.05][0] for s in SEEDS]) for b in BS]
        print(f"{a:28s} {ar:15s} " + " ".join(f"{v:7.0f}" for v in m))

print("\n===== SUMMARY over 27 cells per arm (3 archs x 3 FRR x 3 seeds) =====")
for a in ARMS:
    rho, dl, fa0, fa1 = [], [], [], []
    for f in FRRS:
        for ar in ARCHS:
            for s in SEEDS:
                lat = [R[(a, ar, b, s)][f][0] for b in BS]
                rho.append(spearman(BS, lat)); dl.append(lat[-1] - lat[0])
    rho, dl = np.array(rho), np.array(dl)
    print(f"[{a}] rho(b,lat): median {np.median(rho):+.2f} min {rho.min():+.2f} max {rho.max():+.2f} | "
          f"rho<=-0.9: {(rho <= -0.9).sum()}/27  rho<=-0.6: {(rho <= -0.6).sum()}/27  rho>-0.3: {(rho > -0.3).sum()}/27 | "
          f"d(b1-b0) ms: median {np.median(dl):.0f} min {dl.min():.0f} max {dl.max():.0f}")
print("\nPre-set rule: trailing speech is the driver if no-trail median rho is near the frame arm (~-0.4) with b1-b0 change of tens of ms;")
print("not the driver if no-trail stays near the window arm (rho ~ -1, change ~200 ms); in between = partial.")
