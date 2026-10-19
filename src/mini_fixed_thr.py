import numpy as np
SR, N_WIN = 16000, 31
TEND = 2.0 + 0.1 * np.arange(N_WIN)
for arch in ["crnn", "dilated_resnet", "repcnn"]:
    print(arch)
    for b in [0.0, 0.1, 0.2, 0.33, 0.5, 1.0]:
        d = np.load(f"results/mini/{arch}_b{b}_s0.npz")
        P, N = d["pos_p"], d["neg_p"]
        ks, ke = d["pos_ks"] / SR, d["pos_ke"] / SR
        Pv = np.where(TEND[None, :] >= ks[:, None], P, -1.0)
        det = Pv.max(1) >= 0.5
        first = (Pv >= 0.5).argmax(1)
        lat = (TEND[first] - ke)[det] * 1000
        fa = int((N.max(1) >= 0.5).sum())
        print(f"  b={b:<5} FRR={100 * (1 - det.mean()):5.1f}%  lat_mean={lat.mean() if det.any() else float('nan'):7.0f}ms  FA/hr={fa / (len(N) * 4 / 3600):7.2f}")
