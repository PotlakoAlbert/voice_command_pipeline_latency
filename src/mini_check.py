import sys
import numpy as np
import torch
sys.path.insert(0, "src")
from latency_loss import latency_controlled_loss
from mini_data import MiniData, WindowFeats, window_logits, SR, SEG, N_WIN
from models import CRNN, DilatedResNet, RepCNN

torch.set_num_threads(4)
ok = True


def check(name, cond, info=""):
    global ok
    ok = ok and bool(cond)
    print(("PASS " if cond else "FAIL ") + name, info, flush=True)


# 1. loss checks
torch.manual_seed(0)
lg = torch.randn(6, 30, 2)
lab = torch.tensor([0, 1, 0, 1, 0, 1])
cls = torch.log_softmax(lg, -1)[torch.arange(6), :, lab]
ar = torch.arange(6)
t0 = torch.where(lab == 0, cls.argmin(1), cls.argmax(1))
e0 = -cls[ar, t0].mean()
l0 = latency_controlled_loss(lg, lab, b=0.0, negative_indices=(0,))
check("binary loss b=0 matches manual", torch.allclose(l0, e0, atol=1e-6), f"{l0.item():.5f} vs {e0.item():.5f}")
t1 = torch.where(lab == 0, cls.argmin(1), (cls.argmax(1) - 1).clamp(min=0))
e1 = -cls[ar, t1].mean()
l1 = latency_controlled_loss(lg, lab, b=1.0, negative_indices=(0,))
check("binary loss b=1 shifts positives 1 step earlier", torch.allclose(l1, e1, atol=1e-6), f"{l1.item():.5f} vs {e1.item():.5f}")
lg12 = torch.randn(8, 30, 12)
lab12 = torch.tensor([0, 3, 10, 11, 5, 10, 9, 11])
check("12-class default unchanged",
      torch.allclose(latency_controlled_loss(lg12, lab12, b=0.0),
                     latency_controlled_loss(lg12, lab12, b=0.0, negative_indices=(10, 11))))

# 2. cache VAD bounds for all splits, then generator checks on val
for s in ("train", "val", "test"):
    MiniData(s)
    print("bounds cached:", s, flush=True)
m = MiniData("val")
rng = np.random.default_rng(0)
ks_l, ke_l, bad = [], [], 0
segs = []
for i in range(200):
    w, ks, ke = m.make_segment(rng, True)
    if not (np.isfinite(w).all() and np.abs(w).max() <= 1.0 and len(w) == SEG and 1.5 * SR <= ks < ke <= SEG - SR):
        bad += 1
    ks_l.append(ks / SR); ke_l.append(ke / SR)
    segs.append(w)
check("positive segments well-formed", bad == 0,
      f"kw start {np.mean(ks_l):.2f}s, end {np.mean(ke_l):.2f}s, mean len {np.mean(np.array(ke_l)-np.array(ks_l)):.2f}s")
negs = [m.make_segment(rng, False)[0] for _ in range(20)]
check("negative segments finite", all(np.isfinite(w).all() for w in negs))

# 3. shapes + models
wav = torch.from_numpy(np.stack(segs[:4]))
feats = WindowFeats()
f = feats(wav)
check("window feature shape", tuple(f.shape) == (4, N_WIN, 40, 98), str(tuple(f.shape)))
models = {
    "crnn": CRNN(n_mels=40, n_classes=2, conv_channels=37, gru_hidden=72),
    "dilated_resnet": DilatedResNet(n_mels=40, n_classes=2),
    "repcnn": RepCNN(n_mels=40, n_classes=2, c0=48, c1=56, c2=64, n_branches=2),
}

# 4. causality of window posteriors: perturb audio after 3.0 s
wav2 = wav.clone()
wav2[:, 3 * SR:] = torch.randn(4, SEG - 3 * SR) * 0.1
for name, mod in models.items():
    mod.eval()
    with torch.no_grad():
        a = window_logits(mod, feats, wav)
        b = window_logits(mod, feats, wav2)
    check(f"{name}: logits shape", tuple(a.shape) == (4, N_WIN, 2), str(tuple(a.shape)))
    d = (a[:, :11] - b[:, :11]).abs().max().item()   # windows ending <= 3.0 s
    d_late = (a[:, 11:] - b[:, 11:]).abs().max().item()
    check(f"{name}: windows ending before 3.0 s unaffected by later audio", d < 1e-5,
          f"max diff early {d:.2e}, late {d_late:.2e}")
print("ALL PASSED" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
