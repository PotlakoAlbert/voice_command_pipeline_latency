import sys
sys.path.insert(0, "src")
import torch
from latency_loss import latency_controlled_loss

torch.manual_seed(0)
lg = torch.randn(6, 101, 2)
lab = torch.tensor([0, 1, 0, 1, 0, 1])
cls = torch.log_softmax(lg, -1)[torch.arange(6), :, lab]
ar = torch.arange(6)
for shift in (1, 5, 10):
    t = torch.where(lab == 0, cls.argmin(1), (cls.argmax(1) - shift).clamp(min=0))
    exp = -cls[ar, t].mean()
    got = latency_controlled_loss(lg, lab, b=1.0, negative_indices=(0,), shift=shift)
    assert torch.allclose(got, exp, atol=1e-6), (shift, got.item(), exp.item())
    print("PASS shift", shift, "%.5f vs %.5f" % (got.item(), exp.item()))
d = latency_controlled_loss(lg, lab, b=1.0, negative_indices=(0,))
assert torch.allclose(d, latency_controlled_loss(lg, lab, b=1.0, negative_indices=(0,), shift=1))
lg12 = torch.randn(8, 30, 12)
lab12 = torch.tensor([0, 3, 10, 11, 5, 10, 9, 11])
assert torch.allclose(latency_controlled_loss(lg12, lab12, b=0.0), latency_controlled_loss(lg12, lab12, b=0.0, shift=7))
print("PASS default unchanged; b=0 unaffected by shift")
print("SHIFT CHECK PASSED")
