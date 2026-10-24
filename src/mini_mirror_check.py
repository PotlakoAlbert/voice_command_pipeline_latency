import sys
sys.path.insert(0, "src")
import numpy as np
from mini_data import MiniData, SR

m = MiniData("val")
rng = np.random.default_rng(0)


def tail_rms(**kw):
    r = []
    for _ in range(100):
        w, ks, ke = m.make_segment(rng, **kw)
        r.append(np.sqrt(np.mean(w[4 * SR:] ** 2)))      # last second of the track
    return float(np.mean(r))


pos_nt = tail_rms(positive=True, trail=False)
neg_old = tail_rms(positive=False, trail=False)
neg_mir = tail_rms(positive=False, trail=False, mirror=True)
print("tail RMS (4-5 s): no-trail positives %.4f | old-style negatives %.4f | mirrored negatives %.4f" % (pos_nt, neg_old, neg_mir))
assert neg_old > 3 * pos_nt, "old negatives should be much louder than no-trail positives (the shortcut)"
assert neg_mir < 2 * pos_nt and pos_nt < 2 * neg_mir, "mirrored negatives should match no-trail positives"
print("MIRROR CHECK PASSED")
