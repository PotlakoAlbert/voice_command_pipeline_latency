import sys
sys.path.insert(0, "src")
import numpy as np
from mini_data import MiniData, SR

m = MiniData("val")
rng = np.random.default_rng(0)
res = {}
for trail in (True, False):
    r = []
    for _ in range(100):
        w, ks, ke = m.make_segment(rng, True, trail)
        r.append(np.sqrt(np.mean(w[ke + int(0.2 * SR):] ** 2)))
    res[trail] = float(np.mean(r))
    print("trail", trail, "mean RMS 0.2 s after keyword end: %.4f" % res[trail])
assert res[True] > 3 * res[False], "no-trail positives should be much quieter after the keyword"
print("TRAIL CHECK PASSED")
