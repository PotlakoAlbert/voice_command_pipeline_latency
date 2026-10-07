import json, sys
sys.path.insert(0, "src")
from analyze_streaming_error import analyze

out = {}
for arch in ["crnn", "dilated_resnet", "repcnn"]:
    for tl in [0.0, 0.1, 0.2, 0.3]:
        path = f"results/{arch}_ep_tl{tl}_latency.json"
        r = analyze(path, "data/mfa_aligned_v2", f"{arch} ep_tl={tl}")
        out[f"{arch}_ep_tl{tl}"] = r
        print(arch, tl, {k: (round(v, 3) if isinstance(v, float) else v)
                         for k, v in r.items() if isinstance(v, (int, float))})
json.dump(out, open("results/endpoint_streaming_error_sweep.json", "w"), indent=2)
print("SAVED results/endpoint_streaming_error_sweep.json")
