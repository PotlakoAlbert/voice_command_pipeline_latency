import json, glob, os

def analyze(results_path, mfa_dir, name):
    loaded = json.load(open(results_path))
    # Support both old flat-list format and new {b, arch, threshold, results} format
    results = loaded['results'] if isinstance(loaded, dict) and 'results' in loaded else loaded

    total_streams = len(results)
    total_hours = total_streams * 6.0 / 3600.0

    early = []
    on_unknown = 0
    on_noise = 0
    in_gap = 0
    missing_mfa = 0

    for r in results:
        if r['detection_time'] is None:
            continue
        if r['detection_time'] >= r['gt_onset']:
            continue
        sid = r['stream_id']
        mfa_path = os.path.join(mfa_dir, f"stream_{sid:05d}.json")
        if not os.path.exists(mfa_path):
            missing_mfa += 1
            continue
        mfa = json.load(open(mfa_path))
        words = mfa['tiers']['words']['entries']
        unk_start, unk_end, _ = words[0]
        dt = r['detection_time']
        early.append(r)
        if unk_start <= dt <= unk_end:
            on_unknown += 1
        elif dt < unk_start:
            on_noise += 1
        else:
            in_gap += 1

    n_early = len(early)
    false_triggers = on_unknown + on_noise
    streaming_error_rate = false_triggers / total_hours if total_hours > 0 else 0
    n_detected = sum(1 for r in results if r['detection_time'] is not None)

    print(f"=== {name} ===")
    print(f"Total streams: {total_streams} ({total_hours:.4f} hours of audio)")
    print(f"Detected: {n_detected}/{total_streams}")
    print(f"Total early detections: {n_early}")
    if n_early > 0:
        print(f"  On unknown word (false trigger): {on_unknown} ({100*on_unknown/n_early:.1f}%)")
        print(f"  On lead-in noise (false trigger): {on_noise} ({100*on_noise/n_early:.1f}%)")
        print(f"  In gap (ambiguous): {in_gap} ({100*in_gap/n_early:.1f}%)")
    print(f"  Missing MFA files: {missing_mfa}")
    print(f"Genuine false triggers: {false_triggers}")
    print(f"Streaming Error rate: {streaming_error_rate:.3f} false triggers/hour")
    print()
    return {
        "name": name, "total_streams": total_streams, "total_hours": total_hours,
        "n_detected": n_detected, "n_early": n_early, "on_unknown": on_unknown,
        "on_noise": on_noise, "in_gap": in_gap, "missing_mfa": missing_mfa,
        "false_triggers": false_triggers, "streaming_error_rate": streaming_error_rate,
    }

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["crnn", "dilated_resnet", "repcnn"])
    ap.add_argument("--b-values", nargs="+", type=float,
                     default=[0.0, 0.1, 0.2, 0.33, 0.5, 1.0])
    ap.add_argument("--mfa-dir", default="data/mfa_aligned_v2")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    all_results = {}
    for b in args.b_values:
        if b == 0.0:
            if args.arch == "dilated_resnet":
                baseline_name = "resnet"
            elif args.arch == "repcnn":
                baseline_name = "repcnn"
            else:
                baseline_name = "crnn"
            path = f"results/{baseline_name}_latency_v2.json"
        else:
            path = f"results/{args.arch}_b{b}_latency.json"
        if not os.path.exists(path):
            print(f"Skipping b={b}: {path} not found")
            continue
        r = analyze(path, args.mfa_dir, f"{args.arch} b={b}")
        all_results[str(b)] = r

    json.dump(all_results, open(args.output, "w"), indent=2)
    print(f"Saved sweep summary to {args.output}")
