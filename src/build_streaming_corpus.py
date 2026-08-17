"""
build_streaming_corpus.py

Constructs the streaming evaluation corpus described in proposal
methodology 4.6: concatenates a randomly selected keyword recording with
a randomly selected background noise clip and a non-keyword ("unknown")
utterance into a continuous 5-8 second audio stream, simulating the
realistic deployment scenario where a KWS system runs continuously and
must fire exactly once per keyword occurrence.

This produces the raw concatenated audio files plus a metadata JSON
recording, for each stream: which keyword was used, its approximate
position in the stream, and the constituent source files. The metadata's
"approximate position" is NOT the ground-truth onset time — that comes
from a separate forced-alignment pass (build_alignment_corpus.py, run
once MFA is available on the cluster) which produces frame-accurate
onset times required for detection-latency measurement per methodology
4.4 and 4.6.

Usage:
    python src/build_streaming_corpus.py \
        --data ~/voice_command_pipeline_latency/data \
        --output ~/voice_command_pipeline_latency/data/streaming_eval \
        --n-utterances 1000
"""

import os
import json
import random
import argparse
from pathlib import Path

import torch
import torchaudio

SAMPLE_RATE = 16000

TARGET_LABELS = [
    "yes", "no", "up", "down", "left", "right",
    "on", "off", "stop", "go",
]


def load_file_list(data_root, list_filename):
    """Reads Speech Commands' own train/val/test split files, which list
    relative paths like 'yes/0a7c2a8d_nohash_0.wav'."""
    path = os.path.join(data_root, list_filename)
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def index_by_label(data_root, split_root):
    """
    Walks the Speech Commands directory and groups .wav file paths by
    their label (the parent folder name). Returns two dicts:
    keyword_files (only the 10 target labels) and unknown_files
    (everything else, excluding _background_noise_).
    """
    keyword_files = {label: [] for label in TARGET_LABELS}
    unknown_files = []

    for entry in os.scandir(split_root):
        if not entry.is_dir():
            continue
        label = entry.name
        if label == "_background_noise_":
            continue

        wavs = [
            os.path.join(split_root, label, f)
            for f in os.listdir(entry.path)
            if f.endswith(".wav")
        ]

        if label in keyword_files:
            keyword_files[label] = wavs
        else:
            unknown_files.extend(wavs)

    return keyword_files, unknown_files


def load_noise_clips(split_root):
    noise_dir = os.path.join(split_root, "_background_noise_")
    if not os.path.isdir(noise_dir):
        return []
    return [
        os.path.join(noise_dir, f)
        for f in os.listdir(noise_dir)
        if f.endswith(".wav")
    ]


def load_and_resample(path, target_sr=SAMPLE_RATE):
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # force mono
    return waveform.squeeze(0)


def random_noise_segment(noise_waveform, length_samples, noise_scale=0.1):
    """Extracts a random segment of the right length from a (typically
    much longer) background noise clip, scaled down so it doesn't
    drown out the speech."""
    if noise_waveform.shape[0] <= length_samples:
        # tile if the noise clip is shorter than needed
        repeats = (length_samples // noise_waveform.shape[0]) + 1
        noise_waveform = noise_waveform.repeat(repeats)
    start = random.randint(0, noise_waveform.shape[0] - length_samples)
    segment = noise_waveform[start:start + length_samples]
    return segment * noise_scale


def build_one_stream(keyword_label, keyword_path, unknown_path, noise_waveform,
                      target_duration_s=6.0, sample_rate=SAMPLE_RATE):
    """
    Builds a single continuous stream: [noise] -> [unknown word] -> [noise]
    -> [KEYWORD] -> [noise], padded/trimmed to approximately
    target_duration_s. Returns the waveform and the approximate keyword
    start sample (pre-forced-alignment estimate, refined later by MFA).
    """
    keyword_wav = load_and_resample(keyword_path)
    unknown_wav = load_and_resample(unknown_path)

    total_samples = int(target_duration_s * sample_rate)
    speech_samples = keyword_wav.shape[0] + unknown_wav.shape[0]
    remaining = max(total_samples - speech_samples, sample_rate)  # ensure some noise padding

    # split remaining noise budget across three gaps: lead-in, mid, tail
    gap1 = remaining // 3
    gap2 = remaining // 3
    gap3 = remaining - gap1 - gap2

    lead_in = random_noise_segment(noise_waveform, gap1)
    mid_gap = random_noise_segment(noise_waveform, gap2)
    tail = random_noise_segment(noise_waveform, gap3)

    stream = torch.cat([lead_in, unknown_wav, mid_gap, keyword_wav, tail])

    # approximate keyword onset (sample index) — refined by MFA later
    approx_keyword_start = lead_in.shape[0] + unknown_wav.shape[0] + mid_gap.shape[0]

    return stream, approx_keyword_start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                     help="Path to the data/ dir containing SpeechCommands/speech_commands_v0.02")
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-utterances", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--duration", type=float, default=6.0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    split_root = os.path.join(
        os.path.expanduser(args.data),
        "SpeechCommands", "speech_commands_v0.02",
    )
    if not os.path.isdir(split_root):
        raise FileNotFoundError(
            f"Speech Commands directory not found at {split_root}. "
            f"Run the download step (see data.py) first."
        )

    os.makedirs(args.output, exist_ok=True)
    audio_out_dir = os.path.join(args.output, "audio")
    os.makedirs(audio_out_dir, exist_ok=True)

    keyword_files, unknown_files = index_by_label(args.data, split_root)
    noise_paths = load_noise_clips(split_root)

    if not noise_paths:
        raise FileNotFoundError(
            "No files found in _background_noise_ — check the dataset download."
        )
    if not unknown_files:
        raise FileNotFoundError("No unknown-class files found — check dataset structure.")

    for label, files in keyword_files.items():
        if not files:
            raise FileNotFoundError(f"No files found for keyword '{label}'")

    print(f"Indexed {sum(len(v) for v in keyword_files.values())} keyword files "
          f"across {len(keyword_files)} labels")
    print(f"Indexed {len(unknown_files)} unknown-class files")
    print(f"Found {len(noise_paths)} background noise clips")

    # Pre-load noise clips (small number of files, safe to hold in memory)
    noise_waveforms = [load_and_resample(p) for p in noise_paths]

    metadata = []

    for i in range(args.n_utterances):
        keyword_label = random.choice(TARGET_LABELS)
        keyword_path = random.choice(keyword_files[keyword_label])
        unknown_path = random.choice(unknown_files)
        noise_waveform = random.choice(noise_waveforms)

        stream, approx_start = build_one_stream(
            keyword_label, keyword_path, unknown_path, noise_waveform,
            target_duration_s=args.duration,
        )

        out_filename = f"stream_{i:05d}.wav"
        out_path = os.path.join(audio_out_dir, out_filename)
        torchaudio.save(out_path, stream.unsqueeze(0), SAMPLE_RATE)

        metadata.append({
            "stream_id": i,
            "filename": out_filename,
            "keyword_label": keyword_label,
            "keyword_source": keyword_path,
            "unknown_source": unknown_path,
            "approx_keyword_start_sample": approx_start,
            "approx_keyword_start_seconds": approx_start / SAMPLE_RATE,
            "duration_seconds": stream.shape[0] / SAMPLE_RATE,
        })

        if (i + 1) % 100 == 0:
            print(f"  built {i + 1}/{args.n_utterances} streams")

    metadata_path = os.path.join(args.output, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone. {args.n_utterances} streams written to {audio_out_dir}")
    print(f"Metadata written to {metadata_path}")
    print("\nNOTE: approx_keyword_start_* fields are estimates from concatenation "
          "offsets, not ground truth. Run forced alignment (MFA) separately to "
          "get frame-accurate onset times for detection-latency measurement.")


if __name__ == "__main__":
    main()
