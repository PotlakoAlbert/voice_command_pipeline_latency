import json, torch, torchaudio, numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '/home-mscluster/pledwaba/voice_command_pipeline_latency/src')
from models import CRNN, DilatedResNet

TARGET_LABELS = ["yes","no","up","down","left","right","on","off","stop","go"]
LABEL_TO_IDX = {label:i for i,label in enumerate(TARGET_LABELS)}

def pcen(mel_spec, smooth=0.025, gain=0.98, bias=2.0, power=0.5, eps=1e-6):
    mel_spec = mel_spec + eps
    smoother = torch.zeros_like(mel_spec)
    smoother[..., 0] = mel_spec[..., 0]
    for t in range(1, mel_spec.shape[-1]):
        smoother[..., t] = (1 - smooth) * smoother[..., t-1] + smooth * mel_spec[..., t]
    agc = mel_spec / (smoother + eps) ** gain
    return (agc + bias) ** power - bias ** power

def sliding_window_detection(model, audio_path, device, keyword_label, threshold=0.5, window_sec=1.0, hop_sec=0.1):
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform = waveform.squeeze(0)
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, hop_length=160, n_mels=40)
    window_samples = int(window_sec * 16000)
    hop_samples = int(hop_sec * 16000)
    target_idx = LABEL_TO_IDX[keyword_label]
    for start in range(0, waveform.shape[0] - window_samples + 1, hop_samples):
        segment = waveform[start:start+window_samples]
        mel = mel_transform(segment.unsqueeze(0))
        feat = pcen(mel).squeeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(feat)
            prob = torch.softmax(out, dim=1)
            # Check probability of the target label only
            conf = prob[0, target_idx].item()
            if conf >= threshold:
                detection_time = start / 16000 + window_sec
                return detection_time, conf
    return None, None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', choices=['crnn', 'dilated_resnet'], required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.arch == 'crnn':
        model = CRNN(n_mels=40, n_classes=12, conv_channels=37, gru_hidden=72).to(device)
    else:
        model = DilatedResNet(n_mels=40, n_classes=12).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded {args.arch} on {device}")

    meta_path = Path('/home-mscluster/pledwaba/voice_command_pipeline_latency/data/streaming_eval/metadata.json')
    aligned_dir = Path('/home-mscluster/pledwaba/voice_command_pipeline_latency/data/mfa_aligned_v2')
    meta = json.load(open(meta_path))
    results = []
    for entry in meta:
        stream_id = entry['stream_id']
        aligned_file = aligned_dir / f"stream_{stream_id:05d}.json"
        if not aligned_file.exists():
            continue
        align = json.load(open(aligned_file))
        words = align['tiers']['words']['entries']
        if len(words) < 2:
            continue
        gt_onset = words[1][0]  # second word start
        keyword_label = words[1][2]  # the actual keyword word
        audio_path = Path('/home-mscluster/pledwaba/voice_command_pipeline_latency/data/streaming_eval/audio') / entry['filename']
        det_time, conf = sliding_window_detection(model, audio_path, device, keyword_label, threshold=args.threshold)
        latency = det_time - gt_onset if det_time is not None else None
        results.append({
            'stream_id': stream_id,
            'keyword_label': keyword_label,
            'gt_onset': gt_onset,
            'detection_time': det_time,
            'latency': latency,
            'confidence': conf,
            'threshold': args.threshold
        })
    json.dump(results, open(args.output, 'w'), indent=2)
    detected = [r for r in results if r['detection_time'] is not None]
    if detected:
        latencies = [r['latency'] for r in detected]
        print(f"Detected {len(detected)}/{len(results)}")
        print(f"Mean latency: {np.mean(latencies):.3f}s, Std: {np.std(latencies):.3f}s")
        print(f"Min: {np.min(latencies):.3f}s, Max: {np.max(latencies):.3f}s")
    else:
        print("No detections at threshold", args.threshold)

if __name__ == '__main__':
    main()
