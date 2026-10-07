"""
endpoint_loss.py

Endpoint-masking baseline: latency-aware max-pooling loss of Fu et al.
[2022] "Unified Speculation, Detection, and Verification Keyword Spotting"
(ICASSP 2022), Eq. (1).

Positives: max-pool only over frames i with (i - e) <= t_l (e = keyword end
frame, t_l = target latency); later frames are discarded.
Negatives: unchanged (frame with lowest score on the true class).

Requires the keyword endpoint e. Speech Commands training clips have no
forced alignment (MFA only covers the streaming eval corpus), so e is
estimated with an energy VAD. State this as a limitation in the writeup.

Label scheme (data.py LABEL_LIST): 0-9 keywords, 10 silence, 11 unknown.
"""

import torch
import torch.nn.functional as F

from latency_loss import NEGATIVE_CLASS_INDICES


def estimate_endpoint_frame(waveform, hop_length=160, win_length=480, rel_db=-35.0):
    """Last mel-frame whose RMS is within `rel_db` dB of the clip's peak RMS.

    Mirrors MelSpectrogram(n_fft=480, hop_length=160, center=True): a 1 s
    clip at 16 kHz gives 101 frames. Silent clips return the last frame.
    """
    x = waveform
    if x.dim() > 1:
        if x.shape[0] > 1:
            x = x.mean(dim=0, keepdim=True)
        x = x.reshape(-1)
    x = x.float()
    x = F.pad(x[None, None, :], (win_length // 2, win_length // 2), mode="reflect")
    x = x.squeeze(0).squeeze(0)
    frames = x.unfold(0, win_length, hop_length)          # (T, win_length)
    rms_db = 20.0 * torch.log10(frames.pow(2).mean(dim=1).sqrt() + 1e-8)
    active = rms_db >= (rms_db.max() + rel_db)
    idx = torch.nonzero(active).reshape(-1)
    return int(idx[-1]) if idx.numel() > 0 else frames.shape[0] - 1


def endpoint_masked_loss(logits, labels, end_frames, n_mel_frames, target_latency_s):
    """
    Args:
        logits: (batch, time, n_classes) from model.forward_frames().
        labels: (batch,) class indices (0-9 keywords, 10 silence, 11 unknown).
        end_frames: (batch,) keyword end frame in MEL-frame units.
        n_mel_frames: mel frames per clip (101 for 1 s at hop 160); rescales
            end_frames to the model's output frame rate.
        target_latency_s: t_l in seconds after the keyword end (may be < 0).
    Returns:
        scalar loss (mean over batch).
    """
    batch, time, _ = logits.shape
    device = logits.device

    log_probs = F.log_softmax(logits, dim=-1)
    class_lp = log_probs.gather(
        2, labels.view(-1, 1, 1).expand(-1, time, 1)
    ).squeeze(2)                                          # (batch, time)

    neg = torch.tensor(NEGATIVE_CLASS_INDICES, device=device)
    is_negative = torch.isin(labels, neg)

    end_frames = end_frames.to(device).float().clamp(0, n_mel_frames - 1)
    e_out = end_frames * (time / float(n_mel_frames))     # clip is 1 s long
    # Clamp so frame 0 is always legitimately allowed (avoids argmax over an
    # all -inf row when t_l is very negative).
    limit = torch.clamp(e_out + target_latency_s * time, min=0.0)

    frame_idx = torch.arange(time, device=device).unsqueeze(0).float()
    allowed = frame_idx <= limit.unsqueeze(1)

    masked = class_lp.masked_fill(~allowed, float("-inf"))
    t_pos = torch.argmax(masked, dim=1)
    t_neg = torch.argmin(class_lp, dim=1)
    t = torch.where(is_negative, t_neg, t_pos)

    batch_idx = torch.arange(batch, device=device)
    return -class_lp[batch_idx, t].mean()


if __name__ == "__main__":
    from latency_loss import latency_controlled_loss

    torch.manual_seed(0)
    sr = 16000

    # 1) VAD: burst occupying samples 4000-9000 -> end frame ~57
    wav = torch.zeros(1, sr)
    tt = torch.arange(5000).float() / sr
    wav[0, 4000:9000] = 0.5 * torch.sin(2 * torch.pi * 440 * tt)
    e = estimate_endpoint_frame(wav)
    print("VAD endpoint frame:", e, "(expect ~57)")
    assert 55 <= e <= 58

    # 2) finite loss + gradients, including a very negative t_l
    B, T, C = 4, 51, 12
    labels = torch.tensor([0, 3, 10, 11])
    ends = torch.full((B,), e)
    for tl in (0.0, 0.1, -0.2, -5.0):
        lg = torch.randn(B, T, C, requires_grad=True)
        loss = endpoint_masked_loss(lg, labels, ends, 101, tl)
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(lg.grad).all()
        print(f"t_l={tl:+.1f}s  loss={loss.item():.4f}  OK")

    # 3) masking can only raise the positive-class loss; negatives unchanged
    lg = torch.zeros(B, T, C)
    for i, y in enumerate(labels.tolist()):
        lg[i, -1, y] = 5.0            # model "fires" only on the last frame
    unmasked = latency_controlled_loss(lg, labels, b=0.0)
    masked = endpoint_masked_loss(lg, labels, ends, 101, 0.0)
    print(f"unmasked={unmasked.item():.4f}  masked={masked.item():.4f}")
    assert masked > unmasked
    neg_only = torch.tensor([10, 11])
    u2 = latency_controlled_loss(lg[2:], neg_only, b=0.0)
    m2 = endpoint_masked_loss(lg[2:], neg_only, ends[2:], 101, 0.0)
    assert torch.allclose(u2, m2)
    print("ALL CHECKS PASSED")
