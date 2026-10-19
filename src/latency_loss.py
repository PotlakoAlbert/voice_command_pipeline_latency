"""
latency_loss.py

Latency-controlled max-pooling loss, following Jose et al. [2022]
"Latency Control for Keyword Spotting" (arXiv:2206.07261), adapted from
their binary "Amazon" wake-word setup to this project's 12-way multi-
keyword classification (10 target words + silence + unknown).

Standard max-pooling loss (Sun et al. 2016) picks, per example, the frame
that best supports the label:
  - positive example (a real keyword, y in TARGET_LABELS): frame with
    max posterior for class y
  - negative example (silence/unknown): frame with min posterior for
    that negative class (== hardest frame to reject)

Jose et al. extend this for positive examples only: with probability b,
shift the selected frame one step earlier than the argmax frame (clamped
at 0), pushing the model to fire earlier without needing the keyword's
true endpoint. b=0 recovers standard max-pooling loss exactly.

Expects model output from `forward_frames()`: logits of shape
(batch, time, n_classes).

Vectorized across the batch dimension (no per-example Python loop) for
GPU throughput.
"""

import torch
import torch.nn.functional as F

# Matches LABEL_LIST in data.py: TARGET_LABELS + [SILENCE_LABEL, UNKNOWN_LABEL]
# Indices 0-9 are real keywords, 10-11 are negative classes.
NEGATIVE_CLASS_INDICES = (10, 11)


def latency_controlled_loss(logits, labels, b=0.0, negative_indices=None):
    """
    Args:
        logits: (batch, time, n_classes) raw model outputs per frame.
        labels: (batch,) integer class label per example, matching
            data.py's LABEL_LIST ordering (0-9 = keywords, 10 = silence,
            11 = unknown).
        b: Bernoulli probability of shifting the target frame one step
            earlier for positive (real-keyword) examples. b=0.0 ->
            standard max-pooling.

    Returns:
        scalar loss (mean over batch).
    """
    batch, time, n_classes = logits.shape
    device = logits.device
    log_probs = F.log_softmax(logits, dim=-1)  # (batch, time, n_classes)

    # Posteriors for each example's own true class, at every frame: (batch, time)
    batch_idx = torch.arange(batch, device=device)
    class_log_probs = log_probs[batch_idx.unsqueeze(1), :, labels.unsqueeze(1)]
    # class_log_probs shape: (batch, time) after squeeze
    if class_log_probs.dim() == 3:
        class_log_probs = class_log_probs.squeeze(1)

    is_negative = torch.zeros(batch, dtype=torch.bool, device=device)
    neg_set = NEGATIVE_CLASS_INDICES if negative_indices is None else negative_indices
    for neg_idx in neg_set:
        is_negative |= (labels == neg_idx)
    is_positive = ~is_negative

    t_max = torch.argmax(class_log_probs, dim=1)   # (batch,) best frame per example
    t_min = torch.argmin(class_log_probs, dim=1)   # (batch,) worst frame per example

    # Bernoulli shift, only applied where relevant (positives)
    beta = torch.bernoulli(torch.full((batch,), float(b), device=device)).long()
    t_pos = torch.clamp(t_max - beta, min=0)

    # Select final frame index per example: negatives use t_min, positives use t_pos
    t = torch.where(is_negative, t_min, t_pos)

    selected_log_probs = class_log_probs[batch_idx, t]  # (batch,)
    loss = -selected_log_probs.mean()
    return loss
