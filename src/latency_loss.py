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
"""

import torch
import torch.nn.functional as F

# Matches LABEL_LIST in data.py: TARGET_LABELS + [SILENCE_LABEL, UNKNOWN_LABEL]
# Indices 0-9 are real keywords, 10-11 are negative classes.
NEGATIVE_CLASS_INDICES = {10, 11}


def latency_controlled_loss(logits, labels, b=0.0):
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
    log_probs = F.log_softmax(logits, dim=-1)  # (batch, time, n_classes)

    losses = []
    for i in range(batch):
        y = labels[i].item()
        lp = log_probs[i]  # (time, n_classes)

        if y in NEGATIVE_CLASS_INDICES:
            # Negative example: pick the hardest frame to reject
            # (min posterior for the true negative class).
            t = torch.argmin(lp[:, y]).item()
        else:
            # Positive example (real keyword): frame with max posterior
            # for the true class, optionally shifted earlier by Bernoulli(b).
            t_max = torch.argmax(lp[:, y]).item()
            beta = int(torch.bernoulli(torch.tensor(float(b))).item())
            t = max(t_max - beta, 0)

        losses.append(-lp[t, y])

    return torch.stack(losses).mean()
