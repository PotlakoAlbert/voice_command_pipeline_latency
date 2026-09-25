"""
train_kws.py

Trainer for KWS architectures (CRNN, Dilated ResNet).

For --b 0.0 (default): standard cross-entropy on utterance-level output
(model.forward()), exactly matching the Phase 1 baseline training path —
untouched for reproducibility.

For --b > 0.0: Phase 2 latency-control training. Uses per-timestep
posteriors (model.forward_frames()) with the Jose et al. [2022]
Bernoulli-shifted max-pooling loss (src/latency_loss.py), pushing the
model to fire earlier in the keyword window at some accuracy cost.

Includes wandb (offline mode) logging and early stopping to avoid
under- or over-training: training halts once validation accuracy shows
no improvement for `--patience` epochs.
"""

import os
import json
import argparse
import time

import torch
import torch.nn as nn
import wandb

from data import get_dataloaders, LABEL_LIST
from models import CRNN, DilatedResNet, RepCNN, count_parameters
from latency_loss import latency_controlled_loss


def build_model(arch, n_mels=40, n_classes=12):
    if arch == "crnn":
        return CRNN(n_mels=n_mels, n_classes=n_classes, conv_channels=37, gru_hidden=72)
    elif arch == "dilated_resnet":
        return DilatedResNet(n_mels=n_mels, n_classes=n_classes, channels=13)
    elif arch == "repcnn":
        return RepCNN(n_mels=n_mels, n_classes=n_classes, c0=48, c1=56, c2=64, n_branches=2)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


@torch.no_grad()
def evaluate(model, loader, device, b=0.0):
    """For b=0.0 (baseline): evaluates via the pooled forward() path,
    matching how the model was trained (whole-clip cross-entropy).

    For b>0.0: the model was trained via forward_frames() with a
    max-pooling-style loss, so evaluation must match that convention:
    take the per-class max posterior across frames (max-pooling
    inference), not the pooled-hidden-state forward() path, which was
    never optimized for and collapses to majority-class prediction."""
    model.eval()
    correct = 0
    total = 0
    for feat, label in loader:
        feat, label = feat.to(device), label.to(device)
        if b == 0.0:
            logits = model(feat)
        else:
            frame_logits = model.forward_frames(feat)  # (batch, time, n_classes)
            logits, _ = frame_logits.max(dim=1)  # max-pool over time per class
        pred = logits.argmax(dim=1)
        correct += (pred == label).sum().item()
        total += label.numel()
    return 100.0 * correct / total


def train_one_epoch(model, loader, optimizer, device, b, log_every=100):
    model.train()
    running = 0.0
    epoch_loss_total = 0.0
    for i, (feat, label) in enumerate(loader, start=1):
        feat, label = feat.to(device), label.to(device)

        optimizer.zero_grad(set_to_none=True)

        if b == 0.0:
            logits = model(feat)
            loss = nn.functional.cross_entropy(logits, label)
        else:
            frame_logits = model.forward_frames(feat)
            loss = latency_controlled_loss(frame_logits, label, b=b)

        loss.backward()
        optimizer.step()

        running += loss.item()
        epoch_loss_total += loss.item()
        if i % log_every == 0:
            avg = running / log_every
            print(f"  step {i:5d}/{len(loader)}  loss {avg:.4f}")
            running = 0.0

    return epoch_loss_total / len(loader)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["crnn", "dilated_resnet", "repcnn"])
    ap.add_argument("--b", type=float, default=0.0,
                     help="Jose et al. latency-control param. b=0.0 -> standard "
                          "cross-entropy baseline. b>0.0 -> latency-controlled "
                          "max-pooling loss via forward_frames().")
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=5,
                     help="Stop if val accuracy doesn't improve for this many epochs")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()

    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print(f"Architecture: {args.arch} | b={args.b}")
    print(f"torch: {torch.__version__} | device: {device}")

    os.environ["WANDB_MODE"] = "offline"
    wandb.init(
        project="voice-kws-phase2" if args.b != 0.0 else "voice-kws-phase1",
        name=f"{args.arch}_b{args.b}",
        config={
            "arch": args.arch, "b": args.b, "epochs": args.epochs,
            "batch_size": args.batch_size, "lr": args.lr, "patience": args.patience,
        },
    )

    train_loader, val_loader, test_loader, labels = get_dataloaders(
        root=args.data, batch_size=args.batch_size, num_workers=args.num_workers,
    )
    print(f"Classes ({len(labels)}): {labels}")
    print(f"Train batches: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")

    model = build_model(args.arch, n_classes=len(labels)).to(device)
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    history = []
    best_val_acc = 0.0
    epochs_without_improvement = 0
    start_time = time.time()
    final_epoch = args.epochs

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, device, b=args.b)

        val_acc = evaluate(model, val_loader, device, b=args.b)
        print(f"  val accuracy: {val_acc:.2f}%")

        wandb.log({"epoch": epoch, "train_loss": avg_train_loss, "val_accuracy": val_acc})

        history.append({"epoch": epoch, "val_accuracy": val_acc, "train_loss": avg_train_loss})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            ckpt_path = os.path.join(
                args.checkpoint_dir, f"{args.arch}_b{args.b}_best.pt"
            )
            torch.save({
                "model_state_dict": model.state_dict(),
                "arch": args.arch,
                "b": args.b,
                "epoch": epoch,
                "val_accuracy": val_acc,
                "n_params": n_params,
            }, ckpt_path)
            print(f"  saved new best checkpoint: {ckpt_path}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs "
                  f"(best val accuracy: {best_val_acc:.2f}% at epoch {epoch - epochs_without_improvement})")
            final_epoch = epoch
            break

    wandb.finish()

    test_acc = evaluate(model, test_loader, device, b=args.b)
    elapsed = time.time() - start_time
    print(f"\nFinal test accuracy: {test_acc:.2f}%")
    print(f"Total training time: {elapsed:.1f}s")

    # RepCNN: save a fused/deploy checkpoint from the best (not final) weights.
    # Fusion is one-way (branches are deleted), so this always loads a fresh
    # instance from the best checkpoint rather than fusing the in-memory model.
    if args.arch == "repcnn":
        best_ckpt_path = os.path.join(
            args.checkpoint_dir, f"{args.arch}_b{args.b}_best.pt"
        )
        fused_model = build_model(args.arch, n_classes=len(labels)).to(device)
        best_state = torch.load(best_ckpt_path, map_location=device)
        fused_model.load_state_dict(best_state["model_state_dict"])
        fused_model.switch_to_deploy()
        fused_params = count_parameters(fused_model)
        fused_ckpt_path = os.path.join(
            args.checkpoint_dir, f"{args.arch}_b{args.b}_best_fused.pt"
        )
        torch.save({
            "model_state_dict": fused_model.state_dict(),
            "arch": args.arch,
            "b": args.b,
            "epoch": best_state["epoch"],
            "val_accuracy": best_state["val_accuracy"],
            "n_params": fused_params,
            "fused": True,
        }, fused_ckpt_path)
        print(f"Fused deploy checkpoint saved: {fused_ckpt_path} ({fused_params} params)")

    results = {
        "arch": args.arch,
        "b": args.b,
        "n_params": n_params,
        "epochs_requested": args.epochs,
        "epochs_run": final_epoch,
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_acc,
        "training_time_seconds": elapsed,
        "history": history,
    }
    results_path = os.path.join(args.results_dir, f"{args.arch}_b{args.b}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {results_path}")


if __name__ == "__main__":
    main()
