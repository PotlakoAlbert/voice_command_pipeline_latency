"""
train_kws.py

Phase 1 baseline trainer for KWS architectures (CRNN, Dilated ResNet).

Trains a single architecture with standard cross-entropy loss on the
12-class Speech Commands task (Warden 2018) and reports Top-One accuracy.

Includes wandb (offline mode) logging and early stopping to avoid
under- or over-training: training halts once validation accuracy shows
no improvement for `--patience` epochs, restoring confidence that the
saved checkpoint reflects the model's actual convergence point.

NOTE on --b: the Jose et al. [2022] latency-control mechanism requires
frame-level logits; current models only produce utterance-level output.
--b is accepted for CLI compatibility but has no effect yet (Phase 2 work).
"""

import os
import json
import argparse
import time

import torch
import torch.nn as nn
import wandb

from data import get_dataloaders, LABEL_LIST
from models import CRNN, DilatedResNet, count_parameters


def build_model(arch, n_mels=40, n_classes=12):
    if arch == "crnn":
        return CRNN(n_mels=n_mels, n_classes=n_classes, conv_channels=37, gru_hidden=72)
    elif arch == "dilated_resnet":
        return DilatedResNet(n_mels=n_mels, n_classes=n_classes, channels=13)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for feat, label in loader:
        feat, label = feat.to(device), label.to(device)
        logits = model(feat)
        pred = logits.argmax(dim=1)
        correct += (pred == label).sum().item()
        total += label.numel()
    return 100.0 * correct / total


def train_one_epoch(model, loader, optimizer, loss_fn, device, log_every=100):
    model.train()
    running = 0.0
    epoch_loss_total = 0.0
    for i, (feat, label) in enumerate(loader, start=1):
        feat, label = feat.to(device), label.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(feat)
        loss = loss_fn(logits, label)
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
    ap.add_argument("--arch", required=True, choices=["crnn", "dilated_resnet"])
    ap.add_argument("--b", type=float, default=0.0,
                     help="Jose et al. latency-control param. Not yet implemented.")
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

    if args.b != 0.0:
        print(f"WARNING: --b={args.b} was passed but latency-control is not yet "
              f"implemented. Training will proceed with standard cross-entropy loss.")

    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print(f"Architecture: {args.arch}")
    print(f"torch: {torch.__version__} | device: {device}")

    os.environ["WANDB_MODE"] = "offline"
    wandb.init(
        project="voice-kws-phase1",
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
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    history = []
    best_val_acc = 0.0
    epochs_without_improvement = 0
    start_time = time.time()
    final_epoch = args.epochs

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)

        val_acc = evaluate(model, val_loader, device)
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

    test_acc = evaluate(model, test_loader, device)
    elapsed = time.time() - start_time
    print(f"\nFinal test accuracy: {test_acc:.2f}%")
    print(f"Total training time: {elapsed:.1f}s")

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
