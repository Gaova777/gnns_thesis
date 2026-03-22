"""
Training script for a single GNN configuration.

Usage:
    uv run python scripts/run_training.py --model GCN --scenario "1:10" --balancing focal_loss
    uv run python scripts/run_training.py --model TAGCN --scenario "1:1" --balancing none --epochs 5 --dry-run
"""

import argparse
import torch
import yaml
from pathlib import Path

from src.data.loader import load_elliptic, print_dataset_stats
from src.data.preprocessing import preprocess
from src.data.imbalance import create_imbalance_scenario
from src.training.trainer import Trainer, build_model
from src.balancing.losses import get_loss_function


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GNN on Elliptic Dataset")
    parser.add_argument("--model", type=str, default="GCN", choices=["GCN", "GraphSAGE", "GAT", "TAGCN"])
    parser.add_argument("--scenario", type=str, default="1:10", help="Imbalance ratio (e.g., 1:1, 1:10)")
    parser.add_argument("--balancing", type=str, default="none", choices=["none", "class_weighting", "focal_loss"])
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Train for minimal epochs to verify")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    return parser.parse_args()


def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Load and preprocess data
    print("\n1. Loading Elliptic dataset...")
    data = load_elliptic()
    print_dataset_stats(data)

    print("\n2. Preprocessing...")
    preprocess(data)

    # Apply imbalance scenario
    ratio_map = {"1:1": 1.0, "1:10": 0.1, "1:50": 0.02, "1:100": 0.01}
    ratio = ratio_map.get(args.scenario, 0.1)
    print(f"\n3. Creating imbalance scenario: {args.scenario}")
    data = create_imbalance_scenario(data, target_ratio=ratio, seed=args.seed)

    # Build model
    print(f"\n4. Building model: {args.model}")
    model = build_model(
        args.model,
        in_channels=data.num_node_features,
        hidden_channels=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss function
    loss_fn = get_loss_function(
        args.balancing, labels=data.y, mask=data.train_mask, device=device
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    epochs = 5 if args.dry_run else args.epochs
    run_name = f"{args.model}_{args.scenario}_{args.balancing}_s{args.seed}"

    print(f"\n5. Training ({epochs} epochs)...")
    trainer = Trainer(
        model=model, loss_fn=loss_fn, optimizer=optimizer,
        device=device, patience=args.patience,
    )
    results = trainer.train(data, epochs=epochs, run_name=run_name)

    # Results
    print(f"\n{'='*60}")
    print(f"Training Complete: {run_name}")
    print(f"{'='*60}")
    print(f"Best epoch:   {results['best_epoch']}")
    print(f"Best val MCC: {results['best_val_mcc']:.4f}")
    print(f"Test metrics:")
    for k, v in results["test_metrics"].items():
        print(f"  {k:10s}: {v:.4f}")


if __name__ == "__main__":
    main()
