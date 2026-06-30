"""
Preprocessing pipeline for the Elliptic dataset.

Handles:
  - Feature normalization (StandardScaler fit on train only)
  - Temporal (time-based) train/val/test splitting
  - Mask generation for transductive learning
"""

import torch
import numpy as np
from sklearn.preprocessing import RobustScaler
from typing import Tuple


def create_time_split_masks(
    data,
    train_range: Tuple[int, int] = (1, 34),
    val_range: Tuple[int, int] = (35, 42),
    test_range: Tuple[int, int] = (43, 49),
) -> None:
    """
    Create train/val/test masks based on temporal timesteps.

    The Elliptic dataset has 49 timesteps. We split causally:
      - Train: timesteps 1-34  (~70%)
      - Val:   timesteps 35-42 (~15%)
      - Test:  timesteps 43-49 (~15%)

    Only labeled nodes (y >= 0) are included in masks.

    Args:
        data: PyG Data object (must have data.timestep or be derived from
              the feature matrix where the first feature is the timestep).
        train_range: Inclusive (start, end) timestep range for training.
        val_range: Inclusive (start, end) timestep range for validation.
        test_range: Inclusive (start, end) timestep range for testing.
    """
    labeled = data.y >= 0

    # The Elliptic dataset encodes timestep as the first local feature.
    # In PyG's EllipticBitcoinDataset, timestep info may be in a separate
    # attribute or encoded in features. We extract it from data.
    if hasattr(data, "timestep"):
        ts = data.timestep
    else:
        # Fallback: attempt to infer timestep from the first feature
        # (the original Elliptic dataset has timestep as feature 0)
        ts = data.x[:, 0].long()

    train_ts = (ts >= train_range[0]) & (ts <= train_range[1])
    val_ts = (ts >= val_range[0]) & (ts <= val_range[1])
    test_ts = (ts >= test_range[0]) & (ts <= test_range[1])

    data.train_mask = train_ts & labeled
    data.val_mask = val_ts & labeled
    data.test_mask = test_ts & labeled

    # Print split statistics
    for name, mask in [("Train", data.train_mask), ("Val", data.val_mask), ("Test", data.test_mask)]:
        total = mask.sum().item()
        illicit = (data.y[mask] == 1).sum().item()
        licit = (data.y[mask] == 0).sum().item()
        if illicit > 0 and licit > 0:
            ratio_str = f"ratio 1:{licit/illicit:.1f}"
        elif illicit == 0:
            ratio_str = "no illicit"
        else:
            ratio_str = "no licit"
        print(f"  {name:5s}: {total:6,} nodes ({illicit:,} illicit, {licit:,} licit, {ratio_str})")


def normalize_features(data, fit_mask=None) -> None:
    """
    Normalize node features using StandardScaler.

    Fits the scaler ONLY on the training set to prevent data leakage.

    Args:
        data: PyG Data object with data.x.
        fit_mask: Boolean mask indicating which nodes to use for fitting
                  the scaler. If None, uses data.train_mask.
    """
    if fit_mask is None:
        fit_mask = data.train_mask

    # RobustScaler uses median/IQR instead of mean/std — resistant to outliers
    # common in financial datasets (Elliptic has extreme skew in some features)
    scaler = RobustScaler()
    x_np = data.x.numpy()

    # Fit only on training data
    scaler.fit(x_np[fit_mask.numpy()])

    # Transform ALL nodes (including unknown)
    x_normalized = scaler.transform(x_np)

    # Clip to [-10, 10] after scaling: prevents exploding activations from
    # extreme test-set values never seen during training (observed max ~18 000σ)
    x_normalized = np.clip(x_normalized, -10.0, 10.0)

    data.x = torch.tensor(x_normalized, dtype=torch.float32)

    # Store scaler for potential inverse transform later
    data._scaler = scaler
    clipped_pct = (np.abs(x_normalized) >= 9.99).mean() * 100
    print(f"  Features normalized (RobustScaler + clip[-10,10]): "
          f"mean~{data.x[fit_mask].mean():.4f}, std~{data.x[fit_mask].std():.4f}, "
          f"clipped={clipped_pct:.2f}% of values")


def preprocess(
    data,
    train_range: Tuple[int, int] = (1, 34),
    val_range: Tuple[int, int] = (35, 42),
    test_range: Tuple[int, int] = (43, 49),
    normalize: bool = True,
) -> None:
    """
    Full preprocessing pipeline: time-split + normalization.

    Modifies the data object in-place.

    Args:
        data: PyG Data object from load_elliptic().
        train_range: Timestep range for training split.
        val_range: Timestep range for validation split.
        test_range: Timestep range for test split.
        normalize: Whether to apply feature normalization.
    """
    print("Preprocessing Elliptic dataset...")

    # 1. Create temporal splits
    print("Creating time-based splits:")
    create_time_split_masks(data, train_range, val_range, test_range)

    # 2. Normalize features (fit on train only)
    if normalize:
        print("Normalizing features (fit on train set):")
        normalize_features(data, fit_mask=data.train_mask)

    print("Preprocessing complete.")
