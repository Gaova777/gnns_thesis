"""
Merge experiment results from all machines into a single CSV.

Sources (in priority order — first found wins for a given config key):
  1. results_machineA/   — RTX 4090 (highest quality)
  2. results_machineA_4060/ — fallback A configs run on 4060
  3. results_machineB/   — GCN+SAGE, scenarios 1:1 and 1:10
  4. results_machineC/   — GCN+SAGE, scenarios 1:50 and 1:100

Output:
  results/results_merged.csv

Usage:
    python scripts/merge_results.py
    python scripts/merge_results.py --results-root /path/to/results_*
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


# Key that uniquely identifies a configuration (without timestamp/seed noise)
CONFIG_KEY = ["scenario", "architecture", "balancing", "explainer"]


def find_csv(results_dir: Path) -> Path | None:
    """Return the first CSV found in results_dir, or None."""
    if not results_dir.exists():
        return None
    csvs = sorted(results_dir.glob("*.csv"))
    return csvs[0] if csvs else None


def load_source(results_dir: Path, label: str) -> pd.DataFrame | None:
    """Load a CSV from results_dir, tag with source label."""
    csv_path = find_csv(results_dir)
    if csv_path is None:
        print(f"  [{label}] Not found or empty: {results_dir}")
        return None
    df = pd.read_csv(csv_path)
    df["_source"] = label
    print(f"  [{label}] Loaded {len(df)} rows from {csv_path}")
    return df


def merge(base_dir: Path) -> pd.DataFrame:
    """Load all sources and merge with priority rules."""

    sources = [
        (base_dir / "results_machineA",      "machineA"),
        (base_dir / "results_machineA_4060", "machineA_4060"),
        (base_dir / "results_machineB",      "machineB"),
        (base_dir / "results_machineC",      "machineC"),
    ]

    frames = []
    for path, label in sources:
        df = load_source(path, label)
        if df is not None:
            frames.append(df)

    if not frames:
        print("ERROR: No result CSVs found in any of the expected directories.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)

    # ── Deduplication with priority ───────────────────────────────────────────
    # Priority order: machineA > machineA_4060 > machineB > machineC
    priority = {"machineA": 0, "machineA_4060": 1, "machineB": 2, "machineC": 3}
    combined["_priority"] = combined["_source"].map(priority).fillna(99)

    # For any config key that appears in both machineA and machineA_4060, warn
    if "machineA" in combined["_source"].values and "machineA_4060" in combined["_source"].values:
        a_keys = set(
            combined[combined["_source"] == "machineA"][CONFIG_KEY]
            .apply(tuple, axis=1)
        )
        a4060_keys = set(
            combined[combined["_source"] == "machineA_4060"][CONFIG_KEY]
            .apply(tuple, axis=1)
        )
        overlap = a_keys & a4060_keys
        if overlap:
            print(f"\nWARNING: {len(overlap)} config(s) appear in both machineA and machineA_4060.")
            print("         Keeping machineA (4090) results — machineA_4060 rows discarded.")
            for cfg in sorted(overlap)[:5]:
                print(f"         {cfg}")
            if len(overlap) > 5:
                print(f"         ... and {len(overlap) - 5} more")

    # Sort so highest-priority rows come first, then drop duplicates keeping first
    combined = combined.sort_values("_priority")
    before = len(combined)
    combined = combined.drop_duplicates(subset=CONFIG_KEY, keep="first")
    after = len(combined)
    if before != after:
        print(f"\n  Deduplicated: {before} → {after} rows ({before - after} duplicates removed)")

    combined = combined.drop(columns=["_priority"])
    return combined


def print_summary(df: pd.DataFrame) -> None:
    """Print a human-readable summary of the merged results."""
    print("\n" + "=" * 70)
    print("MERGE SUMMARY")
    print("=" * 70)
    print(f"Total rows:        {len(df)}")
    print(f"Sources present:   {sorted(df['_source'].unique())}")

    print("\nDistribution by scenario:")
    if "scenario" in df.columns:
        print(df.groupby("scenario").size().to_string())

    print("\nDistribution by architecture:")
    if "architecture" in df.columns:
        print(df.groupby("architecture").size().to_string())

    print("\nDistribution by balancing:")
    if "balancing" in df.columns:
        print(df.groupby("balancing").size().to_string())

    print("\nDistribution by explainer:")
    if "explainer" in df.columns:
        print(df.groupby("explainer").size().to_string())

    # OOM retries
    oom_col = next((c for c in df.columns if "oom_retries" in c), None)
    if oom_col:
        oom_nonzero = df[df[oom_col].notna() & (df[oom_col] > 0)]
        if not oom_nonzero.empty:
            print(f"\nConfigs with shap_oom_retries > 0: {len(oom_nonzero)}")
            print(oom_nonzero[CONFIG_KEY + [oom_col]].to_string(index=False))
        else:
            print("\nNo SHAP OOM retries recorded.")

    # Error rows
    err_col = next((c for c in df.columns if c == "stab_error"), None)
    if err_col:
        errors = df[df[err_col].notna()]
        if not errors.empty:
            print(f"\nConfigs with errors: {len(errors)}")
            print(errors[CONFIG_KEY + [err_col]].to_string(index=False))

    # Coverage check: expect 144 unique config keys
    unique_configs = df[CONFIG_KEY].drop_duplicates()
    print(f"\nUnique config keys:  {len(unique_configs)} (expected 144 for full experiment)")
    if len(unique_configs) < 144:
        print(f"  WARNING: {144 - len(unique_configs)} configs are missing from the merge!")
    elif len(unique_configs) == 144:
        print("  All 144 configs present.")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Merge distributed experiment results")
    parser.add_argument(
        "--results-root", type=str, default=".",
        help="Directory that contains results_machineA/, results_machineB/, etc. (default: .)"
    )
    parser.add_argument(
        "--output", type=str, default="results/results_merged.csv",
        help="Output CSV path (default: results/results_merged.csv)"
    )
    args = parser.parse_args()

    base_dir = Path(args.results_root)
    output_path = Path(args.output)

    print(f"Merging results from: {base_dir.resolve()}")
    print(f"Output:               {output_path}")
    print()

    merged = merge(base_dir)
    print_summary(merged)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Drop internal _source column before saving if user doesn't need it
    save_df = merged.copy()
    output_path_str = str(output_path)
    save_df.to_csv(output_path_str, index=False)
    print(f"\nSaved {len(save_df)} rows to {output_path}")


if __name__ == "__main__":
    main()
