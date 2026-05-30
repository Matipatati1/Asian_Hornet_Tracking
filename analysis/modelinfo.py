import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
METRIC_ALIASES = {
    "map50-95": ["metrics/mAP50-95(B)", "metrics/mAP_0.5:0.95", "mAP_0.5:0.95"],
    "map50":    ["metrics/mAP50(B)",    "metrics/mAP_0.5",       "mAP_0.5"],
    "precision":["metrics/precision(B)","metrics/P",             "P"],
    "recall":   ["metrics/recall(B)",   "metrics/R",             "R"],
    "fitness":  ["fitness"],
}

ARGS_KEYS_OF_INTEREST = [
    "model", "data", "epochs", "imgsz", "batch",
    "optimizer", "lr0", "weight_decay", "augment",
    "project", "name",
]


def find_column(headers: List[str], candidates: List[str]) -> Optional[str]:
    clean = [h.strip() for h in headers]
    for c in candidates:
        if c.strip() in clean:
            return c.strip()
    return None


def parse_results_csv(csv_path: Path) -> Dict[str, str]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    # Strip whitespace from keys and values
    return {k.strip(): v.strip() for k, v in rows[-1].items()}


def parse_args_yaml(yaml_path: Path) -> Dict[str, str]:
    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}
    return {k: data.get(k, "—") for k in ARGS_KEYS_OF_INTEREST}


def discover_runs(root: Path) -> List[Path]:
    runs = []
    for dirpath, dirnames, filenames in os.walk(root):
        p = Path(dirpath)
        if "results.csv" in filenames and "args.yaml" in filenames:
            runs.append(p)
    return sorted(runs)


def safe_float(value: str) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Rank YOLO training runs by validation performance."
    )
    parser.add_argument(
        "--root", default=".", help="Root folder containing training run sub-folders."
    )
    parser.add_argument(
        "--metric", default="map50-95",
        choices=list(METRIC_ALIASES.keys()),
        help="Metric to rank by (default: map50-95)."
    )
    parser.add_argument(
        "--output", default="model_ranking.csv",
        help="Output CSV path (default: model_ranking.csv)."
    )
    parser.add_argument(
        "--top", type=int, default=0,
        help="Print only the top-N models to terminal (0 = all)."
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"[ERROR] Root folder not found: {root}", file=sys.stderr)
        sys.exit(1)

    print(f"🔍  Scanning: {root}")
    runs = discover_runs(root)
    if not runs:
        print("[ERROR] No completed training runs found (no folder has both results.csv and args.yaml).")
        sys.exit(1)
    print(f"✅  Found {len(runs)} run(s)\n")

    records = []
    for run_path in runs:
        result_row = parse_results_csv(run_path / "results.csv")
        config     = parse_args_yaml(run_path / "args.yaml")

        # Resolve the target metric column
        candidates = METRIC_ALIASES[args.metric]
        col = find_column(list(result_row.keys()), candidates)
        metric_value = safe_float(result_row.get(col, "nan")) if col else float("nan")

        # Compute fitness if not present but mAP50 and mAP50-95 are available
        if args.metric == "fitness" and col is None:
            c50    = find_column(list(result_row.keys()), METRIC_ALIASES["map50"])
            c5095  = find_column(list(result_row.keys()), METRIC_ALIASES["map50-95"])
            if c50 and c5095:
                metric_value = (
                    0.1 * safe_float(result_row.get(c50, "nan"))
                    + 0.9 * safe_float(result_row.get(c5095, "nan"))
                )

        # Collect all metrics for the report
        all_metrics = {}
        for name, cols in METRIC_ALIASES.items():
            if name == "fitness":
                continue
            c = find_column(list(result_row.keys()), cols)
            all_metrics[name] = safe_float(result_row.get(c, "nan")) if c else float("nan")

        # Add computed fitness
        if all_metrics.get("map50") and all_metrics.get("map50-95"):
            all_metrics["fitness"] = round(
                0.1 * all_metrics["map50"] + 0.9 * all_metrics["map50-95"], 6
            )
        else:
            all_metrics["fitness"] = float("nan")

        # Last epoch / total epochs
        epoch_col = find_column(list(result_row.keys()), ["epoch", "Epoch"])
        last_epoch = result_row.get(epoch_col, "?") if epoch_col else "?"

        record = {
            "run_path":   str(run_path.relative_to(root)),
            "run_name":   run_path.name,
            # Config
            "model":      config.get("model", "—"),
            "data":       config.get("data", "—"),
            "epochs_cfg": config.get("epochs", "—"),
            "last_epoch": last_epoch,
            "imgsz":      config.get("imgsz", "—"),
            "batch":      config.get("batch", "—"),
            "optimizer":  config.get("optimizer", "—"),
            "lr0":        config.get("lr0", "—"),
            "weight_decay": config.get("weight_decay", "—"),
            # Metrics
            **{f"metric_{k}": v for k, v in all_metrics.items()},
            # Sort key
            "_sort_value": metric_value,
        }
        records.append(record)

    # Sort descending (higher = better)
    import math
    records.sort(key=lambda r: r["_sort_value"] if not math.isnan(r["_sort_value"]) else -1, reverse=True)

    # Assign rank
    for i, r in enumerate(records, 1):
        r["rank"] = i

    # -----------------------------------------------------------------------
    # Terminal output
    # -----------------------------------------------------------------------
    display = records if args.top == 0 else records[: args.top]
    col_w = 28

    header_line = (
        f"{'Rank':<5} {'Run name':<25} {'Model':<30} {'Data':<20} "
        f"{'Epochs':>7} {'ImgSz':>6} {'Batch':>5} "
        f"{'mAP50':>8} {'mAP50-95':>10} {'Precision':>10} {'Recall':>8} {'Fitness':>9}"
    )
    print("=" * len(header_line))
    print(f"  Ranking by: {args.metric.upper()}  |  Root: {root}")
    print("=" * len(header_line))
    print(header_line)
    print("-" * len(header_line))

    for r in display:
        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) and not math.isnan(v) else "  N/A"

        print(
            f"{r['rank']:<5} "
            f"{str(r['run_name'])[:24]:<25} "
            f"{str(r['model'])[:29]:<30} "
            f"{str(r['data'])[:19]:<20} "
            f"{str(r['last_epoch']):>7} "
            f"{str(r['imgsz']):>6} "
            f"{str(r['batch']):>5} "
            f"{fmt(r['metric_map50']):>8} "
            f"{fmt(r['metric_map50-95']):>10} "
            f"{fmt(r['metric_precision']):>10} "
            f"{fmt(r['metric_recall']):>8} "
            f"{fmt(r['metric_fitness']):>9}"
        )

    print("=" * len(header_line))
    print(f"\n🏆  Best run:  {records[0]['run_name']}  ({args.metric} = {records[0]['_sort_value']:.4f})")
    print(f"   Model:     {records[0]['model']}")
    print(f"   Data:      {records[0]['data']}")
    print(f"   Epochs:    {records[0]['last_epoch']} / {records[0]['epochs_cfg']}")

    # -----------------------------------------------------------------------
    # CSV output
    # -----------------------------------------------------------------------
    output_path = Path(args.output)
    fieldnames = [
        "rank", "run_name", "run_path",
        "model", "data", "epochs_cfg", "last_epoch", "imgsz", "batch",
        "optimizer", "lr0", "weight_decay",
        "metric_map50", "metric_map50-95", "metric_precision", "metric_recall", "metric_fitness",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    print(f"\n📄  Full report saved to: {output_path.resolve()}\n")


if __name__ == "__main__":
    main()