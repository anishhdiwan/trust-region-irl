"""Export wandb run summary metrics to a CSV, keyed by "(exp_id)(seed)".

Each wandb run in this project is one training seed: the run's wandb id
(e.g. "dl9zouzo") is the experiment id, and the run's name (e.g.
"1789307885") is the seed used for that run. This writes one row per run
with an "id" column formatted as "<exp_id><seed>" (e.g.
"dl9zouzo1789307885"), followed by all of that run's summary metrics.

Run with the same environment used for training (conda env "trirl"), from
the experiments/ directory:

    conda activate trirl
    cd experiments
    python export_wandb_csv.py s-mohddheia-tu-darmstadt/ICRA_TRIRL_10_2F
    python export_wandb_csv.py s-mohddheia-tu-darmstadt/ICRA_TRIRL_10_2F --out summary.csv
"""
import argparse
import csv
import sys

import wandb


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project", type=str, help="wandb project path, e.g. <entity>/<project>")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path (default: <project_name>_summary.csv)")
    args = parser.parse_args()

    api = wandb.Api()
    runs = api.runs(args.project)

    rows = []
    metric_keys = []
    for run in runs:
        summary = {k: v for k, v in dict(run.summary).items() if not k.startswith("_") and not k.startswith("wandb")}
        row = {"id": f"{run.id}{run.name}", "group": run.group, **summary}
        rows.append(row)
        for k in summary:
            if k not in metric_keys:
                metric_keys.append(k)

    if not rows:
        sys.exit(f"No runs found under {args.project}")

    fieldnames = ["id", "group"] + metric_keys
    out_path = args.out or f"{args.project.split('/')[-1]}_summary.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} runs to {out_path}")


if __name__ == "__main__":
    main()
