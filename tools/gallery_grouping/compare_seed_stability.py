#!/usr/bin/env python3
"""Compare multiple Gallery runs without relying on unstable person_xxx IDs.

The stable face key is (photo, face_index). Cluster labels are compared with
label-invariant partition metrics (ARI/Rand score), so UUIDs/person folder
numbers may differ freely between runs.
"""
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, Tuple

from sklearn.metrics import adjusted_rand_score, rand_score


def parse_args():
    parser = argparse.ArgumentParser(description="Compare arrival-order stability across Gallery runs.")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run directories containing summary.json and face_assignments.csv.",
    )
    parser.add_argument(
        "--output",
        default="data/gallery_seed_stability",
        help="Output directory for comparison JSON/CSV.",
    )
    return parser.parse_args()


def load_run(path: Path) -> dict:
    summary_path = path / "summary.json"
    faces_path = path / "face_assignments.csv"
    clusters_path = path / "clusters.csv"
    if not summary_path.exists() or not faces_path.exists():
        raise FileNotFoundError(f"Run must contain summary.json and face_assignments.csv: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    faces: Dict[Tuple[str, str], dict] = {}
    with faces_path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = (row["photo"], row["face_index"])
            faces[key] = row
    singleton_clusters = None
    non_singleton_clusters = None
    if clusters_path.exists():
        with clusters_path.open("r", newline="", encoding="utf-8-sig") as f:
            cluster_rows = list(csv.DictReader(f))
        singleton_clusters = sum(int(row.get("cluster_face_count", 0)) == 1 for row in cluster_rows)
        non_singleton_clusters = sum(int(row.get("cluster_face_count", 0)) >= 2 for row in cluster_rows)

    auto_merges = sum(int(r.get("auto_merges", 0)) for r in summary.get("consolidation_runs", []))
    auto_splits = sum(int(r.get("auto_splits", 0)) for r in summary.get("consolidation_runs", []))
    return {
        "path": str(path),
        "summary": summary,
        "faces": faces,
        "seed": summary.get("seed"),
        "auto_merges": auto_merges,
        "auto_splits": auto_splits,
        "singleton_clusters": singleton_clusters,
        "non_singleton_clusters": non_singleton_clusters,
    }


def pairwise_metrics(left: dict, right: dict) -> dict:
    common = sorted(set(left["faces"]) & set(right["faces"]))
    same_state = sum(
        left["faces"][key]["assignment_state"] == right["faces"][key]["assignment_state"]
        for key in common
    )
    confirmed = [
        key for key in common
        if left["faces"][key]["assignment_state"] in ("confirmed", "manual")
        and right["faces"][key]["assignment_state"] in ("confirmed", "manual")
        and left["faces"][key]["cluster_id"]
        and right["faces"][key]["cluster_id"]
    ]
    left_labels = [left["faces"][key]["cluster_id"] for key in confirmed]
    right_labels = [right["faces"][key]["cluster_id"] for key in confirmed]
    ari = adjusted_rand_score(left_labels, right_labels) if confirmed else None
    rand = rand_score(left_labels, right_labels) if confirmed else None
    return {
        "seed_a": left["seed"],
        "seed_b": right["seed"],
        "common_faces": len(common),
        "assignment_state_agreement": same_state / len(common) if common else None,
        "common_confirmed_faces": len(confirmed),
        "adjusted_rand_index": ari,
        "rand_index": rand,
    }


def main() -> int:
    args = parse_args()
    runs = [load_run(Path(p).expanduser().resolve()) for p in args.runs]
    if len(runs) < 2:
        raise ValueError("Provide at least two runs")

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    run_rows = []
    for run in runs:
        s = run["summary"]
        states = s.get("assignment_states", {})
        run_rows.append({
            "seed": run["seed"],
            "scheduled_photos": s.get("scheduled_photos"),
            "accepted_faces": s.get("accepted_faces"),
            "active_clusters": s.get("active_clusters"),
            "singleton_clusters": run.get("singleton_clusters"),
            "non_singleton_clusters": run.get("non_singleton_clusters"),
            "confirmed": states.get("confirmed", 0),
            "ambiguous": states.get("ambiguous", 0),
            "unassigned": states.get("unassigned", 0),
            "manual": states.get("manual", 0),
            "final_suspicious_faces": s.get("final_suspicious_faces"),
            "ungrouped_photos": s.get("ungrouped_photos"),
            "auto_merges": run["auto_merges"],
            "auto_splits": run["auto_splits"],
            "path": run["path"],
        })

    pair_rows = [pairwise_metrics(a, b) for a, b in combinations(runs, 2)]
    aris = [row["adjusted_rand_index"] for row in pair_rows if row["adjusted_rand_index"] is not None]
    state_agreements = [row["assignment_state_agreement"] for row in pair_rows if row["assignment_state_agreement"] is not None]

    payload = {
        "test": "gallery_multi_seed_stability_v2",
        "run_count": len(runs),
        "seeds": [run["seed"] for run in runs],
        "runs": run_rows,
        "pairwise": pair_rows,
        "aggregate": {
            "cluster_count_min": min(int(row["active_clusters"]) for row in run_rows),
            "cluster_count_max": max(int(row["active_clusters"]) for row in run_rows),
            "cluster_count_range": max(int(row["active_clusters"]) for row in run_rows) - min(int(row["active_clusters"]) for row in run_rows),
            "singleton_cluster_count_min": min(int(row["singleton_clusters"]) for row in run_rows if row["singleton_clusters"] is not None) if any(row["singleton_clusters"] is not None for row in run_rows) else None,
            "singleton_cluster_count_max": max(int(row["singleton_clusters"]) for row in run_rows if row["singleton_clusters"] is not None) if any(row["singleton_clusters"] is not None for row in run_rows) else None,
            "non_singleton_cluster_count_min": min(int(row["non_singleton_clusters"]) for row in run_rows if row["non_singleton_clusters"] is not None) if any(row["non_singleton_clusters"] is not None for row in run_rows) else None,
            "non_singleton_cluster_count_max": max(int(row["non_singleton_clusters"]) for row in run_rows if row["non_singleton_clusters"] is not None) if any(row["non_singleton_clusters"] is not None for row in run_rows) else None,
            "mean_adjusted_rand_index": sum(aris) / len(aris) if aris else None,
            "min_adjusted_rand_index": min(aris) if aris else None,
            "mean_assignment_state_agreement": sum(state_agreements) / len(state_agreements) if state_agreements else None,
        },
        "notes": [
            "Faces are matched across runs by (photo, face_index), not UUID.",
            "ARI/Rand are invariant to cluster label renaming, so person_xxx/cluster UUID changes do not matter.",
            "Use the same exact Gallery file set for every compared seed.",
            "Singleton clusters are reported separately because a one-photo person is valid gallery evidence and is intentionally protected from auto-merge.",
        ],
    }

    (output / "stability_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    def write_csv(name, rows):
        if not rows:
            return
        with (output / name).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    write_csv("runs.csv", run_rows)
    write_csv("pairwise_stability.csv", pair_rows)

    print("=== MULTI-SEED STABILITY ===")
    print(f"Seeds:              {payload['seeds']}")
    print(f"Cluster range:      {payload['aggregate']['cluster_count_min']}..{payload['aggregate']['cluster_count_max']} (spread={payload['aggregate']['cluster_count_range']})")
    if payload['aggregate']['non_singleton_cluster_count_min'] is not None:
        print(f"Non-singletons:     {payload['aggregate']['non_singleton_cluster_count_min']}..{payload['aggregate']['non_singleton_cluster_count_max']}")
        print(f"Singletons:         {payload['aggregate']['singleton_cluster_count_min']}..{payload['aggregate']['singleton_cluster_count_max']}")
    if payload["aggregate"]["mean_adjusted_rand_index"] is not None:
        print(f"Mean ARI:           {payload['aggregate']['mean_adjusted_rand_index']:.6f}")
        print(f"Minimum ARI:        {payload['aggregate']['min_adjusted_rand_index']:.6f}")
    if payload["aggregate"]["mean_assignment_state_agreement"] is not None:
        print(f"Mean state agree:   {payload['aggregate']['mean_assignment_state_agreement']:.6f}")
    print(f"Summary:            {output / 'stability_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
