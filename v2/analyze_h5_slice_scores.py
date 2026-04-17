"""Analyze the H5b single-agent test output and produce a slice-weights file.

The H5b notebook writes one JSON per (parent_id, run) into
``v2/h5_test_outputs/``.  Each file is a list of dicts with ``parent_id``,
``clear_phase``, ``case_index``, ``actual``, ``expected``, ``overall_behavior_score``
(and all the sub-scores used by the grader).

This script:
    1. Loads the latest H5 output file per parent_id (or a specific run id if
       ``--run-ts`` is given).
    2. Computes mean ``overall_behavior_score``, ``phase_alignment``,
       ``persona_alignment`` and letter-grade distribution per
       ``(parent_id, clear_phase)``.
    3. Computes ``collapsed_across_phases`` per ``(parent_id, case_index)``
       using the exact same heuristic as the H5b notebook's
       ``summarize_phase_variation`` (5 phase responses look too alike to be
       phase-differentiated).
    4. Writes a ``slice_weights.json`` file that the v2 dataset generator can
       read to over-sample the weakest slices in the next regen pass.

Usage:
    python v2/analyze_h5_slice_scores.py
    python v2/analyze_h5_slice_scores.py --run-ts 20260417T141338_763447+0000
    python v2/analyze_h5_slice_scores.py --in-dir v2/h5_test_outputs \
        --out training_data/parent/slice_weights.json

The generated slice_weights.json has the shape::

    {
        "run_ts": "...",
        "per_parent_phase": {
            "anne_palmer|COUNSEL": {
                "mean_overall_behavior_score": 0.52,
                "mean_phase_alignment": 0.30,
                "mean_persona_alignment": 0.55,
                "grade_mix": {"D": 9, "C": 2},
                "collapsed_case_count": 5,
                "weight": 1.82
            },
            ...
        },
        "collapsed_cases": [[parent_id, case_index], ...],
        "recommend_extra_rows": {
            "anne_palmer|COUNSEL": 180,
            ...
        }
    }
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
DEFAULT_IN_DIR = ROOT / "h5_test_outputs"
DEFAULT_OUT = ROOT.parent / "training_data" / "parent" / "slice_weights.json"
CLEAR_PHASES = ("COUNSEL", "LISTEN", "EMPATHIZE", "ANSWER", "RECOMMEND")


# Mirror of H5b behavior_grade thresholds.
def behavior_grade(score: float) -> str:
    if score >= 0.85:
        return "A"
    if score >= 0.70:
        return "B"
    if score >= 0.55:
        return "C"
    if score >= 0.40:
        return "D"
    return "F"


# Same normalization the H5b grader uses for semantic similarity.
def normalize_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text).strip().lower())
    cleaned = re.sub(r"[^a-z0-9\s]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def pairwise_similarity(texts: Sequence[str]) -> float:
    if len(texts) < 2:
        return 1.0
    normalized = [normalize_text(t) for t in texts]
    sims: List[float] = []
    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            sims.append(
                SequenceMatcher(None, normalized[i], normalized[j]).ratio()
            )
    return mean(sims) if sims else 1.0


def load_run(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "results" in data:
        rows = data["results"]
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError(f"unrecognized H5 output format in {path}")
    return [r for r in rows if isinstance(r, dict)]


def pick_latest_per_parent(
    in_dir: Path, run_ts: str | None = None
) -> Dict[str, Path]:
    """Pick one H5 JSON per parent.  If ``run_ts`` is given, filter by it;
    otherwise fall back to the most recently modified file per parent."""
    candidates = sorted(in_dir.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"no .json files in {in_dir}")

    by_parent: Dict[str, Path] = {}
    run_ts_lc = run_ts.lower() if run_ts else None
    for path in candidates:
        stem = path.stem.lower()
        if run_ts_lc and run_ts_lc not in stem:
            continue
        parent = None
        for candidate in ("anne_palmer", "maya_pena"):
            if candidate in stem:
                parent = candidate
                break
        if parent is None:
            # Fallback: peek at the first row.
            rows = load_run(path)
            parent = rows[0].get("parent_id") if rows else None
            if not parent:
                continue
        prior = by_parent.get(parent)
        if prior is None or path.stat().st_mtime > prior.stat().st_mtime:
            by_parent[parent] = path
    if not by_parent:
        raise FileNotFoundError(
            "no matching H5 JSON files found"
            + (f" for run_ts={run_ts}" if run_ts else "")
        )
    return by_parent


def slice_analysis(
    rows: Iterable[Dict[str, Any]],
    collapse_threshold: float = 0.9,
) -> Dict[str, Any]:
    rows = list(rows)
    if not rows:
        return {
            "per_parent_phase": {},
            "collapsed_cases": [],
            "counts": {"total_rows": 0},
        }

    per_slice_scores: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    per_slice_phase: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    per_slice_persona: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    per_slice_grades: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    per_case_responses: Dict[Tuple[str, int], List[str]] = defaultdict(list)

    for row in rows:
        parent = str(row.get("parent_id", "unknown"))
        phase = str(row.get("clear_phase", "unknown"))
        case_index = int(row.get("case_index", 0))
        overall = float(row.get("overall_behavior_score", 0.0) or 0.0)
        per_slice_scores[(parent, phase)].append(overall)
        per_slice_phase[(parent, phase)].append(
            float(row.get("phase_alignment", 0.0) or 0.0)
        )
        per_slice_persona[(parent, phase)].append(
            float(row.get("persona_alignment", 0.0) or 0.0)
        )
        per_slice_grades[(parent, phase)][behavior_grade(overall)] += 1
        per_case_responses[(parent, case_index)].append(str(row.get("actual", "")))

    collapsed_cases: List[Tuple[str, int]] = []
    per_slice_collapse_count: Counter = Counter()
    for (parent, case_index), responses in per_case_responses.items():
        if len(responses) >= 2 and pairwise_similarity(responses) >= collapse_threshold:
            collapsed_cases.append((parent, case_index))
            # Mark all (parent, phase) slices for this case as collapsed.
            for phase in CLEAR_PHASES:
                per_slice_collapse_count[(parent, phase)] += 1

    per_parent_phase: Dict[str, Dict[str, Any]] = {}
    for key in sorted(per_slice_scores.keys()):
        parent, phase = key
        scores = per_slice_scores[key]
        per_parent_phase[f"{parent}|{phase}"] = {
            "rows": len(scores),
            "mean_overall_behavior_score": round(mean(scores), 4),
            "mean_phase_alignment": round(mean(per_slice_phase[key]), 4),
            "mean_persona_alignment": round(mean(per_slice_persona[key]), 4),
            "grade_mix": dict(per_slice_grades[key]),
            "collapsed_case_count": int(per_slice_collapse_count[key]),
        }

    return {
        "per_parent_phase": per_parent_phase,
        "collapsed_cases": [list(c) for c in collapsed_cases],
        "counts": {"total_rows": len(rows)},
    }


def compute_slice_weights(
    per_parent_phase: Dict[str, Dict[str, Any]],
    base_rows_per_slice: int = 135,  # 1350 per persona / 10 phases
    target_total: int = 2700,
    floor_score: float = 0.4,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Convert per-slice mean scores into (relative weight, row count).

    A slice with a lower mean gets a higher weight so the next dataset
    regen pass over-samples it.  The formula is intentionally simple so
    weights stay interpretable by humans reading ``slice_weights.json``::

        raw_weight = 1.0 + 2.0 * max(0, (1.0 - mean_overall))
        weight     = raw_weight / mean(raw_weight across slices)
        row_count  = round(weight * base_rows_per_slice)

    The collapsed_case_count adds an extra penalty so slices that collapse
    across phases also get more samples.
    """
    if not per_parent_phase:
        return {}, {}

    raw: Dict[str, float] = {}
    for key, stats in per_parent_phase.items():
        overall = float(stats.get("mean_overall_behavior_score", 0.0))
        collapsed = int(stats.get("collapsed_case_count", 0))
        rows_in_slice = max(1, int(stats.get("rows", 0)))
        collapse_ratio = collapsed / rows_in_slice
        penalty = 1.0 + 2.0 * max(0.0, floor_score + 0.5 - overall)  # bigger when low
        penalty += 1.5 * collapse_ratio
        raw[key] = penalty
    avg_raw = mean(raw.values()) or 1.0
    weights = {k: round(v / avg_raw, 4) for k, v in raw.items()}

    extra_rows: Dict[str, int] = {
        k: max(0, int(round(w * base_rows_per_slice))) for k, w in weights.items()
    }
    # Rescale if needed to hit a rough target_total.
    if target_total > 0 and sum(extra_rows.values()) > 0:
        scale = target_total / sum(extra_rows.values())
        extra_rows = {
            k: max(1, int(round(v * scale))) for k, v in extra_rows.items()
        }
    return weights, extra_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=str, default=str(DEFAULT_IN_DIR))
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--run-ts", type=str, default=None)
    parser.add_argument(
        "--collapse-threshold",
        type=float,
        default=0.9,
        help="pairwise_similarity >= this is considered 'collapsed across phases'",
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=2700,
        help="aim to recommend this many single-turn rows for the next regen",
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_path = Path(args.out)

    per_parent_files = pick_latest_per_parent(in_dir, run_ts=args.run_ts)
    print("Using H5 output files:")
    for parent, path in per_parent_files.items():
        print(f"  {parent}: {path.name}")

    all_rows: List[Dict[str, Any]] = []
    for path in per_parent_files.values():
        all_rows.extend(load_run(path))

    analysis = slice_analysis(all_rows, collapse_threshold=args.collapse_threshold)
    weights, extra_rows = compute_slice_weights(
        analysis["per_parent_phase"],
        target_total=args.target_total,
    )

    run_ts_label = args.run_ts or next(iter(per_parent_files.values())).stem
    payload = {
        "run_ts": run_ts_label,
        "per_parent_phase": analysis["per_parent_phase"],
        "collapsed_cases": analysis["collapsed_cases"],
        "counts": analysis["counts"],
        "slice_weights": weights,
        "recommend_extra_rows": extra_rows,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nPer-(parent, phase) slice stats (sorted by mean overall):")
    slices = sorted(
        analysis["per_parent_phase"].items(),
        key=lambda kv: kv[1]["mean_overall_behavior_score"],
    )
    for key, stats in slices:
        grade = behavior_grade(stats["mean_overall_behavior_score"])
        print(
            f"  {key:28s} "
            f"overall={stats['mean_overall_behavior_score']:.3f} ({grade}) "
            f"phase={stats['mean_phase_alignment']:.3f} "
            f"persona={stats['mean_persona_alignment']:.3f} "
            f"collapsed={stats['collapsed_case_count']}/{stats['rows']}"
        )
    print(
        f"\nCollapsed-across-phases cases: {len(analysis['collapsed_cases'])} "
        f"(target = 0 before shipping a checkpoint)"
    )
    print("\nRecommended extra rows by slice (for the next regen pass):")
    for key in sorted(extra_rows.keys()):
        print(f"  {key:28s} {extra_rows[key]}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
