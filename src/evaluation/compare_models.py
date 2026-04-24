from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    from evaluation.metrics import evaluate_directory
except ModuleNotFoundError:
    from src.evaluation.metrics import evaluate_directory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="Folders containing .mid files")
    ap.add_argument("--reference", type=str, default=None, help="Optional reference MIDI for pitch histogram similarity")
    ap.add_argument("--out", type=str, default="comparison.json")
    args = ap.parse_args()
    summary = {}
    reference = Path(args.reference) if args.reference else None
    for d in args.dirs:
        rows = evaluate_directory(d, reference_midi=reference, save_plots_to=None)
        if not rows:
            summary[Path(d).name] = {}
            continue
        entry = {
            "mean_rhythm_diversity": sum(r["rhythm_diversity"] for r in rows) / len(rows),
            "mean_repetition_ratio": sum(r["repetition_ratio"] for r in rows) / len(rows),
            "n": len(rows),
        }
        if rows and "pitch_hist_sim" in rows[0]:
            entry["mean_pitch_hist_sim"] = sum(r["pitch_hist_sim"] for r in rows) / len(rows)
        summary[Path(d).name] = entry
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
