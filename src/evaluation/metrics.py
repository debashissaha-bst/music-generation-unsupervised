
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    import config
    from evaluation.pitch_histogram import histogram_similarity, pitch_histogram
    from evaluation.rhythm_score import repetition_ratio, rhythm_diversity
except ModuleNotFoundError:
    from src import config
    from src.evaluation.pitch_histogram import histogram_similarity, pitch_histogram
    from src.evaluation.rhythm_score import repetition_ratio, rhythm_diversity


def summarize_midi(path: Path) -> Dict[str, float]:
    return {
        "rhythm_diversity": rhythm_diversity(path),
        "repetition_ratio": repetition_ratio(path),
    }


def evaluate_directory(
    midi_dir: str | Path,
    reference_midi: Path | None = None,
    save_plots_to: Path | None = None,
) -> List[Dict[str, Any]]:
    midi_dir = Path(midi_dir)
    rows: List[Dict[str, Any]] = []
    midis = sorted(midi_dir.glob("*.mid")) + sorted(midi_dir.glob("*.midi"))
    ref_hist = pitch_histogram(reference_midi) if reference_midi and Path(reference_midi).is_file() else None

    for p in midis:
        row = {"file": str(p.name), **summarize_midi(p)}
        if ref_hist is not None:
            row["pitch_hist_sim"] = histogram_similarity(ref_hist, pitch_histogram(p))
        rows.append(row)

    if save_plots_to:
        save_plots_to = Path(save_plots_to)
        save_plots_to.mkdir(parents=True, exist_ok=True)
        with open(save_plots_to / "metrics_summary.json", "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)

    return rows


def main():
    ap = argparse.ArgumentParser(description="Evaluate a folder of generated MIDIs")
    ap.add_argument("--midi_dir", type=str, default=str(config.OUTPUTS_MIDI))
    ap.add_argument("--reference", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=str(config.OUTPUTS_PLOTS))
    args = ap.parse_args()

    rows = evaluate_directory(
        args.midi_dir,
        reference_midi=Path(args.reference) if args.reference else None,
        save_plots_to=Path(args.out_dir) if args.out_dir else None,
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
