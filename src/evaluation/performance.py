import json
import csv
import re
from pathlib import Path
import matplotlib.pyplot as plt
from mido import MidiFile

ROOT = Path(__file__).resolve().parents[2]
plots_dir = ROOT / "outputs/plots"
survey_file = ROOT / "outputs/survey_results/listening_survey.csv"

metrics_file = plots_dir / "metrics_summary.json"
ppl_file = plots_dir / "transformer_perplexity_report.json"

metrics = json.loads(metrics_file.read_text())
ppl_report = json.loads(ppl_file.read_text())

groups = {
    "Task 1: Autoencoder": [],
    "Task 2: VAE Multi-Genre": [],
    "Task 3: Transformer": [],
    "Task 4: RLHF-Tuned Model": []
}

for row in metrics:
    fname = row["file"].lower()
    if fname.startswith("ae_"):
        groups["Task 1: Autoencoder"].append(row)
    elif fname.startswith("vae_"):
        groups["Task 2: VAE Multi-Genre"].append(row)
    elif fname.startswith("transformer_"):
        groups["Task 3: Transformer"].append(row)
    elif fname.startswith("rlhf_"):
        groups["Task 4: RLHF-Tuned Model"].append(row)

def mean_value(rows, key):
    if not rows:
        return "--"
    return f"{sum(r[key] for r in rows) / len(rows):.4f}"

def clean_label(label):
    return re.sub(r"^Task\s+\d+:\s*", "", label)

# Use actual numeric values for baselines
rand_div = 0.2946428571428571
markov_div = 0.432515192357712

human_scores = {}
if survey_file.exists():
    with open(survey_file, newline="") as f:
        reader = csv.DictReader(f)
        temp = {}
        for row in reader:
            model_id = row["id"].lower()
            score = float(row["score"])
            if model_id.startswith("ae"):
                temp.setdefault("Task 1: Autoencoder", []).append(score)
            elif model_id.startswith("vae"):
                temp.setdefault("Task 2: VAE Multi-Genre", []).append(score)
            elif model_id.startswith("transformer"):
                temp.setdefault("Task 3: Transformer", []).append(score)
        for model, scores in temp.items():
            human_scores[model] = f"{sum(scores) / len(scores):.1f}"

rows = [
    ["Random Generator", "--", f"{rand_div:.4f}", "--", "None"],
    ["Markov Chain", "--", f"{markov_div:.4f}", "--", "Weak"],
    ["Task 1: Autoencoder", "--", mean_value(groups["Task 1: Autoencoder"], "rhythm_diversity"), human_scores.get("Task 1: Autoencoder", "--"), "Single Genre"],
    ["Task 2: VAE Multi-Genre", "--", mean_value(groups["Task 2: VAE Multi-Genre"], "rhythm_diversity"), human_scores.get("Task 2: VAE Multi-Genre", "--"), "Moderate"],
    ["Task 3: Transformer", f"{ppl_report['perplexity']:.2f}", mean_value(groups["Task 3: Transformer"], "rhythm_diversity"), human_scores.get("Task 3: Transformer", "--"), "Strong"],
    ["Task 4: RLHF-Tuned Model", "--", mean_value(groups["Task 4: RLHF-Tuned Model"], "rhythm_diversity"), "Not collected", "Strongest"]
]

latex = r"""\begin{table}[!htb]
\caption{Performance Comparison of Baseline and Task-Level Models}
\label{tab:performance_comparison}
\centering
\begin{tabular}{lcccc}
\hline
\textbf{Model} & \textbf{Perplexity} & \textbf{Rhythm Diversity} & \textbf{Human Score} & \textbf{Genre Control} \\
\hline
"""

for r in rows:
    latex += " & ".join(r) + r" \\" + "\n"

latex += r"""\hline
\end{tabular}
\end{table}
"""

out_file = plots_dir / "performance_comparison_table.tex"
out_file.write_text(latex)

model_labels = [r[0] for r in rows]
rhythm_vals = []

for r in rows:
    try:
        rhythm_vals.append(float(r[2]))
    except:
        mapping = {"Low": 0.1, "Medium": 0.3, "High": 0.5, "Very High": 0.7}
        rhythm_vals.append(mapping.get(r[2], 0.0))

hs_labels = []
hs_vals = []

for r in rows:
    try:
        score = float(r[3])
        hs_labels.append(clean_label(r[0]))
        hs_vals.append(score)
    except:
        pass

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(hs_labels, hs_vals, color="#4C72B0")
ax.set_xlabel("Models")
ax.set_ylabel("Human Score")
ax.set_title("Human Score Comparison Across Models")
ax.set_ylim(0, 5)
plt.xticks(rotation=0, ha="center")
plt.tight_layout()
plt.savefig(plots_dir / "human_score_comparison.png")
plt.show()

rd_labels = [clean_label(lbl) for lbl in model_labels]
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(rd_labels, rhythm_vals, color="#55A868")
ax.set_xlabel("Models")
ax.set_ylabel("Rhythm Diversity")
ax.set_title("Rhythm Diversity Comparison Across Models")
ax.set_ylim(0, 1)
plt.xticks(rotation=0, ha="center")
plt.tight_layout()
plt.savefig(plots_dir / "rhythm_diversity_comparison.png")
plt.show()