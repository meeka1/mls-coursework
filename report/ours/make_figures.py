"""Figures for the report, from the committed result files. Run from the repository root:
    python report/ours/make_figures.py
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False})

# Figure 1: Recall@10 against the migrated fraction (dev, mean and std over 5 seeds).
d = json.loads((ROOT / "rag/results/migration_dev.json").read_text())
style = {"map-docs": ("#1b7837", "o"), "dual-read": ("#2166ac", "s"), "z-calibrated": ("#9970ab", "^"),
         "naive": ("#d6604d", "v"), "no-map": ("#878787", "x")}
BASELINE = 0.475  # all records re-encoded: every strategy is the supplied pipeline
fig, ax = plt.subplots(figsize=(2.75, 2.1))
for name, (color, marker) in style.items():
    fr = [float(f) for f in d["summary"][name]]  # 0 ... 1.0 (1.0 = baseline)
    mean = [v["Recall@10"][0] for v in d["summary"][name].values()]
    std = [v["Recall@10"][1] for v in d["summary"][name].values()]
    ax.errorbar([100 * f for f in fr], mean, yerr=std, color=color, marker=marker, ms=3, lw=1, capsize=1.5, label=name)
ax.axhline(BASELINE, color="black", lw=0.6, ls="--")
ax.text(2, 0.483, "baseline (all enc_a)", fontsize=6.5)
ax.set_xlabel("records re-encoded with enc_a (%)")
ax.set_ylabel("Recall@10 (dev)")
ax.set_ylim(0, 0.52)
ax.set_xticks([0, 10, 25, 50, 75, 90, 100])
ax.legend(fontsize=6.5, loc="lower right", handlelength=1.5)
fig.tight_layout(pad=0.2)
fig.savefig(OUT / "migration_recall.pdf")

# Figure 2: GPU time per operation on test_audio.wav, before and after trim + KV cache.
def profile(name):
    ops = {}
    for line in (ROOT / "results/asr" / name).read_text().splitlines():
        m = re.match(r"\s+(\w+)\s+(cuda|torch)\s+\d+\s+([\d.]+)\s+[\d.]+%", line)
        if m:
            ops[m.group(1)] = float(m.group(3))
    return ops

runs = [("ours\nbefore", "profile_cuda.txt"), ("ours\nafter", "profile_cuda_final.txt"),
        ("PyTorch\nbefore", "profile_torch.txt"), ("PyTorch\nafter", "profile_torch_final.txt")]
groups = [("linear", "#4d4d4d"), ("attention", "#2166ac"), ("rope", "#d6604d"), ("other", "#bababa")]
fig, ax = plt.subplots(figsize=(2.75, 2.1))
for i, (label, f) in enumerate(runs):
    ops = profile(f)
    vals = {"linear": ops["linear"], "attention": ops["attention"], "rope": ops["rope"]}
    vals["other"] = sum(ops.values()) - sum(vals.values())
    bottom = 0
    for g, color in groups:
        ax.bar(i, vals[g] / 1000, bottom=bottom, color=color, width=0.6, label=g if i == 0 else None)
        bottom += vals[g] / 1000
    ax.text(i, bottom + 0.05, f"{bottom:.2f}", ha="center", fontsize=6.5)
ax.set_xticks(range(len(runs)), [r[0] for r in runs], fontsize=6.5)
ax.set_ylabel("GPU time per transcription (s)")
ax.legend(fontsize=6.5, loc="upper right")
fig.tight_layout(pad=0.2)
fig.savefig(OUT / "asr_profile.pdf")
print("written", sorted(p.name for p in OUT.iterdir()))
