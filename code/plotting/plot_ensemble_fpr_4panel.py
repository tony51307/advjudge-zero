"""Plot ensemble FPR (AdvJudge-Zero vs.\ Master-RM) across 6 general-purpose
judges, one bar pair per judge, four panels (one per dataset) in a single row.

Data source: Table 1 (paper/tables/table_model_specific_fpr.tex). Numbers are
hard-coded here so the figure is always consistent with the table; if Table 1
changes, update this script.

Outputs:
  paper/figures/ensemble_fpr_4panel.pdf
  paper/figures/ensemble_fpr_4panel.png
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 9

OUT_DIR = Path(__file__).resolve().parents[2] / "paper" / "figures"

# Data from Table 1 — general-purpose judges only.
JUDGES = ['Qwen3-4B', 'Qwen2.5-7B', 'Qwen3-30B', 'Llama-3.2-3B', 'gemma-3-4b', 'Llama-3.3-70B']
JUDGE_SHORT = ['Q3-4B', 'Q2.5-7B', 'Q3-30B', 'L3.2-3B', 'G3-4B', 'L3.3-70B']
ADV = {
    'AIME':      [100.00, 99.25,  100.00,  100.00,  99.25,  93.35],
    'MATH':      [100.00, 99.88,  100.00,  99.80,   99.92,  99.85],
    'GSM8K':     [100.00, 99.96,  100.00,  69.34,   99.92,  99.56],
    'MultiRLVR': [100.00, 79.45,  100.00,  93.28,   100.00, 95.78],
}
MRM = {
    'AIME':      [87.25,  37.41,  81.78,   86.07,   41.59,  32.69],
    'MATH':      [89.25,  55.14,  82.88,   75.18,   60.70,  67.99],
    'GSM8K':     [97.90,  92.96,  98.29,   99.16,   92.20,  98.65],
    'MultiRLVR': [83.33,  13.78,  79.17,   28.22,   76.70,  45.58],
}
DATASETS = ['AIME', 'MATH', 'GSM8K', 'MultiRLVR']

ADV_COLOR = '#2ca02c'   # green for AdvJudge-Zero
MRM_COLOR = '#ff7f0e'   # orange for Master-RM


def main():
    n = len(JUDGES)
    x = np.arange(n)
    bar_w = 0.38

    fig, axes = plt.subplots(1, 4, figsize=(11, 2.4), sharey=True)

    for ax, ds in zip(axes, DATASETS):
        ax.bar(x - bar_w/2, ADV[ds], bar_w, color=ADV_COLOR, label='AdvJudge-Zero')
        ax.bar(x + bar_w/2, MRM[ds], bar_w, color=MRM_COLOR, label='Master-RM (10 tokens)')
        ax.set_xticks(x)
        ax.set_xticklabels(JUDGE_SHORT, rotation=35, ha='right', fontsize=8)
        ax.set_title(ds, fontsize=10)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='y', labelsize=8)
        # Light "90% threshold" line for the 22-of-24 framing
        ax.axhline(90, color='gray', linestyle=':', linewidth=0.7, alpha=0.7)

    axes[0].set_ylabel('Ensemble FPR (%)', fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2,
               bbox_to_anchor=(0.5, 1.05), frameon=False, fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    pdf = OUT_DIR / 'ensemble_fpr_4panel.pdf'
    png = OUT_DIR / 'ensemble_fpr_4panel.png'
    plt.savefig(pdf, bbox_inches='tight', dpi=200)
    plt.savefig(png, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'saved {pdf}')
    print(f'saved {png}')


if __name__ == '__main__':
    main()
