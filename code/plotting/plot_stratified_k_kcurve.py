"""Plot the Stratified-K cross-architecture K-curve comparison.

Shows that the K=50 catastrophe is a pool-composition property of Omni-Judge's
chat-template-injection-dominant Llama pool, not a procedure property: the
same Stratified-K sweep on Qwen2.5-RLVR is essentially flat at 0.5-1.0% FPR,
while Omni-Judge ramps monotonically from 1.25% to 99.92% on FPR. (Joint
defense quality on Omni is U-shaped because TPR collapses to 83.5% at K=5;
the figure plots FPR only, on which the curve is a single-direction ramp.)

Data sources:
  - Omni-Judge: Table 3 in the body (sections/mitigations.tex)
  - Qwen2.5-RLVR: Appendix tab:kcurve_qwen (sections/appendix.tex)
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

K_VALUES = [5, 10, 20, 50]
OMNI_FPR = [1.25, 3.99, 12.50, 99.92]    # Llama-base, chat-injection-dominant pool
QWEN_FPR = [1.00, 0.50, 1.00, 0.50]      # Qwen-base, mechanism-balanced pool

OMNI_COLOR = '#d62728'
QWEN_COLOR = '#2ca02c'


def main():
    fig, ax = plt.subplots(1, 1, figsize=(4.2, 2.6))
    ax.plot(K_VALUES, OMNI_FPR, marker='o', color=OMNI_COLOR, linewidth=2.0, markersize=7,
            label='Omni-Judge (Llama-base, imbalanced pool)')
    ax.plot(K_VALUES, QWEN_FPR, marker='s', color=QWEN_COLOR, linewidth=2.0, markersize=7,
            label='Qwen2.5-RLVR (Qwen-base, balanced pool)')

    ax.set_yscale('symlog', linthresh=1.0)
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_xlabel('Stratified-K (tokens per training pool)')
    ax.set_ylabel('Mean held-out FPR (%)')
    ax.set_ylim(0, 300)
    ax.grid(True, alpha=0.3, which='both')

    # Annotate K=50 spike (place ANNOTATION at bottom-right so it doesn't collide with the top-left legend)
    ax.annotate('K=50 catastrophe\n(99.92%)',
                xy=(50, 99.92), xytext=(38, 6),
                fontsize=8, color=OMNI_COLOR,
                ha='center',
                arrowprops=dict(arrowstyle='->', color=OMNI_COLOR, lw=0.8))
    ax.annotate('flat 0.5–1%',
                xy=(20, 1.0), xytext=(15, 0.15),
                fontsize=8, color=QWEN_COLOR,
                ha='center',
                arrowprops=dict(arrowstyle='->', color=QWEN_COLOR, lw=0.8))

    ax.legend(loc='upper left', fontsize=7.5, frameon=True, framealpha=0.92)
    plt.tight_layout()
    pdf = OUT_DIR / 'stratified_k_crossjudge.pdf'
    png = OUT_DIR / 'stratified_k_crossjudge.png'
    plt.savefig(pdf, bbox_inches='tight', dpi=200)
    plt.savefig(png, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'saved {pdf}')
    print(f'saved {png}')


if __name__ == '__main__':
    main()
