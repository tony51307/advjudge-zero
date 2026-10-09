"""Plot GRPO training curves for the n=10 vs n=10 experiment, both datasets.

For each of {MATH, GSM8K}, generate a 2-panel figure:
  (a) Cumulative false positives (per seed, with mean overlay)
  (b) Mean response length (per seed, with collapse threshold)

The cumulative-FP panel is the load-bearing visual: shows that 5/10 (MATH)
or 3/10 (GSM8K) baseline seeds escape into runaway FP, while 0/10 hardened
seeds do.

Reads from results/grpo_metrics/{baseline,hardened}_seed{0..9}/grpo_metrics_{MATH,GSM8K}.json,
all committed to the repo (seeds 0-4 originally from local results/, seeds 5-9
fetched back from GCP via gcloud scp from tuli-a100s + tuli-a100-80g).
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 9

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, '..', '..'))   # advjudge-zero-paper/
METRICS = f"{PAPER}/results/grpo_metrics"
FIG_DIR = f"{PAPER}/paper/figures"
MAX_STEP = 500


def metric_path(condition, seed, dataset):
    return f'{METRICS}/{condition}_seed{seed}/grpo_metrics_{dataset}.json'


def load_seeds(condition, dataset):
    """Returns dict with cum_fp (n_seeds, MAX_STEP) and len (n_seeds, MAX_STEP)."""
    cum_fps, lens = [], []
    for s in range(10):
        path = metric_path(condition, s, dataset)
        if not os.path.exists(path):
            print(f"  missing: {path}")
            continue
        m = json.load(open(path))
        if 'metrics' in m: m = m['metrics']
        m = [s for s in m if s['step'] < MAX_STEP]
        if len(m) < 50:
            print(f"  too short: {path}")
            continue
        fps = np.array([s['fp_count'] for s in m])
        lns = np.array([s['avg_response_length'] for s in m])
        # Pad to MAX_STEP
        if len(fps) < MAX_STEP:
            fps = np.concatenate([fps, np.full(MAX_STEP - len(fps), 0)])
            lns = np.concatenate([lns, np.full(MAX_STEP - len(lns), lns[-1])])
        cum_fps.append(np.cumsum(fps[:MAX_STEP]))
        lens.append(lns[:MAX_STEP])
    return np.array(cum_fps), np.array(lens)


def smooth(arr, window=15):
    """Centered moving-average smoothing with edge-reflection padding.
    Avoids the boundary drop-to-zero artifact that np.convolve(mode='same')
    introduces at the start/end of the series.
    """
    if arr.ndim == 1:
        pad = window // 2
        padded = np.pad(arr, pad, mode='edge')
        kernel = np.ones(window) / window
        out = np.convolve(padded, kernel, mode='valid')
        return out[:len(arr)]
    return np.array([smooth(row, window) for row in arr])


def plot_dataset(dataset, out_pdf, out_png):
    print(f"\n=== {dataset} ===")
    base_fp, base_len = load_seeds('baseline', dataset)
    hard_fp, hard_len = load_seeds('hardened', dataset)
    print(f"  baseline: {len(base_fp)} seeds | hardened: {len(hard_fp)} seeds")

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))
    x = np.arange(MAX_STEP)

    def line(ax, arr, color, label):
        """Plot mean line only (no band)."""
        mean = arr.mean(axis=0)
        ax.plot(x, mean, color=color, linewidth=2.2, label=label)

    # Panel (a): cumulative FP — mean only
    ax = axes[0]
    line(ax, base_fp, '#d62728', f'Baseline-30B (n={len(base_fp)})')
    line(ax, hard_fp, '#2ca02c', f'Hardened-30B (n={len(hard_fp)})')
    ax.axhline(200, color='black', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.text(495, 215, 'FP-collapse (≥200)', ha='right', va='bottom', fontsize=7, color='black', alpha=0.8)
    ax.set_xlabel('GRPO step')
    ax.set_ylabel('Cumulative false positives')
    ax.set_title(f'(a) Cumulative FPs — {dataset}', fontsize=10)
    ax.set_yscale('symlog', linthresh=10)
    ax.set_ylim(0, max(1500, base_fp.max() * 1.1))
    ax.set_xlim(0, MAX_STEP)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper left', fontsize=8, frameon=True, framealpha=0.9)

    # Panel (b): response length — smoothed mean only
    ax = axes[1]
    sm_base = smooth(base_len, 25)
    sm_hard = smooth(hard_len, 25)
    line(ax, sm_base, '#d62728', f'Baseline-30B (n={len(base_fp)})')
    line(ax, sm_hard, '#2ca02c', f'Hardened-30B (n={len(hard_fp)})')
    ax.axhline(30, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
    ax.text(495, 33, 'length-collapse (<30w)', ha='right', va='bottom', fontsize=7, color='gray')
    ax.set_xlabel('GRPO step')
    ax.set_ylabel('Mean response length (words)')
    ax.set_title(f'(b) Response length — {dataset}', fontsize=10)
    ax.set_ylim(0, 250)
    ax.set_xlim(0, MAX_STEP)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8, frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches='tight', dpi=200)
    plt.savefig(out_png, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"  saved {out_pdf}")
    print(f"  saved {out_png}")


def main():
    plot_dataset('MATH',
                 out_pdf=f'{FIG_DIR}/grpo_n10_MATH.pdf',
                 out_png=f'{FIG_DIR}/grpo_n10_MATH.png')
    plot_dataset('GSM8K',
                 out_pdf=f'{FIG_DIR}/grpo_n10_GSM8K.pdf',
                 out_png=f'{FIG_DIR}/grpo_n10_GSM8K.png')


if __name__ == '__main__':
    main()
