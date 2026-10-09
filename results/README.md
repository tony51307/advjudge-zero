# Results artifact guide

These files are selected derived outputs accompanying the paper. They are not raw benchmark datasets. CSV artifacts contain evaluation outputs and references such as prompt indices/candidate strings, but not the original benchmark question and solution text.

- `eval/`: aggregate baseline and hardened static-evaluation metrics.
- `per_token_flips/`: per-prompt/token evaluation records for baseline and hardened judges across the listed datasets.
- `ablations/`: aggregate JSON summaries for selected defense comparisons; not every intermediate ablation file is included.
- `geometric/`: aggregate geometric-analysis outputs.
- `grpo_metrics/`: per-seed training/evaluation metrics for the included GRPO conditions.

Consult the paper for protocols, denominators, filtering, seed counts, and metric definitions. In particular, replacement and append-only evaluations are different threat models; do not compare their FPRs as if they measured the same condition. The replacement result is a controlled diagnostic, while append-only preserves the original incorrect response.

Benchmark examples should be obtained from their upstream sources and used under their respective terms. See [`../data/README.md`](../data/README.md).
