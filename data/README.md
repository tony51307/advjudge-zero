# Data notes

This directory contains selected derived research artifacts, not a mirror of the underlying benchmarks.

- `n7_analysis/`: discovered candidate strings and related analysis outputs grouped by judge and dataset.
- `token_splits/`: token pools/splits used in selected defense experiments.
- `taxonomy/`: assignments and clustering outputs used in taxonomy analyses.

The result CSVs generally contain evaluation records such as pool labels, prompt indices, and candidate strings; they omit the original benchmark questions and reference answers. The scripts load examples from upstream datasets, including [MATH](https://huggingface.co/datasets/nlile/hendrycks-MATH-benchmark), [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [AIME](https://huggingface.co/datasets/gneubig/aime-1983-2024), and [Multi-subject RLVR](https://huggingface.co/datasets/virtuoussy/Multi-subject-RLVR). Dataset names/configurations can change; consult the scripts and paper for the exact evaluation protocol and split.

The released token strings include control-like text because they are the object of this robustness study. Treat them as untrusted input and do not insert them into prompts or systems outside an isolated, authorized evaluation. Follow upstream dataset terms and model-provider access requirements.
