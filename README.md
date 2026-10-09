# AdvJudge-Zero

Code and selected derived data for **“AdvJudge-Zero: Binary Decision Flips in LLM-as-a-Judge via Adversarial Control Tokens.”** The paper was accepted as a poster at the NeurIPS 2026 Main Track.

This repository includes the main evaluation, defense, discovery, and GRPO scripts; selected derived outputs; and the camera-ready PDF. LaTeX source and build files are intentionally not included.

## Paper

- Camera-ready paper: [`paper/main.pdf`](paper/main.pdf)
- arXiv: [arXiv:2512.17375](https://arxiv.org/abs/2512.17375)

## What is included

```text
code/       Selected discovery, evaluation, defense, GRPO, and plotting scripts
data/       Discovered candidates, defense splits, and taxonomy assignments
results/    Aggregate metrics, per-prompt flip records, and GRPO metrics
paper/      Camera-ready PDF only
```

The release focuses on the artifacts most useful for understanding and checking the paper's central results. It does not include model weights, raw benchmark text, every intermediate file, or the full camera-ready authoring source. Dataset acquisition and model access are handled through the original providers; see [`data/README.md`](data/README.md).

## Main findings

- **Replacement diagnostic:** the discovered ensemble exceeds 90% false-positive rate in 22 of 24 evaluated judge/dataset cells. This is a controlled upper-bound diagnostic, not the realistic deployment estimate.
- **Append-only setting:** reported false-positive rates include 58.1% on Omni-Judge, 27.5% on Llama-3.2-3B, and at most 23% on the evaluated Qwen/Gemma judges. The incorrect response is preserved in this setting.
- **Filtering:** strict candidate filtering reduces Omni-Judge's held-out replacement FPR from 30.4% to 1.4%, while Qwen3-30B remains at 94.5% under the reported filter.
- **Hardening:** mechanism-stratified LoRA reduces Omni-Judge cross-family held-out FPR from 32.3% to 10.0%; on the round-2 adaptive pool, it reaches 12.25% versus 31% for naive hardening.
- **GRPO:** in the reported Qwen3-30B setup, FP-collapse occurs in 5/10 unhardened seeds and 0/10 hardened seeds per task. This supports hardening in that setup; it is not end-to-end evidence isolating the stratification benefit.

See the paper for metric definitions, protocols, caveats, and full comparisons.

## Code map

- Discovery: `code/discovery/`
- Static evaluation: `code/evaluation/`
- Defense data construction and LoRA fine-tuning: `code/defense/`
- GRPO runs and generation analysis: `code/rl/`
- Geometric analyses and figure generation: `code/geometric/`, `code/plotting/`

Some scripts require substantial GPU memory, gated model access, and compatible versions of PyTorch, Transformers, vLLM, PEFT, or TRL. Dataset/model terms remain those of their respective providers. Check each script's arguments and configuration before running.

## Citation

```bibtex
@inproceedings{li2026advjudgezero,
  title     = {AdvJudge-Zero: Binary Decision Flips in LLM-as-a-Judge via Adversarial Control Tokens},
  author    = {Li, Tung-Ling and Wu, Yuhao and Liu, Hongliang},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Work context

The work began while Tung-Ling Li was at Palo Alto Networks; additional experiments were conducted during the rebuttal period while he was at Crusoe.ai. Yuhao Wu and Hongliang Liu remain at Palo Alto Networks. These affiliations describe the authors' work context and do not imply endorsement by either organization.

## License

No license is included in this prepared snapshot. Before public release, the authors should confirm and add suitable licensing for the code, derived data, and paper PDF, including any applicable employer or third-party terms. Until then, public visibility should not be interpreted as permission to reuse or redistribute these materials.

## Contact

For artifact questions, contact Tung-Ling Li at `tony51307@gmail.com`.
