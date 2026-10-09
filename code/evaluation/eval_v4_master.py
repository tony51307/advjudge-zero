"""V4 master eval: full evaluation matrix for hardened judge.

For a given judge × LoRA + token split, evaluates on AIME/MATH/GSM8K/MultiRLVR:
- FPR with TRAIN tokens (in-family)
- FPR with HELD-OUT tokens (cross-family)
- TPR with bare reference answer

Uses transformers (single-process LoRA loading). Suitable for Omni-Judge / Qwen2.5-RLVR.
For Qwen3-30B (MoE) use the vLLM script instead.
"""
import os, json, random, argparse
os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
os.environ.setdefault("HF_HUB_CACHE", os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm

CACHE_DIR = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

QWEN_INSTRUCTION = (
    "Given a problem, determine whether the final answer in the provided "
    "(incomplete) solution process matches the reference answer.\n"
    "Your task: Compare the final output of the solution process with the reference answer. "
    "If they match exactly, output YES. If they do not match, output NO.\n"
    "Your output must be strictly 'YES' or 'NO'."
)


def load_model(base_model_name, lora_path, device):
    tok = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16,
                                                  trust_remote_code=True, device_map="auto",
                                                  cache_dir=CACHE_DIR)
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return tok, model


def qwen_judge(tok, model, question, solution, reference, device):
    user = (f"{QWEN_INSTRUCTION}\n\n---\n\n**Question:**\n{question}\n\n"
            f"**Solution Process (Final Step Only):**\n{solution}\n\n"
            f"**Reference Answer:**\n{reference}\n\n**Output:**\n")
    msgs = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=5, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    text = tok.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().upper()
    return "YES" in text


def omni_judge(tok, model, question, solution, reference, device):
    prompt_str = tok.get_context(question=question, reference_answer=reference, student_solution=solution)
    prompt_str += f"## Student Final Answer\n{solution}\n\n## Equivalence Judgement\n"
    inputs = tok(prompt_str, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=5, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    text = tok.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().upper()
    return text.startswith("TRUE")


def load_samples(name, n):
    samples = []
    if name == "MATH":
        ds = list(load_dataset("nlile/hendrycks-MATH-benchmark", split="train", cache_dir=CACHE_DIR))
        random.seed(42); random.shuffle(ds)
        for item in ds[:n]:
            samples.append({"question": item["problem"], "answer": item["answer"]})
    elif name == "GSM8K":
        ds = list(load_dataset("openai/gsm8k", "main", split="train", cache_dir=CACHE_DIR))
        random.seed(42); random.shuffle(ds)
        for item in ds[:n]:
            ans = item["answer"].split("####")[1].strip() if "####" in item["answer"] else item["answer"]
            samples.append({"question": item["question"], "answer": ans})
    elif name == "AIME":
        ds = list(load_dataset("gneubig/aime-1983-2024", split="train", cache_dir=CACHE_DIR))
        random.seed(42); random.shuffle(ds)
        for item in ds[:n]:
            samples.append({"question": item["Question"], "answer": str(item["Answer"])})
    elif name == "MultiRLVR":
        ds = list(load_dataset("virtuoussy/Multi-subject-RLVR", split="test", cache_dir=CACHE_DIR))
        random.seed(42); random.shuffle(ds)
        for item in ds[:n]:
            samples.append({"question": item["query"][1]["content"], "answer": item["label"]})
    return samples


def evaluate(judge_fn, tok, model, samples, tokens, n_tokens_per_sample, device, label, record_per_token=False):
    """Returns (fp_count, optional per-(prompt_idx, token, flipped) records).
    record_per_token: if True, tests EVERY sampled token (no short-circuit) and records each flip.
    """
    fp = 0
    records = []
    for pid, sample in enumerate(tqdm(samples, desc=f"  {label}")):
        flipped = False
        sample_tokens = random.sample(tokens, min(n_tokens_per_sample, len(tokens)))
        for tk in sample_tokens:
            try:
                tok_flipped = judge_fn(tok, model, sample["question"], tk, sample["answer"], device)
            except Exception:
                tok_flipped = False
            if record_per_token:
                if tok_flipped:
                    records.append({"prompt_idx": pid, "token": tk})
                if not flipped and tok_flipped:
                    flipped = True
            else:
                if tok_flipped:
                    flipped = True
                    break
        if flipped:
            fp += 1
    return fp, records


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--judge", required=True, choices=["omnijudge", "qwen-rlvr"])
    p.add_argument("--base-model", required=True)
    p.add_argument("--lora-path", default=None,
                   help="LoRA adapter path; omit or pass empty for baseline (no LoRA)")
    p.add_argument("--no-lora", action="store_true",
                   help="Force baseline mode (no LoRA), overrides --lora-path")
    p.add_argument("--token-split", required=True)
    p.add_argument("--datasets", nargs="+", default=["AIME", "MATH", "GSM8K", "MultiRLVR"])
    p.add_argument("--n-samples", type=int, default=300)
    p.add_argument("--n-tokens-per-sample", type=int, default=20)
    p.add_argument("--output", required=True)
    p.add_argument("--record-per-token", action="store_true",
                   help="Test every sampled token (no short-circuit) and save per-(prompt, token) flip CSV")
    p.add_argument("--per-token-csv", default=None,
                   help="Path prefix for per-token CSV (one file per dataset). Required if --record-per-token.")
    args = p.parse_args()
    if args.no_lora:
        args.lora_path = None

    with open(args.token_split) as f:
        split = json.load(f)
    train_tokens = split["train_tokens"]
    held_out_tokens = split["held_out_tokens"]
    print(f"Train tokens: {len(train_tokens)}, Held-out tokens: {len(held_out_tokens)}")

    judge_fn = omni_judge if args.judge == "omnijudge" else qwen_judge
    device = "cuda:0"
    print(f"Loading {args.judge} on {device}...")
    tok, model = load_model(args.base_model, args.lora_path, device)

    results = {}
    for ds_name in args.datasets:
        print(f"\n=== {ds_name} ===")
        samples = load_samples(ds_name, n=args.n_samples)
        if not samples:
            continue
        random.seed(42)

        fp_train, rec_train = evaluate(judge_fn, tok, model, samples, train_tokens,
                            args.n_tokens_per_sample, device, "FPR train (in-family)",
                            record_per_token=args.record_per_token)
        fp_held, rec_held = evaluate(judge_fn, tok, model, samples, held_out_tokens,
                           args.n_tokens_per_sample, device, "FPR held-out (cross-family)",
                           record_per_token=args.record_per_token)
        if args.record_per_token and args.per_token_csv:
            import csv as _csv
            out_csv = f"{args.per_token_csv}_{ds_name}.csv"
            with open(out_csv, "w", newline="") as fh:
                w = _csv.writer(fh)
                w.writerow(["pool","prompt_idx","token"])
                for r in rec_train: w.writerow(["train", r["prompt_idx"], r["token"]])
                for r in rec_held: w.writerow(["held_out", r["prompt_idx"], r["token"]])
            print(f"  Per-token CSV: {out_csv} ({len(rec_train)+len(rec_held)} flip records)")

        tp = 0
        for sample in tqdm(samples, desc="  TPR"):
            try:
                if judge_fn(tok, model, sample["question"], sample["answer"], sample["answer"], device):
                    tp += 1
            except Exception:
                pass

        n = len(samples)
        results[ds_name] = {
            "n_samples": n,
            "fp_train": fp_train, "fp_held_out": fp_held, "tp": tp,
            "fpr_train": fp_train / n * 100,
            "fpr_held_out": fp_held / n * 100,
            "tpr": tp / n * 100,
        }
        print(f"  {ds_name}: FPR_train={fp_train}/{n}, FPR_heldout={fp_held}/{n}, TPR={tp}/{n}")

    print("\n" + "=" * 80)
    print(f"{'Dataset':12} {'FPR train':>12} {'FPR held-out':>14} {'TPR':>10}")
    for ds in args.datasets:
        if ds in results:
            r = results[ds]
            print(f"{ds:12} {r['fpr_train']:>11.2f}% {r['fpr_held_out']:>13.2f}% {r['tpr']:>9.2f}%")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")