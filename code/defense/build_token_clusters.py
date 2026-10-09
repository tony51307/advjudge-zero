"""Build semantic clusters for the AdvJudge-Zero in-family Llama tokens (Omni-Judge),
using either:
  (1) effect-based: cluster tokens by their per-prompt flip pattern across the 4
      v4-clean baseline evals (each token's row in a 1200-prompt binary matrix)
  (2) embedding-based: encode each token with sentence-transformers and cluster
      in embedding space.

Output: a JSON mapping token -> cluster_id, plus a summary.
"""
import os, json, csv, argparse
from collections import defaultdict
import numpy as np


def load_baseline_flip_matrix(token_split_path, csv_dir, judge="omnijudge"):
    """Build a (n_tokens × n_prompts) binary matrix from baseline per-token CSVs.
    Prompts are concatenated across the 4 datasets so each token has a 1200-dim row
    indicating which of 1200 prompts it flipped at baseline.
    """
    split = json.load(open(token_split_path))
    tokens = sorted(split["train_tokens"])
    tok_idx = {t: i for i, t in enumerate(tokens)}

    flips = np.zeros((len(tokens), 4 * 300), dtype=np.int8)
    for ds_off, ds in enumerate(["AIME", "MATH", "GSM8K", "MultiRLVR"]):
        path = os.path.join(csv_dir, f"{judge}_BASELINE_{ds}.csv")
        if not os.path.exists(path):
            print(f"  warn: {path} missing")
            continue
        with open(path) as f:
            for r in csv.DictReader(f):
                if r["pool"] != "train": continue
                tk = r["token"]
                pid = int(r["prompt_idx"])
                if tk in tok_idx:
                    flips[tok_idx[tk], ds_off * 300 + pid] = 1
    return tokens, flips


def cluster_effect(tokens, flips, k=9):
    from sklearn.cluster import KMeans
    # Tokens with all-zero flip rows still get clustered (their rows are identical)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(flips)
    return labels


def cluster_embedding(tokens, k=9):
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embs = enc.encode([t[:200] for t in tokens], show_progress_bar=False, normalize_embeddings=True)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embs)
    return labels


def cluster_umap_hdbscan(tokens, source="embedding", min_cluster_size=8, n_neighbors=15, n_components=5, random_state=42):
    """UMAP -> HDBSCAN: auto-K density-based clustering with noise labels.
    Noise points (label = -1) are reassigned by HDBSCAN's approximate_predict to
    their nearest cluster so every token is sampleable in the diverse mixture.
    """
    import umap
    import hdbscan
    if source == "embedding":
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        feats = enc.encode([t[:200] for t in tokens], show_progress_bar=False, normalize_embeddings=True)
    elif source == "effect":
        # passed-in effect matrix is required; fetch via global hack to keep call symmetric
        feats = globals()["_EFFECT_MATRIX"]
    else:
        raise ValueError(source)

    # UMAP to 5-D space, then HDBSCAN
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components,
                         metric="cosine", random_state=random_state)
    embs = reducer.fit_transform(feats)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                  metric="euclidean", cluster_selection_method="eom",
                                  prediction_data=True)
    labels = clusterer.fit_predict(embs)

    # Reassign noise points (label=-1) to nearest cluster centroid (so diverse
    # sampling treats every token as cluster-attached; alternatively, keep
    # noise as its own "noise" cluster).
    if (labels == -1).any():
        import numpy as _np
        valid_mask = labels != -1
        if valid_mask.sum() > 0:
            cluster_ids = sorted(set(labels[valid_mask].tolist()))
            centroids = _np.stack([embs[labels == c].mean(0) for c in cluster_ids])
            noise_idx = _np.where(labels == -1)[0]
            for i in noise_idx:
                d = _np.linalg.norm(centroids - embs[i], axis=1)
                labels[i] = cluster_ids[d.argmin()]
    return labels


def summarize(tokens, labels, label):
    print(f"\n=== {label} clusters ({len(set(labels))} groups) ===")
    by_cluster = defaultdict(list)
    for t, l in zip(tokens, labels):
        by_cluster[int(l)].append(t)
    for c, ts in sorted(by_cluster.items(), key=lambda x: -len(x[1])):
        sample = ts[0][:50].replace("\n", "\\n").replace("\t", "\\t")
        print(f"  cluster {c:>2}: {len(ts):>4} tokens   sample={sample!r}")
    return by_cluster


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    p.add_argument("--token-split", default=os.path.join(
        repo_root, "data/token_splits/robust_v4clean_omnijudge_token_split.json"))
    p.add_argument("--csv-dir", default=os.path.join(repo_root, "results/per_token_flips"))
    p.add_argument("--judge", default="omnijudge")
    p.add_argument("--k", type=int, default=9)
    p.add_argument("--method", required=True, choices=["effect", "embedding", "umap_hdbscan_emb", "umap_hdbscan_eff"])
    p.add_argument("--min-cluster-size", type=int, default=8)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    tokens, flips = load_baseline_flip_matrix(args.token_split, args.csv_dir, args.judge)
    print(f"Loaded {len(tokens)} tokens, baseline flip matrix shape {flips.shape}, "
          f"avg flips/token = {flips.sum()/len(tokens):.1f}")

    if args.method == "effect":
        labels = cluster_effect(tokens, flips, args.k)
    elif args.method == "embedding":
        labels = cluster_embedding(tokens, args.k)
    elif args.method == "umap_hdbscan_emb":
        labels = cluster_umap_hdbscan(tokens, source="embedding",
                                       min_cluster_size=args.min_cluster_size)
    elif args.method == "umap_hdbscan_eff":
        globals()["_EFFECT_MATRIX"] = flips.astype("float32")
        labels = cluster_umap_hdbscan(tokens, source="effect",
                                       min_cluster_size=args.min_cluster_size)

    by_cluster = summarize(tokens, labels, args.method)

    with open(args.output, "w") as f:
        json.dump({
            "method": args.method, "k": args.k,
            "token_to_cluster": {t: int(l) for t, l in zip(tokens, labels)},
            "clusters": {str(c): ts for c, ts in by_cluster.items()},
        }, f, indent=2)
    print(f"\nSaved cluster assignment -> {args.output}")
