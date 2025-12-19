from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from logging_config import get_logger

logger = get_logger(__name__)

def recall_at_k(ranks: List[int], k: int) -> float:
    # ranks: 1-based rank positions of true positives in the ranked list
    return float(np.mean([1.0 if r <= k else 0.0 for r in ranks])) if ranks else 0.0


def ndcg_at_k(ranks: List[int], k: int) -> float:
    # DCG with multiple positives (if test has multiple items)
    dcg = sum(1.0 / np.log2(r + 1) for r in ranks if r <= k)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(ranks), k)))
    return float(dcg / ideal) if ideal > 0 else 0.0

def get_rr(ranks: List[int]) -> float:
    # Reciprocal Rank
    if not ranks: return 0.0
    return 1.0 / ranks[0] # Best rank is the first one

def get_hr(ranks: List[int], k: int) -> float:
    for r in ranks:
        if r <= k:
            return 1.0
    return 0.0


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_model_class(model_name: str):
    name = model_name.strip().lower()
    if name == "gat":
        from train_gat_neighbor import HeteroGAT  # type: ignore
        return HeteroGAT
    if name == "sage":
        from train_sage_neighbor import HeteroSAGE  # type: ignore
        return HeteroSAGE

    raise ValueError(f"Unknown model.name='{model_name}'. Expected one of: gat | sage")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config_eval.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    proc_dir = Path(cfg["data"]["proc_dir"])
    graph_path = Path(cfg["data"]["graph_path"])

    model_name = cfg["model"]["name"]
    ckpt_path = Path(cfg["model"]["ckpt_path"])
    hidden = int(cfg["model"].get("hidden", 32))
    heads = int(cfg["model"].get("heads", 2))

    ks = [int(x) for x in cfg["eval"].get("ks", [5, 10])]
    num_neg = int(cfg["eval"].get("num_neg", 99))
    seed = int(cfg["eval"].get("seed", 42))

    device = torch.device("cpu")

    logger.info("Loading graph...")
    data = torch.load(graph_path, weights_only=False).to(device)

    logger.info("Loading mappings...")
    with open(proc_dir / "mappings.json") as f:
        mp = json.load(f)

    num_users = mp["num_users"]
    num_books = mp["num_books"]
    num_movies = mp["num_movies"]

    # instantiate model
    logger.info(f"Loading model class: {model_name}")
    ModelCls = load_model_class(model_name)

    # GAT signature: (num_users, num_books, num_movies, hidden=, heads=)
    # SAGE signature: (num_users, num_books, num_movies, hidden=)  [heads ignored]
    if model_name.lower() == "gat":
        model = ModelCls(num_users, num_books, num_movies, hidden=hidden, heads=heads).to(device)
    else:
        model = ModelCls(num_users, num_books, num_movies, hidden=hidden).to(device)

    logger.info(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    logger.info("Loading splits...")
    train_df = pd.read_parquet(proc_dir / "movies_train.parquet")
    val_df = pd.read_parquet(proc_dir / "movies_val.parquet")
    test_df = pd.read_parquet(proc_dir / "movies_test.parquet")

    # user -> seen movie set (train + val + test) to avoid sampling seen items as negatives
    seen: Dict[int, set] = {}
    for df in (train_df, val_df, test_df):
        for u, g in df.groupby("uid"):
            seen.setdefault(int(u), set()).update(map(int, g["iid"]))

    test_by_user = test_df.groupby("uid")["iid"].apply(list).to_dict()

    logger.info("Computing embeddings...")
    with torch.no_grad():
        z = model(data)  # returns dict: {"user": [U,H], "movie":[M,H], ...}
        ZU = z["user"]
        ZM = z["movie"]

    rng = np.random.default_rng(seed)
    metrics = {f"Recall@{k}": [] for k in ks} | \
          {f"NDCG@{k}": [] for k in ks} | \
          {f"HR@{k}": [] for k in ks} | \
          {"MRR": []} 

    logger.info("Evaluating...")
    for u, pos_items in test_by_user.items():
        if not pos_items:
            continue

        # sample negatives
        negs = []
        while len(negs) < num_neg:
            cand = int(rng.integers(0, num_movies))
            if cand not in seen.get(int(u), set()):
                negs.append(cand)

        candidates = list(map(int, pos_items)) + negs

        # dot product score
        scores = (ZM[candidates] * ZU[int(u)]).sum(dim=1)

        ranked = torch.argsort(scores, descending=True)
        ranked_items = [candidates[i] for i in ranked.tolist()]

        # ranks of positives (1-based)
        pos_set = set(map(int, pos_items))
        ranks = []
        for idx, it in enumerate(ranked_items, start=1):
            if it in pos_set:
                ranks.append(idx)

        for k in ks:
            metrics[f"Recall@{k}"].append(recall_at_k(ranks, k))
            metrics[f"NDCG@{k}"].append(ndcg_at_k(ranks, k))
            metrics[f"HR@{k}"].append(get_hr(ranks, k))
        metrics["MRR"].append(get_rr(ranks)) 

    logger.info("\n=== Evaluation Results ===")
    for k in ks:
        logger.info(f"Recall@{k}: {np.mean(metrics[f'Recall@{k}']):.4f}")
        logger.info(f"NDCG@{k}:   {np.mean(metrics[f'NDCG@{k}']):.4f}")
        logger.info(f"HR@{k}:     {np.mean(metrics[f'HR@{k}']):.4f}")
    logger.info(f"MRR:        {np.mean(metrics['MRR']):.4f}")
if __name__ == "__main__":
    main()