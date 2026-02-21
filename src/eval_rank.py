from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import yaml

try:
    from logging_config import get_logger
    from data_loader import HeteroDataLoader, read_inter
except ImportError:
    from src.logging_config import get_logger
    from src.data_loader import HeteroDataLoader, read_inter

logger = get_logger(__name__)

def recall_at_k(ranks: List[int], k: int) -> float:
    return float(np.mean([1.0 if r <= k else 0.0 for r in ranks])) if ranks else 0.0


def ndcg_at_k(ranks: List[int], k: int) -> float:
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


def normalize_model_name(model_name: str) -> str:

    name = model_name.strip().lower()
    if "gat" in name:
        return "gat"
    if "sage" in name:
        return "sage"
    # If exact match, return as is
    if name in ["gat", "sage"]:
        return name
    raise ValueError(f"Unknown model name '{model_name}'. Must contain 'gat' or 'sage'")


def load_model_class(model_name: str):
    name = model_name.strip().lower()
    if name == "gat":
        try:
            from train_gat_neighbor import HeteroGAT  # type: ignore
            return HeteroGAT
        except ImportError:
            from src.train_gat_neighbor import HeteroGAT  # type: ignore
            return HeteroGAT
    if name == "sage":
        try:
            from train_sage_neighbor import HeteroSAGE  # type: ignore
            return HeteroSAGE
        except ImportError:
            from src.train_sage_neighbor import HeteroSAGE  # type: ignore
            return HeteroSAGE

    raise ValueError(f"Unknown model.name='{model_name}'. Expected one of: gat | sage")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config_eval.yaml")
    parser.add_argument("--model", type=str, default=None, help="Model name: 'gat' or 'sage' (overrides config)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    proc_dir = Path(cfg["data"]["proc_dir"])

    # Use command line argument if provided, otherwise use config
    model_name_cli = args.model if args.model is not None else cfg["model"]["name"]
    # Normalize model name: CLI can accept any name, but internally we use 'gat' or 'sage'
    model_name = normalize_model_name(model_name_cli)
    
    # If model name provided via CLI, construct checkpoint path from it
    # Otherwise use the path from config
    if args.model is not None:
        # Construct path: data/processed/models/{model_name_cli}.pt
        ckpt_path = proc_dir / "models" / f"{model_name_cli}.pt"
    else:
        ckpt_path = Path(cfg["model"]["ckpt_path"])
    hidden = int(cfg["model"].get("hidden", 32))
    heads = int(cfg["model"].get("heads", 2))

    ks = [int(x) for x in cfg["eval"].get("ks", [5, 10])]
    num_neg = int(cfg["eval"].get("num_neg", 99))
    seed = int(cfg["eval"].get("seed", 42))

    device = torch.device("cpu")

    # Load data using HeteroDataLoader
    logger.info("Loading data with HeteroDataLoader...")
    raw_dir = Path("data/raw")
    data_loader = HeteroDataLoader(raw_dir=raw_dir)
    train_data, val_data, test_data, metadata = data_loader.load_data()
    
    # Use train_data as the graph for embedding generation
    data = train_data.to(device)

    num_users = metadata["num_users"]
    num_books = metadata["num_books"]
    num_movies = metadata["num_movies"]
    
    # Get mappings
    user2idx = metadata["user2idx"]
    movie2idx = metadata["movie2idx"]

    # instantiate model
    logger.info(f"Loading model class: {model_name} (from CLI: {model_name_cli})")
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

    logger.info("Loading and processing splits...")
    # Load raw inter files
    movies_train_raw = read_inter(raw_dir / "AmazonMovies.train.inter")
    movies_valid_raw = read_inter(raw_dir / "AmazonMovies.valid.inter")
    movies_test_raw = read_inter(raw_dir / "AmazonMovies.test.inter")
    
    # Apply mappings to get indices
    def apply_mapping(df, user_map, item_map):
        df = df.copy()
        # Filter to only include known entities
        df = df[df["user_id"].isin(user_map) & df["item_id"].isin(item_map)]
        df["uid"] = df["user_id"].map(user_map)
        df["iid"] = df["item_id"].map(item_map)
        return df[["uid", "iid"]]
    
    train_df = apply_mapping(movies_train_raw, user2idx, movie2idx)
    val_df = apply_mapping(movies_valid_raw, user2idx, movie2idx)
    test_df = apply_mapping(movies_test_raw, user2idx, movie2idx)
    
    logger.info(f"  Train: {len(train_df):,} edges")
    logger.info(f"  Val:   {len(val_df):,} edges")
    logger.info(f"  Test:  {len(test_df):,} edges")
    
    train_edges = set(zip(train_df["uid"], train_df["iid"]))
    test_edges = set(zip(test_df["uid"], test_df["iid"]))
    overlap = train_edges & test_edges
    if overlap:
        logger.warning(f"⚠️  WARNING: {len(overlap)} train-test edges overlap! This causes data leakage.")
    else:
        logger.info(" No train-test overlap detected")

    seen: Dict[int, set] = {}
    for df in (train_df, val_df):
        for u, g in df.groupby("uid"):
            seen.setdefault(int(u), set()).update(map(int, g["iid"]))

    test_by_user = test_df.groupby("uid")["iid"].apply(list).to_dict()
    
    logger.info(f"  Test users: {len(test_by_user):,}")
    logger.info(f"  Avg items per test user: {np.mean([len(v) for v in test_by_user.values()]):.2f}")

    logger.info("Computing embeddings...")
    with torch.no_grad():
        z = model(data) 
        ZU = z["user"]
        ZM = z["movie"]

    rng = np.random.default_rng(seed)
    metrics = {f"Recall@{k}": [] for k in ks} | \
          {f"NDCG@{k}": [] for k in ks} | \
          {f"HR@{k}": [] for k in ks} | \
          {"MRR": []} 

    logger.info("Evaluating...")
    n_users_evaluated = 0
    for u, pos_items in test_by_user.items():
        if not pos_items:
            continue

        # Select exactly 1 positive item randomly from test set
        pos_item = int(rng.choice(pos_items))
        pos_items = [pos_item]

        # sample negatives 
        user_seen = seen.get(int(u), set())
        negs = []
        attempts = 0
        max_attempts = num_neg * 100  # Avoid infinite loop
        while len(negs) < num_neg and attempts < max_attempts:
            cand = int(rng.integers(0, num_movies))
            if cand not in user_seen and cand != pos_item:
                negs.append(cand)
            attempts += 1
        
        if len(negs) < num_neg:
            logger.warning(f"User {u}: Only found {len(negs)}/{num_neg} negatives after {attempts} attempts")

        candidates = pos_items + negs

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
        n_users_evaluated += 1
    
    logger.info(f"Evaluated {n_users_evaluated} users") 

    logger.info("\nEvaluation Results: ")
    for k in ks:
        logger.info(f"Recall@{k}: {np.mean(metrics[f'Recall@{k}']):.4f}")
        logger.info(f"NDCG@{k}:   {np.mean(metrics[f'NDCG@{k}']):.4f}")
        logger.info(f"HR@{k}:     {np.mean(metrics[f'HR@{k}']):.4f}")
    logger.info(f"MRR:        {np.mean(metrics['MRR']):.4f}")
if __name__ == "__main__":
    main()