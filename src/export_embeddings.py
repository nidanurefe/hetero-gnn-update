from __future__ import annotations
import torch_sparse, torch_scatter
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

try:
    from data_loader import HeteroDataLoader
except ImportError:
    from src.data_loader import HeteroDataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

RAW = Path("data/raw")


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def infer_hidden_from_state_dict(state: Dict[str, torch.Tensor]) -> int | None:
    w = state.get("user_emb.weight")
    if isinstance(w, torch.Tensor) and w.ndim == 2:
        return int(w.shape[1])
    return None


def infer_num_nodes_from_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for node_type in ("user", "book", "movie"):
        w = state.get(f"{node_type}_emb.weight")
        if isinstance(w, torch.Tensor) and w.ndim == 2:
            out[node_type] = int(w.shape[0])
    return out


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
    parser = argparse.ArgumentParser(description="Export node embeddings from a trained hetero GNN checkpoint.")
    parser.add_argument("--config", type=str, default="config/config_export.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model_name = cfg["model"]["name"]
    ckpt_path = Path(cfg["model"]["ckpt_path"])
    out_dir = Path(cfg["export"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    use_ckpt_sizes = bool(cfg["export"].get("use_checkpoint_sizes", False))

    device = torch.device(cfg["export"].get("device", "cpu"))
    outfile = cfg["export"].get("outfile", f"{model_name.lower()}_embeddings.pt")
    out_path = out_dir / outfile

    logger.info("="*80)
    logger.info("EXPORTING EMBEDDINGS FROM TRAINED MODEL")
    logger.info("="*80)

    # Load data using HeteroDataLoader
    logger.info("\n Loading data with HeteroDataLoader...")
    data_loader = HeteroDataLoader(raw_dir=RAW)
    train_data, val_data, test_data, metadata = data_loader.load_data()

    # Use train_data for full graph embeddings
    data = train_data.to(device)

    num_users_map = metadata["num_users"]
    num_books_map = metadata["num_books"]
    num_movies_map = metadata["num_movies"]

    logger.info(f"\n Dataset info:")
    logger.info(f"  Users:  {num_users_map:,}")
    logger.info(f"  Books:  {num_books_map:,}")
    logger.info(f"  Movies: {num_movies_map:,}")

    logger.info(f"\n Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(state, dict):
        raise TypeError(f"Expected checkpoint to be a state_dict (dict). Got: {type(state)}")

    hidden_cfg = cfg["model"].get("hidden")
    hidden_inferred = infer_hidden_from_state_dict(state) if isinstance(state, dict) else None
    hidden = int(hidden_inferred if hidden_inferred is not None else hidden_cfg)
    if hidden_cfg is not None and int(hidden_cfg) != hidden:
        logger.warning(f"Config hidden={hidden_cfg} but checkpoint implies hidden={hidden}. Using hidden={hidden}.")

    heads = int(cfg["model"].get("heads", 2))

    sizes = infer_num_nodes_from_state_dict(state)
    num_users_ckpt = sizes.get("user", num_users_map)
    num_books_ckpt = sizes.get("book", num_books_map)
    num_movies_ckpt = sizes.get("movie", num_movies_map)

    if (num_users_ckpt, num_books_ckpt, num_movies_ckpt) != (num_users_map, num_books_map, num_movies_map):
        logger.warning(
            "Node-count mismatch between mappings.json and checkpoint embeddings.\n"
            f"- mappings: users={num_users_map}, books={num_books_map}, movies={num_movies_map}\n"
            f"- checkpoint: users={num_users_ckpt}, books={num_books_ckpt}, movies={num_movies_ckpt}\n"
            "If this checkpoint was trained with a different preprocessing run, prefer exporting with the matching "
            "proc_dir/graph_path. Alternatively set export.use_checkpoint_sizes: 1 to force-load and export anyway."
        )

    if use_ckpt_sizes:
        num_users, num_books, num_movies = num_users_ckpt, num_books_ckpt, num_movies_ckpt
    else:
        num_users, num_books, num_movies = num_users_map, num_books_map, num_movies_map

    logger.info(f"\n Instantiating model: {model_name} (hidden={hidden}, heads={heads})")
    ModelCls = load_model_class(model_name)
    if model_name.lower() == "gat":
        model = ModelCls(num_users, num_books, num_movies, hidden=hidden, heads=heads).to(device)
    else:
        model = ModelCls(num_users, num_books, num_movies, hidden=hidden).to(device)

    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        msg = (
            f"Failed to load checkpoint into model.\n"
            f"- model.name={model_name}\n"
            f"- ckpt_path={ckpt_path}\n"
            f"- hidden={hidden} (cfg={hidden_cfg}, inferred={hidden_inferred})\n"
            f"- heads={heads} (used only for GAT)\n"
            f"\nOriginal error:\n{e}"
        )
        raise RuntimeError(msg) from e

    model.eval()
    logger.info("\n Computing full-graph embeddings...")
    with torch.no_grad():
        z = model(data)  # Dict[str, Tensor]

    z_cpu = {k: v.detach().cpu() for k, v in z.items()}
    
    logger.info("\n Saving embeddings...")
    torch.save(
        {
            "model": model_name.lower(),
            "ckpt_path": str(ckpt_path),
            "hidden": hidden,
            "num_nodes": {
                "user": num_users,
                "book": num_books,
                "movie": num_movies,
            },
            "embeddings": z_cpu,
            "metadata": {
                "num_users": num_users,
                "num_books": num_books,
                "num_movies": num_movies,
            },
        },
        out_path,
    )

    logger.info("\n" + "*"*80)
    logger.info(f"Embeddings saved to: {out_path}")
    for k, v in z_cpu.items():
        logger.info(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")


if __name__ == "__main__":
    main()


