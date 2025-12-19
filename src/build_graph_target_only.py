from __future__ import annotations
from pathlib import Path
import json
import torch
import pandas as pd
from torch_geometric.data import HeteroData
from logging_config import get_logger

logger = get_logger(__name__)


PROC = Path("data/processed")

def empty_edge_index(device=None):
    return torch.empty((2, 0), dtype=torch.long, device=device)

def main():
    with open(PROC / "mappings.json") as f:
        mp = json.load(f)

    num_users = mp["num_users"]
    num_books = mp["num_books"]
    num_movies = mp["num_movies"]

    # movies train interactions 
    mtrain = pd.read_parquet(PROC / "movies_train.parquet")

    data = HeteroData()
    data["user"].num_nodes = num_users
    data["book"].num_nodes = num_books
    data["movie"].num_nodes = num_movies

    # target only
    um_src = torch.tensor(mtrain["uid"].to_numpy(), dtype=torch.long)
    um_dst = torch.tensor(mtrain["iid"].to_numpy(), dtype=torch.long)

    data[("user", "rates_movie", "movie")].edge_index = torch.stack([um_src, um_dst], dim=0)
    data[("movie", "rev_rates_movie", "user")].edge_index = torch.stack([um_dst, um_src], dim=0)

    data[("user", "rates_book", "book")].edge_index = empty_edge_index()
    data[("book", "rev_rates_book", "user")].edge_index = empty_edge_index()

    out_path = PROC / "hetero_graph_target_only.pt"
    torch.save(data, out_path)
    print(f"Saved target-only graph to: {out_path}")
    print(data)

if __name__ == "__main__":
    main()