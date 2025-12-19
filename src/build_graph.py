from __future__ import annotations
from pathlib import Path
import json
import torch
import pandas as pd
from torch_geometric.data import HeteroData
from logging_config import get_logger

logger = get_logger(__name__)

INP = Path("data/processed")
OUT = INP

def main():
    # load train data
    books = pd.read_parquet(INP/"books_train.parquet")
    mtrain = pd.read_parquet(INP/"movies_train.parquet")

    # read mappings
    with open(INP/"mappings.json") as f:
        mp = json.load(f)

    # build hetero graph
    num_users = mp["num_users"]
    num_books = mp["num_books"]
    num_movies = mp["num_movies"]

    data = HeteroData()
    data["user"].num_nodes = num_users
    data["book"].num_nodes = num_books
    data["movie"].num_nodes = num_movies

    # edges: user->book
    ub_src = torch.tensor(books["uid"].to_numpy(), dtype=torch.long)
    ub_dst = torch.tensor(books["iid"].to_numpy(), dtype=torch.long)
    data[("user","rates_book","book")].edge_index = torch.stack([ub_src, ub_dst], dim=0)
    data[("book","rev_rates_book","user")].edge_index = torch.stack([ub_dst, ub_src], dim=0)

    # edges: user->movie (train only)
    um_src = torch.tensor(mtrain["uid"].to_numpy(), dtype=torch.long)
    um_dst = torch.tensor(mtrain["iid"].to_numpy(), dtype=torch.long)
    data[("user","rates_movie","movie")].edge_index = torch.stack([um_src, um_dst], dim=0)
    data[("movie","rev_rates_movie","user")].edge_index = torch.stack([um_dst, um_src], dim=0)

    # save hetero graph
    torch.save(data, OUT/"hetero_graph.pt")
    print("Saved hetero_graph.pt")
    print(data)

if __name__ == "__main__":
    main()