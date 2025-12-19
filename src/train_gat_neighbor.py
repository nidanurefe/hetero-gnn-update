from __future__ import annotations
import torch_sparse, torch_scatter
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import GATConv, HeteroConv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROC = Path("data/processed")

MOVIE_REL = ("user", "rates_movie", "movie")
BOOK_REL  = ("user", "rates_book", "book")


class HeteroGAT(nn.Module):
    def __init__(self, num_users: int, num_books: int, num_movies: int, hidden: int = 32, heads: int = 2):
        super().__init__()
        self.hidden = hidden

        self.user_emb = nn.Embedding(num_users, hidden)
        self.book_emb = nn.Embedding(num_books, hidden)
        self.movie_emb = nn.Embedding(num_movies, hidden)

        # First layer
        convs1 = {
            ("user", "rates_book", "book"): GATConv((hidden, hidden), hidden, heads=heads, concat=False, add_self_loops=False),
            ("book", "rev_rates_book", "user"): GATConv((hidden, hidden), hidden, heads=heads, concat=False, add_self_loops=False),
            ("user", "rates_movie", "movie"): GATConv((hidden, hidden), hidden, heads=heads, concat=False, add_self_loops=False),
            ("movie", "rev_rates_movie", "user"): GATConv((hidden, hidden), hidden, heads=heads, concat=False, add_self_loops=False),
        }
        self.conv1 = HeteroConv(convs1, aggr="sum")

        # Second layer
        convs2 = {
            ("user", "rates_book", "book"): GATConv((hidden, hidden), hidden, heads=heads, concat=False, add_self_loops=False),
            ("book", "rev_rates_book", "user"): GATConv((hidden, hidden), hidden, heads=heads, concat=False, add_self_loops=False),
            ("user", "rates_movie", "movie"): GATConv((hidden, hidden), hidden, heads=heads, concat=False, add_self_loops=False),
            ("movie", "rev_rates_movie", "user"): GATConv((hidden, hidden), hidden, heads=heads, concat=False, add_self_loops=False),
        }
        self.conv2 = HeteroConv(convs2, aggr="sum")

        self.lin = nn.ModuleDict({
            "user": nn.Linear(hidden, hidden),
            "book": nn.Linear(hidden, hidden),
            "movie": nn.Linear(hidden, hidden),
        })

    def forward(self, batch: HeteroData) -> Dict[str, torch.Tensor]:
        def pick_x(node_type: str, emb: nn.Embedding) -> torch.Tensor:
            store = batch[node_type]

            if hasattr(store, "n_id") and store.n_id is not None:
                return emb.weight[store.n_id]  # only sampled nodes
            
            return emb.weight  # all nodes

        x = {
            "user": pick_x("user", self.user_emb),
            "book": pick_x("book", self.book_emb),
            "movie": pick_x("movie", self.movie_emb),
        }

        x = self.conv1(x, batch.edge_index_dict)
        x = {k: F.relu(v) for k, v in x.items()}

        x = self.conv2(x, batch.edge_index_dict)
        x = {k: F.relu(self.lin[k](v)) for k, v in x.items()}
        
        return x

def link_bce_loss(z_user: torch.Tensor, z_item: torch.Tensor,
                  edge_label_index: torch.Tensor, edge_label: torch.Tensor) -> torch.Tensor:
    src = edge_label_index[0]
    dst = edge_label_index[1]
    scores = (z_user[src] * z_item[dst]).sum(dim=-1)
    return F.binary_cross_entropy_with_logits(scores, edge_label.float())


def make_link_loader(
    data: HeteroData,
    rel: Tuple[str, str, str],
    edge_index: torch.Tensor, # Train data
    batch_size: int,
    num_neighbors: list[int],
    neg_ratio: float,
    shuffle: bool = True,
) -> LinkNeighborLoader:
    edge_label_index = (rel, edge_index)
    edge_label = torch.ones(edge_index.size(1), dtype=torch.float)

    return LinkNeighborLoader(
        data=data,
        num_neighbors=num_neighbors,         
        edge_label_index=edge_label_index,    
        edge_label=edge_label,               
        neg_sampling_ratio=neg_ratio,        
        batch_size=batch_size,
        shuffle=shuffle,
    )


def get_edge_count(data: HeteroData, rel: Tuple[str, str, str]) -> int:
    try:
        return int(data[rel].edge_index.size(1))
    except Exception:
        return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config_neighbor.yaml", help="Path to config file")
    args_cmd = parser.parse_args()

    with open(args_cmd.config, "r") as f:
        config = yaml.safe_load(f)

    graph_path = config["training"]["graph_path"]
    save_path = config["training"]["save_path"] + "/gat.pt"
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    lr = config["training"]["lr"]
    weight_decay = config["training"]["weight_decay"]
    num_neighbors = config["training"]["num_neighbors"]
    neg_ratio = config["training"]["neg_ratio"]
    
    hidden = config["model"]["hidden"]
    heads = config["model"]["heads"]
    
    use_books_loss = config["loss"]["use_books_loss"]
    lambda_book = config["loss"]["lambda_book"]

    logger.info("Starting training (neighbor sampling)...")
    device = torch.device("cpu")
    logger.info(f"Device: {device}")

    logger.info(f"Loading graph: {graph_path}")
    data: HeteroData = torch.load(graph_path, weights_only=False)

    with open(PROC / "mappings.json") as f:
        mp = json.load(f)

    num_users = mp["num_users"]
    num_books = mp["num_books"]
    num_movies = mp["num_movies"]

    e_movie = get_edge_count(data, MOVIE_REL)
    e_book  = get_edge_count(data, BOOK_REL)
    logger.info(f"Graph sizes - Users={num_users} Books={num_books} Movies={num_movies}")
    logger.info(f"Edge counts - movie={e_movie} book={e_book}")

    movies_df = pd.read_parquet(PROC / "movies_train.parquet")
    movie_edge_index = torch.from_numpy(
        movies_df[["uid", "iid"]].to_numpy().T
    ).long()

    use_books = bool(use_books_loss) and (e_book > 0)
    if use_books_loss and not use_books:
        logger.warning("use_books_loss=1 ama graph'ta book edge yok -> otomatik kapatıldı.")

    if use_books:
        books_df = pd.read_parquet(PROC / "books_train.parquet")
        book_edge_index = torch.from_numpy(
        books_df[["uid", "iid"]].to_numpy().T
    ).long()
    else:
        book_edge_index = None


    movie_loader = make_link_loader(
        data=data,
        rel=MOVIE_REL,
        edge_index=movie_edge_index,
        batch_size=batch_size,
        num_neighbors=num_neighbors,
        neg_ratio=neg_ratio,
        shuffle=True,
    )

    book_loader: Optional[LinkNeighborLoader] = None
    if use_books:
        book_loader = make_link_loader(
            data=data,
            rel=BOOK_REL,
            edge_index=book_edge_index,
            batch_size=batch_size,
            num_neighbors=num_neighbors,
            neg_ratio=neg_ratio,
            shuffle=True,
        )
        book_iter = iter(book_loader)


    model = HeteroGAT(num_users, num_books, num_movies, hidden=hidden, heads=heads).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    logger.info(
        f"Config: epochs={epochs}, batch_size={batch_size}, num_neighbors={num_neighbors}, "
        f"neg_ratio={neg_ratio}, use_books={int(use_books)}, lambda_book={lambda_book}"
    )

    data = data.to(device)  

    for ep in range(1, epochs + 1):
        model.train()
        total = total_movie = total_book = 0.0
        n_steps = 0

        for mbatch in movie_loader:
            mbatch = mbatch.to(device)

            z = model(mbatch)


            m_ei = mbatch[MOVIE_REL].edge_label_index
            m_y  = mbatch[MOVIE_REL].edge_label
            loss_movie = link_bce_loss(z["user"], z["movie"], m_ei, m_y)

            loss = loss_movie


            if use_books:
                try:
                    bbatch = next(book_iter)
                except StopIteration:
                    book_iter = iter(book_loader)
                    bbatch = next(book_iter)

                bbatch = bbatch.to(device)
                bz = model(bbatch)

                b_ei = bbatch[BOOK_REL].edge_label_index
                b_y  = bbatch[BOOK_REL].edge_label
                loss_book = link_bce_loss(bz["user"], bz["book"], b_ei, b_y)

                loss = loss + lambda_book * loss_book
            else:
                loss_book = torch.tensor(0.0, device=device)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += float(loss.item())
            total_movie += float(loss_movie.item())
            total_book += float(loss_book.item())
            n_steps += 1

        if use_books:
            logger.info(
                f"[Epoch {ep:03d}] loss={total/max(n_steps,1):.4f} | movie={total_movie/max(n_steps,1):.4f} | book={total_book/max(n_steps,1):.4f}"
            )
        else:
            logger.info(
                f"[Epoch {ep:03d}] loss={total/max(n_steps,1):.4f} | movie={total_movie/max(n_steps,1):.4f}"
            )

    logger.info(f"Saving model to: {save_path}")
    torch.save(model.state_dict(), save_path)
    logger.info("✓ Saved")


if __name__ == "__main__":
    main()