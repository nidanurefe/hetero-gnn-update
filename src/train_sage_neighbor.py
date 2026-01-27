from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import matplotlib.pyplot as plt
import yaml
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import HeteroConv, SAGEConv

from logging_config import get_logger

logger = get_logger(__name__)

PROC = Path("data/processed")

MOVIE_REL = ("user", "rates_movie", "movie")
BOOK_REL  = ("user", "rates_book", "book")


# Model (2 layer HeteroSAGE)
# Uses neighbor sampling
# Batch[node_type].n_id => global node IDs in the subgraph
# Select embeddings only for those nodes
class HeteroSAGE(nn.Module):
    def __init__(self, num_users: int, num_books: int, num_movies: int, hidden: int = 64):
        super().__init__()
        self.hidden = hidden

        self.user_emb  = nn.Embedding(num_users, hidden)
        self.book_emb  = nn.Embedding(num_books, hidden)
        self.movie_emb = nn.Embedding(num_movies, hidden)
        
        # Initialize embeddings with Xavier uniform
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.book_emb.weight)
        nn.init.xavier_uniform_(self.movie_emb.weight)

        # Layer 1
        convs1 = {
            ("user", "rates_book", "book"): SAGEConv((hidden, hidden), hidden),
            ("book", "rev_rates_book", "user"): SAGEConv((hidden, hidden), hidden),
            ("user", "rates_movie", "movie"): SAGEConv((hidden, hidden), hidden),
            ("movie", "rev_rates_movie", "user"): SAGEConv((hidden, hidden), hidden),
        }
        self.conv1 = HeteroConv(convs1, aggr="sum")

        # Layer 2
        convs2 = {
            ("user", "rates_book", "book"): SAGEConv((hidden, hidden), hidden),
            ("book", "rev_rates_book", "user"): SAGEConv((hidden, hidden), hidden),
            ("user", "rates_movie", "movie"): SAGEConv((hidden, hidden), hidden),
            ("movie", "rev_rates_movie", "user"): SAGEConv((hidden, hidden), hidden),
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
                return emb.weight[store.n_id]   # sampled nodes only
            return emb.weight                  # full graph

        x = {
            "user": pick_x("user", self.user_emb),
            "book": pick_x("book", self.book_emb),
            "movie": pick_x("movie", self.movie_emb),
        }

        # conv stack as usual
        x = self.conv1(x, batch.edge_index_dict)
        x = {k: F.relu(v) for k, v in x.items()}

        x = self.conv2(x, batch.edge_index_dict)
        x = {k: F.relu(self.lin[k](v)) for k, v in x.items()}
        return x


# Loss (edge label BCE)
def link_bce_loss(
    z_user: torch.Tensor,
    z_item: torch.Tensor,
    edge_label_index: torch.Tensor,
    edge_label: torch.Tensor,
) -> torch.Tensor:
    src = edge_label_index[0]
    dst = edge_label_index[1]
    scores = (z_user[src] * z_item[dst]).sum(dim=-1)  # dot product
    return F.binary_cross_entropy_with_logits(scores, edge_label.float())

def weighted_bce_loss(z_user, z_item, edge_label_index, edge_label, pos_weight: float):
    src = edge_label_index[0]
    dst = edge_label_index[1]
    scores = (z_user[src] * z_item[dst]).sum(dim=-1)
    
    pw = torch.tensor([pos_weight], device=scores.device, dtype=scores.dtype)
    return F.binary_cross_entropy_with_logits(scores, edge_label.float(), pos_weight=pw)


# DataLoader (neighbor sampling)
def make_link_loader(
    data: HeteroData,
    rel: Tuple[str, str, str],
    edge_index: torch.Tensor,
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
    parser.add_argument("--config", type=str, default="config/config_neighbor.yaml")
    args_cmd = parser.parse_args()

    with open(args_cmd.config, "r") as f:
        config = yaml.safe_load(f)

    graph_path    = config["training"]["graph_path"]
    save_path     = config["training"]["save_path"] + "/sage.pt"
    epochs        = config["training"]["epochs"]
    batch_size    = config["training"]["batch_size"]
    lr            = config["training"]["lr"]
    weight_decay  = config["training"]["weight_decay"]
    num_neighbors = config["training"]["num_neighbors"]
    neg_ratio     = config["training"]["neg_ratio"]

    hidden        = config["model"]["hidden"]

    use_books_loss = config["loss"]["use_books_loss"]
    lambda_book    = config["loss"]["lambda_book"]

    logger.info("Starting SAGE training (neighbor sampling)...")
    device = torch.device("cpu")
    logger.info(f"Device: {device}")

    logger.info(f"Loading graph: {graph_path}")
    data: HeteroData = torch.load(graph_path, weights_only=False)

    with open(PROC / "mappings.json") as f:
        mp = json.load(f)

    num_users  = mp["num_users"]
    num_books  = mp["num_books"]
    num_movies = mp["num_movies"]

    e_movie = get_edge_count(data, MOVIE_REL)
    e_book  = get_edge_count(data, BOOK_REL)
    logger.info(f"Graph sizes - Users={num_users} Books={num_books} Movies={num_movies}")
    logger.info(f"Edge counts - movie={e_movie} book={e_book}")

    # edge list from parquet (fast conversion)
    movies_df = pd.read_parquet(PROC / "movies_train.parquet")
    movie_edge_index = torch.from_numpy(movies_df[["uid", "iid"]].to_numpy().T).long()

    use_books = bool(use_books_loss) and (e_book > 0)
    if use_books_loss and not use_books:
        logger.warning("use_books_loss=1 ama graph'ta book edge yok -> otomatik kapatıldı.")

    if use_books:
        books_df = pd.read_parquet(PROC / "books_train.parquet")
        book_edge_index = torch.from_numpy(books_df[["uid", "iid"]].to_numpy().T).long()
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

    model = HeteroSAGE(num_users, num_books, num_movies, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    logger.info(
        f"Config: epochs={epochs}, batch_size={batch_size}, num_neighbors={num_neighbors}, "
        f"neg_ratio={neg_ratio}, use_books={int(use_books)}, lambda_book={lambda_book}, hidden={hidden}"
    )
    
    data = data.to(device)
    
    # Calculate pos_weight for weighted BCE
    pos_weight = neg_ratio + 1.0  # total_samples / positive_samples
    logger.info(f"Using pos_weight={pos_weight:.2f} for weighted BCE loss")
    
    # Loss tracking for plotting
    epoch_losses = []
    epoch_movie_losses = []
    epoch_book_losses = []

    for ep in range(1, epochs + 1):
        model.train()
        total = total_movie = total_book = 0.0
        n_steps = 0

        for mbatch in movie_loader:
            mbatch = mbatch.to(device)
            z = model(mbatch)

            m_ei = mbatch[MOVIE_REL].edge_label_index
            m_y  = mbatch[MOVIE_REL].edge_label
            loss_movie = weighted_bce_loss(z["user"], z["movie"], m_ei, m_y, pos_weight)

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
                loss_book = weighted_bce_loss(bz["user"], bz["book"], b_ei, b_y, pos_weight)

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

        # Logging
        avg_loss = total/max(n_steps,1)
        avg_movie_loss = total_movie/max(n_steps,1)
        avg_book_loss = total_book/max(n_steps,1)
        
        epoch_losses.append(avg_loss)
        epoch_movie_losses.append(avg_movie_loss)
        epoch_book_losses.append(avg_book_loss)
        
        log_parts = [f"[Epoch {ep:03d}] loss={avg_loss:.4f}"]
        if use_books:
            log_parts.append(f"movie={avg_movie_loss:.4f}")
            log_parts.append(f"book={avg_book_loss:.4f}")
        else:
            log_parts.append(f"movie={avg_movie_loss:.4f}")
        logger.info(" | ".join(log_parts))

    # Plot training loss curves
    logger.info("Creating loss plot...")
    plt.figure(figsize=(10, 6))
    
    epochs_range = range(1, epochs + 1)
    plt.plot(epochs_range, epoch_losses, 'b-', label='Total Loss', linewidth=2)
    plt.plot(epochs_range, epoch_movie_losses, 'r--', label='Movie Loss', linewidth=1.5)
    
    if use_books:
        plt.plot(epochs_range, epoch_book_losses, 'g--', label='Book Loss', linewidth=1.5)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('SAGE Training Loss Over Epochs', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_path = save_path.replace('.pt', '_loss_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Loss plot saved to: {plot_path}")
    plt.close()

    logger.info(f"Saving model to: {save_path}")
    torch.save(model.state_dict(), save_path)
    logger.info("✓ Saved")


if __name__ == "__main__":
    main()