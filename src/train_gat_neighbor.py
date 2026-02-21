from __future__ import annotations
import torch_sparse, torch_scatter
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import GATConv, HeteroConv

try:
    from data_loader import HeteroDataLoader
except ImportError:
    from src.data_loader import HeteroDataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROC = Path("data/processed")
RAW = Path("data/raw")

MOVIE_REL = ("user", "rates_movie", "movie")
BOOK_REL  = ("user", "rates_book", "book")


class HeteroGAT(nn.Module):
    def __init__(self, num_users: int, num_books: int, num_movies: int, hidden: int = 32, heads: int = 2):
        super().__init__()
        self.hidden = hidden

        self.user_emb = nn.Embedding(num_users, hidden)
        self.book_emb = nn.Embedding(num_books, hidden)
        self.movie_emb = nn.Embedding(num_movies, hidden)
        
        # Initialize embeddings with Xavier uniform (better than default)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.book_emb.weight)
        nn.init.xavier_uniform_(self.movie_emb.weight)

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

def weighted_bce_loss(z_user, z_item, edge_label_index, edge_label, pos_weight: float):
    src = edge_label_index[0]
    dst = edge_label_index[1]
    scores = (z_user[src] * z_item[dst]).sum(dim=-1)

    pw = torch.tensor([pos_weight], device=scores.device, dtype=scores.dtype)
    return F.binary_cross_entropy_with_logits(scores, edge_label.float(), pos_weight=pw)


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

    logger.info("="*80)
    logger.info("Starting GAT training with HeteroDataLoader...")
    logger.info("="*80)
    device = torch.device("cpu")
    logger.info(f"Device: {device}")

    # Load data using HeteroDataLoader
    logger.info("\n Loading data with HeteroDataLoader...")
    data_loader = HeteroDataLoader(raw_dir=RAW)
    train_data, val_data, test_data, metadata = data_loader.load_data()

    num_users = metadata["num_users"]
    num_books = metadata["num_books"]
    num_movies = metadata["num_movies"]

    logger.info(f"\nDataset loaded:")
    logger.info(f"  Users:  {num_users:,}")
    logger.info(f"  Books:  {num_books:,}")
    logger.info(f"  Movies: {num_movies:,}")
    logger.info(f"  Train edges - Movies: {metadata['train_stats']['num_movie_edges']:,}, Books: {metadata['train_stats']['num_book_edges']:,}")

    # Use train_data as the main graph
    data = train_data
    
    e_movie = get_edge_count(data, MOVIE_REL)
    e_book  = get_edge_count(data, BOOK_REL)

    # Get edge indices for training
    movie_edge_index = data[MOVIE_REL].edge_index
    book_edge_index = data[BOOK_REL].edge_index if e_book > 0 else None

    use_books = bool(use_books_loss) and (e_book > 0)
    if use_books_loss and not use_books:
        logger.warning("use_books_loss=1 ama graph'ta book edge yok -> otomatik kapatıldı.")

    use_movies = movie_edge_index.size(1) > 0
    if not use_movies:
        logger.warning("Movie edges empty. Movie training will be skipped.")
        if not use_books:
            raise RuntimeError("Both movie and book training data are empty. Cannot train model.")

    movie_loader: Optional[LinkNeighborLoader] = None
    if use_movies:
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
    book_iter = None
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
        f"neg_ratio={neg_ratio}, use_movies={int(use_movies)}, use_books={int(use_books)}, lambda_book={lambda_book}"
    )

    data = data.to(device)  

    # Loss tracking for plotting
    epoch_losses = []
    epoch_movie_losses = []
    epoch_book_losses = []

    for ep in range(1, epochs + 1):
        model.train()
        total = total_movie = total_book = 0.0
        n_steps = 0

        # Determine which loader to use as primary
        if use_movies:
            primary_loader = movie_loader
        elif use_books:
            primary_loader = book_loader
        else:
            raise RuntimeError("No training data available (both movies and books are empty)")

        for primary_batch in primary_loader:
            primary_batch = primary_batch.to(device)
            z = model(primary_batch)

            loss = torch.tensor(0.0, device=device)
            loss_movie = torch.tensor(0.0, device=device)
            loss_book = torch.tensor(0.0, device=device)

            # Movie loss
            if use_movies:
                m_ei = primary_batch[MOVIE_REL].edge_label_index
                m_y  = primary_batch[MOVIE_REL].edge_label
                loss_movie = weighted_bce_loss(
                    z["user"], z["movie"], m_ei, m_y,
                    pos_weight=neg_ratio 
                )
                # loss_movie = link_bce_loss(z["user"], z["movie"], m_ei, m_y)
                loss = loss + loss_movie

            # Book loss
            if use_books:
                if use_movies:
                    # If using movies as primary, get book batch separately
                    try:
                        bbatch = next(book_iter)
                    except StopIteration:
                        book_iter = iter(book_loader)
                        bbatch = next(book_iter)
                    bbatch = bbatch.to(device)
                    bz = model(bbatch)
                else:
                    # If using books as primary, use the current batch
                    bbatch = primary_batch
                    bz = z

                b_ei = bbatch[BOOK_REL].edge_label_index
                b_y  = bbatch[BOOK_REL].edge_label
            
                loss_book = weighted_bce_loss(bz["user"], bz["book"], b_ei, b_y, pos_weight=neg_ratio )
                # loss_book = link_bce_loss(bz["user"], bz["book"], b_ei, b_y)
                loss = loss + lambda_book * loss_book

            opt.zero_grad()
            loss.backward()
            
            # Check gradients (debug)
            if ep == 1 and n_steps == 0:
                total_grad_norm = 0.0
                param_count = 0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param_grad_norm = param.grad.data.norm(2)
                        total_grad_norm += param_grad_norm.item() ** 2
                        param_count += 1
                total_grad_norm = total_grad_norm ** (1. / 2)
                logger.info(f"  [Debug] Gradient norm: {total_grad_norm:.6f}, Parameters with grad: {param_count}")
                logger.info(f"  [Debug] Loss values - total: {loss.item():.6f}, book: {loss_book.item():.6f}, movie: {loss_movie.item():.6f}")
            
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
        if use_movies:
            log_parts.append(f"movie={avg_movie_loss:.4f}")
        if use_books:
            log_parts.append(f"book={avg_book_loss:.4f}")
        logger.info(" | ".join(log_parts))

    # Plot training loss curves
    logger.info("Creating loss plot...")
    plt.figure(figsize=(10, 6))
    
    epochs_range = range(1, epochs + 1)
    plt.plot(epochs_range, epoch_losses, 'b-', label='Total Loss', linewidth=2)
    
    if use_movies:
        plt.plot(epochs_range, epoch_movie_losses, 'r--', label='Movie Loss', linewidth=1.5)
    if use_books:
        plt.plot(epochs_range, epoch_book_losses, 'g--', label='Book Loss', linewidth=1.5)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
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
    logger.info(" Saved")


if __name__ == "__main__":
    main()