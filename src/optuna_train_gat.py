from __future__ import annotations
import torch_sparse, torch_scatter
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

try:
    import optuna
    from optuna.trial import Trial, TrialState
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_slice,
    )
except ImportError:
    raise ImportError(
        "Optuna is not installed"
    )

from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import GATConv, HeteroConv

try:
    from data_loader import HeteroDataLoader
    from train_gat_neighbor import (
        HeteroGAT, 
        get_edge_count, 
        weighted_bce_loss, 
        make_link_loader,
        MOVIE_REL,
        BOOK_REL,
    )
except ImportError:
    from src.data_loader import HeteroDataLoader
    from src.train_gat_neighbor import (
        HeteroGAT, 
        get_edge_count, 
        weighted_bce_loss, 
        make_link_loader,
        MOVIE_REL,
        BOOK_REL,
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROC = Path("data/processed")
RAW = Path("data/raw")


def compute_recall_at_k(
    model: nn.Module,
    data: HeteroData,
    train_edges: Dict[str, torch.Tensor],
    test_edges: Dict[str, torch.Tensor],
    k: int = 10,
    num_neg: int = 99,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
) -> float:

    model.eval()
    
    with torch.no_grad():
        # Get embeddings for all nodes
        data = data.to(device)
        z = model(data)
        
        user_emb = z["user"]
        movie_emb = z["movie"]
        num_movies = movie_emb.size(0)
        
        # For simplicity, we'll compute recall on movie domain
        movie_test = test_edges.get("movie")
        movie_train = train_edges.get("movie")
        
        if movie_test is None or movie_test.size(1) == 0:
            logger.warning("No movie test edges available for evaluation")
            return 0.0
        
        # Build seen items per user from training data
        seen: Dict[int, set] = {}
        if movie_train is not None:
            train_np = movie_train.cpu().numpy()
            for u, m in zip(train_np[0], train_np[1]):
                seen.setdefault(int(u), set()).add(int(m))
        
        # Group test edges by user
        movie_test_np = movie_test.cpu().numpy()
        user_to_test_items = {}
        for u, m in zip(movie_test_np[0], movie_test_np[1]):
            if u not in user_to_test_items:
                user_to_test_items[u] = []
            user_to_test_items[u].append(m)
        
        # Evaluate using standard protocol: 1 positive + num_neg negatives
        rng = np.random.default_rng(seed)
        recalls = []
        
        for user_id, test_items in user_to_test_items.items():
            if user_id >= user_emb.size(0) or not test_items:
                continue
            
            # Sample 1 positive item
            pos_item = int(rng.choice(test_items))
            
            # Sample negative items (not seen by this user)
            user_seen = seen.get(int(user_id), set())
            negatives = []
            attempts = 0
            max_attempts = num_neg * 100
            
            while len(negatives) < num_neg and attempts < max_attempts:
                cand = int(rng.integers(0, num_movies))
                if cand not in user_seen and cand != pos_item:
                    negatives.append(cand)
                attempts += 1
            
            if len(negatives) < num_neg:
                continue
            
            candidates = [pos_item] + negatives
            candidate_tensor = torch.tensor(candidates, device=device)
            
            user_vec = user_emb[user_id].unsqueeze(0)  # [1, hidden]
            item_vecs = movie_emb[candidate_tensor]     # [num_neg+1, hidden]
            scores = (user_vec * item_vecs).sum(dim=1)  # [num_neg+1]
            
            # Rank candidates
            ranked_indices = torch.argsort(scores, descending=True)
            
            # Find rank of positive item (which is at index 0 in candidates)
            pos_rank = (ranked_indices == 0).nonzero(as_tuple=True)[0].item() + 1  # 1-based
            
            # Check if positive is in top-K
            recall = 1.0 if pos_rank <= k else 0.0
            recalls.append(recall)
        
        return float(np.mean(recalls)) if recalls else 0.0


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    movie_loader: Optional[LinkNeighborLoader],
    book_loader: Optional[LinkNeighborLoader],
    lambda_book: float,
    neg_ratio: float,
    use_movies: bool,
    use_books: bool,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.train()
    total = total_movie = total_book = 0.0
    n_steps = 0
    
    # Determine primary loader
    if use_movies:
        primary_loader = movie_loader
    elif use_books:
        primary_loader = book_loader
    else:
        raise RuntimeError("No training data available")
    
    book_iter = iter(book_loader) if book_loader is not None else None
    
    for primary_batch in primary_loader:
        primary_batch = primary_batch.to(device)
        z = model(primary_batch)
        
        loss = torch.tensor(0.0, device=device)
        loss_movie = torch.tensor(0.0, device=device)
        loss_book = torch.tensor(0.0, device=device)
        
        # Movie loss
        if use_movies:
            m_ei = primary_batch[MOVIE_REL].edge_label_index
            m_y = primary_batch[MOVIE_REL].edge_label
            loss_movie = weighted_bce_loss(
                z["user"], z["movie"], m_ei, m_y, pos_weight=neg_ratio
            )
            loss = loss + loss_movie
        
        # Book loss
        if use_books:
            if use_movies:
                # Get book batch separately
                try:
                    bbatch = next(book_iter)
                except StopIteration:
                    book_iter = iter(book_loader)
                    bbatch = next(book_iter)
                bbatch = bbatch.to(device)
                bz = model(bbatch)
            else:
                bbatch = primary_batch
                bz = z
            
            b_ei = bbatch[BOOK_REL].edge_label_index
            b_y = bbatch[BOOK_REL].edge_label
            loss_book = weighted_bce_loss(
                bz["user"], bz["book"], b_ei, b_y, pos_weight=neg_ratio
            )
            loss = loss + lambda_book * loss_book
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total += float(loss.item())
        total_movie += float(loss_movie.item())
        total_book += float(loss_book.item())
        n_steps += 1
    
    avg_loss = total / max(n_steps, 1)
    avg_movie = total_movie / max(n_steps, 1)
    avg_book = total_book / max(n_steps, 1)
    
    return avg_loss, avg_movie, avg_book


def objective(
    trial: Trial,
    config: Dict[str, Any],
    train_data: HeteroData,
    val_data: HeteroData,
    metadata: Dict[str, Any],
    device: torch.device,
) -> float:
    search_space = config["optuna"]["search_space"]
    
    if search_space["lr"]["type"] == "loguniform":
        lr = trial.suggest_float(
            "lr",
            search_space["lr"]["low"],
            search_space["lr"]["high"],
            log=True,
        )
    
    # Weight decay
    if search_space["weight_decay"]["type"] == "loguniform":
        weight_decay = trial.suggest_float(
            "weight_decay",
            search_space["weight_decay"]["low"],
            search_space["weight_decay"]["high"],
            log=True,
        )
    
    # Negative ratio
    neg_ratio = trial.suggest_categorical("neg_ratio", search_space["neg_ratio"]["choices"])
    
    # Model architecture
    hidden = trial.suggest_categorical("hidden", search_space["hidden"]["choices"])
    heads = trial.suggest_categorical("heads", search_space["heads"]["choices"])
    
    # Neighbor sampling
    num_neighbors_1 = trial.suggest_categorical(
        "num_neighbors_1", search_space["num_neighbors_1"]["choices"]
    )
    num_neighbors_2 = trial.suggest_categorical(
        "num_neighbors_2", search_space["num_neighbors_2"]["choices"]
    )
    num_neighbors = [num_neighbors_1, num_neighbors_2]
    
    # Lambda book
    lambda_book = trial.suggest_float(
        "lambda_book",
        search_space["lambda_book"]["low"],
        search_space["lambda_book"]["high"],
    )
    
    # Fixed hyperparameters from config
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    use_books_loss = config["loss"]["use_books_loss"]
    
    # Log trial parameters
    logger.info(f"\n{'='*80}")
    logger.info(f"Trial {trial.number}: Testing hyperparameters:")
    logger.info(f"  lr={lr:.6f}, weight_decay={weight_decay:.6f}")
    logger.info(f"  neg_ratio={neg_ratio}, hidden={hidden}, heads={heads}")
    logger.info(f"  num_neighbors={num_neighbors}, lambda_book={lambda_book:.3f}")
    logger.info(f"{'='*80}\n")
    
    # Prepare data
    num_users = metadata["num_users"]
    num_books = metadata["num_books"]
    num_movies = metadata["num_movies"]
    
    e_movie = get_edge_count(train_data, MOVIE_REL)
    e_book = get_edge_count(train_data, BOOK_REL)
    
    use_books = bool(use_books_loss) and (e_book > 0)
    use_movies = e_movie > 0
    
    if not use_movies and not use_books:
        raise RuntimeError("No training data available")
    
    # Create data loaders
    movie_loader = None
    if use_movies:
        movie_edge_index = train_data[MOVIE_REL].edge_index
        movie_loader = make_link_loader(
            train_data, MOVIE_REL, movie_edge_index,
            batch_size, num_neighbors, neg_ratio, shuffle=True
        )
    
    book_loader = None
    if use_books:
        book_edge_index = train_data[BOOK_REL].edge_index
        book_loader = make_link_loader(
            train_data, BOOK_REL, book_edge_index,
            batch_size, num_neighbors, neg_ratio, shuffle=True
        )
    
    # Create model
    model = HeteroGAT(num_users, num_books, num_movies, hidden=hidden, heads=heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    train_data = train_data.to(device)
    
    # Training loop
    best_metric = 0.0 if "recall" in config["evaluation"]["metric"] else float("inf")
    patience_counter = 0
    patience = config.get("pruning", {}).get("patience", 5)
    
    for epoch in range(1, epochs + 1):
        # Train one epoch
        avg_loss, avg_movie, avg_book = train_one_epoch(
            model, optimizer, movie_loader, book_loader,
            lambda_book, neg_ratio, use_movies, use_books, device
        )
        
        # Evaluate every 5 epochs or at the end
        if epoch % 5 == 0 or epoch == epochs:
            metric_name = config["evaluation"]["metric"]
            
            if metric_name == "loss":
                current_metric = avg_loss
                is_better = current_metric < best_metric
            elif "recall" in metric_name:
                # Extract k value (e.g., "recall@10" -> 10)
                k = int(metric_name.split("@")[1])
                
                # Prepare training and validation edges
                train_edges = {
                    "movie": train_data[MOVIE_REL].edge_index if e_movie > 0 else None,
                    "book": train_data[BOOK_REL].edge_index if e_book > 0 else None,
                }
                val_edges = {
                    "movie": val_data[MOVIE_REL].edge_index if e_movie > 0 else None,
                    "book": val_data[BOOK_REL].edge_index if e_book > 0 else None,
                }
                
                # Use standard evaluation protocol: 1 positive + 99 negatives
                current_metric = compute_recall_at_k(
                    model, train_data, train_edges, val_edges, 
                    k=k, num_neg=99, device=device, seed=42
                )
                is_better = current_metric > best_metric
            else:
                raise ValueError(f"Unknown metric: {metric_name}")
            
            # Update best metric
            if is_better:
                best_metric = current_metric
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Log progress
            logger.info(
                f"  [Epoch {epoch:03d}] loss={avg_loss:.4f} | "
                f"{metric_name}={current_metric:.4f} | best={best_metric:.4f}"
            )
            
            # Report to Optuna for pruning
            trial.report(best_metric, epoch)
            
            # Prune unpromising trials
            if config.get("pruning", {}).get("enabled", False):
                if trial.should_prune():
                    logger.info(f"  Trial {trial.number} pruned at epoch {epoch}")
                    raise optuna.TrialPruned()
                
                # Early stopping
                if patience_counter >= patience:
                    logger.info(f"  Early stopping at epoch {epoch}")
                    break
    
    logger.info(f"Trial {trial.number} completed with {metric_name}={best_metric:.4f}\n")
    
    # Return objective value
    # For recall/ndcg: higher is better (Optuna maximizes by default)
    # For loss: lower is better (return negative)
    if metric_name == "loss":
        return -best_metric  # Negate for minimization
    else:
        return best_metric


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization for HeteroGAT")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_optuna.yaml",
        help="Path to Optuna configuration file",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Name for the Optuna study (default: auto-generated with timestamp)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing study with the same name",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Database URL for Optuna storage (e.g., sqlite:///optuna_studies.db)",
    )
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    logger.info("="*80)
    logger.info("Optuna Hyperparameter Optimization for HeteroGAT")
    logger.info("="*80)
    
    # Automatic device selection: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Device: {device} (GPU: {torch.cuda.get_device_name(0)})")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info(f"Device: {device} (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        logger.info(f"Device: {device}")
    
    # Load data
    logger.info("\n Loading data...")
    data_loader = HeteroDataLoader(raw_dir=RAW)
    train_data, val_data, test_data, metadata = data_loader.load_data()
    
    logger.info(f"Dataset loaded:")
    logger.info(f"  Users:  {metadata['num_users']:,}")
    logger.info(f"  Books:  {metadata['num_books']:,}")
    logger.info(f"  Movies: {metadata['num_movies']:,}")
    logger.info(
        f"  Train edges - Movies: {metadata['train_stats']['num_movie_edges']:,}, "
        f"Books: {metadata['train_stats']['num_book_edges']:,}"
    )
    
    # Create study name
    if args.study_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name = f"hetero_gat_{timestamp}"
    else:
        study_name = args.study_name
    
    logger.info(f"\n Creating Optuna study: {study_name}")
    
    # Create or load study
    storage = args.storage if args.storage else None
    load_if_exists = args.resume
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=load_if_exists,
        direction="maximize",  # We want to maximize recall (or minimize -loss)
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    
    # Optimize
    n_trials = config["optuna"]["n_trials"]
    timeout = config["optuna"].get("timeout")
    
    logger.info(f"\n Starting optimization with {n_trials} trials...")
    logger.info(f"Optimization metric: {config['evaluation']['metric']}")
    
    study.optimize(
        lambda trial: objective(trial, config, train_data, val_data, metadata, device),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )
    
    # Results
    logger.info("\n" + "="*80)
    logger.info("OPTIMIZATION COMPLETED")
    logger.info("="*80)
    
    logger.info(f"\nNumber of finished trials: {len(study.trials)}")
    logger.info(f"Number of pruned trials: {len([t for t in study.trials if t.state == TrialState.PRUNED])}")
    logger.info(f"Number of completed trials: {len([t for t in study.trials if t.state == TrialState.COMPLETE])}")
    
    # Best trial
    best_trial = study.best_trial
    logger.info(f"\nBest trial: {best_trial.number}")
    logger.info(f"  Value ({config['evaluation']['metric']}): {best_trial.value:.4f}")
    logger.info("  Best hyperparameters:")
    for key, value in best_trial.params.items():
        logger.info(f"    {key}: {value}")
    
    # Save best hyperparameters
    output_dir = Path(config["training"]["save_path"]).parent / "optuna_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_params_file = output_dir / f"{study_name}_best_params.json"
    with open(best_params_file, "w") as f:
        json.dump(best_trial.params, f, indent=2)
    logger.info(f"\nBest hyperparameters saved to: {best_params_file}")
    
    # Save study statistics
    study_stats = {
        "study_name": study_name,
        "n_trials": len(study.trials),
        "n_completed": len([t for t in study.trials if t.state == TrialState.COMPLETE]),
        "n_pruned": len([t for t in study.trials if t.state == TrialState.PRUNED]),
        "best_trial": best_trial.number,
        "best_value": best_trial.value,
        "best_params": best_trial.params,
        "optimization_metric": config["evaluation"]["metric"],
    }
    
    stats_file = output_dir / f"{study_name}_stats.json"
    with open(stats_file, "w") as f:
        json.dump(study_stats, f, indent=2)
    logger.info(f"Study statistics saved to: {stats_file}")
    
    # Create results table with all trials
    logger.info("\n Creating results table...")
    
    trials_data = []
    for trial in study.trials:
        if trial.state == TrialState.COMPLETE:
            trial_dict = {
                "trial_number": trial.number,
                "value": trial.value,
                "state": trial.state.name,
            }
            # Add all parameters
            trial_dict.update(trial.params)
            trials_data.append(trial_dict)
    
    # Create DataFrame and sort by value (best first)
    trials_df = pd.DataFrame(trials_data)
    
    if len(trials_df) > 0:
        # Sort by value (descending for recall, ascending for loss)
        metric_name = config["evaluation"]["metric"]
        ascending = True if metric_name == "loss" else False
        trials_df = trials_df.sort_values("value", ascending=ascending)
        
        # Save to CSV
        csv_file = output_dir / f"{study_name}_all_trials.csv"
        trials_df.to_csv(csv_file, index=False, float_format="%.6f")
        logger.info(f"All trials saved to CSV: {csv_file}")
        
        # Save to Excel (more readable)
        try:
            excel_file = output_dir / f"{study_name}_all_trials.xlsx"
            trials_df.to_excel(excel_file, index=False, float_format="%.6f")
            logger.info(f"All trials saved to Excel: {excel_file}")
        except Exception as e:
            logger.warning(f"Could not save Excel file: {e}. Install openpyxl: pip install openpyxl")
        
        # Print top 10 trials to console
        logger.info(f"\n{'='*100}")
        logger.info(f"TOP 10 TRIALS (sorted by {metric_name}):")
        logger.info(f"{'='*100}")
        
        top_n = min(10, len(trials_df))
        display_columns = ["trial_number", "value"] + list(trials_df.columns[3:])  # Skip state
        
        # Print header
        header = " | ".join([f"{col:>12}" for col in display_columns])
        logger.info(header)
        logger.info("-" * len(header))
        
        # Print top trials
        for idx, row in trials_df.head(top_n).iterrows():
            values = []
            for col in display_columns:
                if col == "trial_number":
                    values.append(f"{int(row[col]):>12}")
                elif col == "value":
                    values.append(f"{row[col]:>12.4f}")
                elif isinstance(row[col], float):
                    values.append(f"{row[col]:>12.6f}")
                else:
                    values.append(f"{row[col]:>12}")
            logger.info(" | ".join(values))
        
        logger.info(f"{'='*100}\n")
    else:
        logger.warning("No completed trials to display")
    
    # Create visualizations
    logger.info("\n Creating visualizations...")
    
    try:
        # Optimization history
        fig = plot_optimization_history(study)
        fig.write_image(str(output_dir / f"{study_name}_history.png"))
        logger.info(f"  - Optimization history: {output_dir / f'{study_name}_history.png'}")
    except Exception as e:
        logger.warning(f"Could not create optimization history plot: {e}")
    
    try:
        # Parameter importances
        fig = plot_param_importances(study)
        fig.write_image(str(output_dir / f"{study_name}_importances.png"))
        logger.info(f"  - Parameter importances: {output_dir / f'{study_name}_importances.png'}")
    except Exception as e:
        logger.warning(f"Could not create parameter importances plot: {e}")
    
    try:
        # Slice plot
        fig = plot_slice(study)
        fig.write_image(str(output_dir / f"{study_name}_slice.png"))
        logger.info(f"  - Slice plot: {output_dir / f'{study_name}_slice.png'}")
    except Exception as e:
        logger.warning(f"Could not create slice plot: {e}")
    
    # Train final model with best hyperparameters
    logger.info("\n Training final model with best hyperparameters...")
    
    best_params = best_trial.params
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    
    # Create final model
    model = HeteroGAT(
        metadata["num_users"],
        metadata["num_books"],
        metadata["num_movies"],
        hidden=best_params["hidden"],
        heads=best_params["heads"],
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
    )
    
    # Create loaders with best params
    num_neighbors = [best_params["num_neighbors_1"], best_params["num_neighbors_2"]]
    neg_ratio = best_params["neg_ratio"]
    lambda_book = best_params["lambda_book"]
    
    e_movie = get_edge_count(train_data, MOVIE_REL)
    e_book = get_edge_count(train_data, BOOK_REL)
    
    use_movies = e_movie > 0
    use_books = bool(config["loss"]["use_books_loss"]) and (e_book > 0)
    
    movie_loader = None
    if use_movies:
        movie_edge_index = train_data[MOVIE_REL].edge_index
        movie_loader = make_link_loader(
            train_data, MOVIE_REL, movie_edge_index,
            batch_size, num_neighbors, neg_ratio, shuffle=True
        )
    
    book_loader = None
    if use_books:
        book_edge_index = train_data[BOOK_REL].edge_index
        book_loader = make_link_loader(
            train_data, BOOK_REL, book_edge_index,
            batch_size, num_neighbors, neg_ratio, shuffle=True
        )
    
    train_data = train_data.to(device)
    
    # Training loop for final model
    epoch_losses = []
    for epoch in range(1, epochs + 1):
        avg_loss, avg_movie, avg_book = train_one_epoch(
            model, optimizer, movie_loader, book_loader,
            lambda_book, neg_ratio, use_movies, use_books, device
        )
        epoch_losses.append(avg_loss)
        
        if epoch % 5 == 0 or epoch == epochs:
            logger.info(
                f"  [Epoch {epoch:03d}] loss={avg_loss:.4f} | "
                f"movie={avg_movie:.4f} | book={avg_book:.4f}"
            )
    
    # Save final model
    model_save_path = output_dir / f"{study_name}_best_model.pt"
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"\nFinal model saved to: {model_save_path}")
    
    # Plot final training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), epoch_losses, 'b-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Final Model Training Loss', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    loss_plot_path = output_dir / f"{study_name}_final_loss.png"
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Final loss plot saved to: {loss_plot_path}")
    plt.close()
    
    logger.info("\n" + "="*80)
    logger.info("OPTIMIZATION PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    logger.info(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
