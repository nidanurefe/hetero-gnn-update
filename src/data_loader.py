from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
import torch
from torch_geometric.data import HeteroData

try:
    from logging_config import get_logger
except ImportError:
    from src.logging_config import get_logger

logger = get_logger(__name__)

RAW = Path("data/raw")
OUT = Path("data/processed")


def read_inter(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        skiprows=1,
        low_memory=False,
        names=["user_id", "item_id", "rating", "timestamp"]
    )
    
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["rating"] = df["rating"].astype(float)
    df["timestamp"] = df["timestamp"].astype(float)
    
    return df


class HeteroDataLoader:    
    def __init__(
        self,
        raw_dir: Path = RAW,
    ):

        self.raw_dir = raw_dir
        
        # File paths
        self.movies_train_path = raw_dir / "AmazonMovies.train.inter"
        self.movies_valid_path = raw_dir / "AmazonMovies.valid.inter"
        self.movies_test_path = raw_dir / "AmazonMovies.test.inter"
        self.books_train_path = raw_dir / "AmazonBooks.train.inter"
        
        # Check if all files exist
        self._check_files_exist()
        
        # Will be populated after loading
        self.user2idx: Optional[Dict[str, int]] = None
        self.movie2idx: Optional[Dict[str, int]] = None
        self.book2idx: Optional[Dict[str, int]] = None
        
        self.idx2user: Optional[Dict[int, str]] = None
        self.idx2movie: Optional[Dict[int, str]] = None
        self.idx2book: Optional[Dict[int, str]] = None
        
    def _check_files_exist(self):
        required_files = [
            self.movies_train_path,
            self.movies_valid_path,
            self.movies_test_path,
            self.books_train_path,
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        logger.info("All required data files found")
    
    def load_data(self) -> Tuple[HeteroData, HeteroData, HeteroData, Dict]:
        logger.info("Load graph data:")
        logger.info("=" * 70)
        
        # Load all raw data
        logger.info("\nLoading raw interaction files")
        movies_train_raw = read_inter(self.movies_train_path)
        movies_valid_raw = read_inter(self.movies_valid_path)
        movies_test_raw = read_inter(self.movies_test_path)
        books_train_raw = read_inter(self.books_train_path)
        
        logger.info(f"Movies train: {len(movies_train_raw):,} interactions")
        logger.info(f"Movies valid: {len(movies_valid_raw):,} interactions")
        logger.info(f"Movies test: {len(movies_test_raw):,} interactions")
        logger.info(f"Books train: {len(books_train_raw):,} interactions")
        
        movies_train = movies_train_raw.copy()
        movies_valid = movies_valid_raw.copy()
        movies_test = movies_test_raw.copy()
        books_train = books_train_raw.copy()
        
        # Build global mappings from ALL data (train, val, test)
        logger.info("\nBuilding global ID mappings from all data...")
        
        # Combine all data to get all entities
        all_users = set()
        all_users.update(movies_train["user_id"].unique())
        all_users.update(movies_valid["user_id"].unique())
        all_users.update(movies_test["user_id"].unique())
        all_users.update(books_train["user_id"].unique())
        
        all_movies = set(movies_train["item_id"].unique())
        all_movies.update(movies_valid["item_id"].unique())
        all_movies.update(movies_test["item_id"].unique())
        
        all_movies.update(movies_valid["item_id"].unique())
        all_movies.update(movies_test["item_id"].unique())
        
        all_books = set(books_train["item_id"].unique())
        
        # Create mappings
        self.user2idx = {uid: idx for idx, uid in enumerate(sorted(all_users))}
        self.movie2idx = {mid: idx for idx, mid in enumerate(sorted(all_movies))}
        self.book2idx = {bid: idx for idx, bid in enumerate(sorted(all_books))}
        
        # Reverse mappings
        self.idx2user = {idx: uid for uid, idx in self.user2idx.items()}
        self.idx2movie = {idx: mid for mid, idx in self.movie2idx.items()}
        self.idx2book = {idx: bid for bid, idx in self.book2idx.items()}
        
        num_users = len(self.user2idx)
        num_movies = len(self.movie2idx)
        num_books = len(self.book2idx)
        
        logger.info(f"  Total users:  {num_users:,}")
        logger.info(f"  Total movies: {num_movies:,}")
        logger.info(f"  Total books:  {num_books:,}")
                
        movies_valid_before = len(movies_valid)
        movies_test_before = len(movies_test)
        
        movies_valid = movies_valid[
            movies_valid["user_id"].isin(self.user2idx) &
            movies_valid["item_id"].isin(self.movie2idx)
        ].copy()
        
        movies_test = movies_test[
            movies_test["user_id"].isin(self.user2idx) &
            movies_test["item_id"].isin(self.movie2idx)
        ].copy()
        
        logger.info(f"  Movies valid: {len(movies_valid):,} ({100*len(movies_valid)/movies_valid_before:.1f}% kept)")
        logger.info(f"  Movies test:  {len(movies_test):,} ({100*len(movies_test)/movies_test_before:.1f}% kept)")
        
        # Apply entity ID mappings
        logger.info("\n[Step 4] Applying ID mappings...")
        
        movies_train["user_idx"] = movies_train["user_id"].map(self.user2idx)
        movies_train["item_idx"] = movies_train["item_id"].map(self.movie2idx)
        
        movies_valid["user_idx"] = movies_valid["user_id"].map(self.user2idx)
        movies_valid["item_idx"] = movies_valid["item_id"].map(self.movie2idx)
        
        movies_test["user_idx"] = movies_test["user_id"].map(self.user2idx)
        movies_test["item_idx"] = movies_test["item_id"].map(self.movie2idx)
        
        books_train["user_idx"] = books_train["user_id"].map(self.user2idx)
        books_train["item_idx"] = books_train["item_id"].map(self.book2idx)
        
        # Build HeteroData objects
        logger.info("\n[Step 5] Building HeteroData objects...")
        
        train_data = self._build_hetero_data(movies_train, books_train, num_users, num_movies, num_books)
        val_data = self._build_hetero_data(movies_valid, None, num_users, num_movies, num_books)
        test_data = self._build_hetero_data(movies_test, None, num_users, num_movies, num_books)
        
        logger.info(f"  Train graph: {train_data}")
        logger.info(f"  Val graph:   {val_data}")
        logger.info(f"  Test graph:  {test_data}")
        
        # Prepare metadata
        metadata = {
            "num_users": num_users,
            "num_movies": num_movies,
            "num_books": num_books,
            "user2idx": self.user2idx,
            "movie2idx": self.movie2idx,
            "book2idx": self.book2idx,
            "idx2user": self.idx2user,
            "idx2movie": self.idx2movie,
            "idx2book": self.idx2book,
            "train_stats": {
                "num_movie_edges": len(movies_train),
                "num_book_edges": len(books_train),
                "total_edges": len(movies_train) + len(books_train),
            },
            "val_stats": {
                "num_movie_edges": len(movies_valid),
            },
            "test_stats": {
                "num_movie_edges": len(movies_test),
            },
        }
        
        logger.info("Data loading completed:")
        
        return train_data, val_data, test_data, metadata
    
    def _build_hetero_data(
        self,
        movies_df: Optional[pd.DataFrame],
        books_df: Optional[pd.DataFrame],
        num_users: int,
        num_movies: int,
        num_books: int,
    ) -> HeteroData:
        data = HeteroData()
        
        # Set number of nodes
        data["user"].num_nodes = num_users
        data["movie"].num_nodes = num_movies
        data["book"].num_nodes = num_books
        
        # Add movie edges if available
        if movies_df is not None and len(movies_df) > 0:
            # Use numpy array directly for faster conversion
            import numpy as np
            movie_edges = np.stack([movies_df["user_idx"].values, movies_df["item_idx"].values], axis=0)
            movie_edge_index = torch.from_numpy(movie_edges).long()
            data["user", "rates_movie", "movie"].edge_index = movie_edge_index
            
            # Add reverse edges
            data["movie", "rev_rates_movie", "user"].edge_index = movie_edge_index.flip([0])
        else:
            # Empty edge tensors if no movie data
            data["user", "rates_movie", "movie"].edge_index = torch.empty((2, 0), dtype=torch.long)
            data["movie", "rev_rates_movie", "user"].edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Add book edges if available
        if books_df is not None and len(books_df) > 0:
            # Use numpy array directly for faster conversion
            import numpy as np
            book_edges = np.stack([books_df["user_idx"].values, books_df["item_idx"].values], axis=0)
            book_edge_index = torch.from_numpy(book_edges).long()
            data["user", "rates_book", "book"].edge_index = book_edge_index
            
            # Add reverse edges
            data["book", "rev_rates_book", "user"].edge_index = book_edge_index.flip([0])
        else:
            # Empty edge tensors if no book data
            data["user", "rates_book", "book"].edge_index = torch.empty((2, 0), dtype=torch.long)
            data["book", "rev_rates_book", "user"].edge_index = torch.empty((2, 0), dtype=torch.long)
        
        return data
    
    def save_metadata(self, metadata: Dict, output_path: Path = OUT / "hetero_metadata.json"):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a serializable version 
        serializable_metadata = {
            "num_users": metadata["num_users"],
            "num_movies": metadata["num_movies"],
            "num_books": metadata["num_books"],
            "train_stats": metadata["train_stats"],
            "val_stats": metadata["val_stats"],
            "test_stats": metadata["test_stats"],
        }
        
        with open(output_path, "w") as f:
            json.dump(serializable_metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {output_path}")
    
    def save_mappings(self, metadata: Dict, output_dir: Path = OUT):
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save user mappings
        user_mapping_path = output_dir / "user_mapping.json"
        with open(user_mapping_path, "w") as f:
            json.dump(metadata["user2idx"], f, indent=2)
        logger.info(f"User mapping saved to {user_mapping_path}")
        
        # Save movie mappings
        movie_mapping_path = output_dir / "movie_mapping.json"
        with open(movie_mapping_path, "w") as f:
            json.dump(metadata["movie2idx"], f, indent=2)
        logger.info(f"Movie mapping saved to {movie_mapping_path}")
        
        # Save book mappings
        book_mapping_path = output_dir / "book_mapping.json"
        with open(book_mapping_path, "w") as f:
            json.dump(metadata["book2idx"], f, indent=2)
        logger.info(f"Book mapping saved to {book_mapping_path}")


def main():
    logger.info("Starting data loading process...")
    
    # Initialize loader
    loader = HeteroDataLoader(
        raw_dir=RAW,
    )
    
    # Load data
    train_data, val_data, test_data, metadata = loader.load_data()
    
    # Save metadata and mappings
    loader.save_metadata(metadata)
    loader.save_mappings(metadata)
    
    # Save HeteroData objects
    torch.save(train_data, OUT / "hetero_train.pt")
    torch.save(val_data, OUT / "hetero_val.pt")
    torch.save(test_data, OUT / "hetero_test.pt")
    
    logger.info(f"\nAll data saved to {OUT}/")
    logger.info("  - hetero_train.pt")
    logger.info("  - hetero_val.pt")
    logger.info("  - hetero_test.pt")
    logger.info("  - hetero_metadata.json")
    logger.info("  - user_mapping.json")
    logger.info("  - movie_mapping.json")
    logger.info("  - book_mapping.json")


if __name__ == "__main__":
    main()
