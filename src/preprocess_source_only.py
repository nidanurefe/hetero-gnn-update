from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np

from logging_config import get_logger

logger = get_logger(__name__)

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

def read_inter(path: Path) -> pd.DataFrame:
    """Read interaction data from RecBole format."""
    df = pd.read_csv(path, sep="\t", header=None, skiprows=1, low_memory=False,
                     names=["user_id", "item_id", "rating", "timestamp"])
    
    # type casting
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["rating"] = df["rating"].astype(float)
    df["timestamp"] = df["timestamp"].astype(float)
    return df


def filter_overlap_users(
    books: pd.DataFrame,
    movies: pd.DataFrame,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
):

    # Find overlap users (users in both domains)
    ub = books["user_id"].value_counts()
    um = movies["user_id"].value_counts()
    overlap = ub.index.intersection(um.index)
    
    logger.info(f"Step 1 - Overlap users found: {len(overlap):,}")

    # Filter overlap users with minimum interactions in BOTH domains
    books_min_mask = ub.loc[overlap] >= min_user_interactions
    movies_min_mask = um.loc[overlap] >= min_user_interactions
    
    logger.info(f"Step 2a - Users with >= {min_user_interactions} book interactions: {books_min_mask.sum():,}")
    logger.info(f"Step 2b - Users with >= {min_user_interactions} movie interactions: {movies_min_mask.sum():,}")
    
    good_users = overlap[books_min_mask & movies_min_mask]
    good_users = set(good_users.tolist())
    
    logger.info(f"Step 2c - Users with >= {min_user_interactions} interactions in BOTH domains: {len(good_users):,}")
    logger.info(f"  → Removed: {len(overlap) - len(good_users):,} users ({100*(len(overlap) - len(good_users))/len(overlap):.2f}%)")

    # Filter books and movies dataframes to keep only good users
    books_f = books[books["user_id"].isin(good_users)].copy()
    movies_f = movies[movies["user_id"].isin(good_users)].copy()
    
    logger.info(f"Step 3 - After user filtering: {len(good_users):,} users")
    logger.info(f"  Books edges: {len(books_f):,}, Movies edges: {len(movies_f):,}")

    # Item filtering (min interactions per item)
    logger.info("Step 4 - Filtering items with minimum interactions...")
    book_counts = books_f["item_id"].value_counts()
    movie_counts = movies_f["item_id"].value_counts()
    
    num_books_before = books_f["item_id"].nunique()
    num_movies_before = movies_f["item_id"].nunique()
    logger.info(f"  Before item filtering - Books items: {num_books_before:,}, Movies items: {num_movies_before:,}")

    # Filter items with enough interactions
    good_books = set(book_counts[book_counts >= min_item_interactions].index)
    good_movies = set(movie_counts[movie_counts >= min_item_interactions].index)
    
    logger.info(f"  Items with >= {min_item_interactions} interactions - Books: {len(good_books):,}, Movies: {len(good_movies):,}")
    logger.info(f"  → Removed - Books: {num_books_before - len(good_books):,} items, Movies: {num_movies_before - len(good_movies):,} items")

    # Apply item filtering
    books_f_before = len(books_f)
    movies_f_before = len(movies_f)
    books_f = books_f[books_f["item_id"].isin(good_books)].copy()
    movies_f = movies_f[movies_f["item_id"].isin(good_movies)].copy()
    
    logger.info(f"  After item filtering - Books edges: {len(books_f):,} (removed {books_f_before - len(books_f):,})")
    logger.info(f"  After item filtering - Movies edges: {len(movies_f):,} (removed {movies_f_before - len(movies_f):,})")
    logger.info(f"  After item filtering - Books items: {books_f['item_id'].nunique():,}, Movies items: {movies_f['item_id'].nunique():,}")

    final_users = books_f['user_id'].nunique()
    logger.info("=" * 60)
    logger.info(f"Step 5 - FINAL RESULTS:")
    logger.info(f"  Final overlap users: {final_users:,}")
    logger.info(f"  Books edges: {len(books_f):,}")
    logger.info(f"  Movies edges: {len(movies_f):,}")
    logger.info(f"  Total reduction: {len(overlap):,} → {final_users:,} users ({100*(len(overlap) - final_users)/len(overlap):.2f}% removed)")
    logger.info("=" * 60)

    return books_f, movies_f


def source_only_split(movies: pd.DataFrame, seed: int = 42):

    np.random.seed(seed)
    
    movies = movies.copy()
    val_parts, test_parts = [], []
    
    logger.info("SOURCE-ONLY SPLIT: All target domain interactions go to val/test")
    
    for u, g in movies.groupby("user_id", sort=False):
        n_total = len(g)
        
        # Randomly shuffle interactions
        indices = np.arange(n_total)
        np.random.shuffle(indices)
        
        # Split 50-50 into val and test
        mid = n_total // 2
        
        if mid == 0:
            # Only 1 interaction, put in test
            val_idx = indices[:0]
            test_idx = indices
        else:
            val_idx = indices[:mid]
            test_idx = indices[mid:]
        
        g_val = g.iloc[val_idx]
        g_test = g.iloc[test_idx]
        
        if len(g_val) > 0:
            val_parts.append(g_val)
        if len(g_test) > 0:
            test_parts.append(g_test)
    
    # Empty train DataFrame with same structure
    train_df = movies.iloc[:0].copy()
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else movies.iloc[:0].copy()
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else movies.iloc[:0].copy()
    
    # Log statistics
    logger.info(f"Source-only split results:")
    logger.info(f"  Train: {len(train_df)} (0% - SOURCE-ONLY)")
    logger.info(f"  Val:   {len(val_df)} (~50%)")
    logger.info(f"  Test:  {len(test_df)} (~50%)")
    
    return train_df, val_df, test_df


def make_id_maps(books_inter: pd.DataFrame, movies_inter: pd.DataFrame):
    # global user mapping 
    users = pd.Index(sorted(set(books_inter["user_id"]).union(movies_inter["user_id"])))
    user2id = {u: i for i, u in enumerate(users)}

    # domain-specific item mapping
    book_items = pd.Index(sorted(books_inter["item_id"].unique()))
    movie_items = pd.Index(sorted(movies_inter["item_id"].unique()))
    book2id = {it: i for i, it in enumerate(book_items)}
    movie2id = {it: i for i, it in enumerate(movie_items)}

    return user2id, book2id, movie2id


def apply_maps(df: pd.DataFrame, user2id: dict, item2id: dict, item_col="item_id"):
    """Apply ID mappings to dataframe."""
    out = df.copy()
    out["uid"] = out["user_id"].map(user2id).astype(int)
    out["iid"] = out[item_col].map(item2id).astype(int)
    return out


def main():
    logger.info("=" * 80)
    logger.info("SOURCE-ONLY TRAINING PREPROCESSING")
    logger.info("Target domain (movies) will NOT be used for training")
    logger.info("=" * 80)
    logger.info("Reading interaction data...")

    books_inter = read_inter(RAW / "AmazonBooks/AmazonBooks.inter")
    logger.info(f"Books interactions: {len(books_inter):,} records")

    movies_inter = read_inter(RAW / "AmazonMov/AmazonMov.inter")
    logger.info(f"Movies interactions: {len(movies_inter):,} records")

    # Log initial user counts (before any filtering)
    num_books_users = books_inter["user_id"].nunique()
    num_movies_users = movies_inter["user_id"].nunique()
    overlap_users_initial = set(books_inter["user_id"]).intersection(set(movies_inter["user_id"]))
    logger.info("=" * 60)
    logger.info("INITIAL USER COUNTS (before any filtering):")
    logger.info(f"  Books domain users: {num_books_users:,}")
    logger.info(f"  Movies domain users: {num_movies_users:,}")
    logger.info(f"  Overlap users (in both domains): {len(overlap_users_initial):,}")
    logger.info("=" * 60)

    # Overlap users + min-5 interactions in BOTH domains + item filtering
    logger.info("Filtering overlap users (min-5 interactions in both domains)...")
    books_f, movies_f = filter_overlap_users(
        books_inter, movies_inter,
        min_user_interactions=5,
        min_item_interactions=5
    )
    logger.info(
        f"After overlap filtering - users: {books_f['user_id'].nunique():,}, "
        f"books edges: {len(books_f):,}, movies edges: {len(movies_f):,}"
    )
    logger.info("=" * 60)
    logger.info(f"SPLIT STRATEGY (SOURCE-ONLY):")
    logger.info(f"  SOURCE domain (books): ALL {len(books_f):,} interactions → TRAIN")
    logger.info(f"  TARGET domain (movies): NONE → TRAIN (0 interactions)")
    logger.info(f"  TARGET domain (movies): ALL → VAL/TEST (split 50-50)")
    logger.info("=" * 60)
    
    m_train, m_val, m_test = source_only_split(movies_f, seed=42)
    
    logger.info(
        f"Final overlap users: {books_f['user_id'].nunique():,} | "
        f"Books edges (all in train): {len(books_f):,} | "
        f"Movies split - Train: {len(m_train):,}, Val: {len(m_val):,}, Test: {len(m_test):,}"
    )

    # Verify that movie train is empty
    assert len(m_train) == 0, "ERROR: Movie train should be empty for source-only training!"
    logger.info("✓ Verified: Movie training set is empty (source-only)")

    # create ID mappings from filtered data
    logger.info("Creating ID mappings...")
    user2id, book2id, movie2id = make_id_maps(books_f, movies_f)
    logger.info(f"Users: {len(user2id):,}, Books: {len(book2id):,}, Movies: {len(movie2id):,}")

    # apply mappings
    logger.info("Applying ID mappings...")
    books_train = apply_maps(books_f, user2id, book2id)
    movies_train = apply_maps(m_train, user2id, movie2id)  # Will be empty
    movies_val = apply_maps(m_val, user2id, movie2id)
    movies_test = apply_maps(m_test, user2id, movie2id)

    # sanity checks
    logger.info("=" * 60)
    logger.info("SANITY CHECKS:")
    
    # Check movie train is empty
    if len(movies_train) == 0:
        logger.info(f"  [✓] Movies train is EMPTY (source-only mode)")
    else:
        logger.error(f"  [✗] Movies train is NOT empty: {len(movies_train)} (SHOULD BE 0!)")
    
    # Check all users in test are also in books_train
    test_users = set(movies_test["uid"].unique())
    train_users = set(books_train["uid"].unique())
    missing_users = test_users - train_users
    
    if len(missing_users) == 0:
        logger.info(f"  [✓] All test users exist in books training set")
    else:
        logger.warning(f"  [!] {len(missing_users)} test users missing in books training (should be 0)")
    
    # Val/test split check
    val_users = len(movies_val["uid"].unique())
    test_users_count = len(movies_test["uid"].unique())
    logger.info(f"  [✓] Val users: {val_users}, Test users: {test_users_count}")
    
    logger.info("=" * 60)

    # save parquet + mappings
    logger.info("Saving parquet files...")
    books_train.to_parquet(OUT / "books_train.parquet", index=False)
    movies_train.to_parquet(OUT / "movies_train.parquet", index=False)  # Empty file
    movies_val.to_parquet(OUT / "movies_val.parquet", index=False)
    movies_test.to_parquet(OUT / "movies_test.parquet", index=False)
    logger.info(f"  ✓ Saved to {OUT}/")

    logger.info("Saving mappings...")
    with open(OUT / "mappings.json", "w") as f:
        json.dump(
            {
                "num_users": len(user2id),
                "num_books": len(book2id),
                "num_movies": len(movie2id),
                "user2id": user2id,
                "book2id": book2id,
                "movie2id": movie2id,
                "split": {
                    "type": "source_only",
                    "train_percent_target": 0.0,
                    "seed": 42,
                    "min_user_interactions": 5,
                    "min_item_interactions": 5,
                    "description": "Source-only training: All source (books) in train, NO target (movies) in train"
                },
            },
            f,
        )
    logger.info(f"  ✓ Saved mappings.json")

    # Final summary
    logger.info("=" * 80)
    logger.info("PREPROCESSING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info(f"Dataset Statistics:")
    logger.info(f"  Users: {len(user2id):,} (overlap users only)")
    logger.info(f"  Books: {len(book2id):,}")
    logger.info(f"  Movies: {len(movie2id):,}")
    logger.info(f"")
    logger.info(f"Training Data (SOURCE-ONLY):")
    logger.info(f"  Books (source): {len(books_train):,} interactions (100% of filtered)")
    logger.info(f"  Movies (target): {len(movies_train):,} interactions (0% - NONE)")
    logger.info(f"")
    logger.info(f"Evaluation Data:")
    logger.info(f"  Movies Val:  {len(movies_val):,}")
    logger.info(f"  Movies Test: {len(movies_test):,}")
    logger.info(f"")
    logger.info(f"Training Strategy:")
    logger.info(f"  - Model will ONLY see source domain (books) during training")
    logger.info(f"  - Movie embeddings will be learned through transfer/GNN propagation")
    logger.info(f"  - Pure cross-domain transfer learning scenario")
    logger.info(f"")
    logger.info(f"Output location: {OUT}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
