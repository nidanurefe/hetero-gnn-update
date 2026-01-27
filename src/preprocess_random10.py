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
    # user_id, item_id, rating, timestamp -> Tab separated values, skip first row
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
    # Find overlap users 
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


def random_percent_split_per_user(movies: pd.DataFrame, train_percent: float = 0.1, seed: int = 42):
    np.random.seed(seed)
    
    movies = movies.copy()
    train_parts, val_parts, test_parts = [], [], []
    
    stats_train_counts = []  # Track train counts per user for logging
    
    for u, g in movies.groupby("user_id", sort=False):
        n_total = len(g)
        n_train = max(1, int(n_total * train_percent))  # at least 1 for train
        
        # Randomly select indices for train
        indices = np.arange(n_total)
        np.random.shuffle(indices)
        
        train_idx = indices[:n_train]
        rest_idx = indices[n_train:]
        
        g_train = g.iloc[train_idx]
        rest = g.iloc[rest_idx]
        
        train_parts.append(g_train)
        stats_train_counts.append(n_train)
        
        # Split rest into val/test
        n_rest = len(rest)
        if n_rest == 0:
            continue
        
        mid = n_rest // 2
        if mid == 0:
            # Only 1 left, put in test
            g_val = rest.iloc[:0]
            g_test = rest
        else:
            g_val = rest.iloc[:mid]
            g_test = rest.iloc[mid:]
        
        if len(g_val) > 0:
            val_parts.append(g_val)
        if len(g_test) > 0:
            test_parts.append(g_test)
    
    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else movies.iloc[:0].copy()
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else movies.iloc[:0].copy()
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else movies.iloc[:0].copy()
    
    # Log statistics
    logger.info(f"Random split statistics (train_percent={train_percent:.1%}):")
    logger.info(f"  Train interactions per user - min: {min(stats_train_counts)}, max: {max(stats_train_counts)}, mean: {np.mean(stats_train_counts):.2f}")
    logger.info(f"  Total splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
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
    logger.info("Starting preprocessing with RANDOM 10% TARGET DOMAIN SPLIT")
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
    train_percent = 0.10  # 10% of each user's interactions go to train
    logger.info("=" * 60)
    logger.info(f"SPLIT STRATEGY:")
    logger.info(f"  SOURCE domain (books): ALL {len(books_f):,} filtered interactions → TRAIN")
    logger.info(f"  TARGET domain (movies): RANDOM {train_percent:.0%} per user → TRAIN, rest → VAL/TEST")
    logger.info("=" * 60)
    
    m_train, m_val, m_test = random_percent_split_per_user(movies_f, train_percent=train_percent, seed=42)
    
    logger.info(
        f"Final overlap users: {books_f['user_id'].nunique():,} | "
        f"Books edges (all in train): {len(books_f):,} | "
        f"Movies split - Train: {len(m_train):,}, Val: {len(m_val):,}, Test: {len(m_test):,}"
    )

    # create ID mappings from filtered data
    logger.info("Creating ID mappings...")
    user2id, book2id, movie2id = make_id_maps(books_f, movies_f)
    logger.info(f"Users: {len(user2id):,}, Books: {len(book2id):,}, Movies: {len(movie2id):,}")

    # apply mappings
    logger.info("Applying ID mappings...")
    books_train = apply_maps(books_f, user2id, book2id)
    movies_train = apply_maps(m_train, user2id, movie2id)
    movies_val = apply_maps(m_val, user2id, movie2id)
    movies_test = apply_maps(m_test, user2id, movie2id)

    # sanity checks
    logger.info("=" * 60)
    logger.info("SANITY CHECKS:")
    if len(movies_train) > 0:
        per_user = movies_train.groupby("uid").size()
        logger.info(
            f"  [✓] Movies train per-user stats: "
            f"min={per_user.min()}, max={per_user.max()}, mean={per_user.mean():.2f}"
        )
    else:
        logger.warning("  [!] movies_train is empty!")

    tr_edges = set(zip(movies_train["uid"].tolist(), movies_train["iid"].tolist())) if len(movies_train) > 0 else set()
    te_edges = set(zip(movies_test["uid"].tolist(), movies_test["iid"].tolist()))
    overlap_edges = len(tr_edges & te_edges)
    if overlap_edges == 0:
        logger.info(f"  [✓] Train-test edge overlap: {overlap_edges} (GOOD)")
    else:
        logger.warning(f"  [!] Train-test edge overlap: {overlap_edges} (SHOULD BE 0!)")

    missing_users = set(movies_test["uid"].unique()) - set(movies_train["uid"].unique())
    if len(missing_users) == 0:
        logger.info(f"  [✓] Test users missing in train: {len(missing_users)} (GOOD)")
    else:
        logger.warning(f"  [!] Test users missing in train: {len(missing_users)} (SHOULD BE 0!)")
    logger.info("=" * 60)

    # save parquet + mappings
    logger.info("Saving parquet files...")
    books_train.to_parquet(OUT / "books_train.parquet", index=False)
    movies_train.to_parquet(OUT / "movies_train.parquet", index=False)
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
                    "type": "random_percent",
                    "train_percent": train_percent,
                    "seed": 42,
                    "min_user_interactions": 5,
                    "min_item_interactions": 5,
                    "description": "Random 10% of target domain interactions per user for training, source domain fully in training"
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
    logger.info(f"  Users: {len(user2id):,}")
    logger.info(f"  Books: {len(book2id):,}")
    logger.info(f"  Movies: {len(movie2id):,}")
    logger.info(f"")
    logger.info(f"Training Data:")
    logger.info(f"  Books (source): {len(books_train):,} interactions (100% of filtered)")
    logger.info(f"  Movies (target): {len(movies_train):,} interactions (~{train_percent:.0%} per user)")
    logger.info(f"")
    logger.info(f"Evaluation Data:")
    logger.info(f"  Movies Val: {len(movies_val):,}")
    logger.info(f"  Movies Test: {len(movies_test):,}")
    logger.info(f"")
    logger.info(f"Output location: {OUT}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
