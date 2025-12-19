from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

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


# Will be used later
def read_item(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, skiprows=1, low_memory=False,
                     names=["item_id","categories","title","price","sales_type","sales_rank","brand"])
    df["item_id"] = df["item_id"].astype(str)
    return df


# Filter overlap users with item filtering and reapply user min-k
# k_src: min interactions per user in source domain (books)
# k_tgt: min interactions per user in target domain (movies)
# min_item_interactions: min interactions per item in both domains

def filter_overlap_users(
    books: pd.DataFrame,
    movies: pd.DataFrame,
    k_src: int = 20,
    k_tgt: int = 20,
    min_item_interactions: int = 5,
):
    # initial overlap + user min-k
    ub = books["user_id"].value_counts() # user interaction counts in books
    um = movies["user_id"].value_counts() # user interaction counts in movies
    overlap = ub.index.intersection(um.index) # users in both domains

    # Filter users with minimum interactions in both domains
    good_users = overlap[(ub.loc[overlap] >= k_src) & (um.loc[overlap] >= k_tgt)]
    good_users = set(good_users.tolist())

    # Filter books and movies dataframes to keep only good users
    books_f = books[books["user_id"].isin(good_users)].copy()
    movies_f = movies[movies["user_id"].isin(good_users)].copy()

    # item filtering (min interactions per item) 
    logger.info("Filtering items with minimum interactions...")
    book_counts = books_f["item_id"].value_counts() # book interaction counts in filtered books
    movie_counts = movies_f["item_id"].value_counts() # movie interaction counts in filtered movies

    # Filter items with enough interactions
    good_books = set(book_counts[book_counts >= min_item_interactions].index)
    good_movies = set(movie_counts[movie_counts >= min_item_interactions].index)

    # Apply item filtering
    books_f = books_f[books_f["item_id"].isin(good_books)].copy()
    movies_f = movies_f[movies_f["item_id"].isin(good_movies)].copy()

    logger.info(
        f"After item filtering - Books items: {books_f['item_id'].nunique()}, Movies items: {movies_f['item_id'].nunique()}"
    )

    # reapply user min-k after item filtering (interactions may have dropped below k due to item filtering)
    logger.info("Re-applying user min-k after item filtering...")
    ub2 = books_f["user_id"].value_counts() # user interaction counts in filtered books
    um2 = movies_f["user_id"].value_counts() # user interaction counts in filtered movies

    overlap2 = ub2.index.intersection(um2.index)
    good_users2 = overlap2[(ub2.loc[overlap2] >= k_src) & (um2.loc[overlap2] >= k_tgt)]
    good_users2 = set(good_users2.tolist())

    books_f = books_f[books_f["user_id"].isin(good_users2)].copy()
    movies_f = movies_f[movies_f["user_id"].isin(good_users2)].copy()

    logger.info(
        f"After re-applying user min-k - Overlap users: {len(good_users2)}, "
        f"Books edges: {len(books_f)}, Movies edges: {len(movies_f)}"
    )

    return books_f, movies_f



# For each user, train only the first k interactions in the target domain,
# and separate the remaining interactions into val/test in chronological order.
# k_shot_train: number of interactions per user to keep in train set
def cold_start_split_per_user(movies: pd.DataFrame, k_shot_train: int = 1):
    movies = movies.sort_values(["user_id", "timestamp"]).copy() # sort by user and timestamp
    # Previously watched movies -> train
    # Remaining movies -> val/test split

    train_parts, val_parts, test_parts = [], [], [] # lists to hold split parts

    # each user and its interactions
    for u, g in movies.groupby("user_id", sort=False):
        n = len(g) 

        # ensure we can do train + val + test
        if n <= k_shot_train + 1:
            # too few interactions; skip user (or keep only train)
            continue

        g_train = g.iloc[:k_shot_train] # first k interactions for train
        rest = g.iloc[k_shot_train:] # remaining interactions for val/test

        # split rest into val/test
        mid = len(rest) // 2

        if mid == 0:
            # if only 1 left, put it into test
            g_val = rest.iloc[:0]
            g_test = rest
        else:
            g_val = rest.iloc[:mid]
            g_test = rest.iloc[mid:]

        train_parts.append(g_train)
        if len(g_val) > 0:
            val_parts.append(g_val)
        test_parts.append(g_test)

    if not train_parts or not test_parts:
        raise RuntimeError("Cold-start split produced empty train/test. Check k_shot_train and data.")

    train_df = pd.concat(train_parts, ignore_index=True) 
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else movies.iloc[:0].copy()
    test_df = pd.concat(test_parts, ignore_index=True)

    return train_df, val_df, test_df


def make_id_maps(books_inter: pd.DataFrame, movies_inter: pd.DataFrame):
    # global user mapping 
    users = pd.Index(sorted(set(books_inter["user_id"]).union(movies_inter["user_id"])))
    user2id = {u:i for i,u in enumerate(users)}

    # domain-specific item mapping
    book_items  = pd.Index(sorted(books_inter["item_id"].unique()))
    movie_items = pd.Index(sorted(movies_inter["item_id"].unique()))
    book2id  = {it:i for i,it in enumerate(book_items)}
    movie2id = {it:i for i,it in enumerate(movie_items)}

    return user2id, book2id, movie2id

# Apply ID mappings to dataframe
def apply_maps(df: pd.DataFrame, user2id: dict, item2id: dict, item_col="item_id"):
    out = df.copy()
    out["uid"] = out["user_id"].map(user2id).astype(int)
    out["iid"] = out[item_col].map(item2id).astype(int)
    return out


def sanity_checks():
    tr = pd.read_parquet(OUT/"movies_train.parquet")
    va = pd.read_parquet(OUT/"movies_val.parquet")
    te = pd.read_parquet(OUT/"movies_test.parquet")

    # 1-shot check
    per_user = tr.groupby("uid").size()
    logger.info(f"[check] movies_train interactions per user: min={per_user.min()}, max={per_user.max()}, mean={per_user.mean():.3f}")

    # train-test edge overlap check
    tr_edges = set(zip(tr["uid"].tolist(), tr["iid"].tolist()))
    te_edges = set(zip(te["uid"].tolist(), te["iid"].tolist()))
    overlap = tr_edges & te_edges
    logger.info(f"[check] train-test edge overlap: {len(overlap)} (should be 0)")

    # test users in train check
    missing_users = set(te["uid"].unique()) - set(tr["uid"].unique())
    logger.info(f"[check] test users missing in train: {len(missing_users)} (should be 0)")


def calculate_sparsity(df_books, df_movies, num_users, num_books, num_movies):
    n_interactions = len(df_books) + len(df_movies)
    
    total_possible = (num_users * num_books) + (num_users * num_movies)
    
    sparsity = 1.0 - (n_interactions / total_possible)
    density = (n_interactions / total_possible) * 100
    
    logger.info("=== Data Sparsity Statistics ===")
    logger.info(f"Total Users: {num_users}")
    logger.info(f"Total Items: {num_books + num_movies} (Books: {num_books}, Movies: {num_movies})")
    logger.info(f"Total Interactions: {n_interactions}")
    logger.info(f"Sparsity: {sparsity:.6f} ({sparsity*100:.4f}%)")
    logger.info(f"Density:  {density:.6f}%")
    
    avg_book = len(df_books) / num_users
    avg_movie = len(df_movies) / num_users
    logger.info(f"Avg Books per User: {avg_book:.2f}")
    logger.info(f"Avg Movies per User: {avg_movie:.2f}")



def main():
    logger.info("Starting preprocessing...")
    logger.info("Reading interaction data...")

    books_inter = read_inter(RAW / "AmazonBooks/AmazonBooks.inter")
    logger.info(f"Books interactions: {len(books_inter)} records")

    movies_inter = read_inter(RAW / "AmazonMov/AmazonMov.inter")
    logger.info(f"Movies interactions: {len(movies_inter)} records")

    # overlap + item filtering + re-apply user min-k 
    logger.info("Filtering overlap users (with item filtering + re-apply min-k)...")
    k_src, k_tgt = 5, 5
    books_f, movies_f = filter_overlap_users(
        books_inter, movies_inter,
        k_src=k_src, k_tgt=k_tgt,
        min_item_interactions=5
    )
    logger.info(
        f"After overlap filtering - users: {books_f['user_id'].nunique()}, "
        f"books edges: {len(books_f)}, movies edges: {len(movies_f)}"
    )

    # cold-start split on TARGET (movies): 1-shot train
    logger.info("Cold-start splitting movies (1-shot train)...")
    m_train, m_val, m_test = cold_start_split_per_user(movies_f, k_shot_train=1)

    # keep only users that survived the cold-start split
    logger.info("Filtering books/movies to users that survived cold-start split...")
    survived_users = set(m_train["user_id"]) & set(m_test["user_id"])


    books_f = books_f[books_f["user_id"].isin(survived_users)].copy()
    movies_f = movies_f[movies_f["user_id"].isin(survived_users)].copy()

    m_train = m_train[m_train["user_id"].isin(survived_users)].copy()
    m_val   = m_val[m_val["user_id"].isin(survived_users)].copy()
    m_test  = m_test[m_test["user_id"].isin(survived_users)].copy()

    logger.info(
        f"Survived users: {len(survived_users)} | "
        f"Books edges: {len(books_f)} | Movies edges (all): {len(movies_f)} | "
        f"Movies split sizes: train={len(m_train)}, val={len(m_val)}, test={len(m_test)}"
    )

    # create ID mappings from filtered data
    logger.info("Creating ID mappings...")
    user2id, book2id, movie2id = make_id_maps(books_f, movies_f)
    logger.info(f"Users: {len(user2id)}, Books: {len(book2id)}, Movies: {len(movie2id)}")

    # apply mappings
    logger.info("Applying ID mappings...")
    books_train  = apply_maps(books_f, user2id, book2id)
    movies_train = apply_maps(m_train, user2id, movie2id)
    movies_val   = apply_maps(m_val, user2id, movie2id)
    movies_test  = apply_maps(m_test, user2id, movie2id)

    # sanity checks
    logger.info("[check] Verifying cold-start + leakage...")
    per_user = movies_train.groupby("uid").size()
    logger.info(
        f"[check] movies_train per-user: min={per_user.min()}, "
        f"max={per_user.max()}, mean={per_user.mean():.3f}"
    )

    tr_edges = set(zip(movies_train["uid"].tolist(), movies_train["iid"].tolist()))
    te_edges = set(zip(movies_test["uid"].tolist(), movies_test["iid"].tolist()))
    logger.info(f"[check] train-test edge overlap: {len(tr_edges & te_edges)} (should be 0)")

    missing_users = set(movies_test["uid"].unique()) - set(movies_train["uid"].unique())
    logger.info(f"[check] test users missing in train: {len(missing_users)} (should be 0)")

    # save parquet + mappings
    logger.info("Saving parquet files...")
    books_train.to_parquet(OUT / "books_train.parquet", index=False)
    movies_train.to_parquet(OUT / "movies_train.parquet", index=False)
    movies_val.to_parquet(OUT / "movies_val.parquet", index=False)
    movies_test.to_parquet(OUT / "movies_test.parquet", index=False)

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
                    "type": "cold_start",
                    "k_shot_train": 1,
                    "k_src": k_src,
                    "k_tgt": k_tgt,
                    "min_item_interactions": 5,
                },
            },
            f,
        )

    all_movies = pd.concat([movies_train, movies_val, movies_test])
    
    # calculate sparsity and log
    calculate_sparsity(
        books_train, 
        all_movies, 
        len(user2id), 
        len(book2id), 
        len(movie2id)
    )

    logger.info("Done.")
    logger.info(f"Users: {len(user2id)}, Books: {len(book2id)}, Movies: {len(movie2id)}")
    logger.info(f"Movies split sizes - Train: {len(movies_train)}, Val: {len(movies_val)}, Test: {len(movies_test)}")
    logger.info(f"Output saved to {OUT}")


if __name__ == "__main__":
    main()