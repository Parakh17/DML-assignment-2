"""
Data Preparation Module.
Deduplicates records, standardizes formats, and performs a strictly chronological
(time-based) train/test split to avoid lookahead bias.
Outputs processed datasets into data/processed/ in Parquet format.
"""

from datetime import datetime
from pathlib import Path
import pandas as pd


def prepare_data(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    test_ratio: float = 0.20
) -> dict:
    """
    Cleans raw data and performs a time-based train/test split on interaction logs.
    """
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    ratings_path = raw_path / "ratings.csv"
    items_path = raw_path / "items.csv"
    users_path = raw_path / "users.csv"

    if not (ratings_path.exists() and items_path.exists() and users_path.exists()):
        raise FileNotFoundError("Raw datasets missing. Run ingestion first.")

    df_ratings = pd.read_csv(ratings_path)
    df_items = pd.read_csv(items_path)
    df_users = pd.read_csv(users_path)

    print(f"[Preparation] Raw records loaded: {len(df_ratings):,} ratings, {len(df_items):,} items, {len(df_users):,} users.")

    # 1. Clean and deduplicate ratings
    # If a user rated the same item multiple times, sort by timestamp and keep the latest rating
    df_ratings.sort_values(by=["timestamp"], inplace=True)
    initial_len = len(df_ratings)
    df_ratings.drop_duplicates(subset=["user_id", "item_id"], keep="last", inplace=True)
    deduped_count = initial_len - len(df_ratings)
    print(f"[Preparation] Deduplication: Removed {deduped_count} duplicate user-item interactions (kept latest).")

    # 2. Time-Based Split (Global Cutoff)
    # Calculate cutoff timestamp at (1 - test_ratio) percentile
    cutoff_idx = int(len(df_ratings) * (1 - test_ratio))
    split_timestamp = df_ratings.iloc[cutoff_idx]["timestamp"]
    split_date_str = datetime.utcfromtimestamp(split_timestamp).strftime('%Y-%m-%d %H:%M:%S UTC')

    df_train = df_ratings[df_ratings["timestamp"] <= split_timestamp].copy()
    df_test = df_ratings[df_ratings["timestamp"] > split_timestamp].copy()

    print(f"[Preparation] Time-based split applied at timestamp {split_timestamp} ({split_date_str}):")
    print(f"  - Train Set: {len(df_train):,} interactions ({len(df_train)/len(df_ratings)*100:.1f}%)")
    print(f"  - Test Set:  {len(df_test):,} interactions ({len(df_test)/len(df_ratings)*100:.1f}%)")

    # 3. Standardize and Save as Parquet
    train_file = processed_path / "train_ratings.parquet"
    test_file = processed_path / "test_ratings.parquet"
    items_file = processed_path / "items.parquet"
    users_file = processed_path / "users.parquet"

    df_train.to_parquet(train_file, index=False)
    df_test.to_parquet(test_file, index=False)
    df_items.to_parquet(items_file, index=False)
    df_users.to_parquet(users_file, index=False)

    print(f"[Preparation] Processed files saved to {processed_path}:")
    print(f"  - {train_file.name}")
    print(f"  - {test_file.name}")
    print(f"  - {items_file.name}")
    print(f"  - {users_file.name}")

    return {
        "train_ratings": str(train_file),
        "test_ratings": str(test_file),
        "items": str(items_file),
        "users": str(users_file),
        "split_timestamp": int(split_timestamp),
        "split_date": split_date_str
    }


if __name__ == "__main__":
    prepare_data()

