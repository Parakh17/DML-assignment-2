"""
Data Transformation Module.
Builds derived datasets from training interactions:
1. User Activity Summary (activity counts, rating means/stds, temporal activity span)
2. Item Popularity Summary (interaction counts, average ratings, popularity rank)
3. User-Item Interaction Matrix (sparse scipy CSR matrix and mapping arrays)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz


def transform_data(
    processed_dir: str = "data/processed"
) -> dict:
    """
    Computes derived summary datasets and interaction matrix from processed train data.
    """
    processed_path = Path(processed_dir)
    train_ratings_file = processed_path / "train_ratings.parquet"
    items_file = processed_path / "items.parquet"

    if not train_ratings_file.exists():
        raise FileNotFoundError("train_ratings.parquet not found. Run preparation first.")

    df_train = pd.read_parquet(train_ratings_file)
    df_items = pd.read_parquet(items_file) if items_file.exists() else None

    print(f"[Transformation] Computing derived tables from {len(df_train):,} training interactions...")

    # 1. User Activity Summary
    user_summary = df_train.groupby("user_id").agg(
        interaction_count=("rating", "count"),
        avg_rating=("rating", "mean"),
        std_rating=("rating", "std"),
        min_timestamp=("timestamp", "min"),
        max_timestamp=("timestamp", "max")
    ).reset_index()

    user_summary["std_rating"] = user_summary["std_rating"].fillna(0.0)
    user_summary["active_span_days"] = (
        (user_summary["max_timestamp"] - user_summary["min_timestamp"]) / (24 * 3600)
    ).round(2)

    user_summary_file = processed_path / "user_activity_summary.parquet"
    user_summary.to_parquet(user_summary_file, index=False)
    print(f"  ✅ Created User Activity Summary -> {user_summary_file} ({len(user_summary)} users)")

    # 2. Item Popularity Summary
    item_summary = df_train.groupby("item_id").agg(
        interaction_count=("rating", "count"),
        avg_rating=("rating", "mean")
    ).reset_index()

    item_summary.sort_values(by=["interaction_count", "avg_rating"], ascending=[False, False], inplace=True)
    item_summary["popularity_rank"] = np.arange(1, len(item_summary) + 1)

    if df_items is not None:
        item_summary = item_summary.merge(
            df_items[["item_id", "title"]],
            on="item_id",
            how="left"
        )

    item_summary_file = processed_path / "item_popularity_summary.parquet"
    item_summary.to_parquet(item_summary_file, index=False)
    print(f"  ✅ Created Item Popularity Summary -> {item_summary_file} ({len(item_summary)} items)")

    # 3. User-Item Interaction Matrix (Sparse CSR)
    # Map raw user_id and item_id to contiguous 0-indexed matrix indices
    unique_users = np.sort(df_train["user_id"].unique())
    unique_items = np.sort(df_train["item_id"].unique())

    user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    item_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}

    row_indices = df_train["user_id"].map(user_to_idx).values
    col_indices = df_train["item_id"].map(item_to_idx).values
    ratings_vals = df_train["rating"].values

    matrix_shape = (len(unique_users), len(unique_items))
    interaction_matrix = csr_matrix((ratings_vals, (row_indices, col_indices)), shape=matrix_shape)

    matrix_file = processed_path / "interaction_matrix.npz"
    user_map_file = processed_path / "user_ids.npy"
    item_map_file = processed_path / "item_ids.npy"

    save_npz(matrix_file, interaction_matrix)
    np.save(user_map_file, unique_users)
    np.save(item_map_file, unique_items)

    sparsity = 1.0 - (interaction_matrix.nnz / (matrix_shape[0] * matrix_shape[1]))
    print(f"  ✅ Created Interaction Matrix -> {matrix_file} (Shape: {matrix_shape}, Sparsity: {sparsity*100:.2f}%)")

    return {
        "user_summary": str(user_summary_file),
        "item_summary": str(item_summary_file),
        "interaction_matrix": str(matrix_file),
        "matrix_shape": matrix_shape,
        "sparsity": float(sparsity)
    }


if __name__ == "__main__":
    transform_data()

