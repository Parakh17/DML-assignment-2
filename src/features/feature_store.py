"""
Feature Management & Feature Store Module.
Generates reusable offline features for Users, Items, and User-Item Interactions.
Outputs feature tables to data/features/ in Parquet format and creates an automated
Feature Catalog table in reports/feature_catalog.md.
"""

from pathlib import Path
import pandas as pd
import numpy as np


GENRE_COLS = [
    "unknown", "Action", "Adventure", "Animation", "Childrens", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film_Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci_Fi", "Thriller", "War", "Western"
]


def build_features(
    processed_dir: str = "data/processed",
    features_dir: str = "data/features",
    reports_dir: str = "reports"
) -> dict:
    """
    Computes User, Item, and Interaction feature matrices and writes the Feature Catalog.
    """
    processed_path = Path(processed_dir)
    features_path = Path(features_dir)
    reports_path = Path(reports_dir)

    features_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_parquet(processed_path / "train_ratings.parquet")
    df_items = pd.read_parquet(processed_path / "items.parquet")
    df_users = pd.read_parquet(processed_path / "users.parquet")

    # Reference timestamp for recency calculation (max timestamp in train split)
    reference_timestamp = df_train["timestamp"].max()

    print(f"[Features] Building feature store tables using reference timestamp {reference_timestamp}...")

    # -------------------------------------------------------------
    # 1. User Features
    # -------------------------------------------------------------
    # User aggregates from interactions
    user_agg = df_train.groupby("user_id").agg(
        user_rating_count=("rating", "count"),
        user_avg_rating=("rating", "mean"),
        user_rating_std=("rating", "std"),
        user_last_timestamp=("timestamp", "max")
    ).reset_index()

    user_agg["user_rating_std"] = user_agg["user_rating_std"].fillna(0.0).round(3)
    user_agg["user_avg_rating"] = user_agg["user_avg_rating"].round(3)
    user_agg["days_since_last_interaction"] = (
        (reference_timestamp - user_agg["user_last_timestamp"]) / (24 * 3600)
    ).round(2)
    user_agg.drop(columns=["user_last_timestamp"], inplace=True)

    # Compute preferred genre per user
    df_train_with_genres = df_train.merge(df_items[["item_id"] + [g for g in GENRE_COLS if g in df_items.columns]], on="item_id", how="left")
    available_genres = [g for g in GENRE_COLS if g in df_items.columns]

    user_genre_counts = {}
    for g in available_genres:
        user_genre_counts[g] = df_train_with_genres.groupby("user_id")[g].sum()

    df_user_genres = pd.DataFrame(user_genre_counts)
    user_preferred_genre = df_user_genres.idxmax(axis=1).reset_index()
    user_preferred_genre.columns = ["user_id", "user_preferred_genre"]

    # Merge demographics from users table
    df_user_features = df_users.merge(user_agg, on="user_id", how="left")
    df_user_features = df_user_features.merge(user_preferred_genre, on="user_id", how="left")

    # Impute for cold users (if any exist without train interactions)
    global_avg_rating = df_train["rating"].mean()
    df_user_features["user_rating_count"] = df_user_features["user_rating_count"].fillna(0).astype(int)
    df_user_features["user_avg_rating"] = df_user_features["user_avg_rating"].fillna(global_avg_rating).round(3)
    df_user_features["user_rating_std"] = df_user_features["user_rating_std"].fillna(0.0)
    df_user_features["days_since_last_interaction"] = df_user_features["days_since_last_interaction"].fillna(999.0)
    df_user_features["user_preferred_genre"] = df_user_features["user_preferred_genre"].fillna("unknown")

    # Encode categorical fields
    df_user_features["gender_code"] = (df_user_features["gender"] == "M").astype(int)

    user_features_file = features_path / "user_features.parquet"
    df_user_features.to_parquet(user_features_file, index=False)
    print(f"  ✅ Built User Features -> {user_features_file} ({df_user_features.shape})")

    # -------------------------------------------------------------
    # 2. Item Features
    # -------------------------------------------------------------
    item_agg = df_train.groupby("item_id").agg(
        item_rating_count=("rating", "count"),
        item_avg_rating=("rating", "mean")
    ).reset_index()

    item_agg["item_avg_rating"] = item_agg["item_avg_rating"].round(3)
    item_agg.sort_values(by=["item_rating_count", "item_avg_rating"], ascending=[False, False], inplace=True)
    item_agg["item_popularity_rank"] = np.arange(1, len(item_agg) + 1)

    df_item_features = df_items.merge(item_agg, on="item_id", how="left")
    df_item_features["item_rating_count"] = df_item_features["item_rating_count"].fillna(0).astype(int)
    df_item_features["item_avg_rating"] = df_item_features["item_avg_rating"].fillna(global_avg_rating).round(3)
    max_rank = df_item_features["item_popularity_rank"].max()
    df_item_features["item_popularity_rank"] = df_item_features["item_popularity_rank"].fillna(max_rank + 1).astype(int)

    item_features_file = features_path / "item_features.parquet"
    df_item_features.to_parquet(item_features_file, index=False)
    print(f"  ✅ Built Item Features -> {item_features_file} ({df_item_features.shape})")

    # -------------------------------------------------------------
    # 3. Interaction Features
    # -------------------------------------------------------------
    df_interaction_features = df_train.merge(
        df_user_features[["user_id", "user_avg_rating"]], on="user_id", how="left"
    ).merge(
        df_item_features[["item_id", "item_avg_rating"]], on="item_id", how="left"
    )

    df_interaction_features["user_rating_residual"] = (
        df_interaction_features["rating"] - df_interaction_features["user_avg_rating"]
    ).round(3)
    df_interaction_features["item_rating_residual"] = (
        df_interaction_features["rating"] - df_interaction_features["item_avg_rating"]
    ).round(3)
    df_interaction_features["interaction_recency_days"] = (
        (reference_timestamp - df_interaction_features["timestamp"]) / (24 * 3600)
    ).round(2)

    interaction_features_file = features_path / "interaction_features.parquet"
    df_interaction_features.to_parquet(interaction_features_file, index=False)
    print(f"  ✅ Built Interaction Features -> {interaction_features_file} ({df_interaction_features.shape})")

    # -------------------------------------------------------------
    # 4. Generate Feature Catalog Documentation
    # -------------------------------------------------------------
    catalog_md = """# Feature Catalog

This catalog documents the offline features generated by the feature engineering pipeline.
All features are standardized and reusable across offline model training and online serving.

| Feature Name | Entity | Data Type | Description | Computation / Source Logic | Reusable in Serving |
|---|---|---|---|---|---|
| `user_avg_rating` | User | Float | Historical mean rating given by the user | Mean of ratings in train split | Yes |
| `user_rating_count` | User | Integer | Total number of ratings submitted by user | Count of ratings in train split | Yes |
| `user_rating_std` | User | Float | Standard deviation of user's ratings | Std of user ratings in train split | Yes |
| `days_since_last_interaction` | User | Float | Inactivity recency in days | `(split_timestamp - max(user_timestamp)) / 86400` | Yes |
| `user_preferred_genre` | User | String | Most frequently rated genre by user | Mode across user's rated movie genres | Yes |
| `age` | User | Integer | Age of the user | Demographics table (`u.user`) | Yes |
| `gender_code` | User | Integer | Binary gender indicator (1=Male, 0=Female) | Encoded from `gender` | Yes |
| `item_popularity_rank` | Item | Integer | Global popularity rank | Ranked descending by interaction frequency | Yes |
| `item_avg_rating` | Item | Float | Historical average rating of the movie | Mean of ratings received in train split | Yes |
| `item_rating_count` | Item | Integer | Total volume of ratings received | Count of ratings received in train split | Yes |
| `genre_*` (19 columns) | Item | Binary (0/1) | Genre multi-hot indicators (Action, Drama, etc.) | Metadata table (`u.item`) | Yes |
| `user_rating_residual` | Interaction | Float | User-specific rating deviation | `rating - user_avg_rating` | Training Only |
| `item_rating_residual` | Interaction | Float | Item-specific rating deviation | `rating - item_avg_rating` | Training Only |
| `interaction_recency_days` | Interaction | Float | Elapsed time since interaction occurred | `(split_timestamp - timestamp) / 86400` | Yes |
"""
    catalog_file = reports_path / "feature_catalog.md"
    with open(catalog_file, "w") as f:
        f.write(catalog_md)
    print(f"  ✅ Exported Feature Catalog -> {catalog_file}")

    return {
        "user_features": str(user_features_file),
        "item_features": str(item_features_file),
        "interaction_features": str(interaction_features_file),
        "catalog_file": str(catalog_file)
    }


if __name__ == "__main__":
    build_features()

