"""
Data Ingestion Module.
Ingests User, Item, and Interaction datasets for the recommendation pipeline.
Supports automated download via KaggleHub or direct GroupLens archive fallback.
"""

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
import pandas as pd


def download_movielens(raw_dir: str = "data/raw", force_download: bool = False) -> dict:
    """
    Downloads and extracts the MovieLens 100K dataset into the raw directory.
    Uses kagglehub if available, with resilient fallback to official GroupLens repository.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    ratings_csv = raw_path / "ratings.csv"
    items_csv = raw_path / "items.csv"
    users_csv = raw_path / "users.csv"

    if not force_download and ratings_csv.exists() and items_csv.exists() and users_csv.exists():
        print(f"[Ingestion] Raw files already exist at {raw_path}. Skipping download.")
        return {
            "ratings": str(ratings_csv),
            "items": str(items_csv),
            "users": str(users_csv)
        }

    downloaded_folder = None

    # Strategy 1: Attempt kagglehub download
    try:
        import kagglehub
        print("[Ingestion] Attempting download via kagglehub...")
        downloaded_folder = kagglehub.dataset_download("prajitdatta/movielens-100k-dataset")
        print(f"[Ingestion] Downloaded via kagglehub to: {downloaded_folder}")
    except Exception as e:
        print(f"[Ingestion] Kagglehub download failed or unauthenticated ({e}). Trying GroupLens fallback...")

    # Strategy 2: Direct GroupLens official archive fallback
    if not downloaded_folder or not os.path.exists(downloaded_folder):
        fallback_zip = raw_path / "ml-100k.zip"
        url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
        print(f"[Ingestion] Downloading from {url} ...")
        urllib.request.urlretrieve(url, fallback_zip)
        with zipfile.ZipFile(fallback_zip, "r") as zip_ref:
            zip_ref.extractall(raw_path)
        downloaded_folder = str(raw_path / "ml-100k")
        if fallback_zip.exists():
            fallback_zip.unlink()

    # Search for files within the extracted/downloaded folder
    found_ratings = None
    found_items = None
    found_users = None

    for root, _, files in os.walk(downloaded_folder):
        for f in files:
            if f in ["u.data", "ratings.csv"]:
                found_ratings = os.path.join(root, f)
            elif f in ["u.item", "movies.csv", "items.csv"]:
                found_items = os.path.join(root, f)
            elif f in ["u.user", "users.csv"]:
                found_users = os.path.join(root, f)

    if not (found_ratings and found_items and found_users):
        raise FileNotFoundError(f"Could not locate u.data, u.item, or u.user inside {downloaded_folder}")

    # Standardize and save as CSV into data/raw/
    # 1. Parse Interactions (u.data)
    if "u.data" in found_ratings:
        df_ratings = pd.read_csv(
            found_ratings,
            sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"],
            engine="python"
        )
    else:
        df_ratings = pd.read_csv(found_ratings)
        df_ratings.rename(columns={"userId": "user_id", "movieId": "item_id"}, inplace=True)

    # 2. Parse Items (u.item)
    genre_cols = [
        "unknown", "Action", "Adventure", "Animation", "Childrens", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film_Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci_Fi", "Thriller", "War", "Western"
    ]
    if "u.item" in found_items:
        item_cols = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + genre_cols
        df_items = pd.read_csv(
            found_items,
            sep="|",
            names=item_cols,
            encoding="latin-1",
            engine="python"
        )
    else:
        df_items = pd.read_csv(found_items, encoding="latin-1")
        df_items.rename(columns={"movieId": "item_id"}, inplace=True)

    # 3. Parse Users (u.user)
    if "u.user" in found_users:
        df_users = pd.read_csv(
            found_users,
            sep="|",
            names=["user_id", "age", "gender", "occupation", "zip_code"],
            engine="python"
        )
    else:
        df_users = pd.read_csv(found_users)
        df_users.rename(columns={"userId": "user_id"}, inplace=True)

    # Write out standardized CSVs
    df_ratings.to_csv(ratings_csv, index=False)
    df_items.to_csv(items_csv, index=False)
    df_users.to_csv(users_csv, index=False)

    print(f"[Ingestion] Successfully ingested:")
    print(f"  - {len(df_ratings):,} interaction records -> {ratings_csv}")
    print(f"  - {len(df_items):,} item records -> {items_csv}")
    print(f"  - {len(df_users):,} user records -> {users_csv}")

    return {
        "ratings": str(ratings_csv),
        "items": str(items_csv),
        "users": str(users_csv)
    }


if __name__ == "__main__":
    download_movielens()

