"""
Model Training Module with MLflow Tracking.
Implements:
1. Baseline Popularity Recommender
2. Personalized Matrix Factorization (Regularized SVD with Biases)
Tracks parameters, artifacts, and metrics using MLflow.
"""

import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import mlflow


class PopularityRecommender:
    """
    Baseline Recommender that ranks items based on global interaction frequency and mean rating.
    """
    def __init__(self, top_k: int = 10):
        self.top_k = top_k
        self.global_mean_ = 3.5
        self.item_stats_ = None
        self.ranked_items_ = []

    def fit(self, df_train: pd.DataFrame):
        self.global_mean_ = float(df_train["rating"].mean())
        stats = df_train.groupby("item_id").agg(
            rating_count=("rating", "count"),
            avg_rating=("rating", "mean")
        ).reset_index()

        # Weighted score (Bayesian average / IMDB weighted formula)
        # score = (v / (v + m)) * R + (m / (v + m)) * C
        m = 10  # minimum ratings threshold
        C = self.global_mean_
        stats["score"] = (
            (stats["rating_count"] / (stats["rating_count"] + m)) * stats["avg_rating"]
            + (m / (stats["rating_count"] + m)) * C
        )
        stats.sort_values(by="score", ascending=False, inplace=True)

        self.item_stats_ = stats
        self.ranked_items_ = stats["item_id"].tolist()
        return self

    def recommend(self, user_id: int, user_seen_items: set, k: int = 10) -> list:
        recs = [item for item in self.ranked_items_ if item not in user_seen_items]
        return recs[:k]

    def predict_rating(self, user_id: int, item_id: int) -> float:
        if self.item_stats_ is not None and item_id in self.item_stats_["item_id"].values:
            return float(self.item_stats_.loc[self.item_stats_["item_id"] == item_id, "avg_rating"].values[0])
        return self.global_mean_


class MatrixFactorizationSVD:
    """
    Personalized Matrix Factorization with explicit User and Item Biases:
    r_hat(u, i) = mu + b_u + b_i + P_u . Q_i^T
    Trained via regularized Stochastic Gradient Descent (SGD).
    """
    def __init__(
        self,
        n_factors: int = 20,
        lr: float = 0.005,
        reg: float = 0.02,
        n_epochs: int = 25,
        random_state: int = 42
    ):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.random_state = random_state

        self.global_mean_ = 0.0
        self.user_biases_ = {}
        self.item_biases_ = {}
        self.P_ = {}  # User latent factors
        self.Q_ = {}  # Item latent factors

        self.user_seen_items_ = {}
        self.all_items_ = []

    def fit(self, df_train: pd.DataFrame):
        rng = np.random.default_rng(self.random_state)
        self.global_mean_ = float(df_train["rating"].mean())

        users = df_train["user_id"].unique()
        items = df_train["item_id"].unique()
        self.all_items_ = list(items)

        # Initialize biases to 0
        self.user_biases_ = {u: 0.0 for u in users}
        self.item_biases_ = {i: 0.0 for i in items}

        # Initialize latent factors normally distributed around 0
        self.P_ = {u: rng.normal(0, 0.1, self.n_factors) for u in users}
        self.Q_ = {i: rng.normal(0, 0.1, self.n_factors) for i in items}

        # Cache seen items for recommendation filtering
        self.user_seen_items_ = df_train.groupby("user_id")["item_id"].apply(set).to_dict()

        records = df_train[["user_id", "item_id", "rating"]].to_numpy(copy=True)

        # SGD Optimization
        for epoch in range(1, self.n_epochs + 1):
            perm = rng.permutation(len(records))
            shuffled_records = records[perm]
            loss = 0.0
            for u_val, i_val, r_val in shuffled_records:
                u, i, r = int(u_val), int(i_val), float(r_val)
                # Prediction
                pred = (
                    self.global_mean_
                    + self.user_biases_[u]
                    + self.item_biases_[i]
                    + np.dot(self.P_[u], self.Q_[i])
                )
                err = r - pred
                loss += err ** 2

                # Updates
                self.user_biases_[u] += self.lr * (err - self.reg * self.user_biases_[u])
                self.item_biases_[i] += self.lr * (err - self.reg * self.item_biases_[i])

                p_u = self.P_[u]
                q_i = self.Q_[i]

                self.P_[u] += self.lr * (err * q_i - self.reg * p_u)
                self.Q_[i] += self.lr * (err * p_u - self.reg * q_i)

        return self

    def predict_rating(self, user_id: int, item_id: int) -> float:
        b_u = self.user_biases_.get(user_id, 0.0)
        b_i = self.item_biases_.get(item_id, 0.0)
        p_u = self.P_.get(user_id, np.zeros(self.n_factors))
        q_i = self.Q_.get(item_id, np.zeros(self.n_factors))

        pred = self.global_mean_ + b_u + b_i + float(np.dot(p_u, q_i))
        # Clip to valid rating range [1.0, 5.0]
        return max(1.0, min(5.0, pred))

    def recommend(self, user_id: int, user_seen_items: set, k: int = 10) -> list:
        candidate_items = [i for i in self.all_items_ if i not in user_seen_items]
        if not candidate_items:
            return []

        p_u = self.P_.get(user_id, None)
        b_u = self.user_biases_.get(user_id, 0.0)

        if p_u is None:
            # Fallback for completely unseen cold users
            scores = [(i, self.global_mean_ + self.item_biases_.get(i, 0.0)) for i in candidate_items]
        else:
            scores = []
            for i in candidate_items:
                b_i = self.item_biases_.get(i, 0.0)
                q_i = self.Q_.get(i, np.zeros(self.n_factors))
                pred = self.global_mean_ + b_u + b_i + np.dot(p_u, q_i)
                scores.append((i, pred))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in scores[:k]]


def train_models(
    processed_dir: str = "data/processed",
    models_dir: str = "models",
    tracking_uri: str = "sqlite:///mlflow.db"
) -> dict:
    """
    Trains baseline and personalized recommenders, logs metrics and artifacts to MLflow.
    """
    processed_path = Path(processed_dir)
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_parquet(processed_path / "train_ratings.parquet")
    print(f"[Training] Loading {len(df_train):,} training interactions for model training...")

    # MLflow Setup
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "recommendation_system_pipeline"
    mlflow.set_experiment(experiment_name)

    # -------------------------------------------------------------
    # 1. Train Baseline Popularity Recommender
    # -------------------------------------------------------------
    with mlflow.start_run(run_name="baseline_popularity"):
        print("[Training] Fitting Baseline Popularity Recommender...")
        baseline = PopularityRecommender(top_k=10)
        baseline.fit(df_train)

        baseline_path = models_path / "baseline_popularity.pkl"
        with open(baseline_path, "wb") as f:
            pickle.dump(baseline, f)

        # Log params & artifacts
        mlflow.log_param("model_type", "popularity_baseline")
        mlflow.log_param("training_samples", len(df_train))
        mlflow.log_artifact(str(baseline_path), artifact_path="models")
        print(f"  ✅ Saved Popularity Model -> {baseline_path}")

    # -------------------------------------------------------------
    # 2. Train Matrix Factorization (SVD) Recommender
    # -------------------------------------------------------------
    with mlflow.start_run(run_name="personalized_svd"):
        print("[Training] Fitting Personalized SVD Matrix Factorization...")
        svd_params = {
            "n_factors": 20,
            "lr": 0.005,
            "reg": 0.02,
            "n_epochs": 25,
            "random_state": 42
        }
        svd_model = MatrixFactorizationSVD(**svd_params)
        svd_model.fit(df_train)

        svd_path = models_path / "personalized_svd.pkl"
        with open(svd_path, "wb") as f:
            pickle.dump(svd_model, f)

        # Log params & artifacts
        for k, v in svd_params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("model_type", "collaborative_filtering_svd")
        mlflow.log_param("training_samples", len(df_train))
        mlflow.log_artifact(str(svd_path), artifact_path="models")
        print(f"  ✅ Saved Personalized SVD Model -> {svd_path}")

    return {
        "baseline_model": str(baseline_path),
        "svd_model": str(svd_path)
    }


if __name__ == "__main__":
    train_models()
