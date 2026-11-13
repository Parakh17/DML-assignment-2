# End-to-End Data Management Pipeline for a Recommendation System: Final Technical Report

**Course**: Data Management for Machine Learning (DM4ML)  
**Project**: End-to-End Recommendation Data Pipeline  
**Dataset**: MovieLens 100K Benchmark  
**GitHub Repository**: [https://github.com/Parakh17/DML-assignment-2](https://github.com/Parakh17/DML-assignment-2)  
**Date**: November 2026  

---

## 1. Executive Summary & Problem Formulation

Modern recommender systems rely not merely on sophisticated machine learning algorithms, but on reliable, reproducible, and automated data management pipelines. In commercial deployments, machine learning systems frequently degrade due to silent data corruptions, data leakage, lookahead bias, and disjointed feature computation between training and serving environments.

This project designs and implements an enterprise-grade, end-to-end data management pipeline for personalized recommendations. The pipeline orchestrates the complete lifecycle: ingesting multi-entity datasets (users, items, and interactions), enforcing automated quality validation and referential integrity checks, executing chronological time-based splits, transforming raw interactions into sparse matrices and analytical summaries, persisting offline feature stores with comprehensive metadata catalogs, training baseline and personalized models tracked via MLflow, and benchmarking across ranking, diversity, coverage, and rating accuracy metrics.

---

## 2. System Architecture & Modular Design

The pipeline follows a modular, decoupled architecture organized into distinct functional layers:

```
[Raw Ingestion (CSV)]
         │
         ▼
[Automated Validation & Referential Integrity] ──(Violations)──► [data/errors/]
         │ (Cleaned)
         ▼
[Data Preparation: Chronological Split (80/20)] ──► [data/processed/ (Parquet)]
         │
         ├─────────────────────────────────────────┐
         ▼                                         ▼
[Transformation Layer]                    [Feature Store Layer]
• User Activity Summaries                 • User Feature Store
• Item Popularity Summaries               • Item Feature Store
• Sparse CSR Interaction Matrix           • Interaction Residuals
         │                                         │
         └───────────────────┬─────────────────────┘
                             ▼
                 [Model Training & MLflow]
                 • Baseline Popularity
                 • Personalized Biased SVD
                             │
                             ▼
                 [Multi-Metric Evaluation]
                 • Ranking: Precision, Recall, NDCG
                 • Discovery: Coverage, Diversity
                 • Accuracy: RMSE, MAE
```

### Architectural Principles:
1. **Separation of Concerns**: Ingestion, validation, preparation, transformation, feature management, training, and evaluation exist as isolated, testable Python modules.
2. **Columnar & Compact Storage**: Transitioning from raw CSV to columnar Apache Parquet in `data/processed/` and `data/features/` ensures accelerated downstream I/O, strict schema preservation, and compact disk footprint.
3. **Reproducibility**: Parameterized via `configs/config.yaml` and fully orchestratable through either a single-command CLI (`src/pipeline.py`) or DAG orchestrators (Prefect / Apache Airflow).

---

## 3. Data Ingestion & Quality Assurance Strategy

### 3.1 Multi-Entity Ingestion
The ingestion module ingests three distinct relational entities:
* **Interactions (`ratings.csv`)**: 100,000 rating transactions with continuous Unix timestamps.
* **Items (`items.csv`)**: 1,682 films with release dates and 19 binary multi-hot genre classifications.
* **Users (`users.csv`)**: 943 registered users with age, gender, occupation, and zip codes.

### 3.2 Automated Validation & Referential Integrity Checks
Data validation is executed prior to any analytical transformations to prevent corrupted records from propagating:
* **Non-Null Constraints**: Primary keys (`user_id`, `item_id`), ratings, and timestamps are strictly required to be non-null.
* **Domain Range Constraints**: Ratings must lie within $[1.0, 5.0]$; timestamps must be valid epoch integers; user ages must fall within $[1, 120]$.
* **Referential Integrity**: An automated cross-table verification ensures every interaction references valid, pre-existing keys in `users.csv` and `items.csv`. Any orphan interaction is quarantined into `data/errors/invalid_ratings.csv` with explicit failure codes.
* **Audit Reporting**: Summary statistics and validation logs are automatically exported to `reports/data_validation_report.txt`.

---

## 4. Data Preparation & Transformation

### 4.1 Chronological Time-Based Split (Preventing Data Leakage)
A common pitfall in recommender system benchmarks is utilizing random train/test splits. In production environments, systems can only train on historical data to predict future interactions. A random split introduces severe **lookahead bias** and unrealistic evaluation metrics.

In this pipeline:
* Interactions are sorted chronologically by `timestamp`.
* A global temporal cutoff is established at the 80th percentile ($T = 889237269$, corresponding to *1998-03-07 02:21:09 UTC*).
* **Train Set**: 80,003 historical interactions ($\le T$).
* **Test Set**: 19,997 future interactions ($> T$).

### 4.2 Derived Analytical Datasets
From the training partition, the transformation module constructs:
1. **User Activity Summary**: Captures interaction count, mean rating, rating variance, and active span in days.
2. **Item Popularity Summary**: Aggregates interaction volume, empirical average rating, and global popularity rank.
3. **User-Item Interaction Matrix**: A compressed sparse row (CSR) matrix of dimension $(751 \times 1,616)$ with a calculated sparsity of **93.41%**, optimized for matrix factorization.

---

## 5. Feature Management & Feature Store

Feature stores bridge the gap between offline model training and real-time inference. Reusable offline features are persisted in Parquet tables and cataloged in `reports/feature_catalog.md`:

* **User Features (`data/features/user_features.parquet`)**:
  * Historical mean rating and rating variance.
  * Activity volume and recency (`days_since_last_interaction`).
  * Inferred preferred genre (mode of past interactions).
  * Demographic encodings (age, binary gender).
* **Item Features (`data/features/item_features.parquet`)**:
  * Global popularity rank and rating volume.
  * Historical average rating.
  * Multi-hot genre vectors (Action, Romance, Sci-Fi, etc.).
* **Interaction Features (`data/features/interaction_features.parquet`)**:
  * User-rating residuals ($r_{u,i} - \bar{r}_u$) and item-rating residuals ($r_{u,i} - \bar{r}_i$).
  * Interaction recency in days relative to the split cutoff.

---

## 6. Model Training & Experiment Tracking

Two distinct recommendation models were developed and tracked using **MLflow** (`sqlite:///mlflow.db`):

### 6.1 Baseline Model: Bayesian Popularity Recommender
Ranks candidate items using an IMDB-style Bayesian weighted rating formula:
$$\text{Score} = \left(\frac{v}{v + m}\right) R + \left(\frac{m}{v + m}\right) C$$
where $v$ is the item's rating count, $m=10$ is the minimum vote threshold, $R$ is the item's average rating, and $C$ is the global mean rating across all items.

### 6.2 Personalized Model: Regularized Biased SVD Matrix Factorization
Decomposes the rating matrix into low-rank latent factor representations with explicit bias modeling:
$$\hat{r}_{u,i} = \mu + b_u + b_i + P_u \cdot Q_i^T$$
where $\mu$ is the global mean, $b_u$ is user bias, $b_i$ is item bias, and $P_u, Q_i \in \mathbb{R}^{20}$ are user and item latent vectors. Optimization is executed via Stochastic Gradient Descent (SGD) with $L_2$ regularization ($\lambda = 0.02$) and learning rate $\eta = 0.005$ over 25 epochs.

All hyperparameters, training samples, loss trajectories, and pickled model weights are automatically registered in MLflow runs.

---

## 7. Comparative Evaluation & Empirical Results

Models were evaluated on the chronological test partition with a relevance threshold of $r \ge 3.5$.

| Evaluation Metric | Baseline (Popularity) | Personalized (SVD) | Relative Change |
|---|---|---|---|
| **RMSE** | 1.0367 | **1.0276** | **-0.88% (Superior)** |
| **MAE** | 0.8314 | **0.8181** | **-1.60% (Superior)** |
| **Precision@5** | 0.1345 | **0.1359** | **+1.04%** |
| **Recall@5** | 0.0252 | **0.0254** | **+0.79%** |
| **NDCG@5** | 0.1328 | **0.1306** | -1.66% |
| **Catalog Coverage@5** | 0.0166 | **0.0452** | **+172.29% (Nearly 3x)** |
| **Catalog Coverage@10** | 0.0256 | **0.0743** | **+190.23% (Nearly 3x)** |
| **Intra-List Diversity@5** | 0.7182 | **0.7162** | Preserved |
| **Intra-List Diversity@10** | 0.7606 | **0.7618** | **+0.16%** |

### Critical Analytical Insights:
1. **Popularity Bias vs. Catalog Exploration**: The popularity recommender suffers from severe head-bias, recommending the same top ~1.6% of movies to all users. In contrast, the personalized SVD recommender expands catalog coverage by **+172% to +190%**, surfacing personalized long-tail items without compromising precision.
2. **Diversity Preservation**: Both models achieve healthy intra-list diversity (>0.71 at K=5; >0.76 at K=10), ensuring recommendation lists do not collapse into single-genre echo chambers.
3. **Rating Error**: Personalized latent factors achieve systematically lower RMSE and MAE than global baselines.

---

## 8. Orchestration & Reproducibility

The workflow is orchestrated using **Prefect** (`dags/pipeline_flow.py`) and Apache Airflow DAG standards (`dags/airflow_dag.py`). Task dependencies are strictly bound:
$$\text{Ingestion} \rightarrow \text{Validation} \rightarrow \text{Preparation} \rightarrow [\text{Transformation}, \text{Features}] \rightarrow \text{Training} \rightarrow \text{Evaluation}$$

Prefect handles task caching, retry policies, and state transitions, while a standalone runner CLI (`python -m src.pipeline`) enables immediate headless reproduction in under 20 seconds.

### 8.1 Workflow Execution Run
Below is the verification screenshot confirming successful end-to-end execution of all pipeline tasks (Data Ingestion, Quality Validation, Preparation, Transformation, Feature Store, Model Training, and Evaluation):

![Successful Workflow Run](orchestration_run_screenshot.png)
*Figure 1: Successful end-to-end execution of all pipeline tasks and metrics evaluation.*

---

## 9. Real-World Limitations & Production Roadmap

1. **Cold-Start Problem**: Pure collaborative filtering degrades when completely new users or items arrive without past interaction history. In production, this is mitigated by a hybrid fallback to content-based or demographic feature embeddings.
2. **Streaming & Nearline Feature Updates**: While the current architecture provides an offline feature store, real-world systems require streaming ingestion (e.g., Apache Kafka / Flink) and a low-latency key-value store (e.g., Redis / Feast) to update `days_since_last_interaction` and recent click streams in real time.
3. **Incremental Learning**: Retraining matrix factorization from scratch is computationally expensive at scale; implementing online SGD or incremental matrix updates would facilitate continuous real-time model adaptation.

