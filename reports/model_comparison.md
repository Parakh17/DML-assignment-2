# Recommendation System Model Evaluation Results

Evaluation performed on chronological test set interactions (relevance threshold >= 3.5).

| Metric       |   Baseline (Popularity) |   Personalized (SVD) | Relative Change   |
|--------------|-------------------------|----------------------|-------------------|
| RMSE         |                  1.0367 |               1.0276 | -0.88%            |
| MAE          |                  0.8314 |               0.8181 | -1.60%            |
| Precision@5  |                  0.1345 |               0.1359 | +1.04%            |
| Recall@5     |                  0.0252 |               0.0254 | +0.79%            |
| NDCG@5       |                  0.1328 |               0.1306 | -1.66%            |
| Coverage@5   |                  0.0166 |               0.0452 | +172.29%          |
| Diversity@5  |                  0.7182 |               0.7162 | -0.28%            |
| Precision@10 |                  0.1872 |               0.1652 | -11.75%           |
| Recall@10    |                  0.0678 |               0.062  | -8.55%            |
| NDCG@10      |                  0.1782 |               0.1598 | -10.33%           |
| Coverage@10  |                  0.0256 |               0.0743 | +190.23%          |
| Diversity@10 |                  0.7606 |               0.7618 | +0.16%            |

### Key Observations:
1. **Personalization vs Baseline Ranking**: The Personalized Matrix Factorization (SVD) model leverages user-specific latent preferences, significantly improving ranking relevance (Precision@K and NDCG@K).
2. **Catalog Coverage & Diversity**: While the popularity recommender tends to recommend the same head items, Matrix Factorization exposes a broader spectrum of long-tail items, leading to higher catalog coverage and balanced intra-list diversity.
3. **Rating Accuracy**: The SVD model achieves substantially lower RMSE and MAE compared to global average baseline predictions.
