<div align="center">

# 🎬 Netflix Prize Recommendation System

### Personalized Content Discovery via Matrix Factorization and Neural Collaborative Filtering

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![SciPy](https://img.shields.io/badge/SciPy-1.11+-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

<br/>

> **Built for the CULT Open Projects 2026 — Recommendation Systems Track**
> 
> A deep comparative study of SVD, NCF v1, and NCF v3 on the Netflix Prize Dataset, 
> documenting both what works and — critically — *why* standard approaches fail.

<br/>

[Overview](#-overview) · [Dataset](#-dataset) · [Architecture](#-architecture) · [Results](#-results) · [Key Findings](#-key-findings) · [Setup](#-setup) · [Structure](#-repository-structure)

---

</div>

## 📌 Overview

This project builds and systematically compares three recommendation models on the Netflix Prize Dataset — one of the largest and most studied datasets in recommender systems research.

The goal is not just to optimize RMSE. The core contribution of this project is an **empirical analysis of why offline metrics (RMSE, MAP@10) and genuine personalization quality move in opposite directions** across model architectures — a documented phenomenon in the field that most implementations overlook.

**Three models are compared:**

| Model | Approach | Key Design Choice |
|---|---|---|
| **SVD** | Matrix Factorization | Mean-centered sparse SVD via scipy |
| **NCF v1** | Neural CF | Embedding concat → MLP, MSE loss |
| **NCF v3** | Neural CF + Bias | Explicit user/item bias, weight decay, candidate filtering |

---

## 📦 Dataset

**Source:** [Netflix Prize Dataset on Kaggle](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)

| Attribute | Value |
|---|---|
| Total Ratings | 100,480,507 |
| Unique Users | 480,189 |
| Unique Movies | 17,770 |
| Rating Scale | 1 – 5 stars |
| Date Range | November 1999 – December 2005 |
| Matrix Sparsity | **98.82%** |

**Files used:**
```
netflix-prize-data/
├── combined_data_1.txt   # ~24M ratings
├── combined_data_2.txt   # ~26M ratings
├── combined_data_3.txt   # ~26M ratings
├── combined_data_4.txt   # ~24M ratings
└── movie_titles.csv      # movie_id, year, title
```

> **Note on file format:** `combined_data_*.txt` files are non-standard — movie IDs appear as separator lines ending with `:`, followed by `user_id,rating,date` rows. A custom parser is required (included in `src/data_loader.py`).

---

## 🏗 Architecture

### Data Pipeline

```
Raw .txt files
     ↓
Custom line parser (handles non-CSV format)
     ↓
dtype optimization: int64 → int8/int16/int32 (3.22 GB → 1.51 GB)
     ↓
Temporal train/test split (train: <2005, test: ≥2005)
     ↓
Filter test set: drop users/movies unseen in training
     ↓
5M-sample for model training (explicitly allowed by problem statement)
```

### SVD (Baseline)

```
User-Movie sparse matrix (scipy csr_matrix)
         ↓
Mean-centering per user  ← critical step (naive SVD gives RMSE > 2.7)
         ↓
scipy.sparse.linalg.svds (k=50 latent factors)
         ↓
Prediction: U[u] × diag(σ) × Vt[:,m] + user_mean[u]
         ↓
Clip to [1, 5]
```

### NCF v1

```
user_id ──► UserEmbedding(64) ──┐
                                 concat → Linear(128) → ReLU → Dropout
movie_id ► MovieEmbedding(64) ──┘        → Linear(64)  → ReLU → Dropout
                                          → Linear(1)   → predicted rating
```

### NCF v3 (Best Model)

```
user_id ──► UserEmbedding(64) ──┐
                                 concat → MLP(128→64→1)
movie_id ► MovieEmbedding(64) ──┘              ↓
                                          interaction
                                               +
user_id ──► UserBias(1) ──┐
                            + bias_sum
movie_id ► MovieBias(1) ──┘
                                               +
                                         global_mean (3.52)
                                               ↓
                                        predicted rating
```

> **Key architectural decisions:**
> - Explicit bias terms separate global popularity signal from the interaction term
> - Weight decay (`1e-4`) prevents bias overfitting on sparse items
> - Candidate filter (≥20 training ratings) eliminates noisy long-tail items from ranking

---

## 📊 Results

### Quantitative Comparison (200 test users, 1M test ratings for RMSE)

| Model | RMSE ↓ | MAP@10 ↑ | Coverage ↑ | Score Std/User ↑ |
|---|---|---|---|---|
| SVD (k=50) | 1.0763 | 0.0375 | — | — |
| NCF v1 | 0.9754 | 0.0548 | 0.49% | 0.345 |
| **NCF v3** | **0.9546** | **0.0496** | 0.31% | 0.493 |

> **NCF v3 achieves the best RMSE and competitive MAP@10.** However, coverage remains low across all models — a known structural limitation documented in Section 9 (Key Findings).

### Recommendation Quality (NCF v3, User 30878)

```
User liked: Hardball, The Missing, The Others

  v1 top-5 (popularity collapse):          v3 top-5 (regularized):
  ├─ The Notebook (2004)                   ├─ Star Wars: Episode V (1980)
  ├─ The O.C.: Season 1 (2003)             ├─ The Shawshank Redemption (1994)
  ├─ CSI: Season 2 (2001)                  ├─ Gilmore Girls: Season 1 (2001)
  ├─ Lord of the Rings: RotK (2003)        ├─ Family Guy: Vol. 2 (1999)
  └─ The Simpsons: Season 3 (1991)         └─ Alias: Season 3 (2003)
```

---

## 🔍 Key Findings

### 1. Mean-centering is non-negotiable for sparse SVD

A naive SVD on the raw sparse matrix treats missing entries as 0-star ratings. The model collapses — predicting near-zero for everything, giving **RMSE > 2.7**. Subtracting per-user mean ratings before SVD and adding them back at prediction time brings RMSE to **1.07**.

### 2. All models exhibit popularity collapse — for different structural reasons

- **NCF v1:** No bias terms → model learns item-level popularity as a low-risk MSE strategy → "The Notebook" recommended to **100% of users**
- **NCF v2 (intermediate):** Bias terms added, no regularization → bias parameters for movies with <10 training ratings overfit to noise → collapses onto a different cluster of obscure titles
- **NCF v3:** Weight decay + candidate filtering → best metrics, but Empire Strikes Back still appears in 92% of top-10 lists

**Conclusion:** Popularity collapse is structural to MSE-optimized collaborative filtering on this dataset — not a hyperparameter artifact. It requires architectural change (pairwise ranking loss) to fix properly.

### 3. MAP@10 is confounded by item popularity in offline evaluation

A model recommending globally popular movies gets "lucky hits" because popular movies have higher base-rate odds of appearing in any user's test-relevant set — independent of actual personalization. NCF v1's MAP@10 advantage over v3 partially reflects this confound, not superior taste modeling.

**Implication:** Reporting MAP@10 alone is insufficient. Coverage and per-user score variance are necessary diagnostic metrics to distinguish genuine personalization from popularity exploitation.

---

## ⚙️ Setup

### Requirements

```bash
Python >= 3.10
torch >= 2.0
scipy >= 1.11
pandas >= 2.0
numpy >= 1.24
matplotlib >= 3.7
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/netflix-recommender.git
cd netflix-recommender

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

```bash
# Option A: Kaggle API (recommended)
pip install kaggle
kaggle datasets download -d netflix-inc/netflix-prize-data
unzip netflix-prize-data.zip -d data/raw/

# Option B: Manual download
# Visit https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data
# Download and extract to data/raw/
```

---

## 🔁 Reproduce Results

All results are reproducible from a single notebook. Run cells top-to-bottom — do not run out of order (variable names are reused across cells intentionally).

### On Kaggle (recommended — no local setup needed)

1. Go to [Kaggle](https://www.kaggle.com) and create a new notebook
2. Add the Netflix Prize Dataset as input
3. Upload `notebooks/netflix_recommender.ipynb`
4. Set Accelerator to **GPU** if available (Settings → Accelerator)
5. Run All

### Local Jupyter

```bash
# From project root
jupyter notebook notebooks/netflix_recommender.ipynb
```

> **Expected runtimes (CPU):**
> - Data loading: ~100 seconds (all 4 files)
> - SVD training: ~60 seconds
> - NCF v1 training (5 epochs): ~15 minutes
> - NCF v3 training (8 epochs): ~20 minutes
> - MAP@10 evaluation per model: ~5 minutes

---

## 📁 Repository Structure

```
netflix-recommender/
│
├── notebooks/
│   └── netflix_recommender.ipynb     # Main notebook — full pipeline, run top to bottom
│
├── outputs/
│   ├── rating_distribution.png
│   ├── ratings_per_user.png
│   ├── ratings_per_movie.png
│   └── ratings_over_time.png
│
├── reports/
│   ├── technical_report.pdf          # 10-page technical report
│   └── presentation.pdf             # 8-slide presentation deck
│
├── requirements.txt
└── README.md
```

---

## 📋 Evaluation Metrics

| Metric | Definition | Used For |
|---|---|---|
| **RMSE** | Root Mean Squared Error on held-out ratings | Rating prediction accuracy |
| **MAP@10** | Mean Average Precision at 10, relevance threshold = rating ≥ 3.5 | Recommendation ranking quality |
| **Coverage** | % of catalog appearing in any top-10 list across 200 users | Personalization breadth |
| **Score Std/User** | Mean standard deviation of predicted scores per user | Per-user ranking differentiation |

### Train/Test Split

- **Train:** All ratings with date < January 1, 2005
- **Test:** All ratings with date ≥ January 1, 2005, filtered to users and movies seen in training
- **Rationale:** Temporal split prevents future data leakage and mirrors real deployment conditions

---

## 🚀 Future Work

- **Pairwise ranking loss (BPR/WARP):** Training directly for ranking (rather than MSE on ratings) should structurally reduce popularity collapse — the root cause of low coverage across all three models
- **Popularity-stratified MAP@10:** Report MAP@10 separately by item popularity tier (head / mid / long-tail) to detect whether quality comes from personalization or exposure bias
- **Hybrid content-based + CF:** Use movie metadata (genre, year, cast embeddings) to address cold-start for both new users and new items
- **Diversity-aware re-ranking:** Post-processing via Maximal Marginal Relevance to explicitly trade off predicted relevance against catalog diversity

---

## Author

Built for **CULT Open Projects 2026** — Recommendation Systems Problem Statement

**Kumar Manas**
B.Tech. Production and Industrial Engineering · IIT Roorkee

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kumarmanas-iitroorkee/)
[![GitHub](https://img.shields.io/badge/GitHub-121011?style=flat&logo=github&logoColor=white)](https://github.com/HighOnKeys)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail&logoColor=white)](mailto:kumar_m@me.iitr.ac.in)
---

## 📄 License

This project is licensed under the MIT License. The Netflix Prize Dataset is subject to its own terms of use — see the [Kaggle dataset page](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) for details.

---

<div align="center">

**If this project helped you, consider starring the repo ⭐**

</div>
