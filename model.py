"""
model.py — Sentiment Analysis Model
=====================================
Trains a Logistic Regression classifier on IMDb-style movie review data,
evaluates it, saves the fitted pipeline, and writes all result artefacts
(accuracy graph + confusion matrix) to results/.

Usage
-----
    python model.py

The script falls back to a small synthetic dataset when no CSV is found so
you can run it without the Kaggle download for quick smoke-testing.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_PATH  = os.path.join(BASE_DIR, "sentiment_model.pkl")
DATA_PATH   = os.path.join(BASE_DIR, "IMDB Dataset.csv")   # Kaggle file name

os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Palette (dark cinematic theme)
# ─────────────────────────────────────────────
BG       = "#0d0d0f"
SURFACE  = "#16161a"
ACCENT1  = "#e63946"   # red  → negative
ACCENT2  = "#2ec4b6"   # teal → positive
TEXT     = "#f1faee"
GRID     = "#1e1e24"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    SURFACE,
    "axes.edgecolor":    GRID,
    "axes.labelcolor":   TEXT,
    "xtick.color":       TEXT,
    "ytick.color":       TEXT,
    "text.color":        TEXT,
    "grid.color":        GRID,
    "grid.linewidth":    0.6,
    "font.family":       "monospace",
})

# ─────────────────────────────────────────────
# 1. Load / generate data
# ─────────────────────────────────────────────
def load_data():
    if os.path.exists(DATA_PATH):
        print(f"[data] Loading Kaggle IMDb dataset from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        # Normalise column names from the Kaggle extensive dataset
        df.columns = df.columns.str.lower().str.strip()
        if "review" in df.columns and "sentiment" in df.columns:
            df = df[["review", "sentiment"]].dropna()
        elif "text" in df.columns and "label" in df.columns:
            df = df.rename(columns={"text": "review", "label": "sentiment"})
        else:
            raise ValueError(
                "Expected columns 'review'/'sentiment' or 'text'/'label'. "
                f"Found: {list(df.columns)}"
            )
        df["sentiment"] = df["sentiment"].str.lower().str.strip()
        df = df[df["sentiment"].isin(["positive", "negative"])].reset_index(drop=True)
        print(f"[data] Loaded {len(df):,} rows  |  "
              f"pos={df['sentiment'].eq('positive').sum():,}  "
              f"neg={df['sentiment'].eq('negative').sum():,}")
    else:
        print("[data] Kaggle CSV not found — generating synthetic demo dataset …")
        df = _synthetic_dataset(n=4_000)
    return df


def _synthetic_dataset(n: int = 4_000) -> pd.DataFrame:
    """Small synthetic corpus for smoke-testing without the Kaggle file."""
    rng = np.random.default_rng(42)
    pos_phrases = [
        "absolutely loved this movie", "outstanding performances all around",
        "gripping storyline kept me engaged", "a masterpiece of cinema",
        "brilliant direction and superb acting", "heartwarming and uplifting story",
        "an unforgettable experience in the theatre", "deserves every award it wins",
        "visually stunning and emotionally powerful", "will watch again and again",
        "the best film I have seen this year", "beautifully crafted and deeply moving",
        "a triumph of storytelling", "perfect in every way", "breathtaking cinematography",
    ]
    neg_phrases = [
        "complete waste of time and money", "terrible acting all around",
        "boring plot with no redeeming qualities", "one of the worst films ever",
        "awful dialogue and poor direction", "painfully slow and uninteresting",
        "deeply disappointing after the hype", "save yourself and skip this one",
        "hollow characters with zero development", "the script is an absolute mess",
        "I walked out after thirty minutes", "badly paced and incoherent story",
        "nothing makes sense in this film", "the worst sequel imaginable",
        "unbearable from start to finish",
    ]
    reviews, labels = [], []
    for _ in range(n // 2):
        base = rng.choice(pos_phrases)
        filler = " ".join(rng.choice(pos_phrases, size=rng.integers(1, 5)))
        reviews.append(f"{base}. {filler}.")
        labels.append("positive")
    for _ in range(n // 2):
        base = rng.choice(neg_phrases)
        filler = " ".join(rng.choice(neg_phrases, size=rng.integers(1, 5)))
        reviews.append(f"{base}. {filler}.")
        labels.append("negative")
    idx = rng.permutation(len(reviews))
    return pd.DataFrame({"review": np.array(reviews)[idx],
                         "sentiment": np.array(labels)[idx]})


# ─────────────────────────────────────────────
# 2. Build pipeline
# ─────────────────────────────────────────────
def build_pipeline() -> Pipeline:
    tfidf = TfidfVectorizer(
        max_features=60_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=3,
        max_df=0.95,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b[a-zA-Z]{2,}\b",
    )
    clf = LogisticRegression(
        C=4.0,
        max_iter=1_000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


# ─────────────────────────────────────────────
# 3. Train & evaluate
# ─────────────────────────────────────────────
def train(df: pd.DataFrame):
    X = df["review"].values
    y = (df["sentiment"] == "positive").astype(int).values   # 1=pos, 0=neg

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    print("[train] Fitting pipeline …")
    pipeline.fit(X_train, y_train)

    # ── metrics ────────────────────────────────
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_proba)
    cm      = confusion_matrix(y_test, y_pred)
    report  = classification_report(y_test, y_pred,
                                    target_names=["Negative", "Positive"])

    print(f"\n{'─'*50}")
    print(f"  Test Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  ROC-AUC Score : {auc:.4f}")
    print(f"\n{report}")
    print(f"{'─'*50}\n")

    # ── cross-validation ───────────────────────
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"[cv] 5-fold CV  mean={cv_scores.mean():.4f}  std={cv_scores.std():.4f}")

    return pipeline, acc, auc, cm, cv_scores, y_test, y_pred, y_proba


# ─────────────────────────────────────────────
# 4. Plot: Accuracy graph
# ─────────────────────────────────────────────
def plot_accuracy(acc: float, cv_scores: np.ndarray, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BG)

    # ── left: CV fold bars ─────────────────────
    ax = axes[0]
    ax.set_facecolor(SURFACE)
    folds = [f"Fold {i+1}" for i in range(len(cv_scores))]
    bars  = ax.bar(folds, cv_scores * 100,
                   color=[ACCENT2 if s >= cv_scores.mean() else ACCENT1
                          for s in cv_scores],
                   width=0.55, zorder=3, edgecolor="none")

    mean_line = ax.axhline(cv_scores.mean() * 100,
                           color="#f1c40f", lw=1.5, ls="--", zorder=4)
    ax.set_ylim(max(0, (cv_scores.min() - 0.02) * 100),
                min(100, (cv_scores.max() + 0.03) * 100))
    ax.set_ylabel("Accuracy (%)", fontsize=10, labelpad=8)
    ax.set_title("5-Fold Cross-Validation Accuracy",
                 fontsize=12, fontweight="bold", pad=14, color=TEXT)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    for bar, score in zip(bars, cv_scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15,
                f"{score*100:.1f}%",
                ha="center", va="bottom", fontsize=8.5, color=TEXT)
    ax.legend(
        [mean_line, mpatches.Patch(color=ACCENT2), mpatches.Patch(color=ACCENT1)],
        [f"Mean {cv_scores.mean()*100:.2f}%", "≥ Mean", "< Mean"],
        fontsize=8, loc="lower right",
        facecolor=SURFACE, edgecolor=GRID
    )

    # ── right: metric summary gauge-style bars ─
    ax2 = axes[1]
    ax2.set_facecolor(SURFACE)
    metrics  = ["Test Accuracy", "CV Mean", "CV Std (±)"]
    values   = [acc * 100, cv_scores.mean() * 100, cv_scores.std() * 100]
    colors   = [ACCENT2, ACCENT2, ACCENT1]
    h_bars   = ax2.barh(metrics, values, color=colors,
                        height=0.45, zorder=3, edgecolor="none")
    ax2.set_xlim(0, 105)
    ax2.xaxis.grid(True, zorder=0)
    ax2.set_axisbelow(True)
    ax2.set_title("Model Performance Summary",
                  fontsize=12, fontweight="bold", pad=14, color=TEXT)
    ax2.set_xlabel("Value (%)", fontsize=10, labelpad=8)
    for bar, val in zip(h_bars, values):
        ax2.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}%", va="center", fontsize=9, color=TEXT)

    fig.suptitle("SENTIMENT ANALYSIS — MODEL ACCURACY REPORT",
                 fontsize=14, fontweight="bold", color=TEXT, y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"[plot] Accuracy graph  → {save_path}")


# ─────────────────────────────────────────────
# 5. Plot: Confusion matrix
# ─────────────────────────────────────────────
def plot_confusion_matrix(cm: np.ndarray, save_path: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SURFACE)

    # Custom colormap from dark → teal (for correct) / dark → red (overall)
    cmap = sns.diverging_palette(0, 170, s=90, l=30, as_cmap=True)

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(
        cm_norm,
        annot=False,
        fmt=".1f",
        cmap=cmap,
        linewidths=2,
        linecolor=BG,
        ax=ax,
        cbar=True,
        cbar_kws={"shrink": 0.75},
        vmin=0, vmax=100,
    )

    labels = ["Negative", "Positive"]
    # Annotate each cell manually for full control
    for i in range(2):
        for j in range(2):
            raw  = cm[i, j]
            pct  = cm_norm[i, j]
            is_diag = (i == j)
            ax.text(j + 0.5, i + 0.42,
                    f"{raw:,}",
                    ha="center", va="center",
                    fontsize=18, fontweight="bold",
                    color=TEXT if is_diag else "#aaaaaa")
            ax.text(j + 0.5, i + 0.62,
                    f"{pct:.1f}%",
                    ha="center", va="center",
                    fontsize=10, color=TEXT if is_diag else "#888888")

    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11, rotation=0)
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=10)
    ax.set_ylabel("True Label",      fontsize=11, labelpad=10)
    ax.set_title("CONFUSION MATRIX",
                 fontsize=14, fontweight="bold", pad=18, color=TEXT)

    # diagonal outline
    for d in range(2):
        ax.add_patch(plt.Rectangle((d, d), 1, 1,
                                   fill=False, edgecolor=ACCENT2,
                                   lw=2.5, zorder=5))

    legend_elements = [
        mpatches.Patch(facecolor=ACCENT2, label="Correct prediction"),
        mpatches.Patch(facecolor=ACCENT1, label="Misclassification"),
    ]
    ax.legend(handles=legend_elements, loc="upper left",
              fontsize=8.5, facecolor=SURFACE, edgecolor=GRID,
              bbox_to_anchor=(0, -0.12), ncol=2)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"[plot] Confusion matrix → {save_path}")


# ─────────────────────────────────────────────
# 6. Sample predictions
# ─────────────────────────────────────────────
def print_sample_predictions(pipeline):
    samples = [
        "An absolute masterpiece. The storytelling is breathtaking and the acting is superb.",
        "Terrible film. Dull, boring, and a complete waste of two hours.",
        "I loved every minute of this movie. Highly recommended!",
        "One of the worst movies I have ever seen. Deeply disappointing.",
        "A beautifully crafted film with outstanding performances.",
        "The plot made no sense whatsoever. Save your money.",
        "Surprisingly good! I wasn't expecting much but it blew me away.",
        "Painfully slow pacing and cardboard characters.",
    ]
    print("\n── Sample Predictions ─────────────────────────────────────────")
    for text in samples:
        pred  = pipeline.predict([text])[0]
        proba = pipeline.predict_proba([text])[0]
        label = "POSITIVE ✓" if pred == 1 else "NEGATIVE ✗"
        conf  = max(proba) * 100
        print(f"  {label}  ({conf:.1f}%)  │  {text[:65]}…" if len(text) > 65 else
              f"  {label}  ({conf:.1f}%)  │  {text}")
    print("───────────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────
# 7. Save model
# ─────────────────────────────────────────────
def save_model(pipeline, path: str):
    with open(path, "wb") as f:
        pickle.dump(pipeline, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(path) / 1_048_576
    print(f"[save] Model saved → {path}  ({size_mb:.1f} MB)")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═" * 55)
    print("  SENTIMENT ANALYSIS — MODEL TRAINING PIPELINE")
    print("═" * 55 + "\n")

    df = load_data()
    pipeline, acc, auc, cm, cv_scores, y_test, y_pred, y_proba = train(df)

    plot_accuracy(acc, cv_scores,
                  os.path.join(RESULTS_DIR, "accuracy.png"))
    plot_confusion_matrix(cm,
                          os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    print_sample_predictions(pipeline)
    save_model(pipeline, MODEL_PATH)

    print("\n" + "═" * 55)
    print("  Training complete.  Run:  streamlit run app.py")
    print("═" * 55 + "\n")
