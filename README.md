# 🎬 CineScope — AI Sentiment Analysis Dashboard

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**A production-grade NLP pipeline for movie review sentiment classification,**  
**served via a cinematic dark-themed Streamlit dashboard.**

[Live Demo](#-live-demo) · [Quick Start](#-quick-start) · [Docker](#-docker) · [Results](#-results) · [Deployment](#-deployment)

</div>

---

## 📖 Project Overview

CineScope is an end-to-end Natural Language Processing project that trains a **Logistic Regression** classifier on the **IMDb Movie Reviews dataset** (50,000 reviews) and exposes the model through an interactive Streamlit web application.

Given any free-text movie review, the dashboard instantly returns:
- **Sentiment verdict** — Positive ✦ or Negative ✗  
- **Confidence score** with an animated progress bar  
- **Per-class probability breakdown** (positive % / negative %)  
- **Session history** of recent analyses

The project was designed to demonstrate a complete ML lifecycle: data ingestion → preprocessing → feature engineering → model training → evaluation → artefact export → containerised deployment.

---

## 🏗️ Architecture

```
sentiment-dashboard/
├── app.py                  # Streamlit web application (UI + inference)
├── model.py                # Training pipeline, evaluation, artefact export
├── text_utils.py           # Shared text cleaning (HTML strip, contractions) — used by BOTH app.py and model.py
├── requirements.txt        # Python dependencies (pinned versions)
├── Dockerfile              # Multi-stage production container
├── README.md               # This file
├── sentiment_model.pkl     # Serialised trained pipeline (auto-generated)
└── results/
    ├── accuracy.png        # 5-fold CV accuracy + performance summary chart
    └── confusion_matrix.png# Annotated confusion matrix heatmap
```

---

## 🧠 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python 3.11 | Core runtime |
| ML Framework | Scikit-learn 1.4 | TF-IDF vectoriser + Logistic Regression |
| Feature Engineering | TF-IDF (60k n-grams) | Text → numeric representation |
| Web UI | Streamlit 1.35 | Interactive dashboard |
| Visualisation | Matplotlib + Seaborn | Training artefacts |
| Containerisation | Docker | Reproducible deployment |
| Dataset | IMDb 50k Reviews (Kaggle) | Training / evaluation corpus |

### Model Details

- **Algorithm:** Logistic Regression (`lbfgs` solver, `C=1.0`, balanced class weights)
- **Vectoriser:** TF-IDF with `(1,2)` n-grams, 60,000 max features, sublinear TF scaling
- **Preprocessing:** custom `clean_text()` (in `text_utils.py`) strips HTML tags (`<br />`, etc.) and **expands contractions before tokenising** (`don't` → `do not`), so negation isn't silently dropped — the single biggest lever for sentiment accuracy on short reviews
- **Evaluation:** 80/20 stratified train-test split + 5-fold cross-validation
- **Serialisation:** Python `pickle` (HIGHEST_PROTOCOL)

> ⚠️ **Important:** `model.py` falls back to a tiny synthetic dataset if `IMDB Dataset.csv` is missing. That fallback is only meant for smoke-testing the pipeline — it produces a model with a ~650-word vocabulary that cannot generalise to real reviews. **Always train on the real 50k dataset before deploying.**

---

## ⚡ Quick Start

### Prerequisites

- Python 3.9 or higher  
- pip

### 1 — Clone the repository

```bash
git clone https://github.com/Ozse97/cinescope-sentiment.git
cd cinescope-sentiment
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — (Optional) Add the Kaggle dataset

Download the IMDb Extensive Dataset from Kaggle and place the CSV in the project root:

```
sentiment-dashboard/
└── IMDB Dataset.csv   ← put it here
```

> **Without the Kaggle file** the script auto-generates a balanced synthetic corpus so you can run and test everything immediately.

### 4 — Train the model

```bash
python model.py
```

This will:
1. Load and preprocess the dataset  
2. Fit the TF-IDF + Logistic Regression pipeline  
3. Run 5-fold cross-validation  
4. Save `sentiment_model.pkl`  
5. Write `results/accuracy.png` and `results/confusion_matrix.png`  
6. Print sample predictions to stdout

### 5 — Launch the dashboard

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🐳 Docker

### Build the image

```bash
docker build -t cinescope-sentiment .
```

### Run the container

```bash
docker run -p 8501:8501 cinescope-sentiment
```

Open **http://localhost:8501**.

### Run with a local dataset (volume mount)

```bash
docker run -p 8501:8501 \
  -v "$(pwd)/IMDB Dataset.csv:/app/IMDB Dataset.csv" \
  cinescope-sentiment
```

### Docker Compose (optional)

```yaml
version: "3.9"
services:
  sentiment-app:
    build: .
    ports:
      - "8501:8501"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

```bash
docker compose up -d
```

---

## ☁️ Deployment

### Streamlit Community Cloud (free)

1. Push your repository to GitHub (include `sentiment_model.pkl` or let `model.py` be run as a startup step)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
3. Click **New app** → select your repo → set **Main file path** to `app.py`
4. Click **Deploy** — your app will be live in ~2 minutes

> **Tip:** Add a `packages.txt` file if you need system libraries.  
> The model trains automatically on first boot if `sentiment_model.pkl` is absent.

### Railway / Render / Fly.io

All three support Docker deployments. Push your image or connect the repo, set the start command to:

```bash
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

### Environment Variables (optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8501` | Port the Streamlit server listens on |

---

## 📊 Results

### Accuracy & Cross-Validation

![Accuracy Graph](results/accuracy.png)

The model achieves **91.0% test accuracy** on the real IMDb 50k dataset with a
5-fold cross-validation mean of **90.9%** (σ = 0.28%).

| Metric | Value |
|--------|-------|
| Test Accuracy | 91.01% |
| ROC-AUC | 0.971 |
| CV Mean (5-fold) | 90.89% |
| CV Std (5-fold) | 0.28% |

### Confusion Matrix

![Confusion Matrix](results/confusion_matrix.png)

### Sample Predictions

| Review (excerpt) | Prediction | Confidence |
|-----------------|-----------|-----------|
| *"An absolute masterpiece…"* | ✦ Positive | 89.1% |
| *"Terrible film. Dull and boring…"* | ✗ Negative | 99.9% |
| *"I loved every minute of this…"* | ✦ Positive | 94.2% |
| *"One of the worst movies I have ever seen…"* | ✗ Negative | 99.4% |
| *"Surprisingly good! I wasn't expecting…"* | ✦ Positive | 84.1% |

---

## 🖥️ Dashboard Preview

> The dashboard features a **cinematic dark theme** with gradient typography,  
> animated confidence bars, one-click example reviews, and a live session history panel.

Open the sidebar (top-left arrow) to view the accuracy plot and confusion matrix inline.

---

## 🌐 Live Demo

> 🚀 **[Live Demo](https://cinescope-sentiment-wurpdyydmhk9uhjm66smjq.streamlit.app)** 

---

## 🗂️ Dataset

This project uses the **IMDb Extensive Dataset** available on Kaggle:

- **Source:** [simhyunsu/imdbextensivedataset](https://www.kaggle.com/datasets/simhyunsu/imdbextensivedataset)
- **Size:** 50,000 labelled movie reviews (25k positive / 25k negative)
- **Format:** CSV with columns `review` and `sentiment`

Place the downloaded CSV as `IMDB Dataset.csv` in the project root before training.

---

## 🔧 Development

### Running tests (smoke test)

```bash
python -c "
import pickle, os
pipeline = pickle.load(open('sentiment_model.pkl','rb'))
texts = ['I loved this film!', 'Terrible movie, waste of time.']
preds = pipeline.predict(texts)
probas = pipeline.predict_proba(texts)
for t, p, pr in zip(texts, preds, probas):
    label = 'Positive' if p == 1 else 'Negative'
    print(f'{label} ({max(pr)*100:.1f}%)  |  {t}')
"
```

### Retraining

Simply re-run `python model.py`. The script overwrites both the model pickle and the result images.

---

## 📁 File Reference

| File | Description |
|------|-------------|
| `app.py` | Streamlit UI — loads model, handles input, renders predictions |
| `model.py` | Training script — data loading, vectorisation, training, evaluation, plotting |
| `text_utils.py` | Shared `clean_text()` preprocessor imported by both `app.py` and `model.py` — must ship alongside `app.py` for the pickle to load |
| `requirements.txt` | Pinned Python dependencies |
| `Dockerfile` | Production container definition |
| `sentiment_model.pkl` | Serialised sklearn Pipeline (TF-IDF + LR) |
| `results/accuracy.png` | CV accuracy bar chart + performance summary |
| `results/confusion_matrix.png` | Annotated confusion matrix heatmap |

---

## 📜 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

<div align="center">
Made with ☕ and Python · [GitHub](https://github.com/Ozse97/cinescope-sentiment)
</div>
