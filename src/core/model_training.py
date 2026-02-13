from src.core.data_processing import *
import yaml
import pandas as pd
import numpy as np

from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ==============================
# Load Config
# ==============================

def load_training_config(config_filename: str = "config/config.yaml") -> dict:
    config_path = Path(__file__).parent.parent.parent / config_filename
    with open(config_path) as f:
        return yaml.safe_load(f)


# ==============================
# Training Function
# ==============================

def train_sklearn_model(df=None, config_filename: str = "config/config.yaml") -> Tuple:

    config = load_training_config(config_filename)

    # Load data if not provided
    if df is None:
        print("📂 Loading data via preprocessor...")
        preprocessor = SimplePreprocessor()
        df = preprocessor.load_and_preprocess()

    if "clean_text" not in df.columns:
        df["clean_text"] = df["tokens"].apply(" ".join)

    print("📊 Preparing data...")

    X = df[config["data"]["text_column"]]
    y = df[config["data"]["label_column"]]

    # ==============================
    # Train-Test Split
    # ==============================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
        stratify=y,
    )

    # ==============================
    # TF-IDF Vectorization
    # ==============================

    print("🔤 Creating TF-IDF features...")

    vectorizer = TfidfVectorizer(
        max_features=config["model"].get("vocab_size", 20000),
        ngram_range=(1, 2),   # unigrams + bigrams
        stop_words="english"
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # ==============================
    # Logistic Regression
    # ==============================

    print("🧠 Training Logistic Regression...")

    model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)


    # model = LogisticRegression(
    #     max_iter=config["training"].get("max_iter", 1000),
    #     class_weight=config["training"].get("class_weight", None),
    #     n_jobs=-1
    # )

    model.fit(X_train_tfidf, y_train)

    # ==============================
    # Evaluation
    # ==============================

    print("🧪 Evaluating...")

    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"✅ Accuracy: {accuracy:.4f}")

    return vectorizer, model, {
        "accuracy": float(accuracy),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "classes": list(model.classes_)
    }


# ==============================
# Run
# ==============================

if __name__ == "__main__":
    vectorizer, model, metrics = train_sklearn_model()
    print("Metrics:", metrics)
