import logging

# -------------------- Logging configuration -------------------
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_FILE = None  # e.g. 'logs/app.log'

# --------------------- Data Configuration ----------------------
KAGGLE_DATASET = "saurabhshahane/twitter-sentiment-dataset"
COLUMN_NAMES = {
    "text_column": "clean_text",
    "label_column": "category",
    "tokens_column": "tokens",
}
LABEL_MAP = {-1: "Negative", 0: "Neutral", 1: "Positive"}

# -------------------- Preprocessing Configuration -------------------
STOP_WORDS_LANG = "english"

# --------------------- Training Configuration ----------------------
MODEL_NAME = "Logistic Regression"
TEST_SIZE = 0.2
RANDOM_STATE = 42
# Vectorizer parameters
VECTORIZER_PARAMS = {
    "max_features": 20000,
    "ngram_range": (1, 2),  # unigrams + bigrams
    "stop_words": STOP_WORDS_LANG,
}
# Logistic Regression parameters
LOGISTIC_REGRESSION_PARAMS = {"max_iter": 1000, "class_weight": "balanced"}

# --------------------- Prediction Configuration ----------------------
ADD_PROB_COLUMN = True
TEXT_COLUMNS = ["text", "comment", "review", "content", "clean_text"]

# -------------------- Server Configuration -------------------
APP_TITLE = "Sentiment Analysis API"
HOST_NAME = "0.0.0.0"
PORT = 8000
CORS_SETTINGS = {
    "allow_origins": ["http://localhost:8501"],
    "allow_methods": ["*"],
    "allow_headers": ["*"],
    "allow_credentials": True,
}
