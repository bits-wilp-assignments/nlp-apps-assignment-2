BACKEND_BASE_URL = "http://localhost:8000"
APP_TITLE = "Sentiment Analysis App"
LAYOUT = "wide"
SINGLE_TEXT_HEIGHT = 150
HEALTH_TIMEOUT = 3
TIMEOUT_SINGLE = 10
TIMEOUT_BATCH = 60
BATCH_FILE_TYPES = ["csv", "xlsx"]
HELP_TEXT = "File must contain a text column such as: text, comment, review, content, or clean_text"
TOP_CONFIDENTS = 10
METRIC = {
    "columns": 4,
    "labels": ["Total", "Positive", "Negative", "Neutral"]
}