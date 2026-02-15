import pandas as pd
import nltk
from src.util.logging_util import get_logger
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

# Fixed patterns
URL_PATTERN = re.compile(r"http\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<.*?>")


class SentimentDataPreprocessor:
    """Preprocessor class for loading and preprocessing sentiment analysis data.
    This class handles text normalization, tokenization, stopword removal, and lemmatization.
    """

    def __init__(self, data_file_paths: list, stop_words_lang: str = "english"):
        self.logger = get_logger(__name__)
        self.data_file_paths = data_file_paths
        self.stop_words_lang = stop_words_lang
        self._init_nltk()
        self.logger.info(
            f"SentimentDataPreprocessor initialized with data file: {self.data_file_paths}"
        )

    def _init_nltk(self):
        self.logger.info("Initializing NLTK resources...")
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)
        self.stop_words = set(stopwords.words(self.stop_words_lang))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_text(self, text: str):
        """Preprocess a single text string by normalizing, tokenizing, removing stopwords, and lemmatizing.
        Args:
            text (str): The input text to preprocess.
        Returns:
            list: A list of preprocessed tokens.
        """
        self.logger.debug(f"Preprocessing text: {text[:10]}...")  # Log first 10 chars
        # Normalize
        text = str(text).lower()
        text = URL_PATTERN.sub(" ", text)
        HTML_PATTERN.sub(" ", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        self.logger.debug(f"Normalized text: {text[:10]}...")  # Log first 10 chars

        # Tokenize
        tokens = word_tokenize(text)
        self.logger.debug(f"Tokenized text: {tokens[:5]}...")  # Log first 5 tokens

        # Filter stopwords + short words
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 1]
        self.logger.debug(f"Filtered tokens: {tokens[:5]}...")  # Log first 5 tokens

        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        self.logger.debug(f"Lemmatized tokens: {tokens[:5]}...")  # Log first 5 tokens

        return tokens

    def load_and_preprocess(self, **kwargs) -> pd.DataFrame:
        """Load data from the specified file paths and preprocess the text data.
        Args:
            **kwargs: Keyword arguments containing column names and preprocessing parameters.
        Returns:
            pd.DataFrame: A DataFrame containing the original text, preprocessed tokens, and labels.
        """
        if not self.data_file_paths:
            raise FileNotFoundError(f"File not found: {self.data_file_paths}")

        self.logger.info(f"Loading data from file: {self.data_file_paths}")
        df = pd.concat(
            [pd.read_csv(f) for f in self.data_file_paths], ignore_index=True
        )

        self.logger.info(f"Preprocessing '{kwargs.get('text_column')}'...")
        df[kwargs.get("token_column")] = (
            df[kwargs.get("text_column")].astype(str).apply(self.preprocess_text)
        )

        # Clean NaNs (hardcoded logic)
        self.logger.info("Cleaning NaNs...")
        df = df.dropna(subset=[kwargs.get("text_column"), kwargs.get("label_column")])

        self.logger.info(f"Shape: {df.shape}")
        self.logger.info(
            f"Labels: {df[kwargs.get('label_column')].value_counts().to_dict()}"
        )
        return df
