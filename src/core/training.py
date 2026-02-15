from src.core.preprocessing import SentimentDataPreprocessor
from src.util.logging_util import get_logger
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class MLClassifierTrainer:
    """Trainer class for training a Logistic Regression model for sentiment analysis.
    This class handles loading and preprocessing the data, creating TF-IDF features, training the model, and evaluating its performance.
    """

    def __init__(self, data_file_paths: list, stop_words_lang: str = "english"):
        self.logger = get_logger(__name__)
        self.data_file_paths = data_file_paths
        self.stop_words_lang = stop_words_lang
        self.preprocessor = SentimentDataPreprocessor(self.data_file_paths, self.stop_words_lang)
        self.logger.info(f"MLClassifierTrainer initialized with data file: {self.data_file_paths}")


    def train_logistic_reg_model(self, **kwargs) -> Tuple:
        """Train a Logistic Regression model using the provided data and parameters.
        Args:
            **kwargs: Keyword arguments containing column names, test size, random state, vectorizer parameters, and logistic regression parameters.
        Returns:
            Tuple: A tuple containing the trained vectorizer, model, and a dictionary of evaluation metrics.
        """
        self.logger.info("No DataFrame provided, loading and preprocessing data...")
        df = self.preprocessor.load_and_preprocess(**kwargs.get("column_names"))

        if kwargs.get("column_names").get("text_column") not in df.columns:
            df[kwargs.get("column_names").get("text_column")] = df[kwargs.get("column_names").get("tokens_column")].apply(" ".join)

        X = df[kwargs.get("column_names").get("text_column")]
        y = df[kwargs.get("column_names").get("label_column")]

        # Train-test split
        self.logger.info("Performing train-test split...")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=kwargs.get("test_size"),
            random_state=kwargs.get("random_state"),
            stratify=y,
        )
        self.logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

        # TF-IDF Vectorization
        self.logger.info("Creating TF-IDF features...")
        vectorizer = TfidfVectorizer(**kwargs.get("vectorizer_params"))

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        self.logger.info(f"TF-IDF features created with vocabulary size: {len(vectorizer.vocabulary_)}")

        # Logistic Regression
        self.logger.info("Training Logistic Regression model...")
        model = LogisticRegression(**kwargs.get("logistic_regression_params"))
        model.fit(X_train_tfidf, y_train)

        # Evaluation
        self.logger.info("Evaluating model...")
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)

        self.logger.info(f"Training completed with accuracy: {accuracy:.4f}")

        return vectorizer, model, {
            "accuracy": float(accuracy),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "classes": list(model.classes_)
        }
