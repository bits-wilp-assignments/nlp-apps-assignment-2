import pandas as pd
from src.util.logging_util import get_logger
from src.core.training import MLClassifierTrainer


class LRegPredictor:
    """Predictor class that encapsulates the trained Logistic Regression model and vectorizer for inference."""

    def __init__(self, data_file_paths: list, label_map: dict, stop_words_lang: str = "english", **kwargs):
        self.logger = get_logger(__name__)
        self.data_file_paths = data_file_paths
        self.stop_words_lang = stop_words_lang
        self.logger.info("Live training mode during initialization of inference LRegPredictor...")
        self.mlTrainer = MLClassifierTrainer(self.data_file_paths, self.stop_words_lang)
        self.vectorizer, self.model, self.metrics = self.mlTrainer.train_logistic_reg_model(**kwargs)
        self.label_map = label_map
        self.logger.info(f"LRPredictor initialized with model accuracy: {self.metrics['accuracy']:.4f}")


    def predict_single(self, text: str):
        """Predict the sentiment category for a single text input.
        Args:
            text (str): The input text to classify.
        Returns:
            dict: A dictionary containing the predicted category and probabilities for each class.
        """
        self.logger.debug(f"Predicting single text: {text[:10]}...")  # Log first 10 chars
        vector = self.vectorizer.transform([text])
        pred_num = int(self.model.predict(vector)[0])
        probs = self.model.predict_proba(vector)[0]

        prediction = {
            "prediction": self.label_map[pred_num],
            "probabilities": {
                self.label_map[int(k)]: float(v)
                for k, v in zip(self.model.classes_, probs)
            },
        }
        self.logger.debug(f"Prediction result: {prediction}")
        return prediction

    def predict_batch(self, texts: list, add_probs: bool = True) -> pd.DataFrame:
        """Predict sentiment categories for a batch of text inputs.
        Args:
            texts (list): A list of input texts to classify.
            add_probs (bool): Whether to include probabilities for each class in the output DataFrame.
        Returns:
            pd.DataFrame: A DataFrame containing the original texts, predicted categories, and optionally probabilities.
        """
        self.logger.debug(f"Predicting batch of size: {len(texts)}")
        vectors = self.vectorizer.transform(texts)
        preds_num = self.model.predict(vectors)
        probs = self.model.predict_proba(vectors)

        df = pd.DataFrame({
            "text": texts,
            "predicted_category": [self.label_map[int(p)] for p in preds_num],
            "max_prob": probs.max(axis=1),
        })
        if add_probs:
            for i, class_label in enumerate(self.model.classes_):
                label_name = self.label_map[int(class_label)]
                df[f"{label_name}_prob"] = probs[:, i]

        self.logger.debug(f"Batch prediction completed with {len(df)} records")
        return df