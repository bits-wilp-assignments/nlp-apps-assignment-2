import pandas as pd
import yaml
from pathlib import Path
from src.core.model_training import train_sklearn_model


class SKPredictor:
    def __init__(self, config_filename: str = "config/config.yaml"):
        config_path = Path(__file__).parent.parent.parent / config_filename

        self.config = self._load_config(config_path)

        print("🔥 LIVE TRAINING MODE")

        # Now returns: vectorizer, model, metrics
        self.vectorizer, self.classifier, self.metrics = train_sklearn_model()

        self.label_map = self.config["predictor"]["label_map"]

        print(f"✅ Predictor ready! Accuracy: {self.metrics['accuracy']:.4f}")

    def _load_config(self, config_path: str):
        with open(Path(config_path)) as f:
            return yaml.safe_load(f)

    def predict_single(self, text: str):

        # Transform text using TF-IDF
        vector = self.vectorizer.transform([text])

        pred_num = int(self.classifier.predict(vector)[0])
        probs = self.classifier.predict_proba(vector)[0]

        return {
            "prediction": self.label_map[pred_num],
            "probabilities": {
                self.label_map[int(k)]: float(v)
                for k, v in zip(self.classifier.classes_, probs)
            },
        }

    def predict_batch(self, texts: list):

        vectors = self.vectorizer.transform(texts)

        preds_num = self.classifier.predict(vectors)
        probs = self.classifier.predict_proba(vectors)

        df = pd.DataFrame({
            "text": texts,
            "predicted_category": [self.label_map[int(p)] for p in preds_num],
            "max_prob": probs.max(axis=1),
        })

        if self.config["batch"]["add_prob_columns"]:
            for i, class_label in enumerate(self.classifier.classes_):
                label_name = self.label_map[int(class_label)]
                df[f"{label_name}_prob"] = probs[:, i]

        return df
