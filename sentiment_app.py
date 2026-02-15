import pandas as pd
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from io import BytesIO
from typing import Dict
from src.services.inference import LRegPredictor
from src.util.data_loader import load_dataset
from src.util.logging_util import get_logger
from src.config.app_config import *
from src.config import ui_config


logger = get_logger(__name__)
dataset_paths = load_dataset(KAGGLE_DATASET)


class TextRequest(BaseModel):
    """Pydantic model for single text prediction request."""

    text: str


class SentimentAnalysisApp:
    """Main application class for the Sentiment Analysis API."""

    def __init__(self, data_file_paths: list):
        self.data_file_paths = data_file_paths
        self.logger = logger
        logger.info(
            f"Initializing SentimentAnalysisApp with dataset: {self.data_file_paths}"
        )
        self.app = FastAPI(title=APP_TITLE)
        self._setup_cors()
        self._init_predictor()
        self._setup_routes()

    def _setup_cors(self):
        self.app.add_middleware(CORSMiddleware, **CORS_SETTINGS)

    def _init_predictor(self):
        self.logger.info("Initializing LRegPredictor for inference...")
        training_params = {
            "column_names": COLUMN_NAMES,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "vectorizer_params": VECTORIZER_PARAMS,
            "logistic_regression_params": LOGISTIC_REGRESSION_PARAMS,
        }
        self.predictor = LRegPredictor(
            self.data_file_paths,
            label_map=LABEL_MAP,
            stop_words_lang=STOP_WORDS_LANG,
            **training_params,
        )
        self.logger.info("LRegPredictor initialized successfully.")

    def _setup_routes(self):
        @self.app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "model": MODEL_NAME,
            }

        @self.app.get("/config")
        async def get_config():
            """Get application and UI configurations."""
            return {
                "app_config": {
                    "model_name": MODEL_NAME,
                    "kaggle_dataset": KAGGLE_DATASET,
                    "column_names": COLUMN_NAMES,
                    "label_map": LABEL_MAP,
                    "stop_words_lang": STOP_WORDS_LANG,
                    "training": {
                        "test_size": TEST_SIZE,
                        "random_state": RANDOM_STATE,
                        "vectorizer_params": VECTORIZER_PARAMS,
                        "logistic_regression_params": LOGISTIC_REGRESSION_PARAMS,
                    },
                    "prediction": {
                        "add_prob_column": ADD_PROB_COLUMN,
                        "text_columns": TEXT_COLUMNS,
                    },
                    "server": {
                        "app_title": APP_TITLE,
                        "host": HOST_NAME,
                        "port": PORT,
                        "cors_settings": CORS_SETTINGS,
                    },
                },
                "ui_config": {
                    "backend_base_url": ui_config.BACKEND_BASE_URL,
                    "app_title": ui_config.APP_TITLE,
                    "layout": ui_config.LAYOUT,
                    "single_text_height": ui_config.SINGLE_TEXT_HEIGHT,
                    "health_timeout": ui_config.HEALTH_TIMEOUT,
                    "timeout_single": ui_config.TIMEOUT_SINGLE,
                    "timeout_batch": ui_config.TIMEOUT_BATCH,
                    "batch_file_types": ui_config.BATCH_FILE_TYPES,
                    "help_text": ui_config.HELP_TEXT,
                    "top_confidents": ui_config.TOP_CONFIDENTS,
                    "metric": ui_config.METRIC,
                },
            }

        @self.app.post("/predict/single")
        async def predict_single(payload: TextRequest):
            if not payload.text.strip():
                raise HTTPException(status_code=400, detail="Text cannot be empty")

            return self.predictor.predict_single(payload.text)

        @self.app.post("/predict/batch")
        async def predict_batch(file: UploadFile = File(...)):
            return await self._handle_batch_upload(file)

    # Batch Handler
    async def _handle_batch_upload(self, file: UploadFile) -> Dict:
        logger.info(f"Received batch prediction request with file: {file.filename}")
        try:
            contents = await file.read()
            filename = file.filename.lower()

            if filename.endswith(".csv"):
                df = pd.read_csv(BytesIO(contents))
            elif filename.endswith((".xls", ".xlsx")):
                df = pd.read_excel(BytesIO(contents))
            else:
                logger.error(f"Unsupported file type uploaded: {filename}")
                raise HTTPException(
                    status_code=400,
                    detail="Only CSV and Excel files are supported",
                )

            text_col = next((c for c in TEXT_COLUMNS if c in df.columns), None)
            if not text_col:
                logger.error("No valid text column found in the uploaded file.")
                raise HTTPException(
                    status_code=400,
                    detail=f"File must contain one of these columns: {TEXT_COLUMNS}",
                )

            texts = df[text_col].dropna().astype(str).str.strip().tolist()

            if not texts:
                logger.warning("No valid text rows found in the uploaded file.")
                raise HTTPException(status_code=400, detail="No valid text rows found")

            result_df = self.predictor.predict_batch(texts, add_probs=ADD_PROB_COLUMN)
            logger.info(f"Batch prediction completed for {len(result_df)} records.")
            return {
                "processed_count": len(result_df),
                "summary": result_df["predicted_category"].value_counts().to_dict(),
                "results": result_df.to_dict(orient="records"),
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


# Initialize the app with the dataset path
app_instance = SentimentAnalysisApp(dataset_paths)
app = app_instance.app


# Local Run
if __name__ == "__main__":
    logger.info(f"Starting Sentiment Analysis API on {HOST_NAME}:{PORT}...")
    uvicorn.run(
        app,
        host=HOST_NAME,
        port=PORT,
        reload=False,
    )
