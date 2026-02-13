import pandas as pd
import yaml
import uvicorn

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from io import BytesIO
from typing import Dict, List
from src.core.inference import SKPredictor

from src.core.model_training import train_sklearn_model


# =========================
# Pydantic Models
# =========================

class TextRequest(BaseModel):
    text: str


# =========================
# Main API Class
# =========================

class SKSentimentAPI:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.app = FastAPI(title=self.config["server"]["title"])

        self._setup_cors()
        self._init_predictor()
        self._setup_routes()

    # ---------------------
    # Config
    # ---------------------

    def _load_config(self, path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    # ---------------------
    # Middleware
    # ---------------------

    def _setup_cors(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config["cors"]["allow_origins"],
            allow_credentials=True,
            allow_methods=self.config["cors"]["allow_methods"],
            allow_headers=self.config["cors"]["allow_headers"],
        )

    # ---------------------
    # Model Init
    # ---------------------

    def _init_predictor(self):
        print("🚀 Starting BERT Sentiment Service...")
        self.predictor = SKPredictor()
        print("✅ Model ready for inference!")

    # ---------------------
    # Routes
    # ---------------------

    def _setup_routes(self):

        @self.app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "model": "bert-sentiment",
            }

        @self.app.post("/predict/single")
        async def predict_single(payload: TextRequest):
            """
            Expects:
            {
              "text": "some input text"
            }
            """
            if not payload.text.strip():
                raise HTTPException(status_code=400, detail="Text cannot be empty")

            return self.predictor.predict_single(payload.text)

        @self.app.post("/predict/batch")
        async def predict_batch(file: UploadFile = File(...)):
            return await self._handle_batch_upload(file)

    # ---------------------
    # Batch Handler
    # ---------------------

    async def _handle_batch_upload(self, file: UploadFile) -> Dict:
        try:
            contents = await file.read()
            filename = file.filename.lower()

            if filename.endswith(".csv"):
                df = pd.read_csv(BytesIO(contents))
            elif filename.endswith((".xls", ".xlsx")):
                df = pd.read_excel(BytesIO(contents))
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Only CSV and Excel files are supported",
                )

            text_columns: List[str] = self.config["batch"]["text_columns"]
            text_col = next((c for c in text_columns if c in df.columns), None)

            if not text_col:
                raise HTTPException(
                    status_code=400,
                    detail=f"File must contain one of these columns: {text_columns}",
                )

            texts = (
                df[text_col]
                .dropna()
                .astype(str)
                .str.strip()
                .tolist()
            )

            if not texts:
                raise HTTPException(status_code=400, detail="No valid text rows found")

            result_df = self.predictor.predict_batch(texts)

            return {
                "processed_count": len(result_df),
                "summary": result_df["predicted_category"]
                .value_counts()
                .to_dict(),
                "results": result_df.to_dict(orient="records"),
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


# =========================
# App Export (IMPORTANT)
# =========================

api = SKSentimentAPI()
app = api.app


# =========================
# Local Run
# =========================

if __name__ == "__main__":
    print("🚀 Running backend directly...")
    uvicorn.run(
        app,
        host=api.config["server"]["host"],
        port=api.config["server"]["port"],
        reload=False,   # 🔥 IMPORTANT
    )