import streamlit as st
import requests
import pandas as pd
import io
import yaml
from pathlib import Path
from datetime import datetime


class SKStreamlitApp:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.api_url = self.config["api"]["base_url"]
        self._setup_page()

    # --------------------
    # Config
    # --------------------

    def _load_config(self, config_path: str):
        with open(Path(config_path)) as f:
            return yaml.safe_load(f)

    def _setup_page(self):
        st.set_page_config(
            page_title=self.config["app"]["name"],
            layout=self.config["app"]["layout"],
        )
        st.title("🧠 " + self.config["app"]["name"])

    # --------------------
    # Backend Status
    # --------------------

    def _check_backend_status(self):
        st.sidebar.title("Status")
        try:
            resp = requests.get(
                f"{self.api_url}/health",
                timeout=self.config["api"]["health_timeout"],
            )
            if resp.status_code == 200:
                st.sidebar.success("✅ Backend Connected")
            else:
                st.sidebar.error("❌ Backend Offline")
        except Exception:
            st.sidebar.error("❌ Backend Offline")

    # --------------------
    # Single Prediction
    # --------------------

    def _single_prediction_tab(self):
        st.header("Single Text Prediction")

        text = st.text_area(
            "Enter text to classify:",
            height=self.config["ui"]["single_text_height"],
        )

        if st.button("🔮 Predict", type="primary") and text.strip():
            with st.spinner("Predicting..."):
                self._predict_single(text.strip())

    def _predict_single(self, text: str):
        try:
            resp = requests.post(
                f"{self.api_url}/predict/single",
                json={"text": text},
                timeout=self.config["api"]["timeout_single"],
            )

            if resp.status_code == 200:
                result = resp.json()

                st.success(f"**Predicted: {result['prediction']}**")

                probs_df = pd.DataFrame(
                    result["probabilities"].items(),
                    columns=["Category", "Probability"],
                )

                st.subheader("Confidence Scores")
                st.bar_chart(probs_df.set_index("Category"))
                st.dataframe(probs_df.round(3))

            else:
                st.error(resp.text)

        except requests.exceptions.RequestException:
            st.error("Backend not responding")

    # --------------------
    # Batch Prediction
    # --------------------

    def _batch_analysis_tab(self):
        st.header("Batch File Analysis")

        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=self.config["ui"]["batch_file_types"],
            help=self.config["ui"]["batch_help_text"],
        )

        if uploaded_file and st.button("🚀 Analyze Batch", type="primary"):
            with st.spinner("Processing file..."):
                self._analyze_batch(uploaded_file)

    def _analyze_batch(self, uploaded_file):
        try:
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type,
                )
            }

            resp = requests.post(
                f"{self.api_url}/predict/batch",
                files=files,
                timeout=self.config["api"]["timeout_batch"],
            )

            if resp.status_code == 200:
                self._display_batch_results(resp.json())
            else:
                st.error(resp.text)

        except Exception as e:
            st.error(str(e))

    # --------------------
    # Results Display
    # --------------------

    def _display_batch_results(self, data):
        df = pd.DataFrame(data["results"])

        st.success(f"✅ Analyzed {len(df)} comments")

        self._show_metrics(df, data["summary"])

        st.subheader(
            f"📈 Top {self.config['ui']['top_n_confident']} Most Confident"
        )

        top_n = df.nlargest(
            self.config["ui"]["top_n_confident"], "max_prob"
        )[
            ["text", "predicted_category", "max_prob"]
        ]

        st.dataframe(top_n.style.format({"max_prob": "{:.1%}"}))

        self._create_download_button(df)

    def _show_metrics(self, df, summary):
        col1, col2, col3, col4 = st.columns(
            self.config["metrics"]["columns"]
        )

        summary_series = pd.Series(summary)

        col1.metric(self.config["metrics"]["labels"][0], len(df))
        col2.metric(self.config["metrics"]["labels"][1], summary_series.get("Positive", 0))
        col3.metric(self.config["metrics"]["labels"][2], summary_series.get("Negative", 0))
        col4.metric(self.config["metrics"]["labels"][3], summary_series.get("Neutral", 0))

    def _create_download_button(self, df):
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)

        st.download_button(
            "💾 Download Results",
            buffer.getvalue(),
            f"sentiment_results_{datetime.now():%Y%m%d_%H%M}.csv",
            "text/csv",
        )

    # --------------------
    # App Run
    # --------------------

    def run(self):
        self._check_backend_status()

        tab1, tab2 = st.tabs(["📝 Single Prediction", "📊 Batch Analysis"])

        with tab1:
            self._single_prediction_tab()

        with tab2:
            self._batch_analysis_tab()

        # st.sidebar.markdown("---")
        # st.sidebar.markdown("*Live-trained model* 🚀")

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🚀 Model Info")
        st.sidebar.markdown("*TF-IDF + Logistic Regression*")

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 👥 Project Group Details")

        st.sidebar.markdown("""
        **Group Number:** 55  

        **ABHISHEK KUMAR TIWARI**  
        📧 2024aa05192 

        **KRISHANU CHAKRABORTY**  
        📧 2024aa05193

        **VISWANADHA PAVAN KUMAR**  
        📧 2024aa05197

        **B VINOD KUMAR**  
        📧 2024aa05832

        **K ABHINAV**  
        📧 2024ab05168
                                                 
        """)



if __name__ == "__main__":
    app = SKStreamlitApp()
    app.run()
