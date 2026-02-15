import streamlit as st
import requests
import pandas as pd
import altair as alt
import io
from datetime import datetime
from src.config.ui_config import *
from src.util.logging_util import get_logger


class SentimentAnalysisUIApp:
    """Streamlit UI application class for the Sentiment Analysis project.
    This class sets up the Streamlit interface, handles user interactions, and communicates with the backend API for predictions and batch analysis.
    """
    def __init__(self):
        self.logger = get_logger(__name__)
        self.api_url = BACKEND_BASE_URL
        self._setup_page()

    def _setup_page(self):
        self.logger.info("Setting up Streamlit page configuration...")
        st.set_page_config(
            page_title=APP_TITLE,
            page_icon="🎭",
            layout=LAYOUT,
        )
        st.title(APP_TITLE)
        # Custom CSS
        st.markdown(
            """
        <style>
            /* Hide deploy button */
            .stAppDeployButton {
                display: none !important;
                visibility: hidden !important;
            }
            .team-card {
                background: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 15px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                border: 1px solid #e0e0e0;
            }
            .team-name {
                color: #333333;
                font-size: 1.1em;
                font-weight: 600;
                margin-bottom: 5px;
            }
            .team-id {
                color: #666666;
                font-size: 0.9em;
                font-family: 'Courier New', monospace;
            }
        </style>
        """,
            unsafe_allow_html=True,
        )

    def _check_backend_status(self):
        self.logger.info("Checking backend status...")
        st.sidebar.title("Status")
        try:
            resp = requests.get(
                f"{self.api_url}/health",
                timeout=HEALTH_TIMEOUT,
            )
            if resp.status_code == 200:
                st.sidebar.success("Backend Connected")
            else:
                st.sidebar.error("Backend Offline")
                self.logger.error("Backend Offline")
        except Exception:
            st.sidebar.error("Backend Offline")
            self.logger.error("Backend Offline")

    def _single_prediction_tab(self):
        self.logger.info("Rendering Single Prediction tab...")
        st.header("Single Text Prediction")
        text = st.text_area(
            "Enter text to classify:",
            height=SINGLE_TEXT_HEIGHT,
        )
        if st.button("Predict", type="primary") and text.strip():
            with st.spinner("Predicting..."):
                self._predict_single(text.strip())

    def _predict_single(self, text: str):
        self.logger.info(f"Sending single prediction request for text: {text[:30]}...")
        try:
            resp = requests.post(
                f"{self.api_url}/predict/single",
                json={"text": text},
                timeout=TIMEOUT_SINGLE,
            )

            if resp.status_code == 200:
                result = resp.json()

                st.success(f"**Predicted: {result['prediction']}**")

                probs_df = pd.DataFrame(
                    result["probabilities"].items(),
                    columns=["Category", "Probability"],
                )

                # Sort by desired order: Positive, Negative, Neutral
                category_order = ["Positive", "Negative", "Neutral"]
                probs_df["Category"] = pd.Categorical(
                    probs_df["Category"], categories=category_order, ordered=True
                )
                probs_df = probs_df.sort_values("Category")

                st.subheader("Confidence Scores")

                # Create two columns for side-by-side display
                col1, col2, col3 = st.columns(
                    [1, 0.1, 1]
                )  # Add a small spacer column in the middle

                with col1:
                    st.dataframe(
                        probs_df.round(3), use_container_width=True, hide_index=True
                    )
                with col2:
                    st.write("")  # Empty space for separation
                with col3:
                    # Create custom bar chart with reduced bar width
                    chart = (
                        alt.Chart(probs_df)
                        .mark_bar(size=50)
                        .encode(
                            x=alt.X(
                                "Category:N",
                                title="Category",
                                axis=alt.Axis(labelAngle=0),
                                sort=["Positive", "Negative", "Neutral"],
                            ),
                            y=alt.Y(
                                "Probability:Q",
                                title="Probability",
                                scale=alt.Scale(domain=[0, 1]),
                            ),
                            color=alt.Color(
                                "Category:N",
                                legend=None,
                                scale=alt.Scale(
                                    domain=["Positive", "Negative", "Neutral"],
                                    range=["#66BB6A", "#F4511E", "#42A5F5"],
                                ),
                            ),
                        )
                        .properties(height=220)
                    )
                    st.altair_chart(chart, use_container_width=True)
            else:
                st.error(resp.text)

        except requests.exceptions.RequestException:
            st.error("Backend not responding")

    def _batch_analysis_tab(self):
        self.logger.info("Rendering Batch Analysis tab...")
        st.header("Batch File Analysis")
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=BATCH_FILE_TYPES,
            help=HELP_TEXT,
        )
        if uploaded_file and st.button("Analyze Batch", type="primary"):
            with st.spinner("Processing file..."):
                self._analyze_batch(uploaded_file)

    def _analyze_batch(self, uploaded_file):
        self.logger.info(f"Processing batch file: {uploaded_file.name}")
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
                timeout=TIMEOUT_BATCH,
            )

            if resp.status_code == 200:
                self._display_batch_results(resp.json())
            else:
                st.error(resp.text)

        except Exception as e:
            st.error(str(e))

    def _display_batch_results(self, data):
        self.logger.info("Displaying batch analysis results...")
        df = pd.DataFrame(data["results"])

        st.success(f"Analyzed {len(df)} comments")

        self._show_metrics(df, data["summary"])

        st.subheader(f"Top {TOP_CONFIDENTS} Most Confident")

        top_n = df.nlargest(TOP_CONFIDENTS, "max_prob")[
            ["text", "predicted_category", "max_prob"]
        ]

        st.dataframe(top_n.style.format({"max_prob": "{:.1%}"}))
        self._create_download_button(df)

    def _show_metrics(self, df, summary):
        self.logger.info("Showing batch analysis metrics...")
        col1, col2, col3, col4 = st.columns(METRIC["columns"])
        summary_series = pd.Series(summary)

        col1.metric(METRIC["labels"][0], len(df))
        col2.metric(METRIC["labels"][1], summary_series.get("Positive", 0))
        col3.metric(METRIC["labels"][2], summary_series.get("Negative", 0))
        col4.metric(METRIC["labels"][3], summary_series.get("Neutral", 0))

    def _create_download_button(self, df):
        self.logger.info("Creating download button for batch results...")
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)

        st.download_button(
            "Download Results",
            buffer.getvalue(),
            f"sentiment_results_{datetime.now():%Y%m%d_%H%M}.csv",
            "text/csv",
        )

    def _fetch_config(self):
        """Fetch configuration from backend API."""
        try:
            resp = requests.get(
                f"{self.api_url}/config",
                timeout=HEALTH_TIMEOUT,
            )
            if resp.status_code == 200:
                return resp.json()
            else:
                self.logger.error("Failed to fetch config from backend")
                return None
        except Exception as e:
            self.logger.error(f"Error fetching config: {e}")
            return None

    def _about_tab(self):
        self.logger.info("Rendering About tab...")
        st.header("About This Application")

        st.markdown(
            """
        ### Project Overview
        This Sentiment Analysis application automatically classifies text into three sentiment categories:
        **Positive**, **Negative**, and **Neutral**. Built using machine learning techniques,
        it processes both single texts and batch files efficiently.
        """
        )

        st.markdown("---")

        st.subheader("Model Information")

        # Fetch configuration from backend
        config_data = self._fetch_config()

        if config_data and "app_config" in config_data:
            app_config = config_data["app_config"]
            training_config = app_config.get("training", {})
            vectorizer_params = training_config.get("vectorizer_params", {})
            lr_params = training_config.get("logistic_regression_params", {})

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    f"""
                **Vectorization**
                - Method: TF-IDF (Term Frequency-Inverse Document Frequency)
                - Max Features: {vectorizer_params.get('max_features', 'N/A'):,}
                - N-gram Range: {vectorizer_params.get('ngram_range', 'N/A')}
                - Stop Words: {vectorizer_params.get('stop_words', 'N/A').title()}
                """
                )

            with col2:
                label_map = app_config.get("label_map", {})
                categories = ", ".join([v for v in label_map.values()])

                st.markdown(
                    f"""
                **Classification**
                - Algorithm: {app_config.get('model_name', 'N/A')}
                - Max Iterations: {lr_params.get('max_iter', 'N/A'):,}
                - Class Weight: {lr_params.get('class_weight', 'N/A').title()}
                - Categories: {categories}
                """
                )

            # Additional configuration details in expander
            with st.expander("View Full Configuration"):
                st.json(config_data)
        else:
            # Display connection failed status
            st.error("**Connection Failed**")
            st.warning(
                "Unable to fetch configuration from backend. Please ensure the backend server is running and accessible."
            )
            st.info(f"Backend URL: `{self.api_url}/config`")

        st.markdown("---")

        st.subheader("How to Use")

        st.markdown(
            """
        **Single Prediction**
        1. Navigate to the 'Single Prediction' tab
        2. Enter or paste your text in the input box
        3. Click 'Predict' to get instant sentiment analysis
        4. View the predicted sentiment and confidence scores

        **Batch Analysis**
        1. Navigate to the 'Batch Analysis' tab
        2. Upload a CSV or Excel file containing text data
        3. Click 'Analyze Batch' to process all entries
        4. Review the results and download the analyzed data
        """
        )

        st.markdown("---")
        st.header("BITs Pilani - WILP (AI/ML)")
        st.subheader("NLP Applications - Assignment 2")
        st.subheader("Group 55")

        team_members = [
            {"name": "ABHISHEK KUMAR TIWARI", "id": "2024aa05192"},
            {"name": "KRISHANU CHAKRABORTY", "id": "2024aa05193"},
            {"name": "VISWANADHA PAVAN KUMAR", "id": "2024aa05197"},
            {"name": "B VINOD KUMAR", "id": "2024aa05832"},
            {"name": "K ABHINAV", "id": "2024ab05168"},
        ]

        for member in team_members:
            st.markdown(
                f"""
            <div class="team-card">
                <div class="team-name">{member['name']}</div>
                <div class="team-id">{member['id']}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        st.markdown(
            """
        <div style='text-align: center; color: #666; padding: 20px;'>
            Built with ❤️ by <strong>GROUP 55</strong> for WILP NLP Applications Course
        </div>
        """,
            unsafe_allow_html=True,
        )

    # App Run
    def run(self):
        self.logger.info("Running Sentiment Analysis UI App...")
        self._check_backend_status()
        tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Analysis", "About"])
        with tab1:
            self._single_prediction_tab()
        with tab2:
            self._batch_analysis_tab()
        with tab3:
            self._about_tab()


if __name__ == "__main__":
    app = SentimentAnalysisUIApp()
    app.run()
