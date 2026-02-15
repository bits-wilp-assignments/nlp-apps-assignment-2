# Sentiment Analysis Application

A comprehensive sentiment analysis application that classifies text into **Positive**, **Negative**, and **Neutral** categories. Built with FastAPI backend and Streamlit frontend, this application provides both single text prediction and batch file analysis capabilities.

## Table of Contents

- [Directory Structure](#directory-structure)
- [Technologies Used](#technologies-used)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Running the Application](#running-the-application)
- [Usage Guide](#usage-guide)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Team](#team)

## Directory Structure

```
sentiment-analysis-app/
├── .streamlit/
│   └── config.toml              # Streamlit theme configuration
├── src/
│   ├── config/
│   │   ├── app_config.py        # Backend configuration
│   │   └── ui_config.py         # Frontend configuration
│   ├── core/
│   │   ├── preprocessing.py     # Text preprocessing functions
│   │   └── training.py          # Model training logic
│   ├── services/
│   │   └── inference.py         # Prediction service
│   └── util/
│       ├── data_loader.py       # Dataset loading utilities
|       └── logging_util.py      # Logging configuration
├── sentiment_app.py             # FastAPI backend application
├── ui_app.py                    # Streamlit frontend application
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## Technologies Used

### Backend

- **FastAPI**: High-performance web framework for building APIs
- **scikit-learn**: Machine learning library for model training
- **pandas**: Data manipulation and analysis
- **NLTK**: Natural language processing toolkit
- **uvicorn**: ASGI server for FastAPI

### Frontend

- **Streamlit**: Interactive web application framework
- **Altair**: Declarative visualization library
- **requests**: HTTP library for API communication

### Machine Learning

- **TF-IDF Vectorization**: Text feature extraction
- **Logistic Regression**: Classification algorithm
- **Balanced Class Weighting**: Handling imbalanced datasets

## Prerequisites

Before setting up the application, ensure you have the following installed:

- **Python 3.8+** (Python 3.9 or 3.10 recommended)
- **pip** (Python package installer)
- **virtualenv** or **venv** (for creating virtual environments)
- **Git** (optional, for cloning the repository)

### System Requirements

- **OS**: macOS, Linux, or Windows
- **RAM**: Minimum 4GB (8GB recommended)
- **Disk Space**: At least 500MB free space

## Setup Instructions

Follow these steps to set up the application from scratch on your local system:

### 1. Clone or Download the Project

```bash
# If using Git
git clone <repository-url>
cd sentiment-analysis-app

# Or download and extract the ZIP file
```

### 2. Create a Virtual Environment

```bash
# Using venv (Python 3.8+)
python3 -m venv venv

# OR using virtualenv
virtualenv venv
```

### 3. Activate the Virtual Environment

```bash
# On macOS/Linux
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

You should see `(venv)` prefix in your terminal prompt.

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install all required packages including:

- FastAPI & Uvicorn (Backend)
- Streamlit (Frontend)
- scikit-learn (Machine Learning)
- pandas, numpy (Data Processing)
- NLTK (NLP)
- And other dependencies

### 5. Download NLTK Data

The application requires specific NLTK resources. Download them using:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

Alternatively, these resources will be downloaded automatically when the application runs for the first time.

### 6. Verify Installation

```bash
# Check Python version
python --version

# Check if packages are installed
pip list | grep streamlit
pip list | grep fastapi
```

## Running the Application

The application consists of two components that need to run simultaneously:

### Starting the Backend

**Terminal 1 - Start the Backend:**

```bash
# Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run backend API
python sentiment_app.py
```

The backend will:

- Train the model automatically (takes ~30-60 seconds)
- Start API server on `http://localhost:8000`
- Display "Backend is ready!" when complete
- Provide interactive API docs at `http://localhost:8000/docs`

**Terminal 2 - Start the Frontend:**

```bash
# Activate virtual environment (in new terminal)
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run Streamlit frontend
streamlit run ui_app.py
```

The frontend will:

- Open automatically in your default browser
- Run on `http://localhost:8501`
- Connect to the backend API

### Important Notes

- **Start Backend First**: Always ensure the backend is running and trained before using the frontend
- **Training Time**: Initial model training takes 30-60 seconds depending on dataset size
- **Port Conflicts**: Ensure ports 8000 and 8501 are available
- **Network Access**: Both applications run on localhost by default

## Usage Guide

### Single Text Prediction

1. Navigate to the **"Single Prediction"** tab
2. Enter or paste your text in the input box
3. Click the **"Predict"** button
4. View the predicted sentiment and confidence scores
5. See the probability distribution chart

**Example:**

- Input: `"This product is amazing! I love it!"`
- Output: `Positive (98.5% confidence)`

### Batch File Analysis

1. Navigate to the **"Batch Analysis"** tab
2. Prepare a CSV or Excel file with a text column
3. Click **"Browse files"** and upload your file
4. Click **"Analyze Batch"**
5. View summary statistics and top predictions
6. Download results using the **"Download Results"** button

**Supported File Formats:**

- CSV (.csv)
- Excel (.xlsx, .xls)

**File Format Requirements:**

- Must contain a column with text data
- Column name should be specified in configuration
- Text should be in a single column

### About Page

- View model configuration and parameters
- Check backend connection status
- Read usage instructions
- See team information

## API Endpoints

The FastAPI backend provides interactive API documentation that allows you to explore and test all endpoints directly in your browser:

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Health Check

```http
GET http://localhost:8000/health
```

Returns API health status.

### Get Configuration

```http
GET http://localhost:8000/config
```

Returns current application and model configuration.

### Single Prediction

```http
POST http://localhost:8000/predict/single
Content-Type: application/json

{
  "text": "Your text here"
}
```

**Response:**

```json
{
  "text": "Your text here",
  "prediction": "Positive",
  "probabilities": {
    "Positive": 0.856,
    "Negative": 0.089,
    "Neutral": 0.055
  }
}
```

### Batch Prediction

```http
POST http://localhost:8000/predict/batch
Content-Type: multipart/form-data

file: <your-file.csv>
```

**Response:**

```json
{
  "results": [...],
  "summary": {
    "Positive": 45,
    "Negative": 23,
    "Neutral": 12
  }
}
```

## Configuration

### Streamlit Theme (`.streamlit/config.toml`)

- **Primary color**: `#1976D2` (Professional Blue)
- **Background colors**: White main area, light gray sidebar
- Customizable via TOML file

### Backend Config (`src/config/app_config.py`)

- API settings
- CORS configuration
- Model parameters
- Label mappings

### Frontend Config (`src/config/ui_config.py`)

- UI constants
- Layout settings
- Timeout values
- Display preferences

## Team

**BITs Pilani - WILP (AI/ML)**
**NLP Applications - Assignment 2**
**Group 55**

- **ABHISHEK KUMAR TIWARI** - 2024aa05192
- **KRISHANU CHAKRABORTY** - 2024aa05193
- **VISWANADHA PAVAN KUMAR** - 2024aa05197
- **B VINOD KUMAR** - 2024aa05832
- **K ABHINAV** - 2024ab05168

---

Built with ❤️ by Group 55 for WILP NLP Applications Course
