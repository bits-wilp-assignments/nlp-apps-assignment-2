#!/bin/bash
echo "Starting BERT Sentiment App..."

# Terminal 1: Backend (trains model)
echo "🚀 Starting Backend (training model...)"
python -m app.backend &

# Wait for backend
sleep 35

# Terminal 2: Frontend
echo "🌐 Starting Frontend"
streamlit run app/frontend.py --server.port 8501
