# 🚀 AI-Driven Customer Feedback Analyzer & Response Generator

An end-to-end AI-powered system to analyze customer feedback, detect sentiment, extract topics, and auto-generate smart responses — all in a simple Streamlit app.

## ✨ Features

 Upload & Analyze Feedback: Supports CSV/Excel input.
 Data Preprocessing: Cleans and tokenizes feedback text.
 Sentiment Analysis: Hugging Face Transformers.
 Topic Modeling: BERTopic for clustering feedback.
 Emotion Detection (optional).
 LLM Response Generation: Groq-powered (with template fallback).
 Insights & Visualization: Charts + AI-generated recommendations.
 Downloadable Enriched CSV with results.

## 📂 Project Structure
app/
  analysis/        # sentiment, topic, emotion modules
  insights/        # aggregation & chart logic
  response/        # response generation logic
  utils/           # preprocessing helpers
data/
  sample_feedback.csv
streamlit_app.py   # main app entry
pipeline.py        # analysis pipeline
requirements.txt   # dependencies

## ⚡ Quickstart
1. Setup Environment
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: . .venv/Scripts/Activate.ps1
pip install -r requirements.txt

2. Configure LLM (Groq)

Copy .env.example → .env

Set the following:

GROQ_API_KEY = your Groq API key

GROQ_MODEL (optional, default: llama-3.1-8b-instant)

3. Run App
streamlit run streamlit_app.py

## 🧪 Demo Workflow

Launch the app:

streamlit run streamlit_app.py


Upload feedback file (data/sample_feedback.csv included).

(Optional) Enable Emotion Detection in sidebar.

Click Analyze.

Explore Results:

 Table → Sentiment, Topic, Emotion, Draft Response

 Charts → Sentiment pie & Topic bar

 Insights → AI-generated recommendations

Download enriched CSV with all annotations.

## 📊 Example Output

Upload: sample_feedback.csv

Sentiment Distribution → Pie chart

Topic Clusters → Bar chart

Draft Response → Generated using Groq LLM (or fallback template).

(Screenshots/Visuals placeholder — add once available)

## 🔧 Tech Stack

Python 

Streamlit (frontend)

Hugging Face Transformers (sentiment)

BERTopic (topic modeling)

Groq API (LLM responses)

Matplotlib/Altair (visualizations)


## 🤝 Contributing

Contributions are welcome! Please fork the repo and submit a PR.

## 📜 License

MIT License © 2025 Anumitha
