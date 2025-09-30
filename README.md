# AI-Driven Customer Feedback Analyzer & Response Generator

## Quickstart

1. Create virtual env and install deps:
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: . .venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

2. Configure LLM (Groq only):
- Copy `.env.example` to `.env` and set:
  - `GROQ_API_KEY`
  - Optional `GROQ_MODEL` (default: `llama-3.1-8b-instant`)

3. Run the app:
```bash
streamlit run streamlit_app.py
```

## Features
- Upload CSV/Excel feedback
- Preprocessing: clean/tokenize
- Sentiment: Hugging Face transformers
- Topics: BERTopic
- (Optional) Emotions
- LLM Response generation (Groq → template fallback)
- Insight aggregation with charts and LLM recommendations

## Project Structure
```
app/
  analysis/
  insights/
  response/
  utils/
streamlit_app.py
pipeline.py
requirements.txt
```

## Sample Data
- `data/sample_feedback.csv` provided for demo.

## Demo Workflow
1. Launch app: `streamlit run streamlit_app.py`
2. Upload `data/sample_feedback.csv` (or your own with a `feedback_text` column).
3. Optionally enable emotion detection in the sidebar.
4. Click Analyze.
5. Review:
   - Results table with sentiment, topic, (emotion), and a draft response.
   - Summary Insights section showing sentiment breakdown pie and topic bar chart.
   - Recommendations list (Groq-generated if configured).
6. Download the enriched CSV using the Download button.

## Notes
- First run will download transformer and embedding models; allow time and network access.
- If no Groq key is set, draft responses and recommendations use safe templates.
- For large files, consider batching or enabling Streamlit caching (already applied).
