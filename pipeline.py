from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import pandas as pd

from app.utils.preprocess import preprocess_dataframe
from app.analysis.sentiment import SentimentAnalyzer
from app.analysis.topics import TopicModeler

try:
	from app.analysis.emotions import EmotionDetector  # optional
except Exception:  # noqa: S110
	EmotionDetector = None  # type: ignore

try:
	from app.response.generator import ResponseGenerator
except Exception:
	ResponseGenerator = None  # type: ignore

try:
	from app.insights.aggregate import InsightAggregator
except Exception:
	InsightAggregator = None  # type: ignore


@dataclass
class PipelineConfig:
	sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
	embedding_model: str = "all-MiniLM-L6-v2"
	use_emotions: bool = False


class FeedbackPipeline:
	def __init__(self, config: PipelineConfig | None = None):
		self.config = config or PipelineConfig()
		self.sentiment = SentimentAnalyzer(model_name=self.config.sentiment_model)
		self.topics = TopicModeler(embedding_model=self.config.embedding_model)
		self.emotions = EmotionDetector() if (self.config.use_emotions and EmotionDetector) else None
		self.responder = ResponseGenerator() if ResponseGenerator else None
		self.aggregator = InsightAggregator() if InsightAggregator else None

	def run(self, df: pd.DataFrame, text_col: str = "feedback_text") -> pd.DataFrame:
		# Preprocess
		df = preprocess_dataframe(df, text_col=text_col)
		# Sentiment
		df = self.sentiment.add_to_dataframe(df, text_col="clean_text")
		# Topics
		df = self.topics.add_to_dataframe(df, text_col="clean_text")
		# Emotions (optional)
		if self.emotions is not None:
			df = self.emotions.add_to_dataframe(df, text_col="clean_text")
		# Responses
		if self.responder is not None:
			df = self.responder.add_to_dataframe(
				df,
				text_col=text_col,
				sentiment_col="sentiment_label",
				topic_col="topic",
			)
		# Insights
		if self.aggregator is not None:
			insights = self.aggregator.aggregate(df)
			df.attrs["insights"] = insights
		return df
