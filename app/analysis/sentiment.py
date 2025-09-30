from typing import List, Dict
import pandas as pd
import re


class SentimentAnalyzer:
	"""Hybrid sentiment analyzer using rule-based and ML approaches."""

	def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest", device: int | None = None):
		self.model_name = model_name
		self._pipeline = None
		self.is_rating_model = False
		
		# Try to load transformer model
		try:
			from transformers import pipeline
			self._pipeline = pipeline(
				"sentiment-analysis",
				model=self.model_name,
				top_k=None,
				device=device if device is not None else -1,
			)
			print("Loaded transformer sentiment model")
		except Exception as e:
			print(f"Failed to load transformer model: {e}. Using rule-based approach.")
		
		# Define sentiment keywords
		self.negative_words = {
			'bad', 'terrible', 'awful', 'horrible', 'disappointed', 'frustrated', 'angry',
			'broken', 'damaged', 'late', 'slow', 'poor', 'worst', 'hate', 'dislike',
			'problem', 'issue', 'complaint', 'refund', 'return', 'failed', 'error',
			'confusing', 'difficult', 'hard', 'impossible', 'useless', 'waste',
			'expensive', 'overpriced', 'ripoff', 'scam', 'cheap', 'low quality',
			'long lines', 'wait', 'delayed', 'missing', 'out of stock', 'unavailable'
		}
		
		self.positive_words = {
			'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love',
			'perfect', 'best', 'awesome', 'outstanding', 'brilliant', 'superb',
			'helpful', 'friendly', 'quick', 'fast', 'easy', 'simple', 'convenient',
			'satisfied', 'happy', 'pleased', 'impressed', 'recommend', 'worth',
			'quality', 'reliable', 'professional', 'responsive', 'supportive'
		}

	def _rule_based_sentiment(self, text: str) -> Dict:
		"""Rule-based sentiment analysis using keyword matching."""
		text_lower = text.lower()
		
		# Count positive and negative words
		pos_count = sum(1 for word in self.positive_words if word in text_lower)
		neg_count = sum(1 for word in self.negative_words if word in text_lower)
		
		# Check for negation patterns
		negation_words = {'not', 'no', 'never', 'none', 'nothing', 'nobody', 'nowhere'}
		words = text_lower.split()
		
		for i, word in enumerate(words):
			if word in negation_words and i + 1 < len(words):
				next_word = words[i + 1]
				if next_word in self.positive_words:
					pos_count -= 1
					neg_count += 1
				elif next_word in self.negative_words:
					neg_count -= 1
					pos_count += 1
		
		# Determine sentiment
		if neg_count > pos_count:
			return {"label": "NEGATIVE", "score": min(0.9, 0.5 + (neg_count - pos_count) * 0.1)}
		elif pos_count > neg_count:
			return {"label": "POSITIVE", "score": min(0.9, 0.5 + (pos_count - neg_count) * 0.1)}
		else:
			return {"label": "NEUTRAL", "score": 0.5}

	def predict(self, texts: List[str]) -> List[Dict]:
		"""Return list of dicts with label and score using hybrid approach."""
		results = []
		
		for text in texts:
			if not text or len(text.strip()) == 0:
				results.append({"label": "NEUTRAL", "score": 0.5})
				continue
			
			# Try transformer model first
			if self._pipeline is not None:
				try:
					res = self._pipeline([text])[0]
					if isinstance(res, list):
						res = max(res, key=lambda x: x.get("score", 0))
					
					label = res["label"].upper()
					score = float(res["score"])
					
					# Convert common labels
					if "NEGATIVE" in label or "NEG" in label:
						label = "NEGATIVE"
					elif "POSITIVE" in label or "POS" in label:
						label = "POSITIVE"
					else:
						label = "NEUTRAL"
					
					results.append({"label": label, "score": score})
					continue
				except Exception as e:
					print(f"Transformer model failed for text: {e}")
			
			# Fallback to rule-based
			results.append(self._rule_based_sentiment(text))
		
		return results

	def add_to_dataframe(self, df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
		"""Add `sentiment_label` and `sentiment_score` columns to df."""
		preds = self.predict(df[text_col].fillna("").astype(str).tolist())
		labels = [p["label"] for p in preds]
		scores = [p["score"] for p in preds]
		df = df.copy()
		df["sentiment_label"] = labels
		df["sentiment_score"] = scores
		return df
