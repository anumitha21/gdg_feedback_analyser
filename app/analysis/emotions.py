from typing import List
import pandas as pd
from transformers import pipeline


class EmotionDetector:
	def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
		self.model_name = model_name
		self._pipeline = pipeline("text-classification", model=self.model_name, top_k=None)

	def predict(self, texts: List[str]) -> List[str]:
		results = self._pipeline(texts)
		labels = []
		for res in results:
			if isinstance(res, list):
				res = max(res, key=lambda x: x.get("score", 0))
			labels.append(res["label"])
		return labels

	def add_to_dataframe(self, df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
		df = df.copy()
		df["emotion_label"] = self.predict(df[text_col].fillna("").astype(str).tolist())
		return df
