from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


class TopicModeler:
	def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", seed: int = 42):
		self.seed = seed
		self._model = None

	def _simple_topic_modeling(self, texts: List[str]) -> pd.DataFrame:
		"""Simple topic modeling using TF-IDF and K-means for small datasets."""
		n_texts = len(texts)
		
		if n_texts <= 1:
			return pd.DataFrame({"topic": [0] * n_texts, "topic_prob": [1.0] * n_texts})
		
		# Use TF-IDF for feature extraction
		vectorizer = TfidfVectorizer(
			max_features=100,
			stop_words='english',
			ngram_range=(1, 2),
			min_df=1
		)
		
		try:
			tfidf_matrix = vectorizer.fit_transform(texts)
			
			# Determine number of topics (max 3 for small datasets)
			n_topics = min(3, max(1, n_texts // 2))
			
			# Use K-means clustering
			kmeans = KMeans(n_clusters=n_topics, random_state=self.seed, n_init=10)
			topics = kmeans.fit_predict(tfidf_matrix)
			
			# Calculate topic probabilities based on distance to centroids
			distances = kmeans.transform(tfidf_matrix)
			# Convert distances to probabilities (closer = higher probability)
			probs = 1 / (1 + distances.min(axis=1))
			probs = probs / probs.sum() * n_texts  # Normalize
			
			return pd.DataFrame({
				"topic": topics,
				"topic_prob": probs
			})
			
		except Exception as e:
			print(f"Simple topic modeling failed: {e}. Using fallback.")
			return pd.DataFrame({
				"topic": [0] * n_texts,
				"topic_prob": [1.0] * n_texts
			})

	def fit_transform(self, texts: List[str]) -> pd.DataFrame:
		"""Fit topic model and return a DataFrame with topic labels and probabilities."""
		n_texts = len(texts)
		
		# For small datasets, use simple approach
		if n_texts < 10:
			return self._simple_topic_modeling(texts)
		
		# For larger datasets, try BERTopic with better error handling
		try:
			from bertopic import BERTopic
			from sentence_transformers import SentenceTransformer
			from umap import UMAP
			from hdbscan import HDBSCAN
			
			embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
			
			umap_model = UMAP(
				n_components=5,
				n_neighbors=min(15, n_texts - 1),
				min_dist=0.0,
				metric='cosine',
				random_state=self.seed,
				verbose=False
			)
			
			hdbscan_model = HDBSCAN(
				min_cluster_size=10,
				min_samples=5,
				metric='euclidean',
				cluster_selection_method='eom'
			)
			
			model = BERTopic(
				embedding_model=embedding_model,
				umap_model=umap_model,
				hdbscan_model=hdbscan_model,
				calculate_probabilities=True,
				verbose=False
			)
			self._model = model
			topics, probs = model.fit_transform(texts)
			return pd.DataFrame({"topic": topics, "topic_prob": [float(p.max()) if p is not None else 0.0 for p in probs]})
			
		except Exception as e:
			print(f"BERTopic failed: {e}. Using simple topic modeling.")
			return self._simple_topic_modeling(texts)

	def get_model(self):
		"""Return the fitted model if available."""
		if self._model is None:
			raise RuntimeError("Topic model is not fitted yet.")
		return self._model

	def add_to_dataframe(self, df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
		res = self.fit_transform(df[text_col].fillna("").astype(str).tolist())
		df = df.copy()
		df["topic"] = res["topic"].values
		df["topic_prob"] = res["topic_prob"].values
		return df
