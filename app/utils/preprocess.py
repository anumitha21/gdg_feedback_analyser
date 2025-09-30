import re
from typing import List
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure NLTK resources are available
try:
	_ = stopwords.words("english")
except LookupError:
	nltk.download("punkt")
	nltk.download("stopwords")

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_TAG_PATTERN = re.compile(r"<.*?>")
NON_ALPHANUM_PATTERN = re.compile(r"[^a-z0-9\s]")
MULTISPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
	"""Basic cleaning for user feedback text while preserving sentiment context."""
	if not isinstance(text, str):
		return ""
	text = text.lower()
	text = URL_PATTERN.sub(" ", text)
	text = HTML_TAG_PATTERN.sub(" ", text)
	# Preserve punctuation that indicates sentiment (exclamation, question marks)
	text = re.sub(r"[^a-z0-9\s!?]", " ", text)
	text = MULTISPACE_PATTERN.sub(" ", text).strip()
	return text


def tokenize_text(text: str, remove_stopwords: bool = True) -> List[str]:
	"""Tokenize feedback text into words, optionally removing stopwords."""
	tokens = word_tokenize(text)
	if remove_stopwords:
		stop_words = set(stopwords.words("english"))
		tokens = [t for t in tokens if t not in stop_words]
	return tokens


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "feedback_text") -> pd.DataFrame:
	"""Return a copy of df with a `clean_text` column.
	If text_col missing, creates empty column.
	"""
	df = df.copy()
	if text_col not in df.columns:
		df[text_col] = ""
	df["clean_text"] = df[text_col].astype(str).map(clean_text)
	return df
