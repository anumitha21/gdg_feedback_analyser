from __future__ import annotations
import os
from typing import List
import pandas as pd
from pydantic import BaseModel, Field

# Groq only
try:
	from groq import Groq
	_groq_available = True
except Exception:
	_groq_available = False

RESPONSE_SYSTEM_PROMPT = (
	"You are a helpful, empathetic customer support assistant. "
	"Write concise, professional replies that acknowledge concerns, take ownership, and propose next steps."
)


class ResponseInput(BaseModel):
	feedback: str = Field(default="")
	sentiment: str = Field(default="")
	topic: str | int = Field(default="")


def _default_reply(feedback: str) -> str:
	return (
		"Thank you for sharing your feedback. We truly appreciate you taking the time to help us improve. "
		"Our team will review this and follow up with any next steps."
	)


class ResponseGenerator:
	def __init__(self, model: str | None = None):
		self.groq_model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
		self.use_groq = _groq_available and bool(os.getenv("GROQ_API_KEY"))
		self._groq = Groq(api_key=os.getenv("GROQ_API_KEY")) if self.use_groq else None

	def _build_user_prompt(self, item: ResponseInput) -> str:
		return (
			"Create a highly specific, personalized customer service response that directly addresses each issue mentioned in the feedback.\n\n"
			f"Customer Feedback: {item.feedback}\n"
			f"Sentiment: {item.sentiment}\n"
			f"Topic Category: {item.topic}\n\n"
			"Response Requirements:\n"
			"- Start with specific acknowledgment of the exact issues mentioned\n"
			"- For delivery problems: offer specific solutions like 'We'll provide a tracking number within 2 hours' or 'Free expedited shipping on your next order'\n"
			"- For quality issues: offer specific remedies like 'We'll send a replacement via overnight shipping' or 'Full refund plus 20% discount code'\n"
			"- For service problems: provide specific actions like 'Our customer success manager will call you within 24 hours' or 'Direct line to our support supervisor'\n"
			"- For website issues: mention specific improvements like 'Our tech team is implementing a new checkout system this week' or 'We'll send you a direct link to bypass the issue'\n"
			"- For pricing concerns: offer specific solutions like 'We'll match competitor pricing' or 'Here's a 15% discount code for your next purchase'\n"
			"- Include specific timelines, contact methods, and next steps\n"
			"- End with a specific follow-up commitment\n"
			"- Keep it under 120 words but be highly specific and actionable\n"
		)

	def _generate_groq(self, prompt: str) -> str | None:
		if not self._groq:
			return None
		try:
			resp = self._groq.chat.completions.create(
				model=self.groq_model,
				messages=[
					{"role": "system", "content": RESPONSE_SYSTEM_PROMPT},
					{"role": "user", "content": prompt},
				],
				temperature=0.3,
			)
			return resp.choices[0].message.content.strip()
		except Exception:
			return None

	def generate(self, inputs: List[ResponseInput]) -> List[str]:
		replies: List[str] = []
		for item in inputs:
			prompt = self._build_user_prompt(item)
			text = self._generate_groq(prompt)
			replies.append(text if text else _default_reply(item.feedback))
		return replies

	def add_to_dataframe(self, df: pd.DataFrame, text_col: str, sentiment_col: str, topic_col: str) -> pd.DataFrame:
		items = []
		for _, row in df.iterrows():
			# Handle NaN values by converting to empty string
			feedback = str(row[text_col]) if pd.notna(row[text_col]) else ""
			sentiment = str(row.get(sentiment_col, "")) if pd.notna(row.get(sentiment_col, "")) else ""
			topic = str(row.get(topic_col, "")) if pd.notna(row.get(topic_col, "")) else ""
			
			items.append(ResponseInput(feedback=feedback, sentiment=sentiment, topic=topic))
		
		replies = self.generate(items)
		df = df.copy()
		df["draft_response"] = replies
		return df
