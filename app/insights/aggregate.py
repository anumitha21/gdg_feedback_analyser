from __future__ import annotations
from collections import Counter
from typing import Dict, List
import os
import pandas as pd
from .business_insights import BusinessInsightsAnalyzer

# Groq only
try:
	from groq import Groq
	_groq_available = True
except Exception:
	_groq_available = False


class InsightAggregator:
	def __init__(self, model: str | None = None):
		self.groq_model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
		self.use_groq = _groq_available and bool(os.getenv("GROQ_API_KEY"))
		self._groq = Groq(api_key=os.getenv("GROQ_API_KEY")) if self.use_groq else None
		self.business_analyzer = BusinessInsightsAnalyzer()

	def _extract_problems_from_feedback(self, df: pd.DataFrame) -> List[str]:
		"""Extract and summarize actual problems mentioned in customer feedback."""
		problems = []
		negative_feedback = df[df.get('sentiment_label', '') == 'NEGATIVE']
		
		if len(negative_feedback) == 0:
			return ["No specific problems identified in the feedback."]
		
		# Common problem keywords to look for
		problem_keywords = {
			'delivery': ['late', 'delayed', 'slow', 'delivery', 'shipping', 'arrived'],
			'quality': ['broken', 'damaged', 'poor quality', 'defective', 'faulty', 'low quality'],
			'service': ['rude', 'unhelpful', 'poor service', 'bad service', 'customer service'],
			'website': ['confusing', 'difficult', 'hard to use', 'website', 'checkout', 'navigation'],
			'pricing': ['expensive', 'overpriced', 'too expensive', 'cost', 'price'],
			'product': ['missing', 'wrong', 'incorrect', 'not as described', 'product'],
			'support': ['no response', 'slow response', 'support', 'help', 'assistance']
		}
		
		# Analyze each negative feedback
		for _, row in negative_feedback.iterrows():
			feedback_text = str(row.get('feedback_text', '')).lower()
			
			# Check for specific problem categories
			for category, keywords in problem_keywords.items():
				if any(keyword in feedback_text for keyword in keywords):
					# Extract the specific issue
					for keyword in keywords:
						if keyword in feedback_text:
							# Find the sentence containing the keyword
							sentences = feedback_text.split('.')
							for sentence in sentences:
								if keyword in sentence:
									problem = sentence.strip()
									if len(problem) > 10:  # Avoid very short fragments
										problems.append(f"{category.title()} issue: {problem}")
									break
							break
		
		# Remove duplicates and limit to top issues
		unique_problems = list(dict.fromkeys(problems))  # Remove duplicates while preserving order
		return unique_problems[:5]  # Return top 5 problems

	def _analyze_positive_feedback(self, df: pd.DataFrame) -> List[str]:
		"""Analyze positive feedback for business insights."""
		insights = []
		positive_feedback = df[df.get('sentiment_label', '') == 'POSITIVE']
		
		if len(positive_feedback) == 0:
			return insights
		
		# Count positive aspects
		positive_aspects = {
			'service': 0, 'quality': 0, 'delivery': 0, 'website': 0, 
			'pricing': 0, 'product': 0, 'support': 0
		}
		
		for _, row in positive_feedback.iterrows():
			feedback_text = str(row.get('feedback_text', '')).lower()
			
			if any(word in feedback_text for word in ['great service', 'excellent service', 'helpful', 'friendly', 'support']):
				positive_aspects['service'] += 1
			if any(word in feedback_text for word in ['quality', 'good quality', 'excellent quality', 'perfect']):
				positive_aspects['quality'] += 1
			if any(word in feedback_text for word in ['fast', 'quick', 'delivery', 'shipping', 'on time']):
				positive_aspects['delivery'] += 1
			if any(word in feedback_text for word in ['easy', 'simple', 'website', 'user-friendly']):
				positive_aspects['website'] += 1
			if any(word in feedback_text for word in ['value', 'worth', 'affordable', 'price']):
				positive_aspects['pricing'] += 1
		
		# Generate positive insights
		total_positive = len(positive_feedback)
		top_strength = max(positive_aspects.items(), key=lambda x: x[1])
		if top_strength[1] > 0:
			percentage = round((top_strength[1] / total_positive) * 100, 1)
			insights.append(f"Key strength: {top_strength[0].title()} excellence drives {percentage}% of positive feedback - scale these practices")
		
		# Multiple strengths insight
		strength_areas = sum(1 for count in positive_aspects.values() if count > 0)
		if strength_areas > 2:
			insights.append(f"Multiple strengths identified ({strength_areas} areas) - leverage these competitive advantages in marketing and operations")
		
		return insights

	def _generate_insights_from_feedback(self, df: pd.DataFrame) -> List[str]:
		"""Generate comprehensive business insights from customer feedback patterns."""
		insights = []
		
		# Analyze sentiment patterns
		negative_feedback = df[df.get('sentiment_label', '') == 'NEGATIVE']
		positive_feedback = df[df.get('sentiment_label', '') == 'POSITIVE']
		neutral_feedback = df[df.get('sentiment_label', '') == 'NEUTRAL']
		
		# Count problem categories with detailed analysis
		problem_categories = {
			'delivery': {'count': 0, 'issues': []}, 
			'quality': {'count': 0, 'issues': []}, 
			'service': {'count': 0, 'issues': []}, 
			'website': {'count': 0, 'issues': []}, 
			'pricing': {'count': 0, 'issues': []}, 
			'product': {'count': 0, 'issues': []}, 
			'support': {'count': 0, 'issues': []}
		}
		
		# Analyze negative feedback for detailed patterns
		for _, row in negative_feedback.iterrows():
			feedback_text = str(row.get('feedback_text', '')).lower()
			
			if any(word in feedback_text for word in ['late', 'delayed', 'delivery', 'shipping', 'arrived']):
				problem_categories['delivery']['count'] += 1
				problem_categories['delivery']['issues'].append(feedback_text[:50])
			if any(word in feedback_text for word in ['broken', 'damaged', 'poor quality', 'defective', 'faulty']):
				problem_categories['quality']['count'] += 1
				problem_categories['quality']['issues'].append(feedback_text[:50])
			if any(word in feedback_text for word in ['rude', 'unhelpful', 'poor service', 'bad service']):
				problem_categories['service']['count'] += 1
				problem_categories['service']['issues'].append(feedback_text[:50])
			if any(word in feedback_text for word in ['confusing', 'website', 'checkout', 'navigation', 'difficult']):
				problem_categories['website']['count'] += 1
				problem_categories['website']['issues'].append(feedback_text[:50])
			if any(word in feedback_text for word in ['expensive', 'overpriced', 'cost', 'price']):
				problem_categories['pricing']['count'] += 1
				problem_categories['pricing']['issues'].append(feedback_text[:50])
		
		# Generate detailed insights based on patterns
		total_negative = len(negative_feedback)
		total_feedback = len(df)
		
		if total_negative > 0:
			# Find most critical problem category
			top_problem = max(problem_categories.items(), key=lambda x: x[1]['count'])
			if top_problem[1]['count'] > 0:
				percentage = round((top_problem[1]['count'] / total_negative) * 100, 1)
				insights.append(f"Critical Issue: {top_problem[0].title()} problems affect {percentage}% of dissatisfied customers - immediate action required")
			
			# Multiple problem areas insight
			problem_areas = sum(1 for cat in problem_categories.values() if cat['count'] > 0)
			if problem_areas > 2:
				insights.append(f"Systemic Issues: {problem_areas} problem categories identified - indicates need for comprehensive customer experience overhaul")
			
			# Specific problem insights with business impact
			for category, data in problem_categories.items():
				if data['count'] > 0:
					percentage = round((data['count'] / total_feedback) * 100, 1)
					if category == 'delivery':
						insights.append(f"Delivery Impact: {data['count']} delivery complaints ({percentage}% of all feedback) - implement tracking, notifications, and carrier partnerships")
					elif category == 'quality':
						insights.append(f"Quality Risk: {data['count']} quality issues ({percentage}% of feedback) - conduct supplier audits and implement quality control checkpoints")
					elif category == 'service':
						insights.append(f"Service Gap: {data['count']} service complaints ({percentage}% of feedback) - invest in staff training and response time improvements")
					elif category == 'website':
						insights.append(f"UX Problems: {data['count']} website issues ({percentage}% of feedback) - prioritize UX testing and checkout optimization")
					elif category == 'pricing':
						insights.append(f"Pricing Concerns: {data['count']} pricing complaints ({percentage}% of feedback) - review pricing strategy and value communication")
		
		# Add positive feedback insights
		positive_insights = self._analyze_positive_feedback(df)
		insights.extend(positive_insights)
		
		# Neutral feedback opportunity
		if len(neutral_feedback) > 0:
			neutral_pct = round((len(neutral_feedback) / total_feedback) * 100, 1)
			insights.append(f"Growth Opportunity: {neutral_pct}% neutral feedback - implement strategies to convert neutral customers to advocates")
		
		return insights[:6]  # Return top 6 insights

	def _generate_business_impact_analysis(self, df: pd.DataFrame) -> List[str]:
		"""Generate detailed business impact analysis for executives."""
		analysis = []
		total_feedback = len(df)
		negative_feedback = df[df.get('sentiment_label', '') == 'NEGATIVE']
		positive_feedback = df[df.get('sentiment_label', '') == 'POSITIVE']
		
		# Customer satisfaction metrics
		csat_score = round((len(positive_feedback) / total_feedback) * 100, 1) if total_feedback > 0 else 0
		negative_rate = round((len(negative_feedback) / total_feedback) * 100, 1) if total_feedback > 0 else 0
		
		analysis.append(f"Customer Satisfaction Score: {csat_score}% (Industry benchmark: 80%+)")
		analysis.append(f"Customer Dissatisfaction Rate: {negative_rate}% (Critical threshold: >20%)")
		
		# Business impact calculations
		if negative_rate > 20:
			analysis.append(f"Business Risk: High dissatisfaction rate ({negative_rate}%) may lead to customer churn and negative word-of-mouth")
		elif negative_rate > 10:
			analysis.append(f"Business Risk: Moderate dissatisfaction rate ({negative_rate}%) requires attention to prevent escalation")
		
		# Revenue impact estimation
		if total_feedback > 50:  # Only for larger datasets
			estimated_churn_risk = round(negative_rate * 0.3, 1)  # 30% of negative customers likely to churn
			analysis.append(f"Revenue Risk: Estimated {estimated_churn_risk}% of customers at risk of churning due to negative experiences")
		
		return analysis

	def _generate_strategic_recommendations(self, df: pd.DataFrame) -> List[str]:
		"""Generate strategic business recommendations with specific actions."""
		recommendations = []
		negative_feedback = df[df.get('sentiment_label', '') == 'NEGATIVE']
		positive_feedback = df[df.get('sentiment_label', '') == 'POSITIVE']
		
		# Immediate actions (0-30 days)
		immediate_actions = []
		if len(negative_feedback) > 0:
			immediate_actions.append("IMMEDIATE (0-30 days): Implement 24-hour response protocol for all negative feedback")
			immediate_actions.append("IMMEDIATE (0-30 days): Create escalation matrix for critical issues requiring management intervention")
		
		# Short-term actions (1-3 months)
		short_term_actions = []
		if len(negative_feedback) > 0:
			short_term_actions.append("SHORT-TERM (1-3 months): Deploy customer feedback analytics dashboard for real-time monitoring")
			short_term_actions.append("SHORT-TERM (1-3 months): Establish cross-functional customer experience improvement team")
		
		# Long-term strategic initiatives (3-12 months)
		long_term_actions = []
		if len(negative_feedback) > 0:
			long_term_actions.append("LONG-TERM (3-12 months): Implement predictive customer satisfaction modeling")
			long_term_actions.append("LONG-TERM (3-12 months): Develop customer success program to proactively address issues")
		
		# ROI-focused recommendations
		roi_recommendations = []
		if len(positive_feedback) > 0:
			roi_recommendations.append("ROI OPPORTUNITY: Leverage positive feedback for marketing campaigns and case studies")
		if len(negative_feedback) > 0:
			roi_recommendations.append("ROI IMPACT: Each resolved complaint can prevent 3-5 negative reviews and potential customer loss")
		
		recommendations.extend(immediate_actions[:2])
		recommendations.extend(short_term_actions[:2])
		recommendations.extend(long_term_actions[:2])
		recommendations.extend(roi_recommendations[:2])
		
		return recommendations

	def _recommendations_fallback(self, summary_lines: List[str], sentiment_breakdown: Dict, topic_counts: Dict, df: pd.DataFrame) -> List[str]:
		"""Generate comprehensive business insights and strategic recommendations."""
		recommendations = []
		
		# Generate detailed business impact analysis
		business_impact = self._generate_business_impact_analysis(df)
		recommendations.extend(business_impact)
		
		# Generate insights from feedback patterns
		insights = self._generate_insights_from_feedback(df)
		recommendations.extend(insights[:3])  # Top 3 insights
		
		# Generate strategic recommendations
		strategic_recs = self._generate_strategic_recommendations(df)
		recommendations.extend(strategic_recs[:4])  # Top 4 strategic recommendations
		
		# Fallback if no recommendations generated
		if not recommendations:
			recommendations = [
				"Establish customer feedback governance committee with monthly review cycles",
				"Implement customer journey analytics to identify critical touchpoints",
				"Develop customer success metrics dashboard for executive reporting",
				"Create customer feedback action plan with quarterly business reviews"
			]
		
		return recommendations[:8]  # Return top 8 comprehensive recommendations

	def _recommendations_llm(self, summary_lines: List[str], df: pd.DataFrame) -> List[str]:
		# Extract sample negative feedback for context
		negative_feedback = df[df.get('sentiment_label', '') == 'NEGATIVE']
		sample_problems = []
		if len(negative_feedback) > 0:
			for _, row in negative_feedback.head(3).iterrows():
				feedback = str(row.get('feedback_text', ''))[:100]  # First 100 chars
				sample_problems.append(feedback)
		
		prompt = (
			"Analyze these customer feedback patterns and provide 3 strategic business insights and recommendations. "
			"Focus on business implications and actionable strategies, not just repeating the feedback.\n\n"
			"Findings:\n- " + "\n- ".join(summary_lines) + 
			"\n\nSample feedback context:\n- " + "\n- ".join(sample_problems) +
			"\n\nProvide strategic insights and recommendations (each under 30 words) that focus on:\n"
			"1. Business process improvements\n2. Operational changes needed\n3. Strategic initiatives to address root causes"
		)
		text = None
		if self._groq is not None:
			try:
				resp = self._groq.chat.completions.create(
					model=self.groq_model,
					messages=[
						{"role": "system", "content": "You are a concise business analyst."},
						{"role": "user", "content": prompt},
					],
					temperature=0.2,
				)
				text = resp.choices[0].message.content.strip()
			except Exception:
				text = None
		if not text:
			return self._recommendations_fallback(summary_lines)
		lines = [l.strip("- ") for l in text.splitlines() if l.strip()]
		return lines[:3] if lines else self._recommendations_fallback(summary_lines)

	def aggregate(self, df: pd.DataFrame) -> Dict:
		total = max(len(df), 1)
		# Sentiment breakdown
		sentiments = Counter(df.get("sentiment_label", []))
		sentiment_breakdown = {k: round(v / total * 100, 1) for k, v in sentiments.items()}
		# Topic counts
		topic_counts = Counter(df.get("topic", []))
		# Top findings summary lines
		top_sent = ", ".join([f"{k}: {v}%" for k, v in sorted(sentiment_breakdown.items(), key=lambda x: -x[1])])
		common_topics = ", ".join([f"Topic {t} ({c} mentions)" for t, c in topic_counts.most_common(3)])
		
		# Enhanced business-focused findings
		summary_lines = [
			f"Customer sentiment analysis: {top_sent}",
			f"Total feedback entries analyzed: {total}",
		]
		
		# Detailed topic analysis
		if topic_counts:
			sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
			summary_lines.append(f"Most discussed issues (by frequency):")
			for i, (topic, count) in enumerate(sorted_topics[:5], 1):
				percentage = round((count / total) * 100, 1)
				summary_lines.append(f"  {i}. Topic {topic}: {count} mentions ({percentage}% of all feedback)")
		
		# Sentiment insights with business implications
		if sentiment_breakdown:
			neg_pct = sentiment_breakdown.get('NEGATIVE', 0)
			pos_pct = sentiment_breakdown.get('POSITIVE', 0)
			neutral_pct = sentiment_breakdown.get('NEUTRAL', 0)
			
			if neg_pct > 50:
				summary_lines.append(f"Critical Alert: High negative sentiment ({neg_pct}%) indicates urgent customer experience issues requiring immediate attention")
			elif neg_pct > pos_pct:
				summary_lines.append(f"Primary concern: Negative feedback dominates ({neg_pct}% vs {pos_pct}% positive) - customer satisfaction below acceptable levels")
			elif pos_pct > 60:
				summary_lines.append(f"Positive trend: Strong customer satisfaction ({pos_pct}% positive) - maintain and scale successful practices")
			
			if neutral_pct > 30:
				summary_lines.append(f"Opportunity: High neutral sentiment ({neutral_pct}%) suggests room for improvement to convert neutral customers to advocates")

		# Enhanced business insights analysis
		problem_analysis = self.business_analyzer.analyze_problems(df)
		business_impact = problem_analysis.get('business_impact', {})
		executive_summary = self.business_analyzer.generate_executive_summary(df, problem_analysis.get('problem_categories', {}), business_impact)
		
		# Try LLM recommendations first, fallback to data-driven recommendations
		try:
			recommendations = self._recommendations_llm(summary_lines, df)
		except Exception:
			recommendations = self._recommendations_fallback(summary_lines, sentiment_breakdown, dict(topic_counts), df)

		# Add business-specific recommendations
		business_recommendations = problem_analysis.get('recommendations', [])
		enhanced_recommendations = recommendations + business_recommendations[:3]  # Add top 3 business recommendations
		
		return {
			"sentiment_breakdown": sentiment_breakdown,
			"topic_counts": dict(topic_counts),
			"top_findings": summary_lines,
			"recommendations": enhanced_recommendations,
			"business_insights": {
				"problem_analysis": problem_analysis,
				"executive_summary": executive_summary,
				"business_impact": business_impact
			}
		}
