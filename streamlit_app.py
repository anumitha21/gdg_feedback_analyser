import os
import io
import pandas as pd
import streamlit as st
import plotly.express as px
from pipeline import FeedbackPipeline, PipelineConfig

st.set_page_config(page_title="AI Feedback Analyzer", layout="wide")
st.title("AI-Driven Customer Feedback Analyzer & Response Generator")

# Global Plotly styling for a formal look
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = [
	"#2E77B8", "#E15759", "#76B7B2", "#F28E2B", "#59A14F", "#EDC948",
	"#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"
]

# Subtle, formal CSS styling
st.markdown(
	"""
	<style>
		/***** Typography *****/
		section.main > div {max-width: 1300px;}
		h1, h2, h3 {font-family: Inter, Segoe UI, system-ui, -apple-system, sans-serif;}
		/***** Metrics *****/
		[data-testid="stMetric"] div {text-align: left;}
		/***** Tables *****/
		[data-testid="stTable"] td, [data-testid="stTable"] th {font-size: 14px;}
		/***** Expander *****/
		[data-testid="stExpander"] details {border: 1px solid #E5E7EB; border-radius: 8px;}
		/***** Buttons *****/
		.stButton>button {border-radius: 6px; border: 1px solid #D1D5DB;}
	</style>
	""",
	unsafe_allow_html=True,
)
with st.sidebar:
	st.header("Data Source")
	source = st.selectbox("Choose input method", ["Upload file", "Local file path", "Use sample data"])
	upload = None
	local_path = None
	if source == "Upload file":
		upload = st.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"]) 
	elif source == "Local file path":
		local_path = st.text_input("Enter full path to CSV/Excel", value="")
	# Emotion detection is always enabled
	use_emotions = True
	analyze_btn = st.button("Analyze")

def _read_df(file) -> pd.DataFrame:
	"""Read CSV/Excel file without caching to ensure fresh data."""
	if isinstance(file, str):
		lower = file.lower()
		if lower.endswith(".csv"):
			return pd.read_csv(file)
		return pd.read_excel(file)
	if hasattr(file, "name") and file.name.lower().endswith(".csv"):
		return pd.read_csv(file)
	return pd.read_excel(file)

def _detect_feedback_column(df: pd.DataFrame) -> str:
	"""Simple detection - just find the first text column."""
	# Look for any text column
	for col in df.columns:
		if df[col].dtype == 'object':  # String columns
			return col
	
	# Fallback to first column
	return df.columns[0]

def _clean_dataframe(df: pd.DataFrame, feedback_col: str) -> pd.DataFrame:
	"""Keep all columns but ensure feedback column exists."""
	# Keep all original columns
	clean_df = df.copy()
	
	# If feedback column is not 'feedback_text', add it
	if feedback_col != 'feedback_text':
		clean_df['feedback_text'] = clean_df[feedback_col]
	
	# Remove rows with empty feedback
	clean_df = clean_df.dropna(subset=['feedback_text'])
	clean_df = clean_df[clean_df['feedback_text'].astype(str).str.strip() != '']
	
	return clean_df

def _get_pipeline(use_emotions: bool) -> FeedbackPipeline:
	"""Create a fresh pipeline instance for each analysis."""
	cfg = PipelineConfig(use_emotions=use_emotions)
	return FeedbackPipeline(cfg)

col1, col2 = st.columns([1, 1])

def _display_insights(df: pd.DataFrame):
	insights = df.attrs.get("insights") if hasattr(df, "attrs") else None
	if not insights:
		return

	# Enhanced Business Insights - Formal, tabbed UI
	st.header("🎯 Business Insights")
	
	# Executive Summary Section
	if "business_insights" in insights:
		business_insights = insights["business_insights"]
		executive_summary = business_insights.get("executive_summary", {})
		problem_analysis = business_insights.get("problem_analysis", {})
		
		# Tabs for a formal, organized layout
		_tab_exec, _tab_analysis, _tab_dashboard, _tab_recs, _tab_details = st.tabs([
			"Executive Summary",
			"Problem Analysis",
			"Dashboard",
			"Recommendations",
			"Detailed Analysis",
		])

		# Executive Summary
		with _tab_exec:
			st.subheader("📊 Executive Summary")
			exec_metrics = executive_summary.get("executive_metrics", {})
			col1, col2, col3, col4 = st.columns(4)
			with col1:
				st.metric("Customer Satisfaction", exec_metrics.get("customer_satisfaction_score", "N/A"))
			with col2:
				st.metric("Dissatisfaction Rate", exec_metrics.get("customer_dissatisfaction_rate", "N/A"))
			with col3:
				st.metric("Business Risk Level", exec_metrics.get("business_risk_level", "N/A"))
			with col4:
				st.metric("Churn Risk", exec_metrics.get("estimated_churn_risk", "N/A"))
			st.markdown("### 🔍 Key Findings")
			for finding in executive_summary.get("key_findings", []):
				st.markdown("• " + str(finding))
			if problem_analysis.get("problem_categories"):
				short_issues = []
				for category, data in sorted(
					problem_analysis["problem_categories"].items(),
					key=lambda x: (x[1]["severity"], x[1]["percentage"]),
					reverse=True,
				):
					short_issues.append(f"{category.title()} ({data['severity']}, {data['percentage']}%)")
				if short_issues:
					st.markdown("**Issues (short):** " + ", ".join(short_issues))

		# Problem Analysis
		with _tab_analysis:
			if problem_analysis.get("problem_categories"):
				st.subheader("⚠️ Problem Analysis")
				problem_categories = problem_analysis["problem_categories"]
				problem_data = []
				for category, data in problem_categories.items():
					problem_data.append({
						"Problem Category": category.title(),
						"Count": data["count"],
						"Percentage": f"{data['percentage']}%",
						"Severity": data["severity"],
						"Business Impact": data["business_impact"]
					})
				problem_df = pd.DataFrame(problem_data)
				st.dataframe(problem_df, use_container_width=True)

		# Dashboard
		with _tab_dashboard:
			if problem_analysis.get("problem_categories"):
				problem_categories = problem_analysis["problem_categories"]
				from app.insights.business_insights import BusinessInsightsAnalyzer
				analyzer = BusinessInsightsAnalyzer()
				visualizations = analyzer.create_visualizations(df, problem_categories)
				for _, fig in visualizations.items():
					st.plotly_chart(fig, use_container_width=True)

		# Recommendations
		with _tab_recs:
			st.subheader("💡 Strategic Recommendations")
			immediate_actions = executive_summary.get("immediate_actions", [])
			strategic_initiatives = executive_summary.get("strategic_initiatives", [])
			colA, colB = st.columns(2)
			with colA:
				st.markdown("#### Immediate (0–30 days)")
				for action in immediate_actions:
					st.markdown("- " + str(action))
			with colB:
				st.markdown("#### Strategic (3–12 months)")
				for initiative in strategic_initiatives:
					st.markdown("- " + str(initiative))

		# Detailed Analysis (original)
		with _tab_details:
			left, right = st.columns([1, 1])
			with left:
				st.write("Top Findings")
				for item in insights.get("top_findings", []):
					st.markdown("- " + str(item))
			with right:
				st.write("Recommendations")
				for rec in insights.get("recommendations", []):
					st.markdown("- " + str(rec))
			if "sentiment_breakdown" in insights:
				br = insights["sentiment_breakdown"]
				pie = px.pie(values=list(br.values()), names=list(br.keys()), title="Sentiment Breakdown")
				st.plotly_chart(pie, use_container_width=True)
			if "topic_counts" in insights:
				tc = insights["topic_counts"]
				bar = px.bar(x=[str(k) for k in tc.keys()], y=list(tc.values()), title="Topic Counts", labels={"x": "Topic", "y": "Count"})
				st.plotly_chart(bar, use_container_width=True)

# Resolve input dataframe based on selection
input_df: pd.DataFrame | None = None
input_error: str | None = None
feedback_column: str | None = None

if analyze_btn:
	try:
		if source == "Upload file":
			if upload is None:
				input_error = "Please select a file to upload."
			else:
				with st.spinner("Loading file..."):
					raw_df = _read_df(upload)
					feedback_column = _detect_feedback_column(raw_df)
					input_df = _clean_dataframe(raw_df, feedback_column)
		elif source == "Local file path":
			if not local_path:
				input_error = "Please enter a valid file path."
			elif not os.path.exists(local_path):
				input_error = "Path does not exist."
			else:
				with st.spinner("Reading local file..."):
					raw_df = _read_df(local_path)
					feedback_column = _detect_feedback_column(raw_df)
					input_df = _clean_dataframe(raw_df, feedback_column)
		else:  # Use sample data
			with st.spinner("Loading sample data..."):
				raw_df = pd.read_csv("data/sample_feedback.csv")
				feedback_column = _detect_feedback_column(raw_df)
				input_df = _clean_dataframe(raw_df, feedback_column)
	except Exception as e:
		input_error = f"Failed to load data: {e}"

if analyze_btn:
	if input_error:
		st.error(input_error)
		st.stop()
	if input_df is None:
		st.error("No data loaded.")
		st.stop()
	if len(input_df) == 0:
		st.error("No valid feedback data found in the file.")
		st.stop()
	
	# Show detected column and data preview
	st.subheader("Data Processing")
	st.info(f"Detected feedback column: **{feedback_column}** → **feedback_text**")
	st.write(f"Found {len(input_df)} valid feedback entries")
	st.write(f"Total columns: {len(input_df.columns)}")
	
	# Input data preview (on-demand)
	st.subheader("Input Data")
	if st.button("View input data"):
		st.dataframe(input_df.head(), use_container_width=True)
	
	# Clear any cached results to ensure fresh analysis
	st.cache_data.clear()
	
	with st.spinner("Analyzing feedback..."):
		# Create fresh pipeline instance (no caching)
		cfg = PipelineConfig(use_emotions=use_emotions)
		pipeline = FeedbackPipeline(cfg)
		
		# Show progress steps
		progress_bar = st.progress(0)
		status_text = st.empty()
		
		status_text.text("Preprocessing text...")
		progress_bar.progress(20)
		
		status_text.text("Analyzing sentiment...")
		progress_bar.progress(40)
		
		status_text.text("Finding topics...")
		progress_bar.progress(60)
		
		status_text.text("Generating responses...")
		progress_bar.progress(80)
		
		status_text.text("Aggregating insights...")
		progress_bar.progress(100)
		
		result_df = pipeline.run(input_df, text_col="feedback_text")
		
		status_text.text("Analysis complete!")
		progress_bar.empty()
		status_text.empty()
		
		_display_insights(result_df)
		
		# Enhanced Business Insights Section
		if "business_insights" in result_df.attrs.get("insights", {}):
			business_insights = result_df.attrs["insights"]["business_insights"]
			problem_analysis = business_insights.get("problem_analysis", {})
			
			# Detailed Problem Analysis
			if problem_analysis.get("problem_categories"):
				st.subheader("🔍 Detailed Problem Analysis")
				problem_categories = problem_analysis["problem_categories"]
				
				# Sort by severity and impact
				sorted_problems = sorted(
					problem_categories.items(), 
					key=lambda x: (x[1]['severity'], x[1]['percentage']), 
					reverse=True
				)
				
				for category, data in sorted_problems:
					with st.expander(f"🚨 {category.title()} Issues - {data['severity']} Priority", expanded=data['severity'] == 'CRITICAL'):
						col1, col2 = st.columns([1, 1])
						
						with col1:
							st.metric("Affected Customers", data['count'])
							st.metric("Impact Percentage", f"{data['percentage']}%")
							st.metric("Business Impact", data['business_impact'])
						
						with col2:
							st.metric("Severity Level", data['severity'])
							st.metric("Timeline", data.get('timeline', '2-4 weeks'))
							st.metric("ROI Estimate", data.get('roi_estimate', '25% ROI'))
						
						# Sample feedback
						if data.get('sample_feedback'):
							st.write("**Sample Customer Feedback:**")
							for feedback in data['sample_feedback']:
								st.text("• " + str(feedback.get('feedback', '')))
						
						# Solutions
						st.write("**Recommended Solutions:**")
						for solution in data.get('solutions', []):
							st.markdown("• " + str(solution))
			
		st.subheader("📊 Analysis Results")
		st.dataframe(result_df, use_container_width=True)
		
		# Show sample results
		if len(result_df) > 0:
			st.subheader("Sample Analysis")
			sample = result_df.iloc[0]
			col1, col2, col3 = st.columns(3)
			with col1:
				st.metric("Sentiment", sample.get("sentiment_label", "N/A"))
			with col2:
				st.metric("Topic", sample.get("topic", "N/A"))
			with col3:
				st.metric("Confidence", f"{sample.get('sentiment_score', 0):.2f}")
		
		# Download
		csv_bytes = result_df.to_csv(index=False).encode("utf-8")
		st.download_button("Download CSV", data=csv_bytes, file_name="feedback_analysis.csv", mime="text/csv")
else:
	st.info("Choose an input method, then click Analyze.")
