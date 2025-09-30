"""
Example demonstrating the enhanced business insights functionality.
This script shows how the new business insights module provides:
1. Specific problem identification with business impact
2. Detailed solutions for each problem category
3. Visual data analysis with charts and graphs
4. Executive summary with key metrics
"""

import pandas as pd
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.insights.business_insights import BusinessInsightsAnalyzer

def create_sample_feedback_data():
    """Create comprehensive sample feedback data for demonstration."""
    sample_data = [
        # Delivery issues (Critical)
        {"feedback_text": "My package was 3 days late and arrived damaged. This is unacceptable!", "sentiment_label": "NEGATIVE", "sentiment_score": -0.9},
        {"feedback_text": "Delivery was extremely slow, took 2 weeks instead of 3 days", "sentiment_label": "NEGATIVE", "sentiment_score": -0.8},
        {"feedback_text": "Package never arrived, tracking shows delivered but I never received it", "sentiment_label": "NEGATIVE", "sentiment_score": -0.95},
        
        # Quality issues (High)
        {"feedback_text": "Product arrived broken and poor quality, not as described", "sentiment_label": "NEGATIVE", "sentiment_score": -0.7},
        {"feedback_text": "Defective product, low quality materials used", "sentiment_label": "NEGATIVE", "sentiment_score": -0.6},
        
        # Service issues (Medium)
        {"feedback_text": "Customer service was rude and unhelpful when I called", "sentiment_label": "NEGATIVE", "sentiment_score": -0.5},
        {"feedback_text": "Poor service experience, staff was unprofessional", "sentiment_label": "NEGATIVE", "sentiment_score": -0.4},
        
        # Website issues (Medium)
        {"feedback_text": "Website is confusing and difficult to navigate", "sentiment_label": "NEGATIVE", "sentiment_score": -0.3},
        {"feedback_text": "Checkout process is buggy and slow", "sentiment_label": "NEGATIVE", "sentiment_score": -0.4},
        
        # Pricing issues (Low)
        {"feedback_text": "Too expensive for what you get, overpriced", "sentiment_label": "NEGATIVE", "sentiment_score": -0.2},
        
        # Positive feedback
        {"feedback_text": "Excellent product quality and fast delivery!", "sentiment_label": "POSITIVE", "sentiment_score": 0.9},
        {"feedback_text": "Great customer service, very helpful staff", "sentiment_label": "POSITIVE", "sentiment_score": 0.8},
        {"feedback_text": "Love the product, will definitely buy again", "sentiment_label": "POSITIVE", "sentiment_score": 0.7},
        
        # Neutral feedback
        {"feedback_text": "Product is okay, nothing special", "sentiment_label": "NEUTRAL", "sentiment_score": 0.1},
        {"feedback_text": "Average experience, could be better", "sentiment_label": "NEUTRAL", "sentiment_score": 0.0},
    ]
    
    return pd.DataFrame(sample_data)

def demonstrate_business_insights():
    """Demonstrate the enhanced business insights functionality."""
    print("🎯 Enhanced Business Insights Demonstration")
    print("=" * 50)
    
    # Create sample data
    df = create_sample_feedback_data()
    print(f"📊 Analyzing {len(df)} customer feedback entries...")
    
    # Initialize business insights analyzer
    analyzer = BusinessInsightsAnalyzer()
    
    # Analyze problems
    print("\n🔍 Problem Analysis:")
    print("-" * 30)
    problem_analysis = analyzer.analyze_problems(df)
    
    print(f"Problem Summary: {problem_analysis['problem_summary']}")
    print(f"Business Impact: {problem_analysis['business_impact']}")
    
    # Display problem categories
    if problem_analysis['problem_categories']:
        print("\n📋 Problem Categories:")
        for category, data in problem_analysis['problem_categories'].items():
            print(f"\n🚨 {category.upper()} ISSUES:")
            print(f"   • Count: {data['count']} customers affected")
            print(f"   • Percentage: {data['percentage']}% of negative feedback")
            print(f"   • Severity: {data['severity']}")
            print(f"   • Business Impact: {data['business_impact']}")
            print(f"   • Timeline: {data.get('timeline', '2-4 weeks')}")
            print(f"   • ROI Estimate: {data.get('roi_estimate', '25% ROI')}")
            
            print(f"   📝 Sample Feedback:")
            for feedback in data.get('sample_feedback', [])[:2]:
                print(f"      • {feedback['feedback']}")
            
            print(f"   💡 Recommended Solutions:")
            for solution in data.get('solutions', [])[:3]:
                print(f"      • {solution}")
    
    # Generate executive summary
    print("\n📊 Executive Summary:")
    print("-" * 30)
    executive_summary = analyzer.generate_executive_summary(
        df, 
        problem_analysis.get('problem_categories', {}), 
        problem_analysis.get('business_impact', {})
    )
    
    metrics = executive_summary['executive_metrics']
    print(f"Customer Satisfaction Score: {metrics['customer_satisfaction_score']}")
    print(f"Customer Dissatisfaction Rate: {metrics['customer_dissatisfaction_rate']}")
    print(f"Business Risk Level: {metrics['business_risk_level']}")
    print(f"Estimated Churn Risk: {metrics['estimated_churn_risk']}")
    print(f"Revenue Impact: {metrics['revenue_impact']}")
    print(f"Brand Reputation Risk: {metrics['brand_reputation_risk']}")
    
    print("\n🔍 Key Findings:")
    for finding in executive_summary['key_findings']:
        print(f"• {finding}")
    
    print("\n🚨 Immediate Actions (0-30 days):")
    for action in executive_summary['immediate_actions']:
        print(f"• {action}")
    
    print("\n🎯 Strategic Initiatives (3-12 months):")
    for initiative in executive_summary['strategic_initiatives']:
        print(f"• {initiative}")
    
    # Create visualizations
    print("\n📈 Creating Visualizations...")
    visualizations = analyzer.create_visualizations(df, problem_analysis.get('problem_categories', {}))
    print(f"Generated {len(visualizations)} visualization charts:")
    for viz_name in visualizations.keys():
        print(f"• {viz_name}")
    
    print("\n✅ Business Insights Analysis Complete!")
    print("\nThis enhanced system provides:")
    print("• Specific problem identification with business impact metrics")
    print("• Detailed solutions for each problem category")
    print("• Visual data analysis with charts and graphs")
    print("• Executive summary with key metrics and recommendations")
    print("• Timeline and ROI estimates for each problem category")
    print("• Comprehensive business risk assessment")

if __name__ == "__main__":
    demonstrate_business_insights()
