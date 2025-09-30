"""
Enhanced Business Insights Module
Provides comprehensive business analysis with specific problem identification, 
solutions, and visual data analysis for customer feedback.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import Counter, defaultdict
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class BusinessInsightsAnalyzer:
    """Comprehensive business insights analyzer for customer feedback."""
    
    def __init__(self):
        self.problem_categories = {
            'delivery': {
                'keywords': ['late', 'delayed', 'slow', 'delivery', 'shipping', 'arrived', 'tracking', 'package'],
                'business_impact': 'Customer retention and satisfaction',
                'solutions': [
                    'Implement real-time delivery tracking system',
                    'Establish carrier partnerships with performance SLAs',
                    'Create delivery notification system with ETA updates',
                    'Develop delivery performance dashboard for monitoring'
                ]
            },
            'quality': {
                'keywords': ['broken', 'damaged', 'poor quality', 'defective', 'faulty', 'low quality', 'substandard'],
                'business_impact': 'Brand reputation and customer trust',
                'solutions': [
                    'Implement supplier quality audits and certifications',
                    'Establish quality control checkpoints in production',
                    'Create quality feedback loop with suppliers',
                    'Develop quality metrics dashboard for continuous monitoring'
                ]
            },
            'service': {
                'keywords': ['rude', 'unhelpful', 'poor service', 'bad service', 'customer service', 'unprofessional'],
                'business_impact': 'Customer experience and loyalty',
                'solutions': [
                    'Implement comprehensive staff training programs',
                    'Establish customer service performance metrics',
                    'Create escalation procedures for complex issues',
                    'Develop customer service quality assurance program'
                ]
            },
            'website': {
                'keywords': ['confusing', 'difficult', 'hard to use', 'website', 'checkout', 'navigation', 'buggy', 'slow'],
                'business_impact': 'Conversion rates and user experience',
                'solutions': [
                    'Conduct comprehensive UX/UI audit and redesign',
                    'Implement A/B testing for critical user flows',
                    'Optimize website performance and loading times',
                    'Create user testing program for continuous improvement'
                ]
            },
            'pricing': {
                'keywords': ['expensive', 'overpriced', 'too expensive', 'cost', 'price', 'value', 'worth'],
                'business_impact': 'Competitive positioning and sales conversion',
                'solutions': [
                    'Conduct competitive pricing analysis',
                    'Develop value-based pricing strategy',
                    'Create pricing transparency and communication plan',
                    'Implement dynamic pricing optimization'
                ]
            },
            'product': {
                'keywords': ['missing', 'wrong', 'incorrect', 'not as described', 'product', 'incomplete', 'defective'],
                'business_impact': 'Product-market fit and customer satisfaction',
                'solutions': [
                    'Enhance product descriptions and specifications',
                    'Implement product quality assurance processes',
                    'Create customer feedback integration in product development',
                    'Develop product performance monitoring system'
                ]
            },
            'support': {
                'keywords': ['no response', 'slow response', 'support', 'help', 'assistance', 'unresponsive'],
                'business_impact': 'Customer satisfaction and retention',
                'solutions': [
                    'Implement 24/7 customer support system',
                    'Establish response time SLAs and monitoring',
                    'Create multi-channel support strategy',
                    'Develop customer support analytics dashboard'
                ]
            }
        }
    
    def analyze_problems(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze customer problems with detailed categorization and business impact."""
        negative_feedback = df[df.get('sentiment_label', '') == 'NEGATIVE']
        
        if len(negative_feedback) == 0:
            return {
                'problem_summary': 'No negative feedback identified',
                'problem_categories': {},
                'business_impact': 'No immediate business risks identified',
                'recommendations': ['Continue monitoring customer feedback for early issue detection']
            }
        
        # Analyze each problem category
        problem_analysis = {}
        total_negative = len(negative_feedback)
        
        for category, config in self.problem_categories.items():
            category_feedback = []
            category_count = 0
            
            for _, row in negative_feedback.iterrows():
                feedback_text = str(row.get('feedback_text', '')).lower()
                
                # Check if any keywords match
                if any(keyword in feedback_text for keyword in config['keywords']):
                    category_count += 1
                    category_feedback.append({
                        'feedback': feedback_text[:100] + '...' if len(feedback_text) > 100 else feedback_text,
                        'sentiment_score': row.get('sentiment_score', 0)
                    })
            
            if category_count > 0:
                percentage = round((category_count / total_negative) * 100, 1)
                problem_analysis[category] = {
                    'count': category_count,
                    'percentage': percentage,
                    'business_impact': config['business_impact'],
                    'solutions': config['solutions'],
                    'sample_feedback': category_feedback[:3],  # Top 3 examples
                    'severity': self._calculate_severity(percentage, category_count)
                }
        
        return {
            'problem_summary': self._generate_problem_summary(problem_analysis, total_negative),
            'problem_categories': problem_analysis,
            'business_impact': self._calculate_business_impact(problem_analysis, total_negative),
            'recommendations': self._generate_specific_recommendations(problem_analysis)
        }
    
    def _calculate_severity(self, percentage: float, count: int) -> str:
        """Calculate problem severity based on percentage and count."""
        if percentage >= 30 or count >= 10:
            return 'CRITICAL'
        elif percentage >= 15 or count >= 5:
            return 'HIGH'
        elif percentage >= 5 or count >= 2:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _generate_problem_summary(self, problem_analysis: Dict, total_negative: int) -> str:
        """Generate comprehensive problem summary."""
        if not problem_analysis:
            return "No specific problems identified in negative feedback."
        
        total_problems = sum(cat['count'] for cat in problem_analysis.values())
        critical_issues = [cat for cat in problem_analysis.values() if cat['severity'] == 'CRITICAL']
        high_issues = [cat for cat in problem_analysis.values() if cat['severity'] == 'HIGH']
        
        summary_parts = []
        
        if critical_issues:
            top_critical = max(critical_issues, key=lambda x: x['percentage'])
            summary_parts.append(f"CRITICAL: {top_critical['percentage']}% of dissatisfied customers face {list(problem_analysis.keys())[list(problem_analysis.values()).index(top_critical)]} issues")
        
        if high_issues:
            summary_parts.append(f"HIGH PRIORITY: {len(high_issues)} problem categories require immediate attention")
        
        summary_parts.append(f"TOTAL IMPACT: {total_problems} problem instances across {len(problem_analysis)} categories affecting {total_negative} dissatisfied customers")
        
        return " | ".join(summary_parts)
    
    def _calculate_business_impact(self, problem_analysis: Dict, total_negative: int) -> Dict[str, Any]:
        """Calculate detailed business impact metrics."""
        if not problem_analysis:
            return {'risk_level': 'LOW', 'estimated_churn': 0, 'revenue_impact': 'Minimal'}
        
        # Calculate risk metrics
        critical_count = sum(1 for cat in problem_analysis.values() if cat['severity'] == 'CRITICAL')
        high_count = sum(1 for cat in problem_analysis.values() if cat['severity'] == 'HIGH')
        
        # Estimate churn risk
        total_problems = sum(cat['count'] for cat in problem_analysis.values())
        estimated_churn = min(round((total_problems / total_negative) * 30, 1), 50)  # Max 50% churn risk
        
        # Determine risk level
        if critical_count > 0:
            risk_level = 'CRITICAL'
        elif high_count >= 2:
            risk_level = 'HIGH'
        elif high_count >= 1:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        return {
            'risk_level': risk_level,
            'estimated_churn': estimated_churn,
            'revenue_impact': self._calculate_revenue_impact(risk_level, estimated_churn),
            'customer_satisfaction_impact': self._calculate_csat_impact(problem_analysis),
            'brand_reputation_risk': 'HIGH' if critical_count > 0 else 'MEDIUM' if high_count > 0 else 'LOW'
        }
    
    def _calculate_revenue_impact(self, risk_level: str, estimated_churn: float) -> str:
        """Calculate revenue impact based on risk level and churn."""
        if risk_level == 'CRITICAL':
            return f"CRITICAL: Estimated {estimated_churn}% revenue loss risk - immediate intervention required"
        elif risk_level == 'HIGH':
            return f"HIGH: Estimated {estimated_churn}% revenue loss risk - urgent action needed"
        elif risk_level == 'MEDIUM':
            return f"MEDIUM: Estimated {estimated_churn}% revenue loss risk - proactive measures recommended"
        else:
            return f"LOW: Estimated {estimated_churn}% revenue loss risk - monitor and maintain"
    
    def _calculate_csat_impact(self, problem_analysis: Dict) -> str:
        """Calculate customer satisfaction impact."""
        if not problem_analysis:
            return "No significant CSAT impact identified"
        
        total_affected = sum(cat['count'] for cat in problem_analysis.values())
        if total_affected >= 10:
            return "SIGNIFICANT: Multiple problem categories affecting customer satisfaction"
        elif total_affected >= 5:
            return "MODERATE: Several problem areas impacting customer experience"
        else:
            return "MINOR: Limited problem areas with manageable CSAT impact"
    
    def _generate_specific_recommendations(self, problem_analysis: Dict) -> List[Dict[str, Any]]:
        """Generate specific, actionable recommendations for each problem category."""
        recommendations = []
        
        # Sort by severity and impact
        sorted_problems = sorted(
            problem_analysis.items(), 
            key=lambda x: (x[1]['severity'], x[1]['percentage']), 
            reverse=True
        )
        
        for category, data in sorted_problems:
            rec = {
                'category': category.title(),
                'priority': data['severity'],
                'impact_percentage': data['percentage'],
                'affected_customers': data['count'],
                'business_impact': data['business_impact'],
                'immediate_actions': data['solutions'][:2],  # Top 2 immediate actions
                'strategic_initiatives': data['solutions'][2:],  # Strategic initiatives
                'timeline': self._get_timeline(data['severity']),
                'roi_estimate': self._calculate_roi_estimate(category, data['percentage'])
            }
            recommendations.append(rec)
        
        return recommendations
    
    def _get_timeline(self, severity: str) -> str:
        """Get recommended timeline based on severity."""
        timelines = {
            'CRITICAL': '0-7 days',
            'HIGH': '1-2 weeks',
            'MEDIUM': '2-4 weeks',
            'LOW': '1-2 months'
        }
        return timelines.get(severity, '2-4 weeks')
    
    def _calculate_roi_estimate(self, category: str, percentage: float) -> str:
        """Calculate ROI estimate for addressing the problem."""
        base_roi = {
            'delivery': 25, 'quality': 30, 'service': 35, 
            'website': 40, 'pricing': 20, 'product': 25, 'support': 30
        }
        
        roi_multiplier = min(percentage / 10, 3)  # Scale with problem severity
        estimated_roi = base_roi.get(category, 25) * roi_multiplier
        
        return f"Estimated {estimated_roi:.0f}% ROI within 6 months"
    
    def create_visualizations(self, df: pd.DataFrame, problem_analysis: Dict) -> Dict[str, Any]:
        """Create comprehensive visualizations for business insights."""
        visualizations = {}
        
        # 1. Problem Category Distribution
        if problem_analysis:
            categories = list(problem_analysis.keys())
            counts = [data['count'] for data in problem_analysis.values()]
            percentages = [data['percentage'] for data in problem_analysis.values()]
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Problem Count by Category', 'Problem Impact by Percentage'),
                specs=[[{"type": "bar"}, {"type": "pie"}]]
            )
            
            # Bar chart for counts
            fig.add_trace(
                go.Bar(x=categories, y=counts, name='Count', marker_color='red'),
                row=1, col=1
            )
            
            # Pie chart for percentages
            fig.add_trace(
                go.Pie(labels=categories, values=percentages, name='Percentage'),
                row=1, col=2
            )
            
            fig.update_layout(
                title="Customer Problem Analysis Dashboard",
                showlegend=True,
                height=500
            )
            
            visualizations['problem_dashboard'] = fig
        
        # 2. Sentiment vs Problem Correlation
        sentiment_data = df.groupby('sentiment_label').size().reset_index(name='count')
        fig_sentiment = px.pie(
            sentiment_data, 
            values='count', 
            names='sentiment_label',
            title="Overall Customer Sentiment Distribution",
            color_discrete_map={'NEGATIVE': 'red', 'POSITIVE': 'green', 'NEUTRAL': 'orange'}
        )
        visualizations['sentiment_distribution'] = fig_sentiment
        
        # 3. Problem Severity Matrix
        if problem_analysis:
            severity_data = []
            for category, data in problem_analysis.items():
                severity_data.append({
                    'Category': category.title(),
                    'Count': data['count'],
                    'Percentage': data['percentage'],
                    'Severity': data['severity']
                })
            
            severity_df = pd.DataFrame(severity_data)
            fig_severity = px.scatter(
                severity_df,
                x='Count',
                y='Percentage',
                color='Severity',
                size='Count',
                hover_data=['Category'],
                title="Problem Severity Matrix",
                color_discrete_map={
                    'CRITICAL': 'red',
                    'HIGH': 'orange', 
                    'MEDIUM': 'yellow',
                    'LOW': 'green'
                }
            )
            visualizations['severity_matrix'] = fig_severity
        
        # 4. Business Impact Timeline
        if problem_analysis:
            timeline_data = []
            for category, data in problem_analysis.items():
                timeline_data.append({
                    'Category': category.title(),
                    'Timeline': self._get_timeline(data['severity']),
                    'Priority': data['severity'],
                    'Impact': data['percentage']
                })
            
            timeline_df = pd.DataFrame(timeline_data)
            fig_timeline = px.bar(
                timeline_df,
                x='Category',
                y='Impact',
                color='Priority',
                title="Problem Resolution Timeline by Priority",
                color_discrete_map={
                    'CRITICAL': 'red',
                    'HIGH': 'orange',
                    'MEDIUM': 'yellow', 
                    'LOW': 'green'
                }
            )
            visualizations['resolution_timeline'] = fig_timeline
        
        return visualizations
    
    def generate_executive_summary(self, df: pd.DataFrame, problem_analysis: Dict, business_impact: Dict) -> Dict[str, Any]:
        """Generate comprehensive executive summary with key metrics and recommendations."""
        total_feedback = len(df)
        negative_feedback = df[df.get('sentiment_label', '') == 'NEGATIVE']
        positive_feedback = df[df.get('sentiment_label', '') == 'POSITIVE']
        
        csat_score = round((len(positive_feedback) / total_feedback) * 100, 1) if total_feedback > 0 else 0
        dissatisfaction_rate = round((len(negative_feedback) / total_feedback) * 100, 1) if total_feedback > 0 else 0
        
        return {
            'executive_metrics': {
                'total_feedback_analyzed': total_feedback,
                'customer_satisfaction_score': f"{csat_score}%",
                'customer_dissatisfaction_rate': f"{dissatisfaction_rate}%",
                'business_risk_level': business_impact.get('risk_level', 'UNKNOWN'),
                'estimated_churn_risk': f"{business_impact.get('estimated_churn', 0)}%",
                'revenue_impact': business_impact.get('revenue_impact', 'Unknown'),
                'brand_reputation_risk': business_impact.get('brand_reputation_risk', 'UNKNOWN')
            },
            'key_findings': [
                f"Critical Issues: {len([cat for cat in problem_analysis.values() if cat['severity'] == 'CRITICAL'])} categories require immediate attention",
                f"High Priority Issues: {len([cat for cat in problem_analysis.values() if cat['severity'] == 'HIGH'])} categories need urgent action",
                f"Total Problem Categories: {len(problem_analysis)} areas affecting customer experience",
                f"Customer Satisfaction: {csat_score}% (Industry benchmark: 80%+)",
                f"Business Risk: {business_impact.get('risk_level', 'UNKNOWN')} level risk requiring {business_impact.get('revenue_impact', 'Unknown')}"
            ],
            'immediate_actions': [
                "Implement 24-hour response protocol for all negative feedback",
                "Create escalation matrix for critical issues requiring management intervention",
                "Establish cross-functional customer experience improvement team",
                "Deploy real-time customer feedback monitoring dashboard"
            ],
            'strategic_initiatives': [
                "Develop comprehensive customer experience transformation roadmap",
                "Implement predictive customer satisfaction modeling",
                "Create customer success program to proactively address issues",
                "Establish customer feedback governance with quarterly business reviews"
            ]
        }
