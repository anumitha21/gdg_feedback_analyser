#!/usr/bin/env python3
"""
Test enhanced response generation and business insights.
"""

import pandas as pd
from app.response.generator import ResponseGenerator
from app.insights.aggregate import InsightAggregator

def test_enhanced_features():
    # Test enhanced response generation
    df = pd.DataFrame({
        'feedback_text': [
            'The delivery was late and the package was damaged',
            'Website is confusing and checkout failed',
            'Customer service was rude and unhelpful'
        ],
        'sentiment_label': ['NEGATIVE', 'NEGATIVE', 'NEGATIVE'],
        'topic': [0, 1, 2]
    })

    print('=== ENHANCED DRAFTED RESPONSES ===')
    generator = ResponseGenerator()
    for i, (_, row) in enumerate(df.iterrows(), 1):
        from app.response.generator import ResponseInput
        response_input = ResponseInput(
            feedback=row['feedback_text'],
            sentiment=row['sentiment_label'],
            topic=row['topic']
        )
        response = generator.generate([response_input])[0]
        print(f'{i}. Feedback: {row["feedback_text"]}')
        print(f'   Response: {response}')
        print()

    print('=== ENHANCED BUSINESS INSIGHTS ===')
    aggregator = InsightAggregator()
    insights = aggregator.aggregate(df)
    for i, rec in enumerate(insights['recommendations'], 1):
        print(f'{i}. {rec}')

if __name__ == "__main__":
    test_enhanced_features()
