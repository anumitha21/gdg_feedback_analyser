#!/usr/bin/env python3
"""
Quick test script to verify sentiment analysis improvements.
"""

from app.analysis.sentiment import SentimentAnalyzer
from app.utils.preprocess import clean_text

def test_sentiment():
    # Test with the problematic feedback
    test_feedback = [
        "Long linesLimited beetsNo pint container of heavy creamAll frozen pie crust were broken they been in the freezer over 3 months like thatNo frozen stuffed crust pizza with chickenCustomer service had only two workersThen another one came out to helpOther workers walking around the storeI didn't see the store manager walking around the store to check on customer service.Some people refused to wait on the long return line instead they went shopping and they'll check how the long the line is on their way out.",
        "Absolutely love the product quality and the customer service is great!",
        "The delivery was late and the package was damaged.",
        "I found the app confusing to navigate and the checkout failed twice.",
        "Support resolved my issue quickly. Very satisfied."
    ]
    
    print("Testing sentiment analysis...")
    analyzer = SentimentAnalyzer()
    
    for i, feedback in enumerate(test_feedback, 1):
        clean_fb = clean_text(feedback)
        result = analyzer.predict([clean_fb])[0]
        print(f"\n{i}. Original: {feedback[:100]}...")
        print(f"   Cleaned: {clean_fb[:100]}...")
        print(f"   Sentiment: {result['label']} (confidence: {result['score']:.3f})")
        
        # Test rule-based directly
        rule_result = analyzer._rule_based_sentiment(clean_fb)
        print(f"   Rule-based: {rule_result['label']} (confidence: {rule_result['score']:.3f})")

if __name__ == "__main__":
    test_sentiment()
