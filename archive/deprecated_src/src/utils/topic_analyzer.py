#!/usr/bin/env python3
"""
Topic Analyzer Module

This module provides functionality for extracting topics from message content
and analyzing the sentiment associated with each topic. It uses natural language
processing techniques to identify key topics and determine sentiment polarity.
"""

import logging
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

# Configure logging
logger = logging.getLogger(__name__)

# Try to import NLP libraries with fallbacks for each
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    
    # Download required NLTK data if not already present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        
    NLTK_AVAILABLE = True
except ImportError:
    logger.warning("NLTK not available. Using simplified topic extraction.")
    NLTK_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    logger.warning("TextBlob not available. Using simplified sentiment analysis.")
    TEXTBLOB_AVAILABLE = False

# Fallback stop words if NLTK is not available
FALLBACK_STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 
    'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 
    'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 
    'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 
    'won', 'wouldn', 'would', 'could', 'should'
}

def preprocess_text(text: str) -> str:
    """
    Preprocess text for topic extraction.
    
    Args:
        text: Input text to preprocess
        
    Returns:
        Preprocessed text
    """
    if not text:
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """
    Extract keywords from text.
    
    Args:
        text: Text to extract keywords from
        top_n: Number of top keywords to extract
        
    Returns:
        List of keywords
    """
    if not text:
        return []
    
    # Preprocess text
    preprocessed_text = preprocess_text(text)
    
    if NLTK_AVAILABLE:
        # Tokenize
        tokens = word_tokenize(preprocessed_text)
        
        # Remove stop words
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [token for token in tokens if token.lower() not in stop_words and len(token) > 2]
    else:
        # Simple tokenization by splitting on whitespace
        tokens = preprocessed_text.split()
        
        # Remove stop words
        filtered_tokens = [token for token in tokens if token.lower() not in FALLBACK_STOP_WORDS and len(token) > 2]
    
    # Count occurrences
    counter = Counter(filtered_tokens)
    
    # Get top N keywords
    keywords = [item[0] for item in counter.most_common(top_n)]
    
    return keywords

def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyze sentiment of text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with sentiment analysis results
    """
    if not text:
        return {
            "polarity": 0,
            "subjectivity": 0,
            "sentiment": "neutral",
            "confidence": 0
        }
    
    if TEXTBLOB_AVAILABLE:
        # Use TextBlob for sentiment analysis
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Determine sentiment category
        if polarity > 0.1:
            sentiment = "positive"
        elif polarity < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
        # Calculate confidence (simplistic approach)
        confidence = abs(polarity) * (0.5 + subjectivity * 0.5)
    else:
        # Very simple fallback sentiment analysis based on positive/negative word lists
        positive_words = {
            'good', 'great', 'awesome', 'excellent', 'wonderful', 'fantastic',
            'happy', 'love', 'like', 'nice', 'best', 'better', 'amazing',
            'perfect', 'thank', 'thanks', 'appreciated', 'enjoy', 'fun',
            'positive', 'well', 'right', 'yes', 'agree', 'cool', 'fantastic'
        }
        
        negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'worse', 'worst',
            'hate', 'dislike', 'angry', 'sad', 'unhappy', 'upset',
            'wrong', 'no', 'not', 'never', 'fail', 'poor', 'sorry',
            'negative', 'problem', 'issue', 'fault', 'annoying', 'disgusting'
        }
        
        # Count positive and negative words
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        # Calculate polarity (-1 to 1) and subjectivity (0 to 1)
        total_words = len(words)
        if total_words > 0:
            polarity = (positive_count - negative_count) / total_words
            subjectivity = (positive_count + negative_count) / total_words
        else:
            polarity = 0
            subjectivity = 0
            
        # Determine sentiment category
        if polarity > 0.05:
            sentiment = "positive"
        elif polarity < -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
        # Calculate confidence (simplistic approach)
        confidence = abs(polarity) * (0.5 + subjectivity * 0.5)
    
    return {
        "polarity": round(polarity, 2),
        "subjectivity": round(subjectivity, 2),
        "sentiment": sentiment,
        "confidence": round(confidence, 2)
    }

def extract_topics(messages: List[Dict[str, Any]], max_topics: int = 10) -> List[Dict[str, Any]]:
    """
    Extract topics from messages.
    
    Args:
        messages: List of message objects
        max_topics: Maximum number of topics to extract
        
    Returns:
        List of topics with associated messages
    """
    if not messages:
        return []
    
    # Collect all message text
    all_text = ""
    topic_messages = defaultdict(list)
    
    for message in messages:
        text = message.get("text", "")
        if text:
            all_text += " " + text
            
    # Extract keywords from all text
    keywords = extract_keywords(all_text, top_n=max_topics * 2)
    
    # Associate messages with keywords
    for keyword in keywords:
        for message in messages:
            text = message.get("text", "").lower()
            if keyword in text.lower():
                topic_messages[keyword].append(message)
    
    # Sort topics by number of associated messages
    sorted_topics = sorted(
        topic_messages.keys(), 
        key=lambda k: len(topic_messages[k]), 
        reverse=True
    )
    
    # Keep only the top N topics
    top_topics = sorted_topics[:max_topics]
    
    # Create topic objects
    topics = []
    for topic in top_topics:
        associated_messages = topic_messages[topic]
        if not associated_messages:
            continue
            
        # Analyze sentiment for all messages about this topic
        topic_text = " ".join([msg.get("text", "") for msg in associated_messages])
        sentiment = analyze_sentiment(topic_text)
        
        topics.append({
            "topic": topic,
            "message_count": len(associated_messages),
            "sentiment": sentiment,
            "sample_messages": associated_messages[:3]  # Include a few sample messages
        })
    
    return topics

def extract_topics_with_sentiment(
    messages: List[Dict[str, Any]], 
    max_topics: int = 10
) -> Dict[str, Any]:
    """
    Extract topics from messages and analyze their sentiment.
    
    Args:
        messages: List of message objects
        max_topics: Maximum number of topics to extract
        
    Returns:
        Dictionary with topics, keywords, and conversation summary
    """
    try:
        # Extract all message text
        all_text = " ".join([msg.get("text", "") for msg in messages if msg.get("text")])
        
        # Extract topics
        topics = extract_topics(messages, max_topics=max_topics)
        
        # Extract keywords
        keywords = extract_keywords(all_text, top_n=20)
        
        # Analyze overall sentiment
        overall_sentiment = analyze_sentiment(all_text)
        
        # Generate a simple conversation summary
        conversation_summary = generate_conversation_summary(messages, topics, overall_sentiment)
        
        return {
            "topics": topics,
            "keywords": keywords,
            "overall_sentiment": overall_sentiment,
            "conversation_summary": conversation_summary
        }
    except Exception as e:
        logger.error(f"Error extracting topics with sentiment: {e}")
        return {
            "topics": [],
            "keywords": [],
            "overall_sentiment": {"sentiment": "neutral", "polarity": 0, "subjectivity": 0},
            "conversation_summary": "Unable to analyze conversation."
        }

def generate_conversation_summary(
    messages: List[Dict[str, Any]], 
    topics: List[Dict[str, Any]], 
    overall_sentiment: Dict[str, Any]
) -> str:
    """
    Generate a concise summary of the conversation based on topics and sentiment.
    
    Args:
        messages: List of message objects
        topics: List of topics extracted from messages
        overall_sentiment: Overall sentiment analysis
        
    Returns:
        A generated text summary of the conversation
    """
    if not messages or not topics:
        return "No significant conversation content to summarize."
    
    # Count messages by sender
    from_me_count = sum(1 for msg in messages if msg.get("is_from_me"))
    from_them_count = len(messages) - from_me_count
    
    # Determine conversation balance
    if from_me_count > from_them_count * 2:
        balance = "You sent significantly more messages than your contact."
    elif from_them_count > from_me_count * 2:
        balance = "Your contact sent significantly more messages than you."
    else:
        balance = "The conversation was fairly balanced between both participants."
    
    # Format top topics
    topic_phrases = []
    for i, topic in enumerate(topics[:3], 1):  # Top 3 topics
        sentiment = topic.get("sentiment", {}).get("sentiment", "neutral")
        count = topic.get("message_count", 0)
        
        if sentiment == "positive":
            tone = "positive tone"
        elif sentiment == "negative":
            tone = "negative tone"
        else:
            tone = "neutral tone"
            
        topic_phrases.append(f"{topic.get('topic', 'unknown')} (discussed in {count} messages with a {tone})")
    
    topic_text = ", ".join(topic_phrases)
    
    # Overall sentiment description
    sentiment = overall_sentiment.get("sentiment", "neutral")
    if sentiment == "positive":
        sentiment_desc = "generally positive"
    elif sentiment == "negative":
        sentiment_desc = "generally negative"
    else:
        sentiment_desc = "neutral"
    
    # Generate summary
    summary = (
        f"This conversation contains {len(messages)} messages. {balance} "
        f"The conversation tone was {sentiment_desc}. "
        f"Main topics discussed were: {topic_text}."
    )
    
    return summary
