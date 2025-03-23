"""
Contextual sentiment analysis for deeper message understanding.

This module provides enhanced sentiment analysis that considers conversation context,
sarcasm detection, quoted content, and other contextual factors.
"""

import logging
import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional, Union
from textblob import TextBlob

# Configure logging
logger = logging.getLogger(__name__)

# Create a thread pool for CPU-bound NLP tasks
_executor = ThreadPoolExecutor(max_workers=4)

# Patterns for detecting contextual elements
QUOTED_TEXT_PATTERN = re.compile(r'["\']([^"\']+)["\']')
SARCASM_INDICATORS = [
    "yeah right", "sure", "obviously", "totally", "of course", "clearly",
    "oh really", "wow", "great", "nice", "awesome", "perfect", "excellent",
    "🙄", "😏", "😒", "🙃", "/s"
]

async def analyze_contextual_sentiment(message: Dict[str, Any], 
                                       previous_messages: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Analyze sentiment with contextual awareness.
    
    Args:
        message: Current message to analyze
        previous_messages: List of preceding messages for context
        
    Returns:
        dict: Enhanced sentiment analysis with contextual factors
    """
    if not message or "text" not in message or not message["text"]:
        return {
            "raw_sentiment": 0.0,
            "adjusted_sentiment": 0.0,
            "confidence": 0.0,
            "contextual_factors": []
        }
    
    text = message["text"]
    
    try:
        # Try to get the running loop first
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # If there's no running loop, create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Run the analysis in a thread pool to avoid blocking
    result = await loop.run_in_executor(
        _executor, 
        _analyze_contextual_sentiment_sync, 
        text, 
        previous_messages
    )
    
    return result

def _analyze_contextual_sentiment_sync(text: str, 
                                      previous_messages: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Synchronous implementation of contextual sentiment analysis.
    
    Args:
        text: Text to analyze
        previous_messages: Previous messages for context
        
    Returns:
        dict: Enhanced sentiment analysis
    """
    # Initialize result
    result = {
        "raw_sentiment": 0.0,
        "adjusted_sentiment": 0.0,
        "confidence": 0.0,
        "contextual_factors": []
    }
    
    # Get basic sentiment
    try:
        blob = TextBlob(text)
        raw_sentiment = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        result["raw_sentiment"] = raw_sentiment
        result["subjectivity"] = subjectivity
        
        # Start with the raw sentiment
        adjusted_sentiment = raw_sentiment
        confidence = 0.7  # Default confidence
        
        # Track contextual factors that influence sentiment
        contextual_factors = []
        
        # 1. Check for quoted text
        quoted_text = QUOTED_TEXT_PATTERN.findall(text)
        if quoted_text:
            # Analyze the quoted content separately
            quoted_sentiment_sum = 0
            for quote in quoted_text:
                if len(quote) > 3:  # Only analyze meaningful quotes
                    quote_blob = TextBlob(quote)
                    quoted_sentiment_sum += quote_blob.sentiment.polarity
            
            if quoted_text:
                quoted_sentiment_avg = quoted_sentiment_sum / len(quoted_text)
                
                # If quoted sentiment differs from overall sentiment, adjust
                if abs(quoted_sentiment_avg - raw_sentiment) > 0.3:
                    # Reduce the weight of the quoted content in the overall sentiment
                    adjustment = (quoted_sentiment_avg - raw_sentiment) * 0.3
                    adjusted_sentiment -= adjustment
                    
                    contextual_factors.append({
                        "type": "quoted_content",
                        "impact": "detected quote that may not reflect sender's opinion",
                        "adjustment": -adjustment
                    })
                    
                    # Reduce confidence slightly
                    confidence -= 0.1
        
        # 2. Check for sarcasm indicators
        sarcasm_detected = False
        for indicator in SARCASM_INDICATORS:
            if indicator.lower() in text.lower():
                sarcasm_detected = True
                break
        
        # Additional sarcasm detection for common patterns
        if not sarcasm_detected:
            # Check for extreme positive sentiment with negative indicators
            if raw_sentiment > 0.6 and any(neg in text.lower() for neg in ["bad", "terrible", "worst", "awful"]):
                sarcasm_detected = True
                
            # Check for question marks combined with positive sentiment
            elif raw_sentiment > 0.5 and "?" in text:
                sarcasm_detected = True
        
        if sarcasm_detected:
            # Invert sentiment for suspected sarcasm
            adjusted_sentiment = -raw_sentiment * 0.7
            
            contextual_factors.append({
                "type": "sarcasm",
                "impact": "potential sarcasm detected; sentiment may be inverted",
                "adjustment": adjusted_sentiment - raw_sentiment
            })
            
            # Reduce confidence for sarcasm detection
            confidence -= 0.2
        
        # 3. Consider previous message context if available
        if previous_messages and len(previous_messages) > 0:
            # Get the sentiment of previous messages
            prev_sentiments = []
            for prev_msg in previous_messages[-3:]:  # Look at last 3 messages
                if "text" in prev_msg and prev_msg["text"]:
                    prev_blob = TextBlob(prev_msg["text"])
                    prev_sentiments.append(prev_blob.sentiment.polarity)
            
            if prev_sentiments:
                avg_prev_sentiment = sum(prev_sentiments) / len(prev_sentiments)
                
                # Check for large sentiment shifts
                sentiment_shift = raw_sentiment - avg_prev_sentiment
                
                if abs(sentiment_shift) > 0.6:
                    # Extreme shifts might indicate context we're missing
                    # Slightly moderate the shift
                    moderation = sentiment_shift * 0.2
                    adjusted_sentiment -= moderation
                    
                    contextual_factors.append({
                        "type": "sentiment_shift",
                        "impact": "large shift from previous messages; moderating",
                        "adjustment": -moderation
                    })
                    
                    # Reduce confidence for large shifts
                    confidence -= 0.1
        
        # 4. Adjust for message length - very short messages are less reliable
        if len(text.split()) < 4:
            # Reduce the sentiment intensity for very short messages
            adjusted_sentiment *= 0.8
            
            contextual_factors.append({
                "type": "message_length",
                "impact": "very short message; reducing intensity",
                "adjustment": adjusted_sentiment - raw_sentiment
            })
            
            # Reduce confidence for short messages
            confidence -= 0.15
        
        # 5. Look for emphasis that might amplify sentiment
        emphasis_count = text.count('!') + text.count('?') + sum(1 for c in text if c.isupper())
        if emphasis_count > 3:
            # Amplify sentiment for emphasized text
            emphasis_factor = min(emphasis_count * 0.05, 0.3)  # Cap at 30% increase
            emphasis_adjustment = raw_sentiment * emphasis_factor
            adjusted_sentiment += emphasis_adjustment
            
            contextual_factors.append({
                "type": "emphasis",
                "impact": "detected emphasis (!, ?, CAPS); amplifying",
                "adjustment": emphasis_adjustment
            })
        
        # Ensure sentiment is within bounds
        adjusted_sentiment = max(min(adjusted_sentiment, 1.0), -1.0)
        
        # Ensure confidence is within bounds
        confidence = max(min(confidence, 1.0), 0.3)
        
        # Update the result
        result["adjusted_sentiment"] = adjusted_sentiment
        result["confidence"] = confidence
        result["contextual_factors"] = contextual_factors
        
        # Add a category for easier interpretation
        if adjusted_sentiment >= 0.3:
            category = "positive"
        elif adjusted_sentiment <= -0.3:
            category = "negative"
        else:
            category = "neutral"
            
        result["category"] = category
        
    except Exception as e:
        logger.error(f"Error in contextual sentiment analysis: {e}")
        # Return basic sentiment with low confidence
        try:
            blob = TextBlob(text)
            result["raw_sentiment"] = blob.sentiment.polarity
            result["adjusted_sentiment"] = blob.sentiment.polarity
            result["confidence"] = 0.4
        except:
            # If even that fails, return zeros
            pass
    
    return result

async def analyze_message_thread(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Analyze sentiment for a thread of messages with contextual awareness.
    
    Args:
        messages: List of messages in chronological order
        
    Returns:
        List of messages with enhanced sentiment analysis
    """
    if not messages:
        return []
    
    # Process messages in order, providing context from previous messages
    enhanced_messages = []
    
    for i, message in enumerate(messages):
        # Get previous messages for context (up to 3)
        context_start = max(0, i - 3)
        previous_messages = messages[context_start:i]
        
        # Analyze sentiment with context
        sentiment_result = await analyze_contextual_sentiment(message, previous_messages)
        
        # Create enhanced message
        enhanced_message = {**message}  # Copy original message
        enhanced_message["contextual_sentiment"] = sentiment_result
        
        enhanced_messages.append(enhanced_message)
    
    return enhanced_messages

async def analyze_conversation_sentiment(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the overall sentiment of a conversation with contextual awareness.
    
    Args:
        messages: List of messages in chronological order
        
    Returns:
        Dict with enhanced conversation sentiment analysis
    """
    if not messages:
        return {
            "overall_sentiment": 0.0,
            "confidence": 0.0,
            "segments": []
        }
    
    # First analyze individual messages with context
    enhanced_messages = await analyze_message_thread(messages)
    
    # Calculate overall sentiment statistics
    raw_sentiments = [msg["contextual_sentiment"]["raw_sentiment"] for msg in enhanced_messages]
    adjusted_sentiments = [msg["contextual_sentiment"]["adjusted_sentiment"] for msg in enhanced_messages]
    confidence_scores = [msg["contextual_sentiment"]["confidence"] for msg in enhanced_messages]
    
    # Calculate weighted average of sentiments based on confidence
    sentiment_weights = [conf + 0.5 for conf in confidence_scores]  # Ensure even low confidence gets some weight
    
    weighted_sum = sum(s * w for s, w in zip(adjusted_sentiments, sentiment_weights))
    weight_sum = sum(sentiment_weights)
    
    overall_sentiment = weighted_sum / weight_sum if weight_sum > 0 else 0.0
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
    
    # Divide conversation into segments
    segment_count = min(5, len(messages) // 10)  # Aim for 5 segments
    segment_count = max(1, segment_count)  # Ensure at least 1 segment
    
    segment_size = len(messages) // segment_count
    segments = []
    
    for i in range(segment_count):
        start_idx = i * segment_size
        end_idx = start_idx + segment_size if i < segment_count - 1 else len(messages)
        
        segment_messages = enhanced_messages[start_idx:end_idx]
        segment_sentiments = [msg["contextual_sentiment"]["adjusted_sentiment"] for msg in segment_messages]
        
        if segment_sentiments:
            avg_sentiment = sum(segment_sentiments) / len(segment_sentiments)
            
            # Get first and last message dates if available
            start_date = segment_messages[0].get("date", f"Message {start_idx+1}")
            end_date = segment_messages[-1].get("date", f"Message {end_idx}")
            
            segments.append({
                "start_index": start_idx,
                "end_index": end_idx - 1,
                "start_date": start_date,
                "end_date": end_date,
                "sentiment": avg_sentiment,
                "message_count": len(segment_messages)
            })
    
    # Detect sentiment shifts between segments
    sentiment_shifts = []
    for i in range(1, len(segments)):
        prev_sentiment = segments[i-1]["sentiment"]
        curr_sentiment = segments[i]["sentiment"]
        
        shift = curr_sentiment - prev_sentiment
        
        if abs(shift) > 0.3:  # Only report significant shifts
            sentiment_shifts.append({
                "from_segment": i,
                "to_segment": i + 1,
                "shift": shift,
                "from_date": segments[i-1]["start_date"],
                "to_date": segments[i]["end_date"]
            })
    
    # Determine overall category
    if overall_sentiment >= 0.3:
        category = "positive"
    elif overall_sentiment <= -0.3:
        category = "negative"
    else:
        category = "neutral"
    
    return {
        "overall_sentiment": overall_sentiment,
        "category": category,
        "confidence": avg_confidence,
        "segments": segments,
        "sentiment_shifts": sentiment_shifts
    }