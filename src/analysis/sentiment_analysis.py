import logging
import traceback
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union

from src.database.messages_db import MessagesDB
from src.utils.helpers import error_response, validate_date_range, ensure_datetime, resolve_chat_id

logger = logging.getLogger(__name__)

def analyze_sentiment(
    phone_number: str = None,
    chat_id: Union[str, int] = None,
    start_date: str = None,
    end_date: str = None,
    include_individual_messages: bool = False,
    sentiment_threshold: float = 0.3
) -> Dict[str, Any]:
    """Analyze the sentiment of messages in a conversation over time.
    
    This function uses Natural Language Processing techniques to determine
    the sentiment (positive, negative, neutral) of messages and track how
    sentiment changes over time in the conversation.
    
    Args:
        phone_number: Phone number/email of the contact to analyze
        chat_id: ID of the chat to analyze (can be group chat or individual)
        start_date: Optional start date in ISO format (YYYY-MM-DD)
        end_date: Optional end date in ISO format (YYYY-MM-DD)
        include_individual_messages: Whether to include sentiment for individual messages
        sentiment_threshold: Threshold for considering sentiment positive/negative (0-1)
        
    Returns:
        Dictionary containing sentiment analysis results
    """
    logger.info(f"analyze_sentiment called with phone_number={phone_number}, chat_id={chat_id}, "
               f"start_date={start_date}, end_date={end_date}")
    
    # Validate required parameters
    if not phone_number and not chat_id:
        return error_response("MISSING_PARAMETER", "Either phone_number or chat_id is required")
    
    # Validate date range
    error = validate_date_range(start_date, end_date)
    if error:
        return error
    
    try:
        # Try to import TextBlob for sentiment analysis
        try:
            from textblob import TextBlob
        except ImportError:
            return error_response(
                "MISSING_DEPENDENCY", 
                "This analysis requires the TextBlob library. Please install with: pip install textblob"
            )
        
        db = MessagesDB()
        
        # Convert date strings to datetime objects
        start_dt = ensure_datetime(start_date) if start_date else None
        end_dt = ensure_datetime(end_date) if end_date else None
        
        # Resolve chat_id based on phone_number if needed
        resolved_chat_id = resolve_chat_id(db, chat_id, phone_number)
        if not resolved_chat_id:
            return error_response("INVALID_PARAMETER", "Could not find a chat for the provided parameters")
        
        # Get messages for analysis
        messages = db.get_chat_transcript(
            chat_id=resolved_chat_id,
            phone_number=phone_number,
            start_date=start_date,
            end_date=end_date
        )
        
        if not messages:
            return {
                "warning": "No messages found for the specified parameters",
                "overall_sentiment": "neutral",
                "sentiment_stats": {
                    "positive": 0,
                    "negative": 0,
                    "neutral": 0
                }
            }
        
        # Prepare data structures for sentiment analysis
        sentiment_scores = []
        sentiment_by_date = defaultdict(list)
        sentiment_by_sender = defaultdict(list)
        positive_messages = []
        negative_messages = []
        
        # Process each message
        for message in messages:
            # Extract message data
            text = message.get("text", "")
            date = message.get("date")
            sender = message.get("sender", "Unknown")
            
            # Skip empty messages
            if not text or len(text.strip()) < 2:
                continue
            
            # Analyze sentiment using TextBlob
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1 scale
            subjectivity = blob.sentiment.subjectivity  # 0 to 1 scale
            
            # Determine sentiment category
            sentiment_category = "neutral"
            if polarity >= sentiment_threshold:
                sentiment_category = "positive"
            elif polarity <= -sentiment_threshold:
                sentiment_category = "negative"
            
            # Store the sentiment data
            sentiment_data = {
                "text": text[:100] + ("..." if len(text) > 100 else ""),  # Truncate long messages
                "polarity": round(polarity, 3),
                "subjectivity": round(subjectivity, 3),
                "category": sentiment_category,
                "date": date,
                "sender": sender
            }
            
            sentiment_scores.append(polarity)
            
            # Organize by date
            if date:
                try:
                    message_date = ensure_datetime(date)
                    date_key = message_date.strftime("%Y-%m-%d")
                    sentiment_by_date[date_key].append(polarity)
                except:
                    pass
            
            # Organize by sender
            sentiment_by_sender[sender].append(polarity)
            
            # Store very positive and very negative messages
            if polarity >= 0.5:
                positive_messages.append(sentiment_data)
            elif polarity <= -0.5:
                negative_messages.append(sentiment_data)
        
        # Calculate overall sentiment statistics
        total_messages = len(sentiment_scores)
        if total_messages == 0:
            return {
                "warning": "No valid messages for sentiment analysis",
                "overall_sentiment": "neutral",
                "sentiment_stats": {
                    "positive": 0,
                    "negative": 0,
                    "neutral": 0
                }
            }
        
        # Count messages by sentiment category
        positive_count = sum(1 for score in sentiment_scores if score >= sentiment_threshold)
        negative_count = sum(1 for score in sentiment_scores if score <= -sentiment_threshold)
        neutral_count = total_messages - positive_count - negative_count
        
        # Calculate average sentiment
        avg_sentiment = sum(sentiment_scores) / total_messages
        
        # Determine overall sentiment category
        overall_sentiment = "neutral"
        if avg_sentiment >= sentiment_threshold:
            overall_sentiment = "positive"
        elif avg_sentiment <= -sentiment_threshold:
            overall_sentiment = "negative"
        
        # Calculate sentiment trends over time
        daily_sentiment = {}
        for date, scores in sentiment_by_date.items():
            daily_sentiment[date] = round(sum(scores) / len(scores), 3)
        
        # Calculate weekly sentiment averages
        weekly_sentiment = {}
        if sentiment_by_date:
            # Get all dates and sort them
            all_dates = [datetime.strptime(date, "%Y-%m-%d") for date in sentiment_by_date.keys()]
            all_dates.sort()
            
            # Calculate week numbers
            min_date = min(all_dates)
            for date_str, scores in sentiment_by_date.items():
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                days_diff = (date_obj - min_date).days
                week_num = days_diff // 7
                week_key = f"Week {week_num+1}"
                
                if week_key not in weekly_sentiment:
                    weekly_sentiment[week_key] = []
                
                weekly_sentiment[week_key].extend(scores)
            
            # Calculate average for each week
            for week, scores in weekly_sentiment.items():
                weekly_sentiment[week] = round(sum(scores) / len(scores), 3)
        
        # Calculate monthly sentiment if applicable
        monthly_sentiment = {}
        for date_str, scores in sentiment_by_date.items():
            month_key = date_str[:7]  # YYYY-MM format
            if month_key not in monthly_sentiment:
                monthly_sentiment[month_key] = []
            
            monthly_sentiment[month_key].extend(scores)
        
        # Calculate average for each month
        for month, scores in monthly_sentiment.items():
            monthly_sentiment[month] = round(sum(scores) / len(scores), 3)
        
        # Analyze sentiment by participant
        participant_sentiment = {}
        for sender, scores in sentiment_by_sender.items():
            # Skip participants with very few messages
            if len(scores) < 3:
                continue
                
            avg_sender_sentiment = sum(scores) / len(scores)
            
            # Determine sender's sentiment category
            sender_sentiment_category = "neutral"
            if avg_sender_sentiment >= sentiment_threshold:
                sender_sentiment_category = "positive"
            elif avg_sender_sentiment <= -sentiment_threshold:
                sender_sentiment_category = "negative"
            
            participant_sentiment[sender] = {
                "average_sentiment": round(avg_sender_sentiment, 3),
                "category": sender_sentiment_category,
                "message_count": len(scores),
                "positive_percentage": round(sum(1 for s in scores if s >= sentiment_threshold) / len(scores) * 100, 1),
                "negative_percentage": round(sum(1 for s in scores if s <= -sentiment_threshold) / len(scores) * 100, 1)
            }
        
        # Find emotional peaks
        sentiment_volatility = 0
        sentiment_shifts = []
        
        if len(sentiment_scores) >= 2:
            # Calculate volatility (standard deviation)
            import numpy as np
            sentiment_volatility = round(float(np.std(sentiment_scores)), 3)
            
            # Find significant sentiment shifts
            prev_score = sentiment_scores[0]
            for i, score in enumerate(sentiment_scores[1:], 1):
                shift = score - prev_score
                if abs(shift) >= 0.5:  # Significant shift threshold
                    try:
                        shift_data = {
                            "from": round(prev_score, 2),
                            "to": round(score, 2),
                            "shift": round(shift, 2),
                            "message": messages[i].get("text", "")[:100] + ("..." if len(messages[i].get("text", "")) > 100 else ""),
                            "date": messages[i].get("date")
                        }
                        sentiment_shifts.append(shift_data)
                    except IndexError:
                        pass
                prev_score = score
        
        # Sort lists for presentation
        positive_messages.sort(key=lambda x: x["polarity"], reverse=True)
        negative_messages.sort(key=lambda x: x["polarity"])
        sentiment_shifts.sort(key=lambda x: abs(x["shift"]), reverse=True)
        
        # Prepare result
        result = {
            "overall_sentiment": overall_sentiment,
            "average_score": round(avg_sentiment, 3),
            "volatility": sentiment_volatility,
            "sentiment_stats": {
                "positive": positive_count,
                "negative": negative_count,
                "neutral": neutral_count,
                "positive_percentage": round(positive_count / total_messages * 100, 1),
                "negative_percentage": round(negative_count / total_messages * 100, 1),
                "neutral_percentage": round(neutral_count / total_messages * 100, 1)
            },
            "time_trends": {
                "daily": daily_sentiment,
                "weekly": weekly_sentiment,
                "monthly": monthly_sentiment
            },
            "participants": participant_sentiment,
            "emotional_peaks": {
                "most_positive": positive_messages[:5],  # Limit to top 5
                "most_negative": negative_messages[:5],  # Limit to top 5
                "significant_shifts": sentiment_shifts[:10]  # Limit to top 10
            }
        }
        
        # Include individual message sentiment if requested
        if include_individual_messages:
            messages_with_sentiment = []
            for i, message in enumerate(messages):
                try:
                    text = message.get("text", "")
                    # Skip empty messages
                    if not text or len(text.strip()) < 2:
                        continue
                        
                    # Get sentiment data
                    blob = TextBlob(text)
                    polarity = blob.sentiment.polarity
                    
                    # Determine category
                    category = "neutral"
                    if polarity >= sentiment_threshold:
                        category = "positive"
                    elif polarity <= -sentiment_threshold:
                        category = "negative"
                    
                    # Add sentiment data to message
                    message_copy = message.copy()
                    message_copy["sentiment"] = {
                        "score": round(polarity, 3),
                        "category": category
                    }
                    messages_with_sentiment.append(message_copy)
                except:
                    # Skip errors in individual messages
                    pass
            
            result["messages"] = messages_with_sentiment
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        logger.error(traceback.format_exc())
        return error_response("ANALYSIS_ERROR", f"Error analyzing sentiment: {str(e)}") 