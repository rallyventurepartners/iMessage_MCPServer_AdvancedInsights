import logging
import traceback
from collections import defaultdict
import asyncio
from datetime import datetime
import re
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
import spacy
from concurrent.futures import ThreadPoolExecutor

from src.database.async_messages_db import AsyncMessagesDB
from src.utils.redis_cache import cached

# Configure logging
logger = logging.getLogger(__name__)

# Try to load required NLTK data, download if needed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    
# Initialize stop words
STOP_WORDS = set(stopwords.words('english'))

# Try to load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
    # Disable unnecessary pipeline components for better performance
    nlp.disable_pipes("ner", "parser")
except:
    logger.warning("Could not load spaCy model. Running spacy download...")
    import subprocess
    subprocess.call([
        "python", "-m", "spacy", "download", "en_core_web_sm"
    ])
    nlp = spacy.load('en_core_web_sm')
    nlp.disable_pipes("ner", "parser")

# Create a thread pool for CPU-bound NLP tasks
_executor = ThreadPoolExecutor(max_workers=4)

# Batch size for processing messages
BATCH_SIZE = 500

async def analyze_sentiment_async(phone_number=None, chat_id=None, start_date=None, end_date=None, 
                           include_individual_messages=False, sentiment_threshold=0.2):
    """
    Analyze the sentiment of messages in a conversation over time.
    
    Uses NLP to determine the sentiment (positive, negative, neutral) of messages
    and track how sentiment changes over the course of the conversation.
    
    Args:
        phone_number (str, optional): The phone number of the contact.
        chat_id (int, optional): The ID of the group chat.
        start_date (str, optional): Start date for filtering messages (ISO format).
        end_date (str, optional): End date for filtering messages (ISO format).
        include_individual_messages (bool, optional): Whether to include sentiment for each message.
        sentiment_threshold (float, optional): Threshold for classifying messages as positive/negative.
        
    Returns:
        dict: Dictionary containing sentiment analysis results.
    """
    try:
        # Initialize DB connection
        db = AsyncMessagesDB()
        
        # Verify we have either a phone number or chat ID
        if not phone_number and not chat_id:
            return {"error": "Either phone number or chat ID is required"}
            
        # If we have both, prioritize chat_id
        if phone_number and not chat_id:
            # Try to find the chat ID for this contact
            async with db.get_db_connection() as connection:
                query = """
                SELECT DISTINCT chat_id 
                FROM chat_handle_join
                JOIN handle ON chat_handle_join.handle_id = handle.ROWID
                WHERE handle.id = ?
                """
                cursor = await connection.execute(query, (phone_number,))
                result = await cursor.fetchone()
                if result:
                    chat_id = result[0]
                    
        if not chat_id:
            return {"error": f"Could not find a chat with contact {phone_number}"}
            
        # Get chat participants
        participants = await db.get_chat_participants(chat_id)
        
        # Get chat info (name, etc.)
        chat_name = None
        if len(participants) > 1:
            # This is a group chat
            async with db.get_db_connection() as connection:
                query = "SELECT display_name FROM chat WHERE ROWID = ?"
                cursor = await connection.execute(query, (chat_id,))
                result = await cursor.fetchone()
                if result and result[0]:
                    chat_name = result[0]
                    
        # If no display name, use participants to create a name
        if not chat_name:
            if len(participants) > 3:
                names = [p['name'] for p in participants[:3]]
                chat_name = f"{', '.join(names)} & {len(participants)-3} others"
            else:
                names = [p['name'] for p in participants]
                chat_name = ', '.join(names)

        # Get all messages in this chat
        messages = await db.get_chat_messages(chat_id, start_date, end_date)
        
        # Process messages in batches for better performance
        return await process_messages_in_batches(
            messages, 
            participants, 
            chat_name, 
            include_individual_messages, 
            sentiment_threshold
        )
        
    except Exception as e:
        logger.error(f"Error in analyze_sentiment: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

async def process_messages_in_batches(messages, participants, chat_name, include_individual_messages, sentiment_threshold):
    """
    Process messages in batches to improve performance.
    
    Args:
        messages: List of message dictionaries
        participants: List of chat participants
        chat_name: Name of the chat
        include_individual_messages: Whether to include individual messages in the result
        sentiment_threshold: Threshold for classifying messages as positive/negative
        
    Returns:
        dict: Dictionary containing sentiment analysis results
    """
    # Group messages by sender
    messages_by_sender = defaultdict(list)
    for message in messages:
        messages_by_sender[message['sender_name']].append(message)
    
    # Group messages by time periods (day, week, month)
    messages_by_day = defaultdict(list)
    messages_by_week = defaultdict(list)
    messages_by_month = defaultdict(list)
    
    for message in messages:
        if 'date' in message and message['date']:
            date = datetime.fromisoformat(message['date'])
            day_key = date.strftime('%Y-%m-%d')
            week_key = f"{date.year}-W{date.strftime('%V')}"
            month_key = date.strftime('%Y-%m')
            
            messages_by_day[day_key].append(message)
            messages_by_week[week_key].append(message)
            messages_by_month[month_key].append(message)
    
    # Process messages in batches
    # Overall sentiment for all messages
    overall_batches = [messages[i:i+BATCH_SIZE] for i in range(0, len(messages), BATCH_SIZE)]
    overall_sentiments = await asyncio.gather(*[process_batch(batch) for batch in overall_batches])
    
    # Flatten results and calculate overall sentiment
    all_sentiments = []
    for batch_sentiments in overall_sentiments:
        all_sentiments.extend(batch_sentiments)
    
    overall_polarity = calculate_avg_sentiment(all_sentiments)
    overall_positive = sum(1 for s in all_sentiments if s > sentiment_threshold)
    overall_negative = sum(1 for s in all_sentiments if s < -sentiment_threshold)
    overall_neutral = len(all_sentiments) - overall_positive - overall_negative
    
    # Process by sender (in parallel)
    sender_results = {}
    sender_tasks = []
    
    for sender, sender_messages in messages_by_sender.items():
        task = asyncio.create_task(process_sender_messages(sender, sender_messages, sentiment_threshold))
        sender_tasks.append(task)
    
    sender_results_list = await asyncio.gather(*sender_tasks)
    for result in sender_results_list:
        sender_results[result['sender']] = result
    
    # Process by time period (in parallel)
    period_tasks = []
    
    # Process days
    for day, day_messages in messages_by_day.items():
        task = asyncio.create_task(process_day_messages(day, day_messages))
        period_tasks.append(task)
    
    # Process weeks
    for week, week_messages in messages_by_week.items():
        task = asyncio.create_task(process_week_messages(week, week_messages))
        period_tasks.append(task)
    
    # Process months
    for month, month_messages in messages_by_month.items():
        task = asyncio.create_task(process_month_messages(month, month_messages))
        period_tasks.append(task)
    
    period_results = await asyncio.gather(*period_tasks)
    
    # Organize period results
    days_sentiment = {}
    weeks_sentiment = {}
    months_sentiment = {}
    
    for result in period_results:
        if result['type'] == 'day':
            days_sentiment[result['period']] = result['sentiment']
        elif result['type'] == 'week':
            weeks_sentiment[result['period']] = result['sentiment']
        elif result['type'] == 'month':
            months_sentiment[result['period']] = result['sentiment']
    
    # Prepare the result
    result = {
        "chat_name": chat_name,
        "participant_count": len(participants),
        "message_count": len(messages),
        "overall_sentiment": {
            "polarity": overall_polarity,
            "positive_count": overall_positive,
            "negative_count": overall_negative,
            "neutral_count": overall_neutral,
            "positive_percentage": (overall_positive / len(all_sentiments)) * 100 if all_sentiments else 0,
            "negative_percentage": (overall_negative / len(all_sentiments)) * 100 if all_sentiments else 0,
            "neutral_percentage": (overall_neutral / len(all_sentiments)) * 100 if all_sentiments else 0
        },
        "by_sender": sender_results,
        "by_time": {
            "days": days_sentiment,
            "weeks": weeks_sentiment,
            "months": months_sentiment
        }
    }
    
    # Only include individual messages if requested
    if include_individual_messages:
        individual_messages = []
        for message in messages:
            if 'text' in message and message['text']:
                # Calculate sentiment for this message
                sentiment = await get_text_sentiment(message['text'])
                
                individual_messages.append({
                    "id": message.get('id'),
                    "sender": message.get('sender_name'),
                    "text": message.get('text'),
                    "date": message.get('date'),
                    "sentiment": sentiment
                })
        
        result["messages"] = individual_messages
    
    return result

async def process_batch(messages):
    """Process a batch of messages to analyze sentiment."""
    loop = asyncio.get_event_loop()
    
    # Extract text from messages
    texts = []
    for message in messages:
        if 'text' in message and message['text']:
            texts.append(message['text'])
    
    # Run TextBlob in a thread pool to avoid blocking the event loop
    sentiments = []
    if texts:
        # Run in thread pool
        sentiments = await loop.run_in_executor(_executor, process_texts_batch, texts)
    
    return sentiments

def process_texts_batch(texts):
    """Process a batch of text messages with TextBlob (runs in thread pool)."""
    return [TextBlob(text).sentiment.polarity for text in texts]

async def process_sender_messages(sender, messages, sentiment_threshold):
    """Process messages for a single sender."""
    # Process in batches
    batches = [messages[i:i+BATCH_SIZE] for i in range(0, len(messages), BATCH_SIZE)]
    sentiments_batches = await asyncio.gather(*[process_batch(batch) for batch in batches])
    
    # Flatten results
    all_sentiments = []
    for batch in sentiments_batches:
        all_sentiments.extend(batch)
    
    # Calculate metrics
    avg_sentiment = calculate_avg_sentiment(all_sentiments)
    positive = sum(1 for s in all_sentiments if s > sentiment_threshold)
    negative = sum(1 for s in all_sentiments if s < -sentiment_threshold)
    neutral = len(all_sentiments) - positive - negative
    
    return {
        "sender": sender,
        "message_count": len(messages),
        "avg_sentiment": avg_sentiment,
        "positive_count": positive,
        "negative_count": negative,
        "neutral_count": neutral,
        "positive_percentage": (positive / len(all_sentiments)) * 100 if all_sentiments else 0,
        "negative_percentage": (negative / len(all_sentiments)) * 100 if all_sentiments else 0,
        "neutral_percentage": (neutral / len(all_sentiments)) * 100 if all_sentiments else 0
    }

async def process_day_messages(day, messages):
    """Process messages for a single day."""
    batches = [messages[i:i+BATCH_SIZE] for i in range(0, len(messages), BATCH_SIZE)]
    sentiments_batches = await asyncio.gather(*[process_batch(batch) for batch in batches])
    
    # Flatten results
    all_sentiments = []
    for batch in sentiments_batches:
        all_sentiments.extend(batch)
    
    return {
        "type": "day",
        "period": day,
        "sentiment": calculate_avg_sentiment(all_sentiments),
        "message_count": len(messages)
    }

async def process_week_messages(week, messages):
    """Process messages for a single week."""
    batches = [messages[i:i+BATCH_SIZE] for i in range(0, len(messages), BATCH_SIZE)]
    sentiments_batches = await asyncio.gather(*[process_batch(batch) for batch in batches])
    
    # Flatten results
    all_sentiments = []
    for batch in sentiments_batches:
        all_sentiments.extend(batch)
    
    return {
        "type": "week",
        "period": week,
        "sentiment": calculate_avg_sentiment(all_sentiments),
        "message_count": len(messages)
    }

async def process_month_messages(month, messages):
    """Process messages for a single month."""
    batches = [messages[i:i+BATCH_SIZE] for i in range(0, len(messages), BATCH_SIZE)]
    sentiments_batches = await asyncio.gather(*[process_batch(batch) for batch in batches])
    
    # Flatten results
    all_sentiments = []
    for batch in sentiments_batches:
        all_sentiments.extend(batch)
    
    return {
        "type": "month",
        "period": month,
        "sentiment": calculate_avg_sentiment(all_sentiments),
        "message_count": len(messages)
    }

@cached(ttl=3600)  # Cache sentiment analysis for 1 hour
async def get_text_sentiment(text):
    """Get sentiment polarity for a text string.
    
    Now cached for better performance.
    
    Args:
        text (str): The text to analyze
        
    Returns:
        float: The sentiment polarity score (-1 to 1)
    """
    if not text:
        return 0.0
        
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, lambda: TextBlob(text).sentiment.polarity)

def calculate_avg_sentiment(sentiments):
    """Calculate the average sentiment from a list of sentiment values."""
    if not sentiments:
        return 0.0
    return sum(sentiments) / len(sentiments)

async def analyze_sentiment_trends_async(phone_number=None, chat_id=None, start_date=None, end_date=None):
    """
    Analyze sentiment trends over time for a conversation.
    
    Args:
        phone_number (str, optional): The phone number of the contact.
        chat_id (int, optional): The ID of the group chat.
        start_date (str, optional): Start date for filtering messages (ISO format).
        end_date (str, optional): End date for filtering messages (ISO format).
        
    Returns:
        dict: Dictionary containing sentiment trend analysis.
    """
    try:
        # Get basic sentiment analysis
        sentiment = await analyze_sentiment_async(phone_number, chat_id, start_date, end_date)
        
        if "error" in sentiment:
            return sentiment
            
        # Extract relevant data
        if "analysis" not in sentiment or "sentiment_over_time" not in sentiment["analysis"]:
            return {"error": "No sentiment data available for trend analysis"}
            
        sentiment_data = sentiment["analysis"]["sentiment_over_time"]
        
        # Ensure we have enough data points
        if len(sentiment_data) < 3:
            return {
                "chat": sentiment["chat"],
                "participants": sentiment["participants"],
                "warning": "Not enough data for trend analysis",
                "data": sentiment_data
            }
            
        # Calculate trend (simple linear regression)
        dates = list(range(len(sentiment_data)))
        sentiments = [point["avg_sentiment"] for point in sentiment_data]
        
        n = len(dates)
        sum_x = sum(dates)
        sum_y = sum(sentiments)
        sum_xy = sum(x * y for x, y in zip(dates, sentiments))
        sum_xx = sum(x * x for x in dates)
        
        # Calculate slope and intercept
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        # Generate trend line data
        trend_line = [{"date": point["date"], "trend": slope * i + intercept} 
                      for i, point in enumerate(sentiment_data)]
        
        # Determine overall trend
        trend_description = "stable"
        if slope > 0.01:
            trend_description = "improving"
        elif slope < -0.01:
            trend_description = "deteriorating"
            
        # Calculate volatility (standard deviation)
        mean = sum_y / n
        variance = sum((y - mean) ** 2 for y in sentiments) / n
        volatility = variance ** 0.5
        
        # Identify notable shifts
        shifts = []
        for i in range(1, len(sentiment_data)):
            prev = sentiment_data[i-1]["avg_sentiment"]
            curr = sentiment_data[i]["avg_sentiment"]
            change = curr - prev
            
            # Consider a significant shift if the change is more than 2x volatility
            if abs(change) > 2 * volatility:
                shifts.append({
                    "from_date": sentiment_data[i-1]["date"],
                    "to_date": sentiment_data[i]["date"],
                    "change": change,
                    "direction": "positive" if change > 0 else "negative"
                })
                
        # Prepare result
        return {
            "chat": sentiment["chat"],
            "analysis": {
                "trend": {
                    "slope": slope,
                    "description": trend_description,
                    "volatility": volatility
                },
                "trend_line": trend_line,
                "significant_shifts": shifts,
                "raw_data": sentiment_data
            }
        }
    except Exception as e:
        logger.error(f"Error in analyze_sentiment_trends: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)} 