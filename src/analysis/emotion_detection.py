"""
Enhanced emotion detection beyond basic sentiment analysis.

This module provides deeper emotional analysis of messages by detecting
specific emotions rather than just positive/negative sentiment.
"""

import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional, Union
import re
import time

# Configure logging
logger = logging.getLogger(__name__)

# Create a thread pool for CPU-bound NLP tasks
_executor = ThreadPoolExecutor(max_workers=4)

# Dictionary of emotion keywords for rule-based detection
EMOTION_KEYWORDS = {
    "joy": [
        "happy", "excited", "thrilled", "delighted", "ecstatic", "glad", "pleased", 
        "overjoyed", "elated", "jubilant", "yay", "woohoo", "love", "wonderful", 
        "amazing", "fantastic", "awesome", "great", "excellent", "perfect", "😊", 
        "😄", "😁", "🙂", "😀", "😃"
    ],
    "sadness": [
        "sad", "unhappy", "depressed", "miserable", "heartbroken", "disappointed", 
        "upset", "down", "blue", "gloomy", "sorrow", "grief", "regret", "mourn",
        "crying", "tears", "😢", "😭", "😔", "☹️", "😞", "😥"
    ],
    "anger": [
        "angry", "furious", "outraged", "livid", "enraged", "irate", "seething", 
        "mad", "annoyed", "irritated", "frustrated", "hate", "dislike", "despise",
        "resent", "rage", "hostile", "😠", "😡", "🤬", "😤"
    ],
    "fear": [
        "afraid", "scared", "frightened", "terrified", "fearful", "anxious", "worried", 
        "nervous", "panicked", "alarmed", "dreading", "horror", "panic", "paranoid",
        "distressed", "😨", "😰", "😱", "😧", "😦"
    ],
    "surprise": [
        "surprised", "shocked", "astonished", "amazed", "stunned", "startled", 
        "unexpected", "wow", "whoa", "omg", "oh my god", "unbelievable", "incredible",
        "😲", "😮", "😯", "😦", "😧", "🤯"
    ],
    "disgust": [
        "disgusted", "grossed", "repulsed", "revolted", "nauseated", "appalled", 
        "sickened", "yuck", "ew", "gross", "nasty", "unpleasant", "distaste",
        "🤢", "🤮", "😖"
    ],
    "love": [
        "love", "adore", "cherish", "affection", "fond", "care", "devoted", 
        "admire", "appreciate", "treasure", "smitten", "infatuated", "passionate",
        "❤️", "💕", "💓", "💗", "♥️", "😍", "🥰"
    ],
    "gratitude": [
        "grateful", "thankful", "appreciate", "thank you", "thanks", "blessed", 
        "indebted", "obliged", "recognition", "appreciation", "gratitude",
        "🙏", "👍"
    ],
    "confusion": [
        "confused", "puzzled", "perplexed", "bewildered", "unsure", "uncertain", 
        "baffled", "mystified", "unclear", "ambiguous", "huh", "what", "why",
        "🤔", "😕", "❓", "❔"
    ]
}

# Initialize regex patterns for each emotion
EMOTION_PATTERNS = {}
for emotion, keywords in EMOTION_KEYWORDS.items():
    # Create regex pattern for each emotion with word boundaries
    pattern = r'\b(' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
    EMOTION_PATTERNS[emotion] = re.compile(pattern, re.IGNORECASE)

async def detect_emotions(text: str) -> Dict[str, float]:
    """
    Detect emotions in text using a combination of rule-based and model-based approaches.
    
    Args:
        text: The text to analyze
        
    Returns:
        Dict mapping emotion categories to confidence scores (0-1)
    """
    if not text or len(text.strip()) < 2:
        return {emotion: 0.0 for emotion in EMOTION_KEYWORDS.keys()}
    
    try:
        # Try to get the running loop first
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # If there's no running loop, create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Run emotion detection in a thread pool
    return await loop.run_in_executor(_executor, _detect_emotions_sync, text)

def _detect_emotions_sync(text: str) -> Dict[str, float]:
    """
    Synchronous emotion detection implementation to run in thread pool.
    
    Args:
        text: The text to analyze
        
    Returns:
        Dict mapping emotion categories to confidence scores (0-1)
    """
    # Initialize emotion scores
    emotion_scores = {emotion: 0.0 for emotion in EMOTION_KEYWORDS.keys()}
    
    # Rule-based approach with regex patterns
    for emotion, pattern in EMOTION_PATTERNS.items():
        # Find all matches
        matches = pattern.findall(text)
        if matches:
            # Calculate score based on number of matches and text length
            # Normalize to avoid very long texts having too high scores
            normalized_count = min(len(matches) / (len(text.split()) * 0.2), 1.0)
            emotion_scores[emotion] = normalized_count
    
    # Try to use more sophisticated NLP model if available
    try:
        # Import here to avoid dependency issues
        from transformers import pipeline
        
        # Try to use a pre-trained emotion classification model
        classifier = pipeline("text-classification", 
                             model="bhadresh-savani/distilbert-base-uncased-emotion",
                             top_k=None)
        
        # Get model predictions
        model_results = classifier(text)
        
        # Map model results to our emotion categories
        model_mapping = {
            "sadness": "sadness",
            "joy": "joy",
            "love": "love",
            "anger": "anger",
            "fear": "fear",
            "surprise": "surprise"
        }
        
        # Update scores with model results
        for result in model_results[0]:
            model_emotion = result["label"]
            if model_emotion in model_mapping:
                # Map to our emotion categories
                our_emotion = model_mapping[model_emotion]
                # The model gives more accurate scores, so give it more weight
                emotion_scores[our_emotion] = result["score"]
    
    except (ImportError, Exception) as e:
        # If model-based approach fails, just use the rule-based results
        logger.debug(f"Using rule-based emotion detection. Model not available: {e}")
    
    # Ensure scores are between 0 and 1
    for emotion in emotion_scores:
        emotion_scores[emotion] = min(max(emotion_scores[emotion], 0.0), 1.0)
    
    return emotion_scores

async def analyze_emotions_in_messages(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze emotions in a list of messages.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Dict with emotion analysis results
    """
    if not messages:
        return {
            "emotions": {},
            "emotion_trends": {},
            "dominant_emotions": [],
            "emotional_shifts": []
        }
    
    # Process messages in batches
    tasks = []
    for message in messages:
        if "text" in message and message["text"]:
            task = asyncio.create_task(detect_emotions(message["text"]))
            task.message = message  # Attach message to task for later reference
            tasks.append(task)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    emotion_data = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Error detecting emotions: {result}")
            continue
            
        # Get the original message
        message = tasks[i].message
        
        # Add emotion data
        emotion_data.append({
            "message_id": message.get("id", i),
            "date": message.get("date"),
            "sender": message.get("sender"),
            "emotions": result
        })
    
    # Calculate aggregate emotion statistics
    aggregate_emotions = {}
    for emotion in EMOTION_KEYWORDS.keys():
        scores = [data["emotions"][emotion] for data in emotion_data]
        if scores:
            aggregate_emotions[emotion] = {
                "average": sum(scores) / len(scores),
                "max": max(scores),
                "count": sum(1 for score in scores if score > 0.3)  # Count significant occurrences
            }
    
    # Sort emotions by average score
    dominant_emotions = sorted(
        aggregate_emotions.items(), 
        key=lambda x: x[1]["average"], 
        reverse=True
    )
    
    # Calculate emotion trends over time
    emotion_trends = _calculate_emotion_trends(emotion_data)
    
    # Detect significant emotional shifts
    emotional_shifts = _detect_emotional_shifts(emotion_data)
    
    return {
        "emotions": aggregate_emotions,
        "dominant_emotions": dominant_emotions[:3],  # Top 3 emotions
        "emotion_trends": emotion_trends,
        "emotional_shifts": emotional_shifts
    }

def _calculate_emotion_trends(emotion_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate how emotions trend over time.
    
    Args:
        emotion_data: List of emotion data for messages
        
    Returns:
        Dict with emotion trends
    """
    # Skip if we don't have enough data
    if len(emotion_data) < 5:
        return {}
        
    # Sort by date if available
    dated_data = []
    for data in emotion_data:
        if "date" in data and data["date"]:
            try:
                dated_data.append(data)
            except:
                pass
                
    if not dated_data:
        return {}
        
    # Sort by date
    dated_data.sort(key=lambda x: x["date"])
    
    # Split into segments
    segment_count = min(5, len(dated_data) // 5)  # Aim for 5 segments when possible
    if segment_count < 2:
        return {}
        
    segment_size = len(dated_data) // segment_count
    segments = []
    
    for i in range(segment_count):
        start_idx = i * segment_size
        end_idx = start_idx + segment_size if i < segment_count - 1 else len(dated_data)
        segment = dated_data[start_idx:end_idx]
        segments.append(segment)
    
    # Calculate emotion averages for each segment
    trends = {}
    for emotion in EMOTION_KEYWORDS.keys():
        trends[emotion] = []
        
        for i, segment in enumerate(segments):
            scores = [data["emotions"][emotion] for data in segment]
            if scores:
                avg_score = sum(scores) / len(scores)
                
                # Get date range for segment
                start_date = segment[0]["date"]
                end_date = segment[-1]["date"]
                
                trends[emotion].append({
                    "segment": i + 1,
                    "start_date": start_date,
                    "end_date": end_date,
                    "average_score": avg_score
                })
    
    # Calculate trend directions
    for emotion, segments in trends.items():
        if len(segments) >= 2:
            first_score = segments[0]["average_score"]
            last_score = segments[-1]["average_score"]
            change = last_score - first_score
            
            if abs(change) < 0.1:
                trend_direction = "stable"
            elif change > 0:
                trend_direction = "increasing"
            else:
                trend_direction = "decreasing"
                
            trends[emotion + "_trend"] = {
                "direction": trend_direction,
                "change": change
            }
    
    return trends

def _detect_emotional_shifts(emotion_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Detect significant shifts in emotions.
    
    Args:
        emotion_data: List of emotion data for messages
        
    Returns:
        List of detected emotional shifts
    """
    # Skip if we don't have enough data
    if len(emotion_data) < 5:
        return []
        
    # Sort by date if available
    dated_data = []
    for data in emotion_data:
        if "date" in data and data["date"]:
            try:
                dated_data.append(data)
            except:
                pass
                
    if not dated_data:
        return []
        
    # Sort by date
    dated_data.sort(key=lambda x: x["date"])
    
    # Detect significant shifts in dominant emotion
    shifts = []
    window_size = min(3, len(dated_data) // 5)  # Use a sliding window of messages
    
    for i in range(window_size, len(dated_data)):
        # Get current and previous windows
        current_window = dated_data[i-window_size:i]
        next_window = dated_data[i:i+window_size]
        
        if not next_window:
            continue
            
        # Get dominant emotions for each window
        current_dominant = _get_dominant_emotion(current_window)
        next_dominant = _get_dominant_emotion(next_window)
        
        # If dominant emotion changed and the change is significant
        if current_dominant["emotion"] != next_dominant["emotion"]:
            # Calculate the magnitude of the shift
            magnitude = abs(next_dominant["score"] - current_dominant["score"])
            
            if magnitude > 0.2:  # Only report significant shifts
                shifts.append({
                    "from_emotion": current_dominant["emotion"],
                    "to_emotion": next_dominant["emotion"],
                    "magnitude": magnitude,
                    "from_date": current_window[0]["date"],
                    "to_date": next_window[-1]["date"]
                })
    
    # Return top 3 most significant shifts
    shifts.sort(key=lambda x: x["magnitude"], reverse=True)
    return shifts[:3]

def _get_dominant_emotion(emotion_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get the dominant emotion in a window of messages.
    
    Args:
        emotion_data: List of emotion data for messages
        
    Returns:
        Dict with dominant emotion and its score
    """
    # Initialize scores
    aggregate_scores = {emotion: 0.0 for emotion in EMOTION_KEYWORDS.keys()}
    
    # Sum scores for each emotion
    for data in emotion_data:
        for emotion, score in data["emotions"].items():
            aggregate_scores[emotion] += score
    
    # Get dominant emotion
    dominant_emotion = max(aggregate_scores.items(), key=lambda x: x[1])
    
    return {
        "emotion": dominant_emotion[0],
        "score": dominant_emotion[1] / len(emotion_data)  # Average score
    }

async def analyze_emotion_correlation(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze correlations between emotions and topics, senders, or time patterns.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Dict with emotion correlation analysis
    """
    from src.utils.topic_analyzer import extract_topics
    
    # Skip if we don't have enough messages
    if len(messages) < 10:
        return {"error": "Not enough messages for correlation analysis"}
    
    # First perform emotion analysis
    emotion_results = await analyze_emotions_in_messages(messages)
    
    # Extract topics
    topics_result = extract_topics(messages, max_topics=10)
    topics = topics_result.get("topics", [])
    
    # Analyze emotions by sender
    emotions_by_sender = {}
    sender_message_count = {}
    
    # Organize messages by sender
    messages_by_sender = {}
    for message in messages:
        sender = message.get("sender", "Unknown")
        if sender not in messages_by_sender:
            messages_by_sender[sender] = []
        messages_by_sender[sender].append(message)
    
    # Analyze emotions for each sender
    sender_emotion_tasks = []
    for sender, sender_messages in messages_by_sender.items():
        if len(sender_messages) >= 3:  # Only analyze senders with enough messages
            task = asyncio.create_task(analyze_emotions_in_messages(sender_messages))
            task.sender = sender
            sender_emotion_tasks.append(task)
    
    # Wait for all tasks to complete
    sender_results = await asyncio.gather(*sender_emotion_tasks, return_exceptions=True)
    
    # Process sender results
    for i, result in enumerate(sender_results):
        if isinstance(result, Exception):
            logger.error(f"Error analyzing emotions for sender: {result}")
            continue
            
        sender = sender_emotion_tasks[i].sender
        emotions_by_sender[sender] = result
    
    # Find topics with strongest emotional association
    emotional_topics = []
    for topic in topics:
        topic_text = topic["topic"]
        topic_messages = []
        
        # Find messages containing this topic
        for message in messages:
            if "text" in message and message["text"] and topic_text.lower() in message["text"].lower():
                topic_messages.append(message)
        
        if topic_messages:
            # Analyze emotions for this topic
            topic_emotions = await analyze_emotions_in_messages(topic_messages)
            
            # Get dominant emotion for this topic
            if "dominant_emotions" in topic_emotions and topic_emotions["dominant_emotions"]:
                dominant = topic_emotions["dominant_emotions"][0]
                
                emotional_topics.append({
                    "topic": topic_text,
                    "count": topic.get("count", len(topic_messages)),
                    "dominant_emotion": dominant[0],
                    "emotion_score": dominant[1]["average"],
                    "message_count": len(topic_messages)
                })
    
    # Sort topics by emotion strength
    emotional_topics.sort(key=lambda x: x["emotion_score"], reverse=True)
    
    return {
        "overall_emotions": emotion_results,
        "emotions_by_sender": emotions_by_sender,
        "emotional_topics": emotional_topics[:5]  # Top 5 most emotional topics
    }

async def analyze_relationship_dynamics(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze emotional relationship dynamics between conversation participants.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Dict with relationship dynamics analysis
    """
    # Skip if we don't have enough messages
    if len(messages) < 20:
        return {"error": "Not enough messages for relationship dynamics analysis"}
    
    # Identify participants
    participants = set()
    for message in messages:
        sender = message.get("sender")
        if sender:
            participants.add(sender)
    
    participants = list(participants)
    
    # Skip if we don't have enough participants
    if len(participants) < 2:
        return {"error": "Not enough participants for relationship analysis"}
    
    # Organize messages by sender
    messages_by_sender = {}
    for participant in participants:
        messages_by_sender[participant] = []
    
    for message in messages:
        sender = message.get("sender")
        if sender in participants:
            messages_by_sender[sender].append(message)
    
    # Analyze emotions for each participant
    participant_emotions = {}
    for participant, participant_messages in messages_by_sender.items():
        if len(participant_messages) >= 5:  # Only analyze participants with enough messages
            emotion_results = await analyze_emotions_in_messages(participant_messages)
            participant_emotions[participant] = emotion_results
    
    # Analyze interaction patterns
    interaction_patterns = []
    
    # For each pair of participants
    for i, participant1 in enumerate(participants):
        for participant2 in participants[i+1:]:
            # Skip if we don't have emotion data for either participant
            if participant1 not in participant_emotions or participant2 not in participant_emotions:
                continue
                
            # Get emotion data for each participant
            emotions1 = participant_emotions[participant1]
            emotions2 = participant_emotions[participant2]
            
            # Compare dominant emotions
            dominant1 = emotions1.get("dominant_emotions", [])
            dominant2 = emotions2.get("dominant_emotions", [])
            
            if not dominant1 or not dominant2:
                continue
                
            # Check if dominant emotions match or differ
            emotion_match = dominant1[0][0] == dominant2[0][0]
            
            # Find conversation segments where they interact directly
            direct_interactions = []
            for i, message in enumerate(messages[:-1]):
                if i + 1 >= len(messages):
                    continue
                    
                sender1 = message.get("sender")
                sender2 = messages[i+1].get("sender")
                
                if (sender1 == participant1 and sender2 == participant2) or \
                   (sender1 == participant2 and sender2 == participant1):
                    direct_interactions.append((message, messages[i+1]))
            
            # Analyze emotional reciprocity in direct interactions
            reciprocity_score = 0.0
            if direct_interactions:
                reciprocal_count = 0
                for msg1, msg2 in direct_interactions:
                    # Analyze emotions in each message
                    emotions_msg1 = await detect_emotions(msg1.get("text", ""))
                    emotions_msg2 = await detect_emotions(msg2.get("text", ""))
                    
                    # Get dominant emotions
                    dominant_msg1 = max(emotions_msg1.items(), key=lambda x: x[1])
                    dominant_msg2 = max(emotions_msg2.items(), key=lambda x: x[1])
                    
                    # Check if emotions match
                    if dominant_msg1[0] == dominant_msg2[0]:
                        reciprocal_count += 1
                
                reciprocity_score = reciprocal_count / len(direct_interactions)
            
            # Add to interaction patterns
            interaction_patterns.append({
                "participant1": participant1,
                "participant2": participant2,
                "emotion_match": emotion_match,
                "direct_interactions": len(direct_interactions),
                "reciprocity_score": reciprocity_score,
                "relationship_score": (0.5 + (reciprocity_score / 2))  # Scale from 0.5 to 1.0
            })
    
    return {
        "participants": list(participants),
        "participant_emotions": participant_emotions,
        "interaction_patterns": interaction_patterns
    }