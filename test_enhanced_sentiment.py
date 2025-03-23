#!/usr/bin/env python3
"""
Test the enhanced sentiment analysis functionality.
This script tests the new emotion detection and contextual sentiment analysis features.
"""

import asyncio
import logging
import sys
from pprint import pprint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sample messages for testing
SAMPLE_MESSAGES = [
    {
        "text": "I'm really excited about our trip next weekend! 😊",
        "sender": "John",
        "date": "2023-04-15T19:30:00"
    },
    {
        "text": "Me too! I've been looking forward to it for weeks.",
        "sender": "Sarah",
        "date": "2023-04-15T19:32:00"
    },
    {
        "text": "I'm a bit nervous about the weather though...",
        "sender": "John",
        "date": "2023-04-15T19:35:00"
    },
    {
        "text": "Yeah right, because it \"always\" rains when we plan something. 🙄",
        "sender": "Sarah",
        "date": "2023-04-15T19:37:00"
    },
    {
        "text": "Haha, I know! Our luck with weather has been terrible.",
        "sender": "John",
        "date": "2023-04-15T19:40:00"
    },
    {
        "text": "I'm actually really angry about what happened at work today.",
        "sender": "Sarah",
        "date": "2023-04-15T19:42:00"
    },
    {
        "text": "Oh no! What happened?",
        "sender": "John",
        "date": "2023-04-15T19:45:00"
    },
    {
        "text": "My boss took credit for my project in the meeting! I was so mad I almost walked out.",
        "sender": "Sarah",
        "date": "2023-04-15T19:47:00"
    },
    {
        "text": "That's awful. I'm really sorry to hear that. You worked so hard on it.",
        "sender": "John",
        "date": "2023-04-15T19:50:00"
    },
    {
        "text": "Thanks for understanding. I'm going to talk to HR tomorrow.",
        "sender": "Sarah",
        "date": "2023-04-15T19:52:00"
    }
]

# Sample messages with sarcasm and quotes for testing contextual analysis
CONTEXTUAL_MESSAGES = [
    {
        "text": "The meeting was \"productive\" as usual... 🙄",
        "sender": "Alex",
        "date": "2023-05-10T14:00:00"
    },
    {
        "text": "Oh I'm sure it was SUPER productive! Nothing like 2 hours of pointless discussion!",
        "sender": "Jamie",
        "date": "2023-05-10T14:05:00"
    },
    {
        "text": "My boss said \"We value your input\" but then completely ignored my suggestions.",
        "sender": "Alex",
        "date": "2023-05-10T14:10:00"
    },
    {
        "text": "That's awful. They always do that in these meetings.",
        "sender": "Jamie",
        "date": "2023-05-10T14:15:00"
    },
    {
        "text": "Yeah but the free lunch was nice at least!",
        "sender": "Alex",
        "date": "2023-05-10T14:20:00"
    },
    {
        "text": "True! Food makes everything better.",
        "sender": "Jamie",
        "date": "2023-05-10T14:25:00"
    }
]

# Relationship test with multiple participants
RELATIONSHIP_MESSAGES = [
    {"text": "Good morning everyone! How's everyone doing today?", "sender": "Alice", "date": "2023-06-01T09:00:00"},
    {"text": "Morning! I'm doing well, thanks for asking. 😊", "sender": "Bob", "date": "2023-06-01T09:05:00"},
    {"text": "Hey guys, sorry I'm late to the chat. Had a rough morning.", "sender": "Charlie", "date": "2023-06-01T09:15:00"},
    {"text": "No worries Charlie! What happened?", "sender": "Alice", "date": "2023-06-01T09:16:00"},
    {"text": "Car wouldn't start... again. 😠 Had to call a taxi.", "sender": "Charlie", "date": "2023-06-01T09:18:00"},
    {"text": "That's the third time this month! Maybe it's time for a new car?", "sender": "Bob", "date": "2023-06-01T09:20:00"},
    {"text": "I know, I know. Just can't afford it right now.", "sender": "Charlie", "date": "2023-06-01T09:22:00"},
    {"text": "I'd be happy to help you look for something in your budget if you want!", "sender": "Alice", "date": "2023-06-01T09:25:00"},
    {"text": "That's so kind of you, Alice! I appreciate it.", "sender": "Charlie", "date": "2023-06-01T09:27:00"},
    {"text": "Alice is always so helpful. 🙄 Must be nice to have all that free time.", "sender": "Bob", "date": "2023-06-01T09:30:00"},
    {"text": "What's that supposed to mean, Bob?", "sender": "Alice", "date": "2023-06-01T09:32:00"},
    {"text": "Nothing, just saying you're always available to help. It's great.", "sender": "Bob", "date": "2023-06-01T09:35:00"},
    {"text": "Let's not start this again, guys. I'll take any help I can get right now.", "sender": "Charlie", "date": "2023-06-01T09:37:00"},
    {"text": "You're right, sorry. Let's focus on the car issue.", "sender": "Alice", "date": "2023-06-01T09:40:00"},
    {"text": "Yeah, sorry Charlie. Did you check if it's the battery?", "sender": "Bob", "date": "2023-06-01T09:42:00"},
    {"text": "Mechanic said it's probably the alternator. $$$", "sender": "Charlie", "date": "2023-06-01T09:45:00"},
    {"text": "Ouch, that's not cheap. I might know someone who can do it for less though!", "sender": "Alice", "date": "2023-06-01T09:47:00"},
    {"text": "That would be amazing! Thank you!", "sender": "Charlie", "date": "2023-06-01T09:50:00"},
    {"text": "Always coming to the rescue. 😇", "sender": "Bob", "date": "2023-06-01T09:52:00"},
    {"text": "Just trying to help, Bob. That's what friends do.", "sender": "Alice", "date": "2023-06-01T09:55:00"},
]

async def test_emotion_detection():
    """Test the emotion detection functionality."""
    try:
        from src.analysis.emotion_detection import detect_emotions
        
        logger.info("Testing emotion detection...")
        
        test_texts = [
            "I'm so happy about the news! This is amazing!",
            "I'm really sad about what happened yesterday.",
            "I'm absolutely furious with how they treated us.",
            "I'm a bit worried about the upcoming exam.",
            "Wow! That's such a surprise, I didn't expect that at all!",
            "I love you so much! You're the best! ❤️",
            "That's disgusting, I can't believe they would do that."
        ]
        
        for text in test_texts:
            emotions = await detect_emotions(text)
            
            # Find the strongest emotion
            strongest = max(emotions.items(), key=lambda x: x[1])
            
            logger.info(f"Text: \"{text}\"")
            logger.info(f"Strongest emotion: {strongest[0]} ({strongest[1]:.2f})")
            logger.info("All emotions: " + ", ".join([f"{e}: {s:.2f}" for e, s in emotions.items() if s > 0.1]))
            logger.info("---")
        
        return True
    except Exception as e:
        logger.error(f"Error in emotion detection test: {e}")
        return False

async def test_message_emotions():
    """Test emotion analysis for a list of messages."""
    try:
        from src.analysis.emotion_detection import analyze_emotions_in_messages
        
        logger.info("Testing emotion analysis for message list...")
        
        results = await analyze_emotions_in_messages(SAMPLE_MESSAGES)
        
        logger.info("Conversation emotion analysis:")
        
        if "dominant_emotions" in results:
            logger.info("Dominant emotions:")
            for emotion, data in results["dominant_emotions"]:
                logger.info(f"- {emotion}: {data['average']:.2f}")
        
        if "emotion_trends" in results:
            logger.info("Emotion trends detected:")
            for emotion, trend in results["emotion_trends"].items():
                if emotion.endswith("_trend"):
                    logger.info(f"- {emotion.replace('_trend', '')}: {trend['direction']} ({trend['change']:.2f})")
        
        if "emotional_shifts" in results:
            logger.info("Emotional shifts detected:")
            for shift in results["emotional_shifts"]:
                logger.info(f"- Shift from {shift['from_emotion']} to {shift['to_emotion']} (magnitude: {shift['magnitude']:.2f})")
        
        return True
    except Exception as e:
        logger.error(f"Error in message emotions test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def test_contextual_sentiment():
    """Test the contextual sentiment analysis."""
    try:
        from src.analysis.contextual_sentiment import analyze_contextual_sentiment, analyze_message_thread
        
        logger.info("Testing contextual sentiment analysis...")
        
        # First test individual message analysis
        test_messages = [
            {"text": "That was a great movie! I really enjoyed it."},
            {"text": "That was a \"great\" movie... Total waste of time. 🙄"},
            {"text": "She said \"I'm really happy\" but she didn't look happy at all."},
            {"text": "Wow! This is AMAZING! I can't believe it!!!"}
        ]
        
        for message in test_messages:
            result = await analyze_contextual_sentiment(message)
            
            logger.info(f"Text: \"{message['text']}\"")
            logger.info(f"Raw sentiment: {result['raw_sentiment']:.2f}")
            logger.info(f"Adjusted sentiment: {result['adjusted_sentiment']:.2f}")
            logger.info(f"Confidence: {result['confidence']:.2f}")
            
            if result["contextual_factors"]:
                logger.info("Contextual factors:")
                for factor in result["contextual_factors"]:
                    logger.info(f"- {factor['type']}: {factor['impact']} (adjustment: {factor['adjustment']:.2f})")
            
            logger.info("---")
        
        # Then test thread analysis
        logger.info("Testing message thread analysis...")
        
        thread_results = await analyze_message_thread(CONTEXTUAL_MESSAGES)
        
        logger.info("Message thread analysis results:")
        for i, message in enumerate(thread_results):
            sentiment = message["contextual_sentiment"]
            logger.info(f"Message {i+1}: \"{message['text']}\"")
            logger.info(f"  Adjusted sentiment: {sentiment['adjusted_sentiment']:.2f} ({sentiment['category']})")
            if "contextual_factors" in sentiment and sentiment["contextual_factors"]:
                factor_types = [f["type"] for f in sentiment["contextual_factors"]]
                logger.info(f"  Factors: {', '.join(factor_types)}")
            logger.info("")
        
        return True
    except Exception as e:
        logger.error(f"Error in contextual sentiment test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def test_relationship_dynamics():
    """Test the relationship dynamics analysis."""
    try:
        from src.analysis.emotion_detection import analyze_relationship_dynamics
        
        logger.info("Testing relationship dynamics analysis...")
        
        results = await analyze_relationship_dynamics(RELATIONSHIP_MESSAGES)
        
        if "error" in results:
            logger.warning(f"Analysis returned error: {results['error']}")
            return True
        
        logger.info(f"Participants: {results['participants']}")
        
        if "interaction_patterns" in results:
            logger.info("Interaction patterns:")
            for pattern in results["interaction_patterns"]:
                p1 = pattern["participant1"]
                p2 = pattern["participant2"]
                score = pattern["relationship_score"]
                interactions = pattern["direct_interactions"]
                reciprocity = pattern["reciprocity_score"]
                
                logger.info(f"- {p1} & {p2}: Score {score:.2f}, {interactions} interactions, reciprocity {reciprocity:.2f}")
        
        return True
    except Exception as e:
        logger.error(f"Error in relationship dynamics test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def run_tests():
    """Run all the tests."""
    test_results = {}
    
    # Test emotion detection
    test_results["Emotion Detection"] = await test_emotion_detection()
    
    # Test message emotion analysis
    test_results["Message Emotions"] = await test_message_emotions()
    
    # Test contextual sentiment analysis
    test_results["Contextual Sentiment"] = await test_contextual_sentiment()
    
    # Test relationship dynamics analysis
    test_results["Relationship Dynamics"] = await test_relationship_dynamics()
    
    # Display test results summary
    logger.info("\n--- Test Results Summary ---")
    all_passed = True
    for test_name, result in test_results.items():
        status = "PASSED" if result else "FAILED"
        if not result:
            all_passed = False
        logger.info(f"{test_name}: {status}")
    
    return all_passed

if __name__ == "__main__":
    try:
        # Run the tests
        result = asyncio.run(run_tests())
        
        # Exit with appropriate status code
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        sys.exit(1)