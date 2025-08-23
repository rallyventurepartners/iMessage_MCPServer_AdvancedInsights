"""
Communication Style Analysis tools for the iMessage Advanced Insights server.

This module provides tools for deep analysis of communication styles, including
self-analysis, comparative analysis, and style effectiveness insights.
"""

import logging
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter
import statistics
import re
import string

from ..database import get_database
from ..exceptions import DatabaseError, ToolExecutionError
from ..utils.responses import error_response, success_response
from ..utils.sanitization import sanitize_contact_info
from .registry import register_tool

logger = logging.getLogger(__name__)


@register_tool(
    name="analyze_self_communication_style",
    description="Analyze your overall communication style across all contacts"
)
async def analyze_self_communication_style_tool(
    time_period: str = "6 months",
    analysis_depth: str = "comprehensive",
    include_evolution: bool = True,
    top_contacts_limit: int = 20
) -> Dict[str, Any]:
    """
    Analyze your overall communication style across all relationships.
    
    This tool provides:
    - Overall communication profile
    - Writing style analysis (grammar, complexity, patterns)
    - Emotional expression patterns
    - Communication adaptability
    - Style evolution over time
    - Strengths and growth areas
    
    Args:
        time_period: Period to analyze (e.g., "6 months", "1 year")
        analysis_depth: Level of analysis ("basic", "moderate", "comprehensive")
        include_evolution: Whether to track style changes over time
        top_contacts_limit: Number of top contacts to analyze
        
    Returns:
        Comprehensive self-communication analysis
    """
    try:
        # Get database connection
        db = await get_database()
        
        # Parse time period
        from ..utils.decorators import parse_date
        start_date = parse_date(time_period)
        end_date = datetime.now()
        
        # Get top contacts for analysis
        contacts_result = await db.get_contacts(limit=top_contacts_limit, offset=0)
        contacts = contacts_result.get("contacts", [])
        
        # Initialize analysis structures
        self_profile = {
            "writing_style": {},
            "emotional_patterns": {},
            "communication_habits": {},
            "adaptability": {},
            "strengths": [],
            "growth_areas": []
        }
        
        all_messages = []
        contact_styles = {}
        
        # Analyze messages with each contact
        for contact in contacts:
            contact_id = contact.get("phone_number", contact.get("handle_id"))
            
            # Get messages
            messages_result = await db.get_messages_from_contact(
                phone_number=contact_id,
                start_date=start_date,
                end_date=end_date,
                page=1, page_size=1000
            )
            
            messages = messages_result.get("messages", [])
            
            if len(messages) < 20:  # Skip contacts with too few messages
                continue
            
            # Separate your messages
            your_messages = [msg for msg in messages if msg.get("is_from_me")]
            their_messages = [msg for msg in messages if not msg.get("is_from_me")]
            
            if your_messages:
                all_messages.extend(your_messages)
                
                # Analyze style with this contact
                contact_style = _analyze_messages_style(your_messages, their_messages)
                contact_styles[contact_id] = {
                    "contact_name": contact.get("name", contact_id[:10]),
                    "style": contact_style,
                    "message_count": len(your_messages)
                }
        
        if not all_messages:
            return success_response({
                "status": "insufficient_data",
                "message": "Not enough messages for self-analysis"
            })
        
        # Analyze overall writing style
        self_profile["writing_style"] = _analyze_writing_style(all_messages, analysis_depth)
        
        # Analyze emotional patterns
        self_profile["emotional_patterns"] = _analyze_emotional_patterns(all_messages)
        
        # Analyze communication habits
        self_profile["communication_habits"] = _analyze_communication_habits(all_messages)
        
        # Analyze adaptability across contacts
        if len(contact_styles) > 3:
            self_profile["adaptability"] = _analyze_style_adaptability(contact_styles)
        
        # Track evolution if requested
        evolution = None
        if include_evolution:
            evolution = _analyze_style_evolution(all_messages, start_date, end_date)
        
        # Identify strengths and growth areas
        insights = _generate_self_insights(self_profile, contact_styles)
        self_profile["strengths"] = insights["strengths"]
        self_profile["growth_areas"] = insights["growth_areas"]
        
        # Generate communication personality
        personality = _determine_communication_personality(self_profile)
        
        # Create final response
        return success_response({
            "self_profile": self_profile,
            "communication_personality": personality,
            "style_variations": _summarize_style_variations(contact_styles),
            "evolution": evolution,
            "statistics": {
                "total_messages_analyzed": len(all_messages),
                "contacts_analyzed": len(contact_styles),
                "time_period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                }
            },
            "recommendations": _generate_self_recommendations(self_profile, insights)
        })
        
    except DatabaseError as e:
        logger.error(f"Database error in analyze_self_communication_style_tool: {e}")
        return error_response(e)
    except Exception as e:
        logger.error(f"Unexpected error in analyze_self_communication_style_tool: {e}")
        logger.error(traceback.format_exc())
        return error_response(ToolExecutionError(f"Failed to analyze self style: {str(e)}"))


def _analyze_writing_style(messages: List[Dict], depth: str) -> Dict[str, Any]:
    """Analyze writing style characteristics."""
    
    # Basic metrics
    message_lengths = []
    word_counts = []
    sentence_counts = []
    punctuation_usage = defaultdict(int)
    capitalization_patterns = {"proper": 0, "all_lower": 0, "all_upper": 0, "mixed": 0}
    
    for msg in messages:
        text = msg.get("text", "")
        if not text:
            continue
        
        message_lengths.append(len(text))
        words = text.split()
        word_counts.append(len(words))
        
        # Count sentences (approximate)
        sentences = len(re.split(r'[.!?]+', text))
        sentence_counts.append(sentences)
        
        # Analyze punctuation
        for char in text:
            if char in string.punctuation:
                punctuation_usage[char] += 1
        
        # Analyze capitalization
        if text.isupper():
            capitalization_patterns["all_upper"] += 1
        elif text.islower():
            capitalization_patterns["all_lower"] += 1
        elif text[0].isupper() if text else False:
            capitalization_patterns["proper"] += 1
        else:
            capitalization_patterns["mixed"] += 1
    
    # Calculate style metrics
    avg_message_length = statistics.mean(message_lengths) if message_lengths else 0
    avg_word_count = statistics.mean(word_counts) if word_counts else 0
    avg_words_per_sentence = avg_word_count / max(statistics.mean(sentence_counts), 1) if sentence_counts else 0
    
    style = {
        "complexity": {
            "average_message_length": round(avg_message_length, 1),
            "average_word_count": round(avg_word_count, 1),
            "words_per_sentence": round(avg_words_per_sentence, 1),
            "complexity_score": _calculate_complexity_score(avg_message_length, avg_words_per_sentence)
        },
        "punctuation": {
            "usage_frequency": sum(punctuation_usage.values()) / len(messages) if messages else 0,
            "most_used": sorted(punctuation_usage.items(), key=lambda x: x[1], reverse=True)[:5],
            "question_frequency": punctuation_usage.get("?", 0) / len(messages) if messages else 0,
            "exclamation_frequency": punctuation_usage.get("!", 0) / len(messages) if messages else 0
        },
        "capitalization": {
            "primary_style": max(capitalization_patterns.items(), key=lambda x: x[1])[0],
            "distribution": capitalization_patterns
        }
    }
    
    if depth in ["moderate", "comprehensive"]:
        # Add vocabulary analysis
        all_words = []
        for msg in messages:
            if msg.get("text"):
                words = msg["text"].lower().split()
                all_words.extend([w.strip(string.punctuation) for w in words if w.strip(string.punctuation)])
        
        unique_words = len(set(all_words))
        vocabulary_richness = unique_words / len(all_words) if all_words else 0
        
        style["vocabulary"] = {
            "unique_words": unique_words,
            "total_words": len(all_words),
            "richness_score": round(vocabulary_richness, 3),
            "most_common_words": [word for word, _ in Counter(all_words).most_common(10) if len(word) > 3]
        }
    
    if depth == "comprehensive":
        # Add linguistic patterns
        style["linguistic_patterns"] = _analyze_linguistic_patterns(messages)
    
    return style


def _calculate_complexity_score(avg_length: float, words_per_sentence: float) -> str:
    """Calculate writing complexity score."""
    score = (avg_length / 100) + (words_per_sentence / 20)
    
    if score < 0.5:
        return "simple"
    elif score < 1.0:
        return "moderate"
    elif score < 1.5:
        return "complex"
    else:
        return "very_complex"


def _analyze_linguistic_patterns(messages: List[Dict]) -> Dict[str, Any]:
    """Analyze linguistic patterns in messages."""
    
    patterns = {
        "sentence_starters": defaultdict(int),
        "common_phrases": defaultdict(int),
        "filler_words": defaultdict(int)
    }
    
    filler_words = {"like", "just", "really", "actually", "basically", "literally", "you know", "i mean"}
    
    for msg in messages:
        text = msg.get("text", "").lower()
        if not text:
            continue
        
        # Analyze sentence starters
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            words = sentence.strip().split()
            if words:
                patterns["sentence_starters"][words[0]] += 1
        
        # Find common phrases (2-3 word combinations)
        words = text.split()
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}"
            patterns["common_phrases"][phrase] += 1
        
        # Count filler words
        for filler in filler_words:
            count = text.count(filler)
            if count > 0:
                patterns["filler_words"][filler] += count
    
    return {
        "common_sentence_starters": sorted(patterns["sentence_starters"].items(), key=lambda x: x[1], reverse=True)[:5],
        "frequent_phrases": sorted(patterns["common_phrases"].items(), key=lambda x: x[1], reverse=True)[:10],
        "filler_word_usage": dict(patterns["filler_words"]),
        "filler_frequency": sum(patterns["filler_words"].values()) / len(messages) if messages else 0
    }


def _analyze_emotional_patterns(messages: List[Dict]) -> Dict[str, Any]:
    """Analyze emotional expression patterns."""
    
    emotion_words = {
        "joy": ["happy", "joy", "excited", "love", "wonderful", "amazing", "great", "awesome"],
        "sadness": ["sad", "depressed", "down", "unhappy", "crying", "tears", "miss"],
        "anger": ["angry", "mad", "furious", "annoyed", "frustrated", "pissed"],
        "fear": ["scared", "afraid", "worried", "anxious", "nervous", "terrified"],
        "surprise": ["surprised", "shocked", "amazed", "astonished", "wow", "omg"],
        "affection": ["love", "care", "adore", "fond", "dear", "sweetheart"]
    }
    
    emotion_counts = defaultdict(int)
    emoji_usage = defaultdict(int)
    emotional_intensity = []
    
    # Common emojis regex
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", 
        flags=re.UNICODE
    )
    
    for msg in messages:
        text = msg.get("text", "").lower()
        if not text:
            continue
        
        # Count emotion words
        intensity = 0
        for emotion, words in emotion_words.items():
            for word in words:
                if word in text:
                    emotion_counts[emotion] += 1
                    intensity += 1
        
        emotional_intensity.append(intensity)
        
        # Count emojis
        emojis = emoji_pattern.findall(msg.get("text", ""))
        for emoji in emojis:
            emoji_usage[emoji] += 1
    
    total_messages = len(messages)
    
    return {
        "emotion_distribution": dict(emotion_counts),
        "emotional_expression_rate": sum(emotion_counts.values()) / total_messages if total_messages else 0,
        "average_emotional_intensity": statistics.mean(emotional_intensity) if emotional_intensity else 0,
        "dominant_emotion": max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "neutral",
        "emoji_usage": {
            "frequency": sum(emoji_usage.values()) / total_messages if total_messages else 0,
            "variety": len(emoji_usage),
            "top_emojis": sorted(emoji_usage.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    }


def _analyze_communication_habits(messages: List[Dict]) -> Dict[str, Any]:
    """Analyze communication habits and patterns."""
    
    # Time-based analysis
    hourly_distribution = defaultdict(int)
    daily_distribution = defaultdict(int)
    response_times = []
    message_bursts = []
    
    for i, msg in enumerate(messages):
        if msg.get("date"):
            msg_time = datetime.fromisoformat(msg["date"])
            hourly_distribution[msg_time.hour] += 1
            daily_distribution[msg_time.weekday()] += 1
            
            # Detect message bursts
            if i > 0:
                prev_time = datetime.fromisoformat(messages[i-1]["date"])
                time_diff = (msg_time - prev_time).total_seconds()
                
                if time_diff < 300:  # Within 5 minutes
                    if not message_bursts or time_diff > 300:
                        message_bursts.append(1)
                    else:
                        message_bursts[-1] += 1
    
    # Find peak communication times
    peak_hours = sorted(hourly_distribution.items(), key=lambda x: x[1], reverse=True)[:3]
    peak_days = sorted(daily_distribution.items(), key=lambda x: x[1], reverse=True)[:3]
    
    return {
        "temporal_patterns": {
            "peak_hours": [{"hour": h, "count": c} for h, c in peak_hours],
            "peak_days": [{"day": _day_name(d), "count": c} for d, c in peak_days],
            "night_owl_score": sum(hourly_distribution[h] for h in range(0, 6)) / sum(hourly_distribution.values()) if hourly_distribution else 0,
            "early_bird_score": sum(hourly_distribution[h] for h in range(5, 9)) / sum(hourly_distribution.values()) if hourly_distribution else 0
        },
        "messaging_behavior": {
            "average_burst_length": statistics.mean(message_bursts) if message_bursts else 1,
            "burst_frequency": len(message_bursts) / len(messages) if messages else 0,
            "messages_per_day": len(messages) / max((messages[-1]["date"][:10] != messages[0]["date"][:10]), 1) if messages else 0
        }
    }


def _day_name(day: int) -> str:
    """Convert day number to name."""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return days[day]


def _analyze_messages_style(your_messages: List[Dict], their_messages: List[Dict]) -> Dict[str, Any]:
    """Analyze communication style for messages with a specific contact."""
    
    # Calculate formality score
    formal_indicators = ["please", "thank you", "regards", "sincerely", "appreciate"]
    informal_indicators = ["lol", "haha", "gonna", "wanna", "yeah", "nah", "sup"]
    
    formal_count = 0
    informal_count = 0
    
    for msg in your_messages:
        text = msg.get("text", "").lower()
        for indicator in formal_indicators:
            if indicator in text:
                formal_count += 1
        for indicator in informal_indicators:
            if indicator in text:
                informal_count += 1
    
    formality_score = (formal_count - informal_count) / max(len(your_messages), 1)
    
    # Calculate mirroring score (how much you adapt to their style)
    your_avg_length = statistics.mean([len(m.get("text", "")) for m in your_messages if m.get("text")])
    their_avg_length = statistics.mean([len(m.get("text", "")) for m in their_messages if m.get("text")]) if their_messages else your_avg_length
    
    length_difference = abs(your_avg_length - their_avg_length) / max(your_avg_length, their_avg_length)
    mirroring_score = 1 - length_difference
    
    return {
        "formality_score": round(formality_score, 3),
        "mirroring_score": round(mirroring_score, 3),
        "average_message_length": round(your_avg_length, 1),
        "initiation_rate": sum(1 for i, m in enumerate(your_messages) if i == 0 or your_messages[i-1].get("is_from_me") != m.get("is_from_me")) / len(your_messages) if your_messages else 0
    }


def _analyze_style_adaptability(contact_styles: Dict[str, Dict]) -> Dict[str, Any]:
    """Analyze how communication style adapts across different contacts."""
    
    # Extract style metrics for each contact
    formality_scores = []
    mirroring_scores = []
    message_lengths = []
    
    for contact_id, data in contact_styles.items():
        style = data["style"]
        formality_scores.append(style["formality_score"])
        mirroring_scores.append(style["mirroring_score"])
        message_lengths.append(style["average_message_length"])
    
    # Calculate variability
    formality_variance = statistics.stdev(formality_scores) if len(formality_scores) > 1 else 0
    length_variance = statistics.stdev(message_lengths) if len(message_lengths) > 1 else 0
    avg_mirroring = statistics.mean(mirroring_scores) if mirroring_scores else 0
    
    # Determine adaptability profile
    if formality_variance > 0.2:
        adaptability_profile = "highly_adaptive"
        description = "You significantly adjust your communication style based on the relationship"
    elif formality_variance > 0.1:
        adaptability_profile = "moderately_adaptive"
        description = "You make moderate adjustments to your style for different people"
    else:
        adaptability_profile = "consistent"
        description = "You maintain a consistent communication style across relationships"
    
    return {
        "adaptability_score": round(formality_variance + (avg_mirroring * 0.5), 3),
        "adaptability_profile": adaptability_profile,
        "description": description,
        "metrics": {
            "formality_variance": round(formality_variance, 3),
            "average_mirroring": round(avg_mirroring, 3),
            "message_length_variance": round(length_variance, 1)
        },
        "style_range": {
            "most_formal": max(contact_styles.items(), key=lambda x: x[1]["style"]["formality_score"])[1]["contact_name"],
            "most_casual": min(contact_styles.items(), key=lambda x: x[1]["style"]["formality_score"])[1]["contact_name"],
            "best_mirroring": max(contact_styles.items(), key=lambda x: x[1]["style"]["mirroring_score"])[1]["contact_name"]
        }
    }


def _analyze_style_evolution(messages: List[Dict], start_date: datetime, end_date: datetime) -> Dict[str, Any]:
    """Analyze how communication style has evolved over time."""
    
    # Divide time period into quarters
    total_days = (end_date - start_date).days
    quarter_days = total_days // 4
    
    evolution_data = []
    
    for quarter in range(4):
        quarter_start = start_date + timedelta(days=quarter * quarter_days)
        quarter_end = quarter_start + timedelta(days=quarter_days)
        
        # Get messages for this quarter
        quarter_messages = [
            msg for msg in messages 
            if quarter_start <= datetime.fromisoformat(msg["date"]) < quarter_end
        ]
        
        if quarter_messages:
            # Analyze style for this period
            style = _analyze_writing_style(quarter_messages, "basic")
            emotional = _analyze_emotional_patterns(quarter_messages)
            
            evolution_data.append({
                "period": f"Q{quarter + 1}",
                "message_count": len(quarter_messages),
                "avg_length": style["complexity"]["average_message_length"],
                "emotional_expression": emotional["emotional_expression_rate"],
                "emoji_usage": emotional["emoji_usage"]["frequency"]
            })
    
    # Analyze trends
    if len(evolution_data) >= 2:
        length_trend = "increasing" if evolution_data[-1]["avg_length"] > evolution_data[0]["avg_length"] else "decreasing"
        emotion_trend = "increasing" if evolution_data[-1]["emotional_expression"] > evolution_data[0]["emotional_expression"] else "decreasing"
        
        return {
            "quarterly_evolution": evolution_data,
            "trends": {
                "message_length": length_trend,
                "emotional_expression": emotion_trend,
                "overall_direction": "evolving" if abs(evolution_data[-1]["avg_length"] - evolution_data[0]["avg_length"]) > 10 else "stable"
            }
        }
    
    return {"status": "insufficient_data_for_evolution"}


def _generate_self_insights(profile: Dict[str, Any], contact_styles: Dict[str, Dict]) -> Dict[str, Any]:
    """Generate insights about communication strengths and growth areas."""
    
    strengths = []
    growth_areas = []
    
    # Analyze writing style
    complexity = profile["writing_style"]["complexity"]["complexity_score"]
    if complexity in ["moderate", "complex"]:
        strengths.append("Articulate and detailed communication")
    elif complexity == "simple":
        strengths.append("Clear and concise messaging")
    
    # Analyze emotional expression
    emotional_rate = profile["emotional_patterns"]["emotional_expression_rate"]
    if emotional_rate > 0.2:
        strengths.append("Emotionally expressive and open")
    elif emotional_rate < 0.05:
        growth_areas.append("Consider expressing emotions more openly")
    
    # Analyze adaptability
    if profile.get("adaptability", {}).get("adaptability_profile") == "highly_adaptive":
        strengths.append("Excellent at adapting communication style to different relationships")
    elif profile.get("adaptability", {}).get("adaptability_profile") == "consistent":
        growth_areas.append("Consider adapting your style more to different relationships")
    
    # Analyze vocabulary
    if "vocabulary" in profile["writing_style"]:
        if profile["writing_style"]["vocabulary"]["richness_score"] > 0.3:
            strengths.append("Rich and diverse vocabulary")
        elif profile["writing_style"]["vocabulary"]["richness_score"] < 0.15:
            growth_areas.append("Expand vocabulary for more engaging conversations")
    
    # Analyze question usage
    question_freq = profile["writing_style"]["punctuation"]["question_frequency"]
    if question_freq > 0.15:
        strengths.append("Inquisitive and engaging conversationalist")
    elif question_freq < 0.05:
        growth_areas.append("Ask more questions to deepen conversations")
    
    return {
        "strengths": strengths,
        "growth_areas": growth_areas
    }


def _determine_communication_personality(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Determine overall communication personality type."""
    
    # Analyze key dimensions
    complexity = profile["writing_style"]["complexity"]["complexity_score"]
    emotional_expression = profile["emotional_patterns"]["emotional_expression_rate"]
    emoji_usage = profile["emotional_patterns"]["emoji_usage"]["frequency"]
    adaptability = profile.get("adaptability", {}).get("adaptability_score", 0)
    
    # Determine personality type
    if complexity in ["complex", "very_complex"] and emotional_expression > 0.15:
        personality_type = "The Thoughtful Expresser"
        description = "You communicate with depth and emotional intelligence, creating meaningful connections through detailed, heartfelt messages."
    elif complexity == "simple" and emotional_expression < 0.1:
        personality_type = "The Efficient Communicator"
        description = "You value clarity and brevity, getting straight to the point while maintaining practical communication."
    elif adaptability > 0.3:
        personality_type = "The Social Chameleon"
        description = "You expertly adapt your communication style to match different relationships and contexts."
    elif emoji_usage > 0.5:
        personality_type = "The Visual Communicator"
        description = "You enhance your messages with visual elements, creating engaging and expressive conversations."
    elif profile["communication_habits"]["temporal_patterns"]["night_owl_score"] > 0.3:
        personality_type = "The Night Owl Connector"
        description = "You thrive in late-night conversations, often having your deepest exchanges after dark."
    else:
        personality_type = "The Balanced Communicator"
        description = "You maintain a well-rounded communication style that works across various contexts and relationships."
    
    return {
        "type": personality_type,
        "description": description,
        "key_traits": _extract_key_traits(profile)
    }


def _extract_key_traits(profile: Dict[str, Any]) -> List[str]:
    """Extract key communication traits from profile."""
    traits = []
    
    # Complexity trait
    complexity = profile["writing_style"]["complexity"]["complexity_score"]
    if complexity in ["complex", "very_complex"]:
        traits.append("Detail-oriented")
    else:
        traits.append("Concise")
    
    # Emotional trait
    if profile["emotional_patterns"]["emotional_expression_rate"] > 0.2:
        traits.append("Emotionally open")
    
    # Timing trait
    if profile["communication_habits"]["temporal_patterns"]["night_owl_score"] > 0.3:
        traits.append("Night communicator")
    elif profile["communication_habits"]["temporal_patterns"]["early_bird_score"] > 0.3:
        traits.append("Morning person")
    
    # Style trait
    if profile.get("adaptability", {}).get("adaptability_profile") == "highly_adaptive":
        traits.append("Adaptable")
    else:
        traits.append("Consistent")
    
    return traits


def _summarize_style_variations(contact_styles: Dict[str, Dict]) -> Dict[str, Any]:
    """Summarize how communication style varies across contacts."""
    
    if not contact_styles:
        return {"status": "no_data"}
    
    # Group contacts by style similarity
    formal_contacts = []
    casual_contacts = []
    
    for contact_id, data in contact_styles.items():
        if data["style"]["formality_score"] > 0.1:
            formal_contacts.append(data["contact_name"])
        elif data["style"]["formality_score"] < -0.1:
            casual_contacts.append(data["contact_name"])
    
    return {
        "formal_relationships": formal_contacts[:5],
        "casual_relationships": casual_contacts[:5],
        "style_distribution": {
            "formal": len(formal_contacts),
            "neutral": len(contact_styles) - len(formal_contacts) - len(casual_contacts),
            "casual": len(casual_contacts)
        }
    }


def _generate_self_recommendations(profile: Dict[str, Any], insights: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate personalized recommendations for communication improvement."""
    
    recommendations = []
    
    # Based on growth areas
    for area in insights.get("growth_areas", []):
        if "vocabulary" in area:
            recommendations.append({
                "action": "Read more diverse content to expand vocabulary",
                "reason": "Richer vocabulary leads to more engaging conversations",
                "priority": "medium",
                "category": "skill_development"
            })
        elif "emotions" in area:
            recommendations.append({
                "action": "Practice expressing feelings more openly",
                "reason": "Emotional openness deepens relationships",
                "priority": "high",
                "category": "emotional_intelligence"
            })
        elif "questions" in area:
            recommendations.append({
                "action": "Ask open-ended questions in conversations",
                "reason": "Questions show interest and deepen understanding",
                "priority": "high",
                "category": "engagement"
            })
    
    # Based on profile analysis
    if profile["writing_style"]["complexity"]["complexity_score"] == "very_complex":
        recommendations.append({
            "action": "Practice simplifying complex thoughts",
            "reason": "Clarity enhances understanding in quick exchanges",
            "priority": "low",
            "category": "clarity"
        })
    
    # Based on habits
    if profile["communication_habits"]["messaging_behavior"]["burst_frequency"] > 0.5:
        recommendations.append({
            "action": "Practice more thoughtful, consolidated messages",
            "reason": "Reduces message overload for recipients",
            "priority": "low",
            "category": "habits"
        })
    
    return recommendations[:5]  # Top 5 recommendations


@register_tool(
    name="compare_communication_styles_across_contacts",
    description="Compare how your communication style differs across relationships"
)
async def compare_communication_styles_tool(
    contact_ids: Optional[List[str]] = None,
    analysis_aspects: Optional[List[str]] = None,
    time_period: str = "3 months",
    include_effectiveness: bool = True
) -> Dict[str, Any]:
    """
    Compare how your communication style varies across different relationships.
    
    This tool analyzes:
    - Style adaptations for different people
    - Consistency vs variation patterns
    - Relationship-specific communication traits
    - Effectiveness of different styles
    
    Args:
        contact_ids: Specific contacts to compare (None = top contacts)
        analysis_aspects: Aspects to compare (default: all)
        time_period: Period to analyze
        include_effectiveness: Correlate style with relationship health
        
    Returns:
        Comparative analysis of communication styles
    """
    try:
        # Get database connection
        db = await get_database()
        
        # Parse time period
        from ..utils.decorators import parse_date
        start_date = parse_date(time_period)
        end_date = datetime.now()
        
        # Default analysis aspects
        if analysis_aspects is None:
            analysis_aspects = ["formality", "length", "emotion", "timing", "topics"]
        
        # Get contacts to analyze
        if contact_ids:
            contacts_to_analyze = contact_ids[:10]  # Limit to 10
        else:
            # Get top contacts
            contacts_result = await db.get_contacts(limit=10, offset=0)
            contacts_to_analyze = [c["id"] for c in contacts_result.get("contacts", [])]
        
        # Analyze style with each contact
        style_comparisons = {}
        
        for contact_id in contacts_to_analyze:
            # Get contact info
            contact_info = await db.get_contact_info(contact_id=contact_id)
            
            # Get messages
            messages_result = await db.get_messages_from_contact(
                phone_number=contact_id,
                start_date=start_date,
                end_date=end_date,
                page=1, page_size=500
            )
            
            messages = messages_result.get("messages", [])
            
            if len(messages) < 20:
                continue
            
            # Separate messages
            your_messages = [msg for msg in messages if msg.get("is_from_me")]
            their_messages = [msg for msg in messages if not msg.get("is_from_me")]
            
            if not your_messages:
                continue
            
            # Analyze style dimensions
            style_analysis = {}
            
            if "formality" in analysis_aspects:
                style_analysis["formality"] = _analyze_formality_level(your_messages)
            
            if "length" in analysis_aspects:
                style_analysis["message_patterns"] = _analyze_message_patterns(your_messages)
            
            if "emotion" in analysis_aspects:
                style_analysis["emotional_expression"] = _analyze_emotional_expression(your_messages)
            
            if "timing" in analysis_aspects:
                style_analysis["response_patterns"] = _analyze_response_timing(messages)
            
            if "topics" in analysis_aspects:
                style_analysis["topic_preferences"] = _analyze_topic_preferences(your_messages)
            
            # Add effectiveness metrics if requested
            effectiveness = None
            if include_effectiveness:
                effectiveness = await _analyze_style_effectiveness(db, contact_id, messages)
            
            style_comparisons[contact_id] = {
                "contact_info": sanitize_contact_info(contact_info),
                "style_analysis": style_analysis,
                "effectiveness": effectiveness,
                "message_count": len(your_messages)
            }
        
        if not style_comparisons:
            return success_response({
                "status": "insufficient_data",
                "message": "Not enough data for comparison"
            })
        
        # Generate comparative insights
        insights = _generate_comparative_insights(style_comparisons)
        
        # Identify patterns
        patterns = _identify_style_patterns(style_comparisons)
        
        # Generate recommendations
        recommendations = _generate_style_recommendations(insights, patterns)
        
        return success_response({
            "individual_styles": style_comparisons,
            "comparative_insights": insights,
            "patterns": patterns,
            "recommendations": recommendations,
            "metadata": {
                "contacts_analyzed": len(style_comparisons),
                "time_period": time_period,
                "aspects_analyzed": analysis_aspects
            }
        })
        
    except Exception as e:
        logger.error(f"Error in compare_communication_styles_tool: {e}")
        logger.error(traceback.format_exc())
        return error_response(ToolExecutionError(f"Failed to compare styles: {str(e)}"))


def _analyze_formality_level(messages: List[Dict]) -> Dict[str, Any]:
    """Analyze formality level in messages."""
    
    formal_indicators = {
        "greetings": ["hello", "hi", "hey", "good morning", "good evening"],
        "politeness": ["please", "thank you", "thanks", "appreciate", "kindly"],
        "closings": ["regards", "sincerely", "best", "cheers", "take care"]
    }
    
    informal_indicators = {
        "slang": ["lol", "omg", "wtf", "lmao", "rofl", "btw", "fyi"],
        "contractions": ["gonna", "wanna", "gotta", "ain't", "y'all"],
        "casual": ["yeah", "nah", "yep", "nope", "sup", "yo"]
    }
    
    formal_score = 0
    informal_score = 0
    
    for msg in messages:
        text = msg.get("text", "").lower()
        
        # Check formal indicators
        for category, words in formal_indicators.items():
            for word in words:
                if word in text:
                    formal_score += 1
        
        # Check informal indicators
        for category, words in informal_indicators.items():
            for word in words:
                if word in text:
                    informal_score += 1
    
    total_messages = len(messages)
    formality_ratio = (formal_score - informal_score) / max(total_messages, 1)
    
    return {
        "formality_score": round(formality_ratio, 3),
        "formality_level": _categorize_formality(formality_ratio),
        "formal_elements": formal_score,
        "informal_elements": informal_score
    }


def _categorize_formality(score: float) -> str:
    """Categorize formality level."""
    if score > 0.3:
        return "very_formal"
    elif score > 0.1:
        return "formal"
    elif score > -0.1:
        return "neutral"
    elif score > -0.3:
        return "casual"
    else:
        return "very_casual"


def _analyze_message_patterns(messages: List[Dict]) -> Dict[str, Any]:
    """Analyze message length and structure patterns."""
    
    lengths = [len(msg.get("text", "")) for msg in messages if msg.get("text")]
    
    if not lengths:
        return {}
    
    return {
        "average_length": round(statistics.mean(lengths), 1),
        "length_variance": round(statistics.stdev(lengths), 1) if len(lengths) > 1 else 0,
        "short_messages": sum(1 for l in lengths if l < 50),
        "medium_messages": sum(1 for l in lengths if 50 <= l < 150),
        "long_messages": sum(1 for l in lengths if l >= 150),
        "consistency": "consistent" if statistics.stdev(lengths) < 50 else "variable" if statistics.stdev(lengths) < 100 else "highly_variable"
    }


def _analyze_emotional_expression(messages: List[Dict]) -> Dict[str, Any]:
    """Analyze emotional expression in messages."""
    
    emotion_words = {
        "positive": ["love", "happy", "excited", "great", "amazing", "wonderful"],
        "negative": ["sad", "angry", "frustrated", "upset", "worried", "stressed"]
    }
    
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "]+",
        flags=re.UNICODE
    )
    
    emotion_counts = defaultdict(int)
    emoji_count = 0
    
    for msg in messages:
        text = msg.get("text", "").lower()
        
        # Count emotion words
        for emotion_type, words in emotion_words.items():
            for word in words:
                if word in text:
                    emotion_counts[emotion_type] += 1
        
        # Count emojis
        emojis = emoji_pattern.findall(msg.get("text", ""))
        emoji_count += len(emojis)
    
    total_messages = len(messages)
    
    return {
        "emotional_openness": sum(emotion_counts.values()) / total_messages if total_messages else 0,
        "positivity_ratio": emotion_counts["positive"] / max(sum(emotion_counts.values()), 1),
        "emoji_usage_rate": emoji_count / total_messages if total_messages else 0,
        "expression_style": "emoji_heavy" if emoji_count > total_messages * 0.5 else "word_based" if sum(emotion_counts.values()) > total_messages * 0.2 else "reserved"
    }


def _analyze_response_timing(messages: List[Dict]) -> Dict[str, Any]:
    """Analyze response timing patterns."""
    
    response_times = []
    initiation_count = 0
    
    for i in range(1, len(messages)):
        curr_msg = messages[i]
        prev_msg = messages[i-1]
        
        # Check if this is a response
        if curr_msg.get("is_from_me") != prev_msg.get("is_from_me"):
            if curr_msg.get("is_from_me"):
                # Your response
                time_diff = (datetime.fromisoformat(curr_msg["date"]) - datetime.fromisoformat(prev_msg["date"])).total_seconds() / 60
                if 0 < time_diff < 1440:  # Within 24 hours
                    response_times.append(time_diff)
        
        # Check for conversation initiation
        if curr_msg.get("is_from_me") and (i == 0 or (datetime.fromisoformat(curr_msg["date"]) - datetime.fromisoformat(prev_msg["date"])).total_seconds() > 3600):
            initiation_count += 1
    
    return {
        "average_response_time": round(statistics.mean(response_times), 1) if response_times else 0,
        "response_consistency": statistics.stdev(response_times) if len(response_times) > 1 else 0,
        "initiation_rate": initiation_count / len(messages) if messages else 0,
        "response_style": _categorize_response_style(statistics.mean(response_times) if response_times else 0)
    }


def _categorize_response_style(avg_response_time: float) -> str:
    """Categorize response style based on timing."""
    if avg_response_time < 5:
        return "immediate"
    elif avg_response_time < 30:
        return "prompt"
    elif avg_response_time < 120:
        return "relaxed"
    else:
        return "delayed"


def _analyze_topic_preferences(messages: List[Dict]) -> Dict[str, Any]:
    """Analyze topic preferences in messages."""
    
    topic_categories = {
        "personal": ["feel", "think", "believe", "me", "my", "i"],
        "work": ["work", "job", "meeting", "project", "deadline", "office"],
        "social": ["party", "dinner", "drinks", "hangout", "plans", "weekend"],
        "interests": ["movie", "book", "music", "game", "show", "sport"]
    }
    
    topic_counts = defaultdict(int)
    
    for msg in messages:
        text = msg.get("text", "").lower()
        
        for topic, keywords in topic_categories.items():
            for keyword in keywords:
                if keyword in text:
                    topic_counts[topic] += 1
                    break
    
    total_topics = sum(topic_counts.values())
    
    return {
        "dominant_topics": sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3],
        "topic_diversity": len([t for t in topic_counts.values() if t > 0]),
        "topic_distribution": {k: v/total_topics for k, v in topic_counts.items()} if total_topics else {}
    }


async def _analyze_style_effectiveness(db: Any, contact_id: str, messages: List[Dict]) -> Dict[str, Any]:
    """Analyze effectiveness of communication style with this contact."""
    
    # Get relationship health metrics
    from .conversation_intelligence import analyze_conversation_intelligence_tool
    
    try:
        health_result = await analyze_conversation_intelligence_tool(
            contact_id=contact_id,
            analysis_depth="basic",
            include_examples=False
        )
        
        if health_result["success"] and "analysis" in health_result["data"]:
            health_score = health_result["data"]["analysis"].get("health_score", {}).get("score", 50)
            
            return {
                "relationship_health": health_score,
                "effectiveness_rating": "high" if health_score > 70 else "moderate" if health_score > 40 else "low",
                "correlation": "Your communication style works well with this person" if health_score > 70 else "Consider adjusting your approach"
            }
    except:
        pass
    
    return {
        "relationship_health": "unknown",
        "effectiveness_rating": "unknown",
        "correlation": "Unable to determine effectiveness"
    }


def _generate_comparative_insights(comparisons: Dict[str, Dict]) -> Dict[str, Any]:
    """Generate insights from style comparisons."""
    
    insights = {
        "style_consistency": {},
        "adaptation_patterns": {},
        "unique_relationships": []
    }
    
    # Analyze formality variation
    formality_scores = []
    for contact_id, data in comparisons.items():
        if "formality" in data["style_analysis"]:
            formality_scores.append(data["style_analysis"]["formality"]["formality_score"])
    
    if formality_scores:
        formality_variance = statistics.stdev(formality_scores) if len(formality_scores) > 1 else 0
        insights["style_consistency"]["formality_variance"] = round(formality_variance, 3)
        insights["style_consistency"]["consistency_level"] = "high" if formality_variance < 0.1 else "moderate" if formality_variance < 0.2 else "low"
    
    # Identify unique relationships
    for contact_id, data in comparisons.items():
        contact_name = data["contact_info"]["name"]
        
        # Check for extremes
        if "formality" in data["style_analysis"]:
            formality_level = data["style_analysis"]["formality"]["formality_level"]
            if formality_level in ["very_formal", "very_casual"]:
                insights["unique_relationships"].append({
                    "contact": contact_name,
                    "characteristic": f"Unusually {formality_level.replace('_', ' ')} communication"
                })
    
    return insights


def _identify_style_patterns(comparisons: Dict[str, Dict]) -> Dict[str, Any]:
    """Identify patterns in communication styles."""
    
    patterns = {
        "formality_groups": defaultdict(list),
        "emotional_expression_groups": defaultdict(list),
        "response_style_groups": defaultdict(list)
    }
    
    for contact_id, data in comparisons.items():
        contact_name = data["contact_info"]["name"]
        
        # Group by formality
        if "formality" in data["style_analysis"]:
            level = data["style_analysis"]["formality"]["formality_level"]
            patterns["formality_groups"][level].append(contact_name)
        
        # Group by emotional expression
        if "emotional_expression" in data["style_analysis"]:
            style = data["style_analysis"]["emotional_expression"]["expression_style"]
            patterns["emotional_expression_groups"][style].append(contact_name)
        
        # Group by response style
        if "response_patterns" in data["style_analysis"]:
            style = data["style_analysis"]["response_patterns"]["response_style"]
            patterns["response_style_groups"][style].append(contact_name)
    
    # Convert to regular dicts and limit names
    return {
        "formality_groups": {k: v[:3] for k, v in patterns["formality_groups"].items()},
        "emotional_expression_groups": {k: v[:3] for k, v in patterns["emotional_expression_groups"].items()},
        "response_style_groups": {k: v[:3] for k, v in patterns["response_style_groups"].items()}
    }


def _generate_style_recommendations(insights: Dict[str, Any], patterns: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate recommendations based on style analysis."""
    
    recommendations = []
    
    # Based on consistency
    if insights.get("style_consistency", {}).get("consistency_level") == "low":
        recommendations.append({
            "action": "Develop a more consistent communication approach",
            "reason": "High variance in style may confuse some contacts",
            "priority": "medium",
            "category": "consistency"
        })
    
    # Based on formality patterns
    formality_groups = patterns.get("formality_groups", {})
    if len(formality_groups.get("very_formal", [])) > 3:
        recommendations.append({
            "action": "Consider relaxing formality with close contacts",
            "reason": "Overly formal communication may create distance",
            "priority": "low",
            "category": "relationship_building"
        })
    
    # Based on emotional expression
    emotion_groups = patterns.get("emotional_expression_groups", {})
    if len(emotion_groups.get("reserved", [])) > len(emotion_groups.get("emoji_heavy", [])) + len(emotion_groups.get("word_based", [])):
        recommendations.append({
            "action": "Express emotions more openly in appropriate relationships",
            "reason": "Emotional expression strengthens connections",
            "priority": "medium",
            "category": "emotional_intelligence"
        })
    
    return recommendations[:4]