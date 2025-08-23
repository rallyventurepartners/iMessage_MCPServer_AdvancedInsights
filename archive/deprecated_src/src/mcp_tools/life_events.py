"""
Life Event Detection tools for the iMessage Advanced Insights server.

This module provides tools for identifying significant life events and
tracking emotional wellbeing through conversation analysis.
"""

import logging
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict, Counter
import statistics
import re

from ..database import get_database
from ..exceptions import DatabaseError, ToolExecutionError
from ..utils.responses import error_response, success_response
from ..utils.sanitization import sanitize_messages
from .registry import register_tool

logger = logging.getLogger(__name__)


# Life event patterns and keywords
LIFE_EVENT_PATTERNS = {
    "job_change": {
        "keywords": ["new job", "quit", "fired", "laid off", "promotion", "interview", 
                    "first day", "last day", "resignation", "offer", "hired", "position",
                    "career", "workplace", "office", "boss", "manager", "team"],
        "patterns": [r"got a? (?:new )?job", r"start(?:ing|ed) at", r"leaving (?:my )?job"],
        "confidence_boost": ["congratulations", "excited", "nervous", "opportunity"]
    },
    "relationship_change": {
        "keywords": ["dating", "boyfriend", "girlfriend", "engaged", "married", "divorce",
                    "breakup", "broke up", "single", "relationship", "partner", "wedding",
                    "proposal", "anniversary", "moving in", "ex"],
        "patterns": [r"we.re (?:getting )?(?:engaged|married)", r"broke up with", 
                    r"started dating", r"been together"],
        "confidence_boost": ["love", "happy", "excited", "heartbroken", "miss"]
    },
    "health_event": {
        "keywords": ["doctor", "hospital", "surgery", "diagnosis", "sick", "ill", "health",
                    "medical", "treatment", "recovery", "accident", "emergency", "pregnant",
                    "baby", "expecting"],
        "patterns": [r"went to (?:the )?(?:doctor|hospital)", r"diagnosed with",
                    r"having surgery", r"feeling (?:sick|ill|better)"],
        "confidence_boost": ["worried", "scared", "relief", "prayers", "hoping"]
    },
    "academic_milestone": {
        "keywords": ["graduation", "graduate", "degree", "university", "college", "school",
                    "exam", "finals", "accepted", "admission", "semester", "thesis",
                    "defended", "passed", "failed", "studying"],
        "patterns": [r"graduat(?:ing|ed)", r"got accepted", r"passed (?:my )?exam",
                    r"starting (?:college|university|school)"],
        "confidence_boost": ["proud", "stressed", "accomplished", "finally"]
    },
    "relocation": {
        "keywords": ["moving", "moved", "relocating", "new place", "new city", "packing",
                    "apartment", "house", "home", "address", "neighborhood", "leaving"],
        "patterns": [r"moving to", r"just moved", r"new (?:apartment|house|place)",
                    r"relocating to"],
        "confidence_boost": ["excited", "sad to leave", "fresh start", "miss"]
    },
    "financial_change": {
        "keywords": ["bought", "purchased", "loan", "debt", "mortgage", "investment",
                    "salary", "raise", "bonus", "car", "house", "expensive", "budget",
                    "savings", "broke", "money"],
        "patterns": [r"bought a? (?:new )?(?:car|house)", r"got a? raise",
                    r"paid off", r"investing in"],
        "confidence_boost": ["finally", "proud", "worried about money", "celebrating"]
    },
    "loss_or_grief": {
        "keywords": ["passed away", "died", "funeral", "loss", "grieving", "memorial",
                    "condolences", "sorry for your loss", "rest in peace", "miss them"],
        "patterns": [r"passed away", r"lost (?:my|our)", r"died", r"funeral"],
        "confidence_boost": ["sorry", "condolences", "prayers", "strength", "miss"]
    },
    "achievement": {
        "keywords": ["won", "award", "achievement", "accomplished", "milestone", "goal",
                    "published", "launched", "completed", "finished", "succeeded", "record"],
        "patterns": [r"won (?:the|a)", r"achieved", r"reached (?:my )?goal",
                    r"finally (?:did|finished|completed)"],
        "confidence_boost": ["proud", "congratulations", "amazing", "hard work"]
    }
}


@register_tool(
    name="detect_life_events",
    description="Identify significant life events from conversation patterns"
)
async def detect_life_events_tool(
    time_period: str = "6 months",
    event_categories: Optional[List[str]] = None,
    confidence_threshold: float = 0.6,
    include_context: bool = True
) -> Dict[str, Any]:
    """
    Detect significant life events from message patterns.
    
    This tool identifies major life changes like:
    - Job changes (new job, promotion, layoff)
    - Relationship milestones (dating, engagement, breakup)
    - Health events (illness, recovery, pregnancy)
    - Academic achievements (graduation, acceptance)
    - Relocation (moving homes/cities)
    - Financial changes (major purchases, job loss)
    - Loss and grief
    - Major achievements
    
    Args:
        time_period: Period to analyze (e.g., "6 months", "1 year")
        event_categories: Specific categories to detect (None = all)
        confidence_threshold: Minimum confidence score (0.0-1.0)
        include_context: Include surrounding messages for context
        
    Returns:
        Detected life events with context and insights
    """
    try:
        # Get database connection
        db = await get_database()
        
        # Parse time period
        from ..utils.decorators import parse_date
        start_date = parse_date(time_period)
        end_date = datetime.now()
        
        # Determine which categories to analyze
        if event_categories:
            categories_to_check = {k: v for k, v in LIFE_EVENT_PATTERNS.items() 
                                 if k in event_categories}
        else:
            categories_to_check = LIFE_EVENT_PATTERNS
        
        # Get all messages in time period
        all_events = []
        contacts_analyzed = 0
        
        # Get active contacts
        contacts_result = await db.get_contacts(limit=100, offset=0)
        
        for contact in contacts_result.get("contacts", [])[:50]:  # Analyze top 50
            contact_id = contact.get("phone_number", contact.get("handle_id"))
            contacts_analyzed += 1
            
            # Get messages for this contact
            messages_result = await db.get_messages_from_contact(
                phone_number=contact_id,
                start_date=start_date,
                end_date=end_date,
                page=1, page_size=1000
            )
            
            messages = messages_result.get("messages", [])
            
            if len(messages) < 10:  # Skip if too few messages
                continue
            
            # Detect events in these messages
            contact_events = _detect_events_in_messages(
                messages, 
                categories_to_check, 
                confidence_threshold,
                contact_id
            )
            
            # Add context if requested
            if include_context and contact_events:
                for event in contact_events:
                    event["context"] = _get_event_context(messages, event["message_index"])
            
            all_events.extend(contact_events)
        
        # Sort events by confidence and date
        all_events.sort(key=lambda x: (x["confidence"], x["date"]), reverse=True)
        
        # Group events by category
        events_by_category = defaultdict(list)
        for event in all_events:
            events_by_category[event["category"]].append(event)
        
        # Generate insights
        insights = _generate_life_event_insights(all_events, events_by_category)
        
        # Generate timeline
        timeline = _generate_event_timeline(all_events)
        
        return success_response({
            "detected_events": all_events[:50],  # Limit to top 50
            "events_by_category": {k: v[:10] for k, v in events_by_category.items()},  # Top 10 per category
            "timeline": timeline,
            "insights": insights,
            "statistics": {
                "total_events": len(all_events),
                "contacts_analyzed": contacts_analyzed,
                "time_period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "events_per_category": {k: len(v) for k, v in events_by_category.items()}
            }
        })
        
    except DatabaseError as e:
        logger.error(f"Database error in detect_life_events_tool: {e}")
        return error_response(e)
    except Exception as e:
        logger.error(f"Unexpected error in detect_life_events_tool: {e}")
        logger.error(traceback.format_exc())
        return error_response(ToolExecutionError(f"Failed to detect life events: {str(e)}"))


def _detect_events_in_messages(
    messages: List[Dict], 
    categories: Dict[str, Dict],
    confidence_threshold: float,
    contact_id: str
) -> List[Dict[str, Any]]:
    """Detect life events in a set of messages."""
    detected_events = []
    
    for idx, message in enumerate(messages):
        if not message.get("text"):
            continue
        
        text_lower = message["text"].lower()
        
        for category, patterns in categories.items():
            confidence = 0
            matched_keywords = []
            matched_patterns = []
            
            # Check keywords
            for keyword in patterns["keywords"]:
                if keyword in text_lower:
                    confidence += 0.3
                    matched_keywords.append(keyword)
            
            # Check regex patterns
            for pattern in patterns["patterns"]:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    confidence += 0.5
                    matched_patterns.append(pattern)
            
            # Check confidence boosters
            for booster in patterns.get("confidence_boost", []):
                if booster in text_lower:
                    confidence += 0.1
            
            # Normalize confidence
            confidence = min(confidence, 1.0)
            
            if confidence >= confidence_threshold:
                detected_events.append({
                    "category": category,
                    "confidence": round(confidence, 2),
                    "date": message["date"],
                    "contact_id": contact_id,
                    "message_preview": message["text"][:100] + "..." if len(message["text"]) > 100 else message["text"],
                    "matched_keywords": matched_keywords,
                    "matched_patterns": matched_patterns,
                    "message_index": idx,
                    "is_from_me": message.get("is_from_me", False)
                })
    
    return detected_events


def _get_event_context(messages: List[Dict], event_index: int, context_size: int = 3) -> Dict[str, Any]:
    """Get context messages around a detected event."""
    start_idx = max(0, event_index - context_size)
    end_idx = min(len(messages), event_index + context_size + 1)
    
    context_messages = messages[start_idx:end_idx]
    
    # Analyze emotional tone in context
    emotional_words = {
        "positive": ["happy", "excited", "congratulations", "proud", "great", "amazing", "wonderful"],
        "negative": ["sad", "worried", "stressed", "scared", "sorry", "difficult", "hard"],
        "supportive": ["here for you", "support", "help", "love", "care", "thinking of you"]
    }
    
    emotional_tone = defaultdict(int)
    for msg in context_messages:
        if msg.get("text"):
            text_lower = msg["text"].lower()
            for tone, words in emotional_words.items():
                for word in words:
                    if word in text_lower:
                        emotional_tone[tone] += 1
    
    return {
        "surrounding_messages": sanitize_messages(context_messages),
        "emotional_tone": dict(emotional_tone),
        "support_level": "high" if emotional_tone["supportive"] > 2 else "moderate" if emotional_tone["supportive"] > 0 else "low"
    }


def _generate_life_event_insights(all_events: List[Dict], events_by_category: Dict[str, List]) -> Dict[str, Any]:
    """Generate insights from detected life events."""
    insights = {
        "major_themes": [],
        "emotional_journey": {},
        "support_network": {},
        "life_trajectory": {}
    }
    
    # Identify major themes
    if events_by_category:
        top_categories = sorted(events_by_category.items(), key=lambda x: len(x[1]), reverse=True)[:3]
        for category, events in top_categories:
            insights["major_themes"].append({
                "theme": category.replace("_", " ").title(),
                "frequency": len(events),
                "interpretation": _interpret_event_category(category, len(events))
            })
    
    # Analyze emotional journey
    positive_events = ["job_change", "relationship_change", "academic_milestone", "achievement"]
    challenging_events = ["health_event", "loss_or_grief", "financial_change"]
    
    positive_count = sum(len(events_by_category.get(cat, [])) for cat in positive_events)
    challenging_count = sum(len(events_by_category.get(cat, [])) for cat in challenging_events)
    
    insights["emotional_journey"] = {
        "positive_events": positive_count,
        "challenging_events": challenging_count,
        "overall_tone": "growth-oriented" if positive_count > challenging_count else "resilience-building",
        "interpretation": _interpret_emotional_journey(positive_count, challenging_count)
    }
    
    # Analyze support network
    events_shared = [e for e in all_events if not e.get("is_from_me", False)]
    events_experienced = [e for e in all_events if e.get("is_from_me", False)]
    
    insights["support_network"] = {
        "events_shared_with_you": len(events_shared),
        "your_events_shared": len(events_experienced),
        "support_balance": "balanced" if abs(len(events_shared) - len(events_experienced)) < 3 else "imbalanced",
        "interpretation": "You have a reciprocal support network" if len(events_shared) > 0 and len(events_experienced) > 0 else "Consider sharing more with your network"
    }
    
    return insights


def _generate_event_timeline(events: List[Dict]) -> List[Dict[str, Any]]:
    """Generate a timeline of major life events."""
    # Group events by month
    timeline = defaultdict(list)
    
    for event in events:
        event_date = datetime.fromisoformat(event["date"])
        month_key = event_date.strftime("%Y-%m")
        timeline[month_key].append({
            "date": event["date"],
            "category": event["category"],
            "preview": event["message_preview"],
            "confidence": event["confidence"]
        })
    
    # Convert to sorted list
    timeline_list = []
    for month, month_events in sorted(timeline.items()):
        timeline_list.append({
            "month": month,
            "events": sorted(month_events, key=lambda x: x["date"]),
            "event_count": len(month_events)
        })
    
    return timeline_list


def _interpret_event_category(category: str, count: int) -> str:
    """Provide interpretation for event categories."""
    interpretations = {
        "job_change": "Career transitions indicate professional growth and adaptation",
        "relationship_change": "Relationship milestones show evolving personal connections",
        "health_event": "Health events highlight the importance of wellbeing and support",
        "academic_milestone": "Academic achievements demonstrate dedication to learning",
        "relocation": "Moving represents new beginnings and life changes",
        "financial_change": "Financial events reflect changing life circumstances",
        "loss_or_grief": "Loss events show the depth of human connections",
        "achievement": "Achievements celebrate hard work and success"
    }
    return interpretations.get(category, "Significant life changes detected")


def _interpret_emotional_journey(positive: int, challenging: int) -> str:
    """Interpret the emotional journey based on event types."""
    if positive > challenging * 2:
        return "This period has been marked by significant positive growth and achievements"
    elif challenging > positive * 2:
        return "This period has involved navigating significant challenges with resilience"
    else:
        return "This period shows a balance of growth opportunities and challenges"


@register_tool(
    name="analyze_emotional_wellbeing",
    description="Monitor emotional health indicators across conversations"
)
async def analyze_emotional_wellbeing_tool(
    contact_id: Optional[str] = None,
    time_period: str = "3 months",
    include_network_effect: bool = True,
    analysis_depth: str = "comprehensive"
) -> Dict[str, Any]:
    """
    Analyze emotional wellbeing patterns across conversations.
    
    This tool examines:
    - Emotional volatility and stability
    - Support seeking and giving patterns
    - Stress indicators and coping mechanisms
    - Joy and celebration frequency
    - Social isolation risks
    - Overall emotional health trajectory
    
    Args:
        contact_id: Specific contact to analyze (None = overall network)
        time_period: Period to analyze
        include_network_effect: Consider full social network impact
        analysis_depth: Level of analysis ("basic", "moderate", "comprehensive")
        
    Returns:
        Comprehensive emotional wellbeing analysis
    """
    try:
        # Get database connection
        db = await get_database()
        
        # Parse time period
        from ..utils.decorators import parse_date
        start_date = parse_date(time_period)
        end_date = datetime.now()
        
        wellbeing_data = {
            "individual_analyses": {},
            "network_analysis": {},
            "overall_wellbeing": {}
        }
        
        if contact_id:
            # Analyze specific contact
            contact_wellbeing = await _analyze_contact_wellbeing(
                db, contact_id, start_date, end_date, analysis_depth
            )
            wellbeing_data["individual_analyses"][contact_id] = contact_wellbeing
        else:
            # Analyze top contacts
            contacts_result = await db.get_contacts(limit=20, offset=0)
            
            for contact in contacts_result.get("contacts", [])[:10]:
                contact_wellbeing = await _analyze_contact_wellbeing(
                    db, contact.get("phone_number", contact.get("handle_id")), start_date, end_date, analysis_depth
                )
                wellbeing_data["individual_analyses"][contact.get("phone_number", contact.get("handle_id"))] = contact_wellbeing
        
        # Network-wide analysis if requested
        if include_network_effect and not contact_id:
            wellbeing_data["network_analysis"] = _analyze_network_wellbeing(
                wellbeing_data["individual_analyses"]
            )
        
        # Calculate overall wellbeing score
        wellbeing_data["overall_wellbeing"] = _calculate_overall_wellbeing(
            wellbeing_data["individual_analyses"],
            wellbeing_data.get("network_analysis", {})
        )
        
        # Generate insights and recommendations
        insights = _generate_wellbeing_insights(wellbeing_data)
        recommendations = _generate_wellbeing_recommendations(wellbeing_data)
        
        return success_response({
            "wellbeing_analysis": wellbeing_data,
            "insights": insights,
            "recommendations": recommendations,
            "metadata": {
                "time_period": time_period,
                "analysis_depth": analysis_depth,
                "contacts_analyzed": len(wellbeing_data["individual_analyses"])
            }
        })
        
    except Exception as e:
        logger.error(f"Error in analyze_emotional_wellbeing_tool: {e}")
        logger.error(traceback.format_exc())
        return error_response(ToolExecutionError(f"Failed to analyze wellbeing: {str(e)}"))


async def _analyze_contact_wellbeing(
    db: Any,
    contact_id: str,
    start_date: datetime,
    end_date: datetime,
    analysis_depth: str
) -> Dict[str, Any]:
    """Analyze emotional wellbeing for a specific contact."""
    
    # Get messages
    messages_result = await db.get_messages_from_contact(
        phone_number=contact_id,
        start_date=start_date,
        end_date=end_date,
        page=1, page_size=500
    )
    
    messages = messages_result.get("messages", [])
    
    if len(messages) < 20:
        return {
            "status": "insufficient_data",
            "message_count": len(messages)
        }
    
    # Analyze emotional patterns
    emotional_scores = _calculate_emotional_scores(messages)
    volatility = _calculate_emotional_volatility(emotional_scores)
    
    # Analyze support patterns
    support_patterns = _analyze_support_patterns(messages)
    
    # Detect stress indicators
    stress_indicators = _detect_stress_indicators(messages)
    
    # Calculate joy frequency
    joy_metrics = _calculate_joy_metrics(messages)
    
    # Assess isolation risk
    isolation_risk = _assess_isolation_risk(messages, contact_id)
    
    wellbeing_analysis = {
        "emotional_state": {
            "current_mood": _determine_current_mood(emotional_scores[-10:]) if len(emotional_scores) > 10 else "neutral",
            "volatility": volatility,
            "stability_score": max(0, 100 - (volatility * 100)),
            "trend": _calculate_emotional_trend(emotional_scores)
        },
        "support_dynamics": support_patterns,
        "stress_level": stress_indicators,
        "joy_frequency": joy_metrics,
        "isolation_risk": isolation_risk,
        "overall_score": _calculate_contact_wellbeing_score(
            volatility, support_patterns, stress_indicators, joy_metrics, isolation_risk
        )
    }
    
    if analysis_depth == "comprehensive":
        wellbeing_analysis["detailed_metrics"] = {
            "emotional_timeline": _create_emotional_timeline(messages, emotional_scores),
            "coping_mechanisms": _identify_coping_mechanisms(messages),
            "communication_health": _assess_communication_health(messages)
        }
    
    return wellbeing_analysis


def _calculate_emotional_scores(messages: List[Dict]) -> List[float]:
    """Calculate emotional scores for messages over time."""
    scores = []
    
    # Emotional word dictionaries
    positive_words = {
        "happy", "joy", "excited", "love", "great", "amazing", "wonderful",
        "fantastic", "excellent", "blessed", "grateful", "lol", "haha", "yay"
    }
    negative_words = {
        "sad", "angry", "frustrated", "upset", "worried", "stressed", "anxious",
        "depressed", "terrible", "awful", "hate", "scared", "lonely", "tired"
    }
    
    for msg in messages:
        if msg.get("text"):
            text_lower = msg["text"].lower()
            words = set(text_lower.split())
            
            positive_count = len(words & positive_words)
            negative_count = len(words & negative_words)
            
            # Calculate score (-1 to 1)
            if positive_count + negative_count > 0:
                score = (positive_count - negative_count) / (positive_count + negative_count)
            else:
                score = 0
            
            scores.append(score)
    
    return scores


def _calculate_emotional_volatility(scores: List[float]) -> float:
    """Calculate emotional volatility from scores."""
    if len(scores) < 2:
        return 0
    
    # Calculate standard deviation of score changes
    changes = [abs(scores[i] - scores[i-1]) for i in range(1, len(scores))]
    
    if changes:
        volatility = statistics.stdev(changes) if len(changes) > 1 else 0
        return min(volatility, 1.0)  # Normalize to 0-1
    
    return 0


def _analyze_support_patterns(messages: List[Dict]) -> Dict[str, Any]:
    """Analyze support seeking and giving patterns."""
    support_given = 0
    support_received = 0
    support_keywords = {
        "here for you", "support", "help", "listen", "understand", "care",
        "love you", "thinking of you", "hug", "sorry to hear", "praying"
    }
    
    for msg in messages:
        if msg.get("text"):
            text_lower = msg["text"].lower()
            for keyword in support_keywords:
                if keyword in text_lower:
                    if msg.get("is_from_me"):
                        support_given += 1
                    else:
                        support_received += 1
                    break
    
    total_messages = len(messages)
    
    return {
        "support_given_rate": support_given / max(total_messages, 1),
        "support_received_rate": support_received / max(total_messages, 1),
        "support_balance": "balanced" if abs(support_given - support_received) < 5 else "imbalanced",
        "support_frequency": (support_given + support_received) / max(total_messages, 1),
        "interpretation": _interpret_support_patterns(support_given, support_received)
    }


def _detect_stress_indicators(messages: List[Dict]) -> Dict[str, Any]:
    """Detect stress indicators in messages."""
    stress_words = {
        "stressed", "overwhelmed", "anxious", "worried", "pressure", "deadline",
        "exhausted", "burnout", "can't sleep", "too much", "drowning", "struggling"
    }
    
    stress_count = 0
    stress_messages = []
    
    for msg in messages:
        if msg.get("text"):
            text_lower = msg["text"].lower()
            for word in stress_words:
                if word in text_lower:
                    stress_count += 1
                    stress_messages.append(msg["date"])
                    break
    
    # Calculate stress frequency and trend
    stress_rate = stress_count / max(len(messages), 1)
    
    # Determine stress level
    if stress_rate > 0.1:
        stress_level = "high"
    elif stress_rate > 0.05:
        stress_level = "moderate"
    else:
        stress_level = "low"
    
    return {
        "stress_level": stress_level,
        "stress_frequency": stress_rate,
        "stress_mentions": stress_count,
        "recent_stress": len([d for d in stress_messages if 
                            (datetime.now() - datetime.fromisoformat(d)).days < 7]),
        "interpretation": f"Stress indicators are {stress_level}, appearing in {stress_rate*100:.1f}% of messages"
    }


def _calculate_joy_metrics(messages: List[Dict]) -> Dict[str, Any]:
    """Calculate joy and celebration frequency."""
    joy_words = {
        "happy", "excited", "celebrate", "amazing", "wonderful", "blessed",
        "grateful", "joy", "fun", "love", "yay", "party", "congratulations"
    }
    
    celebration_patterns = [
        r"birth?day", r"anniversary", r"promot(?:ion|ed)", r"graduat",
        r"got (?:the )?job", r"engag(?:ed|ement)", r"married", r"baby"
    ]
    
    joy_count = 0
    celebration_count = 0
    
    for msg in messages:
        if msg.get("text"):
            text_lower = msg["text"].lower()
            
            # Check joy words
            for word in joy_words:
                if word in text_lower:
                    joy_count += 1
                    break
            
            # Check celebration patterns
            for pattern in celebration_patterns:
                if re.search(pattern, text_lower):
                    celebration_count += 1
                    break
    
    total_messages = len(messages)
    
    return {
        "joy_frequency": joy_count / max(total_messages, 1),
        "celebration_frequency": celebration_count / max(total_messages, 1),
        "positivity_score": min((joy_count * 2) / max(total_messages, 1), 1.0),
        "recent_celebrations": celebration_count,
        "interpretation": _interpret_joy_metrics(joy_count, celebration_count, total_messages)
    }


def _assess_isolation_risk(messages: List[Dict], contact_id: str) -> Dict[str, Any]:
    """Assess risk of social isolation."""
    # Check message frequency trend
    daily_counts = defaultdict(int)
    
    for msg in messages:
        if msg.get("date"):
            date = datetime.fromisoformat(msg["date"]).date()
            daily_counts[date] += 1
    
    # Calculate recent vs historical activity
    all_dates = sorted(daily_counts.keys())
    if len(all_dates) < 14:
        return {"risk_level": "unknown", "reason": "Insufficient data"}
    
    recent_dates = all_dates[-7:]
    historical_dates = all_dates[:-7]
    
    recent_avg = sum(daily_counts[d] for d in recent_dates) / 7
    historical_avg = sum(daily_counts[d] for d in historical_dates) / len(historical_dates)
    
    # Calculate isolation risk
    if recent_avg < historical_avg * 0.3:
        risk_level = "high"
    elif recent_avg < historical_avg * 0.6:
        risk_level = "moderate"
    else:
        risk_level = "low"
    
    return {
        "risk_level": risk_level,
        "communication_decline": f"{((historical_avg - recent_avg) / historical_avg * 100):.0f}%",
        "recent_activity": f"{recent_avg:.1f} messages/day",
        "historical_baseline": f"{historical_avg:.1f} messages/day",
        "interpretation": _interpret_isolation_risk(risk_level, recent_avg, historical_avg)
    }


def _calculate_contact_wellbeing_score(
    volatility: float,
    support: Dict,
    stress: Dict,
    joy: Dict,
    isolation: Dict
) -> float:
    """Calculate overall wellbeing score for a contact."""
    score = 50  # Start at neutral
    
    # Emotional stability (up to +20)
    score += (1 - volatility) * 20
    
    # Support patterns (up to +15)
    if support["support_frequency"] > 0.05:
        score += 15
    elif support["support_frequency"] > 0.02:
        score += 7
    
    # Stress level (up to -20)
    if stress["stress_level"] == "high":
        score -= 20
    elif stress["stress_level"] == "moderate":
        score -= 10
    
    # Joy frequency (up to +15)
    score += joy["positivity_score"] * 15
    
    # Isolation risk (up to -20)
    if isolation["risk_level"] == "high":
        score -= 20
    elif isolation["risk_level"] == "moderate":
        score -= 10
    
    return max(0, min(100, score))


def _determine_current_mood(recent_scores: List[float]) -> str:
    """Determine current mood from recent emotional scores."""
    if not recent_scores:
        return "neutral"
    
    avg_score = statistics.mean(recent_scores)
    
    if avg_score > 0.5:
        return "positive"
    elif avg_score > 0.2:
        return "optimistic"
    elif avg_score > -0.2:
        return "neutral"
    elif avg_score > -0.5:
        return "concerned"
    else:
        return "struggling"


def _calculate_emotional_trend(scores: List[float]) -> str:
    """Calculate trend in emotional scores."""
    if len(scores) < 10:
        return "insufficient_data"
    
    # Compare recent to historical
    recent = statistics.mean(scores[-10:])
    historical = statistics.mean(scores[:-10])
    
    if recent > historical + 0.2:
        return "improving"
    elif recent < historical - 0.2:
        return "declining"
    else:
        return "stable"


def _create_emotional_timeline(messages: List[Dict], scores: List[float]) -> List[Dict]:
    """Create a timeline of emotional states."""
    timeline = []
    
    # Group by week
    weekly_scores = defaultdict(list)
    
    for msg, score in zip(messages, scores):
        if msg.get("date"):
            week = datetime.fromisoformat(msg["date"]).isocalendar()
            week_key = f"{week[0]}-W{week[1]}"
            weekly_scores[week_key].append(score)
    
    for week, scores in sorted(weekly_scores.items()):
        if scores:
            timeline.append({
                "week": week,
                "average_mood": statistics.mean(scores),
                "mood_label": _score_to_mood(statistics.mean(scores)),
                "volatility": statistics.stdev(scores) if len(scores) > 1 else 0
            })
    
    return timeline[-12:]  # Last 12 weeks


def _score_to_mood(score: float) -> str:
    """Convert emotional score to mood label."""
    if score > 0.5:
        return "very_positive"
    elif score > 0.2:
        return "positive"
    elif score > -0.2:
        return "neutral"
    elif score > -0.5:
        return "negative"
    else:
        return "very_negative"


def _identify_coping_mechanisms(messages: List[Dict]) -> List[str]:
    """Identify coping mechanisms from messages."""
    coping_patterns = {
        "social_support": ["talk", "vent", "share", "tell you", "need to talk"],
        "physical_activity": ["gym", "run", "workout", "exercise", "walk", "yoga"],
        "creative_expression": ["writing", "music", "art", "painting", "creating"],
        "mindfulness": ["meditate", "breathe", "relax", "calm", "peace"],
        "humor": ["laugh", "joke", "funny", "lol", "haha"],
        "problem_solving": ["plan", "figure out", "solve", "work on", "fix"]
    }
    
    identified_mechanisms = set()
    
    for msg in messages:
        if msg.get("text"):
            text_lower = msg["text"].lower()
            for mechanism, patterns in coping_patterns.items():
                for pattern in patterns:
                    if pattern in text_lower:
                        identified_mechanisms.add(mechanism)
    
    return list(identified_mechanisms)


def _assess_communication_health(messages: List[Dict]) -> Dict[str, Any]:
    """Assess overall communication health."""
    # Calculate message length distribution
    lengths = [len(msg.get("text", "")) for msg in messages if msg.get("text")]
    
    if not lengths:
        return {"status": "no_data"}
    
    avg_length = statistics.mean(lengths)
    
    # Check for balanced exchange
    sent = sum(1 for msg in messages if msg.get("is_from_me"))
    received = len(messages) - sent
    
    balance_ratio = sent / max(received, 1)
    
    return {
        "message_depth": "substantial" if avg_length > 50 else "moderate" if avg_length > 20 else "brief",
        "exchange_balance": "balanced" if 0.5 <= balance_ratio <= 2.0 else "imbalanced",
        "engagement_level": "high" if avg_length > 50 and 0.7 <= balance_ratio <= 1.3 else "moderate",
        "average_message_length": round(avg_length, 1)
    }


def _analyze_network_wellbeing(individual_analyses: Dict[str, Dict]) -> Dict[str, Any]:
    """Analyze wellbeing across the entire network."""
    if not individual_analyses:
        return {"status": "no_data"}
    
    # Calculate network-wide metrics
    all_scores = []
    stress_levels = Counter()
    isolation_risks = Counter()
    
    for contact_id, analysis in individual_analyses.items():
        if analysis.get("status") != "insufficient_data":
            all_scores.append(analysis.get("overall_score", 50))
            stress_levels[analysis.get("stress_level", {}).get("stress_level", "unknown")] += 1
            isolation_risks[analysis.get("isolation_risk", {}).get("risk_level", "unknown")] += 1
    
    if not all_scores:
        return {"status": "no_data"}
    
    return {
        "network_health_score": statistics.mean(all_scores),
        "network_volatility": statistics.stdev(all_scores) if len(all_scores) > 1 else 0,
        "stress_distribution": dict(stress_levels),
        "isolation_distribution": dict(isolation_risks),
        "at_risk_contacts": sum(1 for s in all_scores if s < 40),
        "thriving_contacts": sum(1 for s in all_scores if s > 70),
        "network_interpretation": _interpret_network_health(all_scores, stress_levels, isolation_risks)
    }


def _calculate_overall_wellbeing(individual: Dict[str, Dict], network: Dict) -> Dict[str, Any]:
    """Calculate overall wellbeing metrics."""
    individual_scores = [
        analysis.get("overall_score", 50) 
        for analysis in individual.values() 
        if analysis.get("status") != "insufficient_data"
    ]
    
    if not individual_scores:
        return {"status": "no_data"}
    
    overall_score = statistics.mean(individual_scores)
    
    # Determine wellbeing category
    if overall_score >= 80:
        category = "thriving"
    elif overall_score >= 60:
        category = "healthy"
    elif overall_score >= 40:
        category = "managing"
    else:
        category = "struggling"
    
    return {
        "score": round(overall_score, 1),
        "category": category,
        "trend": _calculate_wellbeing_trend(individual),
        "key_factors": _identify_key_wellbeing_factors(individual, network)
    }


def _calculate_wellbeing_trend(individual: Dict[str, Dict]) -> str:
    """Calculate wellbeing trend from individual analyses."""
    trends = []
    
    for analysis in individual.values():
        if analysis.get("emotional_state"):
            trend = analysis["emotional_state"].get("trend")
            if trend and trend != "insufficient_data":
                trends.append(trend)
    
    if not trends:
        return "unknown"
    
    improving = trends.count("improving")
    declining = trends.count("declining")
    
    if improving > declining * 2:
        return "improving"
    elif declining > improving * 2:
        return "declining"
    else:
        return "stable"


def _identify_key_wellbeing_factors(individual: Dict[str, Dict], network: Dict) -> List[str]:
    """Identify key factors affecting wellbeing."""
    factors = []
    
    # Check for high stress
    high_stress_count = sum(
        1 for analysis in individual.values()
        if analysis.get("stress_level", {}).get("stress_level") == "high"
    )
    if high_stress_count > len(individual) * 0.3:
        factors.append("Elevated stress levels across multiple relationships")
    
    # Check for isolation
    isolation_count = sum(
        1 for analysis in individual.values()
        if analysis.get("isolation_risk", {}).get("risk_level") in ["high", "moderate"]
    )
    if isolation_count > len(individual) * 0.4:
        factors.append("Risk of social isolation detected")
    
    # Check for positive factors
    high_joy_count = sum(
        1 for analysis in individual.values()
        if analysis.get("joy_frequency", {}).get("positivity_score", 0) > 0.5
    )
    if high_joy_count > len(individual) * 0.5:
        factors.append("Strong presence of joy and celebration")
    
    return factors


def _generate_wellbeing_insights(wellbeing_data: Dict) -> Dict[str, Any]:
    """Generate insights from wellbeing analysis."""
    insights = {
        "key_findings": [],
        "strengths": [],
        "concerns": []
    }
    
    overall = wellbeing_data.get("overall_wellbeing", {})
    
    # Overall status
    if overall.get("category") == "thriving":
        insights["key_findings"].append("Overall emotional wellbeing is excellent")
    elif overall.get("category") == "struggling":
        insights["key_findings"].append("Emotional wellbeing needs attention and support")
    
    # Network insights
    network = wellbeing_data.get("network_analysis", {})
    if network.get("at_risk_contacts", 0) > 0:
        insights["concerns"].append(f"{network['at_risk_contacts']} contacts may need additional support")
    
    # Individual patterns
    for contact_id, analysis in wellbeing_data.get("individual_analyses", {}).items():
        if analysis.get("support_dynamics", {}).get("support_frequency", 0) > 0.1:
            insights["strengths"].append("Strong mutual support in key relationships")
            break
    
    return insights


def _generate_wellbeing_recommendations(wellbeing_data: Dict) -> List[Dict[str, str]]:
    """Generate recommendations for improving wellbeing."""
    recommendations = []
    
    # Check for high stress
    for contact_id, analysis in wellbeing_data.get("individual_analyses", {}).items():
        if analysis.get("stress_level", {}).get("stress_level") == "high":
            recommendations.append({
                "action": "Practice stress management techniques",
                "reason": "High stress indicators detected in conversations",
                "priority": "high",
                "suggestion": "Consider scheduling regular check-ins or relaxation activities"
            })
            break
    
    # Check for isolation risk
    isolation_risks = [
        analysis for analysis in wellbeing_data.get("individual_analyses", {}).values()
        if analysis.get("isolation_risk", {}).get("risk_level") == "high"
    ]
    if isolation_risks:
        recommendations.append({
            "action": "Increase social connection efforts",
            "reason": f"{len(isolation_risks)} relationships showing signs of isolation",
            "priority": "high",
            "suggestion": "Reach out to distant friends or join social activities"
        })
    
    # Enhance positive patterns
    if wellbeing_data.get("overall_wellbeing", {}).get("score", 0) > 70:
        recommendations.append({
            "action": "Maintain current positive patterns",
            "reason": "Your emotional wellbeing practices are working well",
            "priority": "low",
            "suggestion": "Continue nurturing supportive relationships"
        })
    
    return recommendations[:5]


# Helper interpretation functions
def _interpret_support_patterns(given: int, received: int) -> str:
    """Interpret support patterns."""
    if given > received * 2:
        return "You're a strong support provider - remember self-care is important too"
    elif received > given * 2:
        return "You have a good support network - consider offering support in return"
    else:
        return "Healthy balance of giving and receiving support"


def _interpret_joy_metrics(joy_count: int, celebration_count: int, total: int) -> str:
    """Interpret joy and celebration metrics."""
    joy_rate = joy_count / max(total, 1)
    if joy_rate > 0.2:
        return "High frequency of joy and positive emotions"
    elif joy_rate > 0.1:
        return "Moderate presence of positive emotions"
    else:
        return "Consider celebrating small wins more often"


def _interpret_isolation_risk(risk_level: str, recent: float, historical: float) -> str:
    """Interpret isolation risk."""
    if risk_level == "high":
        return "Significant decline in communication - reconnection recommended"
    elif risk_level == "moderate":
        return "Some reduction in social interaction detected"
    else:
        return "Healthy level of social connection maintained"


def _interpret_network_health(scores: List[float], stress: Counter, isolation: Counter) -> str:
    """Interpret overall network health."""
    avg_score = statistics.mean(scores) if scores else 0
    
    if avg_score > 70:
        return "Your social network is emotionally healthy and supportive"
    elif avg_score > 50:
        return "Your network shows mixed wellbeing - some relationships need attention"
    else:
        return "Several relationships in your network may benefit from increased support"