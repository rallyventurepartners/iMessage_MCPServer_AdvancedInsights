"""
Predictive Analytics tools for the iMessage Advanced Insights server.

This module provides tools for forecasting communication patterns, detecting
anomalies, and predicting relationship trajectories using time series analysis
and machine learning techniques.
"""

import logging
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from collections import defaultdict, Counter
import statistics
import math

from ..database import get_database
from ..exceptions import DatabaseError, ToolExecutionError
from ..utils.responses import error_response, success_response
from ..utils.sanitization import sanitize_contact_info
from .registry import register_tool

logger = logging.getLogger(__name__)


@register_tool(
    name="predict_communication_patterns",
    description="Forecast future communication patterns using ML and time series analysis"
)
async def predict_communication_patterns_tool(
    contact_id: Optional[str] = None,
    prediction_window: str = "30 days",
    include_anomaly_detection: bool = True,
    confidence_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Predict future communication patterns based on historical data.
    
    This tool analyzes:
    - Message frequency trends
    - Response time patterns
    - Peak communication times
    - Relationship trajectory
    - Anomaly detection (unusual silences)
    
    Args:
        contact_id: Optional specific contact to analyze
        prediction_window: Time window for predictions (e.g., "30 days", "1 week")
        include_anomaly_detection: Whether to detect anomalies
        confidence_threshold: Minimum confidence level for predictions
        
    Returns:
        Comprehensive prediction analysis with actionable recommendations
    """
    try:
        # Get database connection
        db = await get_database()
        
        # Parse prediction window
        days_to_predict = _parse_time_window(prediction_window)
        
        # Get historical data for analysis (6 months back)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        predictions = {}
        
        if contact_id:
            # Analyze specific contact
            contact_predictions = await _predict_for_contact(
                db, contact_id, start_date, end_date, days_to_predict
            )
            predictions[contact_id] = contact_predictions
        else:
            # Analyze all active contacts
            contacts_result = await db.get_contacts(limit=50, offset=0)
            
            for contact in contacts_result.get("contacts", [])[:10]:  # Top 10 active
                contact_predictions = await _predict_for_contact(
                    db, contact.get("phone_number", contact.get("handle_id")), start_date, end_date, days_to_predict
                )
                predictions[contact.get("phone_number", contact.get("handle_id"))] = contact_predictions
        
        # Detect anomalies if requested
        anomalies = []
        if include_anomaly_detection:
            anomalies = await _detect_communication_anomalies(db, predictions)
        
        # Generate overall insights and recommendations
        overall_insights = _generate_overall_insights(predictions, anomalies)
        recommendations = _generate_predictive_recommendations(predictions, anomalies)
        
        return success_response({
            "predictions": predictions,
            "anomalies": anomalies,
            "insights": overall_insights,
            "recommendations": recommendations,
            "metadata": {
                "prediction_window": prediction_window,
                "days_predicted": days_to_predict,
                "confidence_threshold": confidence_threshold,
                "contacts_analyzed": len(predictions)
            }
        })
        
    except DatabaseError as e:
        logger.error(f"Database error in predict_communication_patterns_tool: {e}")
        return error_response(e)
    except Exception as e:
        logger.error(f"Unexpected error in predict_communication_patterns_tool: {e}")
        logger.error(traceback.format_exc())
        return error_response(ToolExecutionError(f"Failed to predict patterns: {str(e)}"))


async def _predict_for_contact(
    db: Any,
    contact_id: str,
    start_date: datetime,
    end_date: datetime,
    days_to_predict: int
) -> Dict[str, Any]:
    """Predict communication patterns for a specific contact."""
    
    # Get historical messages
    messages_result = await db.get_messages_from_contact(
        phone_number=contact_id,
        start_date=start_date,
        end_date=end_date,
        page=1, page_size=1000
    )
    
    messages = messages_result.get("messages", [])
    
    if len(messages) < 10:  # Not enough data for prediction
        return {
            "status": "insufficient_data",
            "message": "Not enough historical data for accurate predictions"
        }
    
    # Analyze historical patterns
    daily_counts = _calculate_daily_message_counts(messages)
    hourly_distribution = _calculate_hourly_distribution(messages)
    response_times = _calculate_response_times(messages)
    
    # Calculate trends
    frequency_trend = _calculate_trend(daily_counts)
    
    # Make predictions
    predicted_daily_avg = _predict_daily_average(daily_counts, days_to_predict)
    predicted_total = predicted_daily_avg * days_to_predict
    
    # Calculate confidence based on variance
    confidence = _calculate_prediction_confidence(daily_counts)
    
    # Detect relationship trajectory
    trajectory = _analyze_relationship_trajectory(messages, daily_counts)
    
    # Identify peak times
    peak_times = _identify_peak_times(hourly_distribution)
    
    return {
        "message_volume": {
            "expected_daily_average": round(predicted_daily_avg, 1),
            "expected_total": int(predicted_total),
            "confidence": round(confidence, 2),
            "trend": frequency_trend,
            "peak_times": peak_times
        },
        "relationship_trajectory": trajectory,
        "response_patterns": {
            "average_response_time_minutes": round(statistics.mean(response_times) if response_times else 0, 1),
            "response_time_trend": _calculate_trend(response_times[-30:]) if len(response_times) > 30 else "stable"
        },
        "historical_baseline": {
            "daily_average": round(statistics.mean(daily_counts) if daily_counts else 0, 1),
            "weekly_average": round(statistics.mean(daily_counts) * 7 if daily_counts else 0, 1)
        }
    }


def _calculate_daily_message_counts(messages: List[Dict]) -> List[float]:
    """Calculate daily message counts from messages."""
    if not messages:
        return []
    
    daily_counts = defaultdict(int)
    
    for msg in messages:
        if msg.get("date"):
            date = datetime.fromisoformat(msg["date"]).date()
            daily_counts[date] += 1
    
    # Fill in missing days with zeros
    all_dates = []
    if daily_counts:
        start = min(daily_counts.keys())
        end = max(daily_counts.keys())
        current = start
        while current <= end:
            all_dates.append(daily_counts.get(current, 0))
            current += timedelta(days=1)
    
    return all_dates


def _calculate_hourly_distribution(messages: List[Dict]) -> Dict[int, int]:
    """Calculate message distribution by hour of day."""
    hourly_counts = defaultdict(int)
    
    for msg in messages:
        if msg.get("date"):
            hour = datetime.fromisoformat(msg["date"]).hour
            hourly_counts[hour] += 1
    
    return dict(hourly_counts)


def _calculate_response_times(messages: List[Dict]) -> List[float]:
    """Calculate response times between messages."""
    response_times = []
    
    for i in range(1, len(messages)):
        if messages[i].get("is_from_me") != messages[i-1].get("is_from_me"):
            # This is a response
            try:
                time1 = datetime.fromisoformat(messages[i-1]["date"])
                time2 = datetime.fromisoformat(messages[i]["date"])
                diff_minutes = (time2 - time1).total_seconds() / 60
                if 0 < diff_minutes < 1440:  # Ignore if > 24 hours
                    response_times.append(diff_minutes)
            except:
                pass
    
    return response_times


def _calculate_trend(values: List[float]) -> str:
    """Calculate trend from a series of values."""
    if len(values) < 3:
        return "stable"
    
    # Simple linear regression
    n = len(values)
    if n == 0:
        return "stable"
        
    x_mean = n / 2
    y_mean = sum(values) / n
    
    numerator = sum((i - x_mean) * (val - y_mean) for i, val in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    
    if denominator == 0:
        return "stable"
    
    slope = numerator / denominator
    
    # Determine trend based on slope
    if slope > 0.1:
        return "increasing"
    elif slope < -0.1:
        return "decreasing"
    else:
        return "stable"


def _predict_daily_average(daily_counts: List[float], days_ahead: int) -> float:
    """Predict daily average using simple moving average with trend."""
    if not daily_counts:
        return 0
    
    # Use last 30 days for prediction
    recent_counts = daily_counts[-30:] if len(daily_counts) > 30 else daily_counts
    
    if not recent_counts:
        return 0
    
    # Calculate moving average
    avg = statistics.mean(recent_counts)
    
    # Add trend component
    trend = _calculate_trend(recent_counts)
    if trend == "increasing":
        avg *= 1.1  # 10% increase
    elif trend == "decreasing":
        avg *= 0.9  # 10% decrease
    
    return max(0, avg)


def _calculate_prediction_confidence(daily_counts: List[float]) -> float:
    """Calculate confidence score based on data variance."""
    if len(daily_counts) < 7:
        return 0.5  # Low confidence with little data
    
    # Calculate coefficient of variation
    if daily_counts:
        mean_val = statistics.mean(daily_counts)
        if mean_val > 0:
            std_dev = statistics.stdev(daily_counts) if len(daily_counts) > 1 else 0
            cv = std_dev / mean_val
            # Lower CV means more consistent data, higher confidence
            confidence = max(0.5, min(0.95, 1 - (cv / 2)))
            return confidence
    
    return 0.7  # Default confidence


def _analyze_relationship_trajectory(messages: List[Dict], daily_counts: List[float]) -> Dict[str, Any]:
    """Analyze the trajectory of the relationship."""
    
    # Compare recent activity to historical
    if len(daily_counts) < 30:
        return {
            "direction": "insufficient_data",
            "confidence": 0.5
        }
    
    recent_avg = statistics.mean(daily_counts[-30:])
    historical_avg = statistics.mean(daily_counts[:-30])
    
    if recent_avg > historical_avg * 1.2:
        direction = "strengthening"
    elif recent_avg < historical_avg * 0.8:
        direction = "weakening"
    else:
        direction = "stable"
    
    # Analyze message quality trends (simplified)
    recent_messages = messages[-100:]
    avg_length_recent = statistics.mean([len(m.get("text", "")) for m in recent_messages if m.get("text")])
    
    return {
        "direction": direction,
        "key_indicators": {
            "frequency_change": f"{((recent_avg / historical_avg) - 1) * 100:.1f}%",
            "engagement_quality": "high" if avg_length_recent > 50 else "moderate"
        },
        "confidence": _calculate_prediction_confidence(daily_counts[-30:])
    }


def _identify_peak_times(hourly_distribution: Dict[int, int]) -> List[str]:
    """Identify peak communication times."""
    if not hourly_distribution:
        return []
    
    # Sort hours by message count
    sorted_hours = sorted(hourly_distribution.items(), key=lambda x: x[1], reverse=True)
    
    # Get top 3 hours
    peak_hours = [hour for hour, _ in sorted_hours[:3]]
    
    # Convert to time periods
    time_periods = []
    for hour in sorted(peak_hours):
        if 5 <= hour < 12:
            period = "morning"
        elif 12 <= hour < 17:
            period = "afternoon"
        elif 17 <= hour < 22:
            period = "evening"
        else:
            period = "night"
        
        if period not in time_periods:
            time_periods.append(period)
    
    return time_periods


async def _detect_communication_anomalies(db: Any, predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Detect anomalies in communication patterns."""
    anomalies = []
    
    for contact_id, prediction in predictions.items():
        if prediction.get("status") == "insufficient_data":
            continue
        
        # Get recent activity
        recent_messages = await db.get_messages_from_contact(
            phone_number=contact_id,
            start_date=datetime.now() - timedelta(days=30),
            page=1, page_size=100
        )
        
        messages = recent_messages.get("messages", [])
        
        # Check for silence anomaly
        if messages:
            last_message_date = datetime.fromisoformat(messages[-1]["date"])
            days_silent = (datetime.now() - last_message_date).days
            
            expected_daily = prediction["message_volume"]["expected_daily_average"]
            if expected_daily > 1 and days_silent > 7:
                anomalies.append({
                    "type": "unusual_silence",
                    "contact_id": contact_id,
                    "last_message": last_message_date.isoformat(),
                    "days_silent": days_silent,
                    "normal_frequency": f"{expected_daily:.1f} messages/day",
                    "concern_level": "high" if days_silent > 14 else "medium"
                })
    
    return anomalies


def _generate_overall_insights(predictions: Dict[str, Any], anomalies: List[Dict]) -> Dict[str, Any]:
    """Generate overall insights from predictions."""
    
    if not predictions:
        return {"status": "no_data"}
    
    # Count relationship trajectories
    trajectories = defaultdict(int)
    for _, pred in predictions.items():
        if "relationship_trajectory" in pred:
            direction = pred["relationship_trajectory"].get("direction", "unknown")
            trajectories[direction] += 1
    
    # Calculate overall activity trend
    total_predicted = sum(
        p["message_volume"]["expected_daily_average"] 
        for p in predictions.values() 
        if "message_volume" in p
    )
    
    return {
        "relationship_health": {
            "strengthening": trajectories.get("strengthening", 0),
            "stable": trajectories.get("stable", 0),
            "weakening": trajectories.get("weakening", 0)
        },
        "communication_forecast": {
            "total_daily_messages": round(total_predicted, 1),
            "active_relationships": len(predictions),
            "relationships_at_risk": len([a for a in anomalies if a.get("concern_level") == "high"])
        },
        "key_findings": _generate_key_findings(predictions, anomalies)
    }


def _generate_key_findings(predictions: Dict[str, Any], anomalies: List[Dict]) -> List[str]:
    """Generate key findings for Claude to communicate."""
    findings = []
    
    # Strengthening relationships
    strengthening = [
        contact_id for contact_id, pred in predictions.items()
        if pred.get("relationship_trajectory", {}).get("direction") == "strengthening"
    ]
    if strengthening:
        findings.append(f"{len(strengthening)} relationship(s) showing increased engagement")
    
    # At-risk relationships
    high_risk = [a for a in anomalies if a.get("concern_level") == "high"]
    if high_risk:
        findings.append(f"{len(high_risk)} relationship(s) need immediate attention")
    
    # Communication patterns
    increasing = [
        p for p in predictions.values()
        if p.get("message_volume", {}).get("trend") == "increasing"
    ]
    if increasing:
        findings.append(f"Communication increasing with {len(increasing)} contact(s)")
    
    return findings


def _generate_predictive_recommendations(predictions: Dict[str, Any], anomalies: List[Dict]) -> List[Dict[str, str]]:
    """Generate actionable recommendations based on predictions."""
    recommendations = []
    
    # Handle anomalies first (high priority)
    for anomaly in anomalies:
        if anomaly["type"] == "unusual_silence" and anomaly["concern_level"] == "high":
            recommendations.append({
                "action": f"Reach out to contact {anomaly['contact_id'][:6]}...",
                "reason": f"Unusual {anomaly['days_silent']}-day silence detected",
                "priority": "high",
                "suggested_message": "Hey, just thinking of you! How have you been?"
            })
    
    # Handle weakening relationships
    for contact_id, pred in predictions.items():
        if pred.get("relationship_trajectory", {}).get("direction") == "weakening":
            recommendations.append({
                "action": f"Schedule quality time with {contact_id[:6]}...",
                "reason": "Communication frequency declining",
                "priority": "medium"
            })
    
    # Suggest optimization based on peak times
    peak_times = set()
    for pred in predictions.values():
        if "message_volume" in pred:
            peak_times.update(pred["message_volume"].get("peak_times", []))
    
    if peak_times:
        recommendations.append({
            "action": f"Schedule important conversations during {', '.join(peak_times)}",
            "reason": "These are your most active communication periods",
            "priority": "low"
        })
    
    return recommendations[:5]  # Limit to top 5 recommendations


def _parse_time_window(window_str: str) -> int:
    """Parse time window string to days."""
    window_str = window_str.lower()
    
    if "day" in window_str:
        days = int(''.join(filter(str.isdigit, window_str)) or 1)
    elif "week" in window_str:
        weeks = int(''.join(filter(str.isdigit, window_str)) or 1)
        days = weeks * 7
    elif "month" in window_str:
        months = int(''.join(filter(str.isdigit, window_str)) or 1)
        days = months * 30
    else:
        days = 30  # Default to 30 days
    
    return days


@register_tool(
    name="detect_anomalies",
    description="Identify unusual patterns in communication behavior"
)
async def detect_anomalies_tool(
    time_window: str = "30 days",
    anomaly_types: List[str] = None,
    sensitivity: str = "balanced"
) -> Dict[str, Any]:
    """
    Detect various types of anomalies in communication patterns.
    
    Args:
        time_window: Period to analyze for anomalies
        anomaly_types: Types to detect ["silence", "frequency", "sentiment", "topics"]
        sensitivity: Detection sensitivity ("low", "balanced", "high")
        
    Returns:
        Detected anomalies with context and recommendations
    """
    try:
        if anomaly_types is None:
            anomaly_types = ["silence", "frequency", "sentiment", "topics"]
        
        # Get database connection
        db = await get_database()
        
        # Parse time window
        days_to_analyze = _parse_time_window(time_window)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_to_analyze)
        
        # Get all active contacts
        contacts_result = await db.get_contacts(limit=100, offset=0)
        
        anomalies = []
        
        for contact in contacts_result.get("contacts", [])[:50]:  # Analyze top 50
            contact_id = contact.get("phone_number", contact.get("handle_id"))
            
            # Get messages for this contact
            messages_result = await db.get_messages_from_contact(
                phone_number=contact_id,
                start_date=start_date - timedelta(days=90),  # Get more history for baseline
                end_date=end_date,
                page=1, page_size=500
            )
            
            messages = messages_result.get("messages", [])
            
            if len(messages) < 20:  # Skip if insufficient data
                continue
            
            # Detect different types of anomalies
            if "silence" in anomaly_types:
                silence_anomalies = _detect_silence_anomalies(messages, contact_id, sensitivity)
                anomalies.extend(silence_anomalies)
            
            if "frequency" in anomaly_types:
                frequency_anomalies = _detect_frequency_anomalies(messages, contact_id, sensitivity)
                anomalies.extend(frequency_anomalies)
            
            if "sentiment" in anomaly_types:
                sentiment_anomalies = _detect_sentiment_anomalies(messages, contact_id, sensitivity)
                anomalies.extend(sentiment_anomalies)
            
            if "topics" in anomaly_types:
                topic_anomalies = _detect_topic_anomalies(messages, contact_id, sensitivity)
                anomalies.extend(topic_anomalies)
        
        # Sort anomalies by severity
        anomalies.sort(key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x.get("severity", "low")))
        
        # Generate insights and recommendations
        insights = _generate_anomaly_insights(anomalies)
        recommendations = _generate_anomaly_recommendations(anomalies)
        
        return success_response({
            "anomalies": anomalies[:20],  # Limit to top 20
            "summary": {
                "total_detected": len(anomalies),
                "by_type": Counter(a["type"] for a in anomalies),
                "by_severity": Counter(a.get("severity", "low") for a in anomalies)
            },
            "insights": insights,
            "recommendations": recommendations,
            "metadata": {
                "time_window": time_window,
                "sensitivity": sensitivity,
                "types_analyzed": anomaly_types
            }
        })
        
    except Exception as e:
        logger.error(f"Error in detect_anomalies_tool: {e}")
        logger.error(traceback.format_exc())
        return error_response(ToolExecutionError(f"Failed to detect anomalies: {str(e)}"))


def _detect_silence_anomalies(messages: List[Dict], contact_id: str, sensitivity: str) -> List[Dict]:
    """Detect unusual periods of silence."""
    if not messages:
        return []
    
    # Calculate normal communication frequency
    daily_counts = _calculate_daily_message_counts(messages)
    if not daily_counts:
        return []
    
    avg_daily = statistics.mean(daily_counts)
    
    # Check current silence
    last_message_date = datetime.fromisoformat(messages[-1]["date"])
    days_silent = (datetime.now() - last_message_date).days
    
    # Set thresholds based on sensitivity
    thresholds = {
        "low": {"multiplier": 5, "min_days": 14},
        "balanced": {"multiplier": 3, "min_days": 7},
        "high": {"multiplier": 2, "min_days": 3}
    }
    
    threshold = thresholds[sensitivity]
    expected_gap = 1 / max(avg_daily, 0.1)  # Days between messages
    
    anomalies = []
    
    if days_silent > max(expected_gap * threshold["multiplier"], threshold["min_days"]):
        anomalies.append({
            "type": "silence",
            "contact_id": contact_id,
            "description": f"Unusual {days_silent}-day silence",
            "last_contact": last_message_date.isoformat(),
            "expected_frequency": f"Every {expected_gap:.1f} days",
            "severity": "high" if days_silent > expected_gap * 5 else "medium",
            "context": {
                "historical_average": f"{avg_daily:.1f} messages/day",
                "days_silent": days_silent
            }
        })
    
    return anomalies


def _detect_frequency_anomalies(messages: List[Dict], contact_id: str, sensitivity: str) -> List[Dict]:
    """Detect sudden changes in message frequency."""
    daily_counts = _calculate_daily_message_counts(messages)
    if len(daily_counts) < 30:
        return []
    
    anomalies = []
    
    # Compare recent to historical
    recent_counts = daily_counts[-14:]  # Last 2 weeks
    historical_counts = daily_counts[-90:-14]  # Previous period
    
    if not historical_counts:
        return []
    
    recent_avg = statistics.mean(recent_counts)
    historical_avg = statistics.mean(historical_counts)
    
    # Calculate change percentage
    if historical_avg > 0:
        change_pct = ((recent_avg - historical_avg) / historical_avg) * 100
        
        # Set thresholds based on sensitivity
        thresholds = {
            "low": 70,
            "balanced": 50,
            "high": 30
        }
        
        if abs(change_pct) > thresholds[sensitivity]:
            anomalies.append({
                "type": "frequency_change",
                "contact_id": contact_id,
                "description": f"Message frequency {'increased' if change_pct > 0 else 'decreased'} by {abs(change_pct):.0f}%",
                "direction": "increase" if change_pct > 0 else "decrease",
                "severity": "high" if abs(change_pct) > 100 else "medium",
                "context": {
                    "recent_average": f"{recent_avg:.1f} messages/day",
                    "historical_average": f"{historical_avg:.1f} messages/day",
                    "change_percentage": f"{change_pct:+.0f}%"
                }
            })
    
    return anomalies


def _detect_sentiment_anomalies(messages: List[Dict], contact_id: str, sensitivity: str) -> List[Dict]:
    """Detect unusual changes in emotional tone."""
    # Simplified sentiment detection
    recent_messages = [m for m in messages[-50:] if m.get("text")]
    historical_messages = [m for m in messages[-200:-50] if m.get("text")]
    
    if len(recent_messages) < 10 or len(historical_messages) < 20:
        return []
    
    def calculate_sentiment_score(msgs):
        positive_words = {"love", "great", "awesome", "happy", "excited", "wonderful", "lol", "haha"}
        negative_words = {"sad", "angry", "frustrated", "upset", "worried", "stressed", "sorry", "hate"}
        
        positive_count = 0
        negative_count = 0
        
        for msg in msgs:
            text_lower = msg["text"].lower()
            words = set(text_lower.split())
            positive_count += len(words & positive_words)
            negative_count += len(words & negative_words)
        
        total = positive_count + negative_count
        if total == 0:
            return 0
        return (positive_count - negative_count) / total
    
    recent_sentiment = calculate_sentiment_score(recent_messages)
    historical_sentiment = calculate_sentiment_score(historical_messages)
    
    sentiment_shift = recent_sentiment - historical_sentiment
    
    anomalies = []
    
    # Set thresholds based on sensitivity
    thresholds = {
        "low": 0.5,
        "balanced": 0.3,
        "high": 0.2
    }
    
    if abs(sentiment_shift) > thresholds[sensitivity]:
        anomalies.append({
            "type": "sentiment_shift",
            "contact_id": contact_id,
            "description": f"Emotional tone became more {'positive' if sentiment_shift > 0 else 'negative'}",
            "direction": "positive" if sentiment_shift > 0 else "negative",
            "severity": "medium",
            "context": {
                "sentiment_change": f"{sentiment_shift:+.2f}",
                "interpretation": "Significant emotional shift detected in recent conversations"
            }
        })
    
    return anomalies


def _detect_topic_anomalies(messages: List[Dict], contact_id: str, sensitivity: str) -> List[Dict]:
    """Detect unusual changes in conversation topics."""
    # Simplified topic detection
    recent_messages = [m for m in messages[-50:] if m.get("text")]
    historical_messages = [m for m in messages[-200:-50] if m.get("text")]
    
    if len(recent_messages) < 10 or len(historical_messages) < 20:
        return []
    
    def extract_topics(msgs):
        # Extract words longer than 4 characters as potential topics
        topics = []
        for msg in msgs:
            words = [w.lower() for w in msg["text"].split() if len(w) > 4 and w.isalpha()]
            topics.extend(words)
        return Counter(topics).most_common(10)
    
    recent_topics = dict(extract_topics(recent_messages))
    historical_topics = dict(extract_topics(historical_messages))
    
    # Find new topics
    new_topics = set(recent_topics.keys()) - set(historical_topics.keys())
    disappeared_topics = set(historical_topics.keys()) - set(recent_topics.keys())
    
    anomalies = []
    
    if len(new_topics) >= 3:  # Multiple new topics
        anomalies.append({
            "type": "topic_shift",
            "contact_id": contact_id,
            "description": "Significant change in conversation topics",
            "new_topics": list(new_topics)[:5],
            "severity": "low",
            "context": {
                "interpretation": "Conversations have shifted to new subjects recently"
            }
        })
    
    return anomalies


def _generate_anomaly_insights(anomalies: List[Dict]) -> Dict[str, Any]:
    """Generate insights from detected anomalies."""
    if not anomalies:
        return {"status": "no_anomalies", "message": "All communication patterns appear normal"}
    
    # Group by type
    by_type = defaultdict(list)
    for anomaly in anomalies:
        by_type[anomaly["type"]].append(anomaly)
    
    insights = {
        "key_patterns": [],
        "risk_assessment": {
            "relationships_at_risk": len(set(a["contact_id"] for a in anomalies if a.get("severity") == "high")),
            "immediate_attention_needed": len([a for a in anomalies if a["type"] == "silence" and a.get("severity") == "high"])
        }
    }
    
    if by_type["silence"]:
        insights["key_patterns"].append(
            f"{len(by_type['silence'])} contacts have gone unusually quiet"
        )
    
    if by_type["frequency_change"]:
        increases = [a for a in by_type["frequency_change"] if a["direction"] == "increase"]
        decreases = [a for a in by_type["frequency_change"] if a["direction"] == "decrease"]
        if increases:
            insights["key_patterns"].append(f"Communication increasing with {len(increases)} contacts")
        if decreases:
            insights["key_patterns"].append(f"Communication decreasing with {len(decreases)} contacts")
    
    return insights


def _generate_anomaly_recommendations(anomalies: List[Dict]) -> List[Dict[str, str]]:
    """Generate recommendations based on detected anomalies."""
    recommendations = []
    
    # Priority 1: Address high-severity silence
    silence_anomalies = [a for a in anomalies if a["type"] == "silence" and a.get("severity") == "high"]
    for anomaly in silence_anomalies[:3]:  # Top 3
        recommendations.append({
            "action": f"Reach out to contact immediately",
            "reason": anomaly["description"],
            "priority": "high",
            "contact_id": anomaly["contact_id"]
        })
    
    # Priority 2: Address negative sentiment shifts
    sentiment_anomalies = [a for a in anomalies if a["type"] == "sentiment_shift" and a["direction"] == "negative"]
    for anomaly in sentiment_anomalies[:2]:
        recommendations.append({
            "action": "Check in with emotional support",
            "reason": "Detected negative emotional shift in conversations",
            "priority": "medium",
            "contact_id": anomaly["contact_id"]
        })
    
    # Priority 3: Address frequency decreases
    frequency_anomalies = [a for a in anomalies if a["type"] == "frequency_change" and a["direction"] == "decrease"]
    for anomaly in frequency_anomalies[:2]:
        recommendations.append({
            "action": "Re-engage with quality conversation",
            "reason": anomaly["description"],
            "priority": "medium",
            "contact_id": anomaly["contact_id"]
        })
    
    return recommendations[:5]  # Return top 5