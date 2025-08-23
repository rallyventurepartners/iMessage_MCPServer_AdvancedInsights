"""
Predictive Engagement tool for forecasting communication patterns.

Uses machine learning to predict future engagement levels, optimal contact times,
and potential relationship changes.
"""

import asyncio
import logging
import statistics
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from imessage_mcp_server.db import get_database
from imessage_mcp_server.privacy import apply_privacy_filters, hash_contact_id

logger = logging.getLogger(__name__)


async def predictive_engagement_tool(
    contact_id: str,
    prediction_type: str = "comprehensive",
    horizon_days: int = 30,
    include_recommendations: bool = True,
    db_path: str = "~/Library/Messages/chat.db",
    redact: bool = True,
) -> Dict[str, Any]:
    """
    Predict future engagement patterns using ML.
    
    Features:
    - Response time predictions
    - Activity level forecasting
    - Sentiment trajectory prediction
    - Risk of communication breakdown
    - Optimal engagement strategies
    
    Args:
        contact_id: Contact identifier to analyze
        prediction_type: Type of prediction ("comprehensive", "response_time", "activity", "sentiment", "risk")
        horizon_days: Days to predict into the future
        include_recommendations: Whether to include actionable recommendations
        db_path: Path to iMessage database
        redact: Whether to apply privacy filters
        
    Returns:
        Dict containing predictions and recommendations
    """
    try:
        # Parse horizon
        if horizon_days > 90:
            horizon_days = 90  # Cap at 90 days
        
        # Expand path
        db_path = Path(db_path).expanduser()
        db = await get_database(str(db_path))
        
        # Fetch historical data (need at least 90 days for good predictions)
        messages = await _fetch_contact_messages(db, contact_id, days=180)
        
        if len(messages) < 50:
            return {
                "error": "Insufficient message history for predictions",
                "contact_id": hash_contact_id(contact_id) if redact else contact_id,
                "message_count": len(messages),
                "required_minimum": 50,
            }
        
        # Extract features for ML
        features = await _extract_ml_features(messages)
        
        # Build predictions based on type
        predictions = {}
        
        if prediction_type in ["comprehensive", "response_time"]:
            predictions["response_time"] = await _predict_response_times(
                features, horizon_days
            )
        
        if prediction_type in ["comprehensive", "activity"]:
            predictions["activity_level"] = await _predict_activity_levels(
                features, messages, horizon_days
            )
        
        if prediction_type in ["comprehensive", "sentiment"]:
            predictions["sentiment_trajectory"] = await _predict_sentiment(
                features, messages, horizon_days
            )
        
        if prediction_type in ["comprehensive", "risk"]:
            predictions["relationship_risk"] = await _assess_relationship_risk(
                features, messages
            )
        
        # Calculate engagement score trajectory
        engagement_forecast = _forecast_engagement_score(predictions, features)
        
        # Generate optimal engagement strategies
        strategies = _generate_engagement_strategies(
            predictions, features, messages
        ) if include_recommendations else None
        
        # Identify key inflection points
        inflection_points = _identify_inflection_points(predictions)
        
        # Build result
        result = {
            "contact_id": hash_contact_id(contact_id) if redact else contact_id,
            "prediction_type": prediction_type,
            "horizon_days": horizon_days,
            "predictions": predictions,
            "engagement_forecast": engagement_forecast,
            "inflection_points": inflection_points,
            "confidence_metrics": _calculate_confidence_metrics(features, messages),
            "analysis_period": {
                "start": messages[0]["date"].isoformat() if messages else None,
                "end": messages[-1]["date"].isoformat() if messages else None,
            },
        }
        
        if strategies:
            result["engagement_strategies"] = strategies
        
        # Apply redaction if requested
        if redact:
            result = apply_privacy_filters(result)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in predictive engagement analysis: {e}")
        return {
            "error": str(e),
            "error_type": "prediction_error",
        }


async def _fetch_contact_messages(
    db: Any, contact_id: str, days: int
) -> List[Dict[str, Any]]:
    """Fetch messages with a contact."""
    query = """
    SELECT 
        m.text,
        m.is_from_me,
        m.date,
        h.id as handle_id
    FROM message m
    JOIN handle h ON m.handle_id = h.ROWID
    WHERE h.id = ?
    AND m.text IS NOT NULL
    AND m.date > (strftime('%s', 'now') - ?) * 1000000000
    ORDER BY m.date
    """
    
    seconds = days * 86400
    cursor = await db.execute(query, [contact_id, seconds])
    rows = await cursor.fetchall()
    
    messages = []
    for row in rows:
        messages.append({
            "text": row[0],
            "is_from_me": row[1],
            "date": datetime.fromtimestamp(row[2] / 1000000000 + 978307200),
            "length": len(row[0]),
        })
    
    return messages


async def _extract_ml_features(messages: List[Dict]) -> Dict[str, np.ndarray]:
    """Extract features for machine learning models."""
    features = defaultdict(list)
    
    # Time-based features
    for i, msg in enumerate(messages):
        features["hour"].append(msg["date"].hour)
        features["day_of_week"].append(msg["date"].weekday())
        features["day_of_month"].append(msg["date"].day)
        features["is_weekend"].append(1 if msg["date"].weekday() >= 5 else 0)
        features["is_from_me"].append(1 if msg["is_from_me"] else 0)
        features["message_length"].append(msg["length"])
        
        # Response time (if not first message)
        if i > 0:
            time_diff = (msg["date"] - messages[i-1]["date"]).total_seconds()
            features["time_since_last"].append(time_diff)
        else:
            features["time_since_last"].append(0)
        
        # Conversation context
        if i > 0 and messages[i-1]["is_from_me"] != msg["is_from_me"]:
            features["is_response"].append(1)
        else:
            features["is_response"].append(0)
    
    # Convert to numpy arrays
    for key in features:
        features[key] = np.array(features[key])
    
    # Add rolling averages
    window_sizes = [7, 14, 30]
    for window in window_sizes:
        if len(messages) >= window:
            # Message frequency
            daily_counts = _calculate_daily_message_counts(messages)
            if len(daily_counts) >= window:
                rolling_avg = []
                for i in range(len(daily_counts)):
                    start = max(0, i - window + 1)
                    avg = np.mean(list(daily_counts.values())[start:i+1])
                    rolling_avg.append(avg)
                features[f"rolling_avg_{window}d"] = np.array(rolling_avg[-len(messages):])
    
    return features


async def _predict_response_times(
    features: Dict[str, np.ndarray], horizon_days: int
) -> Dict[str, Any]:
    """Predict future response times using Random Forest."""
    # Prepare training data
    response_indices = np.where(features["is_response"] == 1)[0]
    if len(response_indices) < 20:
        return {
            "status": "insufficient_data",
            "message": "Not enough response data for predictions",
        }
    
    # Features for response time prediction
    feature_names = ["hour", "day_of_week", "is_weekend"]
    X = np.column_stack([features[name][response_indices] for name in feature_names])
    y = features["time_since_last"][response_indices]
    
    # Remove outliers (responses > 24 hours)
    valid_indices = y < 86400
    X = X[valid_indices]
    y = y[valid_indices]
    
    if len(y) < 10:
        return {
            "status": "insufficient_data",
            "message": "Not enough valid response data",
        }
    
    # Train model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
    model.fit(X_scaled, y)
    
    # Generate predictions for next N days
    predictions_by_hour = {}
    for hour in range(24):
        for dow in range(7):
            is_weekend = 1 if dow >= 5 else 0
            X_pred = scaler.transform([[hour, dow, is_weekend]])
            pred_seconds = model.predict(X_pred)[0]
            
            key = f"hour_{hour}_dow_{dow}"
            predictions_by_hour[key] = {
                "predicted_response_time_minutes": round(pred_seconds / 60, 1),
                "hour": hour,
                "day_of_week": dow,
                "is_weekend": bool(is_weekend),
            }
    
    # Find best and worst times
    sorted_predictions = sorted(
        predictions_by_hour.items(),
        key=lambda x: x[1]["predicted_response_time_minutes"]
    )
    
    best_times = [
        {
            "hour": p[1]["hour"],
            "day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][p[1]["day_of_week"]],
            "response_time_minutes": p[1]["predicted_response_time_minutes"],
        }
        for p in sorted_predictions[:5]
    ]
    
    worst_times = [
        {
            "hour": p[1]["hour"],
            "day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][p[1]["day_of_week"]],
            "response_time_minutes": p[1]["predicted_response_time_minutes"],
        }
        for p in sorted_predictions[-5:]
    ]
    
    return {
        "model_score": round(model.score(X_scaled, y), 3),
        "average_response_time_minutes": round(np.mean(y) / 60, 1),
        "best_times_to_contact": best_times,
        "worst_times_to_contact": worst_times,
        "prediction_confidence": _calculate_prediction_confidence(model, X_scaled, y),
    }


async def _predict_activity_levels(
    features: Dict[str, np.ndarray], messages: List[Dict], horizon_days: int
) -> Dict[str, Any]:
    """Predict future activity levels."""
    # Calculate daily message counts
    daily_counts = _calculate_daily_message_counts(messages)
    
    if len(daily_counts) < 30:
        return {
            "status": "insufficient_data",
            "message": "Need at least 30 days of history",
        }
    
    # Convert to time series
    dates = sorted(daily_counts.keys())
    counts = [daily_counts[d] for d in dates]
    
    # Simple time series features
    X = []
    y = []
    window = 7
    
    for i in range(window, len(counts)):
        # Features: last 7 days of counts + day of week
        features_window = counts[i-window:i]
        dow = dates[i].weekday()
        X.append(features_window + [dow])
        y.append(counts[i])
    
    X = np.array(X)
    y = np.array(y)
    
    if len(y) < 20:
        return {
            "status": "insufficient_data",
            "message": "Not enough data points for time series prediction",
        }
    
    # Train model
    model = RandomForestRegressor(n_estimators=30, random_state=42, max_depth=5)
    model.fit(X, y)
    
    # Predict next N days
    predictions = []
    last_counts = counts[-window:]
    current_date = dates[-1]
    
    for day in range(horizon_days):
        future_date = current_date + timedelta(days=day+1)
        dow = future_date.weekday()
        
        X_pred = np.array([last_counts + [dow]])
        pred_count = max(0, model.predict(X_pred)[0])
        
        predictions.append({
            "date": future_date.isoformat(),
            "predicted_messages": round(pred_count, 1),
            "day_of_week": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][dow],
        })
        
        # Update rolling window
        last_counts = last_counts[1:] + [pred_count]
    
    # Calculate trends
    recent_avg = np.mean(counts[-30:])
    predicted_avg = np.mean([p["predicted_messages"] for p in predictions])
    trend = "increasing" if predicted_avg > recent_avg * 1.1 else "decreasing" if predicted_avg < recent_avg * 0.9 else "stable"
    
    return {
        "current_daily_average": round(recent_avg, 1),
        "predicted_daily_average": round(predicted_avg, 1),
        "trend": trend,
        "predictions": predictions[:7],  # First week for clarity
        "weekly_pattern": _analyze_weekly_pattern(daily_counts),
        "model_score": round(model.score(X, y), 3),
    }


async def _predict_sentiment(
    features: Dict[str, np.ndarray], messages: List[Dict], horizon_days: int
) -> Dict[str, Any]:
    """Predict sentiment trajectory."""
    # Simple sentiment analysis based on message characteristics
    sentiments = []
    
    for msg in messages:
        # Basic sentiment indicators
        text = msg["text"].lower()
        
        positive_indicators = ["!", "😊", "😂", "❤️", "👍", "haha", "lol", "love", "great", "awesome", "thanks"]
        negative_indicators = ["?", "😢", "😡", "👎", "sorry", "sad", "angry", "frustrated", "annoyed"]
        
        positive_score = sum(1 for ind in positive_indicators if ind in text)
        negative_score = sum(1 for ind in negative_indicators if ind in text)
        
        # Length-based adjustment (longer messages often more thoughtful)
        length_factor = min(msg["length"] / 100, 2.0)
        
        sentiment = (positive_score - negative_score + length_factor) / (positive_score + negative_score + 1)
        sentiments.append({
            "date": msg["date"],
            "sentiment": sentiment,
            "is_from_me": msg["is_from_me"],
        })
    
    # Aggregate by week
    weekly_sentiments = defaultdict(list)
    for s in sentiments:
        week_start = s["date"] - timedelta(days=s["date"].weekday())
        weekly_sentiments[week_start].append(s["sentiment"])
    
    # Calculate weekly averages
    weekly_avg = {
        week: np.mean(scores) for week, scores in weekly_sentiments.items()
    }
    
    if len(weekly_avg) < 4:
        return {
            "status": "insufficient_data",
            "message": "Need at least 4 weeks of data",
        }
    
    # Simple trend prediction
    weeks = sorted(weekly_avg.keys())
    values = [weekly_avg[w] for w in weeks]
    
    # Linear regression for trend
    x = np.arange(len(values))
    slope, intercept = np.polyfit(x, values, 1)
    
    # Project forward
    future_weeks = []
    for i in range(horizon_days // 7 + 1):
        future_x = len(values) + i
        future_sentiment = slope * future_x + intercept
        future_week = weeks[-1] + timedelta(weeks=i+1)
        
        future_weeks.append({
            "week_start": future_week.isoformat(),
            "predicted_sentiment": round(future_sentiment, 3),
            "trend": "improving" if slope > 0.01 else "declining" if slope < -0.01 else "stable",
        })
    
    return {
        "current_sentiment": round(values[-1], 3),
        "sentiment_trend": "improving" if slope > 0.01 else "declining" if slope < -0.01 else "stable",
        "weekly_predictions": future_weeks,
        "confidence": "high" if abs(slope) > 0.05 else "medium" if abs(slope) > 0.02 else "low",
        "key_factors": _identify_sentiment_factors(sentiments, messages),
    }


async def _assess_relationship_risk(
    features: Dict[str, np.ndarray], messages: List[Dict]
) -> Dict[str, Any]:
    """Assess risk of relationship deterioration."""
    risk_factors = {
        "declining_frequency": 0,
        "increasing_response_time": 0,
        "sentiment_decline": 0,
        "conversation_imbalance": 0,
        "long_silence_periods": 0,
    }
    
    # 1. Check frequency decline
    daily_counts = _calculate_daily_message_counts(messages)
    if len(daily_counts) >= 60:
        first_month = list(daily_counts.values())[:30]
        last_month = list(daily_counts.values())[-30:]
        
        if np.mean(last_month) < np.mean(first_month) * 0.5:
            risk_factors["declining_frequency"] = 0.3
    
    # 2. Check response time increase
    response_times = features["time_since_last"][features["is_response"] == 1]
    if len(response_times) >= 20:
        first_half = response_times[:len(response_times)//2]
        second_half = response_times[len(response_times)//2:]
        
        if np.median(second_half) > np.median(first_half) * 2:
            risk_factors["increasing_response_time"] = 0.25
    
    # 3. Check sentiment decline (simplified)
    message_lengths = [m["length"] for m in messages]
    if len(message_lengths) >= 40:
        first_half_avg = np.mean(message_lengths[:len(message_lengths)//2])
        second_half_avg = np.mean(message_lengths[len(message_lengths)//2:])
        
        if second_half_avg < first_half_avg * 0.6:
            risk_factors["sentiment_decline"] = 0.2
    
    # 4. Check conversation balance
    my_messages = sum(1 for m in messages if m["is_from_me"])
    their_messages = len(messages) - my_messages
    balance_ratio = min(my_messages, their_messages) / max(my_messages, their_messages)
    
    if balance_ratio < 0.3:
        risk_factors["conversation_imbalance"] = 0.15
    
    # 5. Check for long silences
    silence_periods = []
    for i in range(1, len(messages)):
        gap = (messages[i]["date"] - messages[i-1]["date"]).days
        if gap > 7:
            silence_periods.append(gap)
    
    if len(silence_periods) > 3 and max(silence_periods) > 14:
        risk_factors["long_silence_periods"] = 0.1
    
    # Calculate overall risk
    total_risk = sum(risk_factors.values())
    risk_level = "high" if total_risk > 0.5 else "medium" if total_risk > 0.25 else "low"
    
    # Generate mitigation strategies
    mitigation = []
    if risk_factors["declining_frequency"] > 0:
        mitigation.append("Increase communication frequency to previous levels")
    if risk_factors["increasing_response_time"] > 0:
        mitigation.append("Respond more promptly to maintain engagement")
    if risk_factors["sentiment_decline"] > 0:
        mitigation.append("Share more meaningful, longer messages")
    if risk_factors["conversation_imbalance"] > 0:
        mitigation.append("Balance conversation participation")
    if risk_factors["long_silence_periods"] > 0:
        mitigation.append("Avoid extended periods without contact")
    
    return {
        "risk_level": risk_level,
        "risk_score": round(total_risk, 2),
        "risk_factors": {k: round(v, 2) for k, v in risk_factors.items() if v > 0},
        "mitigation_strategies": mitigation,
        "early_warning_signs": _identify_warning_signs(risk_factors, messages),
    }


def _calculate_daily_message_counts(messages: List[Dict]) -> Dict[datetime, int]:
    """Calculate message count per day."""
    daily_counts = defaultdict(int)
    
    for msg in messages:
        day = msg["date"].date()
        daily_counts[day] += 1
    
    # Fill in missing days with 0
    if daily_counts:
        start_date = min(daily_counts.keys())
        end_date = max(daily_counts.keys())
        current = start_date
        
        while current <= end_date:
            if current not in daily_counts:
                daily_counts[current] = 0
            current += timedelta(days=1)
    
    return dict(sorted(daily_counts.items()))


def _forecast_engagement_score(
    predictions: Dict[str, Any], features: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    """Forecast overall engagement score."""
    current_score = 0.5  # Base score
    
    # Adjust based on predictions
    if "response_time" in predictions and predictions["response_time"].get("model_score", 0) > 0:
        # Better response times = higher engagement
        avg_response = predictions["response_time"].get("average_response_time_minutes", 60)
        if avg_response < 30:
            current_score += 0.2
        elif avg_response > 120:
            current_score -= 0.2
    
    if "activity_level" in predictions and predictions["activity_level"].get("trend"):
        if predictions["activity_level"]["trend"] == "increasing":
            current_score += 0.15
        elif predictions["activity_level"]["trend"] == "decreasing":
            current_score -= 0.15
    
    if "sentiment_trajectory" in predictions and predictions["sentiment_trajectory"].get("sentiment_trend"):
        if predictions["sentiment_trajectory"]["sentiment_trend"] == "improving":
            current_score += 0.15
        elif predictions["sentiment_trajectory"]["sentiment_trend"] == "declining":
            current_score -= 0.15
    
    # Cap between 0 and 1
    current_score = max(0, min(1, current_score))
    
    # Project forward
    trend = "stable"
    if current_score > 0.65:
        trend = "positive"
    elif current_score < 0.35:
        trend = "concerning"
    
    return {
        "current_score": round(current_score, 2),
        "predicted_score_30d": round(current_score * 0.9, 2),  # Conservative estimate
        "trend": trend,
        "confidence": "medium",
    }


def _generate_engagement_strategies(
    predictions: Dict[str, Any], features: Dict[str, np.ndarray], messages: List[Dict]
) -> Dict[str, Any]:
    """Generate optimal engagement strategies."""
    strategies = {
        "timing": [],
        "content": [],
        "frequency": [],
        "warnings": [],
    }
    
    # Timing strategies
    if "response_time" in predictions and "best_times_to_contact" in predictions["response_time"]:
        best_times = predictions["response_time"]["best_times_to_contact"][:3]
        for bt in best_times:
            strategies["timing"].append(
                f"Contact on {bt['day']} around {bt['hour']}:00 for fastest response"
            )
    
    # Content strategies based on sentiment
    if "sentiment_trajectory" in predictions:
        sentiment_trend = predictions["sentiment_trajectory"].get("sentiment_trend", "stable")
        if sentiment_trend == "declining":
            strategies["content"].append("Share more positive, engaging content")
            strategies["content"].append("Ask open-ended questions to deepen conversations")
        elif sentiment_trend == "improving":
            strategies["content"].append("Continue current communication style")
    
    # Frequency strategies
    if "activity_level" in predictions:
        current_avg = predictions["activity_level"].get("current_daily_average", 0)
        if current_avg < 1:
            strategies["frequency"].append("Increase contact frequency to daily")
        elif current_avg > 10:
            strategies["frequency"].append("Current high frequency is sustainable")
    
    # Risk warnings
    if "relationship_risk" in predictions:
        risk_level = predictions["relationship_risk"].get("risk_level", "low")
        if risk_level in ["medium", "high"]:
            strategies["warnings"] = predictions["relationship_risk"].get("mitigation_strategies", [])
    
    return strategies


def _identify_inflection_points(predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Identify key moments where patterns might change."""
    inflection_points = []
    
    # Activity level inflections
    if "activity_level" in predictions and "predictions" in predictions["activity_level"]:
        activity_preds = predictions["activity_level"]["predictions"]
        if len(activity_preds) >= 3:
            # Look for significant changes
            for i in range(1, len(activity_preds) - 1):
                prev = activity_preds[i-1]["predicted_messages"]
                curr = activity_preds[i]["predicted_messages"]
                next_val = activity_preds[i+1]["predicted_messages"]
                
                if (curr > prev * 1.5 and curr > next_val * 1.5) or \
                   (curr < prev * 0.5 and curr < next_val * 0.5):
                    inflection_points.append({
                        "date": activity_preds[i]["date"],
                        "type": "activity_spike" if curr > prev else "activity_drop",
                        "significance": "high" if abs(curr - prev) > 5 else "medium",
                    })
    
    return inflection_points


def _calculate_confidence_metrics(
    features: Dict[str, np.ndarray], messages: List[Dict]
) -> Dict[str, float]:
    """Calculate confidence in predictions."""
    confidence_factors = {
        "data_volume": min(len(messages) / 500, 1.0),  # More data = higher confidence
        "data_recency": 1.0,  # Assume recent data
        "pattern_consistency": 0.5,  # Default medium
    }
    
    # Check pattern consistency
    if len(messages) >= 100:
        daily_counts = list(_calculate_daily_message_counts(messages).values())
        if daily_counts:
            cv = statistics.stdev(daily_counts) / statistics.mean(daily_counts) if statistics.mean(daily_counts) > 0 else 1
            confidence_factors["pattern_consistency"] = max(0, 1 - cv / 2)
    
    overall_confidence = sum(confidence_factors.values()) / len(confidence_factors)
    
    return {
        "overall": round(overall_confidence, 2),
        "factors": {k: round(v, 2) for k, v in confidence_factors.items()},
    }


def _calculate_prediction_confidence(model: Any, X: np.ndarray, y: np.ndarray) -> str:
    """Calculate confidence level for predictions."""
    score = model.score(X, y)
    
    if score > 0.7:
        return "high"
    elif score > 0.4:
        return "medium"
    else:
        return "low"


def _analyze_weekly_pattern(daily_counts: Dict[datetime, int]) -> Dict[str, float]:
    """Analyze weekly communication patterns."""
    day_totals = defaultdict(list)
    
    for date, count in daily_counts.items():
        dow = date.weekday()
        day_totals[dow].append(count)
    
    weekly_pattern = {}
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    for dow, day_name in enumerate(days):
        if dow in day_totals:
            weekly_pattern[day_name] = round(np.mean(day_totals[dow]), 1)
        else:
            weekly_pattern[day_name] = 0.0
    
    return weekly_pattern


def _identify_sentiment_factors(
    sentiments: List[Dict], messages: List[Dict]
) -> List[str]:
    """Identify key factors affecting sentiment."""
    factors = []
    
    # Check message length correlation
    long_messages = [s for m, s in zip(messages, sentiments) if m["length"] > 100]
    short_messages = [s for m, s in zip(messages, sentiments) if m["length"] < 50]
    
    if long_messages and short_messages:
        if np.mean([s["sentiment"] for s in long_messages]) > \
           np.mean([s["sentiment"] for s in short_messages]) * 1.2:
            factors.append("Longer messages correlate with positive sentiment")
    
    # Check time of day patterns
    morning_sentiments = [s["sentiment"] for s in sentiments if 6 <= s["date"].hour < 12]
    evening_sentiments = [s["sentiment"] for s in sentiments if 18 <= s["date"].hour < 24]
    
    if morning_sentiments and evening_sentiments:
        if np.mean(morning_sentiments) > np.mean(evening_sentiments) * 1.1:
            factors.append("Morning conversations tend to be more positive")
        elif np.mean(evening_sentiments) > np.mean(morning_sentiments) * 1.1:
            factors.append("Evening conversations tend to be more positive")
    
    return factors


def _identify_warning_signs(
    risk_factors: Dict[str, float], messages: List[Dict]
) -> List[str]:
    """Identify early warning signs of relationship issues."""
    warning_signs = []
    
    if risk_factors.get("declining_frequency", 0) > 0:
        warning_signs.append("Message frequency has dropped by more than 50%")
    
    if risk_factors.get("increasing_response_time", 0) > 0:
        warning_signs.append("Response times have doubled recently")
    
    if risk_factors.get("sentiment_decline", 0) > 0:
        warning_signs.append("Messages are becoming shorter and less engaged")
    
    if risk_factors.get("long_silence_periods", 0) > 0:
        warning_signs.append("Multiple extended periods of no contact detected")
    
    return warning_signs