"""
Test suite for predictive engagement tools.
"""

import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from imessage_mcp_server.tools.predictive_engagement import (
    predictive_engagement_tool,
    _extract_ml_features,
    _predict_response_times,
    _predict_activity_levels,
    _predict_sentiment,
    _assess_relationship_risk,
    _calculate_daily_message_counts,
    _forecast_engagement_score,
    _generate_engagement_strategies,
    _identify_inflection_points,
)


# Fixtures
@pytest.fixture
def mock_db():
    """Create a mock database connection."""
    db = Mock()
    db.execute = AsyncMock()
    return db


@pytest.fixture
def sample_messages_180d():
    """Create 180 days of realistic message history."""
    messages = []
    base_time = datetime.now()
    
    # Create realistic conversation patterns
    for day in range(180):
        date = base_time - timedelta(days=180-day)
        
        # Vary message frequency by day of week
        if date.weekday() < 5:  # Weekday
            num_messages = np.random.poisson(5)
        else:  # Weekend
            num_messages = np.random.poisson(3)
        
        # Create messages throughout the day
        for i in range(num_messages):
            hour = np.random.choice([9, 10, 11, 14, 15, 16, 19, 20, 21], p=[0.1, 0.1, 0.1, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1])
            minute = np.random.randint(0, 60)
            
            msg_time = date.replace(hour=hour, minute=minute)
            is_from_me = bool(np.random.binomial(1, 0.45))  # Slightly imbalanced
            
            # Vary message length
            if np.random.random() < 0.3:  # Short messages
                length = np.random.randint(5, 50)
                text = "Hey!" if length < 10 else "How are you doing today?"
            elif np.random.random() < 0.7:  # Medium messages
                length = np.random.randint(50, 150)
                text = "That sounds great! I was thinking the same thing. Let's definitely do that."
            else:  # Long messages
                length = np.random.randint(150, 300)
                text = "I've been thinking about what you said earlier and I completely agree. " * 3
            
            messages.append({
                "text": text[:length],
                "is_from_me": is_from_me,
                "date": msg_time,
                "length": length,
            })
    
    return sorted(messages, key=lambda x: x["date"])


@pytest.fixture
def sample_messages_sparse():
    """Create sparse message history for risk detection."""
    messages = []
    base_time = datetime.now()
    
    # First 90 days: active
    for day in range(90):
        date = base_time - timedelta(days=180-day)
        for i in range(np.random.poisson(8)):
            hour = np.random.randint(9, 22)
            messages.append({
                "text": "Active period message",
                "is_from_me": bool(np.random.binomial(1, 0.5)),
                "date": date.replace(hour=hour),
                "length": np.random.randint(50, 150),
            })
    
    # Last 90 days: declining
    for day in range(90, 180):
        date = base_time - timedelta(days=180-day)
        if np.random.random() < 0.3:  # Only 30% chance of messages
            for i in range(np.random.poisson(2)):
                hour = np.random.randint(9, 22)
                messages.append({
                    "text": "Declining period",
                    "is_from_me": bool(np.random.binomial(1, 0.7)),  # More one-sided
                    "date": date.replace(hour=hour),
                    "length": np.random.randint(10, 50),  # Shorter
                })
    
    return sorted(messages, key=lambda x: x["date"])


class TestPredictiveEngagementTool:
    """Test the main predictive engagement tool."""
    
    @pytest.mark.asyncio
    async def test_predictive_engagement_comprehensive(self, mock_db, sample_messages_180d):
        """Test comprehensive predictive analysis."""
        # Mock database results
        mock_cursor = AsyncMock()
        rows = []
        for msg in sample_messages_180d:
            timestamp = int((msg["date"].timestamp() - 978307200) * 1000000000)
            rows.append((
                msg["text"],
                1 if msg["is_from_me"] else 0,
                timestamp,
                "contact@example.com"
            ))
        mock_cursor.fetchall = AsyncMock(return_value=rows)
        mock_db.execute.return_value = mock_cursor
        
        with patch("imessage_mcp_server.tools.predictive_engagement.get_database", 
                   AsyncMock(return_value=mock_db)):
            result = await predictive_engagement_tool(
                contact_id="contact@example.com",
                prediction_type="comprehensive",
                horizon_days=30,
                include_recommendations=True,
                redact=True
            )
        
        assert "predictions" in result
        assert "engagement_forecast" in result
        assert "inflection_points" in result
        assert "confidence_metrics" in result
        assert "engagement_strategies" in result
        
        # Check all prediction types are included
        predictions = result["predictions"]
        assert "response_time" in predictions
        assert "activity_level" in predictions
        assert "sentiment_trajectory" in predictions
        assert "relationship_risk" in predictions
    
    @pytest.mark.asyncio
    async def test_predictive_engagement_insufficient_data(self, mock_db):
        """Test handling of insufficient data."""
        # Only 20 messages (below threshold of 50)
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=[
            ("Message", 1, 1700000000000000000, "contact@example.com")
            for _ in range(20)
        ])
        mock_db.execute.return_value = mock_cursor
        
        with patch("imessage_mcp_server.tools.predictive_engagement.get_database", 
                   AsyncMock(return_value=mock_db)):
            result = await predictive_engagement_tool(
                contact_id="contact@example.com",
                prediction_type="comprehensive"
            )
        
        assert "error" in result
        assert "Insufficient message history" in result["error"]
        assert result["message_count"] == 20
        assert result["required_minimum"] == 50
    
    @pytest.mark.asyncio
    async def test_predictive_engagement_response_time_only(self, mock_db, sample_messages_180d):
        """Test response time prediction only."""
        mock_cursor = AsyncMock()
        rows = []
        for msg in sample_messages_180d[:100]:  # Use subset
            timestamp = int((msg["date"].timestamp() - 978307200) * 1000000000)
            rows.append((
                msg["text"],
                1 if msg["is_from_me"] else 0,
                timestamp,
                "contact@example.com"
            ))
        mock_cursor.fetchall = AsyncMock(return_value=rows)
        mock_db.execute.return_value = mock_cursor
        
        with patch("imessage_mcp_server.tools.predictive_engagement.get_database", 
                   AsyncMock(return_value=mock_db)):
            result = await predictive_engagement_tool(
                contact_id="contact@example.com",
                prediction_type="response_time",
                horizon_days=7
            )
        
        assert "predictions" in result
        assert "response_time" in result["predictions"]
        assert "activity_level" not in result["predictions"]
        assert "sentiment_trajectory" not in result["predictions"]


class TestFeatureExtraction:
    """Test ML feature extraction."""
    
    @pytest.mark.asyncio
    async def test_extract_ml_features(self, sample_messages_180d):
        """Test feature extraction from messages."""
        features = await _extract_ml_features(sample_messages_180d[:50])
        
        # Check basic features
        assert "hour" in features
        assert "day_of_week" in features
        assert "day_of_month" in features
        assert "is_weekend" in features
        assert "is_from_me" in features
        assert "message_length" in features
        assert "time_since_last" in features
        assert "is_response" in features
        
        # Check array lengths
        num_messages = len(sample_messages_180d[:50])
        assert len(features["hour"]) == num_messages
        assert len(features["day_of_week"]) == num_messages
        
        # Check value ranges
        assert all(0 <= h <= 23 for h in features["hour"])
        assert all(0 <= d <= 6 for d in features["day_of_week"])
        assert all(v in [0, 1] for v in features["is_weekend"])
    
    def test_calculate_daily_message_counts(self, sample_messages_180d):
        """Test daily message count calculation."""
        daily_counts = _calculate_daily_message_counts(sample_messages_180d[:30])
        
        assert isinstance(daily_counts, dict)
        assert len(daily_counts) > 0
        
        # Check all values are non-negative
        assert all(count >= 0 for count in daily_counts.values())
        
        # Check dates are consecutive
        dates = sorted(daily_counts.keys())
        for i in range(1, len(dates)):
            assert (dates[i] - dates[i-1]).days <= 1


class TestResponseTimePrediction:
    """Test response time prediction functionality."""
    
    @pytest.mark.asyncio
    async def test_predict_response_times(self, sample_messages_180d):
        """Test response time prediction with ML."""
        features = await _extract_ml_features(sample_messages_180d)
        result = await _predict_response_times(features, horizon_days=30)
        
        if result.get("status") != "insufficient_data":
            assert "model_score" in result
            assert "average_response_time_minutes" in result
            assert "best_times_to_contact" in result
            assert "worst_times_to_contact" in result
            assert "prediction_confidence" in result
            
            # Check best times structure
            for time_slot in result["best_times_to_contact"]:
                assert "hour" in time_slot
                assert "day" in time_slot
                assert "response_time_minutes" in time_slot
                assert 0 <= time_slot["hour"] <= 23
    
    @pytest.mark.asyncio
    async def test_predict_response_times_insufficient_data(self):
        """Test response time prediction with insufficient data."""
        # Create minimal features
        features = {
            "hour": np.array([10, 11, 12]),
            "day_of_week": np.array([1, 2, 3]),
            "is_weekend": np.array([0, 0, 0]),
            "is_response": np.array([0, 0, 1]),  # Only 1 response
            "time_since_last": np.array([0, 300, 600]),
        }
        
        result = await _predict_response_times(features, horizon_days=30)
        
        assert result["status"] == "insufficient_data"
        assert "message" in result


class TestActivityPrediction:
    """Test activity level prediction."""
    
    @pytest.mark.asyncio
    async def test_predict_activity_levels(self, sample_messages_180d):
        """Test activity level forecasting."""
        features = await _extract_ml_features(sample_messages_180d)
        result = await _predict_activity_levels(features, sample_messages_180d, horizon_days=30)
        
        if result.get("status") != "insufficient_data":
            assert "current_daily_average" in result
            assert "predicted_daily_average" in result
            assert "trend" in result
            assert "predictions" in result
            assert "weekly_pattern" in result
            assert "model_score" in result
            
            # Check predictions structure
            for pred in result["predictions"]:
                assert "date" in pred
                assert "predicted_messages" in pred
                assert "day_of_week" in pred
                assert pred["predicted_messages"] >= 0
            
            # Check weekly pattern
            assert len(result["weekly_pattern"]) == 7
            assert all(day in result["weekly_pattern"] for day in 
                      ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])


class TestSentimentPrediction:
    """Test sentiment trajectory prediction."""
    
    @pytest.mark.asyncio
    async def test_predict_sentiment(self, sample_messages_180d):
        """Test sentiment prediction."""
        features = await _extract_ml_features(sample_messages_180d)
        result = await _predict_sentiment(features, sample_messages_180d, horizon_days=30)
        
        if result.get("status") != "insufficient_data":
            assert "current_sentiment" in result
            assert "sentiment_trend" in result
            assert "weekly_predictions" in result
            assert "confidence" in result
            assert "key_factors" in result
            
            # Check trend values
            assert result["sentiment_trend"] in ["improving", "declining", "stable"]
            assert result["confidence"] in ["high", "medium", "low"]
            
            # Check predictions
            for pred in result["weekly_predictions"]:
                assert "week_start" in pred
                assert "predicted_sentiment" in pred
                assert "trend" in pred


class TestRiskAssessment:
    """Test relationship risk assessment."""
    
    @pytest.mark.asyncio
    async def test_assess_relationship_risk_healthy(self, sample_messages_180d):
        """Test risk assessment for healthy relationship."""
        features = await _extract_ml_features(sample_messages_180d)
        result = await _assess_relationship_risk(features, sample_messages_180d)
        
        assert "risk_level" in result
        assert "risk_score" in result
        assert "risk_factors" in result
        assert "mitigation_strategies" in result
        assert "early_warning_signs" in result
        
        assert result["risk_level"] in ["low", "medium", "high"]
        assert 0 <= result["risk_score"] <= 1
    
    @pytest.mark.asyncio
    async def test_assess_relationship_risk_declining(self, sample_messages_sparse):
        """Test risk assessment for declining relationship."""
        features = await _extract_ml_features(sample_messages_sparse)
        result = await _assess_relationship_risk(features, sample_messages_sparse)
        
        # Should detect higher risk due to declining pattern
        assert result["risk_level"] in ["medium", "high"]
        assert result["risk_score"] > 0.25
        assert len(result["risk_factors"]) > 0
        assert len(result["mitigation_strategies"]) > 0


class TestEngagementForecasting:
    """Test engagement score forecasting."""
    
    def test_forecast_engagement_score(self):
        """Test engagement score calculation."""
        predictions = {
            "response_time": {
                "model_score": 0.8,
                "average_response_time_minutes": 25
            },
            "activity_level": {
                "trend": "increasing"
            },
            "sentiment_trajectory": {
                "sentiment_trend": "improving"
            }
        }
        
        features = {"dummy": np.array([1, 2, 3])}
        result = _forecast_engagement_score(predictions, features)
        
        assert "current_score" in result
        assert "predicted_score_30d" in result
        assert "trend" in result
        assert "confidence" in result
        
        assert 0 <= result["current_score"] <= 1
        assert 0 <= result["predicted_score_30d"] <= 1
        assert result["trend"] in ["positive", "stable", "concerning"]


class TestEngagementStrategies:
    """Test engagement strategy generation."""
    
    def test_generate_engagement_strategies(self):
        """Test strategy generation based on predictions."""
        predictions = {
            "response_time": {
                "best_times_to_contact": [
                    {"hour": 10, "day": "Tuesday", "response_time_minutes": 15},
                    {"hour": 14, "day": "Wednesday", "response_time_minutes": 20},
                ]
            },
            "sentiment_trajectory": {
                "sentiment_trend": "declining"
            },
            "activity_level": {
                "current_daily_average": 0.5
            },
            "relationship_risk": {
                "risk_level": "medium",
                "mitigation_strategies": ["Increase communication frequency"]
            }
        }
        
        features = {}
        messages = []
        
        strategies = _generate_engagement_strategies(predictions, features, messages)
        
        assert "timing" in strategies
        assert "content" in strategies
        assert "frequency" in strategies
        assert "warnings" in strategies
        
        # Check specific strategies
        assert len(strategies["timing"]) > 0
        assert any("Tuesday" in s for s in strategies["timing"])
        assert any("positive" in s for s in strategies["content"])
        assert any("daily" in s for s in strategies["frequency"])


class TestInflectionPoints:
    """Test inflection point identification."""
    
    def test_identify_inflection_points(self):
        """Test finding key moments in predictions."""
        predictions = {
            "activity_level": {
                "predictions": [
                    {"date": "2025-01-01", "predicted_messages": 5},
                    {"date": "2025-01-02", "predicted_messages": 12},  # Spike
                    {"date": "2025-01-03", "predicted_messages": 4},
                    {"date": "2025-01-04", "predicted_messages": 3},
                    {"date": "2025-01-05", "predicted_messages": 1},  # Drop
                    {"date": "2025-01-06", "predicted_messages": 5},
                ]
            }
        }
        
        inflection_points = _identify_inflection_points(predictions)
        
        assert len(inflection_points) > 0
        
        for point in inflection_points:
            assert "date" in point
            assert "type" in point
            assert "significance" in point
            assert point["type"] in ["activity_spike", "activity_drop"]
            assert point["significance"] in ["high", "medium"]


class TestIntegration:
    """Integration tests for predictive engagement."""
    
    @pytest.mark.asyncio
    async def test_full_prediction_flow(self, mock_db):
        """Test complete prediction flow with all components."""
        # Create comprehensive test data
        base_time = datetime.now()
        messages = []
        
        # Simulate 6 months of evolving relationship
        for day in range(180):
            date = base_time - timedelta(days=180-day)
            
            # Early phase: high activity
            if day < 60:
                num_messages = np.random.poisson(8)
            # Middle phase: stable
            elif day < 120:
                num_messages = np.random.poisson(5)
            # Recent phase: declining
            else:
                num_messages = np.random.poisson(2)
            
            for i in range(num_messages):
                hour = np.random.choice([9, 12, 15, 18, 21])
                is_from_me = np.random.binomial(1, 0.5 if day < 120 else 0.7)
                
                messages.append((
                    f"Message on day {day}",
                    is_from_me,
                    int((date.replace(hour=hour).timestamp() - 978307200) * 1000000000),
                    "test@example.com"
                ))
        
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=messages)
        mock_db.execute.return_value = mock_cursor
        
        with patch("imessage_mcp_server.tools.predictive_engagement.get_database", 
                   AsyncMock(return_value=mock_db)):
            result = await predictive_engagement_tool(
                contact_id="test@example.com",
                prediction_type="comprehensive",
                horizon_days=30,
                include_recommendations=True,
                redact=False  # Don't redact for testing
            )
        
        # Comprehensive validation
        assert result["contact_id"] == "test@example.com"
        assert result["prediction_type"] == "comprehensive"
        assert result["horizon_days"] == 30
        
        # Check all predictions are present
        predictions = result["predictions"]
        assert len(predictions) == 4  # All 4 prediction types
        
        # Verify risk detection (should be medium/high due to decline)
        risk = predictions["relationship_risk"]
        assert risk["risk_level"] in ["medium", "high"]
        assert len(risk["mitigation_strategies"]) > 0
        
        # Check engagement strategies
        strategies = result["engagement_strategies"]
        assert any(len(strategies[key]) > 0 for key in strategies)
        
        # Verify confidence metrics
        confidence = result["confidence_metrics"]
        assert 0 <= confidence["overall"] <= 1