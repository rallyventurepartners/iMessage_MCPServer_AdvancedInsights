"""
Test suite for group dynamics analysis tools.
"""

import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from imessage_mcp_server.tools.group_dynamics import (
    group_dynamics_tool,
    _analyze_participation,
    _analyze_influence,
    _detect_subgroups,
    _assess_group_health,
    _extract_participants,
    _build_interaction_matrix,
    _determine_group_personality,
    _generate_group_insights,
)


# Fixtures
@pytest.fixture
def mock_db():
    """Create a mock database connection."""
    db = Mock()
    db.execute = AsyncMock()
    return db


@pytest.fixture
def sample_group_messages():
    """Create sample group messages for testing."""
    base_time = datetime.now()
    messages = []
    
    # Create a realistic conversation flow
    participants = ["me", "user_alice", "user_bob", "user_charlie", "user_diana"]
    
    # Simulate various conversation patterns
    for day in range(30):
        for hour in [9, 12, 15, 18, 21]:
            for i, sender in enumerate(participants):
                if (day + hour + i) % 3 == 0:  # Skip some to create patterns
                    continue
                    
                msg_time = base_time - timedelta(days=day, hours=23-hour, minutes=i*5)
                messages.append({
                    "text": f"Message from {sender} on day {day}",
                    "is_from_me": sender == "me",
                    "date": msg_time,
                    "sender_id": sender,
                    "length": 50 + (i * 10),
                })
    
    return sorted(messages, key=lambda x: x["date"])


@pytest.fixture
def sample_interaction_matrix():
    """Create sample interaction matrix."""
    participants = {"me", "user_alice", "user_bob", "user_charlie", "user_diana"}
    matrix = {p1: {p2: 0 for p2 in participants} for p1 in participants}
    
    # Add some interactions
    matrix["me"]["user_alice"] = 15
    matrix["user_alice"]["me"] = 12
    matrix["me"]["user_bob"] = 8
    matrix["user_bob"]["me"] = 10
    matrix["user_alice"]["user_bob"] = 5
    matrix["user_bob"]["user_alice"] = 6
    matrix["user_charlie"]["user_diana"] = 20
    matrix["user_diana"]["user_charlie"] = 18
    
    return matrix


class TestGroupDynamicsTool:
    """Test the main group dynamics tool."""
    
    @pytest.mark.asyncio
    async def test_group_dynamics_comprehensive(self, mock_db):
        """Test comprehensive group dynamics analysis."""
        # Mock database results
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=[
            ("Hello everyone!", 0, 1700000000000000000, 1, "user_alice", "group_123"),
            ("Hi Alice!", 1, 1700000060000000000, 0, None, "group_123"),
            ("How's everyone doing?", 0, 1700000120000000000, 2, "user_bob", "group_123"),
            ("Great, thanks!", 0, 1700000180000000000, 3, "user_charlie", "group_123"),
            ("Doing well!", 1, 1700000240000000000, 0, None, "group_123"),
        ])
        mock_db.execute.return_value = mock_cursor
        
        with patch("imessage_mcp_server.tools.group_dynamics.get_database", 
                   AsyncMock(return_value=mock_db)):
            result = await group_dynamics_tool(
                group_id="group_123",
                analysis_type="comprehensive",
                time_period="30d",
                redact=True
            )
        
        assert result["group_id"] != "group_123"  # Should be hashed
        assert result["participant_count"] == 4
        assert result["message_count"] == 5
        assert "dynamics" in result
        assert "participation" in result["dynamics"]
        assert "influence" in result["dynamics"]
        assert "subgroups" in result["dynamics"]
        assert "health" in result["dynamics"]
        assert "insights" in result
    
    @pytest.mark.asyncio
    async def test_group_dynamics_participation_only(self, mock_db):
        """Test participation-focused analysis."""
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=[
            ("Message 1", 0, 1700000000000000000, 1, "user_alice", "group_123"),
            ("Message 2", 1, 1700000060000000000, 0, None, "group_123"),
            ("Message 3", 0, 1700000120000000000, 1, "user_alice", "group_123"),
        ])
        mock_db.execute.return_value = mock_cursor
        
        with patch("imessage_mcp_server.tools.group_dynamics.get_database", 
                   AsyncMock(return_value=mock_db)):
            result = await group_dynamics_tool(
                group_id="group_123",
                analysis_type="participation",
                time_period="7d"
            )
        
        assert "participation" in result["dynamics"]
        assert "influence" not in result["dynamics"]
        assert "subgroups" not in result["dynamics"]
    
    @pytest.mark.asyncio
    async def test_group_dynamics_no_messages(self, mock_db):
        """Test handling of empty group."""
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=[])
        mock_db.execute.return_value = mock_cursor
        
        with patch("imessage_mcp_server.tools.group_dynamics.get_database", 
                   AsyncMock(return_value=mock_db)):
            result = await group_dynamics_tool(
                group_id="empty_group",
                analysis_type="comprehensive"
            )
        
        assert "error" in result
        assert "No messages found" in result["error"]
    
    @pytest.mark.asyncio
    async def test_group_dynamics_error_handling(self, mock_db):
        """Test error handling."""
        mock_db.execute.side_effect = Exception("Database error")
        
        with patch("imessage_mcp_server.tools.group_dynamics.get_database", 
                   AsyncMock(return_value=mock_db)):
            result = await group_dynamics_tool(
                group_id="group_123",
                analysis_type="comprehensive"
            )
        
        assert "error" in result
        assert result["error_type"] == "group_analysis_error"


class TestParticipationAnalysis:
    """Test participation analysis functions."""
    
    @pytest.mark.asyncio
    async def test_analyze_participation(self, sample_group_messages):
        """Test participation pattern analysis."""
        participants = _extract_participants(sample_group_messages)
        result = await _analyze_participation(sample_group_messages, participants)
        
        assert "distribution" in result
        assert "balance_score" in result
        assert "roles" in result
        assert "peak_hours" in result
        
        # Check distribution metrics
        for participant, data in result["distribution"].items():
            assert "message_count" in data
            assert "percentage" in data
            assert "avg_message_length" in data
            assert "active_hours" in data
            assert "conversation_starts" in data
        
        # Check roles
        assert result["roles"]["most_active"] is not None
        assert isinstance(result["roles"]["conversation_starters"], list)
    
    def test_extract_participants(self, sample_group_messages):
        """Test participant extraction."""
        participants = _extract_participants(sample_group_messages)
        
        assert isinstance(participants, set)
        assert "me" in participants
        assert len(participants) == 5  # Based on sample data


class TestInfluenceAnalysis:
    """Test influence network analysis."""
    
    @pytest.mark.asyncio
    async def test_analyze_influence(self, sample_group_messages, sample_interaction_matrix):
        """Test influence network analysis."""
        participants = _extract_participants(sample_group_messages)
        result = await _analyze_influence(
            sample_group_messages, 
            participants, 
            sample_interaction_matrix
        )
        
        assert "metrics" in result
        assert "key_members" in result
        assert "network_density" in result
        
        # Check metrics for each participant
        for participant in participants:
            metrics = result["metrics"][participant]
            assert "receives_responses" in metrics
            assert "initiates_responses" in metrics
            assert "bridge_score" in metrics
            assert "influence_score" in metrics
        
        # Check key members
        assert isinstance(result["key_members"]["influencers"], list)
        assert isinstance(result["key_members"]["connectors"], list)
        assert isinstance(result["key_members"]["conversation_drivers"], list)
    
    def test_build_interaction_matrix(self, sample_group_messages):
        """Test interaction matrix building."""
        participants = _extract_participants(sample_group_messages)
        matrix = _build_interaction_matrix(sample_group_messages, participants)
        
        assert isinstance(matrix, dict)
        for p1 in participants:
            assert p1 in matrix
            for p2 in participants:
                assert p2 in matrix[p1]
                assert isinstance(matrix[p1][p2], int)
                assert matrix[p1][p2] >= 0


class TestSubgroupDetection:
    """Test subgroup detection functionality."""
    
    @pytest.mark.asyncio
    async def test_detect_subgroups(self, sample_interaction_matrix):
        """Test subgroup detection with spectral clustering."""
        participants = set(sample_interaction_matrix.keys())
        result = await _detect_subgroups(sample_interaction_matrix, participants)
        
        assert "clusters" in result
        assert "modularity" in result
        assert "description" in result
        
        # Check cluster properties
        for cluster in result["clusters"]:
            assert "id" in cluster
            assert "members" in cluster
            assert "size" in cluster
            assert "cohesion" in cluster
            assert "type" in cluster
            assert cluster["size"] == len(cluster["members"])
    
    @pytest.mark.asyncio
    async def test_detect_subgroups_small_group(self):
        """Test subgroup detection with too few participants."""
        participants = {"user1", "user2"}
        matrix = {"user1": {"user1": 0, "user2": 5}, 
                  "user2": {"user1": 3, "user2": 0}}
        
        result = await _detect_subgroups(matrix, participants)
        
        assert result["clusters"] == []
        assert "Too few participants" in result["description"]


class TestGroupHealthAssessment:
    """Test group health assessment."""
    
    @pytest.mark.asyncio
    async def test_assess_group_health(self, sample_group_messages, sample_interaction_matrix):
        """Test group health metrics calculation."""
        participants = _extract_participants(sample_group_messages)
        result = await _assess_group_health(
            sample_group_messages,
            participants,
            sample_interaction_matrix
        )
        
        assert "health_score" in result
        assert 0 <= result["health_score"] <= 100
        
        assert "activity_level" in result
        assert "daily_average" in result["activity_level"]
        assert "rating" in result["activity_level"]
        
        assert "inclusivity_score" in result
        assert "responsiveness_score" in result
        assert "sentiment_score" in result
        assert "engagement_trend" in result
        assert "health_indicators" in result
        
        # Check indicators
        assert isinstance(result["health_indicators"], list)
        assert len(result["health_indicators"]) > 0


class TestGroupPersonality:
    """Test group personality determination."""
    
    def test_determine_group_personality(self):
        """Test group personality classification."""
        # Test collaborative active group
        analysis_results = {
            "participation": {"balance_score": 0.8},
            "health": {
                "activity_level": {"rating": "high"},
                "inclusivity_score": 75
            }
        }
        personality = _determine_group_personality(analysis_results)
        assert personality == "collaborative_active"
        
        # Test hierarchical formal group
        analysis_results = {
            "participation": {"balance_score": 0.3},
            "health": {
                "activity_level": {"rating": "moderate"},
                "inclusivity_score": 45
            }
        }
        personality = _determine_group_personality(analysis_results)
        assert personality == "hierarchical_formal"
        
        # Test empty results
        personality = _determine_group_personality({})
        assert personality == "unknown"


class TestGroupInsights:
    """Test insight generation."""
    
    def test_generate_group_insights(self):
        """Test actionable insight generation."""
        analysis_results = {
            "participation": {
                "balance_score": 0.3,
                "roles": {"conversation_starters": ["user1", "user2"]}
            },
            "influence": {
                "key_members": {"influencers": ["user1"]},
                "network_density": 0.25
            },
            "subgroups": {
                "clusters": [
                    {"size": 3, "cohesion": 0.85},
                    {"size": 2, "cohesion": 0.9}
                ]
            },
            "health": {
                "health_score": 45,
                "engagement_trend": "decreasing"
            }
        }
        
        participants = {"user1", "user2", "user3", "user4", "user5"}
        insights = _generate_group_insights(analysis_results, participants)
        
        assert "patterns" in insights
        assert "recommendations" in insights
        assert "strengths" in insights
        assert "concerns" in insights
        
        # Check specific insights based on data
        assert any("dominated by few members" in c for c in insights["concerns"])
        assert any("encourage quieter members" in r for r in insights["recommendations"])
        assert any("declining" in c for c in insights["concerns"])


class TestIntegration:
    """Integration tests for group dynamics tool."""
    
    @pytest.mark.asyncio
    async def test_full_analysis_flow(self, mock_db):
        """Test complete analysis flow with realistic data."""
        # Create more realistic message data
        base_time = 1700000000000000000
        messages = []
        
        # Simulate a week of group chat activity
        for day in range(7):
            for hour in range(24):
                if 8 <= hour <= 22:  # Active hours
                    for i, sender in enumerate(["user_alice", "me", "user_bob", "user_charlie"]):
                        if (day * 24 + hour + i) % 3 != 0:  # Some randomness
                            messages.append((
                                f"Message content {day}-{hour}-{i}",
                                1 if sender == "me" else 0,
                                base_time + (day * 86400 + hour * 3600 + i * 300) * 1000000000,
                                i if sender != "me" else 0,
                                sender if sender != "me" else None,
                                "group_work_chat"
                            ))
        
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=messages)
        mock_db.execute.return_value = mock_cursor
        
        with patch("imessage_mcp_server.tools.group_dynamics.get_database", 
                   AsyncMock(return_value=mock_db)):
            result = await group_dynamics_tool(
                group_id="group_work_chat",
                analysis_type="comprehensive",
                time_period="7d",
                redact=True
            )
        
        # Comprehensive checks
        assert result["participant_count"] == 4
        assert result["message_count"] == len(messages)
        assert result["group_personality"] in [
            "collaborative_active", "collaborative_casual", 
            "hierarchical_formal", "dominated_active", "mixed_moderate"
        ]
        
        # Check all analysis components
        dynamics = result["dynamics"]
        assert dynamics["participation"]["balance_score"] > 0
        assert len(dynamics["participation"]["distribution"]) == 4
        assert dynamics["influence"]["network_density"] > 0
        assert len(dynamics["subgroups"]["clusters"]) >= 0
        assert 0 <= dynamics["health"]["health_score"] <= 100
        
        # Check insights
        insights = result["insights"]
        assert len(insights["patterns"]) > 0
        assert isinstance(insights["recommendations"], list)
        assert isinstance(insights["strengths"], list)
        assert isinstance(insights["concerns"], list)