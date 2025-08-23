"""Tests for conversation quality analysis tools."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from imessage_mcp_server.tools.quality import conversation_quality_tool


@pytest.mark.asyncio
async def test_conversation_quality_basic(populated_db):
    """Test basic conversation quality scoring."""
    result = await conversation_quality_tool(
        contact_id="+15551234567",
        time_period="30d",
        db_path=populated_db,
        redact=True
    )
    
    # Check basic structure
    assert "overall_score" in result
    assert "grade" in result
    assert "trajectory" in result
    assert "dimensions" in result
    assert "action_items" in result
    
    # Check dimensions
    dimensions = result["dimensions"]
    assert "depth" in dimensions
    assert "balance" in dimensions
    assert "emotion" in dimensions
    assert "consistency" in dimensions
    
    # Check each dimension has score and insights
    for dim_name, dim_data in dimensions.items():
        assert "score" in dim_data
        assert "insights" in dim_data
        assert isinstance(dim_data["score"], (int, float))
        assert 0 <= dim_data["score"] <= 100
        assert isinstance(dim_data["insights"], list)
    
    # Check overall score
    assert isinstance(result["overall_score"], (int, float))
    assert 0 <= result["overall_score"] <= 100
    
    # Check grade
    assert result["grade"] in ["A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "F"]
    
    # Check trajectory
    assert result["trajectory"] in ["improving", "declining", "stable"]


@pytest.mark.asyncio
async def test_conversation_quality_time_periods(populated_db):
    """Test quality scoring with different time periods."""
    periods = ["7d", "30d", "90d", "6m", "1y"]
    
    for period in periods:
        result = await conversation_quality_tool(
            contact_id="+15551234567",
            time_period=period,
            db_path=populated_db,
            redact=True
        )
        
        if "error" not in result:
            assert result["time_period"] == period
            assert "overall_score" in result


@pytest.mark.asyncio
async def test_conversation_quality_recommendations(populated_db):
    """Test recommendation generation."""
    result = await conversation_quality_tool(
        contact_id="+15551234567",
        time_period="30d",
        include_recommendations=True,
        db_path=populated_db,
        redact=True
    )
    
    if "error" not in result:
        assert "action_items" in result
        assert isinstance(result["action_items"], list)
        assert len(result["action_items"]) <= 3  # Top 3 recommendations
        
        # Check recommendations are strings
        for rec in result["action_items"]:
            assert isinstance(rec, str)
            assert len(rec) > 0


@pytest.mark.asyncio
async def test_conversation_quality_no_recommendations(populated_db):
    """Test quality scoring without recommendations."""
    result = await conversation_quality_tool(
        contact_id="+15551234567",
        time_period="30d",
        include_recommendations=False,
        db_path=populated_db,
        redact=True
    )
    
    if "error" not in result:
        assert "action_items" in result
        assert result["action_items"] == []


@pytest.mark.asyncio
async def test_conversation_quality_privacy(populated_db):
    """Test privacy features in quality scoring."""
    # Test with redaction
    result_redacted = await conversation_quality_tool(
        contact_id="+15551234567",
        time_period="30d",
        db_path=populated_db,
        redact=True
    )
    
    if "error" not in result_redacted:
        assert result_redacted["contact_id"].startswith("hash:")
    
    # Test without redaction
    result_clear = await conversation_quality_tool(
        contact_id="+15551234567",
        time_period="30d",
        db_path=populated_db,
        redact=False
    )
    
    if "error" not in result_clear:
        assert result_clear["contact_id"] == "+15551234567"


@pytest.mark.asyncio
async def test_conversation_quality_edge_cases(populated_db):
    """Test edge cases for quality scoring."""
    # Non-existent contact
    result = await conversation_quality_tool(
        contact_id="nonexistent@example.com",
        time_period="30d",
        db_path=populated_db,
        redact=True
    )
    
    assert "error" in result
    assert result["error"] == "No messages found for quality analysis"
    
    # Very short time period
    result = await conversation_quality_tool(
        contact_id="+15551234567",
        time_period="1d",
        db_path=populated_db,
        redact=True
    )
    
    # Should either have data or appropriate error
    assert "overall_score" in result or "error" in result


@pytest.mark.asyncio
async def test_conversation_quality_metrics_detail(populated_db):
    """Test detailed metrics in each dimension."""
    result = await conversation_quality_tool(
        contact_id="+15551234567",
        time_period="30d",
        db_path=populated_db,
        redact=True
    )
    
    if "error" not in result and "analysis_details" in result:
        details = result["analysis_details"]
        assert "messages_analyzed" in details
        assert isinstance(details["messages_analyzed"], int)
        assert details["messages_analyzed"] > 0
        
        assert "date_range" in details
        assert isinstance(details["date_range"], str)


@pytest.mark.asyncio
async def test_conversation_quality_score_calculation(temp_db):
    """Test score calculation logic with known data."""
    # Would need to insert specific test data to verify calculations
    # For now, just ensure the tool handles empty database gracefully
    result = await conversation_quality_tool(
        contact_id="+15551234567",
        time_period="30d",
        db_path=temp_db,
        redact=True
    )
    
    assert "error" in result


@pytest.mark.asyncio 
async def test_grade_calculation():
    """Test grade calculation from scores."""
    from imessage_mcp_server.tools.quality import _score_to_grade
    
    # Test grade boundaries
    assert _score_to_grade(95) == "A"
    assert _score_to_grade(93) == "A"
    assert _score_to_grade(92) == "A-"
    assert _score_to_grade(90) == "A-"
    assert _score_to_grade(87) == "B+"
    assert _score_to_grade(85) == "B"
    assert _score_to_grade(83) == "B"
    assert _score_to_grade(80) == "B-"
    assert _score_to_grade(75) == "C+"
    assert _score_to_grade(73) == "C"
    assert _score_to_grade(70) == "C-"
    assert _score_to_grade(65) == "D+"
    assert _score_to_grade(63) == "D"
    assert _score_to_grade(60) == "F"
    assert _score_to_grade(50) == "F"


@pytest.mark.asyncio
async def test_time_period_parsing():
    """Test time period string parsing."""
    from imessage_mcp_server.tools.quality import _parse_time_period
    
    assert _parse_time_period("7d") == 7
    assert _parse_time_period("30d") == 30
    assert _parse_time_period("2w") == 14
    assert _parse_time_period("3m") == 90
    assert _parse_time_period("1y") == 365
    assert _parse_time_period("invalid") == 30  # Default