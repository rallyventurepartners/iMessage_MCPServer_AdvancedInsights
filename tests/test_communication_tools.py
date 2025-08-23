"""Tests for communication pattern analysis tools."""

import sys
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from imessage_mcp_server.tools.communication import (
    cadence_calendar_tool,
    conversation_topics_tool,
    response_time_distribution_tool,
    sentiment_evolution_tool,
)


@pytest.mark.asyncio
async def test_conversation_topics_tool(populated_db):
    """Test conversation topics extraction."""
    result = await conversation_topics_tool(
        db_path=populated_db,
        contact_id=None,
        days=180,
        min_frequency=2,
        redact=True
    )

    assert "topics" in result
    assert "emerging_keywords" in result
    assert "total_messages_analyzed" in result
    assert result["total_messages_analyzed"] > 0

    # Check redaction
    if result["topics"]:
        assert all(kw == "[REDACTED]" for topic in result["topics"] for kw in topic["keywords"])


@pytest.mark.asyncio
async def test_conversation_topics_with_contact(populated_db):
    """Test conversation topics for specific contact."""
    result = await conversation_topics_tool(
        db_path=populated_db,
        contact_id="+15551234567",
        days=180,
        min_frequency=1,
        redact=False
    )

    assert "contact_id" in result
    assert result["contact_id"] == "+15551234567"
    assert "topics" in result


@pytest.mark.asyncio
async def test_sentiment_evolution_tool(populated_db):
    """Test sentiment tracking over time."""
    result = await sentiment_evolution_tool(
        db_path=populated_db,
        contact_id=None,
        days=180,
        bucket_days=7,
        redact=True
    )

    assert "series" in result
    assert "summary" in result
    assert "mean" in result["summary"]
    assert "trend" in result["summary"]
    assert "volatility" in result["summary"]

    # Check series data
    if result["series"]:
        for point in result["series"]:
            assert "date" in point
            assert "score" in point
            assert "messages" in point
            assert -1 <= point["score"] <= 1


@pytest.mark.asyncio
async def test_response_time_distribution(populated_db):
    """Test response time analysis."""
    result = await response_time_distribution_tool(
        db_path=populated_db,
        contact_id="+15551234567",
        days=180,
        redact=True
    )

    assert "your_response_times" in result
    assert "their_response_times" in result

    for key in ["your_response_times", "their_response_times"]:
        times = result[key]
        assert "median_minutes" in times
        assert "p25_minutes" in times
        assert "p75_minutes" in times
        assert "p95_minutes" in times

        # Check ordering
        assert times["p25_minutes"] <= times["median_minutes"]
        assert times["median_minutes"] <= times["p75_minutes"]
        assert times["p75_minutes"] <= times["p95_minutes"]


@pytest.mark.asyncio
async def test_cadence_calendar_tool(populated_db):
    """Test communication frequency heatmap."""
    result = await cadence_calendar_tool(
        db_path=populated_db,
        contact_id=None,
        days=180,
        redact=True
    )

    assert "heatmap" in result
    assert "peak_hours" in result
    assert "peak_days" in result
    assert "total_messages" in result

    # Check heatmap structure
    days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    for day in days:
        assert day in result["heatmap"]
        assert isinstance(result["heatmap"][day], dict)

    # Check peak hours format
    for hour in result["peak_hours"]:
        assert ":" in hour
        assert hour.endswith(":00")


@pytest.mark.asyncio
async def test_error_handling_invalid_db():
    """Test error handling with invalid database path."""
    result = await conversation_topics_tool(
        db_path="/invalid/path/to/db",
        contact_id=None,
        days=90,
        min_frequency=3,
        redact=True
    )

    assert "error" in result
    assert "error_type" in result
    assert result["error_type"] == "analysis_error"


@pytest.mark.asyncio
async def test_empty_results(temp_db):
    """Test handling of empty database."""
    result = await sentiment_evolution_tool(
        db_path=temp_db,
        contact_id=None,
        days=30,
        bucket_days=7,
        redact=True
    )

    assert "series" in result
    assert result["series"] == []
    assert result["summary"]["mean"] == 0
    assert result["summary"]["trend"] == "no_data"
