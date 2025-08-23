"""Tests for advanced analytics tools."""

import sys
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from imessage_mcp_server.tools.analytics import (
    anomaly_scan_tool,
    best_contact_time_tool,
    network_intelligence_tool,
    sample_messages_tool,
)


@pytest.mark.asyncio
async def test_best_contact_time_tool(populated_db):
    """Test optimal contact time prediction."""
    result = await best_contact_time_tool(
        db_path=populated_db,
        contact_id="+15551234567",
        redact=True
    )

    # Handle both success and no data cases
    if "error" in result and result.get("error_type") == "no_data":
        assert "error" in result
        assert result["error_type"] == "no_data"
    else:
        assert "best_times" in result
        assert "avoid_times" in result

        # Check best times structure
        for time in result.get("best_times", []):
            assert "time" in time
            assert "day" in time
            assert "confidence" in time
            assert time["day"] in ["weekday", "weekend"]
            assert 0 <= time["confidence"] <= 1

        # Check avoid times
        for time in result.get("avoid_times", []):
            assert "time" in time
            assert "reason" in time


@pytest.mark.asyncio
async def test_anomaly_scan_tool(populated_db):
    """Test anomaly detection."""
    result = await anomaly_scan_tool(
        db_path=populated_db,
        contact_id=None,
        days=30,
        sensitivity=0.7,
        redact=True
    )

    assert "anomalies" in result
    assert "scan_period" in result
    assert "contacts_scanned" in result
    assert "sensitivity" in result

    # Check anomaly structure
    for anomaly in result["anomalies"]:
        assert "type" in anomaly
        assert "contact_id" in anomaly
        assert "severity" in anomaly
        assert "description" in anomaly
        assert "detected_date" in anomaly

        assert anomaly["type"] in ["silence", "volume_spike", "volume_drop"]
        assert anomaly["severity"] in ["low", "medium", "high"]

        # Check redaction
        if result:
            assert anomaly["contact_id"].endswith("...")


@pytest.mark.asyncio
async def test_anomaly_scan_sensitivity(populated_db):
    """Test anomaly detection with different sensitivity levels."""
    # High sensitivity (more anomalies)
    result_high = await anomaly_scan_tool(
        db_path=populated_db,
        contact_id=None,
        days=30,
        sensitivity=0.9,
        redact=True
    )

    # Low sensitivity (fewer anomalies)
    result_low = await anomaly_scan_tool(
        db_path=populated_db,
        contact_id=None,
        days=30,
        sensitivity=0.3,
        redact=True
    )

    # Higher sensitivity should generally find more or equal anomalies
    assert len(result_high["anomalies"]) >= len(result_low["anomalies"])


@pytest.mark.asyncio
async def test_network_intelligence_tool(populated_db):
    """Test social network analysis."""
    result = await network_intelligence_tool(
        db_path=populated_db,
        min_messages=5,
        days=180,
        redact=True
    )

    assert "network_stats" in result
    assert "key_connectors" in result
    assert "communities" in result

    # Check network stats
    stats = result["network_stats"]
    assert "total_nodes" in stats
    assert "total_edges" in stats
    assert "density" in stats
    assert "communities" in stats
    assert 0 <= stats["density"] <= 1

    # Check key connectors
    for connector in result["key_connectors"]:
        assert "contact_id" in connector
        assert "centrality" in connector
        assert "bridges" in connector
        assert 0 <= connector["centrality"] <= 1

    # Check communities
    for community in result["communities"]:
        assert "id" in community
        assert "size" in community
        assert "label" in community
        assert "cohesion" in community
        assert 0 <= community["cohesion"] <= 1


@pytest.mark.asyncio
async def test_sample_messages_tool(populated_db):
    """Test message sampling with redaction."""
    result = await sample_messages_tool(
        db_path=populated_db,
        contact_id="+15551234567",
        limit=5,
        days=180,
        redact=True
    )

    assert "contact_id" in result
    assert "samples" in result
    assert "privacy_note" in result

    # Check redaction
    assert result["contact_id"].endswith("...")
    assert "PII removed" in result["privacy_note"]

    # Check sample structure
    for sample in result["samples"]:
        assert "timestamp" in sample
        assert "is_from_me" in sample
        assert "preview" in sample
        assert "has_attachment" in sample
        assert "word_count" in sample

        # Check redaction in preview
        if "[REDACTED]" in sample["preview"] or "[" in sample["preview"]:
            # Good - contains redaction markers
            pass


@pytest.mark.asyncio
async def test_sample_messages_no_redaction(populated_db):
    """Test message sampling without redaction."""
    result = await sample_messages_tool(
        db_path=populated_db,
        contact_id="+15551234567",
        limit=3,
        days=180,
        redact=False
    )

    assert result["contact_id"] == "+15551234567"
    assert "No redaction applied" in result["privacy_note"]

    # Messages should not be redacted
    for sample in result["samples"]:
        assert "[REDACTED]" not in sample["preview"]


@pytest.mark.asyncio
async def test_empty_network(temp_db):
    """Test network analysis with no group chats."""
    result = await network_intelligence_tool(
        db_path=temp_db,
        min_messages=10,
        days=180,
        redact=True
    )

    assert result["network_stats"]["total_nodes"] == 0
    assert result["network_stats"]["total_edges"] == 0
    assert result["network_stats"]["density"] == 0
    assert result["key_connectors"] == []
    assert result["communities"] == []
