#!/usr/bin/env python3
"""
Simplified integration tests for iMessage MCP Server using real database.

These tests are designed to match the actual implementation signatures.
"""

import sys
import time
from pathlib import Path

import pytest
import pytest_asyncio

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from imessage_mcp_server.main import (
    check_consent,
    imsg_anomaly_scan,
    imsg_best_contact_time,
    imsg_cadence_calendar,
    imsg_contact_resolve,
    imsg_conversation_topics,
    imsg_health_check,
    imsg_network_intelligence,
    imsg_relationship_intelligence,
    imsg_response_time_distribution,
    imsg_sample_messages,
    imsg_sentiment_evolution,
    imsg_summary_overview,
    initialize,
    request_consent,
    revoke_consent,
)


class TestBasicFunctionality:
    """Test basic functionality with real database."""

    @pytest_asyncio.fixture(autouse=True)
    async def setup_and_teardown(self):
        """Setup consent and cleanup after tests."""
        # Initialize the server
        await initialize()

        # Grant consent using the global consent manager
        from imessage_mcp_server.main import consent_manager
        await consent_manager.grant_consent(expires_hours=1)

        yield

        # Revoke consent after tests
        await consent_manager.revoke_consent()

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test database health check."""
        result = await imsg_health_check()

        assert "db_version" in result
        assert "tables" in result
        assert "indices_ok" in result
        assert "read_only_ok" in result
        assert result["read_only_ok"] is True
        assert "message" in result["tables"]
        assert "handle" in result["tables"]
        assert "chat" in result["tables"]

    @pytest.mark.asyncio
    async def test_summary_overview(self):
        """Test summary overview with actual return format."""
        result = await imsg_summary_overview()

        # Check for actual keys returned
        assert "total_messages" in result
        assert "unique_contacts" in result  # Not "total_contacts"
        assert "date_range" in result
        assert "by_direction" in result
        assert "by_platform" in result
        assert "attachments" in result

        # Verify data types
        assert isinstance(result["total_messages"], int)
        assert isinstance(result["unique_contacts"], int)
        assert isinstance(result["by_direction"], dict)
        assert "sent" in result["by_direction"]
        assert "received" in result["by_direction"]

        print(f"Found {result['total_messages']} messages from {result['unique_contacts']} contacts")

    @pytest.mark.asyncio
    async def test_contact_resolve(self):
        """Test contact resolution."""
        # Test with a phone number
        result = await imsg_contact_resolve(query="+1-555-123-4567")

        assert "contact_id" in result
        assert "display_name" in result
        assert "kind" in result
        assert result["kind"] == "phone"

        # Test with email
        result2 = await imsg_contact_resolve(query="test@example.com")
        assert result2["kind"] == "email"

    @pytest.mark.asyncio
    async def test_relationship_intelligence(self):
        """Test relationship intelligence with actual parameters."""
        # Function takes contact_filters, not contact_id
        result = await imsg_relationship_intelligence(
            window_days=30
        )

        assert "contacts" in result
        assert isinstance(result["contacts"], list)

        if result["contacts"]:
            contact = result["contacts"][0]
            assert "contact_id" in contact  # Uses hashed contact_id, not handle_id
            assert "messages_total" in contact  # Different key name
            assert "sent_pct" in contact  # Percentage, not count
            assert "avg_daily_msgs" in contact
            assert "flags" in contact

    @pytest.mark.asyncio
    async def test_conversation_topics(self):
        """Test conversation topics analysis."""
        result = await imsg_conversation_topics(
            since_days=30,
            top_k=10
        )

        assert "terms" in result
        assert "trend" in result
        assert isinstance(result["terms"], list)

        if result["terms"]:
            term = result["terms"][0]
            assert "term" in term
            assert "count" in term

    @pytest.mark.asyncio
    async def test_sample_messages(self):
        """Test sample messages with preview caps."""
        result = await imsg_sample_messages(limit=5)

        assert "messages" in result
        assert isinstance(result["messages"], list)
        assert len(result["messages"]) <= 5

        for msg in result["messages"]:
            assert "preview" in msg  # Messages have preview, not text
            assert "ts" in msg  # Timestamp field
            assert "direction" in msg  # sent/received instead of is_from_me
            assert "contact_id" in msg
            # Check preview cap (160 chars)
            if msg["preview"]:
                assert len(msg["preview"]) <= 163  # 160 + "..."

    @pytest.mark.asyncio
    async def test_response_time_distribution(self):
        """Test response time analysis."""
        result = await imsg_response_time_distribution()

        assert "p50_s" in result
        assert "p90_s" in result
        assert "p99_s" in result
        assert "histogram" in result
        assert "samples" in result

    @pytest.mark.asyncio
    async def test_sentiment_evolution(self):
        """Test sentiment evolution."""
        result = await imsg_sentiment_evolution(window_days=30)

        assert "series" in result
        assert "summary" in result
        assert isinstance(result["series"], list)

        summary = result["summary"]
        assert "mean" in summary
        assert "delta_30d" in summary

    @pytest.mark.asyncio
    async def test_cadence_calendar(self):
        """Test communication cadence calendar."""
        result = await imsg_cadence_calendar()

        assert "matrix" in result
        assert "hours" in result
        assert "weekdays" in result

        # Check matrix dimensions (24 hours x 7 days)
        assert len(result["matrix"]) == 24
        assert all(len(row) == 7 for row in result["matrix"])

    @pytest.mark.asyncio
    async def test_best_contact_time(self):
        """Test best contact time prediction."""
        result = await imsg_best_contact_time()

        assert "windows" in result
        assert isinstance(result["windows"], list)

        for window in result["windows"]:
            assert "weekday" in window
            assert "hour" in window
            assert "score" in window  # Actual key returned

    @pytest.mark.asyncio
    async def test_anomaly_scan(self):
        """Test anomaly detection."""
        result = await imsg_anomaly_scan(lookback_days=30)

        assert "anomalies" in result
        assert isinstance(result["anomalies"], list)

        for anomaly in result["anomalies"]:
            assert "ts" in anomaly  # Timestamp field
            assert "type" in anomaly
            assert "severity" in anomaly

    @pytest.mark.asyncio
    async def test_network_intelligence(self):
        """Test social network analysis."""
        result = await imsg_network_intelligence(since_days=90)

        assert "nodes" in result
        assert "edges" in result
        assert "communities" in result
        assert "key_connectors" in result

        assert isinstance(result["nodes"], list)
        assert isinstance(result["edges"], list)


class TestConsentManagement:
    """Test consent management functionality."""

    @pytest_asyncio.fixture(autouse=True)
    async def setup(self):
        """Initialize server for consent tests."""
        await initialize()

    @pytest.mark.asyncio
    async def test_consent_workflow(self):
        """Test complete consent workflow."""
        # Check initial state
        check_result = await check_consent()
        initial_consent = check_result["consent"]

        # Revoke any existing consent
        if initial_consent:
            await revoke_consent()

        # Verify no consent
        check_result = await check_consent()
        assert check_result["consent"] is False

        # Try to use a tool without consent
        result = await imsg_summary_overview()
        assert "error" in result
        assert "consent" in result["error"].lower()

        # Grant consent
        grant_result = await request_consent(expiry_hours=1)
        assert grant_result["consent"] is True
        assert "expires_at" in grant_result

        # Verify consent is active
        check_result = await check_consent()
        assert check_result["consent"] is True

        # Now tool should work
        result = await imsg_summary_overview()
        assert "error" not in result
        assert "total_messages" in result

        # Revoke consent
        revoke_result = await revoke_consent()
        assert revoke_result["consent"] is False

        # Verify revoked
        check_result = await check_consent()
        assert check_result["consent"] is False


class TestPerformance:
    """Test performance requirements."""

    @pytest_asyncio.fixture(autouse=True)
    async def setup(self):
        """Setup for performance tests."""
        await initialize()
        from imessage_mcp_server.main import consent_manager
        await consent_manager.grant_consent(expires_hours=1)
        yield
        await consent_manager.revoke_consent()

    @pytest.mark.asyncio
    async def test_health_check_performance(self):
        """Test health check completes within target time."""
        start = time.time()
        result = await imsg_health_check()
        duration = time.time() - start

        assert "tables" in result
        assert duration < 1.5, f"Health check took {duration:.2f}s, exceeding 1.5s target"
        print(f"Health check completed in {duration:.3f}s")

    @pytest.mark.asyncio
    async def test_summary_performance(self):
        """Test summary overview performance."""
        start = time.time()
        result = await imsg_summary_overview()
        duration = time.time() - start

        assert "total_messages" in result
        assert duration < 1.5, f"Summary took {duration:.2f}s, exceeding 1.5s target"
        print(f"Summary completed in {duration:.3f}s with {result['total_messages']} messages")

    @pytest.mark.asyncio
    async def test_multiple_queries_performance(self):
        """Test multiple queries in sequence."""
        operations = [
            ("Health Check", imsg_health_check()),
            ("Summary", imsg_summary_overview()),
            ("Topics", imsg_conversation_topics(since_days=7, top_k=5)),
            ("Response Times", imsg_response_time_distribution()),
            ("Cadence", imsg_cadence_calendar()),
        ]

        total_start = time.time()
        for name, operation in operations:
            start = time.time()
            result = await operation
            duration = time.time() - start
            print(f"{name}: {duration:.3f}s")
            assert duration < 1.5, f"{name} took {duration:.2f}s, exceeding 1.5s target"

        total_duration = time.time() - total_start
        print(f"Total time for {len(operations)} operations: {total_duration:.3f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
