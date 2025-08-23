"""Integration tests for the MCP server."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import imessage_mcp_server.main as main_module


@pytest.mark.asyncio
async def test_server_startup():
    """Test server startup sequence."""
    # Mock the global variables
    main_module.config = Mock()
    main_module.consent_manager = AsyncMock()

    # Test startup
    await main_module.startup()

    # Verify consent manager was initialized
    main_module.consent_manager.initialize.assert_called_once()


@pytest.mark.asyncio
async def test_server_shutdown():
    """Test server shutdown sequence."""
    # Test shutdown
    await main_module.shutdown()
    # Should complete without errors


@pytest.mark.asyncio
async def test_consent_tools():
    """Test consent management tools."""
    # Mock consent manager
    mock_consent = AsyncMock()
    mock_consent.request_consent.return_value = {
        "consent_granted": True,
        "expires_at": "2024-12-20T10:00:00"
    }
    mock_consent.check_consent.return_value = {
        "has_consent": True,
        "expires_at": "2024-12-20T10:00:00",
        "remaining_hours": 24
    }
    mock_consent.revoke_consent.return_value = {
        "consent_revoked": True
    }

    main_module.consent_manager = mock_consent

    # Test request consent
    result = await main_module.request_consent(expiry_hours=48)
    assert result["consent_granted"] is True

    # Test check consent
    result = await main_module.check_consent()
    assert result["has_consent"] is True

    # Test revoke consent
    result = await main_module.revoke_consent()
    assert result["consent_revoked"] is True


@pytest.mark.asyncio
async def test_health_check_tool(temp_db):
    """Test health check tool."""
    result = await main_module.imsg_health_check(db_path=temp_db)

    assert "db_accessible" in result
    assert "schema_valid" in result
    assert "read_only_mode" in result
    assert "indexes" in result

    # Should be accessible with our test DB
    assert result["db_accessible"] is True


@pytest.mark.asyncio
async def test_tool_consent_check():
    """Test that tools check consent before execution."""
    # Mock consent manager to deny consent
    mock_consent = AsyncMock()
    mock_consent.check_consent.return_value = {
        "has_consent": False
    }

    main_module.consent_manager = mock_consent

    # Test various tools - should all return consent error
    tools_to_test = [
        (main_module.imsg_summary_overview, {}),
        (main_module.imsg_contact_resolve, {"query": "test@example.com"}),
        (main_module.imsg_relationship_intelligence, {"contact_id": "test"}),
        (main_module.imsg_conversation_topics, {}),
        (main_module.imsg_sentiment_evolution, {}),
    ]

    for tool_func, kwargs in tools_to_test:
        result = await tool_func(**kwargs)
        assert result["error"] == "No active consent"
        assert result["error_type"] == "consent_required"


@pytest.mark.asyncio
async def test_tool_registration():
    """Test that all tools are properly registered."""
    # Check that MCP instance has tools registered
    assert hasattr(main_module.mcp, '_tools')
    assert len(main_module.mcp._tools) > 0

    # Expected tool names
    expected_tools = [
        "request_consent",
        "check_consent",
        "revoke_consent",
        "imsg_health_check",
        "imsg_summary_overview",
        "imsg_contact_resolve",
        "imsg_relationship_intelligence",
        "imsg_conversation_topics",
        "imsg_sentiment_evolution",
        "imsg_response_time_distribution",
        "imsg_cadence_calendar",
        "imsg_best_contact_time",
        "imsg_anomaly_scan",
        "imsg_network_intelligence",
        "imsg_sample_messages",
        "imsg_semantic_search",
        "imsg_emotion_timeline",
        "imsg_topic_clusters"
    ]

    registered_tools = set(main_module.mcp._tools.keys())

    for tool_name in expected_tools:
        assert tool_name in registered_tools, f"Tool {tool_name} not registered"
