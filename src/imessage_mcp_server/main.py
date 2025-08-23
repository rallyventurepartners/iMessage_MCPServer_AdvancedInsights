#!/usr/bin/env python3
"""
iMessage Advanced Insights MCP Server - Complete Implementation.

This is the production-ready MCP server with all tools properly integrated.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# MCP SDK imports
from mcp.server.fastmcp import FastMCP

# Local imports
from imessage_mcp_server.config import Config, load_config
from imessage_mcp_server.consent import ConsentManager
from imessage_mcp_server.db import close_database, get_database

# Tool imports - Advanced analytics tools
from imessage_mcp_server.tools.analytics import (
    anomaly_scan_tool,
    best_contact_time_tool,
    network_intelligence_tool,
    sample_messages_tool,
)

# Tool imports - Cloud-aware tools
from imessage_mcp_server.tools.cloud_aware import (
    cloud_status_tool,
    smart_query_tool,
    progressive_analysis_tool,
)
from imessage_mcp_server.tools.comparison import relationship_comparison_tool

# Tool imports - Communication pattern tools
from imessage_mcp_server.tools.communication import (
    cadence_calendar_tool,
    conversation_topics_tool,
    response_time_distribution_tool,
    sentiment_evolution_tool,
)
# Enhanced functionality is now integrated into the original tools
from imessage_mcp_server.tools.group_dynamics import group_dynamics_tool
from imessage_mcp_server.tools.predictive_engagement import predictive_engagement_tool

# Tool imports - Core tools
from imessage_mcp_server.tools.consent import (
    check_consent_tool,
    check_tool_consent,
    request_consent_tool,
    revoke_consent_tool,
)
from imessage_mcp_server.tools.contacts import contact_resolve_tool
from imessage_mcp_server.tools.health import health_check_tool

# Tool imports - Optional ML tools
from imessage_mcp_server.tools.ml_tools import (
    emotion_timeline_tool,
    semantic_search_tool,
    topic_clusters_ml_tool,
)
from imessage_mcp_server.tools.overview import summary_overview_tool
from imessage_mcp_server.tools.quality import conversation_quality_tool
from imessage_mcp_server.tools.relationship import relationship_intelligence_tool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("iMessage Advanced Insights")

# Global instances
config: Optional[Config] = None
consent_manager: Optional[ConsentManager] = None


# ===== CONSENT MANAGEMENT TOOLS =====


@mcp.tool()
async def request_consent(expiry_hours: int = 24):
    """Request user consent to access iMessage data."""
    return await request_consent_tool(consent_manager, expiry_hours)


@mcp.tool()
async def check_consent():
    """Check current consent status."""
    return await check_consent_tool(consent_manager)


@mcp.tool()
async def revoke_consent():
    """Revoke user consent to access iMessage data."""
    return await revoke_consent_tool(consent_manager)


# ===== CORE ANALYSIS TOOLS =====


@mcp.tool()
async def imsg_health_check(db_path: str = "~/Library/Messages/chat.db"):
    """Validate DB access, schema presence, index hints, and read-only mode."""
    return await health_check_tool(db_path)


@mcp.tool()
async def imsg_summary_overview(db_path: str = "~/Library/Messages/chat.db", redact: bool = True):
    """Global overview for Claude's kickoff context."""
    if not await check_tool_consent(consent_manager, "imsg_summary_overview"):
        return {"error": "No active consent", "error_type": "consent_required"}

    return await summary_overview_tool(db_path, redact)


@mcp.tool()
async def imsg_contact_resolve(query: str):
    """Resolve phone/email/handle to hashed contact ID."""
    if not await check_tool_consent(consent_manager, "imsg_contact_resolve"):
        return {"error": "No active consent", "error_type": "consent_required"}

    return await contact_resolve_tool(query)


@mcp.tool()
async def imsg_relationship_intelligence(
    contact_id: str,
    window_days: int = 180,
    include_time_series: bool = False,
    time_series_granularity: str = "auto",
    include_visualizations: bool = False,
    db_path: str = "~/Library/Messages/chat.db",
    redact: bool = True,
):
    """
    Comprehensive relationship analysis with optional time series and visualizations.
    
    Can analyze up to 3 years (1095 days) of history with detailed time series
    data and matplotlib/seaborn visualizations showing trends and patterns.
    """
    if not await check_tool_consent(consent_manager, "imsg_relationship_intelligence"):
        return {"error": "No active consent", "error_type": "consent_required"}

    return await relationship_intelligence_tool(
        db_path=db_path,
        contact_id=contact_id,
        window_days=window_days,
        include_time_series=include_time_series,
        time_series_granularity=time_series_granularity,
        include_visualizations=include_visualizations,
        redact=redact
    )


@mcp.tool()
async def imsg_conversation_quality(
    contact_id: str,
    time_period: str = "30d",
    include_recommendations: bool = True,
    db_path: str = "~/Library/Messages/chat.db",
    redact: bool = True,
):
    """Calculate comprehensive conversation quality score with multi-dimensional analysis."""
    if not await check_tool_consent(consent_manager, "imsg_conversation_quality"):
        return {"error": "No active consent", "error_type": "consent_required"}

    return await conversation_quality_tool(contact_id, time_period, include_recommendations, db_path, redact)


@mcp.tool()
async def imsg_relationship_comparison(
    contact_ids: List[str],
    comparison_type: str = "comprehensive",
    include_clusters: bool = True,
    db_path: str = "~/Library/Messages/chat.db",
    redact: bool = True,
):
    """Compare multiple relationships to identify patterns and insights."""
    if not await check_tool_consent(consent_manager, "imsg_relationship_comparison"):
        return {"error": "No active consent", "error_type": "consent_required"}

    return await relationship_comparison_tool(contact_ids, comparison_type, include_clusters, db_path, redact)


# ===== COMMUNICATION PATTERN TOOLS =====


@mcp.tool()
async def imsg_conversation_topics(
    contact_id: Optional[str] = None,
    days: int = 90,
    min_frequency: int = 3,
    db_path: str = "~/Library/Messages/chat.db",
):
    """Extract conversation topics using keyword analysis."""
    if not await check_tool_consent(consent_manager, "imsg_conversation_topics"):
        return {"error": "No active consent", "error_type": "consent_required"}

    return await conversation_topics_tool(db_path, contact_id, days, min_frequency, True)


@mcp.tool()
async def imsg_sentiment_evolution(
    contact_id: Optional[str] = None,
    days: int = 180,
    bucket_days: int = 7,
    db_path: str = "~/Library/Messages/chat.db",
):
    """Track sentiment changes over time."""
    if not await check_tool_consent(consent_manager, "imsg_sentiment_evolution"):
        return {"error": "No active consent", "error_type": "consent_required"}

    return await sentiment_evolution_tool(db_path, contact_id, days, bucket_days, True)


@mcp.tool()
async def imsg_response_time_distribution(
    contact_id: Optional[str] = None, days: int = 90, db_path: str = "~/Library/Messages/chat.db"
):
    """Analyze response time patterns."""
    if not await check_tool_consent(consent_manager, "imsg_response_time_distribution"):
        return {"error": "No active consent", "error_type": "consent_required"}

    return await response_time_distribution_tool(db_path, contact_id, days, True)


@mcp.tool()
async def imsg_cadence_calendar(
    contact_id: Optional[str] = None, 
    time_period: str = "90d",
    granularity: str = "auto",
    include_visualizations: bool = False,
    include_time_series: bool = False,
    comparison_contacts: Optional[List[str]] = None,
    db_path: str = "~/Library/Messages/chat.db"
):
    """
    Generate communication frequency analysis with optional visualizations.
    
    Supports extended time periods (up to 36 months) and multiple granularities.
    Can generate matplotlib/seaborn charts and compare multiple contacts.
    
    Time periods: "90d", "6m", "1y", "36m"
    Granularity: "auto", "hourly", "daily", "weekly", "monthly"
    """
    if not await check_tool_consent(consent_manager, "imsg_cadence_calendar"):
        return {"error": "No active consent", "error_type": "consent_required"}

    return await cadence_calendar_tool(
        db_path=db_path,
        contact_id=contact_id,
        time_period=time_period,
        granularity=granularity,
        include_visualizations=include_visualizations,
        include_time_series=include_time_series,
        comparison_contacts=comparison_contacts,
        redact=True
    )


# ===== ADVANCED ANALYTICS TOOLS =====


@mcp.tool()
async def imsg_best_contact_time(contact_id: str, db_path: str = "~/Library/Messages/chat.db"):
    """Predict optimal times to contact someone."""
    if not await check_tool_consent(consent_manager, "imsg_best_contact_time"):
        return {"error": "No active consent", "error_type": "consent_required"}

    return await best_contact_time_tool(db_path, contact_id, True)


@mcp.tool()
async def imsg_anomaly_scan(
    contact_id: Optional[str] = None,
    days: int = 30,
    sensitivity: float = 0.7,
    db_path: str = "~/Library/Messages/chat.db",
):
    """Detect unusual communication patterns."""
    if not await check_tool_consent(consent_manager, "imsg_anomaly_scan"):
        return {"error": "No active consent", "error_type": "consent_required"}

    return await anomaly_scan_tool(db_path, contact_id, days, sensitivity, True)


@mcp.tool()
async def imsg_network_intelligence(
    min_messages: int = 10, days: int = 180, db_path: str = "~/Library/Messages/chat.db"
):
    """Analyze social network structure from group chats."""
    if not await check_tool_consent(consent_manager, "imsg_network_intelligence"):
        return {"error": "No active consent", "error_type": "consent_required"}

    return await network_intelligence_tool(db_path, min_messages, days, True)


@mcp.tool()
async def imsg_sample_messages(
    contact_id: str, limit: int = 10, days: int = 30, db_path: str = "~/Library/Messages/chat.db"
):
    """Get heavily redacted sample messages for context."""
    if not await check_tool_consent(consent_manager, "imsg_sample_messages"):
        return {"error": "No active consent", "error_type": "consent_required"}

    return await sample_messages_tool(db_path, contact_id, limit, days, True)


# ===== OPTIONAL ML TOOLS =====


@mcp.tool()
async def imsg_semantic_search(
    query: str,
    contact_id: Optional[str] = None,
    k: int = 10,
    db_path: str = "~/Library/Messages/chat.db",
):
    """Search messages using semantic similarity (requires ML dependencies)."""
    if not await check_tool_consent(consent_manager, "imsg_semantic_search"):
        return {"error": "No active consent", "error_type": "consent_required"}

    return await semantic_search_tool(query, db_path, contact_id, k, True)


@mcp.tool()
async def imsg_emotion_timeline(
    contact_id: Optional[str] = None, days: int = 90, db_path: str = "~/Library/Messages/chat.db"
):
    """Track emotional dimensions over time (requires ML dependencies)."""
    if not await check_tool_consent(consent_manager, "imsg_emotion_timeline"):
        return {"error": "No active consent", "error_type": "consent_required"}

    return await emotion_timeline_tool(db_path, contact_id, days)


@mcp.tool()
async def imsg_topic_clusters(
    contact_id: Optional[str] = None, k: int = 10, db_path: str = "~/Library/Messages/chat.db"
):
    """Discover topic clusters using ML (requires ML dependencies)."""
    if not await check_tool_consent(consent_manager, "imsg_topic_clusters"):
        return {"error": "No active consent", "error_type": "consent_required"}

    return await topic_clusters_ml_tool(db_path, contact_id, k)


# ===== CLOUD-AWARE TOOLS =====


@mcp.tool()
async def imsg_cloud_status(
    db_path: str = "~/Library/Messages/chat.db", check_specific_dates: Optional[List[str]] = None
):
    """
    Check cloud vs local message availability.

    Returns detailed information about what data is available locally
    vs stored in iCloud, with recommendations for accessing cloud data.
    """
    if not await check_tool_consent(consent_manager, "imsg_cloud_status"):
        return {"error": "No active consent", "error_type": "consent_required"}

    return await cloud_status_tool(db_path, check_specific_dates)


@mcp.tool()
async def imsg_smart_query(
    query_type: str,
    contact_id: Optional[str] = None,
    date_range: Optional[Dict[str, str]] = None,
    auto_download: bool = False,
    db_path: str = "~/Library/Messages/chat.db",
):
    """
    Intelligently query messages with cloud awareness.

    Query types: 'messages', 'stats', 'patterns'
    Can optionally trigger downloads for missing data.
    """
    if not await check_tool_consent(consent_manager, "imsg_smart_query"):
        return {"error": "No active consent", "error_type": "consent_required"}

    return await smart_query_tool(db_path, query_type, contact_id, date_range, auto_download)


@mcp.tool()
async def imsg_progressive_analysis(
    analysis_type: str,
    contact_id: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    db_path: str = "~/Library/Messages/chat.db",
):
    """
    Perform analysis that adapts to available data.

    Analysis types: 'sentiment', 'topics', 'patterns'
    Returns results with confidence scores based on data availability.
    """
    if not await check_tool_consent(consent_manager, "imsg_progressive_analysis"):
        return {"error": "No active consent", "error_type": "consent_required"}

    return await progressive_analysis_tool(db_path, analysis_type, contact_id, options)


@mcp.tool()
async def imsg_group_dynamics(
    group_id: str,
    analysis_type: str = "comprehensive",
    time_period: str = "90d",
    db_path: str = "~/Library/Messages/chat.db",
    redact: bool = True,
):
    """
    Analyze group chat dynamics and social structures.
    
    Provides insights into:
    - Participation patterns and balance
    - Influence networks and opinion leaders
    - Subgroup/clique detection
    - Group health metrics
    
    Analysis types: "comprehensive", "participation", "influence", "health"
    """
    if not await check_tool_consent(consent_manager, "imsg_group_dynamics"):
        return {"error": "No active consent", "error_type": "consent_required"}
    
    return await group_dynamics_tool(group_id, analysis_type, time_period, db_path, redact)


@mcp.tool()
async def imsg_predictive_engagement(
    contact_id: str,
    prediction_type: str = "comprehensive",
    horizon_days: int = 30,
    include_recommendations: bool = True,
    db_path: str = "~/Library/Messages/chat.db",
    redact: bool = True,
):
    """
    Predict future engagement patterns using ML.
    
    Features:
    - Response time predictions
    - Activity level forecasting
    - Sentiment trajectory prediction
    - Risk of communication breakdown
    - Optimal engagement strategies
    
    Prediction types: "comprehensive", "response_time", "activity", "sentiment", "risk"
    """
    if not await check_tool_consent(consent_manager, "imsg_predictive_engagement"):
        return {"error": "No active consent", "error_type": "consent_required"}
    
    return await predictive_engagement_tool(
        contact_id, prediction_type, horizon_days, include_recommendations, db_path, redact
    )


# ===== SERVER LIFECYCLE =====


async def startup():
    """Initialize server resources on startup."""
    global config, consent_manager

    logger.info("Starting iMessage Advanced Insights MCP Server...")

    # Load configuration
    config = load_config()
    logger.info(f"Loaded configuration: {config}")

    # Initialize consent manager
    consent_manager = ConsentManager()
    logger.info("Consent manager initialized")

    # Initialize database connection
    try:
        db = await get_database(config.db_path)
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        # Continue anyway - tools will handle the error

    logger.info("Server startup complete")


async def shutdown():
    """Clean up resources on shutdown."""
    logger.info("Shutting down iMessage Advanced Insights MCP Server...")

    # Close database connection
    await close_database()

    logger.info("Server shutdown complete")


def main():
    """Main entry point for the server."""
    try:
        # Initialize resources
        asyncio.run(startup())

        # Run the MCP server
        logger.info("Starting MCP server on stdio transport...")
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
    finally:
        # Cleanup resources
        asyncio.run(shutdown())


if __name__ == "__main__":
    main()
