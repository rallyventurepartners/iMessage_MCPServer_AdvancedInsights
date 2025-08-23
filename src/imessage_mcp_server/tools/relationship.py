"""
Relationship analysis tools for iMessage Advanced Insights.

Provides deep analysis of communication patterns and relationship dynamics.
"""

import os
import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict

from imessage_mcp_server.privacy import hash_contact_id


async def relationship_intelligence_tool(
    db_path: str = "~/Library/Messages/chat.db",
    contact_id: str = None,
    window_days: int = 180,
    include_time_series: bool = True,
    time_series_granularity: str = "auto",
    include_visualizations: bool = True,
    redact: bool = True,
) -> Dict[str, Any]:
    """
    Comprehensive relationship analysis with time series and visualizations.

    This tool provides deep insights into communication patterns, engagement
    levels, and relationship dynamics based on messaging metadata only.

    Args:
        db_path: Path to the iMessage database
        contact_id: Hashed contact identifier (required)
        window_days: Analysis window in days (default: 180, max: 1095)
        include_time_series: Include detailed time series data
        time_series_granularity: Granularity for time series ("auto", "daily", "weekly", "monthly")
        include_visualizations: Generate matplotlib/seaborn charts
        redact: Whether to apply privacy redaction (default: True)

    Returns:
        Dict containing:
        - Message statistics (volume, ratios, timing)
        - Response time analysis
        - Engagement metrics
        - Communication patterns
        - Relationship flags and insights
        - Time series data
        - Visualization charts

    Privacy:
        No message content accessed. All analysis based on metadata only.
    """
    # Import and use enhanced functionality directly
    from imessage_mcp_server.tools.communication_enhanced import enhanced_relationship_intelligence_tool
    
    return await enhanced_relationship_intelligence_tool(
        db_path=db_path,
        contact_id=contact_id,
        window_days=window_days,
        include_time_series=include_time_series,
        time_series_granularity=time_series_granularity,
        include_visualizations=include_visualizations,
        redact=redact
    )
