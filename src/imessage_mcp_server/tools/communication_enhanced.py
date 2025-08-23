"""
Enhanced communication analysis tools with extended time periods and visualizations.

Provides comprehensive time series analysis with hourly, daily, weekly, and monthly
aggregations plus matplotlib/seaborn visualizations.
"""

import asyncio
import base64
import logging
import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from imessage_mcp_server.db import get_database
from imessage_mcp_server.privacy import apply_privacy_filters, hash_contact_id

logger = logging.getLogger(__name__)

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


async def enhanced_cadence_calendar_tool(
    db_path: str,
    contact_id: Optional[str] = None,
    time_period: str = "90d",
    granularity: str = "auto",
    include_visualizations: bool = True,
    include_time_series: bool = True,
    comparison_contacts: Optional[List[str]] = None,
    redact: bool = True,
) -> Dict[str, Any]:
    """
    Enhanced communication frequency analysis with extended time periods and visualizations.
    
    Features:
    - Supports up to 36 months of data
    - Multiple granularity levels (hourly, daily, weekly, monthly)
    - Time series data export
    - Matplotlib/seaborn visualizations
    - Multi-contact comparison
    
    Args:
        db_path: Path to iMessage database
        contact_id: Specific contact to analyze (None = all)
        time_period: Period to analyze (e.g., "90d", "6m", "1y", "36m")
        granularity: Data granularity ("auto", "hourly", "daily", "weekly", "monthly")
        include_visualizations: Generate visualization charts
        include_time_series: Include raw time series data
        comparison_contacts: List of contacts to compare against
        redact: Apply privacy filters
        
    Returns:
        Dict containing cadence analysis with optional visualizations
    """
    try:
        # Parse time period
        days = _parse_extended_time_period(time_period)
        
        # Auto-select granularity if needed
        if granularity == "auto":
            granularity = _auto_select_granularity(days)
        
        # Expand path
        db_path = Path(db_path).expanduser()
        db = await get_database(str(db_path))
        
        # Get primary contact data
        primary_data = await _fetch_message_counts(
            db, contact_id, days, granularity
        )
        
        # Get comparison data if requested
        comparison_data = {}
        if comparison_contacts:
            for comp_contact in comparison_contacts[:5]:  # Limit to 5 for readability
                comparison_data[comp_contact] = await _fetch_message_counts(
                    db, comp_contact, days, granularity
                )
        
        # Build result structure
        result = {
            "contact_id": hash_contact_id(contact_id) if contact_id and redact else contact_id,
            "time_period": time_period,
            "days_analyzed": days,
            "granularity": granularity,
            "total_messages": primary_data["total_messages"],
            "daily_average": round(primary_data["total_messages"] / days, 2),
        }
        
        # Add heatmap data (for backward compatibility)
        if granularity in ["hourly", "daily"]:
            result["heatmap"] = _generate_heatmap_data(primary_data["hourly_data"])
            result["peak_hours"] = primary_data["peak_hours"]
            result["peak_days"] = primary_data["peak_days"]
        
        # Add time series data if requested
        if include_time_series:
            result["time_series"] = {
                "primary": primary_data["time_series"],
            }
            if comparison_data:
                result["time_series"]["comparisons"] = {
                    hash_contact_id(cid) if redact else cid: data["time_series"]
                    for cid, data in comparison_data.items()
                }
        
        # Add statistics
        result["statistics"] = _calculate_advanced_statistics(
            primary_data, comparison_data
        )
        
        # Generate visualizations if requested
        if include_visualizations:
            charts = await _generate_cadence_visualizations(
                primary_data, 
                comparison_data,
                contact_id,
                granularity,
                days
            )
            result["visualizations"] = charts
        
        # Apply privacy filters if needed
        if redact:
            result = apply_privacy_filters(result)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in enhanced cadence analysis: {e}")
        return {"error": str(e), "error_type": "analysis_error"}


async def enhanced_relationship_intelligence_tool(
    db_path: str,
    contact_id: str,
    window_days: int = 180,
    include_time_series: bool = True,
    time_series_granularity: str = "auto",
    include_visualizations: bool = True,
    redact: bool = True,
) -> Dict[str, Any]:
    """
    Enhanced relationship intelligence with time series data and visualizations.
    
    Provides comprehensive relationship analysis with:
    - Base relationship metrics and insights
    - Time series message counts
    - Trend visualizations
    - Response time evolution
    - Sentiment trajectory charts
    
    Args:
        db_path: Path to iMessage database
        contact_id: Contact to analyze
        window_days: Days to analyze (up to 1095 / 3 years)
        include_time_series: Include detailed time series data
        time_series_granularity: Granularity for time series
        include_visualizations: Generate visualization charts
        redact: Apply privacy filters
        
    Returns:
        Dict containing enhanced relationship analysis
    """
    if not contact_id:
        return {"error": "contact_id is required", "error_type": "missing_parameter"}

    try:
        # Parse database path and get connection
        db_path = Path(db_path).expanduser()
        db = await get_database(str(db_path))
        
        # Get handle ROWID for the contact
        cursor = await db.execute(
            """
            SELECT ROWID, id FROM handle 
            WHERE id = ? OR id LIKE ?
            """,
            (contact_id, f"%{contact_id}%")
        )
        handle_result = await cursor.fetchone()
        
        if not handle_result:
            return {"error": f"Contact not found: {contact_id}", "error_type": "contact_not_found"}
        
        handle_rowid, full_id = handle_result
        
        # Time window
        now = datetime.now()
        cutoff = now - timedelta(days=window_days)
        cutoff_timestamp = int((cutoff - datetime(2001, 1, 1)).total_seconds() * 1000000000)
        
        # Get all messages in window
        cursor = await db.execute(
            """
            SELECT 
                date,
                is_from_me,
                cache_has_attachments,
                datetime(date/1000000000 + 978307200, 'unixepoch', 'localtime') as timestamp
            FROM message
            WHERE handle_id = ? AND date > ?
            ORDER BY date
            """,
            (handle_rowid, cutoff_timestamp)
        )
        messages = await cursor.fetchall()
        
        if not messages:
            return {
                "contact_id": hash_contact_id(full_id) if redact else full_id,
                "window_days": window_days,
                "messages_total": 0,
                "error": "No messages found in the specified time window",
                "error_type": "no_data",
            }
        
        # Basic statistics
        total_messages = len(messages)
        sent_messages = sum(1 for m in messages if m[1] == 1)
        received_messages = total_messages - sent_messages
        messages_with_attachments = sum(1 for m in messages if m[2] == 1)
        
        # Response time analysis
        response_times = []
        last_sent_time = None
        last_received_time = None
        
        for msg in messages:
            timestamp = datetime.fromisoformat(msg[3])
            is_from_me = msg[1]
            
            if is_from_me:
                if last_received_time:
                    response_time = (timestamp - last_received_time).total_seconds()
                    if 0 < response_time < 86400:  # Within 24 hours
                        response_times.append(("me", response_time))
                last_sent_time = timestamp
            else:
                if last_sent_time:
                    response_time = (timestamp - last_sent_time).total_seconds()
                    if 0 < response_time < 86400:  # Within 24 hours
                        response_times.append(("them", response_time))
                last_received_time = timestamp
        
        # Calculate response metrics
        my_response_times = [t for (who, t) in response_times if who == "me"]
        their_response_times = [t for (who, t) in response_times if who == "them"]
        
        def calculate_response_stats(times):
            if not times:
                return None
            import statistics
            return {
                "median_seconds": statistics.median(times),
                "mean_seconds": statistics.mean(times),
                "p25_seconds": statistics.quantiles(times, n=4)[0] if len(times) > 3 else times[0],
                "p75_seconds": statistics.quantiles(times, n=4)[2] if len(times) > 3 else times[-1],
                "count": len(times),
            }
        
        # Daily patterns
        daily_messages = defaultdict(int)
        hourly_messages = defaultdict(int)
        
        for msg in messages:
            timestamp = datetime.fromisoformat(msg[3])
            daily_messages[timestamp.date()] += 1
            hourly_messages[timestamp.hour] += 1
        
        # Calculate streaks and gaps
        sorted_days = sorted(daily_messages.keys())
        current_streak = 0
        max_streak = 0
        max_gap = 0
        
        if sorted_days:
            current_streak = 1
            for i in range(1, len(sorted_days)):
                gap = (sorted_days[i] - sorted_days[i - 1]).days
                if gap == 1:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    max_gap = max(max_gap, gap)
                    current_streak = 1
        
        # Recent activity
        recent_30d = sum(1 for m in messages if (now - datetime.fromisoformat(m[3])).days <= 30)
        recent_7d = sum(1 for m in messages if (now - datetime.fromisoformat(m[3])).days <= 7)
        
        # Engagement score calculation
        import statistics
        factors = {
            "frequency": min(1.0, total_messages / (window_days * 2)),  # 2 msgs/day = 1.0
            "balance": 1.0 - abs(0.5 - (sent_messages / total_messages)),
            "responsiveness": min(
                1.0, len(my_response_times) / max(1, received_messages) if my_response_times else 0
            ),
            "consistency": min(1.0, len(daily_messages) / window_days),
            "recency": min(1.0, recent_30d / 60),  # 60 messages in 30d = 1.0
        }
        
        engagement_score = sum(factors.values()) / len(factors)
        
        # Determine relationship flags
        flags = []
        
        if sent_messages / total_messages > 0.65:
            flags.append("conversation-initiator")
        elif sent_messages / total_messages < 0.35:
            flags.append("conversation-responder")
        
        if my_response_times and statistics.median(my_response_times) < 300:  # 5 minutes
            flags.append("quick-responder")
        
        if max_streak > 14:
            flags.append("consistent-communication")
        
        if recent_30d == 0 and total_messages > 50:
            flags.append("reconnect-suggested")
        
        if messages_with_attachments / total_messages > 0.2:
            flags.append("media-heavy")
        
        if engagement_score > 0.7:
            flags.append("highly-engaged")
        elif engagement_score < 0.3:
            flags.append("low-engagement")
        
        # Peak communication times
        peak_hour = max(hourly_messages.items(), key=lambda x: x[1])[0] if hourly_messages else None
        
        # Build base response
        base_result = {
            "contact_id": hash_contact_id(full_id) if redact else full_id,
            "window_days": window_days,
            "messages_total": total_messages,
            "messages_sent": sent_messages,
            "messages_received": received_messages,
            "sent_percentage": round(100.0 * sent_messages / total_messages, 1),
            "received_percentage": round(100.0 * received_messages / total_messages, 1),
            "messages_with_media": messages_with_attachments,
            "media_percentage": round(100.0 * messages_with_attachments / total_messages, 1),
            "daily_average": round(total_messages / window_days, 2),
            "response_analysis": {
                "my_responses": calculate_response_stats(my_response_times),
                "their_responses": calculate_response_stats(their_response_times),
            },
            "communication_patterns": {
                "days_with_contact": len(daily_messages),
                "max_streak_days": max_streak,
                "current_streak_days": current_streak,
                "max_gap_days": max_gap,
                "peak_hour": peak_hour,
                "recent_messages_30d": recent_30d,
                "recent_messages_7d": recent_7d,
            },
            "engagement_metrics": {
                "engagement_score": round(engagement_score, 3),
                "frequency_score": round(factors["frequency"], 3),
                "balance_score": round(factors["balance"], 3),
                "responsiveness_score": round(factors["responsiveness"], 3),
                "consistency_score": round(factors["consistency"], 3),
                "recency_score": round(factors["recency"], 3),
            },
            "flags": flags,
            "last_contact": messages[-1][3] if messages else None,
        }
        
        # Generate insights
        insights = []
        
        # Engagement insight
        if engagement_score > 0.7:
            insights.append("This is one of your most engaged relationships")
        elif engagement_score < 0.3:
            insights.append("Engagement with this contact is relatively low")
        
        # Balance insight
        if sent_messages / total_messages > 0.7:
            insights.append("You initiate most conversations with this contact")
        elif sent_messages / total_messages < 0.3:
            insights.append("This contact usually initiates conversations with you")
        
        # Response time insight
        if my_response_times and statistics.median(my_response_times) < 600:
            insights.append("You typically respond very quickly to this contact")
        
        # Activity trend
        if recent_30d > total_messages * 0.3:
            insights.append("Communication has increased recently")
        elif recent_30d < total_messages * 0.05:
            insights.append("Communication has decreased significantly")
        
        # Consistency
        if max_streak > 30:
            insights.append(f"You had a {max_streak}-day streak of daily communication")
        
        base_result["insights"] = insights
        
        # Enhance with time series data if requested
        if include_time_series or include_visualizations:
            # Auto-select granularity
            if time_series_granularity == "auto":
                time_series_granularity = _auto_select_granularity(window_days)
            
            # Get detailed time series
            time_data = await _fetch_relationship_time_series(
                db, contact_id, window_days, time_series_granularity
            )
            
            if include_time_series:
                base_result["time_series"] = time_data["series"]
                base_result["time_series_granularity"] = time_series_granularity
            
            if include_visualizations:
                # Generate relationship-specific visualizations
                charts = await _generate_relationship_visualizations(
                    time_data,
                    base_result,
                    contact_id,
                    window_days
                )
                base_result["visualizations"] = charts
        
        # Apply privacy filters if needed
        if redact:
            base_result = apply_privacy_filters(base_result)
        
        return base_result
        
    except Exception as e:
        logger.error(f"Error in enhanced relationship intelligence: {e}")
        return {"error": str(e), "error_type": "analysis_error"}


async def _fetch_message_counts(
    db: Any, contact_id: Optional[str], days: int, granularity: str
) -> Dict[str, Any]:
    """Fetch message counts with specified granularity."""
    
    # Build appropriate query based on granularity
    if granularity == "hourly":
        date_format = '%Y-%m-%d %H'
        group_format = "strftime('%Y-%m-%d %H', datetime(m.date/1000000000 + 978307200, 'unixepoch'))"
    elif granularity == "daily":
        date_format = '%Y-%m-%d'
        group_format = "date(datetime(m.date/1000000000 + 978307200, 'unixepoch'))"
    elif granularity == "weekly":
        date_format = '%Y-W%W'
        group_format = "strftime('%Y-W%W', datetime(m.date/1000000000 + 978307200, 'unixepoch'))"
    else:  # monthly
        date_format = '%Y-%m'
        group_format = "strftime('%Y-%m', datetime(m.date/1000000000 + 978307200, 'unixepoch'))"
    
    query = f"""
    SELECT 
        {group_format} as time_bucket,
        COUNT(*) as message_count,
        SUM(CASE WHEN m.is_from_me = 1 THEN 1 ELSE 0 END) as sent_count,
        SUM(CASE WHEN m.is_from_me = 0 THEN 1 ELSE 0 END) as received_count,
        strftime('%w', datetime(m.date/1000000000 + 978307200, 'unixepoch')) as day_of_week,
        strftime('%H', datetime(m.date/1000000000 + 978307200, 'unixepoch')) as hour
    FROM message m
    WHERE m.text IS NOT NULL
    AND datetime(m.date/1000000000 + 978307200, 'unixepoch') > datetime('now', '-' || ? || ' days')
    """
    
    params = [days]
    
    if contact_id:
        query += """
        AND m.handle_id IN (
            SELECT h.ROWID 
            FROM handle h 
            WHERE h.id = ?
        )
        """
        params.append(contact_id)
    
    query += f" GROUP BY time_bucket, day_of_week, hour ORDER BY time_bucket"
    
    # Execute query
    cursor = await db.execute(query, params)
    rows = await cursor.fetchall()
    
    # Process results
    time_series = defaultdict(lambda: {"total": 0, "sent": 0, "received": 0})
    hourly_data = defaultdict(lambda: defaultdict(int))
    day_totals = defaultdict(int)
    hour_totals = defaultdict(int)
    
    for row in rows:
        time_bucket, count, sent, received, dow, hour = row
        time_series[time_bucket]["total"] += count
        time_series[time_bucket]["sent"] += sent
        time_series[time_bucket]["received"] += received
        
        # Aggregate for heatmap
        hourly_data[int(dow)][int(hour)] += count
        day_totals[int(dow)] += count
        hour_totals[int(hour)] += count
    
    # Find peak times
    day_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    peak_days = sorted(
        [(day_names[d], c) for d, c in day_totals.items()],
        key=lambda x: x[1],
        reverse=True
    )[:3]
    peak_hours = sorted(
        [(f"{h:02d}:00", c) for h, c in hour_totals.items()],
        key=lambda x: x[1],
        reverse=True
    )[:3]
    
    return {
        "time_series": dict(time_series),
        "hourly_data": dict(hourly_data),
        "peak_days": [d[0] for d in peak_days],
        "peak_hours": [h[0] for h in peak_hours],
        "total_messages": sum(ts["total"] for ts in time_series.values()),
    }


async def _fetch_relationship_time_series(
    db: Any, contact_id: str, days: int, granularity: str
) -> Dict[str, Any]:
    """Fetch detailed relationship time series data."""
    
    # Similar to _fetch_message_counts but with additional metrics
    if granularity == "daily":
        date_format = "date(datetime(m.date/1000000000 + 978307200, 'unixepoch'))"
    elif granularity == "weekly":
        date_format = "strftime('%Y-W%W', datetime(m.date/1000000000 + 978307200, 'unixepoch'))"
    else:  # monthly
        date_format = "strftime('%Y-%m', datetime(m.date/1000000000 + 978307200, 'unixepoch'))"
    
    query = f"""
    SELECT 
        {date_format} as time_bucket,
        COUNT(*) as message_count,
        SUM(CASE WHEN m.is_from_me = 1 THEN 1 ELSE 0 END) as sent_count,
        SUM(CASE WHEN m.is_from_me = 0 THEN 1 ELSE 0 END) as received_count,
        AVG(LENGTH(m.text)) as avg_length,
        COUNT(DISTINCT date(datetime(m.date/1000000000 + 978307200, 'unixepoch'))) as active_days
    FROM message m
    JOIN handle h ON m.handle_id = h.ROWID
    WHERE h.id = ?
    AND m.text IS NOT NULL
    AND datetime(m.date/1000000000 + 978307200, 'unixepoch') > datetime('now', '-' || ? || ' days')
    GROUP BY time_bucket
    ORDER BY time_bucket
    """
    
    cursor = await db.execute(query, [contact_id, days])
    rows = await cursor.fetchall()
    
    series = []
    for row in rows:
        time_bucket, total, sent, received, avg_length, active_days = row
        series.append({
            "period": time_bucket,
            "total": total,
            "sent": sent,
            "received": received,
            "balance": round(sent / total, 2) if total > 0 else 0.5,
            "avg_length": round(avg_length, 1),
            "active_days": active_days,
        })
    
    return {"series": series}


async def _generate_cadence_visualizations(
    primary_data: Dict,
    comparison_data: Dict,
    contact_id: Optional[str],
    granularity: str,
    days: int
) -> Dict[str, str]:
    """Generate matplotlib/seaborn visualizations for cadence data."""
    charts = {}
    
    # 1. Time series line chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot primary data
    ts_data = primary_data["time_series"]
    if ts_data:
        periods = sorted(ts_data.keys())
        values = [ts_data[p]["total"] for p in periods]
        
        # Convert to dates for better x-axis
        if granularity == "daily":
            dates = [datetime.strptime(p, "%Y-%m-%d") for p in periods]
        elif granularity == "weekly":
            # Handle week format
            dates = [datetime.strptime(p + "-1", "%Y-W%W-%w") for p in periods]
        else:  # monthly
            dates = [datetime.strptime(p + "-01", "%Y-%m-%d") for p in periods]
        
        ax.plot(dates, values, linewidth=2, label="Primary Contact", marker='o', markersize=4)
        
        # Add rolling average
        if len(values) > 7:
            window = 7 if granularity == "daily" else 4
            rolling_avg = pd.Series(values).rolling(window=window, center=True).mean()
            ax.plot(dates, rolling_avg, linewidth=1.5, alpha=0.7, linestyle='--', 
                   label=f"{window}-period moving average")
    
    # Plot comparison data
    for i, (cid, data) in enumerate(comparison_data.items()):
        ts = data["time_series"]
        if ts:
            periods = sorted(ts.keys())
            values = [ts[p]["total"] for p in periods]
            ax.plot(dates[:len(values)], values, linewidth=1.5, alpha=0.7, 
                   label=f"Contact {i+2}")
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Message Count")
    ax.set_title(f"Message Volume Over Time ({granularity.capitalize()})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    if granularity == "daily" and days > 90:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    elif granularity == "daily":
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    charts["time_series_chart"] = _fig_to_base64(fig)
    plt.close(fig)
    
    # 2. Heatmap visualization (if hourly data available)
    if "hourly_data" in primary_data and primary_data["hourly_data"]:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for heatmap
        hourly = primary_data["hourly_data"]
        heatmap_data = np.zeros((7, 24))
        
        for dow, hours in hourly.items():
            for hour, count in hours.items():
                heatmap_data[dow][hour] = count
        
        # Create heatmap
        sns.heatmap(
            heatmap_data,
            cmap="YlOrRd",
            cbar_kws={"label": "Message Count"},
            yticklabels=["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
            xticklabels=[f"{h:02d}" for h in range(24)],
            ax=ax
        )
        
        ax.set_title("Communication Heatmap by Day and Hour")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Day of Week")
        
        plt.tight_layout()
        charts["heatmap_chart"] = _fig_to_base64(fig)
        plt.close(fig)
    
    # 3. Message balance chart (sent vs received)
    if ts_data:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        periods = sorted(ts_data.keys())[-30:]  # Last 30 periods for clarity
        sent = [ts_data[p]["sent"] for p in periods]
        received = [ts_data[p]["received"] for p in periods]
        
        # Stacked area chart
        ax1.fill_between(range(len(periods)), sent, alpha=0.7, label="Sent")
        ax1.fill_between(range(len(periods)), sent, 
                        [s + r for s, r in zip(sent, received)], 
                        alpha=0.7, label="Received")
        ax1.set_ylabel("Messages")
        ax1.set_title("Message Flow (Sent vs Received)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Balance ratio
        balance = [s / (s + r) if (s + r) > 0 else 0.5 
                  for s, r in zip(sent, received)]
        ax2.plot(range(len(periods)), balance, linewidth=2, color='purple')
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax2.fill_between(range(len(periods)), 0.5, balance, 
                        where=[b > 0.5 for b in balance], 
                        alpha=0.3, color='blue', label='You initiate more')
        ax2.fill_between(range(len(periods)), balance, 0.5,
                        where=[b < 0.5 for b in balance],
                        alpha=0.3, color='green', label='They initiate more')
        ax2.set_ylabel("Send Ratio")
        ax2.set_xlabel(f"{granularity.capitalize()} Period")
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Simplify x-axis labels
        ax2.set_xticks(range(0, len(periods), max(1, len(periods)//10)))
        ax2.set_xticklabels([periods[i] for i in range(0, len(periods), max(1, len(periods)//10))], 
                           rotation=45)
        
        plt.tight_layout()
        charts["balance_chart"] = _fig_to_base64(fig)
        plt.close(fig)
    
    return charts


async def _generate_relationship_visualizations(
    time_data: Dict,
    base_result: Dict,
    contact_id: str,
    window_days: int
) -> Dict[str, str]:
    """Generate relationship-specific visualizations."""
    charts = {}
    
    series = time_data["series"]
    if not series:
        return charts
    
    # 1. Multi-metric dashboard
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Message volume over time
    ax1 = fig.add_subplot(gs[0, :])
    periods = [s["period"] for s in series]
    totals = [s["total"] for s in series]
    
    ax1.plot(periods, totals, linewidth=2, marker='o', markersize=4)
    ax1.fill_between(range(len(periods)), totals, alpha=0.3)
    ax1.set_title("Message Volume Trend")
    ax1.set_ylabel("Messages")
    ax1.grid(True, alpha=0.3)
    
    # Simplify x-axis
    if len(periods) > 20:
        ax1.set_xticks(range(0, len(periods), len(periods)//10))
        ax1.set_xticklabels([periods[i] for i in range(0, len(periods), len(periods)//10)], 
                           rotation=45)
    
    # Communication balance
    ax2 = fig.add_subplot(gs[1, 0])
    balance = [s["balance"] for s in series]
    ax2.plot(periods, balance, linewidth=2, color='purple')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.fill_between(range(len(periods)), 0.5, balance, alpha=0.3)
    ax2.set_title("Communication Balance")
    ax2.set_ylabel("Your Message Ratio")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Average message length
    ax3 = fig.add_subplot(gs[1, 1])
    avg_lengths = [s["avg_length"] for s in series]
    ax3.plot(periods, avg_lengths, linewidth=2, color='green', marker='s', markersize=4)
    ax3.set_title("Average Message Length")
    ax3.set_ylabel("Characters")
    ax3.grid(True, alpha=0.3)
    
    # Active days (for weekly/monthly granularity)
    ax4 = fig.add_subplot(gs[2, :])
    if "active_days" in series[0]:
        active_days = [s["active_days"] for s in series]
        ax4.bar(range(len(periods)), active_days, alpha=0.7, color='orange')
        ax4.set_title("Active Days per Period")
        ax4.set_ylabel("Days with Messages")
        ax4.set_xlabel("Period")
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f"Relationship Analytics Dashboard - {window_days} days", fontsize=16)
    plt.tight_layout()
    charts["dashboard"] = _fig_to_base64(fig)
    plt.close(fig)
    
    # 2. Engagement score visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate rolling engagement score
    engagement_scores = []
    for i, s in enumerate(series):
        # Simple engagement score based on volume, balance, and consistency
        volume_score = min(s["total"] / 10, 1.0)  # Normalize to 0-1
        balance_score = 1 - abs(s["balance"] - 0.5) * 2  # Closer to 0.5 is better
        
        # Consistency (compare to previous period)
        if i > 0:
            consistency = 1 - abs(s["total"] - series[i-1]["total"]) / max(s["total"], series[i-1]["total"])
        else:
            consistency = 1.0
        
        engagement = (volume_score + balance_score + consistency) / 3
        engagement_scores.append(engagement)
    
    ax.plot(periods, engagement_scores, linewidth=3, marker='o', markersize=6)
    ax.fill_between(range(len(periods)), engagement_scores, alpha=0.3)
    ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Healthy threshold')
    ax.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='Warning threshold')
    
    ax.set_title("Relationship Engagement Score Over Time")
    ax.set_ylabel("Engagement Score (0-1)")
    ax.set_xlabel("Period")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Simplify x-axis
    if len(periods) > 20:
        ax.set_xticks(range(0, len(periods), len(periods)//10))
        ax.set_xticklabels([periods[i] for i in range(0, len(periods), len(periods)//10)], 
                          rotation=45)
    
    plt.tight_layout()
    charts["engagement_score"] = _fig_to_base64(fig)
    plt.close(fig)
    
    return charts


def _parse_extended_time_period(period: str) -> int:
    """Parse time period string to days, supporting extended periods."""
    if period.endswith('d'):
        return int(period[:-1])
    elif period.endswith('w'):
        return int(period[:-1]) * 7
    elif period.endswith('m'):
        return int(period[:-1]) * 30
    elif period.endswith('y'):
        return int(period[:-1]) * 365
    else:
        return 90  # Default


def _auto_select_granularity(days: int) -> str:
    """Automatically select appropriate granularity based on time period."""
    if days <= 7:
        return "hourly"
    elif days <= 90:
        return "daily"
    elif days <= 365:
        return "weekly"
    else:
        return "monthly"


def _generate_heatmap_data(hourly_data: Dict) -> Dict[str, Dict[str, int]]:
    """Convert hourly data to heatmap format."""
    day_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    heatmap = {day: {} for day in day_names}
    
    for dow, hours in hourly_data.items():
        day_name = day_names[dow]
        for hour, count in hours.items():
            heatmap[day_name][f"{hour:02d}:00"] = count
    
    return heatmap


def _calculate_advanced_statistics(
    primary_data: Dict, comparison_data: Dict
) -> Dict[str, Any]:
    """Calculate advanced statistics from time series data."""
    stats = {}
    
    ts = primary_data["time_series"]
    if not ts:
        return stats
    
    values = [v["total"] for v in ts.values()]
    
    # Basic statistics
    stats["mean"] = round(np.mean(values), 2)
    stats["median"] = round(np.median(values), 2)
    stats["std_dev"] = round(np.std(values), 2)
    stats["cv"] = round(stats["std_dev"] / stats["mean"], 2) if stats["mean"] > 0 else 0
    
    # Trend analysis
    if len(values) >= 2:
        # Simple linear regression
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        stats["trend_slope"] = round(slope, 3)
        stats["trend_direction"] = "increasing" if slope > 0.1 else "decreasing" if slope < -0.1 else "stable"
    
    # Seasonality detection (simplified)
    if len(values) >= 14:
        # Check weekly pattern
        weekly_pattern = []
        for i in range(7):
            day_values = values[i::7]
            if day_values:
                weekly_pattern.append(np.mean(day_values))
        
        if weekly_pattern:
            stats["weekly_pattern_strength"] = round(np.std(weekly_pattern) / np.mean(weekly_pattern), 2)
    
    # Comparison statistics
    if comparison_data:
        stats["rank_among_contacts"] = _calculate_rank(primary_data, comparison_data)
    
    return stats


def _calculate_rank(primary_data: Dict, comparison_data: Dict) -> int:
    """Calculate rank among compared contacts."""
    all_totals = [primary_data["total_messages"]]
    all_totals.extend([data["total_messages"] for data in comparison_data.values()])
    all_totals.sort(reverse=True)
    
    return all_totals.index(primary_data["total_messages"]) + 1


def _fig_to_base64(fig: Figure) -> str:
    """Convert matplotlib figure to base64 encoded string."""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    return f"data:image/png;base64,{image_base64}"