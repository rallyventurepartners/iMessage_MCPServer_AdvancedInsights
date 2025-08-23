"""Cloud-aware tools for handling iMessage data split between local and iCloud."""

import asyncio
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from imessage_mcp_server.db import get_database
from imessage_mcp_server.privacy import apply_privacy_filters, hash_contact_id

logger = logging.getLogger(__name__)


async def cloud_status_tool(
    db_path: str = "~/Library/Messages/chat.db",
    check_specific_dates: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Check cloud vs local message availability.

    Returns detailed information about what data is available locally
    vs stored in iCloud, with recommendations for accessing cloud data.
    """
    try:
        db_path = Path(db_path).expanduser()
        db = await get_database(str(db_path))

        # Overall statistics
        query = """
        SELECT 
            COUNT(*) as total_messages,
            SUM(CASE WHEN text IS NULL THEN 1 ELSE 0 END) as cloud_messages,
            SUM(CASE WHEN text IS NOT NULL THEN 1 ELSE 0 END) as local_messages,
            MIN(date) as earliest_date,
            MAX(date) as latest_date
        FROM message
        WHERE date > 0
        """

        cursor = await db.execute(query)
        result = await cursor.fetchone()
        total, cloud, local, earliest, latest = result

        # Calculate percentages
        cloud_pct = (cloud / total * 100) if total > 0 else 0
        local_pct = (local / total * 100) if total > 0 else 0

        # Convert timestamps
        earliest_dt = datetime.fromtimestamp(earliest / 1000000000 + 978307200)
        latest_dt = datetime.fromtimestamp(latest / 1000000000 + 978307200)

        # Analyze by time period
        period_query = """
        SELECT 
            strftime('%Y-%m', datetime(date/1000000000 + 978307200, 'unixepoch')) as month,
            COUNT(*) as total,
            SUM(CASE WHEN text IS NOT NULL THEN 1 ELSE 0 END) as local,
            SUM(CASE WHEN text IS NULL THEN 1 ELSE 0 END) as cloud
        FROM message
        WHERE date > 0
        GROUP BY month
        ORDER BY month DESC
        LIMIT 24
        """

        cursor = await db.execute(period_query)
        periods = await cursor.fetchall()

        # Find gaps
        availability_gaps = []
        for month, total, local, cloud in periods:
            if cloud > local * 10:  # More than 90% in cloud
                availability_gaps.append(
                    {
                        "period": month,
                        "total_messages": total,
                        "cloud_percentage": round(cloud / total * 100, 1),
                    }
                )

        # Check specific dates if requested
        date_availability = {}
        if check_specific_dates:
            for date_str in check_specific_dates:
                date_query = """
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN text IS NOT NULL THEN 1 ELSE 0 END) as available
                FROM message
                WHERE date(datetime(date/1000000000 + 978307200, 'unixepoch')) = date(?)
                """
                cursor = await db.execute(date_query, [date_str])
                total, available = await cursor.fetchone()
                date_availability[date_str] = {
                    "total": total,
                    "available": available,
                    "percentage": round(available / total * 100, 1) if total > 0 else 0,
                }

        # Generate recommendations
        recommendations = []
        if cloud_pct > 50:
            recommendations.append(
                {
                    "priority": "high",
                    "action": "download_from_cloud",
                    "description": f"{cloud_pct:.1f}% of messages are in iCloud",
                    "command": "brctl download ~/Library/Messages/",
                }
            )

        if availability_gaps:
            recommendations.append(
                {
                    "priority": "medium",
                    "action": "target_specific_periods",
                    "description": f"{len(availability_gaps)} time periods have limited local data",
                    "periods": availability_gaps[:5],  # Top 5 gaps
                }
            )

        return {
            "summary": {
                "total_messages": total,
                "local_messages": local,
                "cloud_messages": cloud,
                "local_percentage": round(local_pct, 1),
                "cloud_percentage": round(cloud_pct, 1),
                "date_range": {"start": earliest_dt.isoformat(), "end": latest_dt.isoformat()},
            },
            "availability_gaps": availability_gaps,
            "date_availability": date_availability,
            "recommendations": recommendations,
            "download_status": await _check_download_status(),
        }

    except Exception as e:
        logger.error(f"Error checking cloud status: {e}")
        return {"error": str(e), "error_type": "cloud_status_error"}


async def smart_query_tool(
    db_path: str,
    query_type: str,
    contact_id: Optional[str] = None,
    date_range: Optional[Dict[str, str]] = None,
    auto_download: bool = False,
) -> Dict[str, Any]:
    """
    Intelligently query messages with cloud awareness.

    This tool checks data availability before querying and can
    optionally trigger downloads for missing data.
    """
    try:
        db_path = Path(db_path).expanduser()
        db = await get_database(str(db_path))

        # First check what's available
        availability_query = """
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN text IS NOT NULL THEN 1 ELSE 0 END) as available
        FROM message
        WHERE 1=1
        """
        params = []

        if contact_id:
            availability_query += " AND handle_id IN (SELECT ROWID FROM handle WHERE id = ?)"
            params.append(contact_id)

        if date_range:
            if "start" in date_range:
                start_ts = _date_to_apple_timestamp(date_range["start"])
                availability_query += " AND date >= ?"
                params.append(start_ts)
            if "end" in date_range:
                end_ts = _date_to_apple_timestamp(date_range["end"])
                availability_query += " AND date <= ?"
                params.append(end_ts)

        cursor = await db.execute(availability_query, params)
        total, available = await cursor.fetchone()

        availability_pct = (available / total * 100) if total > 0 else 0

        # If data is mostly in cloud
        if availability_pct < 20 and auto_download:
            # Trigger targeted download
            download_result = await _trigger_targeted_download(
                contact_id=contact_id, date_range=date_range
            )

            # Wait a bit for download to start
            await asyncio.sleep(5)

            # Re-check availability
            cursor = await db.execute(availability_query, params)
            total, available = await cursor.fetchone()
            availability_pct = (available / total * 100) if total > 0 else 0

        # Now perform the actual query based on type
        if query_type == "messages":
            result = await _query_messages(db, contact_id, date_range, limit=100)
        elif query_type == "stats":
            result = await _query_statistics(db, contact_id, date_range)
        elif query_type == "patterns":
            result = await _query_patterns(db, contact_id, date_range)
        else:
            result = {"error": f"Unknown query type: {query_type}"}

        # Add availability metadata
        result["_metadata"] = {
            "total_matching_messages": total,
            "locally_available": available,
            "availability_percentage": round(availability_pct, 1),
            "data_completeness": (
                "full"
                if availability_pct > 95
                else "partial" if availability_pct > 50 else "limited"
            ),
        }

        return result

    except Exception as e:
        logger.error(f"Error in smart query: {e}")
        return {"error": str(e), "error_type": "smart_query_error"}


async def progressive_analysis_tool(
    db_path: str,
    analysis_type: str,
    contact_id: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Perform analysis that adapts to available data.

    This tool performs analysis on available data while tracking
    what's missing, providing partial results with confidence scores.
    """
    try:
        db_path = Path(db_path).expanduser()
        db = await get_database(str(db_path))

        # Map analysis types to time windows
        time_windows = {
            "recent": 30,  # Last 30 days
            "quarterly": 90,  # Last quarter
            "annual": 365,  # Last year
            "historical": 1825,  # Last 5 years
        }

        window_days = time_windows.get(
            options.get("window", "quarterly") if options else "quarterly", 90
        )

        # Analyze in chunks, starting with most recent
        results = []
        confidence_scores = []

        for offset in range(0, window_days, 30):  # 30-day chunks
            chunk_start = offset
            chunk_end = min(offset + 30, window_days)

            # Check chunk availability
            chunk_query = """
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN text IS NOT NULL THEN 1 ELSE 0 END) as available
            FROM message
            WHERE datetime(date/1000000000 + 978307200, 'unixepoch') 
                BETWEEN datetime('now', '-' || ? || ' days') 
                AND datetime('now', '-' || ? || ' days')
            """
            params = [chunk_end, chunk_start]

            if contact_id:
                chunk_query += " AND handle_id IN (SELECT ROWID FROM handle WHERE id = ?)"
                params.append(contact_id)

            cursor = await db.execute(chunk_query, params)
            total, available = await cursor.fetchone()

            if total == 0:
                continue

            chunk_confidence = available / total if total > 0 else 0

            # Perform analysis on available data
            if analysis_type == "sentiment":
                chunk_result = await _analyze_sentiment_chunk(
                    db, contact_id, chunk_start, chunk_end
                )
            elif analysis_type == "topics":
                chunk_result = await _analyze_topics_chunk(db, contact_id, chunk_start, chunk_end)
            elif analysis_type == "patterns":
                chunk_result = await _analyze_patterns_chunk(db, contact_id, chunk_start, chunk_end)
            else:
                chunk_result = {}

            if chunk_result:
                results.append(
                    {
                        "period": f"{chunk_start}-{chunk_end} days ago",
                        "data": chunk_result,
                        "confidence": round(chunk_confidence, 2),
                        "messages_analyzed": available,
                        "messages_total": total,
                    }
                )
                confidence_scores.append(chunk_confidence)

        # Aggregate results
        overall_confidence = (
            sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        )

        return {
            "analysis_type": analysis_type,
            "time_window": f"{window_days} days",
            "chunks_analyzed": len(results),
            "overall_confidence": round(overall_confidence, 2),
            "results": results,
            "data_quality": _assess_data_quality(overall_confidence),
            "recommendations": _generate_analysis_recommendations(overall_confidence, results),
        }

    except Exception as e:
        logger.error(f"Error in progressive analysis: {e}")
        return {"error": str(e), "error_type": "progressive_analysis_error"}


# Helper functions


async def _check_download_status() -> Dict[str, Any]:
    """Check if Messages is currently downloading from iCloud."""
    try:
        # Use brctl to check status
        result = subprocess.run(
            ["brctl", "status", str(Path.home() / "Library/Messages")],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            output = result.stdout
            # Parse brctl output (this is simplified - real parsing would be more complex)
            is_downloading = "downloading" in output.lower()
            return {
                "is_downloading": is_downloading,
                "status": "active" if is_downloading else "idle",
            }
    except:
        pass

    return {"is_downloading": False, "status": "unknown"}


async def _trigger_targeted_download(
    contact_id: Optional[str] = None, date_range: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Attempt to trigger download of specific messages."""
    try:
        # This is a simplified version - real implementation would need
        # to interact with Messages app or use private APIs
        subprocess.run(
            ["brctl", "download", str(Path.home() / "Library/Messages")], capture_output=True
        )
        return {"triggered": True}
    except:
        return {"triggered": False, "error": "Unable to trigger download"}


def _date_to_apple_timestamp(date_str: str) -> int:
    """Convert ISO date string to Apple timestamp."""
    dt = datetime.fromisoformat(date_str)
    apple_epoch = datetime(2001, 1, 1)
    return int((dt - apple_epoch).total_seconds() * 1000000000)


def _assess_data_quality(confidence: float) -> str:
    """Assess overall data quality based on confidence."""
    if confidence > 0.9:
        return "excellent"
    elif confidence > 0.7:
        return "good"
    elif confidence > 0.5:
        return "fair"
    elif confidence > 0.2:
        return "limited"
    else:
        return "poor"


def _generate_analysis_recommendations(
    confidence: float, results: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """Generate recommendations based on analysis confidence."""
    recommendations = []

    if confidence < 0.5:
        recommendations.append(
            {
                "priority": "high",
                "action": "Download more data from iCloud before analysis",
                "reason": f"Only {confidence*100:.0f}% of data is available locally",
            }
        )

    # Find gaps
    low_confidence_periods = [r for r in results if r["confidence"] < 0.3]
    if low_confidence_periods:
        recommendations.append(
            {
                "priority": "medium",
                "action": f"Consider downloading data for {len(low_confidence_periods)} time periods",
                "periods": [p["period"] for p in low_confidence_periods],
            }
        )

    return recommendations


# Analysis chunk functions (simplified examples)


async def _analyze_sentiment_chunk(db, contact_id, start_days, end_days):
    """Analyze sentiment for a time chunk."""
    # Simplified - would use actual sentiment analysis
    return {"average_sentiment": 0.65, "trend": "stable"}


async def _analyze_topics_chunk(db, contact_id, start_days, end_days):
    """Analyze topics for a time chunk."""
    # Simplified - would use actual topic extraction
    return {"top_topics": ["work", "family", "weekend"]}


async def _analyze_patterns_chunk(db, contact_id, start_days, end_days):
    """Analyze patterns for a time chunk."""
    # Simplified - would use actual pattern analysis
    return {"message_frequency": "daily", "peak_hours": ["09:00", "18:00"]}


async def _query_messages(db, contact_id, date_range, limit):
    """Query messages with filters."""
    # Simplified - would build actual query
    return {"messages": [], "count": 0}


async def _query_statistics(db, contact_id, date_range):
    """Query statistics."""
    # Simplified
    return {"total": 0, "sent": 0, "received": 0}


async def _query_patterns(db, contact_id, date_range):
    """Query communication patterns."""
    # Simplified
    return {"patterns": {}}
