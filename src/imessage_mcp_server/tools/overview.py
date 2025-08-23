"""
Overview and summary tools for iMessage Advanced Insights.

Provides high-level statistics and summaries without accessing message content.
"""

import os
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict

from imessage_mcp_server.privacy import hash_contact_id


async def summary_overview_tool(
    db_path: str = "~/Library/Messages/chat.db", redact: bool = True
) -> Dict[str, Any]:
    """
    Get global overview of messaging activity for Claude's context.

    This tool provides high-level statistics about the user's messaging
    activity without accessing any message content. All contact identifiers
    are hashed for privacy.

    Args:
        db_path: Path to the iMessage database
        redact: Whether to hash contact identifiers (default: True)

    Returns:
        Dict containing:
        - Total message counts
        - Active contacts (30d, 90d, all-time)
        - Date ranges
        - Top contacts by message volume
        - Activity trends

    Privacy:
        No message content accessed. Contact IDs are hashed by default.
    """
    try:
        # Expand path
        expanded_path = os.path.expanduser(db_path)
        if not os.path.exists(expanded_path):
            return {
                "error": f"Database not found at {expanded_path}",
                "error_type": "database_not_found",
            }

        # Connect to database
        conn = sqlite3.connect(f"file:{expanded_path}?mode=ro", uri=True)
        cursor = conn.cursor()

        # Total messages
        cursor.execute("SELECT COUNT(*) FROM message")
        total_messages = cursor.fetchone()[0]

        # Total unique contacts
        cursor.execute("SELECT COUNT(DISTINCT id) FROM handle")
        total_contacts = cursor.fetchone()[0]

        # Date range
        cursor.execute(
            """
            SELECT 
                datetime(MIN(date/1000000000 + 978307200), 'unixepoch', 'localtime') as earliest,
                datetime(MAX(date/1000000000 + 978307200), 'unixepoch', 'localtime') as latest
            FROM message 
            WHERE date > 0
        """
        )
        earliest, latest = cursor.fetchone()

        # Active contacts by time period
        now = datetime.now()
        periods = {"30d": 30, "90d": 90, "365d": 365}

        active_contacts = {}
        for period_name, days in periods.items():
            cutoff = now - timedelta(days=days)
            cutoff_timestamp = int((cutoff - datetime(2001, 1, 1)).total_seconds() * 1000000000)

            cursor.execute(
                """
                SELECT COUNT(DISTINCT h.id)
                FROM message m
                JOIN handle h ON m.handle_id = h.ROWID
                WHERE m.date > ?
            """,
                (cutoff_timestamp,),
            )

            active_contacts[f"active_contacts_{period_name}"] = cursor.fetchone()[0]

        # Top contacts by message volume
        cursor.execute(
            """
            SELECT h.id, COUNT(*) as msg_count
            FROM message m
            JOIN handle h ON m.handle_id = h.ROWID
            GROUP BY h.id
            ORDER BY msg_count DESC
            LIMIT 10
        """
        )

        top_contacts = []
        for contact_id, count in cursor.fetchall():
            if redact:
                contact_id = hash_contact_id(contact_id)[:12] + "..."
            top_contacts.append(
                {
                    "contact_id": contact_id,
                    "message_count": count,
                    "percentage": round(100.0 * count / total_messages, 1),
                }
            )

        # Message trends by month (last 12 months)
        cursor.execute(
            """
            SELECT 
                strftime('%Y-%m', datetime(date/1000000000 + 978307200, 'unixepoch', 'localtime')) as month,
                COUNT(*) as count
            FROM message
            WHERE date > ?
            GROUP BY month
            ORDER BY month DESC
            LIMIT 12
        """,
            (int((now - timedelta(days=365) - datetime(2001, 1, 1)).total_seconds() * 1000000000),),
        )

        monthly_trends = []
        for month, count in cursor.fetchall():
            monthly_trends.append({"month": month, "message_count": count})

        # Calculate growth rate
        if len(monthly_trends) >= 2:
            recent_avg = sum(m["message_count"] for m in monthly_trends[:3]) / 3
            older_avg = sum(m["message_count"] for m in monthly_trends[-3:]) / 3
            growth_rate = ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
        else:
            growth_rate = 0

        # Sent vs received ratio
        cursor.execute(
            """
            SELECT 
                SUM(CASE WHEN is_from_me = 1 THEN 1 ELSE 0 END) as sent,
                SUM(CASE WHEN is_from_me = 0 THEN 1 ELSE 0 END) as received
            FROM message
        """
        )
        sent, received = cursor.fetchone()

        conn.close()

        # Build response
        overview = {
            "total_messages": total_messages,
            "total_contacts": total_contacts,
            "active_contacts_30d": active_contacts.get("active_contacts_30d", 0),
            "active_contacts_90d": active_contacts.get("active_contacts_90d", 0),
            "active_contacts_365d": active_contacts.get("active_contacts_365d", 0),
            "date_range": {
                "earliest": earliest,
                "latest": latest,
                "days_span": (
                    (datetime.fromisoformat(latest) - datetime.fromisoformat(earliest)).days
                    if earliest and latest
                    else 0
                ),
            },
            "message_distribution": {
                "sent": sent,
                "received": received,
                "sent_percentage": (
                    round(100.0 * sent / total_messages, 1) if total_messages > 0 else 0
                ),
                "received_percentage": (
                    round(100.0 * received / total_messages, 1) if total_messages > 0 else 0
                ),
            },
            "top_contacts": top_contacts,
            "monthly_trends": monthly_trends,
            "activity_metrics": {
                "avg_messages_per_day": (
                    round(
                        total_messages
                        / max(
                            1,
                            (
                                datetime.fromisoformat(latest) - datetime.fromisoformat(earliest)
                            ).days,
                        ),
                        1,
                    )
                    if earliest and latest
                    else 0
                ),
                "avg_messages_per_contact": round(total_messages / max(1, total_contacts), 1),
                "growth_rate_3m": round(growth_rate, 1),
            },
            "insights": [],
        }

        # Generate insights
        insights = []

        # Activity level insight
        if overview["activity_metrics"]["avg_messages_per_day"] > 50:
            insights.append("You're a very active messager, averaging over 50 messages per day")
        elif overview["activity_metrics"]["avg_messages_per_day"] < 10:
            insights.append(
                "You have a moderate messaging volume, averaging less than 10 messages per day"
            )

        # Contact concentration
        if top_contacts and top_contacts[0]["percentage"] > 20:
            insights.append(
                f"Your top contact represents {top_contacts[0]['percentage']}% of all messages"
            )

        # Growth trend
        if growth_rate > 20:
            insights.append(
                "Your messaging activity has increased significantly over the past year"
            )
        elif growth_rate < -20:
            insights.append("Your messaging activity has decreased notably over the past year")

        # Balance insight
        sent_pct = overview["message_distribution"]["sent_percentage"]
        if sent_pct > 60:
            insights.append("You tend to initiate conversations more often than others")
        elif sent_pct < 40:
            insights.append("You tend to be more responsive than initiating in conversations")

        overview["insights"] = insights

        return overview

    except sqlite3.Error as e:
        return {"error": f"Database error: {str(e)}", "error_type": "database_error"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}", "error_type": "unexpected_error"}
