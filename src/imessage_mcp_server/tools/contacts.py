"""
Contact management and resolution tools for iMessage Advanced Insights.

Handles contact lookup and identification with privacy-preserving hashing.
"""

import os
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict

from imessage_mcp_server.privacy import hash_contact_id


async def contact_resolve_tool(
    query: str, db_path: str = "~/Library/Messages/chat.db"
) -> Dict[str, Any]:
    """
    Resolve a phone number, email, or name fragment to a hashed contact ID.

    This tool helps identify contacts while maintaining privacy through
    consistent hashing. The returned contact ID can be used with other tools.

    Args:
        query: Phone number, email, or name fragment to search
        db_path: Path to the iMessage database

    Returns:
        Dict containing:
        - Matched contact(s) with hashed IDs
        - Basic activity statistics
        - Last contact time

    Privacy:
        All identifiers are SHA-256 hashed. No message content accessed.
    """
    if not query or len(query) < 3:
        return {"error": "Query must be at least 3 characters", "error_type": "invalid_query"}

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

        # Search for handles matching the query
        # Clean the query for SQL LIKE
        search_pattern = f"%{query}%"

        cursor.execute(
            """
            SELECT 
                h.ROWID,
                h.id,
                h.service,
                COUNT(m.ROWID) as message_count,
                MAX(m.date) as last_message_date
            FROM handle h
            LEFT JOIN message m ON m.handle_id = h.ROWID
            WHERE h.id LIKE ?
            GROUP BY h.ROWID, h.id, h.service
            ORDER BY message_count DESC
            LIMIT 10
        """,
            (search_pattern,),
        )

        results = cursor.fetchall()

        if not results:
            return {
                "query": query,
                "matches": [],
                "count": 0,
                "message": "No contacts found matching the query",
            }

        # Process results
        matches = []
        for rowid, handle_id, service, msg_count, last_msg_date in results:
            # Hash the contact ID
            hashed_id = hash_contact_id(handle_id)

            # Format last contact time
            if last_msg_date:
                last_contact = datetime.fromtimestamp(
                    last_msg_date / 1000000000 + 978307200
                ).isoformat()
            else:
                last_contact = None

            # Determine contact type
            if "@" in handle_id:
                contact_type = "email"
            elif handle_id.startswith("+") or handle_id.replace("-", "").isdigit():
                contact_type = "phone"
            else:
                contact_type = "other"

            # Get recent activity
            if last_msg_date:
                cursor.execute(
                    """
                    SELECT COUNT(*) 
                    FROM message 
                    WHERE handle_id = ? AND date > ?
                """,
                    (
                        rowid,
                        int(
                            (
                                datetime.now() - timedelta(days=30) - datetime(2001, 1, 1)
                            ).total_seconds()
                            * 1000000000
                        ),
                    ),
                )
                recent_messages = cursor.fetchone()[0]
            else:
                recent_messages = 0

            match = {
                "contact_id": hashed_id,
                "contact_id_short": hashed_id[:12] + "...",
                "contact_type": contact_type,
                "service": service,
                "total_messages": msg_count or 0,
                "recent_messages_30d": recent_messages,
                "last_contact": last_contact,
                "is_active": recent_messages > 0,
            }

            # Add activity indicator
            if recent_messages > 100:
                match["activity_level"] = "very_active"
            elif recent_messages > 20:
                match["activity_level"] = "active"
            elif recent_messages > 0:
                match["activity_level"] = "occasional"
            else:
                match["activity_level"] = "inactive"

            matches.append(match)

        conn.close()

        # Build response
        response = {
            "query": query,
            "matches": matches,
            "count": len(matches),
            "message": f"Found {len(matches)} contact(s) matching '{query}'",
        }

        # Add usage hint
        if matches:
            response["hint"] = (
                "Use the 'contact_id' value with other tools for analysis. "
                "The full hashed ID ensures privacy while maintaining consistency."
            )

        # If only one match, highlight it
        if len(matches) == 1:
            response["best_match"] = matches[0]
            response["message"] = f"Found exact match for '{query}'"

        return response

    except sqlite3.Error as e:
        return {"error": f"Database error: {str(e)}", "error_type": "database_error"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}", "error_type": "unexpected_error"}
