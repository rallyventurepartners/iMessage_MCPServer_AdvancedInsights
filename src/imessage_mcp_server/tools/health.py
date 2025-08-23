"""
Health check tool for iMessage Advanced Insights.

Validates database access, schema presence, and system readiness.
"""

import os
import sqlite3
from typing import Any, Dict


async def health_check_tool(db_path: str = "~/Library/Messages/chat.db") -> Dict[str, Any]:
    """
    Validate DB access, schema presence, index hints, and read-only mode.

    This tool performs a comprehensive health check of the iMessage database
    and system configuration without accessing any message content.

    Args:
        db_path: Path to the iMessage database

    Returns:
        Dict containing health check results including:
        - Database accessibility
        - Schema validation
        - Performance indexes
        - Message statistics
        - System information

    Privacy:
        No message content is accessed, only metadata and statistics.
    """
    health = {
        "db_accessible": False,
        "schema_valid": False,
        "indexes_present": False,
        "read_only": True,
        "stats": {},
        "errors": [],
    }

    try:
        # Expand path and check existence
        expanded_path = os.path.expanduser(db_path)
        if not os.path.exists(expanded_path):
            health["errors"].append(f"Database not found at {expanded_path}")
            return health

        # Check read permissions
        if not os.access(expanded_path, os.R_OK):
            health["errors"].append("No read permission for database")
            return health

        health["db_accessible"] = True

        # Connect and check schema
        conn = sqlite3.connect(f"file:{expanded_path}?mode=ro", uri=True)
        cursor = conn.cursor()

        # Check critical tables
        required_tables = ["message", "handle", "chat", "chat_message_join"]
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = {row[0] for row in cursor.fetchall()}

        missing_tables = set(required_tables) - existing_tables
        if missing_tables:
            health["errors"].append(f"Missing tables: {missing_tables}")
        else:
            health["schema_valid"] = True

        # Check for performance indexes
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'")
        indexes = [row[0] for row in cursor.fetchall()]
        health["indexes_present"] = len(indexes) > 0
        health["index_count"] = len(indexes)

        # Get basic statistics (no content)
        if health["schema_valid"]:
            # Total messages
            cursor.execute("SELECT COUNT(*) FROM message")
            health["stats"]["total_messages"] = cursor.fetchone()[0]

            # Total contacts
            cursor.execute("SELECT COUNT(DISTINCT id) FROM handle")
            health["stats"]["total_contacts"] = cursor.fetchone()[0]

            # Date range
            cursor.execute(
                "SELECT datetime(MIN(date/1000000000 + 978307200), 'unixepoch', 'localtime'), "
                "datetime(MAX(date/1000000000 + 978307200), 'unixepoch', 'localtime') "
                "FROM message WHERE date > 0"
            )
            min_date, max_date = cursor.fetchone()
            health["stats"]["date_range"] = {"earliest": min_date, "latest": max_date}

            # Database size
            cursor.execute(
                "SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()"
            )
            db_size = cursor.fetchone()[0]
            health["stats"]["database_size_mb"] = round(db_size / 1024 / 1024, 2)

        conn.close()

        # System info
        health["system"] = {
            "platform": "macOS",
            "python_version": "3.9+",
            "mcp_server": "iMessage Advanced Insights v0.1.0",
        }

        # Recommendations
        recommendations = []
        if not health["indexes_present"]:
            recommendations.append(
                "Run 'python scripts/add_performance_indexes.py' for better performance"
            )
        if health["stats"].get("database_size_mb", 0) > 5000:
            recommendations.append("Consider database sharding for optimal performance")

        if recommendations:
            health["recommendations"] = recommendations

    except sqlite3.Error as e:
        health["errors"].append(f"Database error: {str(e)}")
    except Exception as e:
        health["errors"].append(f"Unexpected error: {str(e)}")

    # Overall status
    health["status"] = (
        "healthy"
        if (health["db_accessible"] and health["schema_valid"] and not health["errors"])
        else "unhealthy"
    )

    # Add expected fields for compatibility
    health["healthy"] = health["status"] == "healthy"
    health["schema"] = {
        "valid": health["schema_valid"],
        "tables": list(existing_tables) if "existing_tables" in locals() else [],
    }
    health["indices"] = {"count": health["index_count"], "present": health["indexes_present"]}

    return health
