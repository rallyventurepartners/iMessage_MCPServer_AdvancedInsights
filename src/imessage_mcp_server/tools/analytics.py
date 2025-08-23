"""Advanced analytics tools for iMessage insights."""

import hashlib
import logging
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from imessage_mcp_server.db import get_database
from imessage_mcp_server.privacy import apply_privacy_filters

logger = logging.getLogger(__name__)


async def best_contact_time_tool(
    db_path: str, contact_id: str, redact: bool = True
) -> Dict[str, Any]:
    """Predict optimal times to contact someone based on response patterns."""
    try:
        # Expand path
        db_path = Path(db_path).expanduser()

        # Get database connection
        db = await get_database(str(db_path))

        # Query to analyze response patterns
        query = """
        SELECT 
            m1.date as msg_date,
            m2.date as response_date,
            m1.is_from_me,
            strftime('%w', datetime(m2.date/1000000000 + 978307200, 'unixepoch')) as resp_day,
            strftime('%H', datetime(m2.date/1000000000 + 978307200, 'unixepoch')) as resp_hour,
            (m2.date - m1.date) / 60000000000 as response_time_minutes
        FROM message m1
        INNER JOIN message m2 ON m2.handle_id = m1.handle_id
        WHERE m1.text IS NOT NULL 
        AND m2.text IS NOT NULL
        AND m2.date > m1.date
        AND m2.date < m1.date + 86400000000000  -- Within 24 hours
        AND m1.is_from_me = 1  -- Your messages
        AND m2.is_from_me = 0  -- Their responses
        AND m1.handle_id IN (
            SELECT h.ROWID 
            FROM handle h 
            WHERE h.id = ?
        )
        AND datetime(m1.date/1000000000 + 978307200, 'unixepoch') > datetime('now', '-180 days')
        ORDER BY response_time_minutes
        """

        # Execute query
        cursor = await db.execute(query, [contact_id])
        responses = await cursor.fetchall()

        if not responses:
            return {
                "contact_id": contact_id,
                "error": "No response data available for this contact",
                "error_type": "no_data",
            }

        # Analyze response patterns by hour and day type
        weekday_hours = defaultdict(list)
        weekend_hours = defaultdict(list)

        for _, _, _, day, hour, resp_time in responses:
            if resp_time < 180:  # Only consider responses within 3 hours
                if int(day) in [0, 6]:  # Weekend
                    weekend_hours[int(hour)].append(resp_time)
                else:  # Weekday
                    weekday_hours[int(hour)].append(resp_time)

        # Calculate best times based on quick response rates
        best_times = []

        # Analyze weekday patterns
        for hour, times in weekday_hours.items():
            if len(times) >= 3:  # Need at least 3 data points
                avg_time = statistics.mean(times)
                quick_responses = len([t for t in times if t < 30])  # Under 30 min
                confidence = min(0.95, quick_responses / len(times) * (1 + len(times) / 20))

                if confidence > 0.6:
                    best_times.append(
                        {
                            "time": f"{hour:02d}:00-{(hour+1)%24:02d}:00",
                            "day": "weekday",
                            "avg_response_minutes": round(avg_time, 1),
                            "confidence": round(confidence, 2),
                        }
                    )

        # Analyze weekend patterns
        for hour, times in weekend_hours.items():
            if len(times) >= 2:  # Lower threshold for weekends
                avg_time = statistics.mean(times)
                quick_responses = len([t for t in times if t < 30])
                confidence = min(0.95, quick_responses / len(times) * (1 + len(times) / 10))

                if confidence > 0.5:
                    best_times.append(
                        {
                            "time": f"{hour:02d}:00-{(hour+1)%24:02d}:00",
                            "day": "weekend",
                            "avg_response_minutes": round(avg_time, 1),
                            "confidence": round(confidence, 2),
                        }
                    )

        # Sort by confidence
        best_times.sort(key=lambda x: x["confidence"], reverse=True)

        # Identify times to avoid
        avoid_times = []

        # Late night / early morning
        avoid_times.append({"time": "22:00-08:00", "reason": "Outside typical waking hours"})

        # Check for consistent non-response periods
        all_hours = set(range(24))
        responsive_hours = set(weekday_hours.keys()) | set(weekend_hours.keys())
        quiet_hours = all_hours - responsive_hours

        if quiet_hours:
            for hour in sorted(quiet_hours):
                if 8 <= hour <= 22:  # Only flag unusual quiet times during day
                    avoid_times.append(
                        {
                            "time": f"{hour:02d}:00-{(hour+1)%24:02d}:00",
                            "reason": "No historical responses",
                        }
                    )

        result = {
            "contact_id": contact_id,
            "best_times": best_times[:5],  # Top 5 times
            "avoid_times": avoid_times[:3],  # Top 3 to avoid
            "analysis_note": "Based on historical response patterns",
        }

        if redact:
            result = apply_privacy_filters(result)

        return result

    except Exception as e:
        logger.error(f"Error analyzing best contact times: {e}")
        return {"error": str(e), "error_type": "analysis_error"}


async def anomaly_scan_tool(
    db_path: str,
    contact_id: Optional[str] = None,
    days: int = 30,
    sensitivity: float = 0.7,
    redact: bool = True,
) -> Dict[str, Any]:
    """Detect unusual communication patterns."""
    try:
        # Expand path
        db_path = Path(db_path).expanduser()

        # Get database connection
        db = await get_database(str(db_path))

        anomalies = []

        # Get baseline statistics for each contact
        baseline_query = """
        SELECT 
            h.id as contact,
            COUNT(*) as total_messages,
            COUNT(DISTINCT date(m.date/1000000000 + 978307200, 'unixepoch')) as active_days,
            MAX(m.date) as last_message_date,
            AVG(CASE WHEN m.is_from_me = 1 THEN 1 ELSE 0 END) as sent_ratio
        FROM message m
        JOIN handle h ON m.handle_id = h.ROWID
        WHERE m.text IS NOT NULL
        AND datetime(m.date/1000000000 + 978307200, 'unixepoch') BETWEEN datetime('now', '-' || ? || ' days') AND datetime('now', '-' || ? || ' days')
        """

        if contact_id:
            baseline_query += " AND h.id = ?"
            baseline_params = [days * 4, days, contact_id]
        else:
            baseline_params = [days * 4, days]

        baseline_query += " GROUP BY h.id HAVING total_messages > 10"

        cursor = await db.execute(baseline_query, baseline_params)
        baselines = await cursor.fetchall()

        # Check recent activity against baseline
        for contact, total_msgs, active_days, last_msg, sent_ratio in baselines:
            # Hash contact ID
            contact_hash = hashlib.sha256(contact.encode()).hexdigest()[:12] + "..."

            # Calculate expected daily rate
            daily_rate = total_msgs / (days * 3)  # Based on longer baseline

            # Check recent activity
            recent_query = """
            SELECT 
                COUNT(*) as recent_messages,
                COUNT(DISTINCT date(m.date/1000000000 + 978307200, 'unixepoch')) as recent_days,
                MAX(m.date) as latest_message
            FROM message m
            JOIN handle h ON m.handle_id = h.ROWID
            WHERE h.id = ?
            AND m.text IS NOT NULL
            AND datetime(m.date/1000000000 + 978307200, 'unixepoch') > datetime('now', '-' || ? || ' days')
            """

            cursor = await db.execute(recent_query, [contact, days])
            recent = await cursor.fetchone()
            recent_msgs, recent_days, latest = recent

            # Detect silence (no recent messages when usually active)
            if recent_msgs == 0 and daily_rate > 0.5:
                days_silent = (datetime.now().timestamp() * 1000000000 - last_msg) / 86400000000000
                if days_silent > 7:
                    anomalies.append(
                        {
                            "type": "silence",
                            "contact_id": contact_hash if redact else contact,
                            "severity": "high" if days_silent > 14 else "medium",
                            "description": f"No messages for {int(days_silent)} days (usually {daily_rate:.1f}/day)",
                            "detected_date": datetime.now().date().isoformat(),
                        }
                    )

            # Detect volume spike
            elif recent_msgs > 0:
                recent_daily_rate = recent_msgs / max(recent_days, 1)
                if recent_daily_rate > daily_rate * 3 and recent_daily_rate > 10:
                    anomalies.append(
                        {
                            "type": "volume_spike",
                            "contact_id": contact_hash if redact else contact,
                            "severity": "low" if recent_daily_rate < daily_rate * 5 else "medium",
                            "description": f"{recent_daily_rate:.0f} msgs/day (usually {daily_rate:.1f}/day)",
                            "detected_date": datetime.now().date().isoformat(),
                        }
                    )

                # Detect volume drop
                elif recent_daily_rate < daily_rate * 0.2 and daily_rate > 1:
                    anomalies.append(
                        {
                            "type": "volume_drop",
                            "contact_id": contact_hash if redact else contact,
                            "severity": "low",
                            "description": f"{recent_daily_rate:.1f} msgs/day (usually {daily_rate:.1f}/day)",
                            "detected_date": datetime.now().date().isoformat(),
                        }
                    )

        # Apply sensitivity filter
        if sensitivity < 1.0:
            severity_scores = {"low": 0.3, "medium": 0.6, "high": 0.9}
            anomalies = [
                a for a in anomalies if severity_scores.get(a["severity"], 0) >= (1 - sensitivity)
            ]

        # Sort by severity
        severity_order = {"high": 0, "medium": 1, "low": 2}
        anomalies.sort(key=lambda x: severity_order.get(x["severity"], 3))

        result = {
            "anomalies": anomalies[:20],  # Limit to 20 most significant
            "scan_period": f"Last {days} days",
            "contacts_scanned": len(baselines),
            "sensitivity": sensitivity,
        }

        if redact:
            result = apply_privacy_filters(result)

        return result

    except Exception as e:
        logger.error(f"Error scanning for anomalies: {e}")
        return {"error": str(e), "error_type": "analysis_error"}


async def network_intelligence_tool(
    db_path: str, min_messages: int = 10, days: int = 180, redact: bool = True
) -> Dict[str, Any]:
    """Analyze social network structure from group chats."""
    try:
        # Expand path
        db_path = Path(db_path).expanduser()

        # Get database connection
        db = await get_database(str(db_path))

        # Find group chats
        group_query = """
        SELECT 
            c.chat_identifier,
            c.display_name,
            COUNT(DISTINCT chj.handle_id) as participants,
            COUNT(DISTINCT m.ROWID) as message_count
        FROM chat c
        JOIN chat_message_join cmj ON c.ROWID = cmj.chat_id
        JOIN message m ON cmj.message_id = m.ROWID
        JOIN chat_handle_join chj ON c.ROWID = chj.chat_id
        WHERE c.style = 45  -- Group chat style
        AND datetime(m.date/1000000000 + 978307200, 'unixepoch') > datetime('now', '-' || ? || ' days')
        GROUP BY c.ROWID
        HAVING message_count >= ?
        """

        cursor = await db.execute(group_query, [days, min_messages])
        groups = await cursor.fetchall()

        # Build network graph
        nodes = set()
        edges = defaultdict(int)
        node_groups = defaultdict(set)

        for chat_id, display_name, participants, msg_count in groups:
            # Get participants in this group
            participant_query = """
            SELECT DISTINCT h.id
            FROM chat_handle_join chj
            JOIN handle h ON chj.handle_id = h.ROWID
            WHERE chj.chat_id = (SELECT ROWID FROM chat WHERE chat_identifier = ?)
            """

            cursor = await db.execute(participant_query, [chat_id])
            members = [row[0] for row in await cursor.fetchall()]

            # Add nodes
            for member in members:
                member_hash = hashlib.sha256(member.encode()).hexdigest()[:12] + "..."
                nodes.add(member_hash if redact else member)

                # Track group membership
                group_label = "group_" + hashlib.sha256(chat_id.encode()).hexdigest()[:8]
                node_groups[member_hash if redact else member].add(group_label)

            # Add edges between all participants
            for i, member1 in enumerate(members):
                for member2 in members[i + 1 :]:
                    m1_hash = hashlib.sha256(member1.encode()).hexdigest()[:12] + "..."
                    m2_hash = hashlib.sha256(member2.encode()).hexdigest()[:12] + "..."

                    edge = tuple(
                        sorted([m1_hash if redact else member1, m2_hash if redact else member2])
                    )
                    edges[edge] += msg_count  # Weight by message count

        # Calculate network statistics
        total_nodes = len(nodes)
        total_edges = len(edges)

        # Calculate density
        max_edges = (total_nodes * (total_nodes - 1)) / 2
        density = total_edges / max_edges if max_edges > 0 else 0

        # Find key connectors (highest degree centrality)
        node_degrees = defaultdict(int)
        for (n1, n2), weight in edges.items():
            node_degrees[n1] += 1
            node_degrees[n2] += 1

        # Calculate centrality
        key_connectors = []
        for node, degree in sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:5]:
            centrality = degree / (total_nodes - 1) if total_nodes > 1 else 0
            bridges = len(node_groups.get(node, set()))
            key_connectors.append(
                {"contact_id": node, "centrality": round(centrality, 2), "bridges": bridges}
            )

        # Detect communities (simple approach based on shared groups)
        communities = []
        for group_id in set().union(*node_groups.values()):
            members = [n for n, groups in node_groups.items() if group_id in groups]
            if len(members) >= 3:
                # Calculate cohesion (how connected members are)
                internal_edges = sum(1 for (n1, n2) in edges if n1 in members and n2 in members)
                max_internal = (len(members) * (len(members) - 1)) / 2
                cohesion = internal_edges / max_internal if max_internal > 0 else 0

                communities.append(
                    {
                        "id": len(communities),
                        "size": len(members),
                        "label": f"group_{len(communities)}",
                        "cohesion": round(cohesion, 2),
                    }
                )

        result = {
            "network_stats": {
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "density": round(density, 3),
                "communities": len(communities),
            },
            "key_connectors": key_connectors,
            "communities": sorted(communities, key=lambda x: x["size"], reverse=True)[:5],
        }

        if redact:
            result = apply_privacy_filters(result)

        return result

    except Exception as e:
        logger.error(f"Error analyzing network intelligence: {e}")
        return {"error": str(e), "error_type": "analysis_error"}


async def sample_messages_tool(
    db_path: str, contact_id: str, limit: int = 10, days: int = 30, redact: bool = True
) -> Dict[str, Any]:
    """Get heavily redacted sample messages for context."""
    try:
        # Expand path
        db_path = Path(db_path).expanduser()

        # Get database connection
        db = await get_database(str(db_path))

        # Query recent messages
        query = """
        SELECT 
            m.text,
            m.is_from_me,
            m.date,
            m.cache_has_attachments
        FROM message m
        JOIN handle h ON m.handle_id = h.ROWID
        WHERE h.id = ?
        AND m.text IS NOT NULL
        AND datetime(m.date/1000000000 + 978307200, 'unixepoch') > datetime('now', '-' || ? || ' days')
        ORDER BY m.date DESC
        LIMIT ?
        """

        cursor = await db.execute(query, [contact_id, days, limit * 2])  # Get extra for filtering
        messages = await cursor.fetchall()

        samples = []

        for text, is_from_me, date, has_attachments in messages:
            # Convert timestamp
            timestamp = datetime.fromtimestamp(date / 1000000000 + 978307200)

            # Heavy redaction
            if redact:
                # Redact patterns
                import re

                # Phone numbers
                text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]", text)
                # Emails
                text = re.sub(r"\S+@\S+\.\S+", "[EMAIL]", text)
                # URLs
                text = re.sub(r"https?://\S+", "[URL]", text)
                # Numbers (potential addresses, codes, etc)
                text = re.sub(r"\b\d{3,}\b", "[NUMBER]", text)
                # Names (basic - words following "Hi", "Hey", "Dear", etc)
                text = re.sub(
                    r"(?:Hi|Hey|Hello|Dear)\s+\w+",
                    lambda m: m.group(0).split()[0] + " [NAME]",
                    text,
                )

                # Further redact by keeping only structure
                words = text.split()
                if len(words) > 5:
                    # Keep first 2 and last 2 words
                    preview = " ".join(words[:2] + ["[REDACTED]"] + words[-2:])
                else:
                    preview = " ".join(["[REDACTED]" if len(w) > 3 else w for w in words])
            else:
                preview = text[:100] + "..." if len(text) > 100 else text

            samples.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "is_from_me": bool(is_from_me),
                    "preview": preview,
                    "has_attachment": bool(has_attachments),
                    "word_count": len(text.split()),
                }
            )

            if len(samples) >= limit:
                break

        # Reverse to show chronological order
        samples.reverse()

        result = {
            "contact_id": (
                hashlib.sha256(contact_id.encode()).hexdigest()[:12] + "..."
                if redact
                else contact_id
            ),
            "samples": samples,
            "privacy_note": (
                "Heavy redaction applied. PII removed." if redact else "No redaction applied"
            ),
        }

        if redact:
            result = apply_privacy_filters(result)

        return result

    except Exception as e:
        logger.error(f"Error getting sample messages: {e}")
        return {"error": str(e), "error_type": "analysis_error"}
