"""Communication pattern analysis tools."""

import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from imessage_mcp_server.db import get_database
from imessage_mcp_server.privacy import apply_privacy_filters

logger = logging.getLogger(__name__)


async def conversation_topics_tool(
    db_path: str,
    contact_id: Optional[str] = None,
    days: int = 90,
    min_frequency: int = 3,
    redact: bool = True,
) -> Dict[str, Any]:
    """Extract conversation topics using keyword analysis."""
    try:
        # Expand path
        db_path = Path(db_path).expanduser()

        # Get database connection
        db = await get_database(str(db_path))

        # Build query
        query = """
        SELECT m.text, m.is_from_me, m.date
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

        # Execute query
        cursor = await db.execute(query, params)
        messages = await cursor.fetchall()

        # Analyze topics
        word_freq = Counter()
        topic_words = defaultdict(list)
        total_messages = len(messages)

        # Common topic keywords
        topic_keywords = {
            "work": [
                "meeting",
                "project",
                "deadline",
                "email",
                "office",
                "boss",
                "team",
                "presentation",
            ],
            "family": [
                "kids",
                "mom",
                "dad",
                "dinner",
                "weekend",
                "family",
                "home",
                "sister",
                "brother",
            ],
            "travel": [
                "trip",
                "flight",
                "vacation",
                "hotel",
                "airport",
                "visit",
                "travel",
                "destination",
            ],
            "food": ["lunch", "dinner", "restaurant", "food", "eat", "cook", "meal", "coffee"],
            "health": [
                "doctor",
                "appointment",
                "health",
                "sick",
                "medicine",
                "hospital",
                "feel",
                "pain",
            ],
            "social": ["party", "drinks", "hangout", "meet", "friends", "bar", "club", "event"],
            "tech": ["app", "phone", "computer", "software", "update", "bug", "code", "website"],
            "finance": ["money", "pay", "bill", "bank", "budget", "expense", "cost", "price"],
        }

        for msg_text, _, _ in messages:
            if msg_text:
                words = msg_text.lower().split()
                for word in words:
                    # Clean word
                    word = word.strip('.,!?";:')
                    if len(word) > 3:  # Skip short words
                        word_freq[word] += 1

                        # Categorize by topic
                        for topic, keywords in topic_keywords.items():
                            if word in keywords:
                                topic_words[topic].append(word)

        # Build topic results
        topics = []
        for topic, words in topic_words.items():
            freq = len(words)
            if freq >= min_frequency:
                unique_words = list(set(words))[:5]  # Top 5 unique keywords
                topics.append(
                    {
                        "topic": topic,
                        "frequency": freq,
                        "keywords": (
                            unique_words if not redact else ["[REDACTED]"] * len(unique_words)
                        ),
                    }
                )

        # Sort by frequency
        topics.sort(key=lambda x: x["frequency"], reverse=True)

        # Find emerging topics (high frequency words not in predefined topics)
        covered_words = set()
        for words in topic_words.values():
            covered_words.update(words)

        emerging = []
        for word, count in word_freq.most_common(20):
            if word not in covered_words and count >= min_frequency:
                emerging.append({"word": word if not redact else "[REDACTED]", "count": count})
                if len(emerging) >= 5:
                    break

        result = {
            "contact_id": contact_id,
            "days": days,
            "topics": topics[:10],  # Top 10 topics
            "emerging_keywords": emerging,
            "total_messages_analyzed": total_messages,
        }

        if redact:
            result = apply_privacy_filters(result)

        return result

    except Exception as e:
        logger.error(f"Error analyzing conversation topics: {e}")
        return {"error": str(e), "error_type": "analysis_error"}


async def sentiment_evolution_tool(
    db_path: str,
    contact_id: Optional[str] = None,
    days: int = 180,
    bucket_days: int = 7,
    redact: bool = True,
) -> Dict[str, Any]:
    """Track sentiment changes over time."""
    try:
        # Expand path
        db_path = Path(db_path).expanduser()

        # Get database connection
        db = await get_database(str(db_path))

        # Build query
        query = """
        SELECT m.text, m.is_from_me, m.date
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

        query += " ORDER BY m.date"

        # Execute query
        cursor = await db.execute(query, params)
        messages = await cursor.fetchall()

        # Simple sentiment analysis based on keywords
        positive_words = {
            "good",
            "great",
            "love",
            "awesome",
            "happy",
            "thanks",
            "wonderful",
            "excellent",
            "best",
            "amazing",
        }
        negative_words = {
            "bad",
            "hate",
            "terrible",
            "awful",
            "sad",
            "sorry",
            "worst",
            "horrible",
            "angry",
            "upset",
        }

        # Group messages by time bucket
        buckets = defaultdict(lambda: {"positive": 0, "negative": 0, "neutral": 0, "count": 0})

        for msg_text, is_from_me, date in messages:
            if msg_text:
                # Convert date to datetime
                timestamp = datetime.fromtimestamp(date / 1000000000 + 978307200)
                bucket_key = timestamp.date() - timedelta(
                    days=timestamp.date().weekday()
                )  # Week start

                # Analyze sentiment
                words = set(msg_text.lower().split())
                pos_count = len(words & positive_words)
                neg_count = len(words & negative_words)

                if pos_count > neg_count:
                    buckets[bucket_key]["positive"] += 1
                elif neg_count > pos_count:
                    buckets[bucket_key]["negative"] += 1
                else:
                    buckets[bucket_key]["neutral"] += 1

                buckets[bucket_key]["count"] += 1

        # Build time series
        series = []
        for bucket_date in sorted(buckets.keys()):
            bucket = buckets[bucket_date]
            if bucket["count"] > 0:
                # Calculate sentiment score (-1 to 1)
                score = (bucket["positive"] - bucket["negative"]) / bucket["count"]
                series.append(
                    {
                        "date": bucket_date.isoformat(),
                        "score": round(score, 2),
                        "messages": bucket["count"],
                    }
                )

        # Calculate summary statistics
        if series:
            scores = [s["score"] for s in series]
            mean_score = sum(scores) / len(scores)

            # Detect trend
            if len(scores) >= 2:
                first_half = scores[: len(scores) // 2]
                second_half = scores[len(scores) // 2 :]
                trend = (
                    "improving"
                    if sum(second_half) / len(second_half) > sum(first_half) / len(first_half)
                    else "declining"
                )
            else:
                trend = "stable"

            # Calculate volatility
            if len(scores) > 1:
                volatility = sum(abs(scores[i] - scores[i - 1]) for i in range(1, len(scores))) / (
                    len(scores) - 1
                )
            else:
                volatility = 0

            summary = {
                "mean": round(mean_score, 2),
                "trend": trend,
                "volatility": round(volatility, 2),
            }
        else:
            summary = {"mean": 0, "trend": "no_data", "volatility": 0}

        result = {
            "contact_id": contact_id,
            "series": series[-20:],  # Last 20 buckets
            "summary": summary,
        }

        if redact:
            result = apply_privacy_filters(result)

        return result

    except Exception as e:
        logger.error(f"Error analyzing sentiment evolution: {e}")
        return {"error": str(e), "error_type": "analysis_error"}


async def response_time_distribution_tool(
    db_path: str, contact_id: Optional[str] = None, days: int = 90, redact: bool = True
) -> Dict[str, Any]:
    """Analyze response time patterns."""
    try:
        # Expand path
        db_path = Path(db_path).expanduser()

        # Get database connection
        db = await get_database(str(db_path))

        # Query to get conversation pairs
        query = """
        SELECT 
            m1.date as msg_date,
            m1.is_from_me as msg_from_me,
            m2.date as response_date,
            m2.is_from_me as response_from_me,
            m1.handle_id
        FROM message m1
        INNER JOIN message m2 ON m2.handle_id = m1.handle_id
        WHERE m1.text IS NOT NULL 
        AND m2.text IS NOT NULL
        AND m2.date > m1.date
        AND m2.date < m1.date + 86400000000000  -- Within 24 hours
        AND m1.is_from_me != m2.is_from_me
        AND datetime(m1.date/1000000000 + 978307200, 'unixepoch') > datetime('now', '-' || ? || ' days')
        """

        params = [days]

        if contact_id:
            query += " AND m1.handle_id IN (SELECT h.ROWID FROM handle h WHERE h.id = ?)"
            params.append(contact_id)

        query += " ORDER BY m1.date"

        # Execute query
        cursor = await db.execute(query, params)
        pairs = await cursor.fetchall()

        # Calculate response times
        your_times = []
        their_times = []

        # Track which messages have been used as responses
        used_responses = set()

        for msg_date, msg_from_me, resp_date, resp_from_me, handle in pairs:
            if resp_date not in used_responses:
                response_time = (resp_date - msg_date) / 60000000000  # Convert to minutes

                if msg_from_me:
                    their_times.append(response_time)
                else:
                    your_times.append(response_time)

                used_responses.add(resp_date)

        # Calculate percentiles
        def calculate_percentiles(times: List[float]) -> Dict[str, float]:
            if not times:
                return {"median_minutes": 0, "p25_minutes": 0, "p75_minutes": 0, "p95_minutes": 0}

            times.sort()
            n = len(times)
            return {
                "median_minutes": round(times[n // 2], 1),
                "p25_minutes": round(times[n // 4], 1),
                "p75_minutes": round(times[3 * n // 4], 1),
                "p95_minutes": round(times[int(0.95 * n)], 1) if n >= 20 else round(times[-1], 1),
            }

        result = {
            "contact_id": contact_id,
            "your_response_times": calculate_percentiles(your_times),
            "their_response_times": calculate_percentiles(their_times),
            "analysis_period_days": days,
            "total_conversations_analyzed": len(your_times) + len(their_times),
        }

        if redact:
            result = apply_privacy_filters(result)

        return result

    except Exception as e:
        logger.error(f"Error analyzing response times: {e}")
        return {"error": str(e), "error_type": "analysis_error"}


async def cadence_calendar_tool(
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
    Generate communication frequency analysis with visualizations and time series.
    
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
        Dict containing cadence analysis with visualizations
    """
    # Import enhanced functionality
    from imessage_mcp_server.tools.communication_enhanced import enhanced_cadence_calendar_tool
    
    return await enhanced_cadence_calendar_tool(
        db_path=db_path,
        contact_id=contact_id,
        time_period=time_period,
        granularity=granularity,
        include_visualizations=include_visualizations,
        include_time_series=include_time_series,
        comparison_contacts=comparison_contacts,
        redact=redact
    )
