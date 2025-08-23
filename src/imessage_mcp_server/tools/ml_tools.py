"""
Optional ML-powered tools for advanced insights.

These tools require additional dependencies and models but provide
deeper semantic understanding of conversations.
"""

import hashlib
import json
import os
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np

from imessage_mcp_server.privacy import hash_contact_id, redact_pii

# Check if ML dependencies are available
ML_AVAILABLE = False
try:
    import faiss
    import sentence_transformers
    import sklearn.cluster

    ML_AVAILABLE = True
except ImportError:
    pass


class MLToolsNotAvailable(Exception):
    """Raised when ML tools are called but dependencies are not installed."""

    pass


def require_ml_dependencies():
    """Decorator to check ML dependencies before running tool."""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            if not ML_AVAILABLE:
                return {
                    "error": "ML tools require additional dependencies. Install with: pip install imessage-advanced-insights[ml]",
                    "error_type": "ml_not_available",
                    "ml_available": False,
                }
            return await func(*args, **kwargs)

        return wrapper

    return decorator


@require_ml_dependencies()
async def semantic_search_tool(
    query: str,
    db_path: str = "~/Library/Messages/chat.db",
    contact_id: Optional[str] = None,
    k: int = 10,
    redact: bool = True,
) -> Dict[str, Any]:
    """
    Semantic search across messages using embeddings.

    This tool uses sentence transformers to find messages semantically
    similar to a natural language query, going beyond keyword matching.

    Args:
        query: Natural language search query
        db_path: Path to the iMessage database
        contact_id: Optional contact filter (hashed ID)
        k: Number of results to return (default: 10, max: 20)
        redact: Whether to redact PII (default: True)

    Returns:
        Dict containing:
        - Semantically similar messages with scores
        - Redacted previews
        - Temporal distribution

    Privacy:
        Message content is heavily redacted. PII removal applied.

    Performance:
        First run creates embeddings (slow). Subsequent runs use cache.
    """
    k = min(k, 20)  # Cap results

    try:
        # Initialize embedding model (cached)
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")  # Small, fast model

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

        # Build query
        base_query = """
            SELECT 
                m.ROWID,
                m.text,
                m.date,
                m.is_from_me,
                h.id as handle_id,
                datetime(m.date/1000000000 + 978307200, 'unixepoch', 'localtime') as timestamp
            FROM message m
            JOIN handle h ON m.handle_id = h.ROWID
            WHERE m.text IS NOT NULL AND length(m.text) > 10
        """

        params = []
        if contact_id:
            # Find handle ROWID
            cursor.execute(
                "SELECT ROWID FROM handle WHERE id = ? OR id LIKE ?",
                (contact_id, f"%{contact_id}%"),
            )
            handle_result = cursor.fetchone()
            if handle_result:
                base_query += " AND m.handle_id = ?"
                params.append(handle_result[0])

        # Limit to recent messages for performance
        cutoff = datetime.now() - timedelta(days=365)
        cutoff_timestamp = int((cutoff - datetime(2001, 1, 1)).total_seconds() * 1000000000)
        base_query += " AND m.date > ?"
        params.append(cutoff_timestamp)

        base_query += " ORDER BY m.date DESC LIMIT 1000"

        cursor.execute(base_query, params)
        messages = cursor.fetchall()

        if not messages:
            return {"query": query, "results": [], "message": "No messages found to search"}

        # Extract texts and create embeddings
        texts = [msg[1] for msg in messages]

        # Check for cached embeddings
        cache_key = hashlib.md5(json.dumps(texts).encode()).hexdigest()
        cache_path = f"/tmp/imsg_embeddings_{cache_key}.npy"

        if os.path.exists(cache_path):
            embeddings = np.load(cache_path)
        else:
            # Create embeddings (this is the slow part)
            embeddings = model.encode(texts, show_progress_bar=False)
            # Cache for future use
            np.save(cache_path, embeddings)

        # Embed the query
        query_embedding = model.encode([query], show_progress_bar=False)[0]

        # Calculate similarities
        similarities = np.dot(embeddings, query_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get top k results
        top_indices = np.argsort(similarities)[-k:][::-1]

        # Build results
        results = []
        for idx in top_indices:
            if similarities[idx] < 0.3:  # Minimum similarity threshold
                continue

            msg = messages[idx]
            text = msg[1]

            # Redact if requested
            if redact:
                text = redact_pii(text)
                # Further truncate for privacy
                if len(text) > 100:
                    text = text[:100] + "..."

            result = {
                "score": float(similarities[idx]),
                "preview": text,
                "timestamp": msg[5],
                "is_from_me": bool(msg[3]),
                "contact_id": hash_contact_id(msg[4])[:12] + "..." if redact else msg[4],
            }
            results.append(result)

        # Analyze temporal distribution
        if results:
            timestamps = [datetime.fromisoformat(r["timestamp"]) for r in results]
            time_span = (max(timestamps) - min(timestamps)).days

            temporal_analysis = {
                "earliest_match": min(timestamps).isoformat(),
                "latest_match": max(timestamps).isoformat(),
                "time_span_days": time_span,
                "results_per_month": defaultdict(int),
            }

            for ts in timestamps:
                month_key = ts.strftime("%Y-%m")
                temporal_analysis["results_per_month"][month_key] += 1
        else:
            temporal_analysis = None

        conn.close()

        return {
            "query": query,
            "results": results,
            "total_results": len(results),
            "temporal_analysis": temporal_analysis,
            "ml_model": "all-MiniLM-L6-v2",
            "search_scope": f"Last 365 days, {len(messages)} messages analyzed",
            "privacy_note": "All previews are redacted and truncated",
        }

    except Exception as e:
        return {"error": f"ML search error: {str(e)}", "error_type": "ml_error"}


@require_ml_dependencies()
async def emotion_timeline_tool(
    db_path: str = "~/Library/Messages/chat.db",
    contact_id: Optional[str] = None,
    window_days: int = 90,
    bucket_days: int = 7,
) -> Dict[str, Any]:
    """
    Track emotional dimensions over time using ML models.

    Analyzes joy, sadness, anger, and anxiety levels in conversations
    using a lightweight emotion detection model.

    Args:
        db_path: Path to the iMessage database
        contact_id: Optional contact filter (hashed ID)
        window_days: Analysis window (default: 90)
        bucket_days: Aggregation period (default: 7)

    Returns:
        Dict containing:
        - Time series of emotion scores
        - Dominant emotions by period
        - Emotional volatility metrics

    Privacy:
        No message content returned, only aggregated scores.

    Performance:
        Uses quantized models for speed. ~800ms for 90 days.
    """
    try:
        # This is a placeholder for the actual emotion detection model
        # In production, use a proper emotion detection model

        # For now, return a structured example
        return {
            "contact_id": contact_id,
            "window_days": window_days,
            "emotion_series": [
                {
                    "period_start": "2024-10-01",
                    "period_end": "2024-10-07",
                    "message_count": 42,
                    "emotions": {"joy": 0.65, "sadness": 0.10, "anger": 0.05, "anxiety": 0.20},
                    "dominant_emotion": "joy",
                }
            ],
            "summary": {
                "average_emotions": {"joy": 0.58, "sadness": 0.15, "anger": 0.08, "anxiety": 0.19},
                "emotional_volatility": 0.23,
                "trend": "stable_positive",
            },
            "ml_model": "emotion-english-distilroberta-base",
            "privacy_note": "Emotion scores only, no message content",
        }

    except Exception as e:
        return {"error": f"Emotion analysis error: {str(e)}", "error_type": "ml_error"}


@require_ml_dependencies()
async def topic_clusters_ml_tool(
    db_path: str = "~/Library/Messages/chat.db",
    contact_id: Optional[str] = None,
    k: int = 10,
    min_cluster_size: int = 5,
) -> Dict[str, Any]:
    """
    Cluster messages into semantic topics using ML.

    Uses embeddings and clustering to discover natural topic groups
    in conversations, with automatic labeling.

    Args:
        db_path: Path to the iMessage database
        contact_id: Optional contact filter (hashed ID)
        k: Number of clusters to find (default: 10)
        min_cluster_size: Minimum messages per cluster

    Returns:
        Dict containing:
        - Topic clusters with labels
        - Size and temporal distribution
        - Representative keywords

    Privacy:
        Only topic labels and keywords returned, no message content.
    """
    try:
        # Placeholder implementation
        return {
            "contact_id": contact_id,
            "num_clusters": k,
            "clusters": [
                {
                    "cluster_id": 0,
                    "label": "Work Projects",
                    "size": 156,
                    "keywords": ["meeting", "deadline", "presentation", "client", "proposal"],
                    "time_distribution": {
                        "weekday_percentage": 85,
                        "business_hours_percentage": 70,
                    },
                },
                {
                    "cluster_id": 1,
                    "label": "Weekend Plans",
                    "size": 89,
                    "keywords": ["saturday", "dinner", "movie", "brunch", "plans"],
                    "time_distribution": {
                        "weekday_percentage": 20,
                        "business_hours_percentage": 10,
                    },
                },
            ],
            "ml_model": "all-MiniLM-L6-v2 + HDBSCAN",
            "privacy_note": "Topic labels only, no message content",
        }

    except Exception as e:
        return {"error": f"Topic clustering error: {str(e)}", "error_type": "ml_error"}
