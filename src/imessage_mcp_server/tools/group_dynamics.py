"""
Group Dynamics Analyzer tool for comprehensive group chat analysis.

Analyzes participation patterns, influence networks, subgroup formation,
and overall group health metrics.
"""

import asyncio
import logging
import statistics
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from sklearn.cluster import SpectralClustering

from imessage_mcp_server.db import get_database
from imessage_mcp_server.privacy import apply_privacy_filters, hash_contact_id

logger = logging.getLogger(__name__)


async def group_dynamics_tool(
    group_id: str,
    analysis_type: str = "comprehensive",
    time_period: str = "90d",
    db_path: str = "~/Library/Messages/chat.db",
    redact: bool = True,
) -> Dict[str, Any]:
    """
    Analyze group chat dynamics and social structures.
    
    Provides insights into:
    - Participation patterns and balance
    - Influence networks and opinion leaders
    - Subgroup/clique detection
    - Group health metrics
    - Communication flow patterns
    
    Args:
        group_id: Group chat identifier
        analysis_type: Type of analysis ("comprehensive", "participation", "influence", "health")
        time_period: Analysis period (e.g., "30d", "90d", "6m")
        db_path: Path to iMessage database
        redact: Whether to apply privacy filters
        
    Returns:
        Dict containing group dynamics analysis
    """
    try:
        # Parse time period
        days = _parse_time_period(time_period)
        
        # Expand path
        db_path = Path(db_path).expanduser()
        db = await get_database(str(db_path))
        
        # Fetch group messages
        messages = await _fetch_group_messages(db, group_id, days)
        
        if not messages:
            return {
                "error": "No messages found for group analysis",
                "group_id": hash_contact_id(group_id) if redact else group_id,
                "time_period": time_period,
            }
        
        # Get unique participants
        participants = _extract_participants(messages)
        
        # Build interaction matrix
        interaction_matrix = _build_interaction_matrix(messages, participants)
        
        # Perform modular analysis based on type
        analysis_results = {}
        
        if analysis_type in ["comprehensive", "participation"]:
            analysis_results["participation"] = await _analyze_participation(
                messages, participants
            )
        
        if analysis_type in ["comprehensive", "influence"]:
            analysis_results["influence"] = await _analyze_influence(
                messages, participants, interaction_matrix
            )
        
        if analysis_type in ["comprehensive", "subgroups"]:
            analysis_results["subgroups"] = await _detect_subgroups(
                interaction_matrix, participants
            )
        
        if analysis_type in ["comprehensive", "health"]:
            analysis_results["health"] = await _assess_group_health(
                messages, participants, interaction_matrix
            )
        
        # Calculate group personality
        group_personality = _determine_group_personality(analysis_results)
        
        # Generate insights
        insights = _generate_group_insights(analysis_results, participants)
        
        # Build result
        result = {
            "group_id": hash_contact_id(group_id) if redact else group_id,
            "time_period": time_period,
            "analysis_type": analysis_type,
            "participant_count": len(participants),
            "message_count": len(messages),
            "group_personality": group_personality,
            "dynamics": analysis_results,
            "insights": insights,
            "analysis_period": {
                "start": messages[0]["date"].isoformat() if messages else None,
                "end": messages[-1]["date"].isoformat() if messages else None,
            },
        }
        
        # Apply redaction if requested
        if redact:
            result = _redact_group_results(result, participants)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in group dynamics analysis: {e}")
        return {
            "error": str(e),
            "error_type": "group_analysis_error",
        }


async def _fetch_group_messages(
    db: Any, group_id: str, days: int
) -> List[Dict[str, Any]]:
    """Fetch messages from a group chat."""
    query = """
    SELECT 
        m.text,
        m.is_from_me,
        m.date,
        m.handle_id,
        h.id as sender_id,
        c.chat_identifier
    FROM message m
    JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
    JOIN chat c ON cmj.chat_id = c.ROWID
    LEFT JOIN handle h ON m.handle_id = h.ROWID
    WHERE c.chat_identifier = ?
    AND m.text IS NOT NULL
    AND m.date > (strftime('%s', 'now') - ?) * 1000000000
    ORDER BY m.date
    """
    
    seconds = days * 86400
    cursor = await db.execute(query, [group_id, seconds])
    rows = await cursor.fetchall()
    
    messages = []
    for row in rows:
        sender_id = "me" if row[1] else (row[4] or f"participant_{row[3]}")
        messages.append({
            "text": row[0],
            "is_from_me": row[1],
            "date": datetime.fromtimestamp(row[2] / 1000000000 + 978307200),
            "sender_id": sender_id,
            "length": len(row[0]),
        })
    
    return messages


def _extract_participants(messages: List[Dict]) -> Set[str]:
    """Extract unique participants from messages."""
    participants = set()
    for msg in messages:
        participants.add(msg["sender_id"])
    return participants


def _build_interaction_matrix(
    messages: List[Dict], participants: Set[str]
) -> Dict[str, Dict[str, int]]:
    """Build interaction matrix showing who responds to whom."""
    # Initialize matrix
    matrix = {p1: {p2: 0 for p2 in participants} for p1 in participants}
    
    # Simple approach: assume sequential messages are interactions
    for i in range(1, len(messages)):
        prev_msg = messages[i-1]
        curr_msg = messages[i]
        
        # If different senders and within 30 minutes, count as interaction
        if prev_msg["sender_id"] != curr_msg["sender_id"]:
            time_diff = (curr_msg["date"] - prev_msg["date"]).total_seconds()
            if time_diff < 1800:  # 30 minutes
                matrix[curr_msg["sender_id"]][prev_msg["sender_id"]] += 1
    
    return matrix


async def _analyze_participation(
    messages: List[Dict], participants: Set[str]
) -> Dict[str, Any]:
    """Analyze participation patterns."""
    # Message distribution
    message_counts = Counter(msg["sender_id"] for msg in messages)
    total_messages = len(messages)
    
    # Calculate metrics
    participation_data = {}
    for participant in participants:
        count = message_counts.get(participant, 0)
        participation_data[participant] = {
            "message_count": count,
            "percentage": round((count / total_messages * 100), 1) if total_messages > 0 else 0,
            "avg_message_length": 0,
            "active_hours": set(),
            "conversation_starts": 0,
        }
    
    # Additional metrics
    for i, msg in enumerate(messages):
        participant = msg["sender_id"]
        
        # Average message length
        if participant in participation_data:
            current_avg = participation_data[participant]["avg_message_length"]
            current_count = participation_data[participant]["message_count"]
            if current_count > 0:
                new_avg = (current_avg * (current_count - 1) + msg["length"]) / current_count
                participation_data[participant]["avg_message_length"] = new_avg
        
        # Active hours
        hour = msg["date"].hour
        participation_data[participant]["active_hours"].add(hour)
        
        # Conversation starters (after 1+ hour gap)
        if i > 0:
            time_gap = (msg["date"] - messages[i-1]["date"]).total_seconds()
            if time_gap > 3600:  # 1 hour
                participation_data[participant]["conversation_starts"] += 1
    
    # Convert sets to lists for JSON serialization
    for p_data in participation_data.values():
        p_data["active_hours"] = sorted(list(p_data["active_hours"]))
        p_data["avg_message_length"] = round(p_data["avg_message_length"], 1)
    
    # Calculate balance metrics
    counts = [d["message_count"] for d in participation_data.values()]
    balance_score = 1 - (statistics.stdev(counts) / statistics.mean(counts)) if len(counts) > 1 and statistics.mean(counts) > 0 else 0
    
    # Identify roles
    sorted_participants = sorted(
        participation_data.items(),
        key=lambda x: x[1]["message_count"],
        reverse=True
    )
    
    roles = {
        "most_active": sorted_participants[0][0] if sorted_participants else None,
        "least_active": sorted_participants[-1][0] if len(sorted_participants) > 1 else None,
        "conversation_starters": [
            p for p, data in participation_data.items()
            if data["conversation_starts"] > len(messages) * 0.1
        ],
    }
    
    return {
        "distribution": participation_data,
        "balance_score": round(balance_score, 2),
        "roles": roles,
        "peak_hours": _calculate_peak_hours(messages),
    }


async def _analyze_influence(
    messages: List[Dict], participants: Set[str], interaction_matrix: Dict
) -> Dict[str, Any]:
    """Analyze influence networks using graph theory."""
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for participant in participants:
        G.add_node(participant)
    
    # Add weighted edges based on interactions
    for sender, receivers in interaction_matrix.items():
        for receiver, count in receivers.items():
            if count > 0:
                G.add_edge(sender, receiver, weight=count)
    
    # Calculate centrality metrics
    influence_metrics = {}
    
    try:
        # Different centrality measures
        in_degree = nx.in_degree_centrality(G)
        out_degree = nx.out_degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        eigenvector = nx.eigenvector_centrality(G, max_iter=100) if len(G) > 1 else {}
    except:
        # Handle cases where graph metrics can't be calculated
        in_degree = {n: 0 for n in participants}
        out_degree = {n: 0 for n in participants}
        betweenness = {n: 0 for n in participants}
        eigenvector = {n: 0 for n in participants}
    
    # Compile influence scores
    for participant in participants:
        influence_metrics[participant] = {
            "receives_responses": round(in_degree.get(participant, 0), 3),
            "initiates_responses": round(out_degree.get(participant, 0), 3),
            "bridge_score": round(betweenness.get(participant, 0), 3),
            "influence_score": round(eigenvector.get(participant, 0), 3),
        }
    
    # Identify key members
    sorted_by_influence = sorted(
        influence_metrics.items(),
        key=lambda x: x[1]["influence_score"],
        reverse=True
    )
    
    key_members = {
        "influencers": [
            p for p, m in sorted_by_influence[:3]
            if m["influence_score"] > 0.1
        ],
        "connectors": [
            p for p, m in influence_metrics.items()
            if m["bridge_score"] > 0.1
        ],
        "conversation_drivers": [
            p for p, m in influence_metrics.items()
            if m["initiates_responses"] > 0.2
        ],
    }
    
    return {
        "metrics": influence_metrics,
        "key_members": key_members,
        "network_density": nx.density(G) if len(G) > 1 else 0,
    }


async def _detect_subgroups(
    interaction_matrix: Dict, participants: Set[str]
) -> Dict[str, Any]:
    """Detect subgroups using spectral clustering."""
    if len(participants) < 3:
        return {
            "clusters": [],
            "modularity": 0,
            "description": "Too few participants for subgroup analysis",
        }
    
    # Convert interaction matrix to numpy array
    participant_list = sorted(participants)
    n = len(participant_list)
    matrix = np.zeros((n, n))
    
    for i, p1 in enumerate(participant_list):
        for j, p2 in enumerate(participant_list):
            if p1 in interaction_matrix and p2 in interaction_matrix[p1]:
                matrix[i][j] = interaction_matrix[p1][p2]
    
    # Make symmetric (undirected)
    matrix = matrix + matrix.T
    
    # Determine optimal number of clusters (2-4)
    n_clusters = min(3, max(2, len(participants) // 3))
    
    try:
        # Perform spectral clustering
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42
        )
        labels = clustering.fit_predict(matrix)
        
        # Group participants by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[label].append(participant_list[i])
        
        # Analyze each cluster
        cluster_analysis = []
        for label, members in clusters.items():
            # Calculate internal vs external interactions
            internal_interactions = 0
            external_interactions = 0
            
            for member in members:
                for other, count in interaction_matrix.get(member, {}).items():
                    if other in members:
                        internal_interactions += count
                    else:
                        external_interactions += count
            
            cohesion = internal_interactions / (internal_interactions + external_interactions) if (internal_interactions + external_interactions) > 0 else 0
            
            cluster_analysis.append({
                "id": f"cluster_{label}",
                "members": members,
                "size": len(members),
                "cohesion": round(cohesion, 2),
                "type": _determine_cluster_type(len(members), cohesion, len(participants)),
            })
        
        # Sort by size
        cluster_analysis.sort(key=lambda x: x["size"], reverse=True)
        
        return {
            "clusters": cluster_analysis,
            "modularity": _calculate_modularity(matrix, labels),
            "description": f"Found {len(clusters)} distinct subgroups",
        }
        
    except Exception as e:
        logger.error(f"Error in subgroup detection: {e}")
        return {
            "clusters": [],
            "modularity": 0,
            "description": "Could not detect subgroups",
        }


async def _assess_group_health(
    messages: List[Dict], participants: Set[str], interaction_matrix: Dict
) -> Dict[str, Any]:
    """Assess overall group health metrics."""
    # Calculate various health indicators
    
    # 1. Activity level
    total_messages = len(messages)
    date_range = (messages[-1]["date"] - messages[0]["date"]).days or 1
    daily_average = total_messages / date_range
    
    # 2. Inclusivity score (based on participation balance)
    message_counts = Counter(msg["sender_id"] for msg in messages)
    counts = list(message_counts.values())
    inclusivity = 1 - (statistics.stdev(counts) / statistics.mean(counts)) if len(counts) > 1 and statistics.mean(counts) > 0 else 0
    
    # 3. Responsiveness (interaction density)
    total_possible_interactions = len(participants) * (len(participants) - 1)
    actual_interactions = sum(
        1 for p1 in interaction_matrix
        for p2 in interaction_matrix[p1]
        if interaction_matrix[p1][p2] > 0 and p1 != p2
    )
    responsiveness = actual_interactions / total_possible_interactions if total_possible_interactions > 0 else 0
    
    # 4. Sentiment (simplified - would need proper sentiment analysis)
    positive_indicators = ["!", "😊", "😂", "❤️", "👍", "haha", "lol", "love", "great", "awesome"]
    negative_indicators = ["😢", "😡", "👎", "hate", "angry", "sad", "sorry", "bad"]
    
    positive_count = sum(
        1 for msg in messages
        if any(indicator in msg["text"].lower() for indicator in positive_indicators)
    )
    negative_count = sum(
        1 for msg in messages
        if any(indicator in msg["text"].lower() for indicator in negative_indicators)
    )
    
    sentiment_score = (positive_count - negative_count) / total_messages if total_messages > 0 else 0
    
    # 5. Engagement trend (compare recent vs older activity)
    midpoint = len(messages) // 2
    recent_messages = messages[midpoint:]
    older_messages = messages[:midpoint]
    
    recent_daily_avg = len(recent_messages) / ((recent_messages[-1]["date"] - recent_messages[0]["date"]).days or 1)
    older_daily_avg = len(older_messages) / ((older_messages[-1]["date"] - older_messages[0]["date"]).days or 1)
    
    engagement_trend = "increasing" if recent_daily_avg > older_daily_avg * 1.1 else "decreasing" if recent_daily_avg < older_daily_avg * 0.9 else "stable"
    
    # Calculate overall health score
    health_score = (
        min(100, daily_average * 10) * 0.25 +  # Activity
        inclusivity * 100 * 0.25 +              # Inclusivity
        responsiveness * 100 * 0.25 +           # Responsiveness
        (sentiment_score + 1) * 50 * 0.25       # Sentiment (normalized to 0-100)
    )
    
    return {
        "health_score": round(health_score, 1),
        "activity_level": {
            "daily_average": round(daily_average, 1),
            "total_messages": total_messages,
            "rating": "high" if daily_average > 20 else "moderate" if daily_average > 5 else "low",
        },
        "inclusivity_score": round(inclusivity * 100, 1),
        "responsiveness_score": round(responsiveness * 100, 1),
        "sentiment_score": round((sentiment_score + 1) * 50, 1),  # Normalized to 0-100
        "engagement_trend": engagement_trend,
        "health_indicators": _generate_health_indicators(
            health_score, inclusivity, responsiveness, sentiment_score
        ),
    }


def _calculate_peak_hours(messages: List[Dict]) -> List[int]:
    """Calculate peak activity hours."""
    hour_counts = Counter(msg["date"].hour for msg in messages)
    sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
    return [hour for hour, _ in sorted_hours[:3]]


def _determine_cluster_type(size: int, cohesion: float, total_participants: int) -> str:
    """Determine the type of a cluster based on its characteristics."""
    size_ratio = size / total_participants
    
    if size_ratio > 0.6:
        return "main_group"
    elif cohesion > 0.7:
        return "tight_clique"
    elif cohesion < 0.3:
        return "loose_association"
    else:
        return "subgroup"


def _calculate_modularity(matrix: np.ndarray, labels: np.ndarray) -> float:
    """Calculate modularity score for clustering quality."""
    # Simplified modularity calculation
    n = len(matrix)
    if n == 0:
        return 0
    
    total_edges = np.sum(matrix) / 2
    if total_edges == 0:
        return 0
    
    modularity = 0
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                expected = (np.sum(matrix[i]) * np.sum(matrix[j])) / (2 * total_edges)
                modularity += (matrix[i][j] - expected) / (2 * total_edges)
    
    return round(modularity, 3)


def _determine_group_personality(analysis_results: Dict) -> str:
    """Determine overall group personality based on metrics."""
    if not analysis_results:
        return "unknown"
    
    # Extract key metrics
    participation = analysis_results.get("participation", {})
    health = analysis_results.get("health", {})
    
    balance = participation.get("balance_score", 0)
    activity = health.get("activity_level", {}).get("rating", "moderate")
    inclusivity = health.get("inclusivity_score", 0)
    
    # Determine personality
    if balance > 0.7 and inclusivity > 70:
        if activity == "high":
            return "collaborative_active"
        else:
            return "collaborative_casual"
    elif balance < 0.4:
        return "hierarchical_formal"
    elif activity == "high" and inclusivity < 50:
        return "dominated_active"
    else:
        return "mixed_moderate"


def _generate_health_indicators(
    health_score: float, inclusivity: float, responsiveness: float, sentiment: float
) -> List[str]:
    """Generate health indicator descriptions."""
    indicators = []
    
    if health_score > 80:
        indicators.append("Healthy, thriving group")
    elif health_score > 60:
        indicators.append("Generally healthy with room for improvement")
    elif health_score > 40:
        indicators.append("Some concerning patterns detected")
    else:
        indicators.append("Group health needs attention")
    
    if inclusivity > 0.8:
        indicators.append("Excellent participation balance")
    elif inclusivity < 0.5:
        indicators.append("Participation is heavily skewed")
    
    if responsiveness > 0.7:
        indicators.append("High interaction between members")
    elif responsiveness < 0.3:
        indicators.append("Limited cross-member interaction")
    
    if sentiment > 75:
        indicators.append("Very positive group atmosphere")
    elif sentiment < 25:
        indicators.append("Negative sentiment detected")
    
    return indicators


def _generate_group_insights(
    analysis_results: Dict, participants: Set[str]
) -> Dict[str, Any]:
    """Generate actionable insights from analysis."""
    insights = {
        "patterns": [],
        "recommendations": [],
        "strengths": [],
        "concerns": [],
    }
    
    # Participation insights
    if "participation" in analysis_results:
        participation = analysis_results["participation"]
        
        if participation["balance_score"] > 0.7:
            insights["strengths"].append("Well-balanced participation across members")
        elif participation["balance_score"] < 0.4:
            insights["concerns"].append("Participation is dominated by few members")
            insights["recommendations"].append("Encourage quieter members to contribute")
        
        if participation["roles"].get("conversation_starters"):
            count = len(participation["roles"]["conversation_starters"])
            insights["patterns"].append(f"{count} members frequently initiate conversations")
    
    # Influence insights
    if "influence" in analysis_results:
        influence = analysis_results["influence"]
        
        if influence["key_members"]["influencers"]:
            insights["patterns"].append(
                f"{len(influence['key_members']['influencers'])} key influencers identified"
            )
        
        if influence["network_density"] < 0.3:
            insights["concerns"].append("Low interaction density - members not engaging with each other")
    
    # Subgroup insights
    if "subgroups" in analysis_results:
        subgroups = analysis_results["subgroups"]
        
        if len(subgroups["clusters"]) > 1:
            insights["patterns"].append(f"{len(subgroups['clusters'])} distinct subgroups detected")
            
            # Check for isolated clusters
            for cluster in subgroups["clusters"]:
                if cluster["cohesion"] > 0.8 and cluster["size"] < len(participants) * 0.3:
                    insights["concerns"].append("Highly isolated subgroup detected")
                    break
    
    # Health insights
    if "health" in analysis_results:
        health = analysis_results["health"]
        
        if health["health_score"] > 80:
            insights["strengths"].append("Group shows excellent health metrics")
        elif health["health_score"] < 50:
            insights["concerns"].append("Group health metrics are concerning")
        
        if health["engagement_trend"] == "decreasing":
            insights["concerns"].append("Group engagement is declining")
            insights["recommendations"].append("Plan engaging activities to revitalize the group")
        elif health["engagement_trend"] == "increasing":
            insights["strengths"].append("Group engagement is growing")
    
    return insights


def _parse_time_period(period: str) -> int:
    """Parse time period string to days."""
    if period.endswith("d"):
        return int(period[:-1])
    elif period.endswith("w"):
        return int(period[:-1]) * 7
    elif period.endswith("m"):
        return int(period[:-1]) * 30
    elif period.endswith("y"):
        return int(period[:-1]) * 365
    else:
        return 90  # Default to 90 days


def _redact_group_results(result: Dict, participants: Set[str]) -> Dict[str, Any]:
    """Apply privacy redaction to group results."""
    # Create participant ID mapping
    id_mapping = {}
    for i, participant in enumerate(sorted(participants)):
        if participant == "me":
            id_mapping[participant] = "me"
        else:
            id_mapping[participant] = f"member_{i:03d}"
    
    # Recursively redact participant IDs
    def redact_value(value):
        if isinstance(value, str) and value in id_mapping:
            return id_mapping[value]
        elif isinstance(value, dict):
            return {k: redact_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [redact_value(v) for v in value]
        else:
            return value
    
    redacted_result = redact_value(result)
    
    # Apply general privacy filters
    return apply_privacy_filters(redacted_result)