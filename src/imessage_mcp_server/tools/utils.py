"""
Utility functions for generating insights and recommendations.

This module contains functions that analyze tool outputs and generate
human-readable insights and actionable recommendations.
"""

from typing import Any, Dict, List


def generate_insights(tool_name: str, metrics: Dict[str, Any]) -> List[str]:
    """
    Generate human-readable insights from tool metrics.

    Args:
        tool_name: Name of the tool that generated the metrics
        metrics: The metrics data from the tool

    Returns:
        List of insight strings
    """
    insights = []

    if tool_name == "relationship_intelligence":
        # Analyze engagement patterns
        for contact in metrics.get("contacts", []):
            if contact.get("engagement_score", 0) > 0.8:
                insights.append(
                    f"Contact {contact['contact_id'][-8:]} is one of your most engaged relationships"
                )

            if "conversation-initiator" in contact.get("flags", []):
                insights.append(
                    f"You typically initiate conversations with {contact['contact_id'][-8:]}"
                )

            if "reconnect-suggested" in contact.get("flags", []):
                insights.append(
                    f"Consider reconnecting with {contact['contact_id'][-8:]} - it's been over a month"
                )

    elif tool_name == "sentiment_evolution":
        summary = metrics.get("summary", {})

        if summary.get("volatility_index", 0) < 0.3:
            insights.append("Your emotional tone is very consistent and stable")
        elif summary.get("volatility_index", 0) > 0.6:
            insights.append(
                "High emotional volatility detected - conversations show significant mood swings"
            )

        if summary.get("delta_30d", 0) > 0.1:
            insights.append("Conversations have become more positive over the last 30 days")
        elif summary.get("delta_30d", 0) < -0.1:
            insights.append(
                "Conversations have become more negative recently - consider addressing any issues"
            )

        peak_times = metrics.get("peak_sentiment_times", {})
        if peak_times.get("pattern") == "morning_person":
            insights.append("You tend to be most positive in the morning hours")
        elif peak_times.get("pattern") == "evening_person":
            insights.append("Your most positive conversations happen in the evening")

    elif tool_name == "network_intelligence":
        health = metrics.get("network_health", {})

        if health.get("risk_level") == "high":
            insights.append(
                "Your communication network could benefit from more diverse connections"
            )
        elif health.get("risk_level") == "low":
            insights.append("You have a healthy, well-connected communication network")

        if health.get("connectivity_score", 0) < 0.3:
            insights.append(
                "Consider strengthening connections between different groups in your network"
            )

        key_connectors = metrics.get("key_connectors", [])
        if len(key_connectors) > 0:
            insights.append(
                f"You have {len(key_connectors)} key people who connect different social groups"
            )

    return insights


def generate_recommendations(tool_name: str, metrics: Dict[str, Any]) -> List[str]:
    """
    Generate actionable recommendations based on metrics.

    Args:
        tool_name: Name of the tool that generated the metrics
        metrics: The metrics data from the tool

    Returns:
        List of recommendation strings
    """
    recommendations = []

    if tool_name == "relationship_intelligence":
        for contact in metrics.get("contacts", [])[:3]:  # Top 3 contacts
            if "reconnect-suggested" in contact.get("flags", []):
                recommendations.append(
                    f"Reach out to {contact['contact_id'][-8:]} - it's been over a month"
                )

            if contact.get("sent_pct", 0) > 80:
                recommendations.append(
                    f"Try asking more questions in conversations with {contact['contact_id'][-8:]}"
                )

            if contact.get("engagement_score", 0) < 0.3:
                recommendations.append(
                    f"Consider scheduling a call with {contact['contact_id'][-8:]} "
                    f"to strengthen the connection"
                )

    elif tool_name == "sentiment_evolution":
        summary = metrics.get("summary", {})

        if summary.get("volatility_index", 0) > 0.6:
            recommendations.append(
                "Consider more consistent communication patterns to stabilize emotional tone"
            )

        if summary.get("emotional_stability") == "volatile":
            recommendations.append("Address sources of stress that may be causing emotional swings")

        peak_times = metrics.get("peak_sentiment_times", {})
        if peak_times.get("most_positive_hour"):
            recommendations.append(
                f"Schedule important conversations around {peak_times['most_positive_hour']}:00"
            )

    elif tool_name == "network_intelligence":
        health = metrics.get("network_health", {})

        if health.get("diversity_score", 0) < 0.4:
            recommendations.append("Expand your social circles - join new groups or activities")

        if health.get("redundancy_score", 0) < 0.3:
            recommendations.append(
                "Introduce friends from different groups to strengthen your network"
            )

        if health.get("risk_level") == "high":
            recommendations.append(
                "Invest time in building deeper connections with existing contacts"
            )

    return recommendations
