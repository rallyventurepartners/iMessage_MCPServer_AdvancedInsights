"""
Report Generation tools for the iMessage Advanced Insights server.

This module provides tools for generating comprehensive insight reports
that synthesize data from multiple analysis tools.
"""

import logging
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from collections import defaultdict, Counter
import json

from ..database import get_database
from ..exceptions import DatabaseError, ToolExecutionError
from ..utils.responses import error_response, success_response
from .registry import register_tool, get_tool

# Import analysis functions from other tools
from .conversation_intelligence import analyze_conversation_intelligence_tool
from .predictive_analytics import predict_communication_patterns_tool, detect_anomalies_tool
from .life_events import detect_life_events_tool, analyze_emotional_wellbeing_tool
from .network_intelligence import analyze_social_network_structure_tool
# from .topic_analysis import analyze_topics_by_contact_tool  # Function doesn't exist in topic_analysis.py

logger = logging.getLogger(__name__)


@register_tool(
    name="generate_insights_report",
    description="Generate comprehensive insights report with visualizations"
)
async def generate_insights_report_tool(
    report_type: str = "monthly",
    focus_areas: Optional[List[str]] = None,
    format: str = "structured",
    include_recommendations: bool = True,
    contact_filter: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate a comprehensive insights report that synthesizes multiple analyses.
    
    This tool creates reports that include:
    - Executive summary of key findings
    - Relationship health overview
    - Communication pattern analysis
    - Emotional wellbeing assessment
    - Life event timeline
    - Network dynamics
    - Predictive insights
    - Actionable recommendations
    
    Args:
        report_type: Type of report ("daily", "weekly", "monthly", "quarterly", "annual", "custom")
        focus_areas: Specific areas to emphasize (default: all areas)
        format: Output format ("structured", "narrative", "visual")
        include_recommendations: Whether to include actionable recommendations
        contact_filter: Optional list of contacts to focus on
        
    Returns:
        Comprehensive insights report formatted for Claude to present
    """
    try:
        # Get database connection
        db = await get_database()
        
        # Determine time period based on report type
        time_period = _get_report_time_period(report_type)
        start_date = time_period["start"]
        end_date = time_period["end"]
        
        # Determine focus areas
        if focus_areas is None:
            focus_areas = [
                "relationships", "wellbeing", "predictions", 
                "network", "life_events", "communication_patterns"
            ]
        
        # Initialize report structure
        report = {
            "metadata": {
                "report_type": report_type,
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                    "days": (end_date - start_date).days
                },
                "generated_at": datetime.now().isoformat(),
                "focus_areas": focus_areas
            },
            "executive_summary": {},
            "sections": {},
            "visualizations": {},
            "recommendations": []
        }
        
        # Generate report sections based on focus areas
        if "relationships" in focus_areas:
            report["sections"]["relationships"] = await _generate_relationships_section(
                db, start_date, end_date, contact_filter
            )
        
        if "wellbeing" in focus_areas:
            report["sections"]["wellbeing"] = await _generate_wellbeing_section(
                start_date, end_date, contact_filter
            )
        
        if "predictions" in focus_areas:
            report["sections"]["predictions"] = await _generate_predictions_section(
                contact_filter
            )
        
        if "network" in focus_areas:
            report["sections"]["network"] = await _generate_network_section(
                time_period["period_str"]
            )
        
        if "life_events" in focus_areas:
            report["sections"]["life_events"] = await _generate_life_events_section(
                time_period["period_str"]
            )
        
        if "communication_patterns" in focus_areas:
            report["sections"]["communication"] = await _generate_communication_section(
                db, start_date, end_date, contact_filter
            )
        
        # Generate executive summary from all sections
        report["executive_summary"] = _generate_executive_summary(report["sections"])
        
        # Generate visualizations data
        report["visualizations"] = _generate_visualization_data(report["sections"])
        
        # Generate overall recommendations if requested
        if include_recommendations:
            report["recommendations"] = _generate_overall_recommendations(report["sections"])
        
        # Format report based on requested format
        if format == "narrative":
            report = _format_as_narrative(report)
        elif format == "visual":
            report = _enhance_with_visual_elements(report)
        
        return success_response(report)
        
    except Exception as e:
        logger.error(f"Error in generate_insights_report_tool: {e}")
        logger.error(traceback.format_exc())
        return error_response(ToolExecutionError(f"Failed to generate report: {str(e)}"))


def _get_report_time_period(report_type: str) -> Dict[str, Any]:
    """Get time period for report based on type."""
    end_date = datetime.now()
    
    if report_type == "daily":
        start_date = end_date - timedelta(days=1)
        period_str = "24 hours"
    elif report_type == "weekly":
        start_date = end_date - timedelta(days=7)
        period_str = "7 days"
    elif report_type == "monthly":
        start_date = end_date - timedelta(days=30)
        period_str = "30 days"
    elif report_type == "quarterly":
        start_date = end_date - timedelta(days=90)
        period_str = "3 months"
    elif report_type == "annual":
        start_date = end_date - timedelta(days=365)
        period_str = "1 year"
    else:  # custom
        start_date = end_date - timedelta(days=30)
        period_str = "30 days"
    
    return {
        "start": start_date,
        "end": end_date,
        "period_str": period_str
    }


async def _generate_relationships_section(
    db: Any,
    start_date: datetime,
    end_date: datetime,
    contact_filter: Optional[List[str]]
) -> Dict[str, Any]:
    """Generate relationships section of the report."""
    
    section = {
        "title": "Relationship Analysis",
        "key_metrics": {},
        "top_relationships": [],
        "relationship_health": {},
        "insights": []
    }
    
    # Get top contacts
    contacts_result = await db.get_contacts(limit=20, offset=0)
    contacts = contacts_result.get("contacts", [])
    
    if contact_filter:
        contacts = [c for c in contacts if 
                    c.get("phone_number") in contact_filter or 
                    c.get("handle_id") in contact_filter]
    
    # Analyze top relationships
    relationship_scores = []
    
    for contact in contacts[:10]:
        try:
            # Get conversation intelligence for this contact
            analysis_result = await analyze_conversation_intelligence_tool(
                contact_id=contact.get("phone_number", contact.get("handle_id")),
                analysis_depth="moderate",
                time_period=f"{(end_date - start_date).days} days",
                include_examples=False
            )
            
            if analysis_result["success"] and "analysis" in analysis_result["data"]:
                analysis = analysis_result["data"]["analysis"]
                health_score = analysis.get("health_score", {}).get("score", 0)
                
                relationship_scores.append({
                    "contact_id": contact.get("phone_number", contact.get("handle_id")),
                    "contact_name": contact.get("name", contact.get("phone_number", contact.get("handle_id"))[:10]),
                    "health_score": health_score,
                    "depth_category": analysis.get("conversation_depth", {}).get("depth_category", "Unknown"),
                    "message_count": analysis_result["data"].get("message_stats", {}).get("total_messages", 0)
                })
        except Exception as e:
            logger.error(f"Error analyzing contact {contact.get("phone_number", contact.get("handle_id"))}: {e}")
    
    # Sort by health score
    relationship_scores.sort(key=lambda x: x["health_score"], reverse=True)
    
    # Calculate key metrics
    if relationship_scores:
        avg_health = sum(r["health_score"] for r in relationship_scores) / len(relationship_scores)
        section["key_metrics"] = {
            "average_health_score": round(avg_health, 1),
            "active_relationships": len(relationship_scores),
            "thriving_relationships": sum(1 for r in relationship_scores if r["health_score"] >= 80),
            "at_risk_relationships": sum(1 for r in relationship_scores if r["health_score"] < 40)
        }
        
        section["top_relationships"] = relationship_scores[:5]
        
        # Generate insights
        if section["key_metrics"]["at_risk_relationships"] > 0:
            section["insights"].append(
                f"{section['key_metrics']['at_risk_relationships']} relationships need attention"
            )
        
        if section["key_metrics"]["thriving_relationships"] >= 3:
            section["insights"].append(
                "Multiple relationships are thriving with high engagement"
            )
    
    return section


async def _generate_wellbeing_section(
    start_date: datetime,
    end_date: datetime,
    contact_filter: Optional[List[str]]
) -> Dict[str, Any]:
    """Generate wellbeing section of the report."""
    
    section = {
        "title": "Emotional Wellbeing Analysis",
        "overall_score": 0,
        "trends": {},
        "support_network": {},
        "stress_indicators": {},
        "insights": []
    }
    
    try:
        # Get wellbeing analysis
        wellbeing_result = await analyze_emotional_wellbeing_tool(
            time_period=f"{(end_date - start_date).days} days",
            include_network_effect=True,
            analysis_depth="moderate"
        )
        
        if wellbeing_result["success"] and "wellbeing_analysis" in wellbeing_result["data"]:
            wellbeing_data = wellbeing_result["data"]["wellbeing_analysis"]
            
            # Extract overall wellbeing
            overall = wellbeing_data.get("overall_wellbeing", {})
            section["overall_score"] = overall.get("score", 0)
            section["wellbeing_category"] = overall.get("category", "unknown")
            
            # Extract network analysis
            network = wellbeing_data.get("network_analysis", {})
            if network:
                section["support_network"] = {
                    "network_health": network.get("network_health_score", 0),
                    "at_risk_contacts": network.get("at_risk_contacts", 0),
                    "thriving_contacts": network.get("thriving_contacts", 0)
                }
            
            # Extract trends
            section["trends"]["emotional"] = overall.get("trend", "stable")
            
            # Generate insights
            insights_data = wellbeing_result["data"].get("insights", {})
            section["insights"] = insights_data.get("key_findings", [])
    
    except Exception as e:
        logger.error(f"Error generating wellbeing section: {e}")
    
    return section


async def _generate_predictions_section(
    contact_filter: Optional[List[str]]
) -> Dict[str, Any]:
    """Generate predictions section of the report."""
    
    section = {
        "title": "Predictive Insights",
        "communication_forecast": {},
        "anomalies": [],
        "relationship_predictions": {},
        "insights": []
    }
    
    try:
        # Get predictive analytics
        predictions_result = await predict_communication_patterns_tool(
            prediction_window="30 days",
            include_anomaly_detection=True
        )
        
        if predictions_result["success"]:
            predictions_data = predictions_result["data"]
            
            # Extract key predictions
            section["communication_forecast"] = predictions_data.get("insights", {}).get("communication_forecast", {})
            
            # Extract anomalies
            anomalies = predictions_data.get("anomalies", [])
            section["anomalies"] = anomalies[:5]  # Top 5 anomalies
            
            # Extract insights
            section["insights"] = predictions_data.get("insights", {}).get("key_findings", [])
            
            # Count relationship trajectories
            predictions = predictions_data.get("predictions", {})
            trajectories = defaultdict(int)
            for _, pred in predictions.items():
                if "relationship_trajectory" in pred:
                    direction = pred["relationship_trajectory"].get("direction", "unknown")
                    trajectories[direction] += 1
            
            section["relationship_predictions"] = dict(trajectories)
    
    except Exception as e:
        logger.error(f"Error generating predictions section: {e}")
    
    return section


async def _generate_network_section(time_period: str) -> Dict[str, Any]:
    """Generate network analysis section of the report."""
    
    section = {
        "title": "Social Network Dynamics",
        "network_metrics": {},
        "key_connectors": [],
        "communities": [],
        "network_health": {},
        "insights": []
    }
    
    try:
        # Get network analysis
        network_result = await analyze_social_network_structure_tool(
            include_communities=True,
            calculate_influence=True,
            time_period=time_period,
            analysis_depth="moderate"
        )
        
        if network_result["success"] and "network_structure" in network_result["data"]:
            network_data = network_result["data"]
            
            # Extract network metrics
            section["network_metrics"] = network_data["network_structure"]["metrics"]
            
            # Extract key nodes
            key_nodes = network_data.get("key_nodes", {})
            section["key_connectors"] = key_nodes.get("connectors", [])[:3]
            
            # Extract communities
            communities = network_data.get("communities", {})
            section["communities"] = [
                {
                    "label": comm.get("label", "Unknown"),
                    "size": len(comm.get("members", [])),
                    "cohesion": comm.get("cohesion", 0)
                }
                for comm in communities.values()
            ][:5]
            
            # Extract network health
            section["network_health"] = network_data.get("network_health", {})
            
            # Extract insights
            insights_data = network_data.get("insights", {})
            section["insights"] = insights_data.get("key_findings", [])
    
    except Exception as e:
        logger.error(f"Error generating network section: {e}")
    
    return section


async def _generate_life_events_section(time_period: str) -> Dict[str, Any]:
    """Generate life events section of the report."""
    
    section = {
        "title": "Life Events & Milestones",
        "detected_events": [],
        "event_categories": {},
        "timeline": [],
        "insights": []
    }
    
    try:
        # Get life events
        events_result = await detect_life_events_tool(
            time_period=time_period,
            confidence_threshold=0.6,
            include_context=False
        )
        
        if events_result["success"] and "detected_events" in events_result["data"]:
            events_data = events_result["data"]
            
            # Extract top events
            section["detected_events"] = events_data["detected_events"][:10]
            
            # Extract event categories
            section["event_categories"] = events_data.get("statistics", {}).get("events_per_category", {})
            
            # Extract timeline
            section["timeline"] = events_data.get("timeline", [])[:6]  # Last 6 months
            
            # Extract insights
            insights_data = events_data.get("insights", {})
            for theme in insights_data.get("major_themes", []):
                section["insights"].append(theme.get("interpretation", ""))
    
    except Exception as e:
        logger.error(f"Error generating life events section: {e}")
    
    return section


async def _generate_communication_section(
    db: Any,
    start_date: datetime,
    end_date: datetime,
    contact_filter: Optional[List[str]]
) -> Dict[str, Any]:
    """Generate communication patterns section of the report."""
    
    section = {
        "title": "Communication Pattern Analysis",
        "overall_activity": {},
        "peak_times": [],
        "topic_trends": {},
        "communication_style": {},
        "insights": []
    }
    
    try:
        # Get overall message statistics
        total_messages = 0
        hourly_distribution = defaultdict(int)
        
        # Get contacts
        contacts_result = await db.get_contacts(limit=50, offset=0)
        contacts = contacts_result.get("contacts", [])
        
        if contact_filter:
            contacts = [c for c in contacts if c.get("phone_number", c.get("handle_id")) in contact_filter]
        
        # Analyze communication patterns
        for contact in contacts[:20]:
            messages_result = await db.get_messages_from_contact(
                phone_number=contact.get("phone_number", contact.get("handle_id")),
                start_date=start_date,
                end_date=end_date,
                page=1, page_size=500
            )
            
            messages = messages_result.get("messages", [])
            total_messages += len(messages)
            
            # Analyze hourly distribution
            for msg in messages:
                if msg.get("date"):
                    hour = datetime.fromisoformat(msg["date"]).hour
                    hourly_distribution[hour] += 1
        
        # Calculate overall activity
        days = (end_date - start_date).days
        section["overall_activity"] = {
            "total_messages": total_messages,
            "daily_average": round(total_messages / days, 1) if days > 0 else 0,
            "active_contacts": len(contacts)
        }
        
        # Identify peak times
        sorted_hours = sorted(hourly_distribution.items(), key=lambda x: x[1], reverse=True)
        peak_hours = sorted_hours[:3]
        
        section["peak_times"] = [
            {
                "hour": hour,
                "period": _hour_to_period(hour),
                "message_count": count
            }
            for hour, count in peak_hours
        ]
        
        # Get topic analysis
        try:
            from .topic_analysis import analyze_conversation_topics_tool
            topics_result = await analyze_conversation_topics_tool(
                time_period=f"{days} days",
                num_topics=10
            )
            
            if topics_result["success"]:
                topics_data = topics_result["data"]
                
                # Extract overall topic trends
                overall_topics = topics_data.get("overall_summary", {}).get("top_topics", [])
                section["topic_trends"]["top_topics"] = [t["topic"] for t in overall_topics[:5]]
                section["topic_trends"]["topic_diversity"] = topics_data.get("overall_summary", {}).get("unique_topics", 0)
        except Exception as e:
            logger.error(f"Error getting topic analysis: {e}")
        
        # Generate insights
        if section["overall_activity"]["daily_average"] > 50:
            section["insights"].append("High communication volume indicates active social engagement")
        
        if peak_hours:
            peak_period = _hour_to_period(peak_hours[0][0])
            section["insights"].append(f"Most active communication occurs during {peak_period}")
    
    except Exception as e:
        logger.error(f"Error generating communication section: {e}")
    
    return section


def _hour_to_period(hour: int) -> str:
    """Convert hour to time period."""
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 22:
        return "evening"
    else:
        return "night"


def _generate_executive_summary(sections: Dict[str, Dict]) -> Dict[str, Any]:
    """Generate executive summary from all sections."""
    
    summary = {
        "headline_metrics": {},
        "key_findings": [],
        "health_status": {},
        "action_items": []
    }
    
    # Extract headline metrics
    if "relationships" in sections:
        rel_metrics = sections["relationships"].get("key_metrics", {})
        summary["headline_metrics"]["active_relationships"] = rel_metrics.get("active_relationships", 0)
        summary["headline_metrics"]["relationship_health"] = rel_metrics.get("average_health_score", 0)
    
    if "wellbeing" in sections:
        summary["headline_metrics"]["wellbeing_score"] = sections["wellbeing"].get("overall_score", 0)
    
    if "communication" in sections:
        comm = sections["communication"].get("overall_activity", {})
        summary["headline_metrics"]["daily_messages"] = comm.get("daily_average", 0)
    
    # Compile key findings from all sections
    for section_name, section_data in sections.items():
        if "insights" in section_data:
            summary["key_findings"].extend(section_data["insights"][:2])  # Top 2 from each
    
    # Determine overall health status
    health_indicators = []
    
    if "relationships" in sections:
        rel_health = sections["relationships"].get("key_metrics", {}).get("average_health_score", 0)
        health_indicators.append(rel_health)
    
    if "wellbeing" in sections:
        wellbeing_score = sections["wellbeing"].get("overall_score", 0)
        health_indicators.append(wellbeing_score)
    
    if "network" in sections:
        network_health = sections["network"].get("network_health", {}).get("score", 0)
        health_indicators.append(network_health)
    
    if health_indicators:
        avg_health = sum(health_indicators) / len(health_indicators)
        summary["health_status"] = {
            "overall_score": round(avg_health, 1),
            "category": _categorize_health(avg_health),
            "trend": "stable"  # Would need historical data for trend
        }
    
    # Extract action items
    if "predictions" in sections:
        anomalies = sections["predictions"].get("anomalies", [])
        for anomaly in anomalies[:2]:
            if anomaly.get("type") == "unusual_silence":
                summary["action_items"].append({
                    "priority": "high",
                    "action": f"Reach out to contact - {anomaly.get('days_silent', 0)} days of silence"
                })
    
    if "relationships" in sections:
        at_risk = sections["relationships"].get("key_metrics", {}).get("at_risk_relationships", 0)
        if at_risk > 0:
            summary["action_items"].append({
                "priority": "medium",
                "action": f"Strengthen {at_risk} at-risk relationships"
            })
    
    return summary


def _categorize_health(score: float) -> str:
    """Categorize health score."""
    if score >= 80:
        return "excellent"
    elif score >= 60:
        return "good"
    elif score >= 40:
        return "fair"
    else:
        return "needs_attention"


def _generate_visualization_data(sections: Dict[str, Dict]) -> Dict[str, Any]:
    """Generate data for visualizations."""
    
    visualizations = {}
    
    # Relationship health chart
    if "relationships" in sections:
        relationships = sections["relationships"].get("top_relationships", [])
        if relationships:
            visualizations["relationship_health_chart"] = {
                "type": "bar",
                "data": [
                    {
                        "label": r["contact_name"],
                        "value": r["health_score"]
                    }
                    for r in relationships
                ],
                "title": "Top Relationship Health Scores"
            }
    
    # Communication timeline
    if "communication" in sections:
        peak_times = sections["communication"].get("peak_times", [])
        if peak_times:
            visualizations["activity_heatmap"] = {
                "type": "heatmap",
                "data": peak_times,
                "title": "Communication Activity by Time"
            }
    
    # Network visualization data
    if "network" in sections:
        communities = sections["network"].get("communities", [])
        if communities:
            visualizations["network_communities"] = {
                "type": "pie",
                "data": [
                    {
                        "label": c["label"],
                        "value": c["size"]
                    }
                    for c in communities
                ],
                "title": "Social Network Communities"
            }
    
    # Life events timeline
    if "life_events" in sections:
        timeline = sections["life_events"].get("timeline", [])
        if timeline:
            visualizations["life_events_timeline"] = {
                "type": "timeline",
                "data": timeline,
                "title": "Life Events Over Time"
            }
    
    return visualizations


def _generate_overall_recommendations(sections: Dict[str, Dict]) -> List[Dict[str, str]]:
    """Generate overall recommendations from all sections."""
    
    recommendations = []
    priority_scores = {"high": 3, "medium": 2, "low": 1}
    
    # Collect recommendations from each section
    
    # From predictions section
    if "predictions" in sections:
        anomalies = sections["predictions"].get("anomalies", [])
        for anomaly in anomalies[:2]:
            if anomaly.get("concern_level") == "high":
                recommendations.append({
                    "action": f"Immediate outreach needed",
                    "reason": anomaly.get("description", "Unusual communication pattern detected"),
                    "priority": "high",
                    "category": "relationships"
                })
    
    # From relationships section
    if "relationships" in sections:
        metrics = sections["relationships"].get("key_metrics", {})
        if metrics.get("at_risk_relationships", 0) > 2:
            recommendations.append({
                "action": "Schedule quality time with at-risk relationships",
                "reason": f"{metrics['at_risk_relationships']} relationships showing low health scores",
                "priority": "high",
                "category": "relationships"
            })
    
    # From wellbeing section
    if "wellbeing" in sections:
        wellbeing_score = sections["wellbeing"].get("overall_score", 50)
        if wellbeing_score < 40:
            recommendations.append({
                "action": "Focus on self-care and emotional wellbeing",
                "reason": "Wellbeing indicators suggest increased stress",
                "priority": "high",
                "category": "wellbeing"
            })
        
        support = sections["wellbeing"].get("support_network", {})
        if support.get("at_risk_contacts", 0) > 0:
            recommendations.append({
                "action": "Check in with contacts who may need support",
                "reason": f"{support['at_risk_contacts']} contacts showing signs of struggle",
                "priority": "medium",
                "category": "support"
            })
    
    # From network section
    if "network" in sections:
        network_health = sections["network"].get("network_health", {})
        if network_health.get("category") == "needs_attention":
            recommendations.append({
                "action": "Expand and diversify your social network",
                "reason": "Network analysis shows limited connectivity",
                "priority": "medium",
                "category": "network"
            })
    
    # From communication section
    if "communication" in sections:
        activity = sections["communication"].get("overall_activity", {})
        if activity.get("daily_average", 0) < 5:
            recommendations.append({
                "action": "Increase daily communication touchpoints",
                "reason": "Low overall communication activity detected",
                "priority": "low",
                "category": "communication"
            })
    
    # Sort by priority and return top recommendations
    recommendations.sort(key=lambda x: priority_scores.get(x["priority"], 0), reverse=True)
    
    return recommendations[:7]  # Return top 7 recommendations


def _format_as_narrative(report: Dict[str, Any]) -> Dict[str, Any]:
    """Format report as narrative text for natural presentation."""
    
    narrative_report = report.copy()
    
    # Convert executive summary to narrative
    exec_summary = report.get("executive_summary", {})
    narrative = []
    
    # Opening
    narrative.append(f"## Communication Insights Report - {report['metadata']['report_type'].title()}")
    narrative.append(f"\n*Report Period: {report['metadata']['period']['start'][:10]} to {report['metadata']['period']['end'][:10]}*\n")
    
    # Health status
    health = exec_summary.get("health_status", {})
    if health:
        narrative.append(f"### Overall Assessment: {health.get('category', 'Unknown').replace('_', ' ').title()}")
        narrative.append(f"Your overall communication health score is **{health.get('overall_score', 0)}/100**.\n")
    
    # Key metrics
    metrics = exec_summary.get("headline_metrics", {})
    if metrics:
        narrative.append("### Key Metrics")
        narrative.append(f"- **{metrics.get('active_relationships', 0)}** active relationships")
        narrative.append(f"- **{metrics.get('daily_messages', 0)}** average messages per day")
        narrative.append(f"- **{metrics.get('wellbeing_score', 0)}/100** emotional wellbeing score\n")
    
    # Key findings
    findings = exec_summary.get("key_findings", [])
    if findings:
        narrative.append("### Key Findings")
        for finding in findings[:5]:
            narrative.append(f"- {finding}")
        narrative.append("")
    
    # Section narratives
    for section_name, section_data in report.get("sections", {}).items():
        if section_data:
            narrative.append(f"### {section_data.get('title', section_name.title())}")
            
            # Add section-specific narrative
            if section_name == "relationships" and "top_relationships" in section_data:
                top_rels = section_data["top_relationships"][:3]
                if top_rels:
                    narrative.append("Your strongest relationships:")
                    for rel in top_rels:
                        narrative.append(f"- **{rel['contact_name']}**: Health score {rel['health_score']}/100")
                    narrative.append("")
            
            elif section_name == "life_events" and "detected_events" in section_data:
                events = section_data["detected_events"][:3]
                if events:
                    narrative.append("Recent significant life events detected:")
                    for event in events:
                        narrative.append(f"- {event['category'].replace('_', ' ').title()} ({event['date'][:10]})")
                    narrative.append("")
    
    # Recommendations
    recs = report.get("recommendations", [])
    if recs:
        narrative.append("### Recommended Actions")
        for i, rec in enumerate(recs[:5], 1):
            narrative.append(f"{i}. **{rec['action']}**")
            narrative.append(f"   *{rec['reason']}*")
        narrative.append("")
    
    narrative_report["narrative"] = "\n".join(narrative)
    
    return narrative_report


def _enhance_with_visual_elements(report: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance report with visual elements descriptions."""
    
    visual_report = report.copy()
    
    # Add ASCII visualizations or emoji indicators
    visual_elements = {
        "health_indicators": {
            "excellent": "🟢",
            "good": "🟡", 
            "fair": "🟠",
            "needs_attention": "🔴"
        },
        "trends": {
            "increasing": "📈",
            "stable": "➡️",
            "decreasing": "📉"
        },
        "priorities": {
            "high": "🚨",
            "medium": "⚠️",
            "low": "ℹ️"
        }
    }
    
    # Enhance executive summary with visuals
    if "executive_summary" in visual_report:
        health = visual_report["executive_summary"].get("health_status", {})
        if health:
            category = health.get("category", "")
            indicator = visual_elements["health_indicators"].get(category, "")
            health["visual_indicator"] = indicator
    
    # Enhance recommendations with priority indicators
    if "recommendations" in visual_report:
        for rec in visual_report["recommendations"]:
            priority = rec.get("priority", "low")
            rec["visual_priority"] = visual_elements["priorities"].get(priority, "")
    
    # Add simple charts using ASCII
    if "visualizations" in visual_report:
        for viz_name, viz_data in visual_report["visualizations"].items():
            if viz_data["type"] == "bar" and "relationship_health_chart" in viz_name:
                # Create simple ASCII bar chart
                chart_lines = ["\n📊 Relationship Health Scores:\n"]
                for item in viz_data["data"][:5]:
                    bar_length = int(item["value"] / 10)
                    bar = "█" * bar_length
                    chart_lines.append(f"{item['label'][:15]:15} {bar} {item['value']}")
                
                viz_data["ascii_chart"] = "\n".join(chart_lines)
    
    return visual_report


@register_tool(
    name="generate_communication_summary",
    description="Generate a quick summary of recent communication activity"
)
async def generate_communication_summary_tool(
    time_period: str = "7 days",
    include_highlights: bool = True
) -> Dict[str, Any]:
    """
    Generate a quick communication summary for a given time period.
    
    This is a lighter-weight alternative to the full insights report,
    focusing on immediate actionable information.
    
    Args:
        time_period: Period to summarize (e.g., "7 days", "24 hours")
        include_highlights: Whether to include conversation highlights
        
    Returns:
        Quick summary with key metrics and actions
    """
    try:
        # Get time bounds
        from ..utils.decorators import parse_date
        start_date = parse_date(time_period)
        end_date = datetime.now()
        
        # Get database
        db = await get_database()
        
        summary = {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "description": time_period
            },
            "activity_overview": {},
            "top_contacts": [],
            "notable_patterns": [],
            "immediate_actions": []
        }
        
        # Get recent activity
        contacts_result = await db.get_contacts(limit=10, offset=0)
        contacts = contacts_result.get("contacts", [])
        
        total_messages = 0
        contact_activity = []
        
        for contact in contacts:
            messages_result = await db.get_messages_from_contact(
                phone_number=contact.get("phone_number", contact.get("handle_id")),
                start_date=start_date,
                end_date=end_date,
                page=1, page_size=100
            )
            
            messages = messages_result.get("messages", [])
            if messages:
                total_messages += len(messages)
                
                # Calculate last message time
                last_message = messages[-1]
                last_message_date = datetime.fromisoformat(last_message["date"])
                days_since = (end_date - last_message_date).days
                
                contact_activity.append({
                    "contact_name": contact.get("name", contact.get("phone_number", contact.get("handle_id"))[:10]),
                    "message_count": len(messages),
                    "days_since_last": days_since,
                    "last_message_preview": last_message.get("text", "")[:50] + "..." if last_message.get("text", "") else ""
                })
        
        # Sort by activity
        contact_activity.sort(key=lambda x: x["message_count"], reverse=True)
        
        # Build summary
        days = (end_date - start_date).days
        summary["activity_overview"] = {
            "total_messages": total_messages,
            "daily_average": round(total_messages / days, 1) if days > 0 else 0,
            "active_contacts": len([c for c in contact_activity if c["message_count"] > 0])
        }
        
        summary["top_contacts"] = contact_activity[:5]
        
        # Identify patterns
        silent_contacts = [c for c in contact_activity if c["days_since_last"] > 7]
        if silent_contacts:
            summary["notable_patterns"].append({
                "pattern": "unusual_silence",
                "description": f"{len(silent_contacts)} contacts haven't communicated in over a week",
                "contacts": [c["contact_name"] for c in silent_contacts[:3]]
            })
        
        very_active = [c for c in contact_activity if c["message_count"] > total_messages * 0.3]
        if very_active:
            summary["notable_patterns"].append({
                "pattern": "high_activity",
                "description": f"High activity with {very_active[0]['contact_name']} ({very_active[0]['message_count']} messages)",
                "insight": "Strong engagement with this contact"
            })
        
        # Generate immediate actions
        for contact in silent_contacts[:2]:
            summary["immediate_actions"].append({
                "action": f"Reach out to {contact['contact_name']}",
                "reason": f"No communication for {contact['days_since_last']} days",
                "priority": "high" if contact["days_since_last"] > 14 else "medium"
            })
        
        # Add highlights if requested
        if include_highlights and contact_activity:
            summary["conversation_highlights"] = []
            for contact in contact_activity[:3]:
                if contact["last_message_preview"]:
                    summary["conversation_highlights"].append({
                        "contact": contact["contact_name"],
                        "preview": contact["last_message_preview"],
                        "when": f"{contact['days_since_last']} days ago" if contact["days_since_last"] > 0 else "today"
                    })
        
        return success_response(summary)
        
    except Exception as e:
        logger.error(f"Error in generate_communication_summary_tool: {e}")
        logger.error(traceback.format_exc())
        return error_response(ToolExecutionError(f"Failed to generate summary: {str(e)}"))