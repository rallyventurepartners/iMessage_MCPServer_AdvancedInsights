"""
Network Intelligence tools for the iMessage Advanced Insights server.

This module provides advanced social network analysis tools for understanding
communication patterns, influence dynamics, and community structures.
"""

import logging
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict, Counter
import statistics
import math

from ..database import get_database
from ..exceptions import DatabaseError, ToolExecutionError
from ..utils.responses import error_response, success_response
from ..utils.sanitization import sanitize_contact_info
from .registry import register_tool

logger = logging.getLogger(__name__)


@register_tool(
    name="analyze_social_network_structure",
    description="Deep analysis of social network dynamics and influence"
)
async def analyze_social_network_structure_tool(
    include_communities: bool = True,
    calculate_influence: bool = True,
    time_period: str = "1 year",
    analysis_depth: str = "comprehensive",
    min_interactions: int = 5
) -> Dict[str, Any]:
    """
    Analyze the structure and dynamics of your social network.
    
    This tool provides:
    - Network topology and graph metrics
    - Community detection and clustering
    - Key connectors and bridge nodes
    - Information flow patterns
    - Influence scores and rankings
    - Network health assessment
    - Evolution over time
    
    Args:
        include_communities: Detect and analyze community structures
        calculate_influence: Calculate influence scores for contacts
        time_period: Period to analyze
        analysis_depth: Level of analysis ("basic", "moderate", "comprehensive")
        min_interactions: Minimum messages to include contact in network
        
    Returns:
        Comprehensive network analysis with visualizations and recommendations
    """
    try:
        # Get database connection
        db = await get_database()
        
        # Parse time period
        from ..utils.decorators import parse_date
        start_date = parse_date(time_period)
        end_date = datetime.now()
        
        # Build network graph
        network_graph = await _build_network_graph(db, start_date, end_date, min_interactions)
        
        if not network_graph["nodes"]:
            return success_response({
                "status": "no_data",
                "message": "No sufficient network data found for analysis"
            })
        
        # Calculate basic network metrics
        network_metrics = _calculate_network_metrics(network_graph)
        
        # Detect communities if requested
        communities = {}
        if include_communities:
            communities = _detect_communities(network_graph)
        
        # Calculate influence scores if requested
        influence_scores = {}
        if calculate_influence:
            influence_scores = _calculate_influence_scores(network_graph)
        
        # Identify key nodes
        key_nodes = _identify_key_nodes(network_graph, influence_scores)
        
        # Analyze information flow
        information_flow = _analyze_information_flow(network_graph)
        
        # Calculate network health
        network_health = _assess_network_health(
            network_metrics, 
            communities, 
            key_nodes,
            network_graph
        )
        
        # Generate evolution analysis if comprehensive
        evolution_analysis = {}
        if analysis_depth == "comprehensive":
            evolution_analysis = await _analyze_network_evolution(
                db, start_date, end_date
            )
        
        # Generate insights and recommendations
        insights = _generate_network_insights(
            network_metrics,
            communities,
            key_nodes,
            network_health
        )
        
        recommendations = _generate_network_recommendations(
            network_metrics,
            communities,
            key_nodes,
            network_health
        )
        
        return success_response({
            "network_structure": {
                "nodes": len(network_graph["nodes"]),
                "edges": len(network_graph["edges"]),
                "metrics": network_metrics
            },
            "communities": communities,
            "influence_analysis": influence_scores,
            "key_nodes": key_nodes,
            "information_flow": information_flow,
            "network_health": network_health,
            "evolution": evolution_analysis,
            "insights": insights,
            "recommendations": recommendations,
            "visualization_data": _prepare_visualization_data(
                network_graph, communities, influence_scores
            )
        })
        
    except DatabaseError as e:
        logger.error(f"Database error in analyze_social_network_structure_tool: {e}")
        return error_response(e)
    except Exception as e:
        logger.error(f"Unexpected error in analyze_social_network_structure_tool: {e}")
        logger.error(traceback.format_exc())
        return error_response(ToolExecutionError(f"Failed to analyze network: {str(e)}"))


async def _build_network_graph(
    db: Any,
    start_date: datetime,
    end_date: datetime,
    min_interactions: int
) -> Dict[str, Any]:
    """Build a network graph from message data."""
    
    # Get all contacts with sufficient interactions
    contacts_result = await db.get_contacts(limit=200, offset=0)
    
    nodes = {}
    edges = defaultdict(lambda: {"weight": 0, "messages": []})
    
    # Add user as central node
    user_node_id = "ME"
    nodes[user_node_id] = {
        "id": user_node_id,
        "label": "Me",
        "type": "user",
        "metrics": {}
    }
    
    # Process each contact
    for contact in contacts_result.get("contacts", []):
        contact_id = contact.get("phone_number", contact.get("handle_id"))
        
        # Get messages with this contact
        messages_result = await db.get_messages_from_contact(
            phone_number=contact_id,
            start_date=start_date,
            end_date=end_date,
            page=1, page_size=1000
        )
        
        messages = messages_result.get("messages", [])
        
        if len(messages) < min_interactions:
            continue
        
        # Add contact as node
        nodes[contact_id] = {
            "id": contact_id,
            "label": contact.get("name", contact_id[:10]),
            "type": "contact",
            "message_count": len(messages),
            "metrics": {}
        }
        
        # Add edge between user and contact
        edge_key = tuple(sorted([user_node_id, contact_id]))
        edges[edge_key]["weight"] = len(messages)
        edges[edge_key]["messages"] = messages
        
        # Look for group connections (simplified - would need group chat data)
        # This is a placeholder for group-based connections
        
    # Convert defaultdict to regular dict
    edges = dict(edges)
    
    return {
        "nodes": nodes,
        "edges": edges
    }


def _calculate_network_metrics(graph: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate various network metrics."""
    
    nodes = graph["nodes"]
    edges = graph["edges"]
    
    # Calculate degree for each node
    node_degrees = defaultdict(int)
    for edge in edges:
        for node in edge:
            node_degrees[node] += 1
    
    # Update node metrics
    for node_id, degree in node_degrees.items():
        if node_id in nodes:
            nodes[node_id]["metrics"]["degree"] = degree
    
    # Calculate network-wide metrics
    degrees = list(node_degrees.values())
    
    metrics = {
        "density": len(edges) / (len(nodes) * (len(nodes) - 1) / 2) if len(nodes) > 1 else 0,
        "average_degree": statistics.mean(degrees) if degrees else 0,
        "max_degree": max(degrees) if degrees else 0,
        "clustering_coefficient": _calculate_clustering_coefficient(graph),
        "centralization": _calculate_centralization(degrees),
        "diameter": _estimate_network_diameter(graph),
        "components": _count_components(graph)
    }
    
    return metrics


def _calculate_clustering_coefficient(graph: Dict[str, Any]) -> float:
    """Calculate the clustering coefficient of the network."""
    # Simplified calculation - measures how connected neighbors are
    
    nodes = graph["nodes"]
    edges = graph["edges"]
    
    # Build adjacency list
    adjacency = defaultdict(set)
    for edge in edges:
        adjacency[edge[0]].add(edge[1])
        adjacency[edge[1]].add(edge[0])
    
    clustering_coeffs = []
    
    for node in nodes:
        neighbors = adjacency[node]
        if len(neighbors) < 2:
            continue
        
        # Count edges between neighbors
        neighbor_edges = 0
        neighbor_list = list(neighbors)
        
        for i in range(len(neighbor_list)):
            for j in range(i + 1, len(neighbor_list)):
                edge_key = tuple(sorted([neighbor_list[i], neighbor_list[j]]))
                if edge_key in edges:
                    neighbor_edges += 1
        
        # Calculate clustering coefficient for this node
        possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
        if possible_edges > 0:
            clustering_coeffs.append(neighbor_edges / possible_edges)
    
    return statistics.mean(clustering_coeffs) if clustering_coeffs else 0


def _calculate_centralization(degrees: List[int]) -> float:
    """Calculate network centralization."""
    if not degrees or len(degrees) < 2:
        return 0
    
    max_degree = max(degrees)
    n = len(degrees)
    
    # Calculate centralization
    sum_diff = sum(max_degree - d for d in degrees)
    max_sum_diff = (n - 1) * (n - 2)
    
    return sum_diff / max_sum_diff if max_sum_diff > 0 else 0


def _estimate_network_diameter(graph: Dict[str, Any]) -> int:
    """Estimate the network diameter (longest shortest path)."""
    # Simplified estimation based on network size
    n = len(graph["nodes"])
    if n <= 1:
        return 0
    elif n <= 10:
        return 2
    elif n <= 50:
        return 3
    else:
        return int(math.log(n, 2))


def _count_components(graph: Dict[str, Any]) -> int:
    """Count the number of connected components."""
    # For our use case, we assume the network is connected through the user
    # In reality, we'd use a graph traversal algorithm
    return 1


def _detect_communities(graph: Dict[str, Any]) -> Dict[str, Any]:
    """Detect communities within the network."""
    # Simplified community detection based on interaction patterns
    
    nodes = graph["nodes"]
    edges = graph["edges"]
    
    # Build interaction matrix
    interaction_strength = defaultdict(lambda: defaultdict(float))
    
    for edge, data in edges.items():
        weight = data["weight"]
        interaction_strength[edge[0]][edge[1]] = weight
        interaction_strength[edge[1]][edge[0]] = weight
    
    # Simple clustering based on interaction strength
    communities = {}
    community_id = 0
    assigned = set()
    
    # Start with nodes that have highest degree
    sorted_nodes = sorted(
        nodes.keys(), 
        key=lambda n: nodes[n]["metrics"].get("degree", 0),
        reverse=True
    )
    
    for node in sorted_nodes:
        if node in assigned:
            continue
        
        # Create new community
        community = {
            "id": f"community_{community_id}",
            "members": [node],
            "core_member": node,
            "cohesion": 0
        }
        
        # Add strongly connected neighbors
        if node in interaction_strength:
            neighbors = sorted(
                interaction_strength[node].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for neighbor, strength in neighbors[:5]:  # Top 5 connections
                if neighbor not in assigned and strength > 10:  # Threshold
                    community["members"].append(neighbor)
                    assigned.add(neighbor)
        
        assigned.add(node)
        
        # Calculate community cohesion
        if len(community["members"]) > 1:
            total_internal_weight = 0
            for i, member1 in enumerate(community["members"]):
                for member2 in community["members"][i+1:]:
                    edge_key = tuple(sorted([member1, member2]))
                    if edge_key in edges:
                        total_internal_weight += edges[edge_key]["weight"]
            
            possible_edges = len(community["members"]) * (len(community["members"]) - 1) / 2
            community["cohesion"] = total_internal_weight / possible_edges if possible_edges > 0 else 0
            
            communities[community["id"]] = community
            community_id += 1
    
    # Add community labels
    _label_communities(communities, nodes)
    
    return communities


def _label_communities(communities: Dict[str, Dict], nodes: Dict[str, Any]) -> None:
    """Add descriptive labels to communities."""
    for comm_id, community in communities.items():
        size = len(community["members"])
        cohesion = community["cohesion"]
        
        if size >= 10:
            community["label"] = "Large Social Circle"
        elif size >= 5:
            community["label"] = "Close Friend Group"
        elif size >= 3:
            community["label"] = "Small Circle"
        else:
            community["label"] = "Pair Connection"
        
        if cohesion > 50:
            community["label"] += " (Very Active)"
        elif cohesion > 20:
            community["label"] += " (Active)"
        else:
            community["label"] += " (Occasional)"


def _calculate_influence_scores(graph: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate influence scores for network members."""
    nodes = graph["nodes"]
    edges = graph["edges"]
    
    influence_scores = {}
    
    for node_id, node_data in nodes.items():
        if node_id == "ME":
            continue
        
        # Calculate influence based on multiple factors
        degree = node_data["metrics"].get("degree", 0)
        message_count = node_data.get("message_count", 0)
        
        # Find edges involving this node
        node_edges = [e for e in edges if node_id in e]
        
        # Calculate message velocity (messages per day)
        total_days = 365  # Assuming 1 year period
        message_velocity = message_count / total_days
        
        # Calculate influence score
        influence = (
            degree * 0.3 +  # Network connectivity
            math.log(message_count + 1) * 0.4 +  # Communication volume
            message_velocity * 0.3  # Communication frequency
        )
        
        influence_scores[node_id] = {
            "score": round(influence, 2),
            "rank": 0,  # Will be updated
            "factors": {
                "connectivity": degree,
                "volume": message_count,
                "frequency": round(message_velocity, 2)
            },
            "category": _categorize_influence(influence)
        }
    
    # Rank influences
    sorted_influences = sorted(
        influence_scores.items(),
        key=lambda x: x[1]["score"],
        reverse=True
    )
    
    for rank, (node_id, data) in enumerate(sorted_influences, 1):
        data["rank"] = rank
    
    return influence_scores


def _categorize_influence(score: float) -> str:
    """Categorize influence level."""
    if score > 20:
        return "key_influencer"
    elif score > 10:
        return "strong_influence"
    elif score > 5:
        return "moderate_influence"
    else:
        return "peripheral"


def _identify_key_nodes(
    graph: Dict[str, Any],
    influence_scores: Dict[str, Any]
) -> Dict[str, List[Dict]]:
    """Identify key nodes in the network."""
    nodes = graph["nodes"]
    edges = graph["edges"]
    
    key_nodes = {
        "connectors": [],  # High degree nodes
        "bridges": [],     # Connect different communities
        "influencers": [], # High influence scores
        "gatekeepers": []  # Control information flow
    }
    
    # Find connectors (top degree nodes)
    degree_sorted = sorted(
        [(nid, n["metrics"].get("degree", 0)) for nid, n in nodes.items() if nid != "ME"],
        key=lambda x: x[1],
        reverse=True
    )
    
    for node_id, degree in degree_sorted[:5]:
        if degree > 3:  # Threshold
            key_nodes["connectors"].append({
                "id": node_id,
                "label": nodes[node_id]["label"],
                "degree": degree,
                "role": "Highly connected individual"
            })
    
    # Find influencers
    for node_id, influence_data in influence_scores.items():
        if influence_data["category"] in ["key_influencer", "strong_influence"]:
            key_nodes["influencers"].append({
                "id": node_id,
                "label": nodes[node_id]["label"],
                "influence_score": influence_data["score"],
                "rank": influence_data["rank"],
                "role": "Strong influence in network"
            })
    
    # Simplified bridge detection (would need community data)
    # For now, identify nodes with diverse connections
    
    return key_nodes


def _analyze_information_flow(graph: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze how information flows through the network."""
    nodes = graph["nodes"]
    edges = graph["edges"]
    
    # Calculate flow metrics
    total_messages = sum(e["weight"] for e in edges.values())
    
    # Identify flow patterns
    flow_patterns = {
        "centralized": False,
        "distributed": False,
        "clustered": False
    }
    
    # Check if flow is centralized (through user)
    user_edges = [e for e in edges if "ME" in e]
    user_message_ratio = sum(e["weight"] for e in edges.values() if e in user_edges) / total_messages
    
    if user_message_ratio > 0.8:
        flow_patterns["centralized"] = True
    elif user_message_ratio < 0.5:
        flow_patterns["distributed"] = True
    else:
        flow_patterns["clustered"] = True
    
    # Calculate flow efficiency
    avg_path_length = _estimate_average_path_length(graph)
    
    return {
        "patterns": flow_patterns,
        "metrics": {
            "total_information_flow": total_messages,
            "centralization_ratio": round(user_message_ratio, 2),
            "average_path_length": avg_path_length,
            "efficiency": round(1 / avg_path_length if avg_path_length > 0 else 0, 2)
        },
        "interpretation": _interpret_information_flow(flow_patterns, user_message_ratio)
    }


def _estimate_average_path_length(graph: Dict[str, Any]) -> float:
    """Estimate average path length in the network."""
    # Simplified estimation
    n = len(graph["nodes"])
    if n <= 2:
        return 1
    
    # For star topology (common in personal networks)
    return 2.0


def _interpret_information_flow(patterns: Dict[str, bool], centralization: float) -> str:
    """Interpret information flow patterns."""
    if patterns["centralized"]:
        return "Information flows primarily through you - you're the central hub"
    elif patterns["distributed"]:
        return "Information flows freely between contacts - well-connected network"
    else:
        return "Information flows within clusters - distinct social groups"


def _assess_network_health(
    metrics: Dict[str, Any],
    communities: Dict[str, Any],
    key_nodes: Dict[str, List],
    graph: Dict[str, Any]
) -> Dict[str, Any]:
    """Assess overall network health."""
    
    health_score = 50  # Start neutral
    health_factors = []
    
    # Factor 1: Network density (not too sparse, not too dense)
    density = metrics["density"]
    if 0.1 <= density <= 0.3:
        health_score += 10
        health_factors.append("Optimal network density")
    elif density < 0.05:
        health_score -= 10
        health_factors.append("Network too sparse")
    elif density > 0.5:
        health_score -= 5
        health_factors.append("Network possibly too dense")
    
    # Factor 2: Community structure
    if len(communities) >= 2:
        health_score += 15
        health_factors.append("Healthy community structure")
    else:
        health_score -= 10
        health_factors.append("Limited community diversity")
    
    # Factor 3: Key node distribution
    if len(key_nodes["connectors"]) >= 3:
        health_score += 10
        health_factors.append("Good distribution of connectors")
    
    if len(key_nodes["influencers"]) >= 2:
        health_score += 10
        health_factors.append("Multiple influence centers")
    
    # Factor 4: Network size
    network_size = len(graph["nodes"])
    if 10 <= network_size <= 150:  # Dunbar's number range
        health_score += 15
        health_factors.append("Optimal network size")
    elif network_size < 5:
        health_score -= 15
        health_factors.append("Very small network")
    
    # Determine health category
    if health_score >= 80:
        category = "excellent"
    elif health_score >= 60:
        category = "good"
    elif health_score >= 40:
        category = "fair"
    else:
        category = "needs_attention"
    
    return {
        "score": health_score,
        "category": category,
        "factors": health_factors,
        "strengths": [f for f in health_factors if "Good" in f or "Optimal" in f or "Healthy" in f],
        "weaknesses": [f for f in health_factors if "Limited" in f or "sparse" in f or "small" in f]
    }


async def _analyze_network_evolution(
    db: Any,
    start_date: datetime,
    end_date: datetime
) -> Dict[str, Any]:
    """Analyze how the network has evolved over time."""
    
    # Divide time period into quarters
    total_days = (end_date - start_date).days
    quarter_days = total_days // 4
    
    evolution_data = []
    
    for quarter in range(4):
        quarter_start = start_date + timedelta(days=quarter * quarter_days)
        quarter_end = quarter_start + timedelta(days=quarter_days)
        
        # Build network for this quarter
        quarter_graph = await _build_network_graph(db, quarter_start, quarter_end, 1)
        
        evolution_data.append({
            "period": f"Q{quarter + 1}",
            "start_date": quarter_start.isoformat(),
            "end_date": quarter_end.isoformat(),
            "nodes": len(quarter_graph["nodes"]),
            "edges": len(quarter_graph["edges"]),
            "total_messages": sum(e["weight"] for e in quarter_graph["edges"].values())
        })
    
    # Analyze trends
    node_trend = "growing" if evolution_data[-1]["nodes"] > evolution_data[0]["nodes"] else "stable"
    activity_trend = "increasing" if evolution_data[-1]["total_messages"] > evolution_data[0]["total_messages"] else "stable"
    
    return {
        "quarterly_data": evolution_data,
        "trends": {
            "network_size": node_trend,
            "activity_level": activity_trend
        },
        "growth_rate": {
            "nodes": f"{((evolution_data[-1]['nodes'] / evolution_data[0]['nodes']) - 1) * 100:.0f}%" if evolution_data[0]['nodes'] > 0 else "N/A",
            "activity": f"{((evolution_data[-1]['total_messages'] / evolution_data[0]['total_messages']) - 1) * 100:.0f}%" if evolution_data[0]['total_messages'] > 0 else "N/A"
        }
    }


def _generate_network_insights(
    metrics: Dict[str, Any],
    communities: Dict[str, Any],
    key_nodes: Dict[str, List],
    health: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate insights from network analysis."""
    
    insights = {
        "key_findings": [],
        "network_type": _determine_network_type(metrics, communities),
        "social_dynamics": []
    }
    
    # Network structure insights
    if metrics["density"] < 0.1:
        insights["key_findings"].append("Your network is loosely connected - consider introducing friends to each other")
    elif metrics["density"] > 0.3:
        insights["key_findings"].append("Your network is highly interconnected - strong community bonds")
    
    # Community insights
    if len(communities) > 3:
        insights["key_findings"].append(f"You maintain {len(communities)} distinct social circles")
    
    # Influence insights
    top_influencers = key_nodes.get("influencers", [])[:3]
    if top_influencers:
        insights["social_dynamics"].append(
            f"Key influencers in your network: {', '.join(i['label'] for i in top_influencers)}"
        )
    
    # Health insights
    if health["category"] == "excellent":
        insights["key_findings"].append("Your social network is exceptionally healthy and well-balanced")
    elif health["category"] == "needs_attention":
        insights["key_findings"].append("Your social network could benefit from diversification")
    
    return insights


def _determine_network_type(metrics: Dict[str, Any], communities: Dict[str, Any]) -> str:
    """Determine the type of social network."""
    
    if metrics["density"] > 0.3 and len(communities) <= 2:
        return "tight_knit"
    elif metrics["density"] < 0.1:
        return "sparse"
    elif len(communities) >= 4:
        return "multi_community"
    elif metrics["centralization"] > 0.7:
        return "hub_and_spoke"
    else:
        return "balanced"


def _generate_network_recommendations(
    metrics: Dict[str, Any],
    communities: Dict[str, Any],
    key_nodes: Dict[str, List],
    health: Dict[str, Any]
) -> List[Dict[str, str]]:
    """Generate recommendations for network optimization."""
    
    recommendations = []
    
    # Based on network health
    if health["score"] < 40:
        recommendations.append({
            "action": "Expand your social network",
            "reason": "Your network is relatively small or disconnected",
            "priority": "high",
            "suggestion": "Reconnect with old friends or join new social activities"
        })
    
    # Based on density
    if metrics["density"] < 0.05:
        recommendations.append({
            "action": "Foster connections between friends",
            "reason": "Your network lacks interconnections",
            "priority": "medium",
            "suggestion": "Organize group activities or introduce compatible friends"
        })
    
    # Based on communities
    if len(communities) < 2:
        recommendations.append({
            "action": "Diversify your social circles",
            "reason": "Limited community diversity detected",
            "priority": "medium",
            "suggestion": "Engage with different groups (work, hobbies, family)"
        })
    
    # Based on key nodes
    if not key_nodes.get("connectors"):
        recommendations.append({
            "action": "Identify and nurture connector relationships",
            "reason": "Lack of bridge connections in network",
            "priority": "low",
            "suggestion": "Strengthen relationships with socially active individuals"
        })
    
    # Maintenance recommendations
    if health["category"] in ["excellent", "good"]:
        recommendations.append({
            "action": "Maintain current network patterns",
            "reason": "Your network structure is healthy",
            "priority": "low",
            "suggestion": "Continue regular engagement with key contacts"
        })
    
    return recommendations[:4]  # Limit to top 4


def _prepare_visualization_data(
    graph: Dict[str, Any],
    communities: Dict[str, Any],
    influence_scores: Dict[str, Any]
) -> Dict[str, Any]:
    """Prepare data for network visualization."""
    
    # Convert graph data to visualization format
    viz_nodes = []
    viz_edges = []
    
    # Prepare nodes
    for node_id, node_data in graph["nodes"].items():
        viz_node = {
            "id": node_id,
            "label": node_data["label"],
            "size": math.log(node_data.get("message_count", 1) + 1) * 5 if node_id != "ME" else 20,
            "color": _get_node_color(node_id, communities, influence_scores),
            "x": 0,  # Would be calculated by layout algorithm
            "y": 0   # Would be calculated by layout algorithm
        }
        viz_nodes.append(viz_node)
    
    # Prepare edges
    for edge, edge_data in graph["edges"].items():
        viz_edges.append({
            "source": edge[0],
            "target": edge[1],
            "weight": math.log(edge_data["weight"] + 1),
            "value": edge_data["weight"]
        })
    
    return {
        "nodes": viz_nodes,
        "edges": viz_edges,
        "layout": "force-directed",
        "communities": [
            {
                "id": comm_id,
                "label": comm["label"],
                "members": comm["members"],
                "color": _get_community_color(idx)
            }
            for idx, (comm_id, comm) in enumerate(communities.items())
        ]
    }


def _get_node_color(
    node_id: str,
    communities: Dict[str, Any],
    influence_scores: Dict[str, Any]
) -> str:
    """Get color for node based on properties."""
    
    if node_id == "ME":
        return "#FF6B6B"  # Red for user
    
    # Color by influence level
    if node_id in influence_scores:
        category = influence_scores[node_id]["category"]
        if category == "key_influencer":
            return "#4ECDC4"  # Teal
        elif category == "strong_influence":
            return "#45B7D1"  # Light blue
        elif category == "moderate_influence":
            return "#96CEB4"  # Green
    
    return "#DDA0DD"  # Default purple


def _get_community_color(index: int) -> str:
    """Get color for community visualization."""
    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
        "#FECA57", "#48C9B0", "#9B59B6", "#E74C3C"
    ]
    return colors[index % len(colors)]


@register_tool(
    name="compare_relationships",
    description="Compare communication patterns across relationships"
)
async def compare_relationships_tool(
    contact_ids: List[str],
    comparison_metrics: Optional[List[str]] = None,
    time_period: str = "6 months",
    include_recommendations: bool = True
) -> Dict[str, Any]:
    """
    Compare communication patterns across multiple relationships.
    
    This tool enables:
    - Side-by-side relationship comparison
    - Pattern identification across relationships
    - Balance assessment
    - Improvement suggestions
    
    Args:
        contact_ids: List of contact IDs to compare
        comparison_metrics: Metrics to compare (default: all)
        time_period: Period to analyze
        include_recommendations: Whether to include improvement suggestions
        
    Returns:
        Comprehensive comparison analysis
    """
    try:
        if not contact_ids or len(contact_ids) < 2:
            return error_response("Please provide at least 2 contact IDs to compare")
        
        if comparison_metrics is None:
            comparison_metrics = ["frequency", "depth", "sentiment", "reciprocity", "topics"]
        
        # Get database connection
        db = await get_database()
        
        # Parse time period
        from ..utils.decorators import parse_date
        start_date = parse_date(time_period)
        end_date = datetime.now()
        
        # Analyze each relationship
        relationship_analyses = {}
        
        for contact_id in contact_ids[:5]:  # Limit to 5 comparisons
            analysis = await _analyze_relationship_for_comparison(
                db, contact_id, start_date, end_date, comparison_metrics
            )
            if analysis:
                relationship_analyses[contact_id] = analysis
        
        if not relationship_analyses:
            return success_response({
                "status": "no_data",
                "message": "No sufficient data found for comparison"
            })
        
        # Perform comparative analysis
        comparison_results = _perform_relationship_comparison(
            relationship_analyses,
            comparison_metrics
        )
        
        # Generate insights
        insights = _generate_comparison_insights(
            relationship_analyses,
            comparison_results
        )
        
        # Generate recommendations if requested
        recommendations = []
        if include_recommendations:
            recommendations = _generate_comparison_recommendations(
                relationship_analyses,
                comparison_results
            )
        
        return success_response({
            "relationships": relationship_analyses,
            "comparison": comparison_results,
            "insights": insights,
            "recommendations": recommendations,
            "metadata": {
                "time_period": time_period,
                "metrics_compared": comparison_metrics,
                "relationships_analyzed": len(relationship_analyses)
            }
        })
        
    except Exception as e:
        logger.error(f"Error in compare_relationships_tool: {e}")
        logger.error(traceback.format_exc())
        return error_response(ToolExecutionError(f"Failed to compare relationships: {str(e)}"))


async def _analyze_relationship_for_comparison(
    db: Any,
    contact_id: str,
    start_date: datetime,
    end_date: datetime,
    metrics: List[str]
) -> Optional[Dict[str, Any]]:
    """Analyze a single relationship for comparison."""
    
    # Get contact info
    contact_info = await db.get_contact_info(contact_id=contact_id)
    
    # Get messages
    messages_result = await db.get_messages_from_contact(
        phone_number=contact_id,
        start_date=start_date,
        end_date=end_date,
        page=1, page_size=1000
    )
    
    messages = messages_result.get("messages", [])
    
    if len(messages) < 10:
        return None
    
    analysis = {
        "contact_info": sanitize_contact_info(contact_info),
        "message_count": len(messages),
        "metrics": {}
    }
    
    # Calculate requested metrics
    if "frequency" in metrics:
        analysis["metrics"]["frequency"] = _calculate_frequency_metrics(messages, start_date, end_date)
    
    if "depth" in metrics:
        analysis["metrics"]["depth"] = _calculate_depth_metrics(messages)
    
    if "sentiment" in metrics:
        analysis["metrics"]["sentiment"] = _calculate_sentiment_metrics(messages)
    
    if "reciprocity" in metrics:
        analysis["metrics"]["reciprocity"] = _calculate_reciprocity_metrics(messages)
    
    if "topics" in metrics:
        analysis["metrics"]["topics"] = _calculate_topic_metrics(messages)
    
    return analysis


def _calculate_frequency_metrics(messages: List[Dict], start_date: datetime, end_date: datetime) -> Dict[str, Any]:
    """Calculate frequency metrics for comparison."""
    total_days = (end_date - start_date).days
    
    # Daily message counts
    daily_counts = defaultdict(int)
    for msg in messages:
        if msg.get("date"):
            date = datetime.fromisoformat(msg["date"]).date()
            daily_counts[date] += 1
    
    active_days = len(daily_counts)
    
    return {
        "messages_per_day": round(len(messages) / total_days, 2),
        "active_days": active_days,
        "activity_rate": round(active_days / total_days, 2),
        "max_daily_messages": max(daily_counts.values()) if daily_counts else 0,
        "consistency_score": round(statistics.stdev(list(daily_counts.values())) if len(daily_counts) > 1 else 0, 2)
    }


def _calculate_depth_metrics(messages: List[Dict]) -> Dict[str, Any]:
    """Calculate conversation depth metrics."""
    message_lengths = [len(msg.get("text", "")) for msg in messages if msg.get("text")]
    
    if not message_lengths:
        return {}
    
    return {
        "average_length": round(statistics.mean(message_lengths), 1),
        "median_length": round(statistics.median(message_lengths), 1),
        "long_messages": sum(1 for l in message_lengths if l > 100),
        "depth_score": min(100, round((statistics.mean(message_lengths) / 50) * 100))
    }


def _calculate_sentiment_metrics(messages: List[Dict]) -> Dict[str, Any]:
    """Calculate sentiment metrics for comparison."""
    positive_count = 0
    negative_count = 0
    
    positive_words = {"love", "great", "awesome", "happy", "excited", "wonderful", "lol", "haha"}
    negative_words = {"sad", "angry", "frustrated", "upset", "worried", "stressed", "sorry"}
    
    for msg in messages:
        if msg.get("text"):
            text_lower = msg["text"].lower()
            words = set(text_lower.split())
            
            if words & positive_words:
                positive_count += 1
            if words & negative_words:
                negative_count += 1
    
    total = len(messages)
    
    return {
        "positivity_rate": round(positive_count / total, 3) if total > 0 else 0,
        "negativity_rate": round(negative_count / total, 3) if total > 0 else 0,
        "sentiment_balance": round((positive_count - negative_count) / total, 3) if total > 0 else 0,
        "emotional_messages": positive_count + negative_count
    }


def _calculate_reciprocity_metrics(messages: List[Dict]) -> Dict[str, Any]:
    """Calculate reciprocity metrics."""
    sent = sum(1 for msg in messages if msg.get("is_from_me"))
    received = len(messages) - sent
    
    # Calculate response patterns
    responses = 0
    for i in range(1, len(messages)):
        if messages[i].get("is_from_me") != messages[i-1].get("is_from_me"):
            responses += 1
    
    return {
        "sent_messages": sent,
        "received_messages": received,
        "balance_ratio": round(sent / max(received, 1), 2),
        "response_rate": round(responses / len(messages), 2) if messages else 0,
        "reciprocity_score": round(100 * (1 - abs(sent - received) / len(messages)), 1) if messages else 50
    }


def _calculate_topic_metrics(messages: List[Dict]) -> Dict[str, Any]:
    """Calculate topic diversity metrics."""
    # Extract words as topics (simplified)
    all_words = []
    
    for msg in messages:
        if msg.get("text"):
            words = [w.lower() for w in msg["text"].split() if len(w) > 4 and w.isalpha()]
            all_words.extend(words)
    
    unique_topics = len(set(all_words))
    total_words = len(all_words)
    
    # Get top topics
    topic_counts = Counter(all_words)
    top_topics = [topic for topic, _ in topic_counts.most_common(5)]
    
    return {
        "unique_topics": unique_topics,
        "topic_diversity": round(unique_topics / total_words, 3) if total_words > 0 else 0,
        "top_topics": top_topics,
        "topic_richness": min(100, round((unique_topics / 100) * 100))
    }


def _perform_relationship_comparison(
    analyses: Dict[str, Dict],
    metrics: List[str]
) -> Dict[str, Any]:
    """Perform comparative analysis across relationships."""
    
    comparison = {
        "rankings": {},
        "averages": {},
        "distributions": {}
    }
    
    # Rank relationships by each metric
    for metric in metrics:
        metric_values = []
        
        for contact_id, analysis in analyses.items():
            if metric in analysis["metrics"]:
                metric_data = analysis["metrics"][metric]
                
                # Get primary value for ranking
                if metric == "frequency":
                    value = metric_data.get("messages_per_day", 0)
                elif metric == "depth":
                    value = metric_data.get("depth_score", 0)
                elif metric == "sentiment":
                    value = metric_data.get("sentiment_balance", 0)
                elif metric == "reciprocity":
                    value = metric_data.get("reciprocity_score", 0)
                elif metric == "topics":
                    value = metric_data.get("topic_richness", 0)
                else:
                    value = 0
                
                metric_values.append((contact_id, value))
        
        # Sort and rank
        metric_values.sort(key=lambda x: x[1], reverse=True)
        
        comparison["rankings"][metric] = [
            {"contact_id": cid, "value": val, "rank": idx + 1}
            for idx, (cid, val) in enumerate(metric_values)
        ]
        
        # Calculate averages
        if metric_values:
            comparison["averages"][metric] = round(
                statistics.mean([val for _, val in metric_values]), 2
            )
    
    return comparison


def _generate_comparison_insights(
    analyses: Dict[str, Dict],
    comparison: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate insights from relationship comparison."""
    
    insights = {
        "patterns": [],
        "imbalances": [],
        "standouts": []
    }
    
    # Identify patterns
    if "frequency" in comparison["rankings"]:
        freq_values = [item["value"] for item in comparison["rankings"]["frequency"]]
        if freq_values and max(freq_values) > min(freq_values) * 3:
            insights["patterns"].append("Large variation in communication frequency across relationships")
    
    # Identify imbalances
    for contact_id, analysis in analyses.items():
        if "reciprocity" in analysis["metrics"]:
            balance_ratio = analysis["metrics"]["reciprocity"]["balance_ratio"]
            if balance_ratio > 2 or balance_ratio < 0.5:
                insights["imbalances"].append({
                    "contact_id": contact_id,
                    "type": "communication_balance",
                    "description": "Significant imbalance in message exchange"
                })
    
    # Identify standouts
    for metric, ranking in comparison["rankings"].items():
        if ranking:
            top = ranking[0]
            if len(ranking) > 1 and top["value"] > ranking[1]["value"] * 1.5:
                insights["standouts"].append({
                    "contact_id": top["contact_id"],
                    "metric": metric,
                    "description": f"Exceptionally high {metric}"
                })
    
    return insights


def _generate_comparison_recommendations(
    analyses: Dict[str, Dict],
    comparison: Dict[str, Any]
) -> List[Dict[str, str]]:
    """Generate recommendations from relationship comparison."""
    
    recommendations = []
    
    # Check for imbalanced relationships
    for contact_id, analysis in analyses.items():
        if "reciprocity" in analysis["metrics"]:
            reciprocity = analysis["metrics"]["reciprocity"]
            if reciprocity["balance_ratio"] > 2:
                recommendations.append({
                    "action": f"Encourage more sharing from {analysis['contact_info']['name']}",
                    "reason": "You're sending many more messages than receiving",
                    "priority": "medium",
                    "contact_id": contact_id
                })
            elif reciprocity["balance_ratio"] < 0.5:
                recommendations.append({
                    "action": f"Engage more actively with {analysis['contact_info']['name']}",
                    "reason": "You're receiving many more messages than sending",
                    "priority": "medium",
                    "contact_id": contact_id
                })
    
    # Check for low-depth relationships
    depth_scores = [
        (cid, a["metrics"]["depth"]["depth_score"])
        for cid, a in analyses.items()
        if "depth" in a["metrics"]
    ]
    
    if depth_scores:
        avg_depth = statistics.mean([score for _, score in depth_scores])
        for contact_id, score in depth_scores:
            if score < avg_depth * 0.5:
                recommendations.append({
                    "action": f"Deepen conversations with {analyses[contact_id]['contact_info']['name']}",
                    "reason": "Conversations tend to be brief",
                    "priority": "low",
                    "suggestion": "Ask open-ended questions or share more detailed updates"
                })
    
    return recommendations[:5]