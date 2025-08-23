"""
Relationship Comparison tool for multi-relationship analysis.

Compares multiple relationships across various dimensions to identify patterns,
clusters, and actionable insights.
"""

import asyncio
import logging
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from imessage_mcp_server.privacy import apply_privacy_filters, hash_contact_id
from imessage_mcp_server.tools.quality import conversation_quality_tool

logger = logging.getLogger(__name__)


async def relationship_comparison_tool(
    contact_ids: List[str],
    comparison_type: str = "comprehensive",
    include_clusters: bool = True,
    db_path: str = "~/Library/Messages/chat.db",
    redact: bool = True,
) -> Dict[str, Any]:
    """
    Compare multiple relationships across quality dimensions.
    
    Analyzes up to 10 relationships to identify:
    - Comparative strengths and weaknesses
    - Relationship clusters (inner circle, professional, casual)
    - Network patterns and dynamics
    - Actionable insights for relationship management
    
    Args:
        contact_ids: List of contact identifiers to compare (max 10)
        comparison_type: Analysis type ("comprehensive", "quick", "focused")
        include_clusters: Whether to perform clustering analysis
        db_path: Path to iMessage database
        redact: Whether to apply privacy filters
        
    Returns:
        Dict containing comparative analysis, clusters, and insights
    """
    try:
        # Validate inputs
        if not contact_ids:
            return {"error": "No contacts provided for comparison"}
        
        if len(contact_ids) > 10:
            return {"error": "Maximum 10 contacts allowed for comparison"}
        
        # Gather quality scores for all contacts in parallel
        quality_tasks = [
            conversation_quality_tool(
                contact_id=cid,
                time_period="30d",
                include_recommendations=False,
                db_path=db_path,
                redact=False  # We'll handle redaction at the end
            )
            for cid in contact_ids
        ]
        
        quality_results = await asyncio.gather(*quality_tasks, return_exceptions=True)
        
        # Filter out errors and prepare data
        valid_results = []
        error_contacts = []
        
        for i, result in enumerate(quality_results):
            if isinstance(result, Exception) or (isinstance(result, dict) and "error" in result):
                error_contacts.append(contact_ids[i])
            else:
                result["original_contact_id"] = contact_ids[i]
                valid_results.append(result)
        
        if not valid_results:
            return {"error": "No valid data available for comparison"}
        
        # Perform comparison analysis
        comparison_data = _analyze_relationships(valid_results, comparison_type)
        
        # Perform clustering if requested
        clusters = {}
        if include_clusters and len(valid_results) >= 3:
            clusters = _identify_relationship_clusters(valid_results)
        
        # Generate insights
        insights = _generate_comparative_insights(comparison_data, clusters, valid_results)
        
        # Build result
        result = {
            "contacts_analyzed": len(valid_results),
            "contacts_errored": len(error_contacts),
            "comparison_type": comparison_type,
            "overview": _generate_overview(valid_results, comparison_data),
            "comparative_matrix": comparison_data["matrix"],
            "insights": insights,
        }
        
        if clusters:
            result["relationship_clusters"] = clusters
        
        if error_contacts:
            result["contacts_with_errors"] = [
                hash_contact_id(c) if redact else c for c in error_contacts
            ]
        
        # Apply redaction if requested
        if redact:
            result = _redact_comparison_results(result, valid_results)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in relationship comparison: {e}")
        return {
            "error": str(e),
            "error_type": "comparison_error",
        }


def _analyze_relationships(
    quality_results: List[Dict], comparison_type: str
) -> Dict[str, Any]:
    """Analyze relationships across multiple dimensions."""
    
    # Extract key metrics for comparison
    metrics = {
        "overall_score": [],
        "depth_score": [],
        "balance_score": [],
        "emotion_score": [],
        "consistency_score": [],
        "messages_analyzed": [],
    }
    
    contact_mapping = {}
    
    for i, result in enumerate(quality_results):
        contact_mapping[i] = result["original_contact_id"]
        
        metrics["overall_score"].append(result["overall_score"])
        metrics["depth_score"].append(result["dimensions"]["depth"]["score"])
        metrics["balance_score"].append(result["dimensions"]["balance"]["score"])
        metrics["emotion_score"].append(result["dimensions"]["emotion"]["score"])
        metrics["consistency_score"].append(result["dimensions"]["consistency"]["score"])
        
        if "analysis_details" in result:
            metrics["messages_analyzed"].append(
                result["analysis_details"].get("messages_analyzed", 0)
            )
        else:
            metrics["messages_analyzed"].append(0)
    
    # Create comparison matrix
    matrix = {}
    for metric_name, values in metrics.items():
        if values:
            matrix[metric_name] = {
                "values": values,
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0,
                "min": min(values),
                "max": max(values),
                "range": max(values) - min(values),
            }
    
    return {
        "matrix": matrix,
        "contact_mapping": contact_mapping,
        "metrics": metrics,
    }


def _identify_relationship_clusters(quality_results: List[Dict]) -> Dict[str, Any]:
    """Identify relationship clusters using K-means clustering."""
    
    # Prepare feature matrix
    features = []
    contact_ids = []
    
    for result in quality_results:
        contact_ids.append(result["original_contact_id"])
        features.append([
            result["overall_score"],
            result["dimensions"]["depth"]["score"],
            result["dimensions"]["balance"]["score"],
            result["dimensions"]["emotion"]["score"],
            result["dimensions"]["consistency"]["score"],
        ])
    
    # Convert to numpy array and scale
    X = np.array(features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal number of clusters (2-4)
    n_clusters = min(3, len(quality_results) - 1)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Analyze clusters
    clusters = defaultdict(list)
    for i, label in enumerate(cluster_labels):
        clusters[label].append({
            "contact_id": contact_ids[i],
            "scores": {
                "overall": features[i][0],
                "depth": features[i][1],
                "balance": features[i][2],
                "emotion": features[i][3],
                "consistency": features[i][4],
            }
        })
    
    # Characterize clusters
    cluster_analysis = []
    cluster_names = ["inner_circle", "regular_contact", "occasional"]
    
    for label, members in clusters.items():
        # Calculate cluster characteristics
        avg_scores = {
            "overall": statistics.mean([m["scores"]["overall"] for m in members]),
            "depth": statistics.mean([m["scores"]["depth"] for m in members]),
            "balance": statistics.mean([m["scores"]["balance"] for m in members]),
            "emotion": statistics.mean([m["scores"]["emotion"] for m in members]),
            "consistency": statistics.mean([m["scores"]["consistency"] for m in members]),
        }
        
        # Determine cluster type based on scores
        cluster_type = _determine_cluster_type(avg_scores, label)
        
        cluster_analysis.append({
            "type": cluster_type,
            "size": len(members),
            "members": [m["contact_id"] for m in members],
            "characteristics": _describe_cluster_characteristics(avg_scores),
            "average_scores": avg_scores,
        })
    
    # Sort clusters by average overall score (descending)
    cluster_analysis.sort(key=lambda x: x["average_scores"]["overall"], reverse=True)
    
    return cluster_analysis


def _determine_cluster_type(avg_scores: Dict[str, float], label: int) -> str:
    """Determine cluster type based on average scores."""
    overall = avg_scores["overall"]
    
    if overall >= 80:
        return "inner_circle"
    elif overall >= 60:
        return "regular_contact"
    elif overall >= 40:
        return "occasional"
    else:
        return "dormant"


def _describe_cluster_characteristics(avg_scores: Dict[str, float]) -> List[str]:
    """Generate descriptive characteristics for a cluster."""
    characteristics = []
    
    # Overall quality
    if avg_scores["overall"] >= 80:
        characteristics.append("High quality relationships")
    elif avg_scores["overall"] >= 60:
        characteristics.append("Moderate quality relationships")
    else:
        characteristics.append("Lower quality relationships")
    
    # Depth
    if avg_scores["depth"] >= 70:
        characteristics.append("Deep, meaningful conversations")
    elif avg_scores["depth"] < 40:
        characteristics.append("Surface-level interactions")
    
    # Balance
    if avg_scores["balance"] >= 80:
        characteristics.append("Well-balanced communication")
    elif avg_scores["balance"] < 50:
        characteristics.append("Imbalanced interaction patterns")
    
    # Emotion
    if avg_scores["emotion"] >= 70:
        characteristics.append("Emotionally rich exchanges")
    elif avg_scores["emotion"] < 40:
        characteristics.append("Limited emotional expression")
    
    # Consistency
    if avg_scores["consistency"] >= 70:
        characteristics.append("Regular, consistent contact")
    elif avg_scores["consistency"] < 40:
        characteristics.append("Sporadic communication")
    
    return characteristics


def _generate_overview(
    quality_results: List[Dict], comparison_data: Dict
) -> Dict[str, Any]:
    """Generate high-level overview of relationships."""
    
    # Find top relationships by different metrics
    sorted_by_overall = sorted(
        quality_results, key=lambda x: x["overall_score"], reverse=True
    )
    sorted_by_messages = sorted(
        quality_results,
        key=lambda x: x.get("analysis_details", {}).get("messages_analyzed", 0),
        reverse=True
    )
    
    overview = {
        "healthiest_relationship": sorted_by_overall[0]["original_contact_id"],
        "highest_quality_score": sorted_by_overall[0]["overall_score"],
    }
    
    if sorted_by_messages and sorted_by_messages[0].get("analysis_details"):
        overview["most_active_relationship"] = sorted_by_messages[0]["original_contact_id"]
        overview["most_messages"] = sorted_by_messages[0]["analysis_details"]["messages_analyzed"]
    
    # Identify relationships needing attention
    needs_attention = [
        r["original_contact_id"] 
        for r in quality_results 
        if r["overall_score"] < 50
    ]
    
    if needs_attention:
        overview["needs_attention"] = needs_attention
    
    # Calculate averages
    avg_scores = comparison_data["matrix"]["overall_score"]["mean"]
    overview["average_quality_score"] = round(avg_scores, 1)
    
    return overview


def _generate_comparative_insights(
    comparison_data: Dict, clusters: List[Dict], quality_results: List[Dict]
) -> Dict[str, Any]:
    """Generate actionable insights from comparison analysis."""
    
    insights = {
        "patterns": [],
        "recommendations": [],
        "strengths": [],
        "opportunities": [],
    }
    
    # Analyze score distributions
    overall_scores = comparison_data["metrics"]["overall_score"]
    if overall_scores:
        score_range = max(overall_scores) - min(overall_scores)
        
        if score_range > 40:
            insights["patterns"].append(
                "Large variation in relationship quality - some relationships need attention"
            )
        elif score_range < 20:
            insights["patterns"].append(
                "Consistent relationship quality across contacts"
            )
    
    # Analyze by dimension
    for dimension in ["depth", "balance", "emotion", "consistency"]:
        dim_key = f"{dimension}_score"
        if dim_key in comparison_data["metrics"]:
            scores = comparison_data["metrics"][dim_key]
            avg_score = statistics.mean(scores) if scores else 0
            
            if avg_score < 50:
                insights["opportunities"].append(
                    f"Overall {dimension} scores are low - focus on improving this area"
                )
            elif avg_score > 80:
                insights["strengths"].append(
                    f"Strong {dimension} across relationships"
                )
    
    # Cluster-based insights
    if clusters:
        for cluster in clusters:
            if cluster["type"] == "inner_circle":
                insights["patterns"].append(
                    f"{cluster['size']} relationships in your inner circle"
                )
            elif cluster["type"] == "dormant" and cluster["size"] > 0:
                insights["recommendations"].append(
                    f"Consider re-engaging with {cluster['size']} dormant relationships"
                )
    
    # Individual relationship insights
    for result in quality_results:
        if result["overall_score"] < 40:
            insights["recommendations"].append(
                f"Relationship with contact needs immediate attention (score: {result['overall_score']})"
            )
        elif result["overall_score"] > 85:
            insights["strengths"].append(
                f"Exceptional relationship quality with one contact (score: {result['overall_score']})"
            )
    
    return insights


def _redact_comparison_results(
    result: Dict, quality_results: List[Dict]
) -> Dict[str, Any]:
    """Apply privacy redaction to comparison results."""
    
    # Create contact ID mapping
    id_mapping = {}
    for i, qr in enumerate(quality_results):
        original_id = qr["original_contact_id"]
        hashed_id = hash_contact_id(original_id)
        id_mapping[original_id] = hashed_id
    
    # Redact overview
    if "overview" in result:
        for key in ["healthiest_relationship", "most_active_relationship"]:
            if key in result["overview"] and result["overview"][key] in id_mapping:
                result["overview"][key] = id_mapping[result["overview"][key]]
        
        if "needs_attention" in result["overview"]:
            result["overview"]["needs_attention"] = [
                id_mapping.get(cid, hash_contact_id(cid))
                for cid in result["overview"]["needs_attention"]
            ]
    
    # Redact clusters
    if "relationship_clusters" in result:
        for cluster in result["relationship_clusters"]:
            cluster["members"] = [
                id_mapping.get(cid, hash_contact_id(cid))
                for cid in cluster["members"]
            ]
    
    # Apply general privacy filters
    return apply_privacy_filters(result)