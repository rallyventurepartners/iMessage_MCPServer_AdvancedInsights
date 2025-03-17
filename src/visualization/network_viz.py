import logging
import traceback
import json
import numpy as np
from typing import Dict, Any, List, Optional

from src.database.messages_db import MessagesDB
from src.utils.helpers import error_response, validate_date_range
from src.analysis.network_analysis import analyze_contact_network_advanced

logger = logging.getLogger(__name__)

def generate_network_visualization(
    start_date: str = None,
    end_date: str = None,
    min_shared_chats: int = 1,
    max_nodes: int = 100,
    include_labels: bool = True,
    layout_algorithm: str = "force_directed",
    color_by: str = "community"
) -> Dict[str, Any]:
    """Generate a detailed visualization of the contact network.
    
    This function prepares data for visualizing the social network of iMessage contacts,
    optimized for different visualization libraries and formats.
    
    Args:
        start_date: Optional start date in ISO format (YYYY-MM-DD)
        end_date: Optional end date in ISO format (YYYY-MM-DD)
        min_shared_chats: Minimum number of shared chats required to consider contacts connected
        max_nodes: Maximum number of nodes to include in the visualization (most connected first)
        include_labels: Whether to include contact name labels in the visualization
        layout_algorithm: Algorithm to use for node positioning ('force_directed', 'circular', 'spectral')
        color_by: Property to use for node coloring ('community', 'connectivity', 'activity')
        
    Returns:
        Dictionary containing visualization data in multiple formats
    """
    logger.info(f"generate_network_visualization called with start_date={start_date}, end_date={end_date}, "
               f"min_shared_chats={min_shared_chats}, max_nodes={max_nodes}, layout={layout_algorithm}")
    
    # Parameter validation
    if max_nodes < 5:
        return error_response("INVALID_PARAMETER", "max_nodes must be at least 5")
    
    if layout_algorithm not in ['force_directed', 'circular', 'spectral']:
        return error_response("INVALID_PARAMETER", "layout_algorithm must be one of: force_directed, circular, spectral")
    
    if color_by not in ['community', 'connectivity', 'activity']:
        return error_response("INVALID_PARAMETER", "color_by must be one of: community, connectivity, activity")
    
    # Validate date range
    error = validate_date_range(start_date, end_date)
    if error:
        return error
    
    try:
        # Get the network data using our advanced analysis function
        network_data = analyze_contact_network_advanced(
            start_date=start_date,
            end_date=end_date,
            min_shared_chats=min_shared_chats,
            include_visualization_data=True
        )
        
        # Check if there was an error or insufficient data
        if "error" in network_data:
            return network_data
        
        if "warning" in network_data:
            return {
                "warning": network_data["warning"],
                "formats": {}
            }
        
        # Get visualization data from the network analysis
        vis_data = network_data.get("visualization", {})
        if not vis_data:
            return error_response("VISUALIZATION_ERROR", "Could not generate visualization data")
        
        # Limit to max_nodes by selecting the most connected nodes
        if len(vis_data.get("nodes", [])) > max_nodes:
            # Sort nodes by size (which corresponds to connectivity)
            sorted_nodes = sorted(vis_data["nodes"], key=lambda x: x.get("size", 0), reverse=True)
            
            # Take the top max_nodes
            selected_nodes = sorted_nodes[:max_nodes]
            selected_node_ids = {node["id"] for node in selected_nodes}
            
            # Filter edges to only include those connecting selected nodes
            selected_edges = [
                edge for edge in vis_data.get("edges", [])
                if edge["source"] in selected_node_ids and edge["target"] in selected_node_ids
            ]
            
            # Update vis_data
            vis_data["nodes"] = selected_nodes
            vis_data["edges"] = selected_edges
        
        # Apply layout algorithm if different from default
        if layout_algorithm != "force_directed" and vis_data.get("nodes"):
            # Import NetworkX if needed
            try:
                import networkx as nx
                
                # Create a NetworkX graph from the visualization data
                G = nx.Graph()
                
                # Add nodes
                for node in vis_data["nodes"]:
                    G.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})
                
                # Add edges
                for edge in vis_data["edges"]:
                    G.add_edge(edge["source"], edge["target"], **{k: v for k, v in edge.items() if k not in ["source", "target"]})
                
                # Apply the selected layout algorithm
                if layout_algorithm == "circular":
                    pos = nx.circular_layout(G)
                elif layout_algorithm == "spectral":
                    pos = nx.spectral_layout(G)
                else:
                    # Fallback to force-directed if something unexpected happens
                    pos = nx.spring_layout(G)
                
                # Update node positions in vis_data
                for node in vis_data["nodes"]:
                    node_id = node["id"]
                    if node_id in pos:
                        node["x"] = float(pos[node_id][0])
                        node["y"] = float(pos[node_id][1])
                
            except ImportError:
                logger.warning("NetworkX library not available, using default layout")
        
        # Apply coloring scheme if different from default
        if color_by != "community" and vis_data.get("nodes"):
            # Get the MessagesDB instance
            db = MessagesDB()
            
            if color_by == "connectivity":
                # Color nodes based on their degree (number of connections)
                # First calculate degree for each node
                node_degrees = {}
                for edge in vis_data["edges"]:
                    node_degrees[edge["source"]] = node_degrees.get(edge["source"], 0) + 1
                    node_degrees[edge["target"]] = node_degrees.get(edge["target"], 0) + 1
                
                # Find min and max degrees for normalization
                if node_degrees:
                    min_degree = min(node_degrees.values())
                    max_degree = max(node_degrees.values())
                    degree_range = max(1, max_degree - min_degree)  # Avoid division by zero
                    
                    # Update node colors
                    for node in vis_data["nodes"]:
                        node_id = node["id"]
                        degree = node_degrees.get(node_id, 0)
                        
                        # Normalize to 0-1 range and convert to color
                        intensity = (degree - min_degree) / degree_range
                        # Use a blue-to-red gradient
                        r = int(255 * intensity)
                        b = int(255 * (1 - intensity))
                        g = int(100 + 50 * abs(0.5 - intensity))  # Peak green in the middle
                        
                        node["color"] = f"#{r:02x}{g:02x}{b:02x}"
            
            elif color_by == "activity":
                # Color nodes based on their message activity
                # This is more complex and requires additional database queries
                
                # Get contact IDs from nodes
                contact_ids = [node["id"] for node in vis_data["nodes"]]
                
                # Create a dictionary to store activity levels
                activity_levels = {}
                
                # For each contact, count their messages
                for contact_id in contact_ids:
                    try:
                        # Count messages for this contact
                        messages = db.get_chat_transcript(
                            chat_id=None,
                            phone_number=contact_id,
                            start_date=start_date,
                            end_date=end_date
                        )
                        
                        # Store the message count
                        activity_levels[contact_id] = len(messages) if messages else 0
                    except:
                        # If there's an error, default to 0
                        activity_levels[contact_id] = 0
                
                # Find min and max activity for normalization
                if activity_levels:
                    min_activity = min(activity_levels.values())
                    max_activity = max(activity_levels.values())
                    activity_range = max(1, max_activity - min_activity)  # Avoid division by zero
                    
                    # Update node colors
                    for node in vis_data["nodes"]:
                        node_id = node["id"]
                        activity = activity_levels.get(node_id, 0)
                        
                        # Normalize to 0-1 range and convert to color
                        intensity = (activity - min_activity) / activity_range
                        # Use a green-to-purple gradient for activity
                        r = int(100 + 155 * intensity)
                        g = int(200 * (1 - intensity))
                        b = int(100 + 155 * intensity)
                        
                        node["color"] = f"#{r:02x}{g:02x}{b:02x}"
        
        # Remove labels if not requested
        if not include_labels and vis_data.get("nodes"):
            for node in vis_data["nodes"]:
                if "label" in node:
                    del node["label"]
        
        # Prepare visualization data in different formats
        
        # 1. D3.js format
        d3_format = {
            "nodes": vis_data.get("nodes", []),
            "links": [
                {
                    "source": edge["source"],
                    "target": edge["target"],
                    "value": edge.get("weight", 1)
                }
                for edge in vis_data.get("edges", [])
            ]
        }
        
        # 2. Sigma.js format (similar to our original vis_data)
        sigma_format = {
            "nodes": vis_data.get("nodes", []),
            "edges": vis_data.get("edges", [])
        }
        
        # 3. Cytoscape.js format
        cytoscape_format = {
            "nodes": [
                {
                    "data": {
                        "id": node["id"],
                        "label": node.get("label", ""),
                        "size": node.get("size", 1),
                        "color": node.get("color", "#666666"),
                        "community": node.get("community", 0)
                    },
                    "position": {
                        "x": node.get("x", 0) * 100,  # Scale positions for Cytoscape
                        "y": node.get("y", 0) * 100
                    }
                }
                for node in vis_data.get("nodes", [])
            ],
            "edges": [
                {
                    "data": {
                        "id": f"e{i}",
                        "source": edge["source"],
                        "target": edge["target"],
                        "weight": edge.get("weight", 1),
                        "label": edge.get("label", "")
                    }
                }
                for i, edge in enumerate(vis_data.get("edges", []))
            ]
        }
        
        # 4. Simple JSON format for custom visualizations
        simple_format = {
            "nodes": [
                {
                    "id": node["id"],
                    "name": node.get("label", ""),
                    "x": node.get("x", 0),
                    "y": node.get("y", 0),
                    "size": node.get("size", 1),
                    "color": node.get("color", "#666666"),
                    "group": node.get("community", 0)
                }
                for node in vis_data.get("nodes", [])
            ],
            "edges": [
                {
                    "from": edge["source"],
                    "to": edge["target"],
                    "weight": edge.get("weight", 1)
                }
                for edge in vis_data.get("edges", [])
            ]
        }
        
        # Gather network metrics for the visualization
        metrics = network_data.get("metrics", {})
        
        return {
            "formats": {
                "d3": d3_format,
                "sigma": sigma_format,
                "cytoscape": cytoscape_format,
                "simple": simple_format
            },
            "metrics": metrics,
            "settings": {
                "layout": layout_algorithm,
                "color_scheme": color_by,
                "max_nodes": max_nodes,
                "include_labels": include_labels,
                "date_range": {
                    "start": start_date,
                    "end": end_date
                }
            }
        }
        
    except ImportError as e:
        logger.error(f"Missing required library for network visualization: {e}")
        return error_response(
            "MISSING_DEPENDENCY", 
            "This visualization requires additional libraries. Please install them with: pip install networkx numpy"
        )
    except Exception as e:
        logger.error(f"Error generating network visualization: {e}")
        logger.error(traceback.format_exc())
        return error_response("VISUALIZATION_ERROR", f"Error generating visualization: {str(e)}") 