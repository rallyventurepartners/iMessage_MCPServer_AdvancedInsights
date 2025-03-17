import logging
import traceback
import networkx as nx
import asyncio
import numpy as np
from collections import defaultdict

from src.analysis.async_network_analysis import analyze_contact_network_async
from src.database.async_messages_db import AsyncMessagesDB

# Configure logging
logger = logging.getLogger(__name__)

async def generate_network_visualization_async(start_date=None, end_date=None, min_shared_chats=1, layout="spring"):
    """
    Generate visualization data for the contact network.
    
    Args:
        start_date (str, optional): Start date for message filtering (ISO format).
        end_date (str, optional): End date for message filtering (ISO format).
        min_shared_chats (int, optional): Minimum number of shared chats to consider a connection.
        layout (str, optional): Layout algorithm to use (spring, circular, kamada_kawai).
    
    Returns:
        dict: Dictionary containing network visualization data.
    """
    try:
        # Get network analysis
        network_data = await analyze_contact_network_async(start_date, end_date, min_shared_chats)
        
        if "error" in network_data:
            return network_data
            
        # Initialize DB connection
        db = AsyncMessagesDB()
        
        # Get centrality information from analysis
        centrality_data = network_data["centrality"]
        
        # Create a new graph for visualization
        G = nx.Graph()
        
        # Add nodes (contacts)
        for node in centrality_data:
            contact_id = node["id"]
            contact_name = node["name"]
            centrality = node["betweenness_centrality"]
            community = node["community"]
            
            # Add node with visualization attributes
            G.add_node(
                contact_id, 
                name=contact_name, 
                centrality=centrality,
                community=community
            )
            
        # Add edges (connections)
        for connection in network_data["strong_connections"]:
            source = connection["source"]["id"]
            target = connection["target"]["id"]
            strength = connection["strength"]
            
            # Only add the edge if both nodes exist
            if source in G.nodes() and target in G.nodes():
                G.add_edge(source, target, weight=strength)
                
        # Generate layout
        pos = generate_layout(G, layout)
        
        # Create mapping of community IDs to distinct colors
        community_colors = {}
        for node in centrality_data:
            if node["community"] is not None and node["community"] not in community_colors:
                # Assign a color index based on community ID
                community_colors[node["community"]] = node["community"]
                
        # Normalize node sizes based on centrality
        centralities = [G.nodes[n]["centrality"] for n in G.nodes()]
        min_centrality = min(centralities) if centralities else 0
        max_centrality = max(centralities) if centralities else 1
        range_centrality = max_centrality - min_centrality
        
        if range_centrality == 0:
            range_centrality = 1  # Avoid division by zero
            
        # Prepare node and edge data
        nodes = []
        for node_id in G.nodes():
            node = G.nodes[node_id]
            
            # Calculate node size (scaled by centrality)
            size = 10 + (node["centrality"] - min_centrality) / range_centrality * 40
            
            # Get node position
            position = pos[node_id]
            
            # Get community color
            color = node.get("community", 0)
            
            nodes.append({
                "id": node_id,
                "name": node["name"],
                "x": float(position[0]),
                "y": float(position[1]),
                "size": float(size),
                "color": color,
                "centrality": node["centrality"]
            })
            
        edges = []
        for source, target, data in G.edges(data=True):
            # Scale edge thickness by weight
            thickness = 1 + data.get("weight", 1) * 0.5
            
            edges.append({
                "source": source,
                "target": target,
                "weight": data.get("weight", 1),
                "thickness": thickness
            })
            
        # Prepare visualization metadata
        layout_algorithm = layout
        node_count = G.number_of_nodes()
        edge_count = G.number_of_edges()
        community_count = len(community_colors)
        
        # Return visualization data
        return {
            "metadata": {
                "layout": layout_algorithm,
                "nodes": node_count,
                "edges": edge_count,
                "communities": community_count,
                "date_range": {
                    "start": start_date,
                    "end": end_date
                }
            },
            "nodes": nodes,
            "edges": edges
        }
    except Exception as e:
        logger.error(f"Error in generate_network_visualization: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

def generate_layout(G, layout_name="spring"):
    """Generate a layout for the graph using the specified algorithm.
    
    Optimized for performance by:
    1. Using a fixed random seed for deterministic results
    2. Limiting iterations to balance quality and speed
    3. Tuning parameters for faster convergence
    4. Implementing simplified calculations for larger graphs
    
    Args:
        G: NetworkX graph
        layout_name: Name of the layout algorithm to use
        
    Returns:
        Dictionary mapping node IDs to position coordinates
    """
    # Calculate graph size to apply adaptive optimizations
    num_nodes = G.number_of_nodes()
    
    # Apply different optimization strategies based on graph size
    if layout_name == "circular":
        # Circular layout is always fast, no optimization needed
        return nx.circular_layout(G)
    elif layout_name == "kamada_kawai":
        try:
            if num_nodes > 100:
                # For large graphs, reduce computational complexity
                logger.info(f"Large graph detected ({num_nodes} nodes). Using optimized kamada_kawai parameters.")
                
                # Pre-compute a simpler layout to use as a starting point
                initial_pos = nx.spring_layout(G, seed=42, iterations=5)
                
                # Use a higher tolerance for faster convergence
                return nx.kamada_kawai_layout(G, pos=initial_pos, tol=1e-2)
            else:
                # For smaller graphs, standard parameters are fine
                return nx.kamada_kawai_layout(G, tol=1e-3)
        except Exception as e:
            logger.warning(f"Kamada-Kawai layout failed: {e}. Falling back to spring layout.")
            # Fall back to spring layout
            return nx.spring_layout(G, seed=42, k=0.3, iterations=30)
    else:
        # Spring layout (default)
        if num_nodes > 200:
            # For very large graphs, prioritize speed over quality
            logger.info(f"Very large graph detected ({num_nodes} nodes). Using fast spring layout.")
            return nx.spring_layout(G, k=0.3, iterations=20, seed=42)
        elif num_nodes > 50:
            # For medium graphs, balanced settings
            return nx.spring_layout(G, k=0.3, iterations=30, seed=42)
        else:
            # For small graphs, prioritize quality
            return nx.spring_layout(G, k=0.3, iterations=50, seed=42) 