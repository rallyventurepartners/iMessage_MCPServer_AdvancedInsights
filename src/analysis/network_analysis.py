import logging
import traceback
import networkx as nx
from collections import defaultdict
from community import best_partition
from typing import Dict, Any, List, Union, Optional

from src.database.messages_db import MessagesDB
from src.utils.helpers import error_response, validate_date_range, ensure_datetime, format_contact_name

logger = logging.getLogger(__name__)

def analyze_contact_network(
    start_date: str = None,
    end_date: str = None,
    min_shared_chats: int = 1
) -> Dict[str, Any]:
    """Analyze the network of contacts based on group chat participation.
    
    This function maps connections between contacts by identifying who appears together
    in the same group chats, creating a social graph of your messaging network.
    
    Args:
        start_date: Optional start date in ISO format (YYYY-MM-DD) to limit analysis timeframe
        end_date: Optional end date in ISO format (YYYY-MM-DD) to limit analysis timeframe
        min_shared_chats: Minimum number of shared chats required to consider contacts connected (default: 1)
        
    Returns:
        Dictionary containing contact network analysis including nodes, connections, and clusters
    """
    logger.info(f"analyze_contact_network called with start_date={start_date}, end_date={end_date}, min_shared_chats={min_shared_chats}")
    
    # Validate date range
    error = validate_date_range(start_date, end_date)
    if error:
        return error
    
    try:
        db = MessagesDB()
        
        # Convert date strings to datetime objects
        start_dt = ensure_datetime(start_date) if start_date else None
        end_dt = ensure_datetime(end_date) if end_date else None
        
        # Get all group chats
        group_chats = db.get_group_chats()
        if not group_chats:
            return {
                "warning": "No group chats found",
                "nodes": [],
                "connections": [],
                "clusters": []
            }
        
        # Track which contacts appear in which group chats
        contact_to_chats = defaultdict(set)
        chat_to_contacts = defaultdict(set)
        total_contacts = set()
        
        # For each group chat, get its participants
        for chat in group_chats:
            chat_id = chat["chat_id"]
            
            # Apply date filter if specified
            if start_dt or end_dt:
                # Check if the chat has activity in the specified date range
                if chat.get("last_message_date"):
                    try:
                        last_message = ensure_datetime(chat["last_message_date"])
                        if (start_dt and last_message < start_dt) or (end_dt and last_message > end_dt):
                            # Skip this chat as it's outside our date range
                            continue
                    except (ValueError, TypeError):
                        # If we can't parse the date, include the chat by default
                        pass
            
            # Get participants for this chat
            participants = db.get_chat_participants(chat_id)
            for participant in participants:
                participant_id = participant.get("id")
                if participant_id:
                    contact_to_chats[participant_id].add(chat_id)
                    chat_to_contacts[chat_id].add(participant_id)
                    total_contacts.add(participant_id)
        
        # Calculate connections between contacts (edges in the graph)
        contact_connections = []
        contact_pairs_processed = set()  # To avoid duplicate connections
        
        for contact_a in total_contacts:
            for contact_b in total_contacts:
                # Skip self-connections and duplicates
                if contact_a == contact_b or (contact_a, contact_b) in contact_pairs_processed or (contact_b, contact_a) in contact_pairs_processed:
                    continue
                
                # Find shared chats
                shared_chats = contact_to_chats[contact_a].intersection(contact_to_chats[contact_b])
                
                # Only create a connection if they share enough chats
                if len(shared_chats) >= min_shared_chats:
                    # Get display names for both contacts
                    contact_a_name = db.get_contact_name(contact_a) or contact_a
                    contact_b_name = db.get_contact_name(contact_b) or contact_b
                    
                    display_a = format_contact_name(contact_a, contact_a_name)
                    display_b = format_contact_name(contact_b, contact_b_name)
                    
                    contact_connections.append({
                        "source": contact_a,
                        "source_display_name": display_a,
                        "target": contact_b,
                        "target_display_name": display_b,
                        "shared_chats": len(shared_chats),
                        "chat_ids": list(shared_chats)
                    })
                    
                    # Mark this pair as processed
                    contact_pairs_processed.add((contact_a, contact_b))
        
        # Identify clusters/communities in the network
        # We'll use a simple approach: connect contacts who share at least one other contact
        clusters = []
        contacts_processed = set()
        
        # Helper function to find contacts in a cluster
        def find_cluster_members(seed_contact, current_cluster):
            """Recursively find all members of a cluster starting from a seed contact."""
            if seed_contact in contacts_processed:
                return
                
            current_cluster.add(seed_contact)
            contacts_processed.add(seed_contact)
            
            # Find all contacts connected to this one
            connected_contacts = set()
            for conn in contact_connections:
                if conn["source"] == seed_contact:
                    connected_contacts.add(conn["target"])
                elif conn["target"] == seed_contact:
                    connected_contacts.add(conn["source"])
            
            # Recursively add connected contacts to cluster
            for contact in connected_contacts:
                if contact not in contacts_processed:
                    find_cluster_members(contact, current_cluster)
        
        # Identify clusters
        for contact in total_contacts:
            if contact not in contacts_processed:
                current_cluster = set()
                find_cluster_members(contact, current_cluster)
                
                if current_cluster:
                    # Get display names for contacts in this cluster
                    cluster_members = []
                    for member in current_cluster:
                        name = db.get_contact_name(member) or member
                        display_name = format_contact_name(member, name)
                        
                        # Get number of connections for this member
                        connection_count = sum(1 for conn in contact_connections 
                                          if conn["source"] == member or conn["target"] == member)
                        
                        cluster_members.append({
                            "id": member,
                            "display_name": display_name,
                            "connections": connection_count,
                            "groups": len(contact_to_chats[member])
                        })
                    
                    # Sort cluster members by connection count
                    cluster_members.sort(key=lambda x: x["connections"], reverse=True)
                    
                    clusters.append({
                        "size": len(cluster_members),
                        "members": cluster_members,
                        "total_connections": len([c for c in contact_connections 
                                               if c["source"] in current_cluster and c["target"] in current_cluster])
                    })
        
        # Sort clusters by size
        clusters.sort(key=lambda x: x["size"], reverse=True)
        
        # Create nodes for the network graph
        nodes = []
        for contact in total_contacts:
            contact_name = db.get_contact_name(contact) or contact
            display_name = format_contact_name(contact, contact_name)
            
            # Count connections for this contact
            connection_count = sum(1 for conn in contact_connections 
                               if conn["source"] == contact or conn["target"] == contact)
            
            # Count groups this contact is in
            group_count = len(contact_to_chats[contact])
            
            nodes.append({
                "id": contact,
                "display_name": display_name,
                "connection_count": connection_count,
                "group_count": group_count
            })
        
        # Sort nodes by connection count
        nodes.sort(key=lambda x: x["connection_count"], reverse=True)
        
        # Sort connections by shared chat count
        contact_connections.sort(key=lambda x: x["shared_chats"], reverse=True)
        
        # Generate network statistics
        stats = {
            "total_contacts": len(total_contacts),
            "total_connections": len(contact_connections),
            "total_clusters": len(clusters),
            "average_connections_per_contact": round(len(contact_connections) * 2 / len(total_contacts), 2) if total_contacts else 0,
            "most_connected_contact": nodes[0]["display_name"] if nodes else None,
            "largest_cluster_size": clusters[0]["size"] if clusters else 0
        }
        
        return {
            "nodes": nodes,
            "connections": contact_connections,
            "clusters": clusters,
            "stats": stats,
            "date_range": {
                "start": start_date,
                "end": end_date
            }
        }
        
    except Exception as e:
        logger.error(f"Error analyzing contact network: {e}")
        logger.error(traceback.format_exc())
        return error_response("ANALYSIS_ERROR", f"Error analyzing contact network: {str(e)}")

def analyze_contact_network_advanced(
    start_date: str = None,
    end_date: str = None,
    min_shared_chats: int = 1,
    include_visualization_data: bool = True
) -> Dict[str, Any]:
    """Perform advanced social network analysis on iMessage contacts using NetworkX.
    
    This function builds a social graph from your iMessage contacts and applies sophisticated
    network analysis algorithms to identify key influencers, communities, and communication patterns.
    
    Args:
        start_date: Optional start date in ISO format (YYYY-MM-DD) to limit analysis timeframe
        end_date: Optional end date in ISO format (YYYY-MM-DD) to limit analysis timeframe
        min_shared_chats: Minimum number of shared chats required to consider contacts connected (default: 1)
        include_visualization_data: Whether to include data formatted for network visualizations (default: True)
        
    Returns:
        Dictionary containing detailed social network metrics and analysis
    """
    logger.info(f"analyze_contact_network_advanced called with start_date={start_date}, end_date={end_date}, min_shared_chats={min_shared_chats}")
    
    # Get basic contact network data using our existing function
    basic_network = analyze_contact_network(start_date, end_date, min_shared_chats)
    
    # Check if there was an error or no data
    if "error" in basic_network:
        return basic_network
    
    if not basic_network.get("connections"):
        return {
            "warning": "Not enough connections to perform network analysis",
            "metrics": {},
            "centrality": [],
            "communities": []
        }
    
    try:
        # Create a NetworkX graph from our connections data
        G = nx.Graph()
        
        # Add nodes (contacts) with attributes
        for node in basic_network["nodes"]:
            G.add_node(
                node["id"],
                display_name=node["display_name"],
                group_count=node["group_count"]
            )
        
        # Add edges (connections) with attributes
        for conn in basic_network["connections"]:
            G.add_edge(
                conn["source"],
                conn["target"],
                weight=conn["shared_chats"],
                shared_chats=conn["shared_chats"]
            )
        
        # Calculate basic network metrics
        metrics = {
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "density": nx.density(G),
            "is_connected": nx.is_connected(G),
            "average_shortest_path_length": nx.average_shortest_path_length(G) if nx.is_connected(G) else None,
            "diameter": nx.diameter(G) if nx.is_connected(G) else None,
            "average_clustering": nx.average_clustering(G),
            "transitivity": nx.transitivity(G)
        }
        
        # Calculate centrality measures
        # Degree centrality - Who has the most connections?
        degree_centrality = nx.degree_centrality(G)
        
        # Betweenness centrality - Who bridges different groups?
        betweenness_centrality = nx.betweenness_centrality(G)
        
        # Closeness centrality - Who can reach others most efficiently?
        closeness_centrality = nx.closeness_centrality(G)
        
        # Eigenvector centrality - Who is connected to other important nodes?
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
        
        # PageRank - Alternative influence measure
        pagerank = nx.pagerank(G)
        
        # Combine centrality measures with contact information
        centrality_data = []
        db = MessagesDB()
        
        for node_id in G.nodes():
            # Get contact display info
            contact_name = db.get_contact_name(node_id) or node_id
            display_name = format_contact_name(node_id, contact_name)
            
            # Create centrality record
            centrality_data.append({
                "id": node_id,
                "display_name": display_name,
                "metrics": {
                    "degree": round(degree_centrality.get(node_id, 0), 4),
                    "betweenness": round(betweenness_centrality.get(node_id, 0), 4),
                    "closeness": round(closeness_centrality.get(node_id, 0), 4),
                    "eigenvector": round(eigenvector_centrality.get(node_id, 0), 4),
                    "pagerank": round(pagerank.get(node_id, 0), 4)
                }
            })
        
        # Sort by degree centrality (most connected first)
        centrality_data.sort(key=lambda x: x["metrics"]["degree"], reverse=True)
        
        # Identify key players
        key_players = {
            "most_connected": centrality_data[0]["display_name"] if centrality_data else None,
            "top_bridger": max(centrality_data, key=lambda x: x["metrics"]["betweenness"])["display_name"] if centrality_data else None,
            "most_central": max(centrality_data, key=lambda x: x["metrics"]["closeness"])["display_name"] if centrality_data else None,
            "most_influential": max(centrality_data, key=lambda x: x["metrics"]["eigenvector"])["display_name"] if centrality_data else None
        }
        
        # Community detection using Louvain method
        communities_dict = best_partition(G)
        
        # Group nodes by community
        community_groups = defaultdict(list)
        for node_id, community_id in communities_dict.items():
            contact_name = db.get_contact_name(node_id) or node_id
            display_name = format_contact_name(node_id, contact_name)
            
            # Get centrality metrics for this contact
            node_metrics = next((item["metrics"] for item in centrality_data if item["id"] == node_id), {})
            
            community_groups[community_id].append({
                "id": node_id,
                "display_name": display_name,
                "centrality": node_metrics
            })
        
        # Format communities data
        communities_data = []
        for community_id, members in community_groups.items():
            # Sort community members by degree centrality
            members.sort(key=lambda x: x.get("centrality", {}).get("degree", 0), reverse=True)
            
            # Identify the leader of this community (most central person)
            community_leader = members[0]["display_name"] if members else "Unknown"
            
            # Calculate community size and density
            community_nodes = [member["id"] for member in members]
            community_subgraph = G.subgraph(community_nodes)
            community_density = nx.density(community_subgraph)
            
            communities_data.append({
                "id": community_id,
                "size": len(members),
                "leader": community_leader,
                "density": round(community_density, 4),
                "cohesion": round(nx.transitivity(community_subgraph) if len(members) > 2 else 0, 4),
                "members": members
            })
        
        # Sort communities by size (largest first)
        communities_data.sort(key=lambda x: x["size"], reverse=True)
        
        # Visualization data (for network graphs)
        visualization_data = None
        if include_visualization_data:
            # Generate positions for nodes using a force-directed layout
            pos = nx.spring_layout(G)
            
            # Prepare nodes data
            nodes_viz = []
            for node_id in G.nodes():
                community_id = communities_dict.get(node_id, 0)
                contact_name = db.get_contact_name(node_id) or node_id
                display_name = format_contact_name(node_id, contact_name)
                
                nodes_viz.append({
                    "id": node_id,
                    "label": display_name,
                    "x": float(pos[node_id][0]),
                    "y": float(pos[node_id][1]),
                    "size": 1 + degree_centrality.get(node_id, 0) * 20,  # Adjust size based on degree
                    "color": f"#{hash(community_id) % 0xFFFFFF:06x}",  # Generate color from community ID
                    "community": community_id
                })
            
            # Prepare edges data
            edges_viz = []
            for source, target, data in G.edges(data=True):
                weight = data.get("weight", 1)
                edges_viz.append({
                    "source": source,
                    "target": target,
                    "weight": weight,
                    "size": 0.5 + (weight / 5),  # Adjust size based on weight
                    "label": f"{weight} shared groups"
                })
            
            visualization_data = {
                "nodes": nodes_viz,
                "edges": edges_viz
            }
        
        # Calculate inter-community connections
        inter_community_connections = []
        for source, target, data in G.edges(data=True):
            source_community = communities_dict.get(source)
            target_community = communities_dict.get(target)
            
            if source_community != target_community:
                source_name = format_contact_name(source, db.get_contact_name(source) or source)
                target_name = format_contact_name(target, db.get_contact_name(target) or target)
                
                inter_community_connections.append({
                    "source": source,
                    "source_name": source_name,
                    "source_community": source_community,
                    "target": target,
                    "target_name": target_name,
                    "target_community": target_community,
                    "weight": data.get("weight", 1)
                })
        
        # Sort by weight
        inter_community_connections.sort(key=lambda x: x["weight"], reverse=True)
        
        # Prepare final analysis result
        result = {
            "metrics": metrics,
            "key_players": key_players,
            "centrality": centrality_data,
            "communities": communities_data,
            "inter_community_connections": inter_community_connections[:20],  # Limit to top 20
            "date_range": basic_network.get("date_range"),
            "raw_network": {
                "nodes": basic_network.get("nodes"),
                "connections": basic_network.get("connections")
            }
        }
        
        # Add visualization data if requested
        if include_visualization_data and visualization_data:
            result["visualization"] = visualization_data
        
        return result
        
    except ImportError as e:
        logger.error(f"Missing required library for network analysis: {e}")
        return error_response(
            "MISSING_DEPENDENCY", 
            "This analysis requires NetworkX and python-louvain libraries. Please install them with: pip install networkx python-louvain"
        )
    except Exception as e:
        logger.error(f"Error performing advanced network analysis: {e}")
        logger.error(traceback.format_exc())
        return error_response("ANALYSIS_ERROR", f"Error performing network analysis: {str(e)}") 