import logging
import traceback
import networkx as nx
import asyncio
from collections import defaultdict, Counter
from datetime import datetime
import pickle
import os
import time

from src.database.async_messages_db import AsyncMessagesDB
from src.utils.redis_cache import AsyncRedisCache

# Configure logging
logger = logging.getLogger(__name__)

# Cache for network graphs to enable incremental updates
_network_cache = {}
_network_cache_last_updated = {}
_network_cache_lock = asyncio.Lock()

async def analyze_contact_network_async(start_date=None, end_date=None, min_shared_chats=1, use_cache=True, cache_ttl=3600):
    """
    Analyze the network of contacts based on group chat participation.
    
    Args:
        start_date (str, optional): Start date for message filtering (ISO format).
        end_date (str, optional): End date for message filtering (ISO format).
        min_shared_chats (int, optional): Minimum number of shared chats to consider a connection. Defaults to 1.
        use_cache (bool, optional): Whether to use cached graph data for incremental updates.
        cache_ttl (int, optional): Cache TTL in seconds. Defaults to 1 hour.
        
    Returns:
        dict: Dictionary containing the contact network analysis.
    """
    try:
        # Create a cache key based on parameters
        cache_key = f"network:{start_date or 'all'}:{end_date or 'all'}:{min_shared_chats}"
        
        # Check if we have a recent cached graph that can be updated incrementally
        if use_cache:
            graph, last_update_time = await get_cached_network(cache_key)
            
            if graph is not None:
                # If cache is recent enough, update it incrementally
                if (time.time() - last_update_time) < cache_ttl:
                    logger.info(f"Using cached network graph with incremental updates from {datetime.fromtimestamp(last_update_time)}")
                    return await update_network_incrementally(graph, last_update_time, start_date, end_date, min_shared_chats)
        
        # If no cache or cache is too old, build the network from scratch
        logger.info("Building network graph from scratch")
        
        # Initialize DB connection
        db = AsyncMessagesDB()
        
        # Get all group chats
        group_chats = await db.get_group_chats()
        
        # Apply date filter if necessary
        if start_date or end_date:
            filtered_chats = []
            for chat in group_chats:
                if chat.get('last_message_date'):
                    # Parse last message date
                    last_message = datetime.fromisoformat(chat['last_message_date'])
                    
                    # Check start date constraint
                    if start_date:
                        start = datetime.fromisoformat(start_date) if isinstance(start_date, str) else start_date
                        if last_message < start:
                            continue
                            
                    # Check end date constraint
                    if end_date:
                        end = datetime.fromisoformat(end_date) if isinstance(end_date, str) else end_date
                        if last_message > end:
                            continue
                            
                    filtered_chats.append(chat)
            group_chats = filtered_chats
        
        # Map contacts to groups they participate in
        contact_groups = defaultdict(set)
        group_participants = {}
        
        # Process each group chat
        for chat in group_chats:
            chat_id = chat['chat_id']
            participants = await db.get_chat_participants(chat_id)
            
            # Store participants for this chat
            group_participants[chat_id] = participants
            
            # Update contact to group mapping
            for participant in participants:
                contact_id = participant['id']
                contact_groups[contact_id].add(chat_id)
        
        # Create graph
        G = nx.Graph()
        
        # Track connections for future analysis
        connections = defaultdict(int)
        
        # Add nodes (contacts)
        contacts = await db.get_contacts()
        for contact in contacts:
            # We'll only add contacts who participate in at least one group chat
            if contact['id'] in contact_groups and contact_groups[contact['id']]:
                G.add_node(contact['id'], name=contact['name'], message_count=contact['message_count'])
        
        # Add edges (connections between contacts)
        for contact1 in G.nodes():
            for contact2 in G.nodes():
                if contact1 >= contact2:  # Avoid duplicate connections and self-connections
                    continue
                    
                # Find shared group chats
                shared_chats = contact_groups[contact1].intersection(contact_groups[contact2])
                if len(shared_chats) >= min_shared_chats:
                    G.add_edge(contact1, contact2, weight=len(shared_chats), shared_chats=list(shared_chats))
                    connections[(contact1, contact2)] = len(shared_chats)
        
        # Analyze the network
        # Node centrality measures
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        
        # Community detection using Louvain algorithm
        try:
            import community
            partition = community.best_partition(G)
        except ImportError:
            logger.warning("python-louvain package not available. Skipping community detection.")
            partition = {node: 0 for node in G.nodes()}
        
        # Prepare centrality data
        centrality_data = []
        for node in G.nodes():
            centrality_data.append({
                "id": node,
                "name": G.nodes[node]["name"],
                "message_count": G.nodes[node]["message_count"],
                "degree_centrality": degree_centrality[node],
                "betweenness_centrality": betweenness_centrality[node],
                "community": partition[node]
            })
            
        # Sort by betweenness centrality
        centrality_data.sort(key=lambda x: x["betweenness_centrality"], reverse=True)
        
        # Identify strongest connections
        strong_connections = []
        for (source, target), strength in sorted(connections.items(), key=lambda x: x[1], reverse=True):
            # Only include the top connections to avoid overwhelming the result
            if len(strong_connections) >= 30:
                break
                
            source_info = next((c for c in centrality_data if c["id"] == source), None)
            target_info = next((c for c in centrality_data if c["id"] == target), None)
            
            if source_info and target_info:
                strong_connections.append({
                    "source": {
                        "id": source,
                        "name": source_info["name"]
                    },
                    "target": {
                        "id": target,
                        "name": target_info["name"]
                    },
                    "strength": strength
                })
        
        # Calculate network metrics
        density = nx.density(G)
        avg_clustering = nx.average_clustering(G)
        
        # Create result dictionary
        result = {
            "metrics": {
                "node_count": G.number_of_nodes(),
                "edge_count": G.number_of_edges(),
                "density": density,
                "avg_clustering": avg_clustering,
                "communities": len(set(partition.values()))
            },
            "centrality": centrality_data,
            "strong_connections": strong_connections
        }
        
        # Save the graph to cache for future incremental updates
        await cache_network(cache_key, G)
        
        return result
    except Exception as e:
        logger.error(f"Error in analyze_contact_network: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

async def update_network_incrementally(G, last_update_time, start_date=None, end_date=None, min_shared_chats=1):
    """
    Update a network graph incrementally with new data since the last update.
    
    Args:
        G: The existing NetworkX graph
        last_update_time: Timestamp of when the graph was last updated
        start_date: Start date for filtering
        end_date: End date for filtering
        min_shared_chats: Minimum shared chats for a connection
        
    Returns:
        dict: Updated network analysis
    """
    try:
        # Convert timestamp to datetime for comparison
        last_update = datetime.fromtimestamp(last_update_time)
        
        # Initialize DB connection
        db = AsyncMessagesDB()
        
        # Get new messages since last update
        new_message_count = await db.get_message_count_since(last_update)
        
        # If there are no new messages, return the existing graph analysis
        if new_message_count == 0:
            logger.info("No new messages since last update, using cached graph without changes")
            return await analyze_graph(G, min_shared_chats)
        
        logger.info(f"Found {new_message_count} new messages since last update, updating graph incrementally")
        
        # Get group chats with recent activity
        active_chats = await db.get_active_chats_since(last_update)
        
        # For each active chat, update the graph
        for chat in active_chats:
            chat_id = chat['chat_id']
            
            # Get updated participants for this chat
            participants = await db.get_chat_participants(chat_id)
            
            # Get existing participant IDs in the graph
            participant_ids = [p['id'] for p in participants]
            
            # Add new nodes for participants not already in the graph
            for participant in participants:
                contact_id = participant['id']
                
                if contact_id not in G.nodes():
                    # Get contact details to add to the graph
                    contact = await db.get_contact_by_id(contact_id)
                    if contact:
                        G.add_node(contact_id, name=contact['name'], message_count=contact['message_count'])
            
            # Update edges between participants in this chat
            for i, p1 in enumerate(participant_ids):
                if p1 in G.nodes():
                    for p2 in participant_ids[i+1:]:
                        if p2 in G.nodes():
                            if G.has_edge(p1, p2):
                                # Update edge weight
                                shared_chats = G.edges[p1, p2].get('shared_chats', [])
                                if chat_id not in shared_chats:
                                    shared_chats.append(chat_id)
                                    G.edges[p1, p2]['weight'] = len(shared_chats)
                                    G.edges[p1, p2]['shared_chats'] = shared_chats
                            else:
                                # Add new edge if it meets the threshold
                                G.add_edge(p1, p2, weight=1, shared_chats=[chat_id])
        
        # Remove edges that no longer meet the min_shared_chats threshold
        edges_to_remove = []
        for u, v, data in G.edges(data=True):
            if data['weight'] < min_shared_chats:
                edges_to_remove.append((u, v))
                
        for edge in edges_to_remove:
            G.remove_edge(*edge)
        
        # Remove isolated nodes (no connections)
        isolated_nodes = list(nx.isolates(G))
        G.remove_nodes_from(isolated_nodes)
        
        # Update the cache with the modified graph
        cache_key = f"network:{start_date or 'all'}:{end_date or 'all'}:{min_shared_chats}"
        await cache_network(cache_key, G)
        
        # Return analysis of the updated graph
        return await analyze_graph(G, min_shared_chats)
        
    except Exception as e:
        logger.error(f"Error in update_network_incrementally: {e}")
        logger.error(traceback.format_exc())
        
        # If incremental update fails, fall back to full rebuild
        logger.info("Falling back to full network rebuild")
        return await analyze_contact_network_async(start_date, end_date, min_shared_chats, use_cache=False)

async def analyze_graph(G, min_shared_chats=1):
    """
    Analyze a NetworkX graph to extract network metrics and centrality information.
    
    Args:
        G: NetworkX graph object
        min_shared_chats: Minimum shared chats threshold
        
    Returns:
        dict: Network analysis results
    """
    try:
        # Track connections for analysis
        connections = {}
        for u, v, data in G.edges(data=True):
            connections[(u, v)] = data['weight']
        
        # Node centrality measures
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        
        # Community detection using Louvain algorithm
        try:
            import community
            partition = community.best_partition(G)
        except ImportError:
            logger.warning("python-louvain package not available. Skipping community detection.")
            partition = {node: 0 for node in G.nodes()}
        
        # Prepare centrality data
        centrality_data = []
        for node in G.nodes():
            centrality_data.append({
                "id": node,
                "name": G.nodes[node]["name"],
                "message_count": G.nodes[node]["message_count"],
                "degree_centrality": degree_centrality[node],
                "betweenness_centrality": betweenness_centrality[node],
                "community": partition[node]
            })
            
        # Sort by betweenness centrality
        centrality_data.sort(key=lambda x: x["betweenness_centrality"], reverse=True)
        
        # Identify strongest connections
        strong_connections = []
        for (source, target), strength in sorted(connections.items(), key=lambda x: x[1], reverse=True):
            # Only include the top connections to avoid overwhelming the result
            if len(strong_connections) >= 30:
                break
                
            source_info = next((c for c in centrality_data if c["id"] == source), None)
            target_info = next((c for c in centrality_data if c["id"] == target), None)
            
            if source_info and target_info:
                strong_connections.append({
                    "source": {
                        "id": source,
                        "name": source_info["name"]
                    },
                    "target": {
                        "id": target,
                        "name": target_info["name"]
                    },
                    "strength": strength
                })
        
        # Calculate network metrics
        density = nx.density(G)
        avg_clustering = nx.average_clustering(G)
        
        # Create result dictionary
        result = {
            "metrics": {
                "node_count": G.number_of_nodes(),
                "edge_count": G.number_of_edges(),
                "density": density,
                "avg_clustering": avg_clustering,
                "communities": len(set(partition.values()))
            },
            "centrality": centrality_data,
            "strong_connections": strong_connections
        }
        
        return result
    except Exception as e:
        logger.error(f"Error in analyze_graph: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

async def cache_network(key, graph):
    """
    Cache a network graph for future incremental updates.
    
    Args:
        key: Cache key
        graph: NetworkX graph object
    """
    try:
        async with _network_cache_lock:
            _network_cache[key] = graph
            _network_cache_last_updated[key] = time.time()
            
        # Also try to cache in Redis if available
        try:
            cache = AsyncRedisCache()
            serialized = pickle.dumps(graph)
            metadata = {
                "timestamp": time.time(),
                "node_count": graph.number_of_nodes(),
                "edge_count": graph.number_of_edges()
            }
            
            # Store metadata in Redis
            await cache.set(f"graph_meta:{key}", metadata, 86400)  # 24 hour TTL
            
            # For the graph itself, we'll use file-based storage instead of Redis
            # since graphs can be very large and may not fit in Redis memory limits
            cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            cache_file = os.path.join(cache_dir, f"graph_{key.replace(':', '_')}.pkl")
            with open(cache_file, 'wb') as f:
                f.write(serialized)
                
            logger.info(f"Network graph cached to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache graph in Redis/file: {e}")
        
    except Exception as e:
        logger.warning(f"Error caching network: {e}")

async def get_cached_network(key):
    """
    Get a cached network graph.
    
    Args:
        key: Cache key
        
    Returns:
        tuple: (graph, last_update_time) or (None, None) if not found
    """
    try:
        # First check in-memory cache
        async with _network_cache_lock:
            if key in _network_cache:
                return _network_cache[key], _network_cache_last_updated[key]
        
        # Try to get from Redis/file cache
        try:
            cache = AsyncRedisCache()
            metadata = await cache.get(f"graph_meta:{key}")
            
            if metadata:
                # Get from file
                cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "cache")
                cache_file = os.path.join(cache_dir, f"graph_{key.replace(':', '_')}.pkl")
                
                if os.path.exists(cache_file):
                    with open(cache_file, 'rb') as f:
                        graph = pickle.loads(f.read())
                        
                    # Store in memory cache
                    async with _network_cache_lock:
                        _network_cache[key] = graph
                        _network_cache_last_updated[key] = metadata["timestamp"]
                        
                    return graph, metadata["timestamp"]
        except Exception as e:
            logger.warning(f"Failed to retrieve graph from Redis/file: {e}")
        
        return None, None
    except Exception as e:
        logger.warning(f"Error retrieving cached network: {e}")
        return None, None

async def identify_key_connectors_async(start_date=None, end_date=None):
    """
    Identify key connectors - contacts who connect different social circles.
    
    Args:
        start_date (str, optional): Start date for message filtering (ISO format).
        end_date (str, optional): End date for message filtering (ISO format).
    
    Returns:
        dict: Dictionary containing the key connectors analysis.
    """
    try:
        # Get network analysis
        network = await analyze_contact_network_async(start_date, end_date)
        
        # Sort contacts by betweenness centrality
        key_connectors = sorted(
            network['centrality'], 
            key=lambda x: x['betweenness_centrality'], 
            reverse=True
        )[:10]  # Top 10 connectors
        
        # Return results
        return {
            'key_connectors': key_connectors,
            'date_range': {
                'start': start_date,
                'end': end_date
            }
        }
    except Exception as e:
        logger.error(f"Error in identify_key_connectors_async: {e}")
        logger.error(traceback.format_exc())
        return {'error': str(e)}

async def analyze_social_circles_async(start_date=None, end_date=None):
    """
    Analyze social circles (communities) within the contact network.
    
    Args:
        start_date (str, optional): Start date for message filtering (ISO format).
        end_date (str, optional): End date for message filtering (ISO format).
    
    Returns:
        dict: Dictionary containing the social circles analysis.
    """
    try:
        # Get network analysis
        network = await analyze_contact_network_async(start_date, end_date)
        
        # Get communities
        communities = network['communities']
        
        # Sort communities by size
        communities.sort(key=lambda x: len(x['members']), reverse=True)
        
        # Return results
        return {
            'social_circles': communities,
            'date_range': {
                'start': start_date,
                'end': end_date
            }
        }
    except Exception as e:
        logger.error(f"Error in analyze_social_circles_async: {e}")
        logger.error(traceback.format_exc())
        return {'error': str(e)} 