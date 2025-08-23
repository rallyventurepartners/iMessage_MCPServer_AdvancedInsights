"""
Network and social graph analysis tools.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

from mcp import Server

from ..config import Config
from ..db import get_database
from ..models import NetworkIntelligenceInput, NetworkIntelligenceOutput
from ..privacy import hash_contact_id

logger = logging.getLogger(__name__)


def register_network_tools(server: Server, config: Config) -> None:
    """Register network analysis tools with the server."""
    
    @server.tool()
    async def imsg_network_intelligence(arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build minimal social graph from group chats.
        
        Analyzes group chat participation to identify social connections,
        communities, and key connectors in the network.
        """
        try:
            # Validate input
            params = NetworkIntelligenceInput(**arguments)
            
            # Get database connection
            db = await get_database(params.db_path)
            
            # Calculate date range
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=params.since_days)
            start_timestamp = int((start_date.timestamp() - 978307200) * 1e9)
            
            # Get group chats with participants
            group_query = """
            SELECT DISTINCT
                c.ROWID as chat_id,
                c.display_name as chat_name,
                c.guid as chat_guid
            FROM chat c
            WHERE c.style = 43 OR c.group_name IS NOT NULL
            """
            
            group_results = await db.execute_query(group_query)
            
            # Build network graph
            edges = defaultdict(int)  # (node1, node2) -> weight
            node_activity = defaultdict(int)  # node -> activity count
            
            for group in group_results:
                chat_id = group['chat_id']
                
                # Get participants in this chat
                participant_query = """
                SELECT DISTINCT h.id as handle_id
                FROM message m
                JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
                JOIN handle h ON m.handle_id = h.ROWID
                WHERE cmj.chat_id = ?
                AND m.date >= ?
                AND h.id IS NOT NULL
                """
                
                participant_results = await db.execute_query(
                    participant_query, 
                    (chat_id, start_timestamp)
                )
                
                participants = [p['handle_id'] for p in participant_results]
                
                # Create edges between all participants
                for i, p1 in enumerate(participants):
                    node_activity[p1] += 1
                    for p2 in participants[i+1:]:
                        edge = tuple(sorted([p1, p2]))
                        edges[edge] += 1
            
            # Convert to node/edge lists
            nodes = []
            for handle_id, activity in node_activity.items():
                node_id = hash_contact_id(handle_id) if config.privacy.hash_identifiers else handle_id
                nodes.append({
                    "id": node_id,
                    "label": None,  # Redacted by default
                    "degree": activity
                })
            
            edge_list = []
            for (p1, p2), weight in edges.items():
                source = hash_contact_id(p1) if config.privacy.hash_identifiers else p1
                target = hash_contact_id(p2) if config.privacy.hash_identifiers else p2
                edge_list.append({
                    "source": source,
                    "target": target,
                    "weight": weight
                })
            
            # Simple community detection (connected components)
            # In production, would use more sophisticated algorithms
            communities = []
            if nodes:
                # Create adjacency list
                adj = defaultdict(set)
                for edge in edge_list:
                    adj[edge['source']].add(edge['target'])
                    adj[edge['target']].add(edge['source'])
                
                # Find connected components
                visited = set()
                community_id = 0
                
                for node in nodes:
                    node_id = node['id']
                    if node_id not in visited:
                        # BFS to find component
                        component = []
                        queue = [node_id]
                        while queue:
                            current = queue.pop(0)
                            if current not in visited:
                                visited.add(current)
                                component.append(current)
                                queue.extend(adj[current] - visited)
                        
                        if len(component) > 1:
                            communities.append({
                                "community_id": community_id,
                                "members": component
                            })
                            community_id += 1
            
            # Identify key connectors (simplified - based on degree)
            key_connectors = sorted(
                nodes, 
                key=lambda x: x['degree'], 
                reverse=True
            )[:5]
            
            key_connector_list = [
                {
                    "contact_id": kc['id'],
                    "score": kc['degree'] / max(len(nodes), 1)
                }
                for kc in key_connectors
            ]
            
            # Build response
            output = NetworkIntelligenceOutput(
                nodes=nodes[:50],  # Limit for performance
                edges=edge_list[:100],  # Limit for performance
                communities=communities[:10],  # Limit for privacy
                key_connectors=key_connector_list
            )
            
            return output.model_dump()
            
        except Exception as e:
            logger.error(f"Network intelligence failed: {e}")
            return {
                "error": str(e),
                "error_type": "analysis_failed"
            }