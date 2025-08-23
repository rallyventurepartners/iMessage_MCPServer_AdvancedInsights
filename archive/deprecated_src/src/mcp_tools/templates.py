#!/usr/bin/env python3
"""
Templates MCP Tool Module

This module provides MCP tools for working with analysis templates.
"""

import logging
from typing import Any, Dict

# Configure logging
logger = logging.getLogger(__name__)


def get_template(template_name: str) -> Dict[str, Any]:
    """
    Get a template for iMessage analysis.

    Args:
        template_name: Name of the template to retrieve

    Returns:
        Dictionary containing the template content
    """
    templates = {
        "conversation_analysis": {
            "title": "Conversation Analysis Template",
            "description": "Template for analyzing conversations with a specific contact",
            "content": """
            # Conversation Analysis
            
            This template provides a structured approach to analyzing your iMessage conversations.
            
            ## Key Metrics to Consider
            
            1. **Message Frequency**: How often do you exchange messages?
            2. **Response Time**: How quickly do you respond to each other?
            3. **Conversation Length**: How long do your conversations typically last?
            4. **Conversation Initiator**: Who typically starts conversations?
            5. **Conversation Topics**: What do you typically talk about?
            6. **Sentiment Analysis**: What's the emotional tone of your conversations?
            
            ## Sample Analysis Structure
            
            Here's a suggested structure for your analysis:
            
            1. **Overview**: Summarize the overall communication patterns
            2. **Frequency Patterns**: Analyze when and how often you communicate
            3. **Content Analysis**: Examine what you talk about
            4. **Sentiment Analysis**: Explore the emotional tone
            5. **Relationship Dynamics**: Analyze the balance of communication
            6. **Trends Over Time**: How has the communication evolved?
            7. **Key Insights**: What are the most interesting discoveries?
            """,
        },
        "group_chat_analysis": {
            "title": "Group Chat Analysis Template",
            "description": "Template for analyzing group chat dynamics",
            "content": """
            # Group Chat Analysis
            
            This template provides a structured approach to analyzing your iMessage group chats.
            
            ## Key Metrics to Consider
            
            1. **Participation Rates**: Who are the most active members?
            2. **Response Patterns**: How quickly do members respond to each other?
            3. **Topic Analysis**: What are the most common discussion topics?
            4. **Sentiment Analysis**: What's the emotional tone of the group?
            5. **Time Patterns**: When is the group most active?
            6. **Sub-group Formation**: Are there smaller conversation clusters?
            
            ## Sample Analysis Structure
            
            Here's a suggested structure for your analysis:
            
            1. **Group Overview**: Summarize the group composition and history
            2. **Participation Analysis**: Examine who participates and how much
            3. **Content Analysis**: Analyze common topics and conversation themes
            4. **Engagement Patterns**: Identify peak activity times and conversation flow
            5. **Relationship Dynamics**: Examine connections between participants
            6. **Trends Over Time**: How has the group dynamic evolved?
            7. **Key Insights**: What are the most interesting discoveries?
            """,
        },
        "network_analysis": {
            "title": "Network Analysis Template",
            "description": "Template for analyzing your social network from messages",
            "content": """
            # iMessage Network Analysis
            
            This template provides a structured approach to analyzing your social network through iMessage data.
            
            ## Key Metrics to Consider
            
            1. **Connection Strength**: How frequently do you message each contact?
            2. **Central Connections**: Who are your most important contacts?
            3. **Connection Clusters**: Are there distinct social groups?
            4. **Communication Flow**: How does information flow through your network?
            5. **Temporal Patterns**: How has your network evolved over time?
            
            ## Sample Analysis Structure
            
            Here's a suggested structure for your analysis:
            
            1. **Network Overview**: Summarize your overall communication network
            2. **Key Relationships**: Identify your most significant connections
            3. **Community Detection**: Identify distinct social groups
            4. **Network Dynamics**: Analyze how information flows
            5. **Temporal Analysis**: Examine how your network has changed
            6. **Key Insights**: What are the most interesting discoveries?
            """,
        },
    }

    # Return the requested template or error
    if template_name in templates:
        return {"success": True, "template": templates[template_name]}
    else:
        return {
            "success": False,
            "error": {
                "type": "not_found",
                "message": f"Template not found: {template_name}",
            },
            "available_templates": list(templates.keys()),
        }
