#!/usr/bin/env python3
"""
Generate Claude Desktop configuration JSON for iMessage Advanced Insights server.
This script creates the necessary configuration to integrate the iMessage Advanced
Insights server with Claude Desktop.
"""

import json
import os
import argparse

def create_parser():
    """Create and return command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate Claude Desktop configuration for iMessage Advanced Insights",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--host', type=str, default='localhost',
                        help='Host address where the server runs')
    parser.add_argument('--port', type=int, default=5001,
                        help='Port where the server runs')
    parser.add_argument('--output', type=str, default='claude_desktop_config.json',
                        help='Output file path for the configuration')
    
    return parser

def main():
    """Main entry point for the configuration generator."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Base URL for the API
    base_url = f"http://{args.host}:{args.port}"
    
    # Create configuration object
    config = {
        "tools": [
            {
                "name": "iMessage_contacts",
                "description": "Get a list of all contacts from iMessage. Use when the user wants to know who they've messaged with.",
                "actionSpec": {
                    "type": "api",
                    "api": {
                        "url": f"{base_url}/api/contacts",
                        "method": "GET"
                    }
                }
            },
            {
                "name": "iMessage_group_chats",
                "description": "Get a list of all group chats from iMessage. Use when the user wants to know their group chats.",
                "actionSpec": {
                    "type": "api",
                    "api": {
                        "url": f"{base_url}/api/group_chats",
                        "method": "GET"
                    }
                }
            },
            {
                "name": "iMessage_analyze_contact",
                "description": "Analyze messages with a specific contact. Use when the user wants insights about conversations with a particular person.",
                "actionSpec": {
                    "type": "api",
                    "api": {
                        "url": f"{base_url}/api/analyze_contact",
                        "method": "POST",
                        "bodyFormat": "json"
                    }
                },
                "inputFields": [
                    {
                        "name": "phone_number",
                        "description": "Phone number of the contact to analyze",
                        "type": "string",
                        "required": True
                    },
                    {
                        "name": "start_date",
                        "description": "Starting date for analysis (YYYY-MM-DD format)",
                        "type": "string",
                        "required": False
                    },
                    {
                        "name": "end_date",
                        "description": "Ending date for analysis (YYYY-MM-DD format)",
                        "type": "string",
                        "required": False
                    },
                    {
                        "name": "page",
                        "description": "Page number for pagination",
                        "type": "number",
                        "required": False,
                        "default": 1
                    },
                    {
                        "name": "page_size",
                        "description": "Number of items per page",
                        "type": "number",
                        "required": False,
                        "default": 100
                    }
                ]
            },
            {
                "name": "iMessage_analyze_group_chat",
                "description": "Analyze messages in a group chat. Use when the user wants insights about a group conversation.",
                "actionSpec": {
                    "type": "api",
                    "api": {
                        "url": f"{base_url}/api/analyze_group_chat",
                        "method": "POST",
                        "bodyFormat": "json"
                    }
                },
                "inputFields": [
                    {
                        "name": "chat_id",
                        "description": "ID of the group chat to analyze",
                        "type": "string",
                        "required": True
                    },
                    {
                        "name": "start_date",
                        "description": "Starting date for analysis (YYYY-MM-DD format)",
                        "type": "string",
                        "required": False
                    },
                    {
                        "name": "end_date",
                        "description": "Ending date for analysis (YYYY-MM-DD format)",
                        "type": "string",
                        "required": False
                    }
                ]
            },
            {
                "name": "iMessage_analyze_network",
                "description": "Analyze the contact network based on group chats. Use when the user wants to visualize their social network from messages.",
                "actionSpec": {
                    "type": "api",
                    "api": {
                        "url": f"{base_url}/api/analyze_network",
                        "method": "POST",
                        "bodyFormat": "json"
                    }
                },
                "inputFields": [
                    {
                        "name": "start_date",
                        "description": "Starting date for analysis (YYYY-MM-DD format)",
                        "type": "string",
                        "required": False
                    },
                    {
                        "name": "end_date",
                        "description": "Ending date for analysis (YYYY-MM-DD format)",
                        "type": "string",
                        "required": False
                    },
                    {
                        "name": "min_shared_chats",
                        "description": "Minimum number of shared chats for a connection",
                        "type": "number",
                        "required": False,
                        "default": 1
                    }
                ]
            },
            {
                "name": "iMessage_visualize_network",
                "description": "Generate visualization data for the contact network. Use when the user wants a visual representation of their messaging relationships.",
                "actionSpec": {
                    "type": "api",
                    "api": {
                        "url": f"{base_url}/api/visualize_network",
                        "method": "POST",
                        "bodyFormat": "json"
                    }
                },
                "inputFields": [
                    {
                        "name": "start_date",
                        "description": "Starting date for visualization (YYYY-MM-DD format)",
                        "type": "string",
                        "required": False
                    },
                    {
                        "name": "end_date",
                        "description": "Ending date for visualization (YYYY-MM-DD format)",
                        "type": "string",
                        "required": False
                    },
                    {
                        "name": "min_shared_chats",
                        "description": "Minimum number of shared chats for a connection",
                        "type": "number",
                        "required": False,
                        "default": 1
                    },
                    {
                        "name": "layout",
                        "description": "Layout algorithm to use (spring, circular, etc.)",
                        "type": "string",
                        "required": False,
                        "default": "spring"
                    }
                ]
            },
            {
                "name": "iMessage_analyze_sentiment",
                "description": "Analyze sentiment in conversations with a contact or group. Use when the user wants to understand emotional patterns in their messages.",
                "actionSpec": {
                    "type": "api",
                    "api": {
                        "url": f"{base_url}/api/analyze_sentiment",
                        "method": "POST",
                        "bodyFormat": "json"
                    }
                },
                "inputFields": [
                    {
                        "name": "phone_number",
                        "description": "Phone number of the contact for sentiment analysis",
                        "type": "string",
                        "required": False
                    },
                    {
                        "name": "chat_id",
                        "description": "ID of the group chat for sentiment analysis",
                        "type": "string",
                        "required": False
                    },
                    {
                        "name": "start_date",
                        "description": "Starting date for analysis (YYYY-MM-DD format)",
                        "type": "string",
                        "required": False
                    },
                    {
                        "name": "end_date",
                        "description": "Ending date for analysis (YYYY-MM-DD format)",
                        "type": "string",
                        "required": False
                    },
                    {
                        "name": "include_individual_messages",
                        "description": "Whether to include sentiment for individual messages",
                        "type": "boolean",
                        "required": False,
                        "default": False
                    }
                ]
            },
            {
                "name": "iMessage_query",
                "description": "Process a natural language query about iMessage data. Use when the user has a specific question about their messages.",
                "actionSpec": {
                    "type": "api",
                    "api": {
                        "url": f"{base_url}/api/query",
                        "method": "POST",
                        "bodyFormat": "json"
                    }
                },
                "inputFields": [
                    {
                        "name": "query",
                        "description": "Natural language query about iMessage data",
                        "type": "string",
                        "required": True
                    }
                ]
            }
        ]
    }
    
    # Write configuration to file
    with open(args.output, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Claude Desktop configuration generated at: {args.output}")
    print(f"The configuration is set up for the server at {base_url}")
    print("\nTo use this configuration:")
    print("1. Make sure the iMessage Advanced Insights server is running")
    print(f"   python3 main_async.py --port {args.port}")
    print("2. Import the generated JSON into Claude Desktop")
    print("3. You can now use the iMessage analysis tools in Claude conversations")

if __name__ == "__main__":
    main() 