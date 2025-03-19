#!/usr/bin/env python3
"""
Generate Claude Desktop configuration JSON for iMessage Advanced Insights server.
This script creates the necessary configuration to integrate the iMessage Advanced
Insights server with Claude Desktop.
"""

import json
import os
import argparse
import subprocess

def create_parser():
    """Create and return command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate Claude Desktop configuration for iMessage Advanced Insights",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--port', type=int, default=5001,
                        help='Port where the server will run')
    parser.add_argument('--output', type=str, default='claude_desktop_config.json',
                        help='Output file path for the configuration')
    parser.add_argument('--server-path', type=str, 
                        default=os.path.abspath('main_async.py'),
                        help='Absolute path to the main_async.py file')
    
    return parser

def main():
    """Main entry point for the configuration generator."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Get the absolute path to the server script
    server_path = args.server_path
    
    # Create configuration object in the correct format
    config = {
        "mcpServers": {
            "iMessage Advanced Insights": {
                "command": "python3",
                "args": [
                    server_path,
                    "--port",
                    str(args.port)
                ]
            }
        }
    }
    
    # Write configuration to file
    with open(args.output, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Claude Desktop configuration generated at: {args.output}")
    print(f"The configuration is set up for the server at port {args.port}")
    print(f"Server path: {server_path}")
    print("\nTo use this configuration:")
    print("1. Import the generated JSON into Claude Desktop settings")
    print("2. Enable the 'iMessage Advanced Insights' MCP server in Claude Desktop")
    print("3. You can now use the iMessage analysis tools in Claude conversations")

if __name__ == "__main__":
    main() 