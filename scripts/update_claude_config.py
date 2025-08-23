#!/usr/bin/env python3
"""
Generate Claude Desktop configuration for iMessage Advanced Insights.

This script generates a configuration file for Claude Desktop that enables
integration with the iMessage Advanced Insights MCP server.
"""

import argparse
import json
from pathlib import Path


def generate_config(
    server_path=None,
    output_path="claude_desktop_config.json",
    port=5000,
    use_sharding=None,
    db_path=None,
):
    """Generate Claude Desktop configuration for the iMessage Advanced Insights server.

    Args:
        server_path: Path to the server script (optional, detected automatically if not provided)
        output_path: Path to write the configuration file
        port: Port number for the server to listen on
        use_sharding: Set to True to enable database sharding, False to disable, None for auto-detection
        db_path: Custom path to the iMessage database (optional)
    """
    # Use default path if not provided
    if not server_path:
        # Get the directory this script is in
        script_dir = Path(__file__).parent.absolute()
        main_script = script_dir / "mcp_server_modular.py"
        if main_script.exists():
            server_path = main_script
        else:
            server_path = script_dir / "src" / "app_async.py"

    # Ensure the server path is absolute
    server_path = Path(server_path).absolute()

    # Generate MCP-compliant configuration
    config = {
        "name": "iMessage Advanced Insights",
        "description": "Analyze and visualize iMessage data with advanced insights and sentiment analysis",
        "version": "2.2.0",
        "transport": {
            "type": "http",
            "host": "127.0.0.1",
            "port": port,
            "path": "/mcp",
        },
        "resources": [
            {
                "name": "contacts",
                "description": "Get contact information from your iMessage database",
                "path": "/contacts",
                "method": "GET",
            },
            {
                "name": "group_chats",
                "description": "Get group chat information",
                "path": "/group_chats",
                "method": "GET",
            },
            {
                "name": "analyze_group_chat",
                "description": "Analyze a group chat with sentiment and message patterns",
                "path": "/analyze_group_chat",
                "method": "POST",
            },
            {
                "name": "analyze_contact",
                "description": "Analyze conversations with a specific contact",
                "path": "/analyze_contact",
                "method": "POST",
            },
            {
                "name": "analyze_network",
                "description": "Analyze your social network based on group chats",
                "path": "/analyze_network",
                "method": "POST",
            },
            {
                "name": "visualize_network",
                "description": "Generate a visualization of your messaging network",
                "path": "/visualize_network",
                "method": "POST",
            },
            {
                "name": "analyze_sentiment",
                "description": "Analyze sentiment patterns in conversations",
                "path": "/analyze_sentiment",
                "method": "POST",
            },
            {
                "name": "query",
                "description": "Ask natural language questions about your messages",
                "path": "/query",
                "method": "POST",
            },
        ],
        "settings": {
            "initialization_timeout": 30,
            "request_timeout": 60,
            "large_request_timeout": 180,
        },
        # This section is for Claude Desktop to know how to start the server
        "startup": {
            "command": "python3",
            "args": [str(server_path)],
            "env": {
                "PORT": str(port),
                "USE_SHARDING": (
                    "auto"
                    if use_sharding is None
                    else "true" if use_sharding else "false"
                ),
            },
        },
    }

    # Add database path if provided
    if db_path:
        config["startup"]["env"]["DB_PATH"] = str(Path(db_path).absolute())

    # Update the path based on the server script
    if "mcp_server_modular.py" in str(server_path):
        config["startup"]["working_directory"] = str(
            Path(server_path).parent.absolute()
        )

    # Write the configuration to file
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Generated Claude Desktop configuration at {output_path}")
    print(f"Server path: {server_path}")
    print(f"Port: {port}")

    # Display database sharding status
    sharding_status = use_sharding if use_sharding is not None else "auto-detect"
    print(f"Database sharding: {sharding_status}")

    # Display custom database path if provided
    if db_path:
        print(f"Database path: {db_path}")

    print("\nTo use this configuration:")
    print("1. Open Claude Desktop")
    print("2. Go to Settings > Model Context Providers")
    print("3. Click 'Import MCP Server Configuration'")
    print("4. Select the generated file")
    print("5. Ensure 'iMessage Advanced Insights' is enabled in the list")


def main():
    """Parse command line arguments and generate configuration."""
    parser = argparse.ArgumentParser(
        description="Generate Claude Desktop configuration for iMessage Advanced Insights"
    )
    parser.add_argument("--server-path", help="Path to the server script")
    parser.add_argument(
        "--output", default="claude_desktop_config.json", help="Output file path"
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port number for the server to listen on"
    )
    parser.add_argument(
        "--use-sharding",
        choices=["auto", "true", "false"],
        default="auto",
        help="Enable or disable database sharding, or auto-detect based on size",
    )
    parser.add_argument("--db-path", help="Path to the iMessage database")
    parser.add_argument(
        "--claude-dir",
        help="Path to Claude Desktop configuration directory (default: ~/.claude)",
    )
    args = parser.parse_args()

    # Convert use_sharding to appropriate value
    use_sharding = None
    if args.use_sharding == "true":
        use_sharding = True
    elif args.use_sharding == "false":
        use_sharding = False

    # Generate the configuration
    generate_config(
        server_path=args.server_path,
        output_path=args.output,
        port=args.port,
        use_sharding=use_sharding,
        db_path=args.db_path,
    )

    # If claude-dir is provided, copy the config file there directly
    if args.claude_dir:
        import shutil

        claude_config_dir = Path(args.claude_dir).expanduser()
        if not claude_config_dir.exists():
            claude_config_dir.mkdir(parents=True, exist_ok=True)

        target_path = claude_config_dir / "mcp_config" / "imessage_insights.json"
        target_path.parent.mkdir(exist_ok=True)

        # Copy the generated config
        shutil.copy2(args.output, target_path)
        print(f"\nConfiguration directly installed to Claude Desktop at: {target_path}")
        print("Restart Claude Desktop for changes to take effect.")


if __name__ == "__main__":
    main()
