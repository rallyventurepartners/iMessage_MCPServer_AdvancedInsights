#!/usr/bin/env python3
"""
iMessage MCP Server Installer and Optimizer

This script simplifies the installation and optimization of the iMessage MCP server.
It handles:
1. Installing required dependencies
2. Detecting database size and applying appropriate optimization strategy
3. Setting up Claude Desktop integration
4. Configuring the server for optimal performance

Usage:
    python install_mcp.py  # Interactive installation
    python install_mcp.py --non-interactive  # Automatic installation with defaults
    python install_mcp.py --optimize-db  # Only database optimization
"""

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default paths
HOME = os.path.expanduser("~")
DEFAULT_DB_PATH = os.path.join(HOME, "Library/Messages/chat.db")
DEFAULT_OUTPUT_DIR = os.path.join(HOME, ".imessage_insights")
CLAUDE_CONFIG_PATH = os.path.join(HOME, ".claude", "config.json")


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        logger.error(f"Python 3.9+ required, found {version.major}.{version.minor}")
        return False
    return True


def check_platform():
    """Check if the platform is supported."""
    system = platform.system()
    if system != "Darwin":
        logger.error(f"macOS required, found {system}")
        return False
    return True


def install_dependencies():
    """Install required Python dependencies."""
    try:
        logger.info("Installing required Python dependencies...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing dependencies: {e}")
        logger.error(f"Output: {e.stdout}")
        logger.error(f"Error: {e.stderr}")
        return False


def analyze_database(db_path):
    """
    Analyze the database and determine the best optimization strategy.

    Args:
        db_path: Path to the database

    Returns:
        dict: Analysis results with recommendations
    """
    if not os.path.exists(db_path):
        logger.error(f"Database file not found: {db_path}")
        return {"error": "Database not found"}

    # Get basic file info
    size_bytes = os.path.getsize(db_path)
    size_gb = size_bytes / (1024 * 1024 * 1024)

    logger.info(f"Database size: {size_gb:.2f} GB ({size_bytes:,} bytes)")

    # Connect to the database to get more information
    try:
        import sqlite3

        conn = sqlite3.connect(db_path)

        # Check message count
        cursor = conn.execute("SELECT COUNT(*) FROM message")
        message_count = cursor.fetchone()[0]

        # Check chat count
        cursor = conn.execute("SELECT COUNT(*) FROM chat")
        chat_count = cursor.fetchone()[0]

        # Check existing indexes
        cursor = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND "
            "name NOT LIKE 'sqlite_%'"
        )
        index_count = cursor.fetchone()[0]

        # Check for FTS
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='message_fts'"
        )
        has_fts = cursor.fetchone() is not None

        # Close connection
        conn.close()

        # Determine optimization strategy
        needs_indexing = index_count < 5
        needs_fts = not has_fts and message_count > 10000
        needs_sharding = size_gb >= 10

        # Return analysis results
        return {
            "size_bytes": size_bytes,
            "size_gb": size_gb,
            "message_count": message_count,
            "chat_count": chat_count,
            "index_count": index_count,
            "has_fts": has_fts,
            "needs_indexing": needs_indexing,
            "needs_fts": needs_fts,
            "needs_sharding": needs_sharding,
            "recommendation": (
                "sharding"
                if needs_sharding
                else "indexing" if needs_indexing else "none"
            ),
        }

    except Exception as e:
        logger.error(f"Error analyzing database: {e}")
        return {"error": str(e)}


def optimize_database(db_path, analysis, interactive=True):
    """
    Optimize the database based on analysis results.

    Args:
        db_path: Path to the database
        analysis: Analysis results from analyze_database()
        interactive: Whether to prompt for confirmation

    Returns:
        dict: Optimization results
    """
    if "error" in analysis:
        logger.error(f"Cannot optimize database: {analysis['error']}")
        return {"success": False, "error": analysis["error"]}

    output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Determine optimization strategy
    if analysis["needs_sharding"]:
        if interactive:
            logger.info(
                "\nYour database is very large (>10 GB). Sharding is recommended."
            )
            logger.info(
                "Sharding will split the database into smaller time-based chunks for better performance."
            )
            proceed = (
                input("Do you want to create database shards? (y/n): ").strip().lower()
            )
            if proceed != "y":
                logger.info("Sharding skipped")
                return {"success": False, "reason": "user_skip"}

        # Create shards
        logger.info("Creating database shards...")
        script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "shard_large_database.py"
        )

        # Determine shard size based on database size
        if analysis["size_gb"] > 50:
            shard_size = 3  # 3 months for extremely large databases
        elif analysis["size_gb"] > 20:
            shard_size = 4  # 4 months for very large databases
        else:
            shard_size = 6  # 6 months for large databases

        shards_dir = os.path.join(output_dir, "shards")

        # Run sharding script
        try:
            cmd = [
                sys.executable,
                script_path,
                "--source-db",
                db_path,
                "--shards-dir",
                shards_dir,
                "--shard-size-months",
                str(shard_size),
            ]

            result = subprocess.run(cmd, check=True, capture_output=True, text=True)

            logger.info("Sharding completed successfully")
            return {
                "success": True,
                "strategy": "sharding",
                "shards_dir": shards_dir,
                "shard_size_months": shard_size,
            }

        except subprocess.CalledProcessError as e:
            logger.error(f"Error creating shards: {e}")
            logger.error(f"Error output: {e.stderr}")
            return {"success": False, "error": f"Sharding failed: {e}"}

    elif analysis["needs_indexing"] or analysis["needs_fts"]:
        if interactive:
            logger.info(
                "\nDatabase optimization is recommended to improve performance."
            )
            logger.info(
                "This will create an indexed copy of your database with better query performance."
            )
            proceed = (
                input("Do you want to create an optimized database copy? (y/n): ")
                .strip()
                .lower()
            )
            if proceed != "y":
                logger.info("Indexing skipped")
                return {"success": False, "reason": "user_skip"}

        # Create indexed copy
        logger.info("Creating optimized database copy...")
        script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "index_imessage_db.py"
        )

        try:
            cmd = [
                sys.executable,
                script_path,
                "--source-db",
                db_path,
                "--output-dir",
                output_dir,
                "--read-only",
            ]

            if not analysis["needs_fts"]:
                cmd.append("--no-fts")

            result = subprocess.run(cmd, check=True, capture_output=True, text=True)

            logger.info("Indexing completed successfully")
            return {
                "success": True,
                "strategy": "indexing",
                "optimized_db": os.path.join(output_dir, "indexed_chat.db"),
            }

        except subprocess.CalledProcessError as e:
            logger.error(f"Error creating indexed copy: {e}")
            logger.error(f"Error output: {e.stderr}")
            return {"success": False, "error": f"Indexing failed: {e}"}
    else:
        logger.info("Database is already optimized")
        return {"success": True, "strategy": "none"}


def setup_claude_desktop_integration(optimization_result):
    """Set up Claude Desktop integration."""
    logger.info("\nSetting up Claude Desktop integration...")

    # Create Claude config directory if it doesn't exist
    claude_config_dir = os.path.dirname(CLAUDE_CONFIG_PATH)
    os.makedirs(claude_config_dir, exist_ok=True)

    # Read existing config if it exists
    config = {}
    if os.path.exists(CLAUDE_CONFIG_PATH):
        try:
            with open(CLAUDE_CONFIG_PATH) as f:
                config = json.load(f)
        except Exception as e:
            logger.warning(f"Error reading Claude config: {e}")

    # Update MCP settings
    if "mcp" not in config:
        config["mcp"] = {}

    # Add our server to the MCP list
    config["mcp"]["servers"] = config.get("mcp", {}).get("servers", [])

    # Check if our server is already in the list
    server_exists = False
    for server in config["mcp"]["servers"]:
        if (
            server.get("name") == "iMessage Insights"
            or server.get("url") == "http://localhost:5000/mcp"
        ):
            server_exists = True
            # Update server config
            server["name"] = "iMessage Insights"
            server["url"] = "http://localhost:5000/mcp"
            server["method"] = "post"
            server["enabled"] = True
            break

    # Add our server if it doesn't exist
    if not server_exists:
        config["mcp"]["servers"].append(
            {
                "name": "iMessage Insights",
                "url": "http://localhost:5000/mcp",
                "method": "post",
                "enabled": True,
            }
        )

    # Write updated config
    try:
        with open(CLAUDE_CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
        logger.info("Claude Desktop integration configured successfully")
        return True
    except Exception as e:
        logger.error(f"Error writing Claude config: {e}")
        return False


def create_startup_script(optimization_result):
    """Create a startup script to launch the server with optimized settings."""
    script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "start_server.sh"
    )

    # Determine command based on optimization strategy
    if optimization_result.get("strategy") == "sharding":
        cmd = (
            f"#!/bin/bash\n"
            f"# Auto-generated startup script for iMessage MCP Server\n"
            f"# Created on {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f'cd "{os.path.dirname(os.path.abspath(__file__))}"\n'
            f"USE_SHARDING=true \\\n"
            f"SHARDS_DIR=\"{optimization_result.get('shards_dir')}\" \\\n"
            f"python mcp_server_modular.py\n"
        )
    elif optimization_result.get("strategy") == "indexing":
        cmd = (
            f"#!/bin/bash\n"
            f"# Auto-generated startup script for iMessage MCP Server\n"
            f"# Created on {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f'cd "{os.path.dirname(os.path.abspath(__file__))}"\n'
            f"DB_PATH=\"{optimization_result.get('optimized_db')}\" \\\n"
            f"python mcp_server_modular.py\n"
        )
    else:
        cmd = (
            f"#!/bin/bash\n"
            f"# Auto-generated startup script for iMessage MCP Server\n"
            f"# Created on {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f'cd "{os.path.dirname(os.path.abspath(__file__))}"\n'
            f"python mcp_server_modular.py\n"
        )

    # Write script
    try:
        with open(script_path, "w") as f:
            f.write(cmd)
        os.chmod(script_path, 0o755)  # Make executable
        logger.info(f"Startup script created at {script_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating startup script: {e}")
        return False


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Install and optimize the iMessage MCP server"
    )
    parser.add_argument(
        "--non-interactive", action="store_true", help="Run in non-interactive mode"
    )
    parser.add_argument(
        "--optimize-db", action="store_true", help="Only perform database optimization"
    )
    parser.add_argument(
        "--source-db",
        help="Path to the source iMessage database",
        default=DEFAULT_DB_PATH,
    )
    parser.add_argument(
        "--skip-dependencies", action="store_true", help="Skip installing dependencies"
    )
    parser.add_argument(
        "--skip-claude", action="store_true", help="Skip Claude Desktop integration"
    )

    args = parser.parse_args()
    interactive = not args.non_interactive
    db_path = os.path.expanduser(args.source_db)

    logger.info("iMessage MCP Server Installer and Optimizer")
    logger.info("==========================================")

    # Check requirements
    if not check_python_version() or not check_platform():
        return 1

    # Only database optimization if requested
    if args.optimize_db:
        logger.info("Performing database optimization only...")
        analysis = analyze_database(db_path)
        optimization_result = optimize_database(db_path, analysis, interactive)

        if optimization_result.get("success"):
            logger.info("Database optimization completed successfully")
            create_startup_script(optimization_result)
        else:
            if optimization_result.get("reason") != "user_skip":
                logger.error("Database optimization failed")
                return 1

        return 0

    # Install dependencies
    if not args.skip_dependencies:
        if not install_dependencies():
            return 1

    # Analyze and optimize database
    logger.info(f"\nAnalyzing iMessage database at {db_path}...")
    analysis = analyze_database(db_path)

    if "error" in analysis:
        if interactive:
            alternate_path = input(
                "\nEnter the path to your iMessage database: "
            ).strip()
            if alternate_path:
                db_path = os.path.expanduser(alternate_path)
                analysis = analyze_database(db_path)
                if "error" in analysis:
                    logger.error("Could not analyze database")
                    return 1
            else:
                logger.error("Database path required")
                return 1
        else:
            logger.error("Could not analyze database")
            return 1

    # Show analysis to user
    logger.info("\nDatabase Analysis:")
    logger.info(f"  Size: {analysis['size_gb']:.2f} GB")
    logger.info(f"  Messages: {analysis['message_count']:,}")
    logger.info(f"  Chats: {analysis['chat_count']:,}")
    logger.info(f"  Indexes: {analysis['index_count']}")
    logger.info(f"  FTS enabled: {analysis['has_fts']}")

    # Recommend optimization
    if analysis["needs_sharding"]:
        logger.info("\nRecommendation: Database Sharding")
        logger.info(
            "  Your database is very large and should be split into smaller time-based chunks"
        )
    elif analysis["needs_indexing"]:
        logger.info("\nRecommendation: Create Indexed Copy")
        logger.info(
            "  Your database would benefit from additional indexes for better performance"
        )
    elif analysis["needs_fts"]:
        logger.info("\nRecommendation: Enable Full-Text Search")
        logger.info("  Add FTS capabilities for faster text searching")
    else:
        logger.info("\nRecommendation: No optimization needed")
        logger.info("  Your database appears to be properly optimized")

    # Optimize database
    optimization_result = optimize_database(db_path, analysis, interactive)

    # Setup Claude Desktop integration
    if not args.skip_claude:
        setup_claude_desktop_integration(optimization_result)

    # Create startup script
    create_startup_script(optimization_result)

    # Final instructions
    logger.info("\n=== Installation Complete ===")
    logger.info("To start the server, run:")
    logger.info("  ./start_server.sh")

    logger.info("\nTo use with Claude Desktop:")
    logger.info("1. Open Claude Desktop")
    logger.info("2. Go to Settings > Model Context Providers")
    logger.info("3. Make sure 'iMessage Insights' is enabled")

    return 0


if __name__ == "__main__":
    sys.exit(main())
