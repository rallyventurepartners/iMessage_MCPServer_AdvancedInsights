#!/usr/bin/env python3
"""
iMessage Advanced Insights - MCP Server

Main entry point for the iMessage Advanced Insights MCP Server.
This script initializes and starts the server, providing a CLI interface
for configuration and management.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from src.exceptions import ConfigurationError, DatabaseError
from src.server.server import MCPServer
from src.utils.logging_config import configure_logging

# Configure basic logging until full configuration is loaded
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="iMessage Advanced Insights MCP Server"
    )
    
    # General options
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        help="Port to run the server on (default: read from config or 5000)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    
    # Database options
    parser.add_argument(
        "--db-path",
        type=str,
        help="Path to iMessage database",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Run in minimal mode with reduced features",
    )
    parser.add_argument(
        "--use-shards",
        action="store_true",
        help="Use time-based database sharding for large databases",
    )
    parser.add_argument(
        "--shards-dir",
        type=str,
        help="Directory containing database shards",
    )
    parser.add_argument(
        "--auto-shard",
        action="store_true",
        help="Automatically detect and use shards if database is large",
    )
    
    # Memory options
    parser.add_argument(
        "--disable-memory-monitor",
        action="store_true",
        help="Disable memory monitoring",
    )
    parser.add_argument(
        "--memory-limit",
        type=int,
        help="Memory limit in MB",
    )
    
    return parser.parse_args()


def set_environment_variables(args: argparse.Namespace) -> None:
    """
    Set environment variables based on command-line arguments.
    
    Args:
        args: Parsed command-line arguments
    """
    # Server settings
    if args.port:
        os.environ["MCP_SERVER_PORT"] = str(args.port)
    
    if args.log_level:
        os.environ["MCP_SERVER_LOG_LEVEL"] = args.log_level
    
    # Database settings
    if args.db_path:
        os.environ["MCP_DATABASE_PATH"] = args.db_path
    
    if args.minimal:
        os.environ["MCP_DATABASE_MINIMAL_MODE"] = "true"
    
    if args.use_shards:
        os.environ["MCP_DATABASE_USE_SHARDING"] = "true"
    
    if args.shards_dir:
        os.environ["MCP_DATABASE_SHARDS_DIR"] = args.shards_dir
    
    if args.auto_shard:
        os.environ["MCP_DATABASE_AUTO_DETECT_SHARDING"] = "true"
    
    # Memory settings
    if args.disable_memory_monitor:
        os.environ["MCP_MEMORY_ENABLE_MONITORING"] = "false"
    
    if args.memory_limit:
        os.environ["MCP_MEMORY_LIMIT_MB"] = str(args.memory_limit)


async def main() -> int:
    """
    Main entry point.
    
    Returns:
        Exit code
    """
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Set environment variables based on arguments
        set_environment_variables(args)
        
        # Initialize the server
        server = MCPServer(config_file=args.config)
        
        # Initialize server components
        success = await server.initialize()
        if not success:
            logger.error("Failed to initialize server")
            return 1
        
        # Start the server
        success = await server.start()
        if not success:
            logger.error("Failed to start server")
            return 1
        
        # Server is running, wait for interruption
        # (the server.start() method is actually blocking,
        # but we have a return value for testing purposes)
        return 0
    
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except DatabaseError as e:
        logger.error(f"Database error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    # Run the main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
