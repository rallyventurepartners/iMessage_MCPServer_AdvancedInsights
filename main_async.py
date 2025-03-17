#!/usr/bin/env python3
"""
iMessage Advanced Insights - MCP Server with Async Support
A powerful server for extracting insights from iMessage conversations.
"""

import os
import sys
import argparse
import logging
import uvicorn
import asyncio
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

def create_parser():
    """Create and return command line argument parser."""
    parser = argparse.ArgumentParser(
        description="iMessage Advanced Insights Server with Async Support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Server configuration
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host address to run the server on')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run the server on')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode with more verbose logging')
    
    # Database configuration
    parser.add_argument('--db-path', type=str, 
                        default=os.path.expanduser('~/Library/Messages/chat.db'),
                        help='Path to the iMessage chat.db file')
    
    # Analysis configuration
    parser.add_argument('--no-sentiment', action='store_true',
                        help='Disable sentiment analysis to improve performance')
    parser.add_argument('--no-network', action='store_true',
                        help='Disable network analysis to improve performance')
    
    return parser

def main():
    """Main entry point for the iMessage Advanced Insights Server."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug mode enabled - verbose logging activated")
    
    # Validate database path
    if not os.path.exists(args.db_path):
        logger.error(f"Database file not found at {args.db_path}")
        logger.error("Please check the path or ensure Messages is properly set up")
        sys.exit(1)
    
    # Print startup message
    logger.info(f"Starting iMessage Advanced Insights Server with Async Support")
    logger.info(f"Server will run on {args.host}:{args.port}")
    logger.info(f"Using database at {args.db_path}")
    
    # Import and initialize the database with the specified path
    from src.database.async_messages_db import AsyncMessagesDB
    db = AsyncMessagesDB(args.db_path)
    
    # Optionally disable certain analysis features
    config = {
        'sentiment_analysis': not args.no_sentiment,
        'network_analysis': not args.no_network,
    }
    
    logger.info(f"Configuration: {config}")
    
    # Import the app
    from src.app_async import app
    
    # Set the database path in the app
    app.db_path = args.db_path
    app.config = config
    
    # Run the server using hypercorn instead of uvicorn
    import hypercorn.asyncio
    from hypercorn.config import Config
    
    config = Config()
    config.bind = [f"{args.host}:{args.port}"]
    config.loglevel = "debug" if args.debug else "info"
    
    # Start the server
    import asyncio
    asyncio.run(hypercorn.asyncio.serve(app, config))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Server shutting down")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        logger.exception(e)
        sys.exit(1) 