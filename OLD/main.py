#!/usr/bin/env python3
"""
iMessage Advanced Insights MCP Server
Main entry point for running the server
"""

import logging
import argparse
import os
import sys

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('imessage_insights.log')
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='iMessage Advanced Insights MCP Server')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run server in debug mode')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                        default='INFO', help='Set the logging level')
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    args = parse_args()
    
    # Set environment variables
    os.environ['PORT'] = str(args.port)
    
    # Set log level
    log_level = getattr(logging, args.log_level)
    logging.getLogger().setLevel(log_level)
    
    # Import after configuring environment
    from src.app import run_server
    
    try:
        logger.info(f"Starting iMessage Advanced Insights MCP Server on port {args.port}")
        run_server()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 