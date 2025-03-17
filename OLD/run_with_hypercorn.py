import os
import sys
import logging
import asyncio
import hypercorn.asyncio
from hypercorn.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the server using hypercorn."""
    try:
        # Set the database path
        db_path = os.path.expanduser("~/Library/Messages/chat.db")
        os.environ["DB_PATH"] = db_path
        
        # Log startup information
        logger.info("Starting iMessage Advanced Insights Server")
        logger.info(f"Using database at {db_path}")
        
        # Import the app
        from src.quart_app import app
        
        # Configure hypercorn
        config = Config()
        config.bind = ["0.0.0.0:49152"]
        config.use_reloader = True
        
        # Run the server
        logger.info(f"Server will run on 0.0.0.0:49152")
        asyncio.run(hypercorn.asyncio.serve(app, config))
        
    except KeyboardInterrupt:
        logger.info("Server shutting down")
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 