import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the server."""
    try:
        # Set the database path
        db_path = os.path.expanduser("~/Library/Messages/chat.db")
        os.environ["DB_PATH"] = db_path
        
        # Log startup information
        logger.info("Starting iMessage Advanced Insights Server")
        logger.info(f"Using database at {db_path}")
        
        # Import and run the app
        from src.flask_app import app
        app.run(host='0.0.0.0', port=23459, debug=True)  # Use a different port
        
    except KeyboardInterrupt:
        logger.info("Server shutting down")
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 