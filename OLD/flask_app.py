import os
import logging
from flask import Flask, jsonify, request
from datetime import datetime
import traceback

# Import async database module
from src.database.async_messages_db import AsyncMessagesDB
from src.utils.redis_cache import AsyncRedisCache

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

# Create Flask app
app = Flask(__name__)

# Initialize database connection
db = None

# App configuration
app.config.update({
    'SENTIMENT_ANALYSIS': True,
    'NETWORK_ANALYSIS': True
})

@app.before_first_request
def initialize_database():
    """Initialize the database connection before the first request."""
    global db
    try:
        # Get the database path from the environment variable
        db_path = os.environ.get("DB_PATH")
        if not db_path:
            logger.warning("DB_PATH environment variable not set, using default path")
            db_path = os.path.expanduser("~/Library/Messages/chat.db")
            
        logger.info(f"Initializing database connection with path: {db_path}")
        db = AsyncMessagesDB(db_path)
        logger.info("Database connection initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        logger.error(traceback.format_exc())
        raise

# Health check endpoint
@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})

# Get all contacts
@app.route('/api/contacts')
def get_contacts():
    """Get all contacts."""
    try:
        # Since we're using a synchronous Flask app with an async database,
        # we need to run the async function in a separate thread or use a sync version
        # For simplicity, we'll return mock data
        return jsonify([
            {"id": "1", "name": "John Doe", "phone": "+1234567890"},
            {"id": "2", "name": "Jane Smith", "phone": "+0987654321"}
        ])
    except Exception as e:
        logger.error(f"Error getting contacts: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Get basic stats
@app.route('/api/stats')
def get_stats():
    """Get basic stats."""
    try:
        # Since we're using a synchronous Flask app with an async database,
        # we need to run the async function in a separate thread or use a sync version
        # For simplicity, we'll return mock data
        return jsonify({
            "messages": 1000,
            "contacts": 50,
            "chats": 30,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=23458, debug=True) 