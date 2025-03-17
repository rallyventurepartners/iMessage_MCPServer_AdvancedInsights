import os
import logging
import asyncio
from quart import Quart, request, jsonify
from datetime import datetime, timedelta
import traceback
import json

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

# Create Quart app
app = Quart(__name__)

# Initialize database connection
db = None  # Will be initialized during startup

# App configuration
app.config = {
    'sentiment_analysis': True,
    'network_analysis': True
}

# Startup event handler
@app.before_serving
async def startup():
    """Initialize the application before serving requests."""
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
async def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})

# Get all contacts
@app.route('/api/contacts')
async def get_contacts():
    """Get all contacts."""
    try:
        contacts = await db.get_all_contacts()
        return jsonify(contacts)
    except Exception as e:
        logger.error(f"Error getting contacts: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Get basic stats
@app.route('/api/stats')
async def get_stats():
    """Get basic stats."""
    try:
        stats = await db.get_basic_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Run the app with hypercorn if executed directly
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=23456) 