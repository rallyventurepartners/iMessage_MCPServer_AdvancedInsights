import os
import logging
import asyncio
from quart import Quart, request, jsonify
from datetime import datetime, timedelta
import traceback
import json
from quart_rate_limiter import RateLimiter, rate_limit

# Import async database module
from src.database.async_messages_db import AsyncMessagesDB

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

# Configure Flask/Quart settings for compatibility
app.config.update({
    "PROVIDE_AUTOMATIC_OPTIONS": True,
    "DEBUG": True,
    "SERVER_NAME": None,
    "APPLICATION_ROOT": "/",
    "PREFERRED_URL_SCHEME": "http",
    "TRAP_HTTP_EXCEPTIONS": False,
    "TRAP_BAD_REQUEST_ERRORS": False,
    "PERMANENT_SESSION_LIFETIME": timedelta(days=31),
    "sentiment_analysis": True,
    "network_analysis": True
})

# Add rate limiting to prevent server overload
limiter = RateLimiter(app)

# Initialize database connection
db = None  # Will be initialized during startup

# App configuration
app.db_path = None  # Set by main_async.py

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
@app.route('/api/health', methods=['GET'])
async def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})

# Get all contacts
@app.route('/api/contacts', methods=['GET'])
@rate_limit(30, timedelta(minutes=1))  # 30 requests per minute
async def get_contacts():
    """Get all contacts from the database."""
    try:
        # Get pagination parameters from query
        page = request.args.get('page', 1, type=int)
        page_size = request.args.get('page_size', 100, type=int)
        
        # Validate pagination parameters
        if page < 1:
            page = 1
        if page_size < 1 or page_size > 500:
            page_size = 100
            
        contacts = await db.get_contacts()
        
        # Calculate total pages
        total_contacts = len(contacts)
        total_pages = (total_contacts + page_size - 1) // page_size
        
        # Apply pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_contacts = contacts[start_idx:end_idx]
        
        return jsonify({
            "contacts": paginated_contacts,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_items": total_contacts,
                "total_pages": total_pages
            }
        })
    except Exception as e:
        logger.error(f"Error retrieving contacts: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Get all group chats
@app.route('/api/group_chats', methods=['GET'])
@rate_limit(30, timedelta(minutes=1))  # 30 requests per minute
async def get_group_chats():
    """Get all group chats from the database."""
    try:
        # Get pagination parameters from query
        page = request.args.get('page', 1, type=int)
        page_size = request.args.get('page_size', 50, type=int)
        
        # Validate pagination parameters
        if page < 1:
            page = 1
        if page_size < 1 or page_size > 200:
            page_size = 50
            
        group_chats = await db.get_group_chats()
        
        # Calculate total pages
        total_chats = len(group_chats)
        total_pages = (total_chats + page_size - 1) // page_size
        
        # Apply pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_chats = group_chats[start_idx:end_idx]
        
        return jsonify({
            "group_chats": paginated_chats,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_items": total_chats,
                "total_pages": total_pages
            }
        })
    except Exception as e:
        logger.error(f"Error retrieving group chats: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Analyze messages with a contact
@app.route('/api/analyze_contact', methods=['POST'])
@rate_limit(10, timedelta(minutes=1))  # 10 requests per minute (resource intensive)
async def analyze_contact():
    """Analyze messages with a specific contact."""
    try:
        data = await request.get_json()
        phone_number = data.get('phone_number')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        # Pagination parameters
        page = data.get('page', 1)
        page_size = data.get('page_size', 100)
        
        if not phone_number:
            return jsonify({"error": "Phone number is required"}), 400
            
        # Asynchronously analyze the contact
        result = await db.analyze_contact(phone_number, start_date, end_date, page, page_size)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error analyzing contact: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Analyze messages in a group chat
@app.route('/api/analyze_group_chat', methods=['POST'])
@rate_limit(10, timedelta(minutes=1))  # 10 requests per minute (resource intensive)
async def analyze_group_chat():
    """Analyze messages in a group chat."""
    try:
        data = await request.get_json()
        chat_id = data.get('chat_id')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        if not chat_id:
            return jsonify({"error": "Chat ID is required"}), 400
            
        # Asynchronously analyze the group chat
        result = await db.analyze_group_chat(chat_id, start_date, end_date)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error analyzing group chat: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Analyze the contact network
@app.route('/api/analyze_network', methods=['POST'])
@rate_limit(5, timedelta(minutes=5))  # 5 requests per 5 minutes (very resource intensive)
async def analyze_network():
    """Analyze the contact network based on group chats."""
    try:
        from analysis.async_network_analysis import analyze_contact_network_async
        
        data = await request.get_json()
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        min_shared_chats = data.get('min_shared_chats', 1)
        
        # Run network analysis asynchronously
        result = await analyze_contact_network_async(start_date, end_date, min_shared_chats)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error analyzing network: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Generate network visualization data
@app.route('/api/visualize_network', methods=['POST'])
@rate_limit(5, timedelta(minutes=5))  # 5 requests per 5 minutes (very resource intensive)
async def visualize_network():
    """Generate visualization data for the contact network."""
    try:
        from visualization.async_network_viz import generate_network_visualization_async
        
        data = await request.get_json()
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        min_shared_chats = data.get('min_shared_chats', 1)
        layout = data.get('layout', 'spring')
        
        # Generate visualization data asynchronously
        result = await generate_network_visualization_async(start_date, end_date, min_shared_chats, layout)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error generating network visualization: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Analyze sentiment in conversations
@app.route('/api/analyze_sentiment', methods=['POST'])
@rate_limit(10, timedelta(minutes=1))  # 10 requests per minute (resource intensive)
async def analyze_sentiment():
    """Analyze sentiment in conversations with a contact or group."""
    try:
        from analysis.async_sentiment_analysis import analyze_sentiment_async
        
        data = await request.get_json()
        phone_number = data.get('phone_number')
        chat_id = data.get('chat_id')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        include_individual_messages = data.get('include_individual_messages', False)
        
        if not phone_number and not chat_id:
            return jsonify({"error": "Either phone number or chat ID is required"}), 400
            
        # Analyze sentiment asynchronously
        result = await analyze_sentiment_async(
            phone_number, 
            chat_id, 
            start_date, 
            end_date, 
            include_individual_messages
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Process natural language queries
@app.route('/api/query', methods=['POST'])
@rate_limit(1, timedelta(seconds=2))  # Limit to 1 request per 2 seconds
async def process_query():
    """Process a natural language query."""
    try:
        from src.utils.async_query_processor import process_natural_language_query_async
        
        data = await request.get_json()
        query = data.get('query')
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
            
        # Process query asynchronously
        result = await process_natural_language_query_async(query)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Error handler for rate limiting
@app.errorhandler(429)
async def ratelimit_handler(e):
    """Handle rate limit exceeded errors."""
    return jsonify({
        "error": "Rate limit exceeded",
        "message": "Too many requests. Please try again later.",
        "status_code": 429
    }), 429

# Run the app with uvicorn if executed directly
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000, log_level="info") 