import os
import logging
import asyncio
from quart import Quart, request, jsonify
from datetime import datetime, timedelta
import traceback
import json
from quart_rate_limiter import RateLimiter, rate_limit
from logging.handlers import RotatingFileHandler
from functools import wraps
import time

# Import database module
from src.database import get_async_db

# Import input validation
from src.utils.input_validation import (
    ValidationError, 
    validate_chat_id,
    validate_phone_number,
    validate_date,
    validate_int_param,
    validate_bool_param,
    validate_json_payload,
    validate_pagination_params,
    safe_parse_json,
    validate_chat_analysis_params
)

# Configure logging
log_file = 'app.log'
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create a rotating file handler (10MB max size, keep 5 backup files)
file_handler = RotatingFileHandler(
    log_file, 
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
file_handler.setFormatter(log_formatter)

# Create a stream handler for console output
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[stream_handler, file_handler]
)
logger = logging.getLogger(__name__)
logger.info("Logging configured with rotation (10MB max size, 5 backup files)")

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
    "network_analysis": True,
    # MCP-specific configurations
    "REQUEST_TIMEOUT": int(os.environ.get("REQUEST_TIMEOUT", 60)),  # Default 60-second timeout
    "LARGE_QUERY_TIMEOUT": int(os.environ.get("LARGE_QUERY_TIMEOUT", 180))  # Default 3-minute timeout for intensive operations
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
        
        # Use the modular database implementation
        db = get_async_db(use_modular=True, db_path=db_path)
        await db.initialize()
        
        logger.info("Database connection initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        logger.error(traceback.format_exc())
        raise

# Shutdown event handler
@app.after_serving
async def shutdown():
    """Clean up resources when the server is shutting down."""
    global db
    try:
        # Close database connections
        if db:
            logger.info("Closing database connections...")
            
            # Close connection pool if available
            if hasattr(db, "_connection_pool"):
                for conn in db._connection_pool:
                    try:
                        await conn.close()
                    except Exception as e:
                        logger.warning(f"Error closing connection: {e}")
            
            # Close cache if available
            if hasattr(db, "cache") and hasattr(db.cache, "close"):
                try:
                    await db.cache.close()
                    logger.info("Cache connections closed")
                except Exception as e:
                    logger.warning(f"Error closing cache: {e}")
            
            # If db has a close method, call it
            if hasattr(db, "close"):
                await db.close()
                
        logger.info("Cleanup complete - server shutting down")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
        logger.error(traceback.format_exc())

# Health check endpoint
@app.route('/api/health', methods=['GET'])
async def health_check():
    """Health check endpoint to verify the service is running.
    
    Returns detailed information about the system status, including:
    - Database connection status
    - Database indexing status
    - Cache status
    - System resource usage
    - Application version information
    """
    try:
        # Basic status check
        system_status = "ok"
        components = {}
        
        # Check database connection
        db_status = "ok"
        db_details = {
            "initialized": db is not None and hasattr(db, "initialized") and db.initialized,
            "type": "modular" if hasattr(db, "search") else "legacy"
        }
        
        try:
            if db and db.initialized:
                # Test a simple query to make sure the connection is working
                async with db.get_db_connection() as conn:
                    async with conn.execute("SELECT 1") as cursor:
                        result = await cursor.fetchone()
                        db_details["query_test"] = result and result[0] == 1
                
                # Get connection pool info if available
                if hasattr(db, "_connection_pool"):
                    db_details["connection_pool_size"] = len(db._connection_pool)
                    db_details["max_connections"] = getattr(db, "_max_connections", "unknown")
                
                # Check if database is indexed
                db_details["indexed"] = False
                try:
                    # Check for specific indexes that should exist in optimized db
                    async with db.get_db_connection() as conn:
                        cursor = await conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='index' AND "
                            "name IN ('idx_message_date', 'idx_chat_message_join_combined')"
                        )
                        indexes = await cursor.fetchall()
                        db_details["indexed"] = len(indexes) >= 2
                        
                        # Check for FTS5 virtual table
                        cursor = await conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='table' AND name='message_fts'"
                        )
                        fts = await cursor.fetchone()
                        db_details["fts_enabled"] = fts is not None
                        
                        # Check for materialized views
                        cursor = await conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='table' AND "
                            "name IN ('mv_contact_message_counts', 'mv_chat_activity')"
                        )
                        views = await cursor.fetchall()
                        db_details["materialized_views"] = len(views)
                        
                        # Get db size
                        db_path = str(getattr(db, "db_path", "unknown"))
                        if os.path.exists(db_path):
                            db_details["size_mb"] = os.path.getsize(db_path) / (1024 * 1024)
                            
                            # Check if this is an indexed copy
                            db_details["is_indexed_copy"] = ".imessage_insights" in db_path
                except Exception as e:
                    logger.warning(f"Error checking database indexes: {e}")
                    db_details["index_check_error"] = str(e)
                
            else:
                db_status = "error"
                db_details["error"] = "Database not initialized"
                system_status = "error"
        except Exception as e:
            db_status = "error"
            db_details["error"] = str(e)
            system_status = "error"
            
        components["database"] = {
            "status": db_status,
            "details": db_details
        }
        
        # Check Redis cache status
        cache_status = "unknown"
        cache_details = {
            "enabled": db is not None and hasattr(db, "cache")
        }
        
        if db and hasattr(db, "cache"):
            try:
                # Get cache stats
                stats = await db.cache.get_stats()
                cache_details["stats"] = stats
                
                # Test cache operation with ping
                ping_result = await db.cache.ping()
                cache_details["ping_test"] = ping_result
                
                cache_status = "ok" if ping_result else "error"
            except Exception as e:
                cache_status = "error"
                cache_details["error"] = str(e)
                system_status = "degraded"  # Cache issue degrades but doesn't fail the system
        
        components["cache"] = {
            "status": cache_status,
            "details": cache_details
        }
        
        # Get system resource information
        try:
            import psutil
            import platform
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage where the app is running
            disk = psutil.disk_usage('/')
            
            system_info = {
                "cpu": {
                    "percent": cpu_percent,
                    "cores": cpu_count
                },
                "memory": {
                    "total_mb": memory.total / (1024 * 1024),
                    "available_mb": memory.available / (1024 * 1024),
                    "percent": memory.percent
                },
                "disk": {
                    "total_gb": disk.total / (1024 * 1024 * 1024),
                    "free_gb": disk.free / (1024 * 1024 * 1024),
                    "percent": disk.percent
                },
                "platform": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "python": platform.python_version()
                }
            }
            
            # Set warning status if resources are constrained
            if memory.percent > 90 or disk.percent > 90 or cpu_percent > 90:
                system_status = "warning" if system_status == "ok" else system_status
                
        except ImportError:
            system_info = {
                "error": "psutil or platform module not available"
            }
        except Exception as e:
            system_info = {
                "error": str(e)
            }
            
        components["system"] = {
            "status": "ok" if "error" not in system_info else "warning",
            "details": system_info
        }
        
        # App version and uptime
        if not hasattr(app, 'start_time'):
            app.start_time = time.time()
            
        uptime_seconds = time.time() - app.start_time
        
        app_info = {
            "version": "2.1.0",
            "uptime_seconds": uptime_seconds,
            "uptime_human": f"{int(uptime_seconds // 3600)}h {int((uptime_seconds % 3600) // 60)}m {int(uptime_seconds % 60)}s"
        }
        
        components["application"] = {
            "status": "ok",
            "details": app_info
        }
        
        # Add database optimization status
        # If database is not indexed, mark as warning
        if "database" in components and components["database"]["status"] == "ok":
            if not db_details.get("indexed", False):
                components["database"]["status"] = "warning"
                if system_status == "ok":
                    system_status = "warning"
                
                # Add optimization advice
                components["database"]["details"]["recommendation"] = (
                    "Database is not fully indexed. For optimal performance, "
                    "run 'python index_imessage_db.py --read-only' to create an indexed copy, "
                    "then restart the server with '--db-path=~/.imessage_insights/indexed_chat.db'"
                )
        
        # Return the comprehensive health check response
        return jsonify({
            "status": system_status,
            "timestamp": datetime.now().isoformat(),
            "components": components
        })
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

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

# Timeout wrapper for potentially long-running operations
async def with_timeout(coro, timeout_seconds=None):
    """Execute a coroutine with a timeout.
    
    Args:
        coro: The coroutine to execute
        timeout_seconds: The timeout in seconds, or None to use the default
        
    Returns:
        The result of the coroutine
        
    Raises:
        asyncio.TimeoutError: If the operation times out
    """
    if timeout_seconds is None:
        timeout_seconds = app.config["REQUEST_TIMEOUT"]
        
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {timeout_seconds} seconds")
        raise

# Analyze messages with a contact
@app.route('/api/analyze_contact', methods=['POST'])
@rate_limit(10, timedelta(minutes=1))  # 10 requests per minute (resource intensive)
async def analyze_contact():
    """Analyze messages with a specific contact."""
    start_time = time.time()
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
            
        # Asynchronously analyze the contact with timeout
        result = await with_timeout(
            db.analyze_contact(phone_number, start_date, end_date, page, page_size),
            app.config["LARGE_QUERY_TIMEOUT"]
        )
        
        logger.info(f"Contact analysis completed in {time.time() - start_time:.2f}s")
        return jsonify(result)
    except asyncio.TimeoutError:
        logger.error(f"Contact analysis timed out after {app.config['LARGE_QUERY_TIMEOUT']} seconds")
        return jsonify({
            "error": "Operation timed out. Please try with a smaller date range or fewer messages."
        }), 504  # Gateway Timeout
    except Exception as e:
        logger.error(f"Error analyzing contact: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Analyze messages in a group chat
@app.route('/api/analyze_group_chat', methods=['POST'])
@rate_limit(10, timedelta(minutes=1))  # 10 requests per minute (resource intensive)
async def analyze_group_chat():
    """Analyze messages in a group chat."""
    start_time = time.time()
    try:
        data = await request.get_json()
        chat_id = data.get('chat_id')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        if not chat_id:
            return jsonify({"error": "Chat ID is required"}), 400
            
        # Asynchronously analyze the group chat with timeout
        # Use a longer timeout for this resource-intensive operation
        result = await with_timeout(
            db.analyze_group_chat(chat_id, start_date, end_date),
            app.config["LARGE_QUERY_TIMEOUT"]
        )
        
        logger.info(f"Group chat analysis completed in {time.time() - start_time:.2f}s")
        return jsonify(result)
    except asyncio.TimeoutError:
        logger.error(f"Group chat analysis timed out after {app.config['LARGE_QUERY_TIMEOUT']} seconds")
        return jsonify({
            "error": "Operation timed out. Please try with a smaller date range or fewer messages."
        }), 504  # Gateway Timeout
    except Exception as e:
        logger.error(f"Error analyzing group chat: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Analyze the contact network
@app.route('/api/analyze_network', methods=['POST'])
@rate_limit(5, timedelta(minutes=5))  # 5 requests per 5 minutes (very resource intensive)
async def analyze_network():
    """Analyze the contact network based on group chats."""
    start_time = time.time()
    try:
        from analysis.async_network_analysis import analyze_contact_network_async
        
        data = await request.get_json()
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        min_shared_chats = data.get('min_shared_chats', 1)
        
        # Run network analysis asynchronously with timeout
        result = await with_timeout(
            analyze_contact_network_async(start_date, end_date, min_shared_chats),
            app.config["LARGE_QUERY_TIMEOUT"]
        )
        
        logger.info(f"Network analysis completed in {time.time() - start_time:.2f}s")
        return jsonify(result)
    except asyncio.TimeoutError:
        logger.error(f"Network analysis timed out after {app.config['LARGE_QUERY_TIMEOUT']} seconds")
        return jsonify({
            "error": "Operation timed out. Network analysis is resource intensive - please try with a smaller date range."
        }), 504  # Gateway Timeout
    except Exception as e:
        logger.error(f"Error analyzing network: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Generate network visualization data
@app.route('/api/visualize_network', methods=['POST'])
@rate_limit(5, timedelta(minutes=5))  # 5 requests per 5 minutes (very resource intensive)
async def visualize_network():
    """Generate visualization data for the contact network."""
    start_time = time.time()
    try:
        from visualization.async_network_viz import generate_network_visualization_async
        
        data = await request.get_json()
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        min_shared_chats = data.get('min_shared_chats', 1)
        layout = data.get('layout', 'spring')
        
        # Generate visualization data asynchronously with timeout
        result = await with_timeout(
            generate_network_visualization_async(start_date, end_date, min_shared_chats, layout),
            app.config["LARGE_QUERY_TIMEOUT"]
        )
        
        logger.info(f"Network visualization completed in {time.time() - start_time:.2f}s")
        return jsonify(result)
    except asyncio.TimeoutError:
        logger.error(f"Network visualization timed out after {app.config['LARGE_QUERY_TIMEOUT']} seconds")
        return jsonify({
            "error": "Operation timed out. Visualization is resource intensive - please try with a smaller date range."
        }), 504  # Gateway Timeout
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
    
    # Get port from environment variable or use default 5000
    port = int(os.environ.get("PORT", 5000))
    
    # Bind only to localhost for security (MCP best practice)
    host = '127.0.0.1'
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info") 