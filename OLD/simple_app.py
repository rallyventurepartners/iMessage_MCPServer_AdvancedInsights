import logging
import asyncio
from quart import Quart, jsonify
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Quart app
app = Quart(__name__)

# Health check endpoint
@app.route('/api/health', methods=['GET'])
async def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})

# Get basic stats
@app.route('/api/stats', methods=['GET'])
async def get_stats():
    """Get basic stats."""
    return jsonify({
        "messages": 1000,
        "contacts": 50,
        "chats": 30,
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    import hypercorn.asyncio
    from hypercorn.config import Config
    
    config = Config()
    config.bind = ["0.0.0.0:49152"]
    config.use_reloader = True
    
    asyncio.run(hypercorn.asyncio.serve(app, config)) 