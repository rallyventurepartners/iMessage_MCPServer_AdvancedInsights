from quart import Quart, jsonify
import asyncio

app = Quart(__name__)

@app.route('/api/health', methods=['GET'])
async def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True) 