from quart import Quart, jsonify

app = Quart(__name__)

@app.route('/api/health')
async def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=23456) 