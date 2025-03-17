import logging
import json
import sys
import os
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('imessage_insights.log')
    ]
)

logger = logging.getLogger(__name__)

# Import modules
from src.database.messages_db import MessagesDB
from src.analysis.network_analysis import analyze_contact_network, analyze_contact_network_advanced
from src.analysis.sentiment_analysis import analyze_sentiment
from src.visualization.network_viz import generate_network_visualization
from src.utils.helpers import (
    error_response, 
    parse_natural_language_time_period, 
    should_analyze_group_chat, 
    should_analyze_all_contacts
)

# Create Flask app
app = Flask(__name__)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "ok", "version": "1.0.0"})

@app.route('/api/contacts', methods=['GET'])
def get_contacts():
    """Get all contacts from the database."""
    try:
        db = MessagesDB()
        contacts = db.get_contacts()
        return jsonify({"contacts": contacts})
    except Exception as e:
        logger.error(f"Error getting contacts: {e}")
        return jsonify(error_response("SERVER_ERROR", f"Error getting contacts: {str(e)}")), 500

@app.route('/api/group_chats', methods=['GET'])
def get_group_chats():
    """Get all group chats from the database."""
    try:
        db = MessagesDB()
        group_chats = db.get_group_chats()
        return jsonify({"group_chats": group_chats})
    except Exception as e:
        logger.error(f"Error getting group chats: {e}")
        return jsonify(error_response("SERVER_ERROR", f"Error getting group chats: {str(e)}")), 500

@app.route('/api/analyze_contact', methods=['GET'])
def analyze_contact_endpoint():
    """Analyze messages with a specific contact."""
    phone_number = request.args.get('phone_number')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    try:
        db = MessagesDB()
        
        # Handle natural language time period
        time_period = request.args.get('time_period')
        if time_period and not (start_date and end_date):
            date_range = parse_natural_language_time_period(time_period)
            if date_range:
                start_date = date_range.get('start_date')
                end_date = date_range.get('end_date')
        
        # Get analysis results
        results = db.analyze_contact(phone_number, start_date, end_date)
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error analyzing contact: {e}")
        return jsonify(error_response("ANALYSIS_ERROR", f"Error analyzing contact: {str(e)}")), 500

@app.route('/api/analyze_group_chat', methods=['GET'])
def analyze_group_chat_endpoint():
    """Analyze messages in a group chat."""
    chat_id = request.args.get('chat_id')
    chat_name = request.args.get('chat_name')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    try:
        db = MessagesDB()
        
        # Handle natural language time period
        time_period = request.args.get('time_period')
        if time_period and not (start_date and end_date):
            date_range = parse_natural_language_time_period(time_period)
            if date_range:
                start_date = date_range.get('start_date')
                end_date = date_range.get('end_date')
        
        # Handle chat_name to chat_id resolution
        if chat_name and not chat_id:
            chat_id = db.find_chat_id_by_name(chat_name)
            if not chat_id:
                return jsonify(error_response("NOT_FOUND", f"Could not find group chat with name: {chat_name}")), 404
        
        # Get analysis results
        results = db.analyze_group_chat(chat_id, start_date, end_date)
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error analyzing group chat: {e}")
        return jsonify(error_response("ANALYSIS_ERROR", f"Error analyzing group chat: {str(e)}")), 500

@app.route('/api/analyze_network', methods=['GET'])
def analyze_network_endpoint():
    """Analyze the network of contacts based on group chat participation."""
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    min_shared_chats = int(request.args.get('min_shared_chats', 1))
    advanced = request.args.get('advanced', 'false').lower() == 'true'
    
    try:
        # Handle natural language time period
        time_period = request.args.get('time_period')
        if time_period and not (start_date and end_date):
            date_range = parse_natural_language_time_period(time_period)
            if date_range:
                start_date = date_range.get('start_date')
                end_date = date_range.get('end_date')
        
        # Get analysis results
        if advanced:
            results = analyze_contact_network_advanced(start_date, end_date, min_shared_chats)
        else:
            results = analyze_contact_network(start_date, end_date, min_shared_chats)
        
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error analyzing network: {e}")
        return jsonify(error_response("ANALYSIS_ERROR", f"Error analyzing network: {str(e)}")), 500

@app.route('/api/visualize_network', methods=['GET'])
def visualize_network_endpoint():
    """Generate visualization data for the contact network."""
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    min_shared_chats = int(request.args.get('min_shared_chats', 1))
    max_nodes = int(request.args.get('max_nodes', 100))
    include_labels = request.args.get('include_labels', 'true').lower() == 'true'
    layout_algorithm = request.args.get('layout', 'force_directed')
    color_by = request.args.get('color_by', 'community')
    
    try:
        # Handle natural language time period
        time_period = request.args.get('time_period')
        if time_period and not (start_date and end_date):
            date_range = parse_natural_language_time_period(time_period)
            if date_range:
                start_date = date_range.get('start_date')
                end_date = date_range.get('end_date')
        
        # Get visualization data
        results = generate_network_visualization(
            start_date, 
            end_date, 
            min_shared_chats, 
            max_nodes, 
            include_labels, 
            layout_algorithm, 
            color_by
        )
        
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error generating network visualization: {e}")
        return jsonify(error_response("VISUALIZATION_ERROR", f"Error generating visualization: {str(e)}")), 500

@app.route('/api/analyze_sentiment', methods=['GET'])
def analyze_sentiment_endpoint():
    """Analyze sentiment of messages in a conversation."""
    phone_number = request.args.get('phone_number')
    chat_id = request.args.get('chat_id')
    chat_name = request.args.get('chat_name')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    include_messages = request.args.get('include_messages', 'false').lower() == 'true'
    
    try:
        db = MessagesDB()
        
        # Handle natural language time period
        time_period = request.args.get('time_period')
        if time_period and not (start_date and end_date):
            date_range = parse_natural_language_time_period(time_period)
            if date_range:
                start_date = date_range.get('start_date')
                end_date = date_range.get('end_date')
        
        # Handle chat_name to chat_id resolution
        if chat_name and not chat_id:
            chat_id = db.find_chat_id_by_name(chat_name)
            if not chat_id:
                return jsonify(error_response("NOT_FOUND", f"Could not find chat with name: {chat_name}")), 404
        
        # Get sentiment analysis results
        results = analyze_sentiment(
            phone_number=phone_number,
            chat_id=chat_id,
            start_date=start_date,
            end_date=end_date,
            include_individual_messages=include_messages
        )
        
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        return jsonify(error_response("ANALYSIS_ERROR", f"Error analyzing sentiment: {str(e)}")), 500

@app.route('/api/process_query', methods=['POST'])
def process_natural_language_query():
    """Process a natural language query about iMessage data."""
    if not request.json or 'query' not in request.json:
        return jsonify(error_response("INVALID_REQUEST", "Missing query parameter")), 400
    
    query = request.json['query']
    logger.info(f"Processing natural language query: {query}")
    
    try:
        # Check if we should analyze a group chat
        group_chat_info = should_analyze_group_chat(query)
        if group_chat_info:
            # Extract chat name/ID and date range
            chat_name = group_chat_info.get('chat_name')
            start_date = group_chat_info.get('start_date')
            end_date = group_chat_info.get('end_date')
            
            db = MessagesDB()
            chat_id = db.find_chat_id_by_name(chat_name)
            
            if chat_id:
                results = db.analyze_group_chat(chat_id, start_date, end_date)
                return jsonify({
                    "query_type": "group_chat_analysis",
                    "chat_name": chat_name,
                    "date_range": {"start": start_date, "end": end_date},
                    "results": results
                })
            else:
                return jsonify(error_response("NOT_FOUND", f"Could not find group chat: {chat_name}")), 404
        
        # Check if we should analyze all contacts (network analysis)
        if should_analyze_all_contacts(query):
            # Extract date range
            date_range = parse_natural_language_time_period(query)
            start_date = date_range.get('start_date') if date_range else None
            end_date = date_range.get('end_date') if date_range else None
            
            # Determine if we should do advanced analysis
            advanced = "detailed" in query.lower() or "advanced" in query.lower()
            
            if advanced:
                results = analyze_contact_network_advanced(start_date, end_date)
            else:
                results = analyze_contact_network(start_date, end_date)
                
            return jsonify({
                "query_type": "network_analysis",
                "advanced": advanced,
                "date_range": {"start": start_date, "end": end_date},
                "results": results
            })
        
        # Default response if we couldn't interpret the query
        return jsonify({
            "query_type": "unknown",
            "message": "I couldn't understand your query. Please try again with a more specific question."
        })
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify(error_response("QUERY_ERROR", f"Error processing query: {str(e)}")), 500

def run_server():
    """Run the Flask server."""
    try:
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    run_server()