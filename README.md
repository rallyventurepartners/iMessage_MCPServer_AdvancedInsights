# iMessage Advanced Insights MCP Server

A powerful analysis server for extracting insights from iMessage conversations. This application provides advanced analytics, sentiment analysis, and network visualization for your iMessage data.

## Features

- **Contact Analysis**: Analyze message patterns with individual contacts
- **Group Chat Analysis**: Get insights into group chat dynamics and participation
- **Network Analysis**: Discover connections between your contacts based on shared group chats
- **Sentiment Analysis**: Track emotional patterns in conversations over time
- **Network Visualization**: Generate interactive visualizations of your social network
- **Natural Language Queries**: Ask questions about your iMessage data in plain English
- **Enhanced Contact Resolution**: Robust contact identification across multiple identifier types
- **Performance Optimizations**: Includes Redis caching with in-memory fallback, database indexes, and more

## Project Structure

The project has been modularized for better organization and maintainability:

```
iMessage_MCPServer_AdvancedInsights/
├── main_async.py            # Main entry point (asynchronous version)
├── requirements.txt         # Python dependencies
├── PERFORMANCE_OPTIMIZATIONS.md # Details of performance optimizations
├── setup.py                 # Package setup script
├── src/                     # Source code directory
│   ├── __init__.py
│   ├── app_async.py         # Quart application with async API endpoints
│   ├── database/            # Database operations
│   │   ├── __init__.py
│   │   ├── async_messages_db.py   # Async iMessage database interactions
│   │   └── db_indexes.sql         # SQL indexes for better performance
│   ├── analysis/            # Analysis modules
│   │   ├── __init__.py
│   │   ├── async_network_analysis.py  # Async contact network analysis
│   │   └── async_sentiment_analysis.py # Async sentiment analysis
│   ├── visualization/       # Visualization modules
│   │   ├── __init__.py
│   │   └── async_network_viz.py   # Async network visualization
│   └── utils/               # Utility functions
│       ├── __init__.py
│       ├── redis_cache.py   # Redis caching with in-memory fallback
│       └── async_query_processor.py # Natural language query processing
```

## Installation

### Prerequisites

- Python 3.8 or higher
- macOS with iMessage enabled
- Read access to the iMessage database
- Redis (optional, falls back to in-memory cache if not available)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/rallyventurepartners/iMessage_MCPServer_AdvancedInsights.git
   cd iMessage_MCPServer_AdvancedInsights
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download additional resources:
   ```bash
   python -m textblob.download_corpora
   python -m spacy download en_core_web_sm
   ```

4. Run the application:
   ```bash
   python main_async.py
   ```

## Enhanced Contact Resolution

The Enhanced Contact Resolution system provides robust identification and display of contacts across various identifiers used in iMessage conversations.

### Key Features

- **Multi-identifier Support**: Resolves phone numbers, emails, and other identifiers
- **Normalized Display**: Consistently formats identifiers with contact names
- **Advanced Caching**: Thread-safe caching for improved performance
- **Contact Framework Integration**: Direct access to macOS Contacts data
- **Intelligent Formatting**: Avoids redundant information in display names

### How It Works

1. **Identifier Detection**: Automatically detects whether an identifier is a phone number, email, or other iMessage ID
2. **Contact Lookup**: Searches macOS Contacts using the appropriate method for each identifier type
3. **Intelligent Formatting**: Creates consistent display formats that include both the contact name and identifier
4. **Caching**: Stores results to improve performance on repeated lookups

### Usage Examples

```python
# Get a MessagesDB instance
db = MessagesDB()

# Resolve a contact
contact_info = db.contact_resolver.resolve_contact("+1234567890")

# Output:
{
    "identifier": "+1234567890",
    "type": "phone",
    "name": "John Smith",
    "display_format": "+1 (234) 567-890",
    "display_name": "John Smith (+1 (234) 567-890)"
}
```

### Integration Points

The enhanced contact resolution is integrated throughout the codebase:

- `get_contacts()`: Enhanced contact info for all contacts
- `get_chat_transcript()`: Better contact display in transcripts
- All visualization and report functions

### Privacy & Permission Handling

The system safely handles Contacts framework permissions:

1. Checks if permission is granted during initialization
2. Gracefully degrades if permission is not available
3. Provides meaningful display even when Contacts access is denied
4. Maintains local caching to avoid repeated permission checks

### Technical Details

- **Phone Number Formatting**: Uses the phonenumbers library for E.164, international, and national formats
- **Email Detection**: Uses regex pattern matching for reliable email identification
- **Threading**: Uses thread locks to ensure cache thread safety
- **Performance**: Intelligent caching with minimal redundant lookups

## Usage

### Running the Server

To run the server:

```bash
# Set the database path (optional)
export DB_PATH=~/Library/Messages/chat.db

# Run the server
python main_async.py
```

The server will start on port 5000, accessing the default iMessage database.

### Command Line Options

```bash
python main_async.py --help

# Options include:
# --host          # Host to bind to (default: 0.0.0.0)
# --port          # Port to listen on (default: 5000)
# --db-path       # Custom path to iMessage database
# --debug         # Enable debug mode
# --no-sentiment  # Disable sentiment analysis
# --no-network    # Disable network analysis
```

### API Endpoints

The server provides the following RESTful API endpoints:

#### Basic Endpoints

- `GET /api/health` - Check if the server is running
- `GET /api/contacts` - Get a list of all contacts
- `GET /api/group_chats` - Get a list of all group chats

#### Analysis Endpoints

- `GET /api/analyze_contact` - Analyze messages with a specific contact
  - Query params: `phone_number`, `start_date`, `end_date`, `time_period`

- `GET /api/analyze_group_chat` - Analyze messages in a group chat
  - Query params: `chat_id` or `chat_name`, `start_date`, `end_date`, `time_period`

- `GET /api/analyze_network` - Analyze the contact network
  - Query params: `start_date`, `end_date`, `min_shared_chats`, `advanced`

- `GET /api/analyze_sentiment` - Analyze sentiment in a conversation
  - Query params: `phone_number` or `chat_id` or `chat_name`, `start_date`, `end_date`, `include_messages`

#### Visualization Endpoints

- `GET /api/visualize_network` - Generate network visualization data
  - Query params: `start_date`, `end_date`, `min_shared_chats`, `max_nodes`, `include_labels`, `layout`, `color_by`

#### Natural Language Processing

- `POST /api/process_query` - Process a natural language query
  - Request body: `{"query": "Analyze my Friends group chat from last month"}`

## Examples

### Analyze a Contact

```bash
curl "http://localhost:5000/api/analyze_contact?phone_number=+15551234567&time_period=last 3 months"
```

### Analyze a Group Chat

```bash
curl "http://localhost:5000/api/analyze_group_chat?chat_name=Friends&time_period=this year"
```

### Analyze Network

```bash
curl "http://localhost:5000/api/analyze_network?advanced=true&min_shared_chats=2"
```

### Natural Language Query

```bash
curl -X POST "http://localhost:5000/api/process_query" \
     -H "Content-Type: application/json" \
     -d '{"query": "Show me sentiment analysis for my Family group chat"}'
```

## Dependencies

- Quart - Asynchronous web server framework
- NetworkX - Network analysis
- TextBlob - Sentiment analysis
- NLTK - Natural language processing
- Python-Louvain - Community detection
- NumPy & Pandas - Data processing
- Matplotlib - Data visualization
- SpaCy - Advanced NLP
- aiosqlite - Asynchronous SQLite access
- hypercorn - ASGI server for Quart

## Redis Cache with Memory Fallback

The application uses Redis for high-performance caching but gracefully falls back to an in-memory cache when Redis is unavailable:

- Automatic detection of Redis availability
- Seamless in-memory fallback when Redis is not installed or not running
- Consistent API regardless of backend
- Thread-safe implementation
- Intelligent key generation based on function parameters
- Configurable TTL for different cache types

## Running the Optimized Server

To run the server with all optimizations enabled:

```bash
python main_async.py
```

You can also customize the optimization parameters:

```bash
python main_async.py --connection-pool-size 15 --cache-ttl 7200 --batch-size 1000
```

## Privacy Notice

This application accesses your local iMessage database to provide insights. All data processing is done locally on your machine, and no data is sent to external servers. The application respects your privacy and does not collect or share any personal information.

## License

MIT License

## Contributors

- David Jelinek, RVP - Rally Venture Partners ([@davidjelinekk](https://github.com/davidjelinekk))

## Acknowledgments

- Inspiration sources
- Libraries and resources used

## Asynchronous Processing Support

This project now includes asynchronous processing capabilities for improved performance. The async implementation:

- Uses `aiosqlite` for non-blocking database operations
- Implements a connection pool for efficient database connections
- Provides all the same functionality with better performance
- Uses Quart instead of Flask for asynchronous HTTP handling
- Supports processing large datasets without blocking

## Performance Optimizations

The application has been optimized for high performance through the following techniques:

### 1. Redis Query Caching
- Implements Redis-based caching for frequently accessed database queries
- Significantly reduces database load for repeated queries
- Intelligent cache invalidation for modified data
- Configurable TTL (time-to-live) for cached results

### 2. Optimized Network Visualization
- Adaptive layout algorithms based on graph size
- Uses fixed random seeds for deterministic rendering
- Balances quality and speed with tuned parameters
- Simplified calculations for larger graphs with minimal visual impact

### 3. Batch Processing for NLP
- Processes text in optimized batches for sentiment analysis
- Uses ThreadPoolExecutor for CPU-bound NLP tasks
- Parallel processing of messages by sender and time period
- Batch size of 500 messages for optimal throughput

### 4. Database Indexes
- Strategic SQL indexes for commonly queried fields
- Optimized indexes for message dates, chat joins, and text search
- Improves query performance by orders of magnitude
- Automatically created during application initialization

### 5. Enhanced Connection Pooling
- Increased connection pool size for better concurrency
- Thread-safe connection management
- Efficient connection reuse to reduce overhead
- Graceful handling of connection failures

### 6. Optimized spaCy Usage
- Disables unnecessary NLP pipeline components
- Uses smaller language models where appropriate
- Optimized for text classification performance
- Memory-efficient NLP processing

### 7. Smart Data Pagination
- Implements cursor-based pagination for large result sets
- Configurable page sizes for different endpoints
- Metadata with total counts and page information
- Prevents memory overload with large conversations

### 8. API Rate Limiting
- Protects server from excessive requests
- Different rate limits based on endpoint resource intensity
- Custom error responses for rate-limited requests
- Configurable time windows for rate limiting

### 9. Improved Error Handling
- Consistent error response format
- Detailed logging for debugging
- Graceful degradation under high load
- Automatic retry mechanisms for transient failures

### 10. Incremental Network Analysis
- Caches network graphs for future updates
- Updates only modified portions of the graph
- File-based storage for large graphs
- Significantly faster network updates for minor changes
