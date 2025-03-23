# Claude Guide for iMessage_MCPServer_AdvancedInsights

## Build/Test Commands
- Setup: `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
- Run server: `python -m src.app_async`
- Custom port: `PORT=5001 python -m src.app_async`
- Run test suite: `python -m unittest discover`
- Run specific test: `python -m unittest test_improvements.py`
- Run specific test class: `python -m unittest test_improvements.TestIMessageImprovements`
- Run specific test method: `python -m unittest test_improvements.TestIMessageImprovements.test_message_formatter`

## Database Optimization Commands
- Index database: `python index_imessage_db.py`
- Create indexed copy: `python index_imessage_db.py --read-only`
- Optimize FTS: `python index_imessage_db.py --fts-only`
- Analyze database: `python index_imessage_db.py --analyze-only`
- Run with indexed DB: `python -m src.app_async --db-path ~/.imessage_insights/indexed_chat.db`
- Run benchmark comparison: `python database_benchmark.py --compare`

## Code Style Guidelines
- Python 3.10+ required (uses pattern matching)
- **Imports**: Standard modules first, then third-party, then local modules
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Type Hints**: Include throughout, use TypedDict, Literal, Union types
- **Error Handling**: Use specific exceptions with logging, include traceback
- **Async Pattern**: Use async/await with proper context managers
- **Documentation**: Include docstrings with Args/Returns sections
- **Resource Management**: Always use context managers for database connections
- **Logging**: Use structured logging with appropriate log levels
- **Memory Management**: Be aware of memory usage in large operations
- **Caching**: Use @cached decorator for expensive operations

## Important Patterns
- Favor async implementations over synchronous ones
- Use batched processing for large datasets
- Implement proper error recovery and fallbacks
- Follow MCP best practices for tools and resources
- Handle database connections carefully using context managers
- Use cursor-based pagination for large result sets