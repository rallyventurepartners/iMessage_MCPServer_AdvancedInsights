"""
Main server class for the MCP Server application.

This module provides the core server functionality, tying together
all the components of the application. It handles initialization,
configuration, and shutdown of the server.
"""

import asyncio
import gc
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from ..database import get_database
from ..exceptions import ConfigurationError, DatabaseError
from ..mcp_tools.registry import set_mcp_instance
from ..utils.config import ServerConfig, init_config
from ..utils.logging_config import configure_logging
from ..utils.memory_monitor import (MemoryMonitor, initialize_memory_monitor,
                                  limit_memory, stop_memory_monitoring)

logger = logging.getLogger(__name__)


class MCPServer:
    """
    Main server class for the MCP Server application.
    
    This class manages the lifecycle of the server, including initialization,
    configuration, and shutdown. It integrates the various components of the
    application and handles the server's event loop.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the server.
        
        Args:
            config_file: Optional path to configuration file
        """
        self.running = False
        self.mcp = None
        self.initialized = False
        self.config_file = config_file
        self.config = None
    
    async def initialize(self) -> bool:
        """
        Initialize the server components.
        
        This includes loading configuration, configuring logging,
        initializing the database, and setting up the MCP instance.
        
        Returns:
            True if initialization was successful
        """
        if self.initialized:
            logger.info("Server already initialized")
            return True
        
        try:
            # Initialize configuration
            logger.info("Initializing configuration...")
            init_config(self.config_file)
            self.config = ServerConfig.get_instance()
            
            # Configure logging
            configure_logging()
            
            # Initialize memory monitoring if enabled
            memory_config = self.config.get_memory_config()
            if memory_config.get("enable_monitoring", True):
                logger.info("Initializing memory monitoring...")
                memory_limit_mb = memory_config.get("limit_mb", 2048)
                await initialize_memory_monitor()
                await limit_memory(memory_limit_mb * 1024 * 1024)
                logger.info(f"Memory monitoring enabled with limit: {memory_limit_mb} MB")
                
                # Configure memory monitor thresholds
                memory_monitor = await MemoryMonitor.get_instance()
                memory_monitor.high_threshold = memory_config.get("high_threshold", 0.75)
                memory_monitor.low_threshold = memory_config.get("low_threshold", 0.5)
                memory_monitor.monitor_interval = memory_config.get("monitor_interval", 5)
                
                # Register emergency callback
                async def memory_emergency_callback(usage_percent, system_memory):
                    logger.critical(
                        f"Memory emergency! Usage: {usage_percent:.1f}% "
                        f"System memory: {system_memory['system']['percent']:.1f}%"
                    )
                    # Force garbage collection
                    import gc
                    gc.collect()
                
                memory_monitor.register_emergency_callback(memory_emergency_callback)
            
            # Initialize the MCP instance
            logger.info("Initializing FastMCP...")
            
            # Get server config
            server_name = self.config.get("server", "name", "iMessage Advanced Insights")
            server_version = self.config.get("server", "version", "2.4.0")
            server_port = self.config.get("server", "port", 5000)
            
            # Set environment variable for port
            os.environ["MCP_PORT"] = str(server_port)
            
            # Create the MCP instance
            self.mcp = FastMCP(name=server_name, version=server_version)
            
            # Register the MCP instance with the tool registry
            set_mcp_instance(self.mcp)
            
            # Initialize the database
            db_config = self.config.get_database_config()
            logger.info("Initializing database...")
            try:
                # Get database connection
                db = await get_database(
                    minimal_mode=db_config.get("minimal_mode", False),
                    use_shards=db_config.get("use_sharding"),
                    shards_dir=db_config.get("shards_dir"),
                )
                
                # Log database initialization
                logger.info("Database initialized successfully")
                
                # Log database settings
                logger.info(f"Database minimal mode: {db_config.get('minimal_mode', False)}")
                logger.info(f"Database sharding: {db_config.get('use_sharding', 'auto')}")
            except DatabaseError as e:
                logger.error(f"Error initializing database: {e}")
                logger.warning("Continuing without database connection")
            
            # Import all tool modules to register them
            self._import_tools()
            
            self.initialized = True
            logger.info("Server initialization complete")
            return True
        except Exception as e:
            logger.error(f"Error initializing server: {e}")
            return False
    
    def _import_tools(self) -> None:
        """Import all tool modules to register them with the tool registry."""
        try:
            # Import consent tools first since they're used by other tools
            from ..mcp_tools import consent
            
            # Import other tool modules
            # These imports are primarily to trigger the tool registration
            try:
                from ..mcp_tools import contacts
            except ImportError:
                logger.warning("Contacts tools not available")
            
            try:
                from ..mcp_tools import messages
            except ImportError:
                logger.warning("Messages tools not available")
            
            try:
                from ..mcp_tools import group_chats
            except ImportError:
                logger.warning("Group chat tools not available")
            
            try:
                from ..mcp_tools import network
            except ImportError:
                logger.warning("Network tools not available")
            
            try:
                from ..mcp_tools import templates
            except ImportError:
                logger.warning("Template tools not available")
            
            try:
                from ..mcp_tools import topic_analysis
            except ImportError:
                logger.warning("Topic analysis tools not available")
            
            try:
                from ..mcp_tools import visualization
            except ImportError:
                logger.warning("Visualization tools not available")
            
            try:
                from ..mcp_tools import conversation_intelligence
            except ImportError:
                logger.warning("Conversation intelligence tools not available")
            
            try:
                from ..mcp_tools import predictive_analytics
            except ImportError:
                logger.warning("Predictive analytics tools not available")
            
            try:
                from ..mcp_tools import life_events
            except ImportError:
                logger.warning("Life events tools not available")
            
            try:
                from ..mcp_tools import network_intelligence
            except ImportError:
                logger.warning("Network intelligence tools not available")
            
            try:
                from ..mcp_tools import report_generation
            except ImportError:
                logger.warning("Report generation tools not available")
            
            try:
                from ..mcp_tools import communication_style
            except ImportError:
                logger.warning("Communication style tools not available")
            
            # Log registered tools count
            from ..mcp_tools.registry import list_tools
            tools = list_tools()
            logger.info(f"Registered {len(tools)} tools")
        except ImportError as e:
            logger.error(f"Error importing tool modules: {e}")
    
    async def start(self) -> bool:
        """
        Start the server.
        
        Returns:
            True if server was started successfully
        """
        if not self.initialized:
            success = await self.initialize()
            if not success:
                logger.error("Failed to initialize server")
                return False
        
        if self.running:
            logger.warning("Server already running")
            return True
        
        try:
            # Log server info
            server_name = self.config.get("server", "name", "iMessage Advanced Insights")
            server_version = self.config.get("server", "version", "2.4.0")
            server_port = self.config.get("server", "port", 5000)
            
            logger.info(f"Starting {server_name} v{server_version} on port {server_port}")
            
            # Set up signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            # Mark as running
            self.running = True
            
            # Start the server (this will block)
            self.mcp.run()
            
            return True
        except Exception as e:
            logger.error(f"Error starting server: {e}")
            self.running = False
            return False
    
    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def handle_signal(sig, frame):
            logger.info(f"Received signal {sig}, shutting down...")
            asyncio.create_task(self.stop())
        
        # Register signal handlers
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
    
    async def stop(self) -> None:
        """Stop the server gracefully."""
        if not self.running:
            logger.warning("Server not running")
            return
        
        logger.info("Stopping server...")
        
        try:
            # Stop memory monitoring
            await stop_memory_monitoring()
            
            # Close database connections
            try:
                from ..database import AsyncMessagesDBBase
                from ..database.async_messages_db_new import AsyncMessagesDB
                
                # Try to close any open database connections
                db_instances = [
                    obj for obj in gc.get_objects()
                    if isinstance(obj, AsyncMessagesDBBase)
                ]
                
                for db in db_instances:
                    await db.close()
                
                logger.info(f"Closed {len(db_instances)} database connections")
            except Exception as e:
                logger.warning(f"Error closing database connections: {e}")
            
            # Mark as not running
            self.running = False
            
            logger.info("Server stopped")
        except Exception as e:
            logger.error(f"Error stopping server: {e}")
    
    async def restart(self) -> bool:
        """
        Restart the server.
        
        Returns:
            True if server was restarted successfully
        """
        logger.info("Restarting server...")
        
        # Stop the server
        await self.stop()
        
        # Re-initialize
        self.initialized = False
        success = await self.initialize()
        
        if not success:
            logger.error("Failed to re-initialize server")
            return False
        
        # Start the server
        return await self.start()
