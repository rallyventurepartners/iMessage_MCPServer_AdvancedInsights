"""
Tool registry for the MCP Server application.

This module provides a centralized registry for MCP tools, making it
easier to manage and organize tools and ensure consistent error handling
and response formatting.
"""

import functools
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, Union, cast

from fastmcp import FastMCP

from ..exceptions import ToolExecutionError
from ..utils.decorators import handle_tool_errors, performance_monitor, requires_consent, run_async
from ..utils.responses import error_response, success_response

logger = logging.getLogger(__name__)

# Type for tool functions
F = TypeVar('F', bound=Callable)
ToolFunction = Callable[..., Dict[str, Any]]


class ToolRegistry:
    """
    Registry for MCP tools.
    
    This class implements the Singleton pattern to ensure only one
    instance exists throughout the application. It provides a centralized
    registry for MCP tools and handles registration with the FastMCP instance.
    """
    
    # Singleton instance
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'ToolRegistry':
        """Get the singleton instance of the ToolRegistry."""
        if cls._instance is None:
            cls._instance = ToolRegistry()
        return cls._instance
    
    def __init__(self):
        """Initialize the ToolRegistry."""
        # Prevent multiple instances
        if ToolRegistry._instance is not None:
            raise RuntimeError("ToolRegistry is a singleton. Use get_instance() instead.")
        
        # Store registered tools
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.mcp_instance: Optional[FastMCP] = None
        self.registered_with_mcp: Set[str] = set()
        
        # Set as singleton instance
        ToolRegistry._instance = self
    
    def set_mcp_instance(self, mcp: FastMCP) -> None:
        """
        Set the FastMCP instance for tool registration.
        
        Args:
            mcp: FastMCP instance
        """
        self.mcp_instance = mcp
        
        # Register any tools that were registered before the MCP instance was set
        self._register_pending_tools()
    
    def _register_pending_tools(self) -> None:
        """Register any tools that are pending registration with the MCP instance."""
        if not self.mcp_instance:
            return
        
        for tool_name, tool_info in self.tools.items():
            if tool_name not in self.registered_with_mcp:
                try:
                    # Register the tool with the MCP instance
                    self.mcp_instance.tool()(tool_info["function"])
                    self.registered_with_mcp.add(tool_name)
                    logger.info(f"Registered tool with MCP: {tool_name}")
                except Exception as e:
                    logger.error(f"Error registering tool {tool_name} with MCP: {e}")
    
    def register_tool(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        requires_user_consent: bool = True,
        monitor_performance: bool = True,
    ) -> Callable[[F], F]:
        """
        Decorator for registering a tool.
        
        Args:
            name: Name of the tool (defaults to function name)
            description: Description of the tool
            requires_user_consent: Whether the tool requires user consent
            monitor_performance: Whether to monitor tool performance
            
        Returns:
            Decorator function
        """
        def decorator(func: F) -> F:
            # Determine tool name
            tool_name = name or func.__name__
            if tool_name.endswith("_tool"):
                # Remove "_tool" suffix from name for cleaner API
                tool_name = tool_name[:-5]
            
            # Determine description
            tool_description = description or func.__doc__ or f"Tool: {tool_name}"
            
            # Determine if function is already async
            is_async = inspect.iscoroutinefunction(func)
            
            # Apply decorators
            wrapped_func = func
            
            # Apply error handling
            wrapped_func = handle_tool_errors(wrapped_func)  # type: ignore
            
            # Apply consent checking if required
            if requires_user_consent:
                wrapped_func = requires_consent(wrapped_func)  # type: ignore
            
            # Apply performance monitoring if enabled
            if monitor_performance:
                wrapped_func = performance_monitor(wrapped_func)  # type: ignore
            
            # Apply run_async if not already async
            if not is_async:
                wrapped_func = run_async(wrapped_func)  # type: ignore
            
            # Store the tool information
            self.tools[tool_name] = {
                "name": tool_name,
                "description": tool_description,
                "function": wrapped_func,
                "requires_consent": requires_user_consent,
                "original_function": func,
            }
            
            # Register with MCP if instance is available
            if self.mcp_instance and tool_name not in self.registered_with_mcp:
                try:
                    self.mcp_instance.tool()(wrapped_func)
                    self.registered_with_mcp.add(tool_name)
                    logger.info(f"Registered tool with MCP: {tool_name}")
                except Exception as e:
                    logger.error(f"Error registering tool {tool_name} with MCP: {e}")
            
            return cast(F, wrapped_func)
        
        return decorator
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """
        Get a list of all registered tools.
        
        Returns:
            List of tool information dictionaries
        """
        return [
            {
                "name": name,
                "description": info["description"],
                "requires_consent": info["requires_consent"],
            }
            for name, info in self.tools.items()
        ]
    
    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific tool.
        
        Args:
            name: Name of the tool
            
        Returns:
            Tool information or None if not found
        """
        # Try with and without "_tool" suffix
        if name in self.tools:
            return self.tools[name]
        
        if name.endswith("_tool"):
            base_name = name[:-5]
            if base_name in self.tools:
                return self.tools[base_name]
        else:
            suffixed_name = f"{name}_tool"
            if suffixed_name in self.tools:
                return self.tools[suffixed_name]
        
        return None


# Singleton instance for convenience
_registry = ToolRegistry.get_instance()


# Convenience function for registering tools
def register_tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    requires_user_consent: bool = True,
    monitor_performance: bool = True,
) -> Callable[[F], F]:
    """
    Decorator for registering a tool.
    
    Args:
        name: Name of the tool (defaults to function name)
        description: Description of the tool
        requires_user_consent: Whether the tool requires user consent
        monitor_performance: Whether to monitor tool performance
        
    Returns:
        Decorator function
    """
    return _registry.register_tool(
        name=name,
        description=description,
        requires_user_consent=requires_user_consent,
        monitor_performance=monitor_performance,
    )


# Convenience function for setting the MCP instance
def set_mcp_instance(mcp: FastMCP) -> None:
    """
    Set the FastMCP instance for tool registration.
    
    Args:
        mcp: FastMCP instance
    """
    _registry.set_mcp_instance(mcp)


# Convenience function for listing tools
def list_tools() -> List[Dict[str, Any]]:
    """
    Get a list of all registered tools.
    
    Returns:
        List of tool information dictionaries
    """
    return _registry.list_tools()


# Convenience function for getting a specific tool
def get_tool(name: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a specific tool.
    
    Args:
        name: Name of the tool
        
    Returns:
        Tool information or None if not found
    """
    return _registry.get_tool(name)
