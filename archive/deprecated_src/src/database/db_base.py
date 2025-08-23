#!/usr/bin/env python3
"""
Abstract Database Base Classes

This module provides abstract base classes for the database layer,
defining interfaces that all database implementations must conform to.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, Tuple, AsyncGenerator
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager


class DatabaseError(Exception):
    """Base exception for database-related errors."""
    pass


class AsyncMessagesDBBase(ABC):
    """
    Abstract base class for async message database implementations.
    
    This class defines the interface that all database implementations
    must implement, ensuring consistent behavior regardless of the
    specific implementation used (standard, sharded, etc.).
    """
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the database connection.
        
        This method must be called before any other methods.
        Implementations should establish database connections,
        initialize caches, and perform any other setup needed.
        
        Raises:
            DatabaseError: If database initialization fails
        """
        pass
        
    @abstractmethod
    async def close(self) -> None:
        """
        Close all database connections.
        
        Implementations should ensure all resources are properly
        released, including database connections and caches.
        """
        pass
    
    @abstractmethod
    @asynccontextmanager
    async def get_db_connection(self):
        """
        Get a database connection from the pool.
        
        This context manager provides a database connection that
        is automatically returned to the pool when the context exits.
        
        Yields:
            A database connection
            
        Raises:
            DatabaseError: If a connection cannot be obtained
        """
        yield None
    
    @abstractmethod
    async def get_contacts(self, limit: int = 100, offset: int = 0, minimal: Optional[bool] = None) -> Dict[str, Any]:
        """
        Get a list of all contacts from the iMessage database.
        
        Args:
            limit: Maximum number of contacts to return
            offset: Number of contacts to skip
            minimal: Whether to return minimal data
            
        Returns:
            Dictionary with contacts list and metadata
        """
        pass
    
    @abstractmethod
    async def get_group_chats(self, limit: int = 20, offset: int = 0) -> Dict[str, Any]:
        """
        Get a list of all group chats.
        
        Args:
            limit: Maximum number of group chats to return
            offset: Number of group chats to skip
            
        Returns:
            Dictionary with group chats list and metadata
        """
        pass
    
    @abstractmethod
    async def get_chat_by_id(self, chat_id: Union[str, int]) -> Dict[str, Any]:
        """
        Get information about a specific chat by its ID.
        
        Args:
            chat_id: The chat ID
            
        Returns:
            Dictionary with chat information
        """
        pass
    
    @abstractmethod
    async def validate_schema_version(self) -> bool:
        """
        Check if the database schema is compatible with this application.
        
        Returns:
            True if schema is compatible, False otherwise
        """
        pass
    
    @abstractmethod
    async def optimize_database(self) -> Dict[str, Any]:
        """
        Perform database optimizations.
        
        This might include creating indexes, updating statistics,
        or other operations to improve query performance.
        
        Returns:
            Dictionary with optimization results
        """
        pass
    
    @abstractmethod
    async def get_messages_from_chat(
        self, 
        chat_id: Union[str, int], 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        search_term: Optional[str] = None,
        page: int = 1,
        page_size: int = 50,
        order_by: str = "date DESC"
    ) -> Dict[str, Any]:
        """
        Get messages from a specific chat.
        
        Args:
            chat_id: The ID of the chat
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            search_term: Optional search term for filtering
            page: Page number for pagination
            page_size: Number of messages per page
            order_by: Field and direction to sort by
            
        Returns:
            Dictionary with messages and metadata
        """
        pass
    
    @abstractmethod
    async def get_messages_from_contact(
        self, 
        contact_id: Union[str, int], 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        search_term: Optional[str] = None,
        page: int = 1,
        page_size: int = 50,
        order_by: str = "date DESC"
    ) -> Dict[str, Any]:
        """
        Get messages from a specific contact.
        
        Args:
            contact_id: The ID or phone number of the contact
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            search_term: Optional search term for filtering
            page: Page number for pagination
            page_size: Number of messages per page
            order_by: Field and direction to sort by
            
        Returns:
            Dictionary with messages and metadata
        """
        pass
    
    @abstractmethod
    async def search_messages(
        self, 
        query: str,
        contact_id: Optional[Union[str, int]] = None,
        chat_id: Optional[Union[str, int]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, Any]:
        """
        Search messages across all conversations or a specific contact/chat.
        
        Args:
            query: Search query
            contact_id: Optional contact to restrict search to
            chat_id: Optional chat to restrict search to
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            page: Page number for pagination
            page_size: Number of messages per page
            
        Returns:
            Dictionary with search results
        """
        pass
    
    @abstractmethod
    async def get_contact_by_phone_or_email(self, identifier: str) -> Dict[str, Any]:
        """
        Get contact information by phone number or email.
        
        Args:
            identifier: Phone number or email to look up
            
        Returns:
            Dictionary with contact information
        """
        pass

    async def should_use_sharding(self) -> bool:
        """
        Determine if database sharding should be used based on database size.
        
        Returns:
            True if sharding is recommended, False otherwise
        """
        return False  # Default implementation
    
    async def get_shards_info(self) -> Dict[str, Any]:
        """
        Get information about database shards (if applicable).
        
        Returns:
            Dictionary with shard information
        """
        return {  # Default implementation for non-sharded databases
            "using_shards": False,
            "shard_count": 0,
            "shards": [],
        }
