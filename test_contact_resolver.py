#!/usr/bin/env python3
# Test script to verify contact resolver functionality

import asyncio
import logging
import sys
from src.utils.contact_resolver import ContactResolverFactory

# Set up logging to see what's happening
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock database for testing
class MockDatabase:
    def __init__(self):
        self.data = {
            "+1234567890": "Test Contact",
            "email@example.com": "Email Contact"
        }
    
    def execute_query(self, query, params=None):
        if not params or not params[0] in self.data:
            return []
        
        # Return mock data for testing
        return [(self.data[params[0]],)]
    
    async def execute_query_async(self, query, params=None):
        if not params or not params[0] in self.data:
            return []
        
        # Return mock data for testing
        return [(self.data[params[0]],)]

async def main():
    try:
        logger.info("Starting contact resolver test")
        
        # Create a mock database
        db = MockDatabase()
        
        # Create a contact resolver
        resolver = ContactResolverFactory.create_resolver(db, force_database_only=True)
        logger.info(f"Created resolver: {type(resolver).__name__}")
        
        # Test synchronous methods
        logger.info("Testing synchronous methods...")
        
        # Test resolve_contact
        contact = resolver.resolve_contact("+1234567890")
        logger.info(f"Resolved contact: {contact}")
        
        # Test format_display_name
        display_name = resolver.format_display_name("+1234567890")
        logger.info(f"Display name: {display_name}")
        
        # Test async methods
        logger.info("Testing asynchronous methods...")
        
        # Test resolve_contact_async
        contact_async = await resolver.resolve_contact_async("+1234567890")
        logger.info(f"Async resolved contact: {contact_async}")
        
        # Test format_display_name_async
        display_name_async = await resolver.format_display_name_async("+1234567890")
        logger.info(f"Async display name: {display_name_async}")
        
        # Test get_contact_image
        contact_image = resolver.get_contact_image("+1234567890")
        logger.info(f"Contact image available: {contact_image is not None}")
        
        logger.info("All tests completed successfully!")
        return 0
    
    except Exception as e:
        logger.error(f"Error in test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))