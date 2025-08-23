#!/usr/bin/env python3
"""
Contact Resolver Module

This module provides functionality to resolve phone numbers and email addresses
to contact names using the macOS Contacts database.
"""

import asyncio
import logging
import re
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Tuple

logger = logging.getLogger(__name__)

class ContactResolver:
    """
    Resolves contact names from macOS Contacts app.
    
    This class provides functionality to look up contact names from
    phone numbers and email addresses using the macOS AddressBook database.
    """
    
    def __init__(self):
        """Initialize the contact resolver."""
        self.initialized = False
        self.cache = {}
        self.cache_expiry = {}
        self.cache_ttl = timedelta(hours=1)  # Cache for 1 hour
        self.contacts_db_path = None
    
    async def initialize(self):
        """Initialize the resolver."""
        self.contacts_db_path = self._find_contacts_db()
        self.initialized = True
        return self
    
    def _find_contacts_db(self) -> Optional[Path]:
        """Find the macOS Contacts database."""
        possible_paths = [
            Path.home() / "Library/Application Support/AddressBook/Sources",
            Path.home() / "Library/Application Support/AddressBook/AddressBook-v22.abcddb",
        ]
        
        # Look for the most recent .abcddb file
        for base_path in possible_paths:
            if base_path.exists():
                if base_path.is_file() and base_path.suffix == '.abcddb':
                    return base_path
                elif base_path.is_dir():
                    # Search for .abcddb files in subdirectories
                    for db_file in base_path.glob("**/*.abcddb"):
                        return db_file
        
        logger.debug("Could not find macOS Contacts database")
        return None
    
    def normalize_identifier(self, identifier: str) -> str:
        """
        Normalize phone numbers and email addresses to standard format.
        
        Args:
            identifier: Phone number or email in any format
            
        Returns:
            Normalized identifier
        """
        if not identifier:
            return identifier
            
        # Clean the identifier
        identifier = identifier.strip()
        
        # Check if it's an email
        if '@' in identifier:
            return identifier.lower()
        
        # It's a phone number - remove all non-digits
        digits = re.sub(r'\D', '', identifier)
        
        # Handle different phone formats
        if len(digits) == 10:  # US number without country code
            return f"+1{digits}"
        elif len(digits) == 11 and digits.startswith('1'):  # US number with 1
            return f"+{digits}"
        elif len(digits) > 10:  # International number
            return f"+{digits}"
        else:
            # Return original if we can't parse it
            return identifier
    
    async def get_contact_by_identifier(self, identifier: str) -> Dict[str, Any]:
        """
        Get contact information by phone number or email.
        
        Args:
            identifier: Phone number or email to look up
            
        Returns:
            Dictionary with contact information or None if not found
        """
        if not self.initialized:
            await self.initialize()
            
        # Normalize the identifier
        normalized = self.normalize_identifier(identifier)
        
        # Check cache first
        if normalized in self.cache:
            expiry = self.cache_expiry.get(normalized)
            if expiry and datetime.now() < expiry:
                return self.cache[normalized]
        
        # Query Contacts database
        contact_info = self._query_contacts_db(normalized)
        
        # If no contact found, create a formatted fallback
        if not contact_info:
            contact_info = {
                "display_name": self.format_identifier_fallback(identifier),
                "identifier": normalized,
                "first_name": None,
                "last_name": None,
                "organization": None,
                "is_placeholder": True
            }
        else:
            contact_info["identifier"] = normalized
            contact_info["is_placeholder"] = False
        
        # Cache the result
        self.cache[normalized] = contact_info
        self.cache_expiry[normalized] = datetime.now() + self.cache_ttl
        
        return contact_info
    
    def _query_contacts_db(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Query the Contacts database for a specific identifier."""
        if not self.contacts_db_path or not self.contacts_db_path.exists():
            return None
            
        try:
            # Connect to the database
            conn = sqlite3.connect(str(self.contacts_db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Try to find by phone number
            if not '@' in identifier:
                # Remove + and country code for matching
                phone_digits = re.sub(r'\D', '', identifier)
                if phone_digits.startswith('1') and len(phone_digits) == 11:
                    phone_digits = phone_digits[1:]  # Remove US country code
                
                query = """
                SELECT DISTINCT
                    r.ZFIRSTNAME as first_name,
                    r.ZLASTNAME as last_name,
                    r.ZORGANIZATION as organization,
                    r.ZNICKNAME as nickname,
                    r.ZJOBTITLE as job_title,
                    r.ZDEPARTMENT as department
                FROM ZABCDPHONENUMBER p
                JOIN ZABCDRECORD r ON p.ZOWNER = r.Z_PK
                WHERE REPLACE(REPLACE(REPLACE(REPLACE(p.ZFULLNUMBER, '+', ''), '-', ''), ' ', ''), '(', '') LIKE ?
                LIMIT 1
                """
                cursor.execute(query, [f"%{phone_digits}%"])
            else:
                # Search by email
                query = """
                SELECT DISTINCT
                    r.ZFIRSTNAME as first_name,
                    r.ZLASTNAME as last_name,
                    r.ZORGANIZATION as organization,
                    r.ZNICKNAME as nickname,
                    r.ZJOBTITLE as job_title,
                    r.ZDEPARTMENT as department
                FROM ZABCDEMAILADDRESS e
                JOIN ZABCDRECORD r ON e.ZOWNER = r.Z_PK
                WHERE LOWER(e.ZADDRESS) = LOWER(?)
                LIMIT 1
                """
                cursor.execute(query, [identifier])
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                # Build contact info
                contact = {
                    'first_name': row['first_name'],
                    'last_name': row['last_name'],
                    'organization': row['organization'],
                    'nickname': row['nickname'],
                    'job_title': row['job_title'],
                    'department': row['department']
                }
                
                # Generate display name
                if contact['first_name'] or contact['last_name']:
                    display_name = f"{contact['first_name'] or ''} {contact['last_name'] or ''}".strip()
                elif contact['organization']:
                    display_name = contact['organization']
                elif contact['nickname']:
                    display_name = contact['nickname']
                else:
                    display_name = None
                
                contact['display_name'] = display_name
                return contact
                
        except Exception as e:
            logger.debug(f"Error querying Contacts database: {e}")
            
        return None
    
    def format_identifier_fallback(self, identifier: str) -> str:
        """
        Format identifier as fallback when no contact name is found.
        
        Args:
            identifier: Phone number or email
            
        Returns:
            Formatted identifier for display
        """
        if '@' in identifier:
            # Email - return as is
            return identifier
        
        # Phone number - format nicely
        digits = re.sub(r'\D', '', identifier)
        
        # US number formatting
        if len(digits) == 10:
            return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        elif len(digits) == 11 and digits.startswith('1'):
            return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
        else:
            # Return original for international numbers
            return identifier
        
    async def search_contacts(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search contacts by name or identifier.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of matching contacts
        """
        if not self.initialized:
            await self.initialize()
            
        # For testing, create placeholder results
        results = []
        for i in range(min(3, limit)):
            results.append({
                "name": f"Search Result {i+1} for '{query}'",
                "phone": f"+1555{i:04d}",
                "email": f"contact{i+1}@example.com",
                "is_placeholder": True
            })
            
        return results

class ContactResolverFactory:
    """Factory for creating contact resolvers."""
    
    @staticmethod
    async def create_async_resolver() -> ContactResolver:
        """
        Create an async contact resolver instance.
        
        Returns:
            ContactResolver instance
        """
        resolver = ContactResolver()
        await resolver.initialize()
        return resolver
