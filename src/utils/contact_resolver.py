import os
import re
import logging
import threading
import phonenumbers
import traceback
import platform
import asyncio
import time
from typing import Dict, Any, Optional, List, Set, Type, Union
from abc import ABC, abstractmethod

# Configure logging
logger = logging.getLogger(__name__)

# Flag to detect if we're running on macOS with Contacts framework
HAS_CONTACTS_FRAMEWORK = False
try:
    import contacts
    from Foundation import NSCompoundPredicate
    HAS_CONTACTS_FRAMEWORK = True
except ImportError:
    pass

# Base abstract class for contact resolvers
class ContactResolverBase(ABC):
    """
    Abstract base class for contact resolvers.
    
    This allows for multiple implementations that can be swapped based on
    platform availability or user preferences.
    """
    
    @abstractmethod
    def resolve_contact(self, identifier):
        """Resolve contact identifier to contact information."""
        pass
        
    @abstractmethod
    async def resolve_contact_async(self, identifier):
        """Async version: Resolve contact identifier to contact information."""
        pass
        
    @abstractmethod
    def format_display_name(self, identifier, contact_name=None, display_format=None, privacy_level="normal"):
        """Format a contact identifier for display with optional privacy controls."""
        pass
        
    @abstractmethod
    async def format_display_name_async(self, identifier, contact_name=None, display_format=None, privacy_level="normal"):
        """Async version: Format a contact identifier for display with optional privacy controls."""
        pass
        
    @abstractmethod
    def get_first_name(self, identifier):
        """Get first name for a contact identifier."""
        pass
        
    @abstractmethod
    def get_message_sender_info(self, message):
        """Get sender info for a message."""
        pass
        
    @abstractmethod
    def resolve_chat_name(self, chat_data, participants=None):
        """Generate a meaningful name for a chat."""
        pass
        
    @abstractmethod
    def resolve_group_chat_names(self, chats):
        """Resolve names for a list of group chats."""
        pass
        
    @abstractmethod
    def get_contact_image(self, identifier):
        """Get contact image if available."""
        pass

class DatabaseOnlyContactResolver(ContactResolverBase):
    """
    Basic contact resolver that only uses the Messages database.
    
    This implementation is platform-independent and doesn't require
    any system-specific contact frameworks.
    """
    
    def __init__(self, db, max_cache_size=1000):
        """Initialize with database connection."""
        self.db = db
        self.contact_cache = {}
        self.cache_lock = threading.RLock()
        self.email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        
        # Add cache management
        self.max_cache_size = max_cache_size
        self.cache_access_times = {}
        self.privacy_settings = {
            "default_level": "normal",  # Options: high, normal, minimal
            "show_full_phone": False,   # Whether to show full phone numbers 
            "show_emails": True         # Whether to show email addresses
        }
    
    def _is_email(self, identifier):
        """Check if the identifier is an email address."""
        if not identifier:
            return False
        return bool(self.email_pattern.match(identifier))
    
    def _is_phone_number(self, identifier):
        """Check if the identifier is a phone number."""
        if not identifier:
            return False
        # Remove common non-digit characters
        digits_only = re.sub(r'[^0-9+]', '', identifier)
        # Basic heuristic: if it has more than 7 digits, treat as phone number
        return len(digits_only) >= 7 and not self._is_email(identifier)
    
    def _format_phone_number(self, phone_number):
        """Format a phone number for better display."""
        if not phone_number:
            return None
            
        try:
            # Better phone number formatting with international support
            digits_only = re.sub(r'[^0-9+]', '', phone_number)
            
            # Format based on phone number pattern
            if len(digits_only) == 10:  # US format without country code
                return f"({digits_only[:3]}) {digits_only[3:6]}-{digits_only[6:]}"
            elif len(digits_only) == 11 and digits_only[0] == '1':  # US with country code
                return f"+1 ({digits_only[1:4]}) {digits_only[4:7]}-{digits_only[7:]}"
            elif digits_only.startswith('+'):
                # Try to use phonenumbers library if available
                try:
                    parsed = phonenumbers.parse(phone_number)
                    formatted = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
                    return formatted
                except Exception:
                    # If parsing fails, return with minimal formatting
                    return phone_number
            return phone_number
        except Exception:
            return phone_number
    
    def normalize_phone_number(self, number):
        """Normalize phone number for consistent comparison."""
        if not number:
            return None
            
        # Remove all non-digit characters
        digits_only = re.sub(r'\D', '', number)
        
        # Ensure number has country code
        if len(digits_only) == 10:  # US number without country code
            return '+1' + digits_only
        elif digits_only.startswith('1') and len(digits_only) == 11:  # US number with '1' prefix
            return '+' + digits_only
        elif digits_only.startswith('+'):
            return digits_only
        else:
            return '+' + digits_only
    
    def resolve_contact(self, identifier):
        """Resolve an identifier using only database information."""
        if not identifier:
            return None
            
        # Check cache
        with self.cache_lock:
            if identifier in self.contact_cache:
                # Update access time for LRU tracking
                self.cache_access_times[identifier] = time.time()
                return self.contact_cache[identifier]
                
        # Query database if available
        result = None
        if self.db:
            try:
                query = """
                SELECT DISTINCT c.display_name
                FROM chat c
                JOIN chat_handle_join chj ON c.ROWID = chj.chat_id
                JOIN handle h ON chj.handle_id = h.ROWID
                WHERE h.id = ? AND c.display_name IS NOT NULL AND c.display_name != ''
                LIMIT 1
                """
                db_result = self.db.execute_query(query, (identifier,))
                if db_result and db_result[0][0]:
                    result = {"name": db_result[0][0]}
            except Exception as e:
                logger.debug(f"Error in database lookup: {e}")
                
        # Cache result (even if None)
        with self.cache_lock:
            # Check cache size before adding new entry
            if len(self.contact_cache) >= self.max_cache_size:
                self._evict_cache_entries()
                
            self.contact_cache[identifier] = result
            self.cache_access_times[identifier] = time.time()
            
        return result
        
    async def resolve_contact_async(self, identifier):
        """Async version of resolve_contact."""
        if not identifier:
            return None
            
        # Check cache
        async with asyncio.Lock():
            if identifier in self.contact_cache:
                # Update access time for LRU tracking
                self.cache_access_times[identifier] = time.time()
                return self.contact_cache[identifier]
                
        # Query database if available
        result = None
        if self.db:
            try:
                # Use the async database method if available
                if hasattr(self.db, 'execute_query_async'):
                    query = """
                    SELECT DISTINCT c.display_name
                    FROM chat c
                    JOIN chat_handle_join chj ON c.ROWID = chj.chat_id
                    JOIN handle h ON chj.handle_id = h.ROWID
                    WHERE h.id = ? AND c.display_name IS NOT NULL AND c.display_name != ''
                    LIMIT 1
                    """
                    db_result = await self.db.execute_query_async(query, (identifier,))
                    if db_result and db_result[0][0]:
                        result = {"name": db_result[0][0]}
                else:
                    # Fallback to sync method in a thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, self.resolve_contact, identifier)
                    return result
            except Exception as e:
                logger.debug(f"Error in async database lookup: {e}")
                
        # Cache result (even if None)
        async with asyncio.Lock():
            # Check cache size before adding new entry
            if len(self.contact_cache) >= self.max_cache_size:
                self._evict_cache_entries()
                
            self.contact_cache[identifier] = result
            self.cache_access_times[identifier] = time.time()
            
        return result
    
    def _evict_cache_entries(self):
        """Evict oldest entries from cache when it gets too large."""
        if not self.cache_access_times:
            return
            
        # Remove 10% of oldest entries
        num_to_remove = max(1, len(self.contact_cache) // 10)
        oldest_keys = sorted(self.cache_access_times, key=self.cache_access_times.get)[:num_to_remove]
        
        for key in oldest_keys:
            if key in self.contact_cache:
                del self.contact_cache[key]
            if key in self.cache_access_times:
                del self.cache_access_times[key]
    
    def format_display_name(self, identifier, contact_name=None, display_format=None, privacy_level="normal"):
        """Format an identifier for display."""
        if not identifier:
            return "Unknown"
            
        # Use provided name if available
        if contact_name:
            return contact_name
            
        # Try to resolve from cache/database
        contact_info = self.resolve_contact(identifier)
        if contact_info and "name" in contact_info and contact_info["name"]:
            return contact_info["name"]
            
        # Format based on type
        if self._is_phone_number(identifier):
            return self._format_phone_number(identifier)
        elif self._is_email(identifier):
            return identifier
        else:
            return identifier
            
    async def format_display_name_async(self, identifier, contact_name=None, display_format=None, privacy_level="normal"):
        """Async version of format_display_name."""
        if not identifier:
            return "Unknown"
            
        # Use provided name if available
        if contact_name:
            return contact_name
            
        # Try to resolve from cache/database
        contact_info = await self.resolve_contact_async(identifier)
        if contact_info and "name" in contact_info and contact_info["name"]:
            return contact_info["name"]
            
        # Format based on type
        if self._is_phone_number(identifier):
            return self._format_phone_number(identifier)
        elif self._is_email(identifier):
            return identifier
        else:
            return identifier
    
    def get_first_name(self, identifier):
        """Get first name for a contact."""
        contact_info = self.resolve_contact(identifier)
        if contact_info and "name" in contact_info:
            # Simple split by space
            return contact_info["name"].split()[0]
        return None
    
    def get_message_sender_info(self, message):
        """Get sender info for a message."""
        if not message:
            return {"display_name": "Unknown"}
            
        sender_id = message.get("sender_id") or message.get("contact_id")
        if not sender_id:
            return {"display_name": "Unknown"}
            
        display_name = self.format_display_name(sender_id)
        return {"display_name": display_name}
    
    def resolve_chat_name(self, chat_data, participants=None):
        """Generate a meaningful name for a chat."""
        if not chat_data:
            return chat_data
            
        # Copy to avoid modifying original
        chat = dict(chat_data)
        
        # Skip if already has a name
        if chat.get('display_name') and chat['display_name'] != "Unnamed Group":
            return chat
            
        # Set a default name based on chat ID
        chat['display_name'] = f"Chat #{chat.get('chat_id', 'Unknown')}"
        return chat
    
    def resolve_group_chat_names(self, chats):
        """Resolve names for multiple chats."""
        if not chats:
            return []
            
        return [self.resolve_chat_name(chat) for chat in chats]
        
    def get_contact_image(self, identifier):
        """Get contact image if available.
        
        In the database-only resolver, this always returns None as
        the Messages database doesn't store contact images.
        """
        return None

class MacOSContactResolver(ContactResolverBase):
    """Enhanced contact resolution using macOS Contacts framework.
    
    This class provides improved contact resolution for various identifier types
    (phone numbers, emails, iMessage IDs) with advanced caching and formatting.
    It specifically leverages macOS Contacts framework when available.
    """
    
    def __init__(self, db):
        """Initialize contact resolver with database connection.
        
        Args:
            db: MessagesDB instance for database access
        """
        self.db = db
        self.contact_cache = {}  # Cache for contact lookups
        self.cache_lock = threading.RLock()  # Reentrant lock for thread safety
        self.email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        
        # Cache for contacts framework results
        self.contacts_framework_cache = {}
        self.contacts_framework_cache_lock = threading.RLock()
        self.max_cache_size = 1000  # Limit cache size to prevent memory issues
        
        # Initialize contacts framework
        self.has_contacts_framework = HAS_CONTACTS_FRAMEWORK
        if self.has_contacts_framework:
            try:
                self.contact_store = contacts.CNContactStore.alloc().init()
                # Request access if needed
                if not self._check_contacts_access():
                    logger.warning("No access to Contacts. Please grant permission in System Settings.")
            except Exception as e:
                logger.error(f"Error initializing Contacts framework: {e}")
                self.has_contacts_framework = False
    
    def _check_contacts_access(self):
        """Check if we have access to Contacts framework."""
        try:
            authorization = self.contact_store.authorizationStatusForEntityType_(
                contacts.CNEntityTypeContacts
            )
            
            if authorization == contacts.CNAuthorizationStatusNotDetermined:
                # Request access
                result = []
                event = threading.Event()
                
                def completion_handler(granted, error):
                    result.append(granted)
                    event.set()
                
                self.contact_store.requestAccessForEntityType_completionHandler_(
                    contacts.CNEntityTypeContacts,
                    completion_handler
                )
                
                # Wait for completion handler to be called
                event.wait(timeout=5.0)  # 5 second timeout
                return result[0] if result else False
                
            return authorization == contacts.CNAuthorizationStatusAuthorized
        except Exception as e:
            logger.error(f"Error checking Contacts access: {e}")
            return False
    
    def _is_email(self, identifier):
        """Check if the identifier is an email address."""
        if not identifier:
            return False
        return bool(self.email_pattern.match(identifier))
    
    def _is_phone_number(self, identifier):
        """Check if the identifier is a phone number."""
        if not identifier:
            return False
        # Remove common non-digit characters
        digits_only = re.sub(r'[^0-9+]', '', identifier)
        # Basic heuristic: if it has more than 7 digits, treat as phone number
        return len(digits_only) >= 7 and not self._is_email(identifier)
    
    def _format_phone_number(self, phone_number):
        """Format a phone number for better display and matching."""
        if not phone_number:
            return None
        
        try:
            # Detect if the number already has a country code
            has_country_code = phone_number.startswith('+')
            
            # Try multiple regions for better international number support
            if not has_country_code:
                # Try to parse with different regions
                for region in ["US", "GB", "CA", "AU", "IN", "DE", "FR"]:
                    try:
                        parsed_number = phonenumbers.parse(phone_number, region)
                        if phonenumbers.is_valid_number(parsed_number):
                            break
                    except Exception:
                        continue
                else:
                    # If no valid parse found, default to US
                    parsed_number = phonenumbers.parse(phone_number, "US")
            else:
                # If it has a country code, we can parse directly
                parsed_number = phonenumbers.parse(phone_number)
                
            if phonenumbers.is_valid_number(parsed_number):
                # Get different formats to improve matching and display
                e164 = phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.E164)
                international = phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
                national = phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.NATIONAL)
                
                # Build a comprehensive set of search values for better matching
                search_values = {
                    phone_number, 
                    e164, 
                    international, 
                    national,
                    re.sub(r'[^0-9+]', '', phone_number)
                }
                
                # Add variants without country code
                if e164.startswith('+'):
                    # For different country codes
                    for i in range(1, 4):
                        if len(e164) > i+1:
                            search_values.add(e164[i+1:])
                
                return {
                    "original": phone_number,
                    "e164": e164,
                    "international": international,
                    "national": national,
                    "country_code": phonenumbers.region_code_for_number(parsed_number),
                    "search_values": search_values
                }
        except Exception as e:
            logger.debug(f"Error formatting phone number {phone_number}: {e}")
        
        # If parsing fails, just return the original with a basic cleanup
        digits_only = re.sub(r'[^0-9+]', '', phone_number)
        return {
            "original": phone_number,
            "e164": phone_number,
            "international": phone_number,
            "national": phone_number,
            "country_code": "unknown",
            "search_values": {phone_number, digits_only}
        }
    
    def _cache_contacts_framework_result(self, lookup_type, key, result):
        """Cache the result of a contacts framework lookup."""
        with self.contacts_framework_cache_lock:
            # Check if we need to evict old entries
            if len(self.contacts_framework_cache) >= self.max_cache_size:
                # Simple strategy: remove oldest (first) entry
                if self.contacts_framework_cache:
                    oldest_key = next(iter(self.contacts_framework_cache))
                    del self.contacts_framework_cache[oldest_key]
            
            # Add to cache
            cache_key = f"{lookup_type}:{key}"
            self.contacts_framework_cache[cache_key] = result

    def _get_cached_contacts_result(self, lookup_type, key):
        """Get a cached contacts framework result if available."""
        with self.contacts_framework_cache_lock:
            cache_key = f"{lookup_type}:{key}"
            return self.contacts_framework_cache.get(cache_key)

    def _lookup_contact_by_phone(self, phone_number):
        """Look up a contact by phone number in macOS Contacts."""
        if not self.has_contacts_framework or not phone_number:
            return None
        
        # Check cache first
        cached_result = self._get_cached_contacts_result('phone', phone_number)
        if cached_result is not None:
            return cached_result
        
        formatted = self._format_phone_number(phone_number)
        if not formatted:
            return None
        
        try:
            # Create search predicate for phone number
            search_values = formatted["search_values"]
            predicates = []
            
            for value in search_values:
                # Skip empty values
                if not value:
                    continue
                # Create predicate for this phone value
                predicate = contacts.CNPhoneNumber.predicateForPhoneNumberMatchingStringWithLabels_(
                    value, None
                )
                predicates.append(predicate)
            
            if not predicates:
                # Cache negative result and return
                self._cache_contacts_framework_result('phone', phone_number, None)
                return None
                
            # Combine with OR predicate
            final_predicate = NSCompoundPredicate.orPredicateWithSubpredicates_(predicates)
            
            # Setup keys to fetch
            keys = [
                contacts.CNContactGivenNameKey, 
                contacts.CNContactFamilyNameKey,
                contacts.CNContactNicknameKey,
                contacts.CNContactOrganizationNameKey,
                contacts.CNContactPhoneNumbersKey
            ]
            
            # Perform search with proper error handling
            matching_contacts = self.contact_store.unifiedContactsMatchingPredicate_keysToFetch_error_(
                final_predicate, keys, None
            )
            
            if matching_contacts and len(matching_contacts) > 0:
                # Process the first matching contact
                contact = matching_contacts[0]
                given_name = contact.givenName() or ""
                family_name = contact.familyName() or ""
                nickname = contact.nickname() or ""
                organization = contact.organizationName() or ""
                
                # Build name based on available parts
                contact_name = ""
                if given_name or family_name:
                    contact_name = f"{given_name} {family_name}".strip()
                elif nickname:
                    contact_name = nickname
                elif organization:
                    contact_name = organization
                
                # Include additional data in the result
                result = {
                    "name": contact_name,
                    "given_name": given_name,
                    "family_name": family_name,
                    "nickname": nickname,
                    "organization": organization
                }
                
                # Cache the result
                self._cache_contacts_framework_result('phone', phone_number, result)
                return result
                
            # Cache negative result
            self._cache_contacts_framework_result('phone', phone_number, None)
            
        except Exception as e:
            logger.debug(f"Error looking up contact by phone {phone_number}: {e}")
            logger.debug(traceback.format_exc())
        
        return None
    
    def _lookup_contact_by_email(self, email):
        """Look up a contact by email address in macOS Contacts."""
        if not self.has_contacts_framework or not email:
            return None
            
        # Check cache first
        cached_result = self._get_cached_contacts_result('email', email)
        if cached_result is not None:
            return cached_result
        
        try:
            # Create search predicate for email
            predicate = contacts.CNContactEmailAddressesContainmentPredicate.predicateForEmailAddressWithStringValue_(email)
            
            # Setup keys to fetch
            keys = [
                contacts.CNContactGivenNameKey, 
                contacts.CNContactFamilyNameKey,
                contacts.CNContactNicknameKey,
                contacts.CNContactOrganizationNameKey,
                contacts.CNContactEmailAddressesKey
            ]
            
            # Perform search with proper error handling
            matching_contacts = self.contact_store.unifiedContactsMatchingPredicate_keysToFetch_error_(
                predicate, keys, None
            )
            
            if matching_contacts and len(matching_contacts) > 0:
                # Process the first matching contact
                contact = matching_contacts[0]
                given_name = contact.givenName() or ""
                family_name = contact.familyName() or ""
                nickname = contact.nickname() or ""
                organization = contact.organizationName() or ""
                
                # Build name based on available parts
                contact_name = ""
                if given_name or family_name:
                    contact_name = f"{given_name} {family_name}".strip()
                elif nickname:
                    contact_name = nickname
                elif organization:
                    contact_name = organization
                
                # Include additional data in the result
                result = {
                    "name": contact_name,
                    "given_name": given_name,
                    "family_name": family_name,
                    "nickname": nickname,
                    "organization": organization
                }
                
                # Cache the result
                self._cache_contacts_framework_result('email', email, result)
                return result
                
            # Cache negative result
            self._cache_contacts_framework_result('email', email, None)
            
        except Exception as e:
            logger.debug(f"Error looking up contact by email {email}: {e}")
            logger.debug(traceback.format_exc())
        
        return None
    
    def resolve_contact(self, identifier):
        """Resolve any type of contact identifier to a name.
        
        Args:
            identifier: Phone number, email, or other contact identifier
            
        Returns:
            Dict with contact information or None if no match
        """
        if not identifier:
            return None
            
        # Check if we have this in cache
        with self.cache_lock:
            if identifier in self.contact_cache:
                return self.contact_cache[identifier]
                
        # Determine identifier type and lookup strategy
        if self._is_phone_number(identifier):
            result = self._lookup_contact_by_phone(identifier)
        elif self._is_email(identifier):
            result = self._lookup_contact_by_email(identifier)
        else:
            # For other types, try to find in the database
            result = self._lookup_in_database(identifier)
            
        # Cache the result (even if None, to avoid repeated lookups)
        with self.cache_lock:
            self.contact_cache[identifier] = result
            
        return result
        
    async def resolve_contact_async(self, identifier):
        """Async version: Resolve any type of contact identifier to a name.
        
        Args:
            identifier: Phone number, email, or other contact identifier
            
        Returns:
            Dict with contact information or None if no match
        """
        if not identifier:
            return None
            
        # Check if we have this in cache
        async with asyncio.Lock():
            if identifier in self.contact_cache:
                return self.contact_cache[identifier]
        
        # Since Contacts framework is not async-friendly, run in a thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.resolve_contact, identifier)
        
        return result
    
    def _lookup_in_database(self, identifier):
        """Try to find a contact name in the Messages database."""
        if not self.db:
            return None
            
        # Check cache first
        cached_result = self._get_cached_contacts_result('db', identifier)
        if cached_result is not None:
            return cached_result
            
        try:
            # Check if this identifier exists in the database
            query = "SELECT id FROM handle WHERE id = ?"
            result = self.db.execute_query(query, (identifier,))
            
            if not result:
                # Cache negative result
                self._cache_contacts_framework_result('db', identifier, None)
                return None
                
            # Try to find a display name in the database
            query = """
            SELECT DISTINCT c.display_name
            FROM chat c
            JOIN chat_handle_join chj ON c.ROWID = chj.chat_id
            JOIN handle h ON chj.handle_id = h.ROWID
            WHERE h.id = ? AND c.display_name IS NOT NULL AND c.display_name != ''
            LIMIT 1
            """
            result = self.db.execute_query(query, (identifier,))
            
            if result and result[0][0]:
                # Found a display name
                db_result = {"name": result[0][0]}
                # Cache the positive result
                self._cache_contacts_framework_result('db', identifier, db_result)
                return db_result
            
            # No display name found, try to find any other information
            query = """
            SELECT h.id, h.service, h.uncanonicalized_id
            FROM handle h
            WHERE h.id = ?
            LIMIT 1
            """
            result = self.db.execute_query(query, (identifier,))
            
            if result:
                # Create a minimal contact record with just the identifier
                service = result[0][1] if len(result[0]) > 1 else "unknown"
                db_result = {
                    "name": identifier,
                    "service": service
                }
                # Cache the result
                self._cache_contacts_framework_result('db', identifier, db_result)
                return db_result
                
            # Cache negative result
            self._cache_contacts_framework_result('db', identifier, None)
            return None
            
        except Exception as e:
            logger.debug(f"Error looking up contact in database {identifier}: {e}")
            logger.debug(traceback.format_exc())
            return None
            
    def _extract_first_name(self, contact_info):
        """Extract the first name from contact info."""
        if not contact_info:
            return None
            
        if isinstance(contact_info, dict):
            # Try to get the first name from structured data
            if "given_name" in contact_info and contact_info["given_name"]:
                return contact_info["given_name"]
                
            # Fall back to splitting the full name
            if "name" in contact_info and contact_info["name"]:
                # Split by whitespace and take the first part
                return contact_info["name"].split()[0]
                
        elif isinstance(contact_info, str):
            # If it's just a string, split by whitespace
            return contact_info.split()[0]
            
        return None
            
    def format_display_name(self, identifier, contact_name=None, display_format=None, privacy_level="normal"):
        """Format a contact identifier for display.
        
        Args:
            identifier: The raw identifier (phone, email, etc.)
            contact_name: Optional contact name if already known
            display_format: Optional format specification
            privacy_level: Level of privacy to apply (normal, high, minimal)
            
        Returns:
            Formatted display name
        """
        if not identifier:
            return "Unknown"
            
        # If we already have a contact name, use it
        if contact_name:
            return contact_name
            
        # Try to resolve the contact
        contact_info = self.resolve_contact(identifier)
        if contact_info and "name" in contact_info and contact_info["name"]:
            return contact_info["name"]
            
        # Format based on identifier type
        if self._is_phone_number(identifier):
            phone_formatted = self._format_phone_number(identifier)
            if phone_formatted and isinstance(phone_formatted, dict):
                return phone_formatted.get("national", identifier)
            return identifier
        elif self._is_email(identifier):
            return identifier
        else:
            return identifier
            
    async def format_display_name_async(self, identifier, contact_name=None, display_format=None, privacy_level="normal"):
        """Async version of format_display_name.
        
        Args:
            identifier: The raw identifier (phone, email, etc.)
            contact_name: Optional contact name if already known
            display_format: Optional format specification
            privacy_level: Level of privacy to apply (normal, high, minimal)
            
        Returns:
            Formatted display name
        """
        if not identifier:
            return "Unknown"
            
        # If we already have a contact name, use it
        if contact_name:
            return contact_name
            
        # Try to resolve the contact asynchronously
        contact_info = await self.resolve_contact_async(identifier)
        if contact_info and "name" in contact_info and contact_info["name"]:
            return contact_info["name"]
            
        # Format based on identifier type
        if self._is_phone_number(identifier):
            phone_formatted = self._format_phone_number(identifier)
            if phone_formatted and isinstance(phone_formatted, dict):
                return phone_formatted.get("national", identifier)
            return identifier
        elif self._is_email(identifier):
            return identifier
        else:
            return identifier
            
    def get_first_name(self, identifier):
        """Get the first name for a contact identifier.
        
        Args:
            identifier: Contact identifier (phone, email, etc.)
            
        Returns:
            First name or None if not available
        """
        contact_info = self.resolve_contact(identifier)
        return self._extract_first_name(contact_info)
            
    def get_message_sender_info(self, message):
        """Get sender info for a message.
        
        Args:
            message: Message dictionary
            
        Returns:
            Dictionary with sender display name
        """
        if not message:
            return {"display_name": "Unknown"}
            
        sender_id = message.get("sender_id") or message.get("contact_id")
        if not sender_id:
            return {"display_name": "Unknown"}
            
        # Try to get sender name
        contact_info = self.resolve_contact(sender_id)
        if contact_info and "name" in contact_info:
            return {
                "display_name": contact_info["name"],
                "contact_info": contact_info
            }
            
        # Format the identifier for display
        return {"display_name": self.format_display_name(sender_id)}

    def normalize_phone_number(self, number):
        """Normalize phone number for consistent comparison.
        
        Args:
            number: Phone number string
            
        Returns:
            Normalized phone number
        """
        if not number:
            return None
            
        try:
            # Try to parse with phonenumbers for accurate normalization
            has_country_code = number.startswith('+')
            
            if has_country_code:
                parsed = phonenumbers.parse(number)
            else:
                # Try with default US region first
                try:
                    parsed = phonenumbers.parse(number, "US")
                except:
                    # If that fails, try with a general approach
                    return self._simple_normalize_phone(number)
                    
            if phonenumbers.is_valid_number(parsed):
                return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
                
            # Fall back to simple normalization
            return self._simple_normalize_phone(number)
            
        except Exception as e:
            logger.debug(f"Error normalizing phone number {number}: {e}")
            return self._simple_normalize_phone(number)
            
    def _simple_normalize_phone(self, number):
        """Simpler phone normalization as fallback."""
        # Remove all non-digit characters
        digits_only = re.sub(r'\D', '', number)
        
        # Ensure number has country code
        if len(digits_only) == 10:  # US number without country code
            return '+1' + digits_only
        elif digits_only.startswith('1') and len(digits_only) == 11:  # US number with '1' prefix
            return '+' + digits_only
        elif digits_only.startswith('+'):
            return digits_only
        else:
            return '+' + digits_only

    def resolve_chat_name(self, chat_data, participants=None):
        """Generate a meaningful name for a group chat based on participants.
        
        Args:
            chat_data: Chat data dictionary
            participants: Optional list of participants
            
        Returns:
            Updated chat data with resolved name
        """
        if not chat_data:
            return chat_data
            
        # Make a copy to avoid modifying original
        chat = dict(chat_data)
        
        # Skip if chat already has a name
        if chat.get('display_name') and chat['display_name'] != "Unnamed Group":
            return chat
            
        # Get participants if not provided
        if not participants and chat.get('chat_id'):
            if self.db:
                # Try to get participants from database
                try:
                    query = """
                    SELECT h.id
                    FROM handle h
                    JOIN chat_handle_join chj ON h.ROWID = chj.handle_id
                    WHERE chj.chat_id = ?
                    """
                    results = self.db.execute_query(query, (chat['chat_id'],))
                    if results:
                        participants = [{'contact_id': r[0]} for r in results]
                except Exception as e:
                    logger.debug(f"Error getting participants for chat {chat['chat_id']}: {e}")
            
        # Skip if we still don't have participants
        if not participants:
            chat['display_name'] = f"Group Chat #{chat.get('chat_id', 'Unknown')}"
            return chat
            
        # Get first names of participants
        first_names = []
        for p in participants:
            contact_id = p.get('contact_id')
            if not contact_id:
                continue
                
            # Try to get first name
            first_name = self.get_first_name(contact_id)
            if first_name:
                first_names.append(first_name)
                
        # Create a name based on participants (up to 3)
        if first_names:
            if len(first_names) == 2:
                chat['display_name'] = f"{first_names[0]} & {first_names[1]}"
            elif len(first_names) >= 3:
                chat['display_name'] = f"{first_names[0]}, {first_names[1]} & {len(first_names)-2} more"
            else:
                chat['display_name'] = f"Chat with {first_names[0]}"
        else:
            # Fall back to chat ID if no names can be resolved
            chat['display_name'] = f"Group Chat #{chat.get('chat_id', 'Unknown')}"
            
        return chat
        
    def resolve_group_chat_names(self, chats):
        """Resolve display names for a list of group chats.
        
        Args:
            chats: List of chat dictionaries
            
        Returns:
            List of chat dictionaries with resolved names
        """
        if not chats:
            return []
            
        resolved_chats = []
        for chat in chats:
            resolved = self.resolve_chat_name(chat)
            resolved_chats.append(resolved)
            
        return resolved_chats
        
    def get_contact_image(self, identifier):
        """Get contact image if available from the macOS Contacts framework.
        
        Args:
            identifier: Contact identifier (phone, email, etc.)
            
        Returns:
            Base64-encoded image data or None if not available
        """
        if not self.has_contacts_framework or not identifier:
            return None
            
        try:
            # Check if this is a phone number or email
            if self._is_phone_number(identifier):
                # Look up by phone number
                contact_info = self._lookup_contact_by_phone(identifier)
            elif self._is_email(identifier):
                # Look up by email
                contact_info = self._lookup_contact_by_email(identifier)
            else:
                # For other types, try database lookup first
                contact_info = self._lookup_in_database(identifier)
                
            if not contact_info:
                return None
                
            # At this point, we would need to query the Contacts framework for the image
            # This is a placeholder implementation - in a real implementation we would:
            # 1. Query the Contacts framework for the contact's image data
            # 2. Convert the image data to base64 for transmission
            # 3. Return the base64-encoded image
            
            logger.debug(f"Contact image requested for {identifier}, but not implemented yet")
            return None
            
        except Exception as e:
            logger.debug(f"Error retrieving contact image for {identifier}: {e}")
            logger.debug(traceback.format_exc())
            return None


class ContactResolverFactory:
    """Factory for creating the appropriate contact resolver based on platform."""
    
    @staticmethod
    def create_resolver(db, force_database_only=False):
        """
        Create and return an appropriate contact resolver instance.
        
        Args:
            db: Database connection to use
            force_database_only: If True, use database-only resolver even on macOS
            
        Returns:
            A ContactResolverBase implementation
        """
        # Check environment variables for override
        env_override = os.environ.get("FORCE_DB_ONLY_RESOLVER", "").lower()
        if env_override in ["1", "true", "yes"]:
            force_database_only = True
        
        # Check platform and availability
        if platform.system() == "Darwin" and HAS_CONTACTS_FRAMEWORK and not force_database_only:
            logger.info("Using macOS Contacts framework for contact resolution")
            return MacOSContactResolver(db)
        else:
            if platform.system() == "Darwin" and not HAS_CONTACTS_FRAMEWORK:
                logger.info("macOS Contacts framework not available, using database-only resolver")
            else:
                logger.info(f"Using database-only contact resolver on {platform.system()}")
            return DatabaseOnlyContactResolver(db)
            

# Legacy compatibility - use EnhancedContactResolver as an alias
EnhancedContactResolver = MacOSContactResolver