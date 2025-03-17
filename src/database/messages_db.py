from pathlib import Path
import os
import sqlite3
import threading
import logging
from contextlib import contextmanager
import re
import phonenumbers
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple

# Configure logging
logger = logging.getLogger(__name__)

# Default database path for macOS
HOME = os.path.expanduser("~")
DB_PATH = Path(f"{HOME}/Library/Messages/chat.db")

# Flag to detect if we're running on macOS with Contacts framework
HAS_CONTACTS_FRAMEWORK = False
try:
    import contacts
    HAS_CONTACTS_FRAMEWORK = True
except ImportError:
    pass


class MessagesDB:
    """A singleton class for handling the messages database.
    
    This class provides methods for interacting with the iMessage database,
    handling contacts, and extracting message data.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, db_path=DB_PATH):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MessagesDB, cls).__new__(cls)
                cls._instance.db_path = Path(db_path)
                cls._instance._initialized = False
                logger.info(f"Created MessagesDB instance with path: {db_path}")
            return cls._instance
    
    def __init__(self, db_path=DB_PATH):
        """Initialize with the database path."""
        # Only initialize once
        if getattr(self, '_initialized', False):
            return
            
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Messages database not found at: {self.db_path}")
        # Create connection pool
        self.connection_pool = []
        for _ in range(5):  # Create 5 connections in the pool
            self.connection_pool.append(sqlite3.connect(str(self.db_path)))
        self.connection_lock = threading.Lock()
        # Add a cache for contact names
        self.contact_name_cache = {}
        self.cache_lock = threading.Lock()
        
        # Initialize enhanced contact resolver
        self.contact_resolver = self.initialize_contact_resolver()
        
        # Mark as initialized
        self._initialized = True
    
    def initialize_contact_resolver(self):
        """Initialize the enhanced contact resolver."""
        try:
            from src.utils.contact_resolver import EnhancedContactResolver
            resolver = EnhancedContactResolver(self)
            logger.info("Enhanced contact resolver initialized")
            return resolver
        except ImportError as e:
            logger.warning(f"Enhanced contact resolver not available: {e}")
            return None
    
    def get_connection(self):
        """Get a connection from the pool or create a new one."""
        with self.connection_lock:
            if self.connection_pool:
                return self.connection_pool.pop()
                
            logger.debug(f"Creating new DB connection to {self.db_path}")
            return sqlite3.connect(self.db_path)
    
    def release_connection(self, conn):
        """Return a connection to the pool."""
        with self.connection_lock:
            self.connection_pool.append(conn)
            
    @contextmanager
    def get_db_connection(self):
        """Context manager for database connections."""
        conn = self.get_connection()
        try:
            yield conn
        finally:
            self.release_connection(conn)
    
    def get_contact_name(self, phone_number):
        """Get contact name from phone number using the system contacts database.
        
        This method first checks if the phone number already has a contact name,
        then tries E.164 format, and finally tries a fallback to the Contacts database.
        
        Args:
            phone_number: The phone number to look up
            
        Returns:
            Contact name or None if not found
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            # Check for "+" prefix and try both with and without
            queries = [phone_number]
            if phone_number and not phone_number.startswith("+"):
                queries.append("+" + phone_number)
            elif phone_number and phone_number.startswith("+"):
                queries.append(phone_number[1:])
            
            # Try various phone formats
            for query in queries:
                cursor.execute(
                    """
                    SELECT id, ROWID 
                    FROM handle 
                    WHERE id = ?
                    """, 
                    [query]
                )
                result = cursor.fetchone()
                if result:
                    break
            
            if not result:
                return None
                
            handle_id = result[1]
            
            # Check if we already have a contact name for this handle
            cursor.execute(
                """
                SELECT id, ROWID 
                FROM handle 
                WHERE id = ?
                """, 
                [phone_number]
            )
            
            # If we have integrated with Contacts framework,
            # we can try to lookup the contact name
            contacts_db_path = Path(f"{HOME}/Library/Application Support/AddressBook/AddressBook-v22.abcddb")
            
            try:
                # First try using Apple's native Contacts framework if available
                if HAS_CONTACTS_FRAMEWORK:
                    try:
                        results = contacts.CNContactStore().unifiedContactsMatchingPredicate_keysToFetch_error_(
                            contacts.CNContact.predicateForContactsMatchingPhoneNumber_(
                                contacts.CNPhoneNumber.phoneNumberWithStringValue_(phone_number)
                            ),
                            [contacts.CNContactGivenNameKey, contacts.CNContactFamilyNameKey],
                            None
                        )[0]
                        if results:
                            contact = results[0]
                            first_name = contact.givenName()
                            last_name = contact.familyName()
                            return f"{first_name} {last_name}".strip()
                    except Exception as e:
                        logger.warning(f"Error looking up contact in Contacts framework: {e}")
                
                # Fallback to parsing the Contacts database if available
                if contacts_db_path.exists():
                    # Define a function to handle contact data safely in a way that can be stopped
                    def handle_contact(contact, stop):
                        first_name = contact.get("First", "")
                        last_name = contact.get("Last", "")
                        phone_numbers = contact.get("Phone", [])
                        
                        # Normalize the phone numbers
                        for phone in phone_numbers:
                            number = phone.get("value", "")
                            try:
                                parsed = phonenumbers.parse(number, "US")
                                if phonenumbers.is_valid_number(parsed):
                                    formatted = phonenumbers.format_number(
                                        parsed, phonenumbers.PhoneNumberFormat.E164
                                    )
                                    if formatted == phone_number or formatted[1:] == phone_number:
                                        stop.set()  # Signal to stop iteration
                                        return f"{first_name} {last_name}".strip()
                            except Exception:
                                # If we can't parse, try direct comparison
                                if number == phone_number:
                                    stop.set()  # Signal to stop iteration
                                    return f"{first_name} {last_name}".strip()
                        
                        return None
                    
                    # Try Contacts database lookup
                    import biplist
                    with open(contacts_db_path, 'rb') as f:
                        plist_data = biplist.readPlist(f)
                        contacts = plist_data.get("ABPersonWithoutPhotos", [])
                        
                        # Use a stop flag that can be set by the handler
                        stop_flag = threading.Event()
                        
                        for contact in contacts:
                            contact_dict = {}
                            for prop in contact.get("properties", []):
                                prop_name = prop.get("name", "")
                                prop_value = prop.get("value", "")
                                contact_dict[prop_name] = prop_value
                            
                            result = handle_contact(contact_dict, stop_flag)
                            if result or stop_flag.is_set():
                                return result
            except Exception as e:
                logger.warning(f"Error looking up contact in Contacts database: {e}")
            
            # If we've made it this far, we haven't found a contact name
            # Just return a string representation of the phone number
            return "Unknown"
        finally:
            self.release_connection(conn)

    def get_contacts(self):
        """Get all contacts from the database with enhanced contact info."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            query = """
            SELECT DISTINCT 
                handle.id as phone_number,
                MAX(message.date) as last_message_date,
                COUNT(message.ROWID) as message_count
            FROM handle
            JOIN message ON handle.ROWID = message.handle_id
            GROUP BY handle.id
            ORDER BY last_message_date DESC
            """
            
            cursor.execute(query)
            results = cursor.fetchall()
            
            contacts_data = []
            for phone_number, last_message_date, message_count in results:
                if phone_number:  # Skip if phone number is None
                    # Convert Apple epoch time to human readable date
                    try:
                        readable_date = self.convert_apple_time_to_datetime(last_message_date)
                        last_message_date = readable_date.isoformat()
                    except:
                        last_message_date = None
                    
                    # Use enhanced contact resolution if available
                    if hasattr(self, 'contact_resolver') and self.contact_resolver:
                        contact_info = self.contact_resolver.resolve_contact(phone_number)
                        contact = {
                            "identifier": phone_number,
                            "name": contact_info["name"] or "Unknown",
                            "display_name": contact_info["display_name"],
                            "formatted_identifier": contact_info["display_format"],
                            "type": contact_info["type"],
                            "message_count": message_count,
                            "last_message_date": last_message_date
                        }
                    else:
                        # Fall back to basic formatting if resolver not available
                        contact_name = self.get_contact_name(phone_number)
                        from src.utils.helpers import format_contact_name
                        display_name = format_contact_name(phone_number, contact_name)
                        contact = {
                            "identifier": phone_number,
                            "name": contact_name or "Unknown",
                            "display_name": display_name,
                            "message_count": message_count,
                            "last_message_date": last_message_date
                        }
                    
                    contacts_data.append(contact)
            
            return contacts_data
        finally:
            self.release_connection(conn)

    def get_chat_participants(self, chat_id):
        """Get all participants in a group chat."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            query = """
            SELECT handle.id, handle.service
            FROM handle
            JOIN chat_handle_join ON handle.ROWID = chat_handle_join.handle_id
            WHERE chat_handle_join.chat_id = ?
            """
            
            cursor.execute(query, [chat_id])
            results = cursor.fetchall()
            
            participants = []
            for handle_id, service in results:
                participants.append({
                    "id": handle_id,
                    "service": service
                })
            
            return participants
        finally:
            self.release_connection(conn)

    def find_chat_id_by_name(self, chat_identifier):
        """Find a chat ID by name or identifier.
        
        This method tries to find a chat by either its ROWID, GUID, or display_name,
        making it possible to reference a chat by any of these identifiers.
        
        Args:
            chat_identifier: Can be a numeric ID, GUID, or display name
            
        Returns:
            The ROWID of the chat if found, otherwise None
        """
        # If already numeric, it might be a ROWID
        if isinstance(chat_identifier, int) or (isinstance(chat_identifier, str) and chat_identifier.isdigit()):
            return int(chat_identifier)
                
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            # Try to find by GUID first (exact match)
            cursor.execute("SELECT ROWID FROM chat WHERE guid = ?", [chat_identifier])
            result = cursor.fetchone()
            if result:
                return result[0]
            
            # Try to find by display_name (case-insensitive)
            cursor.execute("SELECT ROWID FROM chat WHERE LOWER(display_name) = LOWER(?)", [chat_identifier])
            result = cursor.fetchone()
            if result:
                return result[0]
            
            # Try partial name match as a last resort
            cursor.execute("SELECT ROWID FROM chat WHERE LOWER(display_name) LIKE LOWER(?)", [f"%{chat_identifier}%"])
            result = cursor.fetchone()
            if result:
                return result[0]
                
            return None
        finally:
            self.release_connection(conn)
                
    def get_group_chat_messages(self, chat_id, start_date=None, end_date=None):
        """Get messages for a specific group chat."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            query = """
            SELECT 
                message.ROWID,
                message.date,
                message.text,
                message.attributedBody,
                message.is_from_me,
                handle.id
            FROM message
            JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
            LEFT JOIN handle ON message.handle_id = handle.ROWID
            WHERE chat_message_join.chat_id = ?
            """
            
            params = [chat_id]
            
            if start_date:
                # Ensure start_date is a datetime object
                if isinstance(start_date, str):
                    start_date = datetime.fromisoformat(start_date)
                start_epoch = self.convert_datetime_to_apple_time(start_date)
                query += " AND message.date >= ?"
                params.append(start_epoch)
                
            if end_date:
                # Ensure end_date is a datetime object
                if isinstance(end_date, str):
                    end_date = datetime.fromisoformat(end_date)
                end_epoch = self.convert_datetime_to_apple_time(end_date)
                query += " AND message.date <= ?"
                params.append(end_epoch)
                
            query += " ORDER BY message.date ASC"
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            messages = []
            for row in results:
                rowid, date, text, attributed_body, is_from_me, sender_id = row
                
                body = self.extract_message_body(text, attributed_body)
                if body is None:
                    continue
                
                try:
                    readable_date = self.convert_apple_time_to_datetime(date)
                    date = readable_date.isoformat()
                except:
                    date = None
                
                messages.append({
                    "rowid": rowid,
                    "text": body,
                    "date": date,
                    "is_from_me": bool(is_from_me),
                    "sender_id": sender_id
                })
            
            return messages
        finally:
            self.release_connection(conn)

    def get_group_chats(self):
        """Get all group chats."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            # Query to identify group chats (chats with multiple participants)
            query = """
            SELECT 
                chat.ROWID as chat_id,
                chat.guid,
                chat.display_name,
                COUNT(DISTINCT chat_handle_join.handle_id) as participant_count,
                MAX(message.date) as last_message_date
            FROM chat
            JOIN chat_handle_join ON chat.ROWID = chat_handle_join.chat_id
            LEFT JOIN chat_message_join ON chat.ROWID = chat_message_join.chat_id
            LEFT JOIN message ON chat_message_join.message_id = message.ROWID
            GROUP BY chat.ROWID, chat.guid, chat.display_name
            HAVING participant_count > 1 OR chat.display_name IS NOT NULL
            ORDER BY last_message_date DESC
            """
            
            cursor.execute(query)
            results = cursor.fetchall()
            
            chats = []
            for chat_id, guid, display_name, participant_count, last_message_date in results:
                # Format the date from Apple time to ISO
                date_str = None
                if last_message_date:
                    try:
                        date_obj = self.convert_apple_time_to_datetime(last_message_date)
                        date_str = date_obj.isoformat()
                    except:
                        pass
                
                chats.append({
                    "chat_id": chat_id,
                    "guid": guid,
                    "display_name": display_name,
                    "participant_count": participant_count,
                    "last_message_date": date_str
                })
            
            return chats
        finally:
            self.release_connection(conn)

    def iter_messages(self, phone_number=None, start_date=None, end_date=None):
        """Iterator that yields messages one at a time with enhanced contact info.
        
        Args:
            phone_number: Filter messages by phone number
            start_date: Filter messages after this date
            end_date: Filter messages before this date
            
        Yields:
            Dictionary containing message data with enhanced contact information
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            # Base query
            query = """
            SELECT 
                message.ROWID,
                message.date,
                message.text,
                message.attributedBody,
                message.is_from_me,
                message.cache_roomnames,
                chat.display_name,
                handle.id
            FROM message
            LEFT JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
            LEFT JOIN chat ON chat_message_join.chat_id = chat.ROWID
            LEFT JOIN handle ON message.handle_id = handle.ROWID
            WHERE 1=1
            """
            
            params = []
            
            # Add phone number filter if provided
            if phone_number:
                query += " AND handle.id = ?"
                params.append(phone_number)
            
            # Add date filters if provided
            if start_date:
                # Convert start_date to Apple epoch
                start_dt = datetime.fromisoformat(start_date) if isinstance(start_date, str) else start_date
                start_epoch = self.convert_datetime_to_apple_time(start_dt)
                query += " AND message.date >= ?"
                params.append(start_epoch)
                
            if end_date:
                # Convert end_date to Apple epoch
                end_dt = datetime.fromisoformat(end_date) if isinstance(end_date, str) else end_date
                end_epoch = self.convert_datetime_to_apple_time(end_dt)
                query += " AND message.date <= ?"
                params.append(end_epoch)
                
            # Add ordering
            query += " ORDER BY message.date ASC"
            
            cursor.execute(query, params)
            
            for row in cursor:
                rowid, date, text, attributed_body, is_from_me, cache_roomname, group_chat_name, handle_id = row
                
                # Process message body
                body = self.extract_message_body(text, attributed_body)
                if body is None:
                    continue
                
                # Convert date from Apple epoch time
                try:
                    readable_date = self.convert_apple_time_to_datetime(date)
                    date_str = readable_date.isoformat()
                except:
                    date_str = None
                
                # Get enhanced sender information
                sender_info = None
                if hasattr(self, 'contact_resolver') and self.contact_resolver:
                    # Create a temporary message dict for the resolver
                    temp_message = {
                        "is_from_me": bool(is_from_me),
                        "sender_id": handle_id
                    }
                    sender_info = self.contact_resolver.get_message_sender_info(temp_message)
                else:
                    # Fallback to basic sender info
                    if is_from_me:
                        sender_info = {
                            "name": "You",
                            "display_name": "You",
                            "is_self": True
                        }
                    elif handle_id:
                        contact_name = self.get_contact_name(handle_id)
                        from src.utils.helpers import format_contact_name
                        display_name = format_contact_name(handle_id, contact_name)
                        sender_info = {
                            "name": contact_name,
                            "display_name": display_name,
                            "is_self": False
                        }
                
                message_data = {
                    "rowid": rowid,
                    "text": body,
                    "date": date_str,
                    "is_from_me": bool(is_from_me),
                    "sender_id": handle_id,
                    "sender": sender_info,
                    "group_chat_name": group_chat_name,
                    "has_attachments": False  # We'll add attachment support later if needed
                }
                yield message_data
        finally:
            self.release_connection(conn)

    def extract_message_body(self, text, attributed_body):
        """Extract the text body from a message, handling rich text if needed."""
        if text:
            return text
            
        if not attributed_body:
            return None
            
        # Try to extract from attributed body if text is None
        try:
            import biplist
            
            # If it's stored as a binary plist (common in later macOS versions)
            if attributed_body.startswith(b'bplist'):
                attributed_body_data = biplist.readPlistFromString(attributed_body)
                if isinstance(attributed_body_data, list) and len(attributed_body_data) > 0:
                    if isinstance(attributed_body_data[0], dict):
                        return attributed_body_data[0].get('NSString', '')
        except:
            # If we can't parse the attributed body, return None
            pass
            
        return None

    @staticmethod
    def convert_apple_time_to_datetime(apple_time):
        """Convert Apple's time format to a datetime object."""
        # Apple's time format is nanoseconds since 2001-01-01
        seconds_since_epoch = apple_time / 1000000000 + 978307200
        return datetime.fromtimestamp(seconds_since_epoch)
    
    @staticmethod
    def convert_datetime_to_apple_time(dt):
        """Convert a datetime object to Apple's time format."""
        # Apple's time format is nanoseconds since 2001-01-01
        seconds_since_epoch = dt.timestamp() - 978307200
        return int(seconds_since_epoch * 1000000000) 