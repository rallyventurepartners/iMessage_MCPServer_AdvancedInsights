from pathlib import Path
import os
import sys
import traceback
import re
import sqlite3
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import json
from datetime import datetime, timedelta
import contextlib
from contextlib import contextmanager
import io
import statistics
from collections import Counter, defaultdict
import dateutil.parser
import dateutil.relativedelta
import phonenumbers
import threading
from textblob import TextBlob
import glob
import networkx as nx
from community import best_partition  # python-louvain package for community detection
import numpy as np

# Common English stop words to filter out in NLP analysis
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", 
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", 
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", 
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", 
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", 
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", 
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", 
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", 
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", 
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", 
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", 
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", 
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", 
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", 
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", 
    "yours", "yourself", "yourselves",
    # Additional common words in messages
    "just", "like", "get", "got", "yeah", "okay", "hey", "hi", "hello", "thanks", "thank", 
    "will", "can", "now", "know", "going", "good", "great", "well", "time", "also", "one", 
    "two", "day", "way", "thing", "make", "see", "need", "want", "said", "say", "go"
}

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("iMessage-Query-Server")

# Add contacts framework import for macOS
try:
    # The contacts module is provided by PyObjC and gives access to
    # macOS Contacts framework. Pylance may not recognize this as it's 
    # a dynamically loaded macOS-specific module.
    import contacts  # type: ignore
    # Test the contacts access by trying to create a store
    try:
        store = contacts.CNContactStore.alloc().init()
        # Request access to see if permissions are granted
        authorization = store.requestAccessForEntityType_completionHandler_(
            contacts.CNEntityTypeContacts, 
            None  # We'll handle this synchronously
        )
        if authorization:
            HAS_CONTACTS_FRAMEWORK = True
            logger.info("Contacts framework access granted")
        else:
            HAS_CONTACTS_FRAMEWORK = False
            logger.warning("Contacts framework access denied - please check permissions in System Settings")
    except Exception as e:
        HAS_CONTACTS_FRAMEWORK = False
        logger.warning(f"Contacts framework access error: {e}")
except ImportError:
    HAS_CONTACTS_FRAMEWORK = False
    logger.warning("contacts framework not available; contact names will not be displayed")

# Log startup information
logger.info("Starting iMessage Query server")
logger.info(f"Python version: {sys.version}")
logger.info(f"Current directory: {os.getcwd()}")

try:
    logger.info("Importing FastMCP")
    from fastmcp import FastMCP
    logger.info("Importing datetime modules")
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)

try:
    # Initialize FastMCP server
    logger.info("Initializing FastMCP")
    mcp = FastMCP("iMessage Query", dependencies=["phonenumbers", "python-dateutil"])
    logger.info("FastMCP initialized successfully")
except Exception as e:
    logger.error(f"Error initializing FastMCP: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)

# Default to Messages database in user's Library
DEFAULT_DB_PATH = Path.home() / "Library" / "Messages" / "chat.db"
DB_PATH = Path(os.environ.get('SQLITE_DB_PATH', DEFAULT_DB_PATH))

logger.info(f"Using database path: {DB_PATH}")
logger.info(f"Database exists: {DB_PATH.exists()}")

try:
    class AnalysisError(Exception):
        """Exception raised when message analysis fails."""
        pass
except Exception as e:
    logger.error(f"Error defining AnalysisError class: {e}")
    logger.error(traceback.format_exc())

class MessagesDB:
    """Class for handling database operations efficiently."""
    
    # Singleton instance
    _instance = None
    _init_lock = threading.Lock()
    
    def __new__(cls, db_path=DB_PATH):
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = super(MessagesDB, cls).__new__(cls)
                cls._instance.db_path = db_path
                cls._instance.connection_pool = []
                logger.info(f"Creating MessagesDB instance with path: {db_path}")
        return cls._instance
    
    def __init__(self, db_path=DB_PATH):
        # Only initialize once
        if not hasattr(self, 'initialized'):
            self.db_path = db_path
            self.connection_pool = []
            self.pool_lock = threading.Lock()
            self.initialized = True
            
            # Prepare for the DB queries
            logger.info(f"Initializing MessagesDB with path: {db_path}")
            
            if not self.db_path.exists():
                logger.error(f"iMessage database not found at {self.db_path}")
                raise FileNotFoundError(f"iMessage database not found at {self.db_path}")

    def get_connection(self):
        """Get a connection from the pool or create a new one."""
        with self.pool_lock:
            if self.connection_pool:
                return self.connection_pool.pop()
                
            logger.debug(f"Creating new DB connection to {self.db_path}")
            return sqlite3.connect(self.db_path)
    
    def release_connection(self, conn):
        """Return a connection to the pool."""
        with self.pool_lock:
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
                
            handle_id = result[1]  # Get the ROWID of the handle
            
            # Use the handle_id to get the Contact record
            try:
                contacts_db_path = Path(os.path.expanduser("~/Library/Application Support/AddressBook/AddressBook-v22.abcddb"))
                if not contacts_db_path.exists():
                    # Try fallback paths
                    fallback_paths = [
                        Path(os.path.expanduser("~/Library/Application Support/AddressBook/Sources/*/AddressBook-v22.abcddb")),
                        Path(os.path.expanduser("~/Library/Application Support/ContactsAgent/AddressBook-v22.abcddb")),
                        Path("/Users/*/Library/Application Support/AddressBook/AddressBook-v22.abcddb")
                    ]
                    
                    for path_pattern in fallback_paths:
                        found_paths = list(glob.glob(str(path_pattern)))
                        if found_paths:
                            contacts_db_path = Path(found_paths[0])
                            break
                
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
        """Get all contacts you have messaged.
        
        Returns:
            List of contact dictionaries with phone numbers and names.
        """
        logger.info("Getting all contacts")
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            # Query to get all unique message senders
            query = """
            SELECT DISTINCT handle.id, handle.service, 
                   COUNT(message.ROWID) as message_count,
                   MAX(message.date) as last_message_date
            FROM handle
            JOIN message ON handle.ROWID = message.handle_id
            WHERE handle.id IS NOT NULL AND handle.id != ''
            GROUP BY handle.id, handle.service
            ORDER BY last_message_date DESC
            """
            
            cursor.execute(query)
            results = cursor.fetchall()
            
            contacts = []
            for phone_number, service, message_count, last_message_date in results:
                # Skip invalid phone numbers
                if not phone_number or phone_number.strip() == '':
                    continue
                    
                # Format the date from Apple time to ISO
                date_str = None
                if last_message_date:
                    try:
                        date_obj = self.convert_apple_time_to_datetime(last_message_date)
                        date_str = date_obj.isoformat()
                    except:
                        pass
                
                # Get contact name if available
                contact_name = self.get_contact_name(phone_number)
                
                contacts.append({
                    "phone_number": phone_number,
                    "service": service,
                    "name": contact_name if contact_name != "Unknown" else phone_number,
                    "message_count": message_count,
                    "last_message_date": date_str
                })
            
            return contacts
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

    def parse_natural_language_time_period(query: str) -> Tuple[datetime, datetime]:
        """
        Extract a time period from a natural language query.
        
        Args:
            query: Natural language query that might contain time period information
            
        Returns:
            Tuple of (start_date, end_date) as datetime objects
        """
        now = datetime.now()
        
        # Default to last 7 days if nothing matches
        start_date = now - timedelta(days=7)
        end_date = now
        
        # Common patterns
        last_n_days_pattern = r"(?:last|past)\s+(\d+)\s+days?"
        last_n_weeks_pattern = r"(?:last|past)\s+(\d+)\s+weeks?"
        last_n_months_pattern = r"(?:last|past)\s+(\d+)\s+months?"
        yesterday_pattern = r"yesterday"
        today_pattern = r"today"
        this_week_pattern = r"this\s+week"
        this_month_pattern = r"this\s+month"
        this_year_pattern = r"this\s+year"
        
        # Check for days
        days_match = re.search(last_n_days_pattern, query, re.IGNORECASE)
        if days_match:
            days = int(days_match.group(1))
            start_date = now - timedelta(days=days)
            return start_date, end_date
            
        # Check for weeks
        weeks_match = re.search(last_n_weeks_pattern, query, re.IGNORECASE)
        if weeks_match:
            weeks = int(weeks_match.group(1))
            start_date = now - timedelta(weeks=weeks)
            return start_date, end_date
            
        # Check for months
        months_match = re.search(last_n_months_pattern, query, re.IGNORECASE)
        if months_match:
            months = int(months_match.group(1))
            start_date = now - dateutil.relativedelta.relativedelta(months=months)
            return start_date, end_date
            
        # Check for yesterday
        if re.search(yesterday_pattern, query, re.IGNORECASE):
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
            end_date = now.replace(hour=23, minute=59, second=59, microsecond=999999) - timedelta(days=1)
            return start_date, end_date
            
        # Check for today
        if re.search(today_pattern, query, re.IGNORECASE):
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
            return start_date, end_date
            
        # Check for this week
        if re.search(this_week_pattern, query, re.IGNORECASE):
            # Start from Monday of current week
            days_since_monday = now.weekday()
            start_date = (now - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
            return start_date, end_date
            
        # Check for this month
        if re.search(this_month_pattern, query, re.IGNORECASE):
            start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            return start_date, end_date
            
        # Check for this year
        if re.search(this_year_pattern, query, re.IGNORECASE):
            start_date = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            return start_date, end_date
            
        # Try to parse with dateutil as a fallback
        try:
            parsed_date = dateutil.parser.parse(query, fuzzy=True)
            # If we got a valid date, assume they want that specific day
            start_date = parsed_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = parsed_date.replace(hour=23, minute=59, second=59, microsecond=999999)
            return start_date, end_date
        except:
            # If all else fails, return the default (last 7 days)
            pass
            
        return start_date, end_date

    def should_analyze_all_contacts(query: str) -> bool:
        """
        Determine if the query is asking to analyze all contacts.
        
        Args:
            query: Natural language query
            
        Returns:
            Boolean indicating if all contacts should be analyzed
        """
        all_contacts_patterns = [
            r'all\s+contacts',
            r'all\s+conversations',
            r'all\s+messages',
            r'every\s+contact',
            r'every\s+conversation',
            r'everyone',
            r'everybody',
            r'across\s+all',
            r'total\s+messages',
            r'overall\s+stats',
            r'overall\s+statistics',
            r'all\s+chats'
        ]
        
        for pattern in all_contacts_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
                
        return False

    def extract_phone_number(query: str) -> Optional[str]:
        """
        Extract a phone number from a natural language query.
        
        Args:
            query: Natural language query that might contain a phone number
            
        Returns:
            Extracted phone number or None if not found
        """
        # Basic phone number pattern
        phone_patterns = [
            r'(\+\d{1,3}\s?)?(\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}',  # General format
            r'\b\d{10}\b',  # 10 digits
            r'\+\d{1,3}\d{10}'  # International format
        ]
        
        for pattern in phone_patterns:
            match = re.search(pattern, query)
            if match:
                # Try to parse and validate
                try:
                    number = match.group(0)
                    parsed_number = phonenumbers.parse(number, "US")
                    if phonenumbers.is_valid_number(parsed_number):
                        formatted = phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.E164)
                        return formatted
                except:
                    continue
                    
        return None

    def analyze_messages(messages_iter):
        """Analyze messages (supports both lists and iterators).
        
        Args:
            messages_iter: A list or iterator of message objects
            
        Returns:
            Dictionary containing the analysis results
        """
        # Initialize counters
        total_messages = 0
        sent_messages = 0
        received_messages = 0
        total_length = 0
        sent_length = 0
        received_length = 0
        
        # Message lengths for statistical analysis
        sent_lengths = []
        received_lengths = []
        all_lengths = []
        
        # Time analysis
        dates = []
        
        # Word frequency analysis
        word_counter = Counter()
        
        # Process messages one by one to handle both lists and iterators
        all_messages = []  # Store processed messages if needed for further analysis
        
        for msg in messages_iter:
            # Ensure we can iterate multiple times if needed
            all_messages.append(msg)
            
            total_messages += 1
            
            # Text analysis
            text = msg.get("text", "")
            msg_length = len(text)
            total_length += msg_length
            all_lengths.append(msg_length)
            
            # Count words (simple split on whitespace)
            if text:
                words = re.findall(r'\b\w+\b', text.lower())
                word_counter.update(words)
            
            # Sent vs received analysis
            if msg.get("is_from_me"):
                sent_messages += 1
                sent_length += msg_length
                sent_lengths.append(msg_length)
            else:
                received_messages += 1
                received_length += msg_length
                received_lengths.append(msg_length)
            
            # Date analysis
            date_str = msg.get("date")
            if date_str:
                try:
                    date = datetime.fromisoformat(date_str)
                    dates.append(date)
                except:
                    pass
        
        # Calculate statistics
        avg_length = total_length / total_messages if total_messages > 0 else 0
        avg_sent_length = sent_length / sent_messages if sent_messages > 0 else 0
        avg_received_length = received_length / received_messages if received_messages > 0 else 0
        
        max_length = max(all_lengths) if all_lengths else 0
        min_length = min(all_lengths) if all_lengths else 0
        
        # Get most common words (excluding very common words)
        common_words = [word for word, count in word_counter.most_common(20)]
        
        # Time series analysis
        date_counts = Counter()
        for date in dates:
            date_key = date.date().isoformat()
            date_counts[date_key] += 1
        
        # Sort by date
        time_series = [{"date": date, "count": count} for date, count in sorted(date_counts.items())]
        
        # Prepare result
        return {
            "total_messages": total_messages,
            "sent_messages": sent_messages,
            "received_messages": received_messages,
            "response_ratio": round(sent_messages / received_messages, 2) if received_messages > 0 else 0,
            "common_words": common_words,
            "time_series": time_series,
            "message_length": {
                "average": round(avg_length, 2),
                "average_sent": round(avg_sent_length, 2),
                "average_received": round(avg_received_length, 2),
                "maximum": max_length,
                "minimum": min_length
            },
            "analysis_summary": f"Analyzed {total_messages} messages ({sent_messages} sent, {received_messages} received). " +
                               f"Average message length is {round(avg_length, 2)} characters."
        }

    def error_response(code, message, details=None):
        """Create a standardized error response."""
        error = {
            "code": code,
            "message": message
        }
        if details:
            error["details"] = details
        return {"error": error}
    
    def resolve_chat_id(chat_id):
        """Helper function to resolve chat ID from name or ID.
        
        This function is used by all tools that require a chat ID to support
        looking up chats by their display name as well as by ID.
        
        Args:
            chat_id: The ID, GUID, or display name of the chat
            
        Returns:
            Tuple of (resolved_id, error_response) where error_response is None if successful
        """
        if chat_id is None:
            return None, None
            
        try:
            db = MessagesDB()
            resolved_id = db.find_chat_id_by_name(chat_id)
            
            if resolved_id is None:
                return None, error_response("CHAT_NOT_FOUND", f"Group chat with ID or name '{chat_id}' not found")
                
            return resolved_id, None
        except Exception as e:
            logger.error(f"Error resolving chat ID: {e}")
            logger.error(traceback.format_exc())
            return None, error_response("ERROR", f"Error resolving chat ID: {str(e)}")
            
    @mcp.tool()
    def get_contacts() -> Dict[str, Any]:
        """Get all contacts you have messaged.
                
        Returns:
            A dictionary containing contact information and message counts
        """
        logger.info("get_contacts called")
        
        try:
            # Create MessagesDB instance
            db = MessagesDB()
            
            # Get contacts from database
            logger.info("Retrieving contacts from database")
            contacts_data = db.get_contacts()
            
            return {
                "contacts": contacts_data,
                "total_count": len(contacts_data)
            }
        except Exception as e:
            logger.error(f"Error retrieving contacts: {e}")
            logger.error(traceback.format_exc())
            raise

    @mcp.tool()
    def get_chat_transcript(
        phone_number: str = None,
        contact_name: str = None,
        start_date: str = None,
        end_date: str = None
    ) -> Dict[str, Any]:
        """
        Retrieve message transcript for a specific contact.
        
        Args:
            phone_number: Phone number of the contact
            contact_name: Name of the contact (alternative to phone_number)
            start_date: Start date for transcript (format: YYYY-MM-DD)
            end_date: End date for transcript (format: YYYY-MM-DD)
        
        Returns:
            Message transcript with contact information
        """
        # Initialize database connection
        db = MessagesDB()
        
        # Validate that we have either a phone number or contact name
        if not phone_number and not contact_name:
            return error_response(
                "MISSING_PARAMETER", 
                "Either phone_number or contact_name must be provided"
            )
        
        # If contact_name is provided but no phone_number, try to find the phone number
        if contact_name and not phone_number:
            logging.debug(f"Looking up phone number for contact name: {contact_name}")
            contacts = db.get_contacts()
            
            # Case-insensitive partial matching for contact names
            contact_name_lower = contact_name.lower()
            matching_contacts = []
            
            for contact in contacts:
                # Check for matches in display_name or contact_name fields
                if 'display_name' in contact and contact['display_name']:
                    if contact_name_lower in contact['display_name'].lower():
                        matching_contacts.append(contact)
                if 'contact_name' in contact and contact['contact_name']:
                    if contact_name_lower in contact['contact_name'].lower():
                        matching_contacts.append(contact)
            
            if matching_contacts:
                # If we found multiple matches, prefer exact matches or take the first one
                exact_matches = [c for c in matching_contacts 
                               if c.get('contact_name', '').lower() == contact_name_lower or 
                                  c.get('display_name', '').lower() == contact_name_lower]
                
                selected_contact = exact_matches[0] if exact_matches else matching_contacts[0]
                phone_number = selected_contact.get('id')
                logging.debug(f"Found phone number {phone_number} for contact {contact_name}")
            else:
                return error_response(
                    "CONTACT_NOT_FOUND", 
                    f"Could not find a contact matching '{contact_name}'"
                )
        
        # Normalize phone number format if provided directly
        if phone_number:
            # Handle various phone number formats
            phone_number = phone_number.strip()
            
            # If it doesn't start with +, assume it's a US number if it has 10 digits
            if not phone_number.startswith('+'):
                if len(re.sub(r'[^0-9]', '', phone_number)) == 10:
                    phone_number = '+1' + re.sub(r'[^0-9]', '', phone_number)
                elif len(re.sub(r'[^0-9]', '', phone_number)) == 11 and re.sub(r'[^0-9]', '', phone_number).startswith('1'):
                    phone_number = '+' + re.sub(r'[^0-9]', '', phone_number)
            
            # Try to standardize the phone number format
            try:
                parsed_number = phonenumbers.parse(phone_number, "US")
                if phonenumbers.is_valid_number(parsed_number):
                    phone_number = phonenumbers.format_number(
                        parsed_number, phonenumbers.PhoneNumberFormat.E164
                    )
            except Exception as e:
                logging.debug(f"Could not parse phone number {phone_number}: {e}")
                # Continue with the original format if parsing fails
        
        # Validate and convert dates
        error = validate_date_range(start_date, end_date)
        if error:
            return error
            
        start_dt = None if not start_date else ensure_datetime(start_date)
        end_dt = None if not end_date else ensure_datetime(end_date)
        
        # Retrieve the contact name for display
        contact_name = db.get_contact_name(phone_number)
        display_name = format_contact_name(phone_number, contact_name)
        
        # Get messages
        try:
            messages = list(db.iter_messages(phone_number, start_dt, end_dt))
            
            # Return formatted response
            return {
                "contact": {
                    "phone_number": phone_number,
                    "name": contact_name if contact_name != "Unknown" else None,
                    "display_name": display_name
                },
                "date_range": {
                    "start": start_date,
                    "end": end_date
                },
                "message_count": len(messages),
                "messages": messages
            }
        except Exception as e:
            logging.error(f"Error retrieving messages: {e}")
            return error_response(
                "DATABASE_ERROR", 
                f"Error retrieving messages: {str(e)}"
            )

    @mcp.tool()
    def analyze_chat_history(
        query: str
    ) -> Dict[str, Any]:
        """Analyze chat history based on natural language query.
        
        Args:
            query: Natural language query, examples:
                  "Analyze all messages from the past 30 days"
                  "Analyze all messages with John from last week"
                  "Analyze my chat with +16505551234 from Jan 1 to Feb 1"
                  "Analyze the Family group chat from the past month"
        
        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"analyze_chat_history called with query={query}")
        
        # Extract time period from query
        start_date, end_date = parse_natural_language_time_period(query)
        logger.info(f"Extracted time period: {start_date} to {end_date}")
        
        # Check if this is a group chat query
        is_group_chat_query = should_analyze_group_chat(query)
        logger.info(f"Is group chat query: {is_group_chat_query}")
        
        if is_group_chat_query:
            # Extract group chat IDs if present
            group_chat_ids = extract_group_chat_ids(query)
            logger.info(f"Extracted group chat IDs: {group_chat_ids}")
            
            # Extract group chat name if no IDs found
            group_chat_name = None
            if not group_chat_ids:
                group_chat_name = extract_group_chat_name(query)
                logger.info(f"Extracted group chat name: {group_chat_name}")
            
            try:
                # Create MessagesDB instance
                db = MessagesDB()
                
                # If we have multiple chat IDs, analyze them together
                if len(group_chat_ids) > 1:
                    return analyze_multiple_group_chats(
                        group_chat_ids,
                        start_date.isoformat(),
                        end_date.isoformat()
                    )
                
                # Get all group chats if we need to search by name
                if not group_chat_ids:
                    group_chats = db.get_group_chats()
                    
                    # If no group chats found
                    if not group_chats:
                        return error_response("NO_CHATS", "No group chats found")
                    
                    # If group chat name specified, try to find a match
                    target_chat = None
                    if group_chat_name:
                        for chat in group_chats:
                            if chat["name"] and group_chat_name.lower() in chat["name"].lower():
                                target_chat = chat
                                break
                    
                    # If no match or no name specified, use the most recent group chat
                    if not target_chat:
                        target_chat = group_chats[0]
                        logger.info(f"Using most recent group chat: {target_chat['name']}")
                        
                    chat_id = target_chat["chat_id"]
                else:
                    # Use the first ID from the extracted list
                    chat_id = group_chat_ids[0]
                
                # Analyze the group chat
                return analyze_group_chat(
                    chat_id,
                    start_date.isoformat(),
                    end_date.isoformat()
                )
                
            except Exception as e:
                logger.error(f"Error analyzing group chat: {e}")
                logger.error(traceback.format_exc())
                raise AnalysisError(f"Failed to analyze group chat: {str(e)}") from e
        else:
            # Existing code for non-group chat queries
            # Extract phone number from query if present
            phone_number = extract_phone_number(query)
            logger.info(f"Extracted phone number: {phone_number}")
            
            # Determine if we should analyze all contacts
            analyze_all = should_analyze_all_contacts(query)
            logger.info(f"Analyze all contacts: {analyze_all}")
            
            try:
                # Create MessagesDB instance
                db = MessagesDB()
                
                if analyze_all:
                    # Get all messages within the date range
                    messages = db.iter_messages(
                        start_date=start_date,
                        end_date=end_date
                    )
                else:
                    if not phone_number:
                        # If no specific contact and not analyzing all, default to most recent contact
                        contacts = db.get_contacts()
                        if not contacts:
                            return error_response("NO_CONTACTS", "No contacts found")
                        
                        phone_number = contacts[0]["phone_number"]
                        logger.info(f"No specific contact specified, using most recent: {phone_number}")
                    
                    # Get messages for the specific contact within the date range
                    messages = db.iter_messages(
                        phone_number=phone_number,
                        start_date=start_date,
                        end_date=end_date
                    )
                
                # Analyze the messages
                analysis_results = analyze_messages(messages)
                
                # Include query parameters in the results
                analysis_results["query_params"] = {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "phone_number": phone_number,
                    "analyze_all": analyze_all
                }
                
                return analysis_results
            except Exception as e:
                logger.error(f"Error analyzing chat history: {e}")
                logger.error(traceback.format_exc())
                raise

    @mcp.tool()
    def get_group_chats() -> Dict[str, Any]:
        """Get all group chat conversations.
        
        Returns:
            Dictionary containing a list of group chats with their names, IDs, and participant counts.
        """
        logger.info("get_group_chats called")
        
        try:
            db = MessagesDB()
            group_chats = db.get_group_chats()
            
            return {
                "group_chats": group_chats,
                "total_count": len(group_chats)
            }
        except Exception as e:
            logger.error(f"Error retrieving group chats: {e}")
            logger.error(traceback.format_exc())
            raise

    def extract_group_chat_name(query: str) -> Optional[str]:
        """Extract a group chat name from a natural language query.
        
        Args:
            query: Natural language query
            
        Returns:
            Extracted group chat name or None if not found
        """
        # Patterns to match group chat names in quotes
        patterns = [
            r'group\s+(?:chat|conversation)\s+(?:called|named)\s+"([^"]+)"',
            r'group\s+(?:chat|conversation)\s+"([^"]+)"',
            r'"([^"]+)"\s+group\s+(?:chat|conversation)',
            r'group\s+(?:called|named)\s+"([^"]+)"'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None

    def extract_group_chat_ids(query: str) -> List[str]:
        """Extract group chat IDs from a natural language query.
        
        Args:
            query: Natural language query
            
        Returns:
            List of extracted group chat IDs
        """
        # Look for patterns like "chat ID 1234" or "chat IDs 1234 and 5678"
        single_id_pattern = r'chat\s+ID\s+(\d+)'
        multiple_ids_pattern = r'chat\s+IDs\s+([\d\s,and]+)'
        
        # Try multiple IDs pattern first
        multi_match = re.search(multiple_ids_pattern, query, re.IGNORECASE)
        if multi_match:
            # Extract the raw text containing IDs
            ids_text = multi_match.group(1)
            # Extract all numbers from the text
            return re.findall(r'\d+', ids_text)
        
        # Try single ID pattern
        single_matches = re.findall(single_id_pattern, query, re.IGNORECASE)
        if single_matches:
            return single_matches
        
        return []

    def sanitize_name(name):
        """Remove potentially problematic characters from names."""
        if name:
            # Remove control characters and limit length
            return re.sub(r'[\x00-\x1F\x7F]', '', name)[:100]
        return name

    @mcp.tool()
    def analyze_group_chat(
        chat_id: Union[str, int],
        start_date: str = None,
        end_date: str = None
    ) -> Dict[str, Any]:
        """Analyze a specific group chat's messages.
        
        Args:
            chat_id: The ID, GUID, or display name of the group chat
            start_date: Optional start date in ISO format (YYYY-MM-DD)
            end_date: Optional end date in ISO format (YYYY-MM-DD)
            
        Returns:
            Dictionary containing analysis of the group chat
        """
        logger.info(f"analyze_group_chat called with chat_id={chat_id}, start_date={start_date}, end_date={end_date}")
        
        # Validate date range if both dates are provided
        error = validate_date_range(start_date, end_date)
        if error:
            return error
            
        # Resolve the chat ID from name or ID
        actual_chat_id, error = resolve_chat_id(chat_id)
        if error:
            return error
        
        try:
            db = MessagesDB()
            
            # Convert string dates to datetime objects for database operations
            start_dt = ensure_datetime(start_date) if start_date else None
            end_dt = ensure_datetime(end_date) if end_date else None
            
            # Get chat name
            with db.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT display_name FROM chat WHERE ROWID = ?", [actual_chat_id])
                result = cursor.fetchone()
                chat_name = sanitize_name(result[0]) if result and result[0] else "Unnamed Group"
            
            # Get the messages
            messages = db.get_group_chat_messages(actual_chat_id, start_dt, end_dt)
            
            # Get participants
            participants = db.get_chat_participants(actual_chat_id)
            
            logger.info(f"Found {len(messages)} messages for sentiment analysis")
            
            # Analyze sentiment
            sentiment_data = []
            daily_sentiment = defaultdict(list)
            sender_sentiment = defaultdict(list)
            
            for msg in messages:
                text = msg.get("text", "")
                if not text or len(text.strip()) < 3:  # Skip very short messages
                    continue
                    
                # Get sentiment scores
                analysis = TextBlob(text)
                polarity = analysis.sentiment.polarity  # -1 to 1 (negative to positive)
                subjectivity = analysis.sentiment.subjectivity  # 0 to 1 (objective to subjective)
                
                # Get sender info
                is_from_me = msg.get("is_from_me", False)
                sender = "You" if is_from_me else msg.get("sender_id", "Unknown")
                
                # Track sentiment by sender
                sender_sentiment[sender].append(polarity)
                
                # Track sentiment by date
                date_str = msg.get("date")
                if date_str:
                    try:
                        date_obj = datetime.fromisoformat(date_str)
                        date_key = date_obj.date().isoformat()
                        daily_sentiment[date_key].append(polarity)
                    except (ValueError, TypeError):
                        # If date parsing fails, skip the daily tracking for this message
                        pass
                
                # Add the message with sentiment data
                sentiment_data.append({
                    "text": text[:100] + "..." if len(text) > 100 else text,  # Truncate long messages
                    "date": msg.get("date"),
                    "is_from_me": is_from_me,
                    "sender": sender,
                    "polarity": round(polarity, 3),
                    "subjectivity": round(subjectivity, 3),
                    "sentiment": "positive" if polarity > 0.2 else "negative" if polarity < -0.2 else "neutral"
                })
            
            if not sentiment_data:
                logger.warning("No messages found with sufficient content for sentiment analysis")
                return {
                    "warning": "No messages found with sufficient content for sentiment analysis",
                    "message_count": 0
                }
            
            # Calculate daily averages
            daily_averages = []
            for date, scores in sorted(daily_sentiment.items()):
                daily_averages.append({
                    "date": date,
                    "average_sentiment": round(sum(scores) / len(scores), 3),
                    "message_count": len(scores)
                })
            
            # Calculate sentiment by sender
            sender_averages = []
            for sender, scores in sender_sentiment.items():
                avg_sentiment = sum(scores) / len(scores)
                display_name = sender
                if sender != "You" and not sender.startswith("Unknown"):
                    display_name = format_contact_name(sender)
                
                sender_averages.append({
                    "sender": sender,
                    "display_name": display_name,
                    "message_count": len(scores),
                    "average_sentiment": round(avg_sentiment, 3),
                    "sentiment_category": "positive" if avg_sentiment > 0.2 else "negative" if avg_sentiment < -0.2 else "neutral"
                })
            
            # Sort by message count
            sender_averages.sort(key=lambda x: x["message_count"], reverse=True)
            
            # Calculate overall sentiment stats
            sent_polarities = [item["polarity"] for item in sentiment_data]
            sent_subjectivities = [item["subjectivity"] for item in sentiment_data]
            
            # Determine sentiment trends (is it getting more positive/negative over time)
            trend = "neutral"
            if len(daily_averages) > 3:
                first_half = daily_averages[:len(daily_averages)//2]
                second_half = daily_averages[len(daily_averages)//2:]
                
                first_half_avg = sum(day["average_sentiment"] for day in first_half) / len(first_half) if first_half else 0
                second_half_avg = sum(day["average_sentiment"] for day in second_half) / len(second_half) if second_half else 0
                
                difference = second_half_avg - first_half_avg
                if difference > 0.1:
                    trend = "becoming more positive"
                elif difference < -0.1:
                    trend = "becoming more negative"
                else:
                    trend = "remaining steady"
            
            overall_sentiment = sum(sent_polarities) / len(sent_polarities) if sent_polarities else 0
            
            return {
                "message_count": len(sentiment_data),
                "analyzed_message_count": len(sentiment_data),
                "overall_sentiment": round(overall_sentiment, 3),
                "overall_sentiment_category": "positive" if overall_sentiment > 0.2 else "negative" if overall_sentiment < -0.2 else "neutral",
                "overall_subjectivity": round(sum(sent_subjectivities) / len(sent_subjectivities), 3) if sent_subjectivities else 0,
                "sentiment_trend": trend,
                "daily_sentiment": daily_averages,
                "sender_sentiment": sender_averages,
                "sentiment_breakdown": {
                    "positive": sum(1 for item in sentiment_data if item["sentiment"] == "positive"),
                    "neutral": sum(1 for item in sentiment_data if item["sentiment"] == "neutral"),
                    "negative": sum(1 for item in sentiment_data if item["sentiment"] == "negative")
                },
                "message_samples": sentiment_data[:20]  # Return just the first 20 for brevity
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            logger.error(traceback.format_exc())
            return error_response("ANALYSIS_ERROR", f"Error analyzing sentiment: {str(e)}")

    @mcp.tool()
    def analyze_multiple_group_chats(
        chat_ids: List[Union[str, int]],
        start_date: str = None,
        end_date: str = None
    ) -> Dict[str, Any]:
        """Analyze multiple group chats and provide comparative statistics.
        
        Args:
            chat_ids: List of chat IDs to analyze
            start_date: Optional start date in ISO format (YYYY-MM-DD)
            end_date: Optional end date in ISO format (YYYY-MM-DD)
        
        Returns:
            Dictionary containing comparative analysis of multiple group chats
        """
        logger.info(f"analyze_multiple_group_chats called with chat_ids={chat_ids}, start_date={start_date}, end_date={end_date}")
        
        if not chat_ids:
            return error_response("NO_CHATS", "No chat IDs provided")
        
        try:
            results = []
            for chat_id in chat_ids:
                # Get individual chat analysis
                chat_analysis = analyze_group_chat(chat_id, start_date, end_date)
                if "error" not in chat_analysis:
                    # Safe access to dictionary keys with defaults
                    chat_data = {
                        "chat_id": chat_analysis.get("chat_id", chat_id),
                        "chat_name": chat_analysis.get("chat_name", f"Chat {chat_id}"),
                        "message_count": chat_analysis.get("message_count", 0),
                        "participant_count": chat_analysis.get("participant_count", 0)
                    }
                    # Add other data from chat_analysis
                    for key, value in chat_analysis.items():
                        if key not in ["chat_id", "chat_name", "message_count", "participant_count"]:
                            chat_data[key] = value
                            
                    results.append(chat_data)
            
            if not results:
                return error_response("NO_CHATS", "No valid group chats found")
            
            # Compile comparative metrics
            comparative_analysis = {
                "chats": results,
                "total_messages": sum(r.get("message_count", 0) for r in results),
                "time_period": {
                    "start_date": start_date,
                    "end_date": end_date
                },
                "individual_results": results
            }
            
            # Sort chats by message count
            comparative_analysis["chats"].sort(key=lambda x: x.get("message_count", 0), reverse=True)
            
            return comparative_analysis
        except Exception as e:
            logger.error(f"Error analyzing multiple group chats: {e}")
            logger.error(traceback.format_exc())
            return error_response("ANALYSIS_ERROR", f"Error analyzing multiple group chats: {str(e)}")

    @mcp.tool()
    def analyze_sentiment(
        phone_number: str = None,
        chat_id: Union[str, int] = None,
        start_date: str = None,
        end_date: str = None
    ) -> Dict[str, Any]:
        """Analyze the sentiment of messages in a conversation over time.
        
        This function examines message content using NLP techniques to determine
        sentiment (positive, negative, neutral) and tracks changes in sentiment
        patterns over time.
        
        Args:
            phone_number: Optional phone number to filter messages for individual chat
            chat_id: Optional chat ID to filter messages for a group chat
            start_date: Optional start date in ISO format (YYYY-MM-DD)
            end_date: Optional end date in ISO format (YYYY-MM-DD)
            
        Returns:
            Dictionary containing sentiment analysis results, time trends, and comparisons between participants
        """
        logger.info(f"analyze_sentiment called with phone_number={phone_number}, chat_id={chat_id}, start_date={start_date}, end_date={end_date}")
        
        # Validate parameters
        if not phone_number and not chat_id:
            return error_response("MISSING_PARAMETER", "Either phone_number or chat_id must be provided")
        
        # Validate date range
        error = validate_date_range(start_date, end_date)
        if error:
            return error
        
        try:
            db = MessagesDB()
            
            # Convert date strings to datetime objects
            start_dt = ensure_datetime(start_date) if start_date else None
            end_dt = ensure_datetime(end_date) if end_date else None
            
            # Get messages from appropriate source
            if chat_id:
                logger.info(f"Analyzing sentiment for group chat ID: {chat_id}")
                
                # Resolve the chat ID from name or ID
                actual_chat_id, error = resolve_chat_id(chat_id)
                if error:
                    return error
                    
                messages = db.get_group_chat_messages(actual_chat_id, start_dt, end_dt)
                
                # Get chat info
                try:
                    with db.get_db_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT display_name FROM chat WHERE ROWID = ? OR guid = ?", [actual_chat_id, actual_chat_id])
                        result = cursor.fetchone()
                        chat_name = result[0] if result else "Unknown Group"
                except Exception as e:
                    logger.warning(f"Error getting group chat name: {e}")
                    chat_name = "Group Chat"
                    
                # Get participant info for group chats
                try:
                    participants = db.get_chat_participants(actual_chat_id)
                    logger.info(f"Found {len(participants)} participants in group chat")
                    participant_info = {p.get("id"): p.get("display_name", "Unknown") for p in participants}
                except Exception as e:
                    logger.warning(f"Error getting group chat participants: {e}")
                    participant_info = {}
                    
            else:
                logger.info(f"Analyzing sentiment for contact: {phone_number}")
                messages = list(db.iter_messages(phone_number, start_dt, end_dt))
                contact_name = db.get_contact_name(phone_number)
                display_name = format_contact_name(phone_number)
                chat_name = f"Chat with {display_name}"
                participant_info = {phone_number: display_name}
            
            logger.info(f"Found {len(messages)} messages for sentiment analysis")
            
            if not messages:
                return {
                    "warning": "No messages found with sufficient content for sentiment analysis",
                    "message_count": 0,
                    "chat_name": chat_name
                }
                
            # Sort messages by date
            messages = sorted(messages, key=lambda m: m.get("date", "") if isinstance(m.get("date"), str) else "")
            
            # Analyze sentiment
            sentiment_data = []
            daily_sentiment = defaultdict(list)
            sender_sentiment = defaultdict(list)
            weekly_sentiment = defaultdict(list)
            monthly_sentiment = defaultdict(list)
            
            for msg in messages:
                text = msg.get("text", "")
                if not text or len(text.strip()) < 3:  # Skip very short messages
                    continue
                    
                # Get sentiment scores
                analysis = TextBlob(text)
                polarity = analysis.sentiment.polarity  # -1 to 1 (negative to positive)
                subjectivity = analysis.sentiment.subjectivity  # 0 to 1 (objective to subjective)
                
                # Get sender info
                is_from_me = msg.get("is_from_me", False)
                sender = "You" if is_from_me else msg.get("sender_id", "Unknown")
                
                # Get formatted display name
                if sender == "You":
                    display_name = "You"
                elif sender in participant_info:
                    display_name = participant_info[sender]
                else:
                    display_name = format_contact_name(sender)
                    
                # Track sentiment by sender
                sender_sentiment[sender].append(polarity)
                
                # Track sentiment by date
                date_str = msg.get("date")
                if date_str:
                    try:
                        date_obj = datetime.fromisoformat(date_str)
                        date_key = date_obj.date().isoformat()
                        daily_sentiment[date_key].append(polarity)
                        
                        # Weekly tracking - get the Monday of the week
                        week_start = date_obj - timedelta(days=date_obj.weekday())
                        week_key = week_start.date().isoformat()
                        weekly_sentiment[week_key].append(polarity)
                        
                        # Monthly tracking
                        month_key = f"{date_obj.year}-{date_obj.month:02d}"
                        monthly_sentiment[month_key].append(polarity)
                        
                    except (ValueError, TypeError):
                        # If date parsing fails, skip the time tracking for this message
                        pass
                
                # Add the message with sentiment data
                sentiment_data.append({
                    "text": text[:100] + "..." if len(text) > 100 else text,  # Truncate long messages
                    "date": msg.get("date"),
                    "is_from_me": is_from_me,
                    "sender": sender,
                    "display_name": display_name,
                    "polarity": round(polarity, 3),
                    "subjectivity": round(subjectivity, 3),
                    "sentiment": "positive" if polarity > 0.2 else "negative" if polarity < -0.2 else "neutral"
                })
                
            if not sentiment_data:
                logger.warning("No messages found with sufficient content for sentiment analysis")
                return {
                    "warning": "No messages found with sufficient content for sentiment analysis",
                    "message_count": 0,
                    "chat_name": chat_name
                }
            
            # Calculate daily averages
            daily_averages = []
            for date, scores in sorted(daily_sentiment.items()):
                daily_averages.append({
                    "date": date,
                    "average_sentiment": round(sum(scores) / len(scores), 3),
                    "message_count": len(scores)
                })
                
            # Calculate weekly averages
            weekly_averages = []
            for week, scores in sorted(weekly_sentiment.items()):
                week_obj = datetime.fromisoformat(week)
                week_end = week_obj + timedelta(days=6)
                weekly_averages.append({
                    "week_start": week,
                    "week_end": week_end.date().isoformat(),
                    "label": f"{week_obj.strftime('%b %d')} - {week_end.strftime('%b %d, %Y')}",
                    "average_sentiment": round(sum(scores) / len(scores), 3),
                    "message_count": len(scores)
                })
                
            # Calculate monthly averages
            monthly_averages = []
            for month, scores in sorted(monthly_sentiment.items()):
                year, month_num = month.split('-')
                month_obj = datetime(int(year), int(month_num), 1)
                monthly_averages.append({
                    "month": month,
                    "label": month_obj.strftime("%B %Y"),
                    "average_sentiment": round(sum(scores) / len(scores), 3),
                    "message_count": len(scores)
                })
            
            # Calculate sentiment by sender
            sender_averages = []
            for sender, scores in sender_sentiment.items():
                avg_sentiment = sum(scores) / len(scores)
                
                # Get display name for this sender
                if sender == "You":
                    display_name = "You"
                elif sender in participant_info:
                    display_name = participant_info[sender]
                else:
                    display_name = format_contact_name(sender)
                
                sender_averages.append({
                    "sender": sender,
                    "display_name": display_name,
                    "message_count": len(scores),
                    "average_sentiment": round(avg_sentiment, 3),
                    "sentiment_category": "positive" if avg_sentiment > 0.2 else "negative" if avg_sentiment < -0.2 else "neutral"
                })
            
            # Sort by message count
            sender_averages.sort(key=lambda x: x["message_count"], reverse=True)
            
            # Calculate overall sentiment stats
            sent_polarities = [item["polarity"] for item in sentiment_data]
            sent_subjectivities = [item["subjectivity"] for item in sentiment_data]
            
            # Determine sentiment trends (is it getting more positive/negative over time)
            trend = "neutral"
            if len(daily_averages) > 3:
                first_half = daily_averages[:len(daily_averages)//2]
                second_half = daily_averages[len(daily_averages)//2:]
                
                first_half_avg = sum(day["average_sentiment"] for day in first_half) / len(first_half) if first_half else 0
                second_half_avg = sum(day["average_sentiment"] for day in second_half) / len(second_half) if second_half else 0
                
                difference = second_half_avg - first_half_avg
                if difference > 0.1:
                    trend = "becoming more positive"
                elif difference < -0.1:
                    trend = "becoming more negative"
                else:
                    trend = "remaining steady"
            
            overall_sentiment = sum(sent_polarities) / len(sent_polarities) if sent_polarities else 0
            
            # Identify emotional high and low points (most positive/negative days)
            if daily_averages:
                daily_averages_sorted = sorted(daily_averages, key=lambda x: x["average_sentiment"])
                most_negative_day = daily_averages_sorted[0]
                most_positive_day = daily_averages_sorted[-1]
            else:
                most_negative_day = None
                most_positive_day = None
                
            # Find examples of highly positive and negative messages
            positive_examples = sorted([msg for msg in sentiment_data if msg["polarity"] > 0.5], 
                                     key=lambda x: x["polarity"], reverse=True)[:3]
            negative_examples = sorted([msg for msg in sentiment_data if msg["polarity"] < -0.5], 
                                     key=lambda x: x["polarity"])[:3]
            
            # Identify sentiment shifts (days where sentiment changed significantly from previous day)
            sentiment_shifts = []
            for i in range(1, len(daily_averages)):
                prev_sentiment = daily_averages[i-1]["average_sentiment"]
                curr_sentiment = daily_averages[i]["average_sentiment"]
                shift = curr_sentiment - prev_sentiment
                
                # If significant shift (> 0.3 in either direction)
                if abs(shift) > 0.3:
                    sentiment_shifts.append({
                        "from_date": daily_averages[i-1]["date"],
                        "to_date": daily_averages[i]["date"],
                        "shift": round(shift, 3),
                        "direction": "positive" if shift > 0 else "negative"
                    })
            
            # Prepare the comprehensive sentiment analysis
            return {
                "chat_name": chat_name,
                "message_count": len(messages),
                "analyzed_message_count": len(sentiment_data),
                "overall_sentiment": {
                    "score": round(overall_sentiment, 3),
                    "category": "positive" if overall_sentiment > 0.2 else "negative" if overall_sentiment < -0.2 else "neutral",
                    "subjectivity": round(sum(sent_subjectivities) / len(sent_subjectivities), 3) if sent_subjectivities else 0
                },
                "sentiment_trend": trend,
                "time_series": {
                    "daily": daily_averages,
                    "weekly": weekly_averages,
                    "monthly": monthly_averages
                },
                "participant_sentiment": sender_averages,
                "sentiment_breakdown": {
                    "positive": sum(1 for item in sentiment_data if item["sentiment"] == "positive"),
                    "neutral": sum(1 for item in sentiment_data if item["sentiment"] == "neutral"),
                    "negative": sum(1 for item in sentiment_data if item["sentiment"] == "negative")
                },
                "emotional_peaks": {
                    "most_positive_day": most_positive_day,
                    "most_negative_day": most_negative_day,
                    "positive_examples": positive_examples,
                    "negative_examples": negative_examples,
                    "significant_shifts": sentiment_shifts
                },
                "message_samples": sentiment_data[:20]  # Return just the first 20 for brevity
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            logger.error(traceback.format_exc())
            return error_response("ANALYSIS_ERROR", f"Error analyzing sentiment: {str(e)}")

    @mcp.tool()
    def analyze_conversation_flow(
        phone_number: str = None,
        chat_id: Union[str, int] = None,
        start_date: str = None,
        end_date: str = None
    ) -> Dict[str, Any]:
        """Analyze conversation patterns such as response times and interaction frequency."""
        # Validate parameters
        if not phone_number and not chat_id:
            return error_response("MISSING_PARAMETER", "Either phone_number or chat_id must be provided")
        
        # Validate date range
        error = validate_date_range(start_date, end_date)
        if error:
            return error
        
        try:
            db = MessagesDB()
            messages = []
            
            # Convert date strings to datetime objects
            start_dt = ensure_datetime(start_date) if start_date else None
            end_dt = ensure_datetime(end_date) if end_date else None
            
            # Get messages from appropriate source
            if chat_id:
                logger.info(f"Analyzing conversation flow for group chat ID: {chat_id}")
                messages = db.get_group_chat_messages(chat_id, start_dt, end_dt)
                # Get chat participants for group chats
                participants = db.get_chat_participants(chat_id)
                participant_ids = {p.get("identifier"): p.get("display_name", "Unknown") for p in participants}
            else:
                logger.info(f"Analyzing conversation flow for contact: {phone_number}")
                messages = list(db.iter_messages(phone_number, start_dt, end_dt))
                participant_ids = {phone_number: "Contact"}
            
            logger.info(f"Found {len(messages)} messages for conversation flow analysis")
            
            if not messages:
                logger.warning("No messages found for conversation flow analysis")
                return {
                    "warning": "No messages found for the specified parameters",
                    "message_count": 0
                }
            
            # Sort messages by date
            messages = sorted(messages, key=lambda m: m.get("date", "") if isinstance(m.get("date"), str) else "")
            
            # Initialize analysis containers
            hourly_distribution = {hour: 0 for hour in range(24)}
            daily_distribution = {day: 0 for day in range(7)}  # 0 = Monday, 6 = Sunday
            response_times = []
            message_lengths = []
            messages_per_day = defaultdict(int)
            conversation_sessions = []
            sender_stats = defaultdict(lambda: {"sent_count": 0, "total_chars": 0, "avg_response_time": [], "display_name": ""})
            current_session = {"start": None, "end": None, "messages": 0, "participants": set()}
            
            # Create a mapping of sender IDs to display names
            display_names = {}
            for sender_id in set([msg.get("sender_id", "Unknown") for msg in messages if not msg.get("is_from_me", False)]):
                if sender_id != "Unknown":
                    display_names[sender_id] = format_contact_name(sender_id)
            
            # Track messages for turn-taking analysis
            last_message_time = None
            last_sender = None
            session_gap = timedelta(hours=3)  # Define a new session if gap > 3 hours
            
            for i, msg in enumerate(messages):
                # Extract sender info
                is_from_me = msg.get("is_from_me", False)
                sender = "You" if is_from_me else msg.get("sender_id", "Unknown")
                text = msg.get("text", "")
                
                # Track message length
                msg_length = len(text)
                message_lengths.append(msg_length)
                
                # Update sender stats
                sender_stats[sender]["sent_count"] += 1
                sender_stats[sender]["total_chars"] += msg_length
                
                # Parse message date
                try:
                    date_obj = datetime.fromisoformat(msg.get("date", ""))
                    
                    # Hourly distribution
                    hourly_distribution[date_obj.hour] += 1
                    
                    # Daily distribution (weekday)
                    daily_distribution[date_obj.weekday()] += 1
                    
                    # Messages per day
                    day_key = date_obj.date().isoformat()
                    messages_per_day[day_key] += 1
                    
                    # Session tracking
                    if not current_session["start"]:
                        current_session["start"] = date_obj
                        current_session["participants"].add(sender)
                    else:
                        # Check if this is a new session
                        time_diff = date_obj - current_session["end"] if current_session["end"] else timedelta(0)
                        if time_diff > session_gap:
                            # Save previous session if it had at least 3 messages
                            if current_session["messages"] >= 3:
                                conversation_sessions.append({
                                    "start": current_session["start"].isoformat(),
                                    "end": current_session["end"].isoformat(),
                                    "duration_minutes": (current_session["end"] - current_session["start"]).seconds // 60,
                                    "message_count": current_session["messages"],
                                    "participants": list(current_session["participants"])
                                })
                            # Start new session
                            current_session = {
                                "start": date_obj,
                                "end": date_obj,
                                "messages": 1,
                                "participants": {sender}
                            }
                        else:
                            # Continue current session
                            current_session["end"] = date_obj
                            current_session["messages"] += 1
                            current_session["participants"].add(sender)
                    
                    # Response time analysis
                    if last_message_time and sender != last_sender:
                        # This is a response to the previous message
                        response_time = (date_obj - last_message_time).seconds / 60  # minutes
                        response_times.append(response_time)
                        sender_stats[sender]["avg_response_time"].append(response_time)
                    
                    # Update tracking variables
                    last_message_time = date_obj
                    last_sender = sender
                    
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not parse message date: {msg.get('date')}, error: {e}")
            
            # Add the last session if it exists and has enough messages
            if current_session["start"] and current_session["messages"] >= 3:
                conversation_sessions.append({
                    "start": current_session["start"].isoformat(),
                    "end": current_session["end"].isoformat(),
                    "duration_minutes": (current_session["end"] - current_session["start"]).seconds // 60,
                    "message_count": current_session["messages"],
                    "participants": list(current_session["participants"])
                })
            
            # Calculate averages for sender stats
            for sender, stats in sender_stats.items():
                if stats["sent_count"] > 0:
                    stats["avg_chars_per_message"] = round(stats["total_chars"] / stats["sent_count"], 1)
                if stats["avg_response_time"]:
                    stats["avg_response_time"] = round(sum(stats["avg_response_time"]) / len(stats["avg_response_time"]), 1)
                else:
                    stats["avg_response_time"] = None
                stats["percentage"] = round((stats["sent_count"] / len(messages)) * 100, 1)
            
            # Format daily message counts for time series
            daily_message_counts = [{"date": date, "count": count} 
                                   for date, count in sorted(messages_per_day.items())]
            
            # Calculate peak conversation hours
            peak_hours = sorted(hourly_distribution.items(), key=lambda x: x[1], reverse=True)[:5]
            peak_hours = [{"hour": hour, "count": count, "hour_formatted": f"{hour:02d}:00-{(hour+1)%24:02d}:00"} 
                         for hour, count in peak_hours if count > 0]
            
            # Calculate peak conversation days
            days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            peak_days = sorted([(day, count) for day, count in daily_distribution.items()], 
                              key=lambda x: x[1], reverse=True)
            peak_days = [{"day": days_of_week[day], "count": count} 
                        for day, count in peak_days if count > 0]
            
            # Calculate overall stats
            avg_response_time = round(sum(response_times) / len(response_times), 1) if response_times else None
            avg_messages_per_day = round(len(messages) / len(messages_per_day), 1) if messages_per_day else None
            
            # Analyze conversation intensity
            if len(daily_message_counts) > 1:
                # Find periods of high intensity
                high_intensity_days = [item for item in daily_message_counts 
                                      if item["count"] > avg_messages_per_day * 1.5]
                
                # Find the most active day
                most_active_day = max(daily_message_counts, key=lambda x: x["count"]) if daily_message_counts else None
            else:
                high_intensity_days = []
                most_active_day = daily_message_counts[0] if daily_message_counts else None
            
            # Format sender stats for output
            formatted_sender_stats = []
            for sender, stats in sender_stats.items():
                # Get contact name if available
                contact_name = None
                if sender not in ["You", "System"]:
                    contact_name = db.get_contact_name(sender)
                
                display_name = "You" if sender == "You" else (
                    "System" if sender == "System" else 
                    format_contact_name(sender, contact_name)
                )
                
                participant_data = {
                    "sender": sender,
                    "display_name": display_name,
                    "message_count": stats["sent_count"],
                    "percentage": stats["percentage"],
                    "avg_chars_per_message": stats["avg_chars_per_message"],
                    "avg_response_time_minutes": stats["avg_response_time"]
                }
                formatted_sender_stats.append(participant_data)
            
            # Sort by message count
            formatted_sender_stats.sort(key=lambda x: x["message_count"], reverse=True)
            
            # Prepare result
            result = {
                "message_count": len(messages),
                "date_range": {
                    "start": messages[0].get("date") if messages else None,
                    "end": messages[-1].get("date") if messages else None
                },
                "conversation_pace": {
                    "avg_response_time_minutes": avg_response_time,
                    "fastest_responder": min(formatted_sender_stats, 
                                            key=lambda x: x["avg_response_time_minutes"] if x["avg_response_time_minutes"] else float('inf'))["sender"] 
                                            if any(s["avg_response_time_minutes"] for s in formatted_sender_stats) else None,
                    "avg_messages_per_day": avg_messages_per_day,
                    "peak_hours": peak_hours,
                    "peak_days": peak_days
                },
                "participant_stats": formatted_sender_stats,
                "conversation_intensity": {
                    "most_active_day": most_active_day,
                    "high_intensity_days": high_intensity_days[:5],  # Limit to top 5
                    "conversation_sessions": sorted(conversation_sessions, key=lambda x: x["message_count"], reverse=True)[:5]  # Top 5 sessions
                },
                "message_length": {
                    "average": round(sum(message_lengths) / len(message_lengths), 1) if message_lengths else 0,
                    "min": min(message_lengths) if message_lengths else 0,
                    "max": max(message_lengths) if message_lengths else 0
                },
                "daily_message_counts": daily_message_counts
            }
            
            return result
        except Exception as e:
            logger.error(f"Error analyzing conversation flow: {e}")
            logger.error(traceback.format_exc())
            return error_response("ANALYSIS_ERROR", f"Error analyzing conversation flow: {str(e)}")

    @mcp.tool()
    def analyze_attachments(
        phone_number: str = None,
        chat_id: Union[str, int] = None,
        start_date: str = None,
        end_date: str = None
    ) -> Dict[str, Any]:
        """Analyze attachments shared in messages."""
        # Validate parameters
        if not phone_number and not chat_id:
            return error_response("MISSING_PARAMETER", "Either phone_number or chat_id must be provided")
        
        # Validate date range
        error = validate_date_range(start_date, end_date)
        if error:
            return error
        
        try:
            db = MessagesDB()
            messages = []
            
            # Convert date strings to datetime objects
            start_dt = ensure_datetime(start_date) if start_date else None
            end_dt = ensure_datetime(end_date) if end_date else None
            
            # Get messages from appropriate source
            if chat_id:
                logger.info(f"Analyzing attachments for group chat ID: {chat_id}")
                messages = db.get_group_chat_messages(chat_id, start_dt, end_dt)
                # Get chat participants for group chats
                participants = db.get_chat_participants(chat_id)
                participant_ids = {p.get("identifier"): p.get("display_name", "Unknown") for p in participants}
            else:
                logger.info(f"Analyzing attachments for contact: {phone_number}")
                messages = list(db.iter_messages(phone_number, start_dt, end_dt))
                participant_ids = {phone_number: format_contact_name(phone_number)}
            
            logger.info(f"Found {len(messages)} messages to analyze for attachments")
            
            # Initialize tracking containers
            attachment_types = Counter()
            attachment_by_sender = defaultdict(Counter)
            sender_display_names = {}  # Mapping of sender IDs to display names
            attachment_by_date = defaultdict(Counter)
            total_attachment_count = 0
            largest_attachment = {"size": 0, "type": None, "sender": None, "date": None}
            attachment_details = []
            daily_attachment_counts = defaultdict(int)
            
            # Helper function to determine file type from filename or mimetype
            def get_file_type(filename, mime_type=None):
                if not filename and not mime_type:
                    return "unknown"
                    
                if filename:
                    ext = filename.split('.')[-1].lower() if '.' in filename else ""
                    
                    # Image types
                    if ext in ['jpg', 'jpeg', 'png', 'gif', 'heic', 'heif', 'tiff', 'bmp', 'webp']:
                        return "image"
                        
                    # Video types
                    elif ext in ['mp4', 'mov', 'avi', 'wmv', 'flv', 'mkv', 'webm', 'm4v', '3gp']:
                        return "video"
                        
                    # Audio types
                    elif ext in ['mp3', 'm4a', 'wav', 'ogg', 'flac', 'aac', 'wma']:
                        return "audio"
                        
                    # Document types
                    elif ext in ['pdf', 'doc', 'docx', 'txt', 'rtf', 'odt', 'pages']:
                        return "document"
                        
                    # Spreadsheet types
                    elif ext in ['xls', 'xlsx', 'csv', 'numbers', 'ods']:
                        return "spreadsheet"
                        
                    # Presentation types
                    elif ext in ['ppt', 'pptx', 'key', 'odp']:
                        return "presentation"
                        
                    # Archive types
                    elif ext in ['zip', 'rar', '7z', 'tar', 'gz', 'bz2']:
                        return "archive"
                        
                    # Contact card
                    elif ext in ['vcf']:
                        return "contact_card"
                        
                    # Calendar invite
                    elif ext in ['ics']:
                        return "calendar"
                        
                    # Code files
                    elif ext in ['py', 'js', 'html', 'css', 'java', 'cpp', 'c', 'php', 'rb', 'go', 'swift']:
                        return "code"
                        
                    else:
                        return ext or "unknown"
                
                # Use MIME type if available and no extension found
                if mime_type:
                    if 'image/' in mime_type:
                        return "image"
                    elif 'video/' in mime_type:
                        return "video"
                    elif 'audio/' in mime_type:
                        return "audio"
                    elif 'application/pdf' in mime_type:
                        return "document"
                    elif 'text/' in mime_type:
                        return "document"
                    else:
                        return mime_type.split('/')[-1]
                
                return "unknown"
            
            # Process each message
            for msg in messages:
                # Extract attachments if any
                attachments = msg.get("attachments", [])
                if not attachments:
                    continue
                    
                # Extract sender info
                is_from_me = msg.get("is_from_me", False)
                sender = "You" if is_from_me else msg.get("sender_id", "Unknown")
                
                # Track sender display name if not already stored
                if sender not in sender_display_names:
                    if sender == "You":
                        sender_display_names[sender] = "You"
                    elif sender in participant_ids:
                        sender_display_names[sender] = participant_ids[sender]
                    else:
                        sender_display_names[sender] = format_contact_name(sender)
                
                # Parse message date
                date_str = msg.get("date")
                date_key = None
                if date_str:
                    try:
                        date_obj = datetime.fromisoformat(date_str)
                        date_key = date_obj.date().isoformat()
                        # Track daily attachment counts
                        daily_attachment_counts[date_key] += len(attachments)
                    except (ValueError, TypeError):
                        pass
                
                # Process each attachment
                for attachment in attachments:
                    total_attachment_count += 1
                    
                    # Extract attachment details
                    filename = attachment.get("filename", "")
                    mime_type = attachment.get("mime_type", "")
                    file_size = attachment.get("file_size", 0)
                    transfer_name = attachment.get("transfer_name", "")
                    
                    # Determine file type
                    file_type = get_file_type(filename or transfer_name, mime_type)
                    
                    # Update counters
                    attachment_types[file_type] += 1
                    attachment_by_sender[sender][file_type] += 1
                    if date_key:
                        attachment_by_date[date_key][file_type] += 1
                    
                    # Check if this is the largest attachment
                    if file_size and file_size > largest_attachment["size"]:
                        largest_attachment = {
                            "size": file_size,
                            "type": file_type,
                            "sender": sender,
                            "sender_display_name": sender_display_names.get(sender, sender),
                            "date": date_str,
                            "filename": filename or transfer_name
                        }
                    
                    # Add to detailed list (limited to recent/interesting ones)
                    if len(attachment_details) < 50:  # Limit to 50 attachments in results
                        attachment_details.append({
                            "type": file_type,
                            "filename": filename or transfer_name,
                            "size_bytes": file_size,
                            "size_formatted": f"{file_size / (1024*1024):.2f} MB" if file_size > 1024*1024 else f"{file_size / 1024:.2f} KB" if file_size > 1024 else f"{file_size} bytes",
                            "sender": sender,
                            "date": date_str,
                            "sender_display_name": sender_display_names.get(sender, sender)
                        })
            
            # No attachments found
            if total_attachment_count == 0:
                return {
                    "warning": "No attachments found in the specified conversation",
                    "message_count": len(messages),
                    "attachment_count": 0
                }
            
            # Prepare sender attachment summary
            sender_attachment_summary = []
            for sender, type_counter in attachment_by_sender.items():
                total_for_sender = sum(type_counter.values())
                sender_attachment_summary.append({
                    "sender": sender,
                    "display_name": sender_display_names.get(sender, sender),
                    "total_attachments": total_for_sender,
                    "percentage": round(total_for_sender / total_attachment_count * 100, 1),
                    "type_breakdown": {file_type: count for file_type, count in type_counter.items()}
                })
            
            # Sort by total attachments
            sender_attachment_summary.sort(key=lambda x: x["total_attachments"], reverse=True)
            
            # Prepare time series data
            time_series = []
            for date, type_counts in sorted(attachment_by_date.items()):
                total_for_day = sum(type_counts.values())
                time_series.append({
                    "date": date,
                    "total_count": total_for_day,
                    "type_breakdown": {file_type: count for file_type, count in type_counts.items()}
                })
            
            # Find days with most attachments
            if time_series:
                peak_attachment_days = sorted(time_series, key=lambda x: x["total_count"], reverse=True)[:5]
            else:
                peak_attachment_days = []
            
            # Calculate attachment frequency 
            days_with_attachments = len(daily_attachment_counts)
            attachment_frequency = round(total_attachment_count / days_with_attachments, 2) if days_with_attachments > 0 else 0
            
            # Get most common attachment types
            common_types = attachment_types.most_common()
            
            # Format for output
            return {
                "message_count": len(messages),
                "attachment_count": total_attachment_count,
                "unique_attachment_days": days_with_attachments,
                "attachments_per_day": attachment_frequency,
                "type_distribution": {file_type: count for file_type, count in common_types},
                "top_attachment_types": [{"type": file_type, "count": count, "percentage": round(count/total_attachment_count*100, 1)} 
                                                for file_type, count in common_types[:5]] if common_types else [],
                "sender_analysis": sender_attachment_summary,
                "peak_attachment_days": peak_attachment_days,
                "largest_attachment": largest_attachment if largest_attachment["size"] > 0 else None,
                "time_series": time_series,
                "recent_attachments": sorted(attachment_details, key=lambda x: x.get("date", ""), reverse=True)[:20]
            }
        except Exception as e:
            logger.error(f"Error analyzing attachments: {e}")
            logger.error(traceback.format_exc())
            return error_response("ANALYSIS_ERROR", f"Error analyzing attachments: {str(e)}")

    @mcp.tool()
    def extract_topics(
        phone_number: str = None,
        chat_id: Union[str, int] = None,
        start_date: str = None,
        end_date: str = None,
        num_topics: int = 5
    ) -> Dict[str, Any]:
        """Extract key conversation topics using NLP techniques.
        
        Args:
            phone_number: Optional phone number to filter messages
            chat_id: Optional chat ID to filter messages from a group chat
            start_date: Optional start date in ISO format (YYYY-MM-DD)
            end_date: Optional end date in ISO format (YYYY-MM-DD)
            num_topics: Number of topics to extract (default: 5)
            
        Returns:
            Dictionary containing extracted topics and related messages
        """
        logger.info(f"extract_topics called with phone_number={phone_number}, chat_id={chat_id}, start_date={start_date}, end_date={end_date}, num_topics={num_topics}")
        
        # Validate parameters
        if not phone_number and not chat_id:
            return error_response("MISSING_PARAMETER", "Either phone_number or chat_id must be provided")
            
        if num_topics <= 0 or num_topics > 50:
            return error_response("INVALID_PARAMETER", "Number of topics must be between 1 and 50")
        
        # Validate date range
        error = validate_date_range(start_date, end_date)
        if error:
            return error
        
        start_dt = None
        end_dt = None
        
        if start_date:
            start_dt = ensure_datetime(start_date)
        if end_date:
            end_dt = ensure_datetime(end_date)
        
        try:
            db = MessagesDB()
            messages = []
            
            # Fetch messages based on parameters
            if phone_number:
                # Format the phone number
                formatted_number = phone_number
                try:
                    parsed_number = phonenumbers.parse(phone_number, "US")
                    formatted_number = phonenumbers.format_number(
                        parsed_number, phonenumbers.PhoneNumberFormat.E164
                    )
                except:
                    pass
                
                messages_iter = db.iter_messages(formatted_number, start_dt, end_dt)
                messages = list(messages_iter)
                logger.info(f"Found {len(messages)} messages for topic extraction")
                
            elif chat_id:
                actual_chat_id, error = resolve_chat_id(chat_id)
                if error:
                    return error
                
                messages = db.get_group_chat_messages(actual_chat_id, start_dt, end_dt)
                logger.info(f"Found {len(messages)} messages for topic extraction in group chat")
            
            if not messages:
                return {
                    "topics": [],
                    "topic_count": 0,
                    "message_count": 0,
                    "time_period": {
                        "start_date": start_date,
                        "end_date": end_date
                    }
                }
            
            # Extract only text messages that are sufficiently long
            text_messages = []
            for msg in messages:
                text = msg.get("text", "")
                if text and len(text.strip()) > 10:  # Skip very short messages
                    text_messages.append({
                        "text": text,
                        "date": msg.get("date"),
                        "is_from_me": msg.get("is_from_me", False),
                        "sender_id": msg.get("sender_id")
                    })
            
            if not text_messages:
                return {
                    "topics": [],
                    "topic_count": 0,
                    "message_count": len(messages),
                    "time_period": {
                        "start_date": start_date,
                        "end_date": end_date
                    }
                }
            
            # Extract key phrases and ngrams
            all_text = " ".join([msg["text"] for msg in text_messages])
            blob = TextBlob(all_text)
            
            # Extract noun phrases as potential topics
            noun_phrases = blob.noun_phrases
            
            # Count word frequency
            word_counts = Counter()
            for word in blob.words:
                # Skip short words and common stop words
                if len(word) > 3 and word.lower() not in STOP_WORDS:
                    word_counts[word.lower()] += 1
            
            # Generate bigrams for additional context
            bigrams = []
            for i in range(len(blob.words) - 1):
                bigram = f"{blob.words[i]} {blob.words[i+1]}"
                if len(bigram) > 7:  # Skip very short bigrams
                    bigrams.append(bigram.lower())
            
            bigram_counts = Counter(bigrams)
            
            # Combine noun phrases and frequent words/bigrams to identify topics
            topic_candidates = list(noun_phrases)
            
            # Add common words
            topic_candidates.extend([word for word, count in word_counts.most_common(num_topics * 2) 
                                    if count > 1 and word not in STOP_WORDS])
            
            # Add common bigrams
            topic_candidates.extend([bigram for bigram, count in bigram_counts.most_common(num_topics) 
                                    if count > 1])
            
            # Remove duplicates and limit to requested number
            unique_topics = []
            for topic in topic_candidates:
                if topic not in unique_topics and len(unique_topics) < num_topics:
                    # Check if it's not a subset of an existing topic
                    if not any(topic in existing_topic for existing_topic in unique_topics):
                        unique_topics.append(topic)
            
            # Find representative messages for each topic
            topic_data = []
            for topic in unique_topics:
                topic_messages = []
                for msg in text_messages:
                    if topic in msg["text"].lower():
                        # Format the sender information
                        is_from_me = msg.get("is_from_me", False)
                        
                        if phone_number:
                            sender = "You" if is_from_me else format_contact_name(phone_number)
                        else:
                            sender_id = msg.get("sender_id")
                            sender = "You" if is_from_me else format_contact_name(sender_id, db.get_contact_name(sender_id))
                        
                        topic_messages.append({
                            "text": msg["text"],
                            "date": msg["date"],
                            "sender": sender
                        })
                
                if topic_messages:
                    # Sort by date
                    topic_messages.sort(key=lambda x: x["date"] if x["date"] else "", reverse=True)
                    
                    # Limit to 3 representative messages
                    topic_data.append({
                        "topic": topic,
                        "message_count": len(topic_messages),
                        "sample_messages": topic_messages[:3]
                    })
            
            # Sort topics by message count
            topic_data.sort(key=lambda x: x["message_count"], reverse=True)
            
            return {
                "topics": topic_data,
                "topic_count": len(topic_data),
                "message_count": len(messages),
                "time_period": {
                    "start_date": start_date,
                    "end_date": end_date
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            logger.error(traceback.format_exc())
            return error_response("ANALYSIS_ERROR", f"Error extracting topics: {str(e)}")

# Function 2: Replace the existing analyze_contact_network() placeholder
@mcp.tool()
def analyze_contact_network(
    start_date: str = None,
    end_date: str = None,
    min_shared_chats: int = 1
) -> Dict[str, Any]:
    """Analyze the network of contacts based on group chat participation.
    
    This function maps connections between contacts by identifying who appears together
    in the same group chats, creating a social graph of your messaging network.
    
    Args:
        start_date: Optional start date in ISO format (YYYY-MM-DD) to limit analysis timeframe
        end_date: Optional end date in ISO format (YYYY-MM-DD) to limit analysis timeframe
        min_shared_chats: Minimum number of shared chats required to consider contacts connected (default: 1)
        
    Returns:
        Dictionary containing contact network analysis including nodes, connections, and clusters
    """
    logger.info(f"analyze_contact_network called with start_date={start_date}, end_date={end_date}, min_shared_chats={min_shared_chats}")
    
    # Validate date range
    error = validate_date_range(start_date, end_date)
    if error:
        return error
    
    try:
        db = MessagesDB()
        
        # Convert date strings to datetime objects
        start_dt = ensure_datetime(start_date) if start_date else None
        end_dt = ensure_datetime(end_date) if end_date else None
        
        # Get all group chats
        group_chats = db.get_group_chats()
        if not group_chats:
            return {
                "warning": "No group chats found",
                "nodes": [],
                "connections": [],
                "clusters": []
            }
        
        # Track which contacts appear in which group chats
        contact_to_chats = defaultdict(set)
        chat_to_contacts = defaultdict(set)
        total_contacts = set()
        
        # For each group chat, get its participants
        for chat in group_chats:
            chat_id = chat["chat_id"]
            
            # Apply date filter if specified
            if start_dt or end_dt:
                # Check if the chat has activity in the specified date range
                if chat.get("last_message_date"):
                    try:
                        last_message = datetime.fromisoformat(chat["last_message_date"])
                        if (start_dt and last_message < start_dt) or (end_dt and last_message > end_dt):
                            # Skip this chat as it's outside our date range
                            continue
                    except (ValueError, TypeError):
                        # If we can't parse the date, include the chat by default
                        pass
            
            # Get participants for this chat
            participants = db.get_chat_participants(chat_id)
            for participant in participants:
                participant_id = participant.get("id")
                if participant_id:
                    contact_to_chats[participant_id].add(chat_id)
                    chat_to_contacts[chat_id].add(participant_id)
                    total_contacts.add(participant_id)
        
        # Calculate connections between contacts (edges in the graph)
        contact_connections = []
        contact_pairs_processed = set()  # To avoid duplicate connections
        
        for contact_a in total_contacts:
            for contact_b in total_contacts:
                # Skip self-connections and duplicates
                if contact_a == contact_b or (contact_a, contact_b) in contact_pairs_processed or (contact_b, contact_a) in contact_pairs_processed:
                    continue
                
                # Find shared chats
                shared_chats = contact_to_chats[contact_a].intersection(contact_to_chats[contact_b])
                
                # Only create a connection if they share enough chats
                if len(shared_chats) >= min_shared_chats:
                    # Get display names for both contacts
                    contact_a_name = db.get_contact_name(contact_a) or contact_a
                    contact_b_name = db.get_contact_name(contact_b) or contact_b
                    
                    display_a = format_contact_name(contact_a, contact_a_name)
                    display_b = format_contact_name(contact_b, contact_b_name)
                    
                    contact_connections.append({
                        "source": contact_a,
                        "source_display_name": display_a,
                        "target": contact_b,
                        "target_display_name": display_b,
                        "shared_chats": len(shared_chats),
                        "chat_ids": list(shared_chats)
                    })
                    
                    # Mark this pair as processed
                    contact_pairs_processed.add((contact_a, contact_b))
        
        # Identify clusters/communities in the network
        # We'll use a simple approach: connect contacts who share at least one other contact
        clusters = []
        contacts_processed = set()
        
        # Helper function to find contacts in a cluster
        def find_cluster_members(seed_contact, current_cluster):
            """Recursively find all members of a cluster starting from a seed contact."""
            if seed_contact in contacts_processed:
                return
                
            current_cluster.add(seed_contact)
            contacts_processed.add(seed_contact)
            
            # Find all contacts connected to this one
            connected_contacts = set()
            for conn in contact_connections:
                if conn["source"] == seed_contact:
                    connected_contacts.add(conn["target"])
                elif conn["target"] == seed_contact:
                    connected_contacts.add(conn["source"])
            
            # Recursively add connected contacts to cluster
            for contact in connected_contacts:
                if contact not in contacts_processed:
                    find_cluster_members(contact, current_cluster)
        
        # Identify clusters
        for contact in total_contacts:
            if contact not in contacts_processed:
                current_cluster = set()
                find_cluster_members(contact, current_cluster)
                
                if current_cluster:
                    # Get display names for contacts in this cluster
                    cluster_members = []
                    for member in current_cluster:
                        name = db.get_contact_name(member) or member
                        display_name = format_contact_name(member, name)
                        
                        # Get number of connections for this member
                        connection_count = sum(1 for conn in contact_connections 
                                          if conn["source"] == member or conn["target"] == member)
                        
                        cluster_members.append({
                            "id": member,
                            "display_name": display_name,
                            "connections": connection_count,
                            "groups": len(contact_to_chats[member])
                        })
                    
                    # Sort cluster members by connection count
                    cluster_members.sort(key=lambda x: x["connections"], reverse=True)
                    
                    clusters.append({
                        "size": len(cluster_members),
                        "members": cluster_members,
                        "total_connections": len([c for c in contact_connections 
                                               if c["source"] in current_cluster and c["target"] in current_cluster])
                    })
        
        # Sort clusters by size
        clusters.sort(key=lambda x: x["size"], reverse=True)
        
        # Create nodes for the network graph
        nodes = []
        for contact in total_contacts:
            contact_name = db.get_contact_name(contact) or contact
            display_name = format_contact_name(contact, contact_name)
            
            # Count connections for this contact
            connection_count = sum(1 for conn in contact_connections 
                               if conn["source"] == contact or conn["target"] == contact)
            
            # Count groups this contact is in
            group_count = len(contact_to_chats[contact])
            
            nodes.append({
                "id": contact,
                "display_name": display_name,
                "connection_count": connection_count,
                "group_count": group_count
            })
        
        # Sort nodes by connection count
        nodes.sort(key=lambda x: x["connection_count"], reverse=True)
        
        # Sort connections by shared chat count
        contact_connections.sort(key=lambda x: x["shared_chats"], reverse=True)
        
        # Generate network statistics
        stats = {
            "total_contacts": len(total_contacts),
            "total_connections": len(contact_connections),
            "total_clusters": len(clusters),
            "average_connections_per_contact": round(len(contact_connections) * 2 / len(total_contacts), 2) if total_contacts else 0,
            "most_connected_contact": nodes[0]["display_name"] if nodes else None,
            "largest_cluster_size": clusters[0]["size"] if clusters else 0
        }
        
        return {
            "nodes": nodes,
            "connections": contact_connections,
            "clusters": clusters,
            "stats": stats,
            "date_range": {
                "start": start_date,
                "end": end_date
            }
        }
        
    except Exception as e:
        logger.error(f"Error analyzing contact network: {e}")
        logger.error(traceback.format_exc())
        return error_response("ANALYSIS_ERROR", f"Error analyzing contact network: {str(e)}")

    @mcp.tool()
    def search_messages(
        query: str,
        phone_number: str = None,
        chat_id: Union[str, int] = None,
        start_date: str = None,
        end_date: str = None
    ) -> Dict[str, Any]:
        """Search messages with advanced query capabilities and relevance ranking."""
        # Validate parameters
        if not query or query.strip() == "":
            return error_response("MISSING_PARAMETER", "Search query is required")
        
        # Either phone_number or chat_id can be provided, but at least one must be provided for targeted search
        if not phone_number and not chat_id:
            logger.info("No phone_number or chat_id provided, searching across all messages")
        
        # Validate date range
        error = validate_date_range(start_date, end_date)
        if error:
            return error
        
        try:
            db = MessagesDB()
            
            # Convert date strings to datetime objects
            start_dt = ensure_datetime(start_date) if start_date else None
            end_dt = ensure_datetime(end_date) if end_date else None
            
            # Clean and prepare search query
            search_terms = query.lower().strip()
            logger.info(f"Searching for: '{search_terms}'")
            
            # Build list of messages to search based on parameters
            messages_to_search = []
            
            if chat_id:
                logger.info(f"Searching in group chat ID: {chat_id}")
                messages = db.get_group_chat_messages(chat_id, start_dt, end_dt)
                messages_to_search.extend(messages)
                
                # Get participant info for group chat
                try:
                    participants = db.get_chat_participants(chat_id)
                    logger.info(f"Found {len(participants)} participants in group chat")
                except Exception as e:
                    logger.warning(f"Error getting group chat participants: {e}")
                    participants = []
                    
            elif phone_number:
                logger.info(f"Searching in chat with contact: {phone_number}")
                messages = list(db.iter_messages(phone_number, start_dt, end_dt))
                messages_to_search.extend(messages)
            else:
                # Search across all chats (could be a large dataset)
                logger.info("Searching across all messages - this might take a while")
                
                # Get all contacts first
                contacts = db.get_contacts()
                total_message_count = 0
                
                # Set a limit to prevent searching too many messages
                MAX_MESSAGES_TO_SEARCH = 10000
                
                for contact in contacts:
                    # Skip if we've already searched too many messages
                    if total_message_count >= MAX_MESSAGES_TO_SEARCH:
                        logger.warning(f"Reached search limit of {MAX_MESSAGES_TO_SEARCH} messages")
                        break
                        
                    contact_id = contact.get("identifier")
                    if not contact_id:
                        continue
                        
                    # Get messages for this contact
                    try:
                        contact_messages = list(db.iter_messages(contact_id, start_dt, end_dt))
                        messages_to_search.extend(contact_messages)
                        total_message_count += len(contact_messages)
                    except Exception as e:
                        logger.warning(f"Error getting messages for contact {contact_id}: {e}")
            
            logger.info(f"Searching through {len(messages_to_search)} messages")
            
            # Initialize search result containers
            search_results = []
            message_dates = set()
            message_senders = set()
            
            # Simple function to calculate relevance score
            def calculate_relevance(text, query_terms):
                text_lower = text.lower()
                
                # Exact match has highest relevance
                if query_terms in text_lower:
                    # Boost score if the match is a whole word or phrase
                    word_boundaries = r'\b' + re.escape(query_terms) + r'\b'
                    if re.search(word_boundaries, text_lower):
                        return 10.0
                    return 8.0
                
                # Split into individual terms for partial matching
                terms = query_terms.split()
                
                # Count how many terms match
                matched_terms = sum(1 for term in terms if term in text_lower)
                
                # Calculate percentage of terms matched
                if terms:
                    match_percentage = matched_terms / len(terms)
                else:
                    match_percentage = 0
                    
                # Score based on percentage of terms matched
                if match_percentage == 1.0:  # All terms match (but not as a phrase)
                    return 7.0
                elif match_percentage >= 0.7:  # Most terms match
                    return 5.0
                elif match_percentage >= 0.5:  # Half or more terms match
                    return 3.0
                elif match_percentage > 0:  # Some terms match
                    return 1.0
                
                return 0.0  # No match
            
            # Search through the messages
            for msg in messages_to_search:
                text = msg.get("text", "")
                if not text:
                    continue
                    
                # Calculate relevance score
                relevance_score = calculate_relevance(text, search_terms)
                
                # Skip messages that don't match
                if relevance_score <= 0:
                    continue
                    
                # Extract message details
                is_from_me = msg.get("is_from_me", False)
                sender = "You" if is_from_me else msg.get("sender_id", "Unknown")
                date_str = msg.get("date", "")
                
                # Get display name for sender
                display_name = sender
                if sender != "You" and not sender.startswith("Unknown"):
                    display_name = format_contact_name(sender)
                
                # Track unique dates and senders for analytics
                if date_str:
                    message_dates.add(date_str.split("T")[0] if "T" in date_str else date_str)
                message_senders.add(sender)
                
                # For long messages, highlight the matching part
                if len(text) > 200:
                    # Try to find and highlight the matching context
                    match_pos = text.lower().find(search_terms.lower())
                    if match_pos >= 0:
                        # Extract a snippet around the match
                        start_pos = max(0, match_pos - 60)
                        end_pos = min(len(text), match_pos + len(search_terms) + 60)
                        snippet = text[start_pos:end_pos]
                        # Add ellipsis if we truncated the text
                        prefix = "..." if start_pos > 0 else ""
                        suffix = "..." if end_pos < len(text) else ""
                        highlighted_text = f"{prefix}{snippet}{suffix}"
                    else:
                        # If we can't find the exact match (e.g. for partial matches), just truncate
                        highlighted_text = text[:200] + "..."
                else:
                    highlighted_text = text
                
                # Add to search results
                search_results.append({
                    "text": text,
                    "highlighted_text": highlighted_text,
                    "date": date_str,
                    "sender": sender,
                    "display_name": display_name,
                    "is_from_me": is_from_me,
                    "relevance": relevance_score,
                    "has_attachments": bool(msg.get("attachments"))
                })
            
            # Sort search results by relevance score
            search_results.sort(key=lambda x: x["relevance"], reverse=True)
            
            # Generate stats for search
            result_stats = {
                "total_messages_searched": len(messages_to_search),
                "total_matches": len(search_results),
                "unique_dates": len(message_dates),
                "unique_senders": len(message_senders),
                "query": search_terms
            }
            
            # Add time period info if available
            if start_date or end_date:
                result_stats["time_period"] = {
                    "start_date": start_date,
                    "end_date": end_date
                }
            
            # Build response with results and stats
            return {
                "query": search_terms,
                "results": search_results[:100],  # Limit to 100 results to prevent large payloads
                "total_results": len(search_results),
                "has_more": len(search_results) > 100,
                "stats": result_stats
            }
        except Exception as e:
            logger.error(f"Error searching messages: {e}")
            logger.error(traceback.format_exc())
            return error_response("SEARCH_ERROR", f"Error searching messages: {str(e)}")

    @mcp.tool()
    def summarize_conversation(
        phone_number: str = None,
        chat_id: Union[str, int] = None,
        start_date: str = None,
        end_date: str = None
    ) -> Dict[str, Any]:
        """Generate a concise summary of conversation key points."""
        # Validate parameters
        if not phone_number and not chat_id:
            return error_response("MISSING_PARAMETER", "Either phone_number or chat_id must be provided")
        
        # Validate date range
        error = validate_date_range(start_date, end_date)
        if error:
            return error
        
        try:
            db = MessagesDB()
            messages = []
            
            # Convert date strings to datetime objects
            start_dt = ensure_datetime(start_date) if start_date else None
            end_dt = ensure_datetime(end_date) if end_date else None
            
            # Get messages from appropriate source
            if chat_id:
                logger.info(f"Summarizing conversation for group chat ID: {chat_id}")
                messages = db.get_group_chat_messages(chat_id, start_dt, end_dt)
                # Get chat info
                try:
                    with db.get_db_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT display_name FROM chat WHERE ROWID = ? OR guid = ?", [chat_id, chat_id])
                        result = cursor.fetchone()
                        chat_name = result[0] if result else "Unknown Group"
                except Exception as e:
                    logger.warning(f"Error getting group chat name: {e}")
                    chat_name = "Group Chat"
                
                # Get participant info for group chats
                try:
                    participants = db.get_chat_participants(chat_id)
                    logger.info(f"Found {len(participants)} participants in group chat")
                    participant_info = {p.get("identifier"): p.get("display_name", "Unknown") for p in participants}
                except Exception as e:
                    logger.warning(f"Error getting group chat participants: {e}")
                    participant_info = {}
                    
            else:
                logger.info(f"Summarizing conversation with contact: {phone_number}")
                messages = list(db.iter_messages(phone_number, start_dt, end_dt))
                contact_name = db.get_contact_name(phone_number)
                display_name = format_contact_name(phone_number)
                chat_name = f"Chat with {display_name}"
                participant_info = {phone_number: display_name}
            
            # Need at least 5 messages for meaningful summarization
            if len(messages) < 5:
                return {
                    "warning": "Not enough messages to generate a meaningful summary",
                    "message_count": len(messages),
                    "chat_name": chat_name
                }
                
            logger.info(f"Summarizing {len(messages)} messages")
            
            # Sort messages by date
            messages = sorted(messages, key=lambda m: m.get("date", "") if isinstance(m.get("date"), str) else "")
            
            # Extract key conversation metrics
            total_messages = len(messages)
            first_message_date = messages[0].get("date") if messages else None
            last_message_date = messages[-1].get("date") if messages else None
            
            # Count messages by sender
            sender_counts = Counter()
            sender_display_names = {}
            for msg in messages:
                is_from_me = msg.get("is_from_me", False)
                sender = "You" if is_from_me else msg.get("sender_id", "Unknown")
                sender_counts[sender] += 1
                
                # Store display name for this sender if not already stored
                if sender not in sender_display_names:
                    if sender == "You":
                        sender_display_names[sender] = "You"
                    elif sender in participant_info:
                        sender_display_names[sender] = participant_info[sender]
                    else:
                        sender_display_names[sender] = format_contact_name(sender)
            
            # Get top participants
            top_participants = []
            for sender, count in sender_counts.most_common(5):
                top_participants.append({
                    "sender": sender,
                    "display_name": sender_display_names.get(sender, sender),
                    "message_count": count
                })
            
            # Extract important messages (longer messages often contain key information)
            message_scores = []
            for msg in messages:
                text = msg.get("text", "")
                score = 0
                
                # Longer messages get higher scores
                score += min(len(text) / 20, 5)  # Up to 5 points for length
                
                # Messages with question marks might be questions
                if "?" in text:
                    score += 1
                
                # Messages with URLs might contain important links
                if "http://" in text or "https://" in text:
                    score += 2
                
                # Messages with exclamation marks might be important
                if "!" in text:
                    score += 0.5
                
                # Messages with specific keywords get higher scores
                important_keywords = ["meeting", "important", "urgent", "deadline", "tomorrow", 
                                     "schedule", "appointment", "reminder", "don't forget", 
                                     "plan", "event", "update", "news", "announcement"]
                
                for keyword in important_keywords:
                    if keyword in text.lower():
                        score += 1
                
                # Add to scored messages if it has a meaningful score
                if score >= 2:
                    sender = "You" if msg.get("is_from_me", False) else msg.get("sender_id", "Unknown")
                    display_name = sender
                    if sender != "You" and not sender.startswith("Unknown"):
                        if sender in participant_info:
                            display_name = participant_info[sender]
                        else:
                            display_name = format_contact_name(sender)
                            
                    message_scores.append({
                        "text": text,
                        "score": score,
                        "date": msg.get("date"),
                        "sender": sender,
                        "display_name": display_name
                    })
            
            # Sort by score and take top messages
            key_messages = sorted(message_scores, key=lambda x: x["score"], reverse=True)[:10]
            
            # Group messages by day for conversation timeline
            messages_by_day = defaultdict(int)
            for msg in messages:
                date_str = msg.get("date")
                if date_str:
                    try:
                        date_obj = datetime.fromisoformat(date_str)
                        day_key = date_obj.date().isoformat()
                        messages_by_day[day_key] += 1
                    except (ValueError, TypeError):
                        pass
            
            # Find active conversation days (days with more messages than average)
            avg_messages_per_day = sum(messages_by_day.values()) / len(messages_by_day) if messages_by_day else 0
            active_days = [{"date": day, "message_count": count} 
                          for day, count in messages_by_day.items() 
                          if count > avg_messages_per_day * 1.5]
            
            # Sort active days by message count
            active_days.sort(key=lambda x: x["message_count"], reverse=True)
            
            # Extract topics by finding common nouns and proper nouns
            # This is a simple approach - for more sophisticated topic modeling, 
            # we would need NLP libraries like spaCy or NLTK
            word_counts = Counter()
            stopwords = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", 
                       "have", "has", "had", "be", "been", "being", "do", "does", "did", 
                       "will", "would", "shall", "should", "may", "might", "must", "can", 
                       "could", "i", "you", "he", "she", "we", "they", "it", "this", "that", 
                       "these", "those", "me", "him", "her", "us", "them", "of", "in", "on", 
                       "at", "by", "for", "with", "about", "to", "from", "up", "down"}
            
            for msg in messages:
                text = msg.get("text", "").lower()
                # Simple tokenization - split on whitespace and remove punctuation
                words = re.findall(r'\b[a-z]{3,}\b', text)
                # Count words that aren't stopwords
                for word in words:
                    if word not in stopwords:
                        word_counts[word] += 1
            
            # Get top words as potential topics
            potential_topics = [word for word, count in word_counts.most_common(15) if count > 2]
            
            # Generate a narrative summary
            participant_names = [p.get('display_name', p.get('sender', 'Unknown')) for p in top_participants[:3]]
            narrative_summary = f"This conversation between {', '.join(participant_names)} "
            
            if first_message_date and last_message_date:
                narrative_summary += f"spans from {first_message_date.split('T')[0] if 'T' in first_message_date else first_message_date} "
                narrative_summary += f"to {last_message_date.split('T')[0] if 'T' in last_message_date else last_message_date}. "
            
            narrative_summary += f"It contains {total_messages} messages. "
            
            if active_days:
                narrative_summary += f"The most active day was {active_days[0]['date']} with {active_days[0]['message_count']} messages. "
            
            if potential_topics:
                narrative_summary += f"Key topics discussed include {', '.join(potential_topics[:5])}. "
            
            if key_messages:
                narrative_summary += "Important messages in this conversation include: "
            
            # Prepare the final summary
            summary = {
                "chat_name": chat_name,
                "message_count": total_messages,
                "date_range": {
                    "start": first_message_date,
                    "end": last_message_date
                },
                "participants": top_participants,
                "key_messages": key_messages,
                "active_days": active_days[:5],  # Top 5 most active days
                "potential_topics": potential_topics,
                "narrative_summary": narrative_summary
            }
            
            return summary
        except Exception as e:
            logger.error(f"Error summarizing conversation: {e}")
            logger.error(traceback.format_exc())
            return error_response("SUMMARY_ERROR", f"Error summarizing conversation: {str(e)}")

    @mcp.tool()
    def get_contact_activity_report(time_period: str = "last_30_days") -> Dict[str, Any]:
        """Generate a comprehensive report of messaging activity across all contacts.
        
        Args:
            time_period: One of "last_7_days", "last_30_days", "last_90_days", "this_year", "all_time"
            
        Returns:
            Dictionary containing activity data for all contacts
        """
        logger.info(f"get_contact_activity_report called with time_period={time_period}")
        
        # Validate time period
        valid_periods = ["last_7_days", "last_30_days", "last_90_days", "this_year", "all_time"]
        if time_period not in valid_periods:
            return error_response("INVALID_PARAMETER", f"Time period must be one of: {', '.join(valid_periods)}")
        
        try:
            # Create MessagesDB instance
            db = MessagesDB()
            
            # Calculate start date based on time period
            now = datetime.now()
            if time_period == "last_7_days":
                start_date = now - timedelta(days=7)
            elif time_period == "last_30_days":
                start_date = now - timedelta(days=30)
            elif time_period == "last_90_days":
                start_date = now - timedelta(days=90)
            elif time_period == "this_year":
                start_date = datetime(now.year, 1, 1)
            else:  # all_time
                start_date = None
            
            # Convert to Apple epoch time for database queries if a start date is specified
            start_epoch = None
            if start_date:
                start_epoch = db.convert_datetime_to_apple_time(start_date)
            
            # Get all contacts
            contacts = db.get_contacts()
            if not contacts:
                return {
                    "time_period": time_period,
                    "warning": "No contacts found in the database",
                    "contacts": []
                }
            
            # Get activity stats for each contact
            contact_activity = []
            total_messages = 0
            date_range = {}
            
            for contact in contacts:
                phone_number = contact.get("phone_number")
                if not phone_number:
                    continue
                
                # Fetch messages for this contact within time period
                with db.get_db_connection() as conn:
                    cursor = conn.cursor()
                    query = """
                    SELECT 
                        COUNT(*) as message_count,
                        SUM(CASE WHEN is_from_me = 1 THEN 1 ELSE 0 END) as sent_count,
                        SUM(CASE WHEN is_from_me = 0 THEN 1 ELSE 0 END) as received_count,
                        MIN(date) as first_message_date,
                        MAX(date) as last_message_date
                    FROM message
                    JOIN handle ON message.handle_id = handle.ROWID
                    WHERE handle.id = ?
                    """
                    params = [phone_number]
                    
                    # Add date filter if a time period is specified
                    if start_epoch:
                        query += " AND message.date >= ?"
                        params.append(start_epoch)
                    
                    cursor.execute(query, params)
                    result = cursor.fetchone()
                    
                    if not result:
                        continue
                        
                    message_count, sent_count, received_count, first_message_date, last_message_date = result
                    
                    # Skip if no messages in this time period
                    if not message_count:
                        continue
                    
                    # Convert dates from Apple epoch time
                    try:
                        first_date = db.convert_apple_time_to_datetime(first_message_date)
                        last_date = db.convert_apple_time_to_datetime(last_message_date)
                        
                        # Track overall date range
                        if not date_range.get("start") or first_date < date_range.get("start"):
                            date_range["start"] = first_date
                        if not date_range.get("end") or last_date > date_range.get("end"):
                            date_range["end"] = last_date
                    except Exception as e:
                        logger.warning(f"Error converting dates for contact {phone_number}: {e}")
                        first_date = None
                        last_date = None
                    
                    # Calculate daily average
                    if first_date and last_date:
                        days_span = max(1, (last_date - first_date).days + 1)
                        daily_avg = round(message_count / days_span, 2)
                    else:
                        daily_avg = None
                    
                    # Get hour distribution
                    hour_query = """
                    SELECT 
                        strftime('%H', datetime(((message.date / 1000000000) + 978307200), 'unixepoch', 'localtime')) as hour,
                        COUNT(*) as count
                    FROM message
                    JOIN handle ON message.handle_id = handle.ROWID
                    WHERE handle.id = ?
                    """
                    hour_params = [phone_number]
                    
                    if start_epoch:
                        hour_query += " AND message.date >= ?"
                        hour_params.append(start_epoch)
                    
                    hour_query += " GROUP BY hour ORDER BY hour"
                    
                    cursor.execute(hour_query, hour_params)
                    hour_results = cursor.fetchall()
                    
                    hour_distribution = {str(h).zfill(2): 0 for h in range(24)}
                    for hour, count in hour_results:
                        hour_distribution[hour] = count
                    
                    # Get weekday distribution
                    weekday_query = """
                    SELECT 
                        strftime('%w', datetime(((message.date / 1000000000) + 978307200), 'unixepoch', 'localtime')) as weekday,
                        COUNT(*) as count
                    FROM message
                    JOIN handle ON message.handle_id = handle.ROWID
                    WHERE handle.id = ?
                    """
                    weekday_params = [phone_number]
                    
                    if start_epoch:
                        weekday_query += " AND message.date >= ?"
                        weekday_params.append(start_epoch)
                    
                    weekday_query += " GROUP BY weekday ORDER BY weekday"
                    
                    cursor.execute(weekday_query, weekday_params)
                    weekday_results = cursor.fetchall()
                    
                    days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
                    weekday_distribution = {day: 0 for day in days}
                    for weekday, count in weekday_results:
                        weekday_distribution[days[int(weekday)]] = count
                    
                    # Build contact activity record
                    contact_record = {
                        "phone_number": phone_number,
                        "contact_name": contact.get("name", "Unknown"),
                        "display_name": f"{contact.get('name', 'Unknown')} ({phone_number})" if contact.get("name") != "Unknown" else phone_number,
                        "message_stats": {
                            "total": message_count,
                            "sent": sent_count,
                            "received": received_count,
                            "ratio": round(sent_count / max(1, received_count), 2),
                            "daily_average": daily_avg,
                            "first_message": first_date.isoformat() if first_date else None,
                            "last_message": last_date.isoformat() if last_date else None,
                            "days_since_last": (now - last_date).days if last_date else None
                        },
                        "time_distribution": {
                            "hour": hour_distribution,
                            "weekday": weekday_distribution
                        }
                    }
                    
                    contact_activity.append(contact_record)
                    total_messages += message_count
            
            # Sort contacts by message count (most active first)
            contact_activity.sort(key=lambda x: x["message_stats"]["total"], reverse=True)
            
            # Calculate overall statistics
            if date_range.get("start") and date_range.get("end"):
                total_days = max(1, (date_range["end"] - date_range["start"]).days + 1)
                overall_daily_avg = round(total_messages / total_days, 2)
                date_range = {
                    "start": date_range["start"].isoformat(),
                    "end": date_range["end"].isoformat(),
                    "days": total_days
                }
            else:
                overall_daily_avg = None
                date_range = None
            
            # Generate report
            return {
                "time_period": time_period,
                "contacts_analyzed": len(contact_activity),
                "total_messages": total_messages,
                "date_range": date_range,
                "daily_average": overall_daily_avg,
                "contacts": contact_activity
            }
            
        except Exception as e:
            logger.error(f"Error generating contact activity report: {e}")
            logger.error(traceback.format_exc())
            return error_response("ANALYSIS_ERROR", f"Error generating contact activity report: {str(e)}")

    @mcp.tool()
    def analyze_emoji_usage(
        phone_number: str = None,
        chat_id: Union[str, int] = None
    ) -> Dict[str, Any]:
        """Analyze emoji and reaction usage patterns."""
        # Validate parameters
        if not phone_number and not chat_id:
            return error_response("MISSING_PARAMETER", "Either phone_number or chat_id must be provided")
        
        try:
            db = MessagesDB()
            messages = []
            
            # Get messages from appropriate source
            if chat_id:
                logger.info(f"Analyzing emoji usage for group chat ID: {chat_id}")
                messages = db.get_group_chat_messages(chat_id)
                
                # Get chat info
                try:
                    with db.get_db_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT display_name FROM chat WHERE ROWID = ? OR guid = ?", [chat_id, chat_id])
                        result = cursor.fetchone()
                        chat_name = result[0] if result else "Unknown Group"
                except Exception as e:
                    logger.warning(f"Error getting group chat name: {e}")
                    chat_name = "Group Chat"
            else:
                logger.info(f"Analyzing emoji usage with contact: {phone_number}")
                messages = list(db.iter_messages(phone_number))
                contact_name = db.get_contact_name(phone_number)
                display_name = format_contact_name(phone_number)
                chat_name = f"Chat with {display_name}"
                
            logger.info(f"Analyzing emoji usage in {len(messages)} messages")
            
            # Regular expression to match emoji characters
            # This is a simplified pattern that catches most common emojis
            emoji_pattern = re.compile(
                "["
                "\U0001F600-\U0001F64F"  # emoticons
                "\U0001F300-\U0001F5FF"  # symbols & pictographs
                "\U0001F680-\U0001F6FF"  # transport & map symbols
                "\U0001F700-\U0001F77F"  # alchemical symbols
                "\U0001F780-\U0001F7FF"  # Geometric Shapes
                "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                "\U0001FA00-\U0001FA6F"  # Chess Symbols
                "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                "\U00002702-\U000027B0"  # Dingbats
                "\U000024C2-\U0001F251" 
                "]+"
            )
            
            # Initialize counters and trackers
            emoji_counter = Counter()
            emoji_by_sender = defaultdict(Counter)
            messages_with_emoji = 0
            messages_with_emoji_by_sender = defaultdict(int)
            total_emoji_count = 0
            total_messages_by_sender = defaultdict(int)
            sender_display_names = {}  # Mapping of sender IDs to display names
            
            # Process each message
            for msg in messages:
                text = msg.get("text", "")
                if not text:
                    continue
                    
                # Extract sender info
                is_from_me = msg.get("is_from_me", False)
                sender = "You" if is_from_me else msg.get("sender_id", "Unknown")
                
                # Store display name for this sender if not already stored
                if sender not in sender_display_names:
                    if sender == "You":
                        sender_display_names[sender] = "You"
                    else:
                        sender_display_names[sender] = format_contact_name(sender)
                
                # Track total messages by sender
                total_messages_by_sender[sender] += 1
                
                # Find all emojis in the message
                emojis = emoji_pattern.findall(text)
                
                if emojis:
                    # Track messages with emojis
                    messages_with_emoji += 1
                    messages_with_emoji_by_sender[sender] += 1
                    
                    # Count individual emojis
                    for emoji in emojis:
                        emoji_counter[emoji] += 1
                        emoji_by_sender[sender][emoji] += 1
                        total_emoji_count += 1
            
            # Calculate emoji usage by sender
            sender_emoji_stats = []
            for sender, count in total_messages_by_sender.items():
                emoji_count = sum(emoji_by_sender[sender].values())
                with_emoji_count = messages_with_emoji_by_sender.get(sender, 0)
                emoji_percentage = round((with_emoji_count / count) * 100, 1) if count > 0 else 0
                emoji_per_message = round(emoji_count / count, 2) if count > 0 else 0
                
                # Get top emojis for this sender
                top_emojis = emoji_by_sender[sender].most_common(5)
                
                sender_emoji_stats.append({
                    "sender": sender,
                    "display_name": sender_display_names.get(sender, sender),
                    "messages_total": count,
                    "messages_with_emoji": with_emoji_count,
                    "emoji_usage_percentage": emoji_percentage,
                    "emoji_per_message": emoji_per_message,
                    "total_emoji_count": emoji_count,
                    "top_emojis": [{"emoji": emoji, "count": count} for emoji, count in top_emojis]
                })
            
            # Sort by emoji count
            sender_emoji_stats.sort(key=lambda x: x["total_emoji_count"], reverse=True)
            
            # Get top emojis overall
            top_emojis = [{"emoji": emoji, "count": count, "percentage": round(count/total_emoji_count*100, 1) if total_emoji_count > 0 else 0} 
                         for emoji, count in emoji_counter.most_common(10)]
            
            # Calculate emoji diversity (number of unique emojis used)
            emoji_diversity = len(emoji_counter)
            
            # Build the response
            result = {
                "chat_name": chat_name,
                "message_count": len(messages),
                "messages_with_emoji": messages_with_emoji,
                "emoji_usage_percentage": round(messages_with_emoji / len(messages) * 100, 1) if len(messages) > 0 else 0,
                "total_emoji_count": total_emoji_count,
                "unique_emoji_count": emoji_diversity,
                "top_emojis": top_emojis,
                "sender_analysis": sender_emoji_stats
            }
            
            return result
        except Exception as e:
            logger.error(f"Error analyzing emoji usage: {e}")
            logger.error(traceback.format_exc())
            return error_response("ANALYSIS_ERROR", f"Error analyzing emoji usage: {str(e)}")

    def validate_date_range(start_date, end_date):
        """Validate that the provided date range is valid.
        
        Args:
            start_date: Start date in ISO format (YYYY-MM-DD)
            end_date: End date in ISO format (YYYY-MM-DD)
            
        Returns:
            Error response dictionary if invalid, None if valid
        """
        if not start_date or not end_date:
            return None
            
        try:
            start_dt = ensure_datetime(start_date)
            end_dt = ensure_datetime(end_date)
            if start_dt > end_dt:
                return error_response("INVALID_DATE_RANGE", "Start date must be before end date")
            return None
        except ValueError as e:
            return error_response("INVALID_DATE_FORMAT", f"Invalid date format: {str(e)}")

    def ensure_datetime(date_value):
        """Convert various date formats to datetime objects."""
        if not date_value:
            return None
        
        if isinstance(date_value, datetime):
            return date_value
        
        try:
            return dateutil.parser.parse(date_value)
        except (ValueError, TypeError):
            return None
            
    def format_contact_name(phone_number, contact_name=None):
        """Format a contact name with phone number for display in insights.
        
        Args:
            phone_number: The phone number
            contact_name: Optional contact name, will be looked up if not provided
            
        Returns:
            A formatted string showing both name and number if available
        """
        if not phone_number:
            return "Unknown"
            
        # Try to format the phone number for better display
        formatted_number = phone_number
        try:
            # If it looks like a phone number, try to normalize it
            if re.match(r'^\+?[\d\s\-()]+$', phone_number):
                parsed_number = phonenumbers.parse(phone_number, "US")
                formatted_number = phonenumbers.format_number(
                    parsed_number, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
        except Exception:
            # If formatting fails, use the original
            formatted_number = phone_number
            
        if not contact_name or contact_name == "Unknown":
            # Only try to get contact name if the framework is available
            if HAS_CONTACTS_FRAMEWORK:
                try:
                    db = MessagesDB()
                    contact_name = db.get_contact_name(phone_number)
                except Exception as e:
                    logger.debug(f"Error getting contact name: {e}")
                    pass
                
        if contact_name and contact_name != "Unknown":
            # Avoid repetition if the name contains the number
            if phone_number in contact_name or formatted_number in contact_name:
                return contact_name
            else:
                # Always include the formatted number with the contact name
                return f"{formatted_number} ({contact_name})"
        else:
            return formatted_number

    @mcp.tool()
    def export_messages(
        format: str,
        phone_number: str = None,
        chat_id: Union[str, int] = None,
        start_date: str = None,
        end_date: str = None
    ) -> Dict[str, Any]:
        """Export messages to different formats (JSON, CSV, TXT).
        
        Args:
            format: Export format ('json', 'csv', or 'txt')
            phone_number: The phone number or email of the contact
            chat_id: The ID of the group chat
            start_date: Optional start date in ISO format (YYYY-MM-DD)
            end_date: Optional end date in ISO format (YYYY-MM-DD)
            
        Returns:
            Dictionary containing the exported data
        """
        logger.info(f"export_messages called with format={format}, phone_number={phone_number}, chat_id={chat_id}, start_date={start_date}, end_date={end_date}")
        
        # Validate format
        format = format.lower()
        if format not in ['json', 'csv', 'txt']:
            return error_response("INVALID_FORMAT", "Format must be one of: json, csv, txt")
        
        # Validate parameters
        if not phone_number and not chat_id:
            return error_response("MISSING_PARAMETER", "Either phone_number or chat_id must be provided")
        
        # Validate date range
        error = validate_date_range(start_date, end_date)
        if error:
            return error
        
        try:
            db = MessagesDB()
            messages = []
            
            # Convert date strings to datetime objects
            start_dt = ensure_datetime(start_date) if start_date else None
            end_dt = ensure_datetime(end_date) if end_date else None
            
            # Get messages from appropriate source
            if chat_id:
                logger.info(f"Exporting messages for group chat ID: {chat_id}")
                messages = db.get_group_chat_messages(chat_id, start_dt, end_dt)
                
                # Get chat info
                try:
                    with db.get_db_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT display_name FROM chat WHERE ROWID = ? OR guid = ?", [chat_id, chat_id])
                        result = cursor.fetchone()
                        chat_name = result[0] if result else "Unknown Group"
                except Exception as e:
                    logger.warning(f"Error getting group chat name: {e}")
                    chat_name = "Group Chat"
            else:
                logger.info(f"Exporting messages with contact: {phone_number}")
                messages = list(db.iter_messages(phone_number, start_dt, end_dt))
                chat_name = f"Chat with {phone_number}"
            
            logger.info(f"Exporting {len(messages)} messages in {format} format")
            
            # Prepare data for export
            if format == 'json':
                # For JSON, we can include all message details
                export_data = []
                for msg in messages:
                    # Convert message to a serializable format
                    export_msg = {
                        "text": msg.get("text", ""),
                        "date": msg.get("date", ""),
                        "is_from_me": msg.get("is_from_me", False),
                        "sender": "You" if msg.get("is_from_me", False) else msg.get("sender_id", "Unknown"),
                    }
                    
                    # Include attachments if available
                    if msg.get("has_attachments", False) and "attachments" in msg:
                        export_msg["attachments"] = msg["attachments"]
                    
                    export_data.append(export_msg)
                
                # Return JSON data
                return {
                    "format": "json",
                    "chat_name": chat_name,
                    "message_count": len(export_data),
                    "data": export_data
                }
                
            elif format == 'csv':
                # For CSV, create a string with comma-separated values
                csv_data = "Date,Sender,Message\n"
                for msg in messages:
                    date = msg.get("date", "")
                    sender = "You" if msg.get("is_from_me", False) else msg.get("sender_id", "Unknown")
                    text = msg.get("text", "").replace('"', '""')  # Escape quotes for CSV
                    
                    # Add row to CSV
                    csv_data += f'"{date}","{sender}","{text}"\n'
                
                return {
                    "format": "csv",
                    "chat_name": chat_name,
                    "message_count": len(messages),
                    "data": csv_data
                }
                
            elif format == 'txt':
                # For TXT, create a human-readable text format
                txt_data = f"Chat: {chat_name}\n"
                txt_data += f"Exported: {datetime.now().isoformat()}\n"
                txt_data += f"Total Messages: {len(messages)}\n\n"
                
                for msg in messages:
                    date = msg.get("date", "")
                    sender = "You" if msg.get("is_from_me", False) else msg.get("sender_id", "Unknown")
                    text = msg.get("text", "")
                    
                    # Format the message
                    txt_data += f"[{date}] {sender}: {text}\n"
                
                return {
                    "format": "txt",
                    "chat_name": chat_name,
                    "message_count": len(messages),
                    "data": txt_data
                }
            
        except Exception as e:
            logger.error(f"Error exporting messages: {e}")
            logger.error(traceback.format_exc())
            return error_response("EXPORT_ERROR", f"Error exporting messages: {str(e)}")

    @mcp.tool()
    def generate_conversation_timeline(
        phone_number: str = None,
        chat_id: Union[str, int] = None,
        start_date: str = None,
        end_date: str = None,
        resolution: str = "daily"
    ) -> Dict[str, Any]:
        """Generate a timeline visualization of conversation activity.
        
        Args:
            phone_number: The phone number or email of the contact
            chat_id: The ID of the group chat
            start_date: Optional start date in ISO format (YYYY-MM-DD)
            end_date: Optional end date in ISO format (YYYY-MM-DD)
            resolution: Timeline resolution, one of "hourly", "daily", "weekly", "monthly"
            
        Returns:
            Dictionary containing timeline data for visualization
        """
        logger.info(f"generate_conversation_timeline called with phone_number={phone_number}, chat_id={chat_id}, start_date={start_date}, end_date={end_date}, resolution={resolution}")
        
        # Validate parameters
        if not phone_number and not chat_id:
            return error_response("MISSING_PARAMETER", "Either phone_number or chat_id must be provided")
        
        # Validate resolution
        valid_resolutions = ["hourly", "daily", "weekly", "monthly"]
        if resolution not in valid_resolutions:
            return error_response("INVALID_PARAMETER", f"Resolution must be one of: {', '.join(valid_resolutions)}")
        
        # Validate date range
        error = validate_date_range(start_date, end_date)
        if error:
            return error
        
        try:
            db = MessagesDB()
            messages = []
            
            # Convert date strings to datetime objects
            start_dt = ensure_datetime(start_date) if start_date else None
            end_dt = ensure_datetime(end_date) if end_date else None
            
            # Get messages from appropriate source
            if chat_id:
                logger.info(f"Generating timeline for group chat ID: {chat_id}")
                messages = db.get_group_chat_messages(chat_id, start_dt, end_dt)
                
                # Get chat info
                try:
                    with db.get_db_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT display_name FROM chat WHERE ROWID = ? OR guid = ?", [chat_id, chat_id])
                        result = cursor.fetchone()
                        chat_name = result[0] if result else "Unknown Group"
                except Exception as e:
                    logger.warning(f"Error getting group chat name: {e}")
                    chat_name = "Group Chat"
                    
                # Get participants for better detail
                try:
                    participants = db.get_chat_participants(chat_id)
                    participant_info = {p.get("id"): {"name": p.get("id")} for p in participants}
                except Exception as e:
                    logger.warning(f"Error getting group chat participants: {e}")
                    participant_info = {}
            else:
                logger.info(f"Generating timeline for contact: {phone_number}")
                messages = list(db.iter_messages(phone_number, start_dt, end_dt))
                chat_name = f"Chat with {phone_number}"
                participant_info = {phone_number: {"name": phone_number}}
            
            logger.info(f"Processing {len(messages)} messages for timeline")
            
            if not messages:
                return {
                    "warning": "No messages found for the specified parameters",
                    "timeline_data": [],
                    "chat_name": chat_name
                }
            
            # Sort messages by date
            messages = sorted(messages, key=lambda m: m.get("date", "") if isinstance(m.get("date"), str) else "")
            
            # Get date range of messages
            first_msg_date = messages[0].get("date", "")
            last_msg_date = messages[-1].get("date", "")
            
            try:
                first_date = datetime.fromisoformat(first_msg_date.split("T")[0]) if first_msg_date else None
                last_date = datetime.fromisoformat(last_msg_date.split("T")[0]) if last_msg_date else None
            except Exception as e:
                logger.warning(f"Error parsing message dates: {e}")
                first_date = None
                last_date = None
            
            # Group messages according to resolution
            time_buckets = defaultdict(lambda: {"sent": 0, "received": 0, "total": 0, "senders": defaultdict(int)})
            
            for msg in messages:
                try:
                    date_str = msg.get("date", "")
                    if not date_str:
                        continue
                        
                    date_obj = datetime.fromisoformat(date_str)
                    is_from_me = msg.get("is_from_me", False)
                    sender = "You" if is_from_me else msg.get("sender_id", "Unknown")
                    
                    # Create time bucket key based on resolution
                    if resolution == "hourly":
                        # Format: YYYY-MM-DD HH
                        bucket_key = date_obj.strftime("%Y-%m-%d %H")
                    elif resolution == "daily":
                        # Format: YYYY-MM-DD
                        bucket_key = date_obj.strftime("%Y-%m-%d")
                    elif resolution == "weekly":
                        # Get start of week (Monday)
                        start_of_week = date_obj - timedelta(days=date_obj.weekday())
                        bucket_key = start_of_week.strftime("%Y-%m-%d")
                    elif resolution == "monthly":
                        # Format: YYYY-MM
                        bucket_key = date_obj.strftime("%Y-%m")
                    
                    # Update counts
                    time_buckets[bucket_key]["total"] += 1
                    time_buckets[bucket_key]["senders"][sender] += 1
                    
                    if is_from_me:
                        time_buckets[bucket_key]["sent"] += 1
                    else:
                        time_buckets[bucket_key]["received"] += 1
                        
                except Exception as e:
                    logger.warning(f"Error processing message for timeline: {e}")
            
            # Convert to list format for timeline visualization
            timeline_data = []
            for bucket, counts in sorted(time_buckets.items()):
                # Generate label based on resolution
                if resolution == "hourly":
                    date_time = datetime.strptime(bucket, "%Y-%m-%d %H")
                    label = date_time.strftime("%b %d, %H:00")
                    timestamp = date_time.isoformat()
                elif resolution == "daily":
                    date_time = datetime.strptime(bucket, "%Y-%m-%d")
                    label = date_time.strftime("%b %d, %Y")
                    timestamp = date_time.isoformat()
                elif resolution == "weekly":
                    date_time = datetime.strptime(bucket, "%Y-%m-%d")
                    end_of_week = date_time + timedelta(days=6)
                    label = f"{date_time.strftime('%b %d')} - {end_of_week.strftime('%b %d, %Y')}"
                    timestamp = date_time.isoformat()
                elif resolution == "monthly":
                    date_time = datetime.strptime(bucket, "%Y-%m")
                    label = date_time.strftime("%b %Y")
                    timestamp = date_time.replace(day=1).isoformat()
                
                # Format sender data
                sender_data = [{"name": sender, "count": count} for sender, count in counts["senders"].items()]
                sender_data.sort(key=lambda x: x["count"], reverse=True)
                
                timeline_data.append({
                    "time_bucket": bucket,
                    "timestamp": timestamp,
                    "label": label,
                    "total_messages": counts["total"],
                    "sent_messages": counts["sent"],
                    "received_messages": counts["received"],
                    "participants": sender_data
                })
            
            # Calculate activity peaks (times with highest message volumes)
            if timeline_data:
                timeline_data.sort(key=lambda x: x["total_messages"], reverse=True)
                peak_periods = timeline_data[:min(5, len(timeline_data))]
                timeline_data.sort(key=lambda x: x["timestamp"])  # Sort back by time
            else:
                peak_periods = []
            
            # Find conversation patterns
            engagement_patterns = []
            
            # Look for high-activity periods
            if len(timeline_data) > 1:
                avg_messages = sum(item["total_messages"] for item in timeline_data) / len(timeline_data)
                high_activity = [item for item in timeline_data if item["total_messages"] > avg_messages * 1.5]
                
                if high_activity:
                    engagement_patterns.append({
                        "pattern_type": "high_activity",
                        "description": f"Periods of high message volume ({len(high_activity)} periods)",
                        "periods": [item["label"] for item in high_activity[:3]]  # Top 3 examples
                    })
            
            # Look for conversation gaps
            if resolution == "daily" and len(timeline_data) > 2:
                date_gaps = []
                for i in range(1, len(timeline_data)):
                    current = datetime.fromisoformat(timeline_data[i]["timestamp"].split("T")[0])
                    previous = datetime.fromisoformat(timeline_data[i-1]["timestamp"].split("T")[0])
                    gap_days = (current - previous).days - 1
                    
                    if gap_days > 3:  # Gap of more than 3 days
                        date_gaps.append({
                            "start": previous.strftime("%b %d, %Y"),
                            "end": current.strftime("%b %d, %Y"),
                            "days": gap_days
                        })
                
                if date_gaps:
                    engagement_patterns.append({
                        "pattern_type": "conversation_gaps",
                        "description": f"Periods with no messages ({len(date_gaps)} gaps)",
                        "gaps": date_gaps[:3]  # Top 3 examples
                    })
            
            # Generate statistic summaries
            statistics = {
                "total_buckets": len(timeline_data),
                "total_messages": sum(item["total_messages"] for item in timeline_data),
                "avg_messages_per_bucket": round(sum(item["total_messages"] for item in timeline_data) / len(timeline_data), 1) if timeline_data else 0,
                "max_messages_in_bucket": max(item["total_messages"] for item in timeline_data) if timeline_data else 0,
                "first_date": first_date.isoformat() if first_date else None,
                "last_date": last_date.isoformat() if last_date else None
            }
            
            # Build the response
            return {
                "chat_name": chat_name,
                "resolution": resolution,
                "timeline_data": timeline_data,
                "peak_periods": peak_periods,
                "engagement_patterns": engagement_patterns,
                "statistics": statistics
            }
            
        except Exception as e:
            logger.error(f"Error generating conversation timeline: {e}")
            logger.error(traceback.format_exc())
            return error_response("ANALYSIS_ERROR", f"Error generating conversation timeline: {str(e)}")

    @mcp.tool()
    def compare_chats(
        chat_ids: List[Union[str, int]] = None,
        phone_numbers: List[str] = None,
        start_date: str = None,
        end_date: str = None
    ) -> Dict[str, Any]:
        """
        Compare communication patterns across multiple conversations.
        
        Args:
            chat_ids: List of chat IDs for group conversations
            phone_numbers: List of phone numbers for one-on-one conversations
            start_date: Start date for message analysis (format: YYYY-MM-DD)
            end_date: End date for message analysis (format: YYYY-MM-DD)
        
        Returns:
            Comparative analysis of communication patterns
        """
        if not chat_ids and not phone_numbers:
            return error_response(
                "MISSING_PARAMETER", 
                "Either chat_ids or phone_numbers must be provided"
            )
        
        # Validate date range
        error = validate_date_range(start_date, end_date)
        if error:
            return error
            
        # Convert date strings to datetime objects
        start_dt = ensure_datetime(start_date) if start_date else None
        end_dt = ensure_datetime(end_date) if end_date else None
        
        # Initialize database connection
        db = MessagesDB()
        
        # Initialize result structure
        result = {
            "date_range": {
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat()
            },
            "conversations": [],
            "comparative_metrics": {},
        }
        
        # Process chat_ids if provided
        all_chats = []
        if chat_ids:
            for chat_id in chat_ids:
                actual_chat_id = resolve_chat_id(chat_id)
                if actual_chat_id is None:
                    continue
                    
                # Get chat details
                participants = db.get_chat_participants(actual_chat_id)
                # Get a name for the chat
                chat_name = None
                for p in participants:
                    if p.get("group_name"):
                        chat_name = p.get("group_name")
                        break
                
                if not chat_name:
                    # Create a name from participant names
                    person_names = []
                    for p in participants:
                        person_id = p.get("id")
                        if person_id:
                            contact_name = db.get_contact_name(person_id)
                            if contact_name != "Unknown":
                                person_names.append(contact_name)
                            else:
                                formatted_name = format_contact_name(person_id)
                                person_names.append(formatted_name)
                    
                    if len(person_names) > 2:
                        chat_name = f"{person_names[0]} and {len(person_names)-1} others"
                    elif len(person_names) == 2:
                        chat_name = f"{person_names[0]} and {person_names[1]}"
                    elif len(person_names) == 1:
                        chat_name = person_names[0]
                    else:
                        chat_name = f"Group chat {actual_chat_id}"
                
                messages = db.get_group_chat_messages(actual_chat_id, start_dt, end_dt)
                
                all_chats.append({
                    "id": actual_chat_id,
                    "type": "group",
                    "name": chat_name,
                    "messages": messages,
                    "participant_count": len(participants)
                })
        
        # Process phone_numbers if provided
        if phone_numbers:
            for phone_number in phone_numbers:
                # Get contact name
                contact_name = db.get_contact_name(phone_number)
                display_name = format_contact_name(phone_number, contact_name)
                
                # Get messages
                messages = list(db.iter_messages(phone_number, start_dt, end_dt))
                
                all_chats.append({
                    "id": phone_number,
                    "type": "individual",
                    "name": display_name,
                    "messages": messages,
                    "participant_count": 2  # You and the contact
                })
        
        # Calculate metrics for each conversation
        for chat in all_chats:
            messages = chat["messages"]
            
            # Skip if no messages
            if not messages:
                result["conversations"].append({
                    "id": chat["id"],
                    "name": chat["name"],
                    "type": chat["type"],
                    "message_count": 0,
                    "no_messages": True
                })
                continue
            
            # Calculate basic metrics
            message_count = len(messages)
            date_first = datetime.fromisoformat(messages[0]["date"])
            date_last = datetime.fromisoformat(messages[-1]["date"])
            
            # Calculate time span in days
            time_span = (date_last - date_first).total_seconds() / 86400  # seconds in a day
            if time_span < 1:
                time_span = 1  # Minimum of 1 day to avoid division by zero
                
            # Messages per day
            messages_per_day = round(message_count / time_span, 1)
            
            # Message length stats
            message_lengths = [len(m.get("text", "")) for m in messages if m.get("text")]
            if message_lengths:
                avg_message_length = round(sum(message_lengths) / len(message_lengths), 1)
                max_message_length = max(message_lengths)
            else:
                avg_message_length = 0
                max_message_length = 0
            
            # Response time analysis
            response_times = []
            last_time = None
            last_sender = None
            
            for msg in messages:
                current_time = datetime.fromisoformat(msg["date"])
                is_from_me = msg.get("is_from_me", False)
                sender = "You" if is_from_me else chat["id"]
                
                if last_time and sender != last_sender:
                    time_diff = (current_time - last_time).total_seconds() / 60  # minutes
                    response_times.append(time_diff)
                    
                last_time = current_time
                last_sender = sender
            
            if response_times:
                avg_response_time = round(sum(response_times) / len(response_times), 1)
                median_response_time = round(statistics.median(response_times), 1)
            else:
                avg_response_time = 0
                median_response_time = 0
            
            # Time of day analysis
            hour_counts = Counter([datetime.fromisoformat(m["date"]).hour for m in messages])
            peak_hour = hour_counts.most_common(1)[0][0] if hour_counts else 0
            
            # Day of week analysis
            day_counts = Counter([datetime.fromisoformat(m["date"]).strftime("%A") for m in messages])
            peak_day = day_counts.most_common(1)[0][0] if day_counts else "Unknown"
            
            # Get sentiment if possible
            try:
                from textblob import TextBlob
                sentiments = []
                for msg in messages:
                    if msg.get("text") and len(msg.get("text", "").strip()) > 3:
                        analysis = TextBlob(msg["text"])
                        sentiments.append(analysis.sentiment.polarity)
                
                if sentiments:
                    avg_sentiment = round(sum(sentiments) / len(sentiments), 2)
                else:
                    avg_sentiment = 0
            except ImportError:
                avg_sentiment = None
            
            # Store results for this conversation
            chat_result = {
                "id": chat["id"],
                "name": chat["name"],
                "type": chat["type"],
                "message_count": message_count,
                "date_first_message": date_first.isoformat(),
                "date_last_message": date_last.isoformat(),
                "messages_per_day": messages_per_day,
                "avg_message_length": avg_message_length,
                "max_message_length": max_message_length,
                "avg_response_time_minutes": avg_response_time,
                "median_response_time_minutes": median_response_time,
                "peak_hour": f"{peak_hour:02d}:00-{peak_hour+1:02d}:00",
                "peak_day": peak_day,
            }
            
            if avg_sentiment is not None:
                chat_result["avg_sentiment"] = avg_sentiment
                
            result["conversations"].append(chat_result)
        
        # Generate comparative metrics
        if len(result["conversations"]) >= 2:
            # Sort by message count
            by_message_count = sorted(
                [c for c in result["conversations"] if not c.get("no_messages", False)],
                key=lambda x: x["message_count"],
                reverse=True
            )
            
            # Sort by response time
            by_response_time = sorted(
                [c for c in result["conversations"] if not c.get("no_messages", False) and c["avg_response_time_minutes"] > 0],
                key=lambda x: x["avg_response_time_minutes"]
            )
            
            # Sort by messages per day
            by_activity = sorted(
                [c for c in result["conversations"] if not c.get("no_messages", False)],
                key=lambda x: x["messages_per_day"],
                reverse=True
            )
            
            # Most active conversation
            if by_activity:
                result["comparative_metrics"]["most_active"] = {
                    "conversation": by_activity[0]["name"],
                    "messages_per_day": by_activity[0]["messages_per_day"]
                }
                
            # Least active conversation
            if len(by_activity) > 1:
                result["comparative_metrics"]["least_active"] = {
                    "conversation": by_activity[-1]["name"],
                    "messages_per_day": by_activity[-1]["messages_per_day"]
                }
                
            # Fastest responses
            if by_response_time:
                result["comparative_metrics"]["fastest_responses"] = {
                    "conversation": by_response_time[0]["name"],
                    "avg_response_time_minutes": by_response_time[0]["avg_response_time_minutes"]
                }
                
            # Slowest responses
            if len(by_response_time) > 1:
                result["comparative_metrics"]["slowest_responses"] = {
                    "conversation": by_response_time[-1]["name"],
                    "avg_response_time_minutes": by_response_time[-1]["avg_response_time_minutes"]
                }
                
            # Longest messages
            longest_messages = sorted(
                [c for c in result["conversations"] if not c.get("no_messages", False)],
                key=lambda x: x["avg_message_length"],
                reverse=True
            )
            
            if longest_messages:
                result["comparative_metrics"]["longest_messages"] = {
                    "conversation": longest_messages[0]["name"],
                    "avg_length": longest_messages[0]["avg_message_length"]
                }
                
            # Most positive sentiment
            if any("avg_sentiment" in c for c in result["conversations"]):
                by_sentiment = sorted(
                    [c for c in result["conversations"] if not c.get("no_messages", False) and "avg_sentiment" in c],
                    key=lambda x: x["avg_sentiment"],
                    reverse=True
                )
                
                if by_sentiment:
                    result["comparative_metrics"]["most_positive"] = {
                        "conversation": by_sentiment[0]["name"],
                        "avg_sentiment": by_sentiment[0]["avg_sentiment"]
                    }
                    
                    if len(by_sentiment) > 1:
                        result["comparative_metrics"]["most_negative"] = {
                            "conversation": by_sentiment[-1]["name"],
                            "avg_sentiment": by_sentiment[-1]["avg_sentiment"]
                        }
            
            # Calculate common peak times
            all_peak_hours = Counter([c["peak_hour"] for c in result["conversations"] 
                                    if not c.get("no_messages", False)])
            all_peak_days = Counter([c["peak_day"] for c in result["conversations"] 
                                   if not c.get("no_messages", False)])
            
            result["comparative_metrics"]["common_peak_times"] = {
                "hours": [{"hour": hour, "count": count} for hour, count in all_peak_hours.most_common()],
                "days": [{"day": day, "count": count} for day, count in all_peak_days.most_common()]
            }
        
        return result

    @mcp.tool()
    def get_group_chat_transcript(
        chat_id: Union[str, int],
        start_date: str = None,
        end_date: str = None
    ) -> Dict[str, Any]:
        """Get transcript for a specific group chat within a date range.
        
        Args:
            chat_id: The ID, GUID, or display name of the group chat
            start_date: Optional start date in ISO format (YYYY-MM-DD)
            end_date: Optional end date in ISO format (YYYY-MM-DD)
        
        Returns:
            Dictionary containing the group chat transcript data
        """
        logger.info(f"get_group_chat_transcript called with chat_id={chat_id}, start_date={start_date}, end_date={end_date}")
        
        # Resolve the chat ID from name or ID
        actual_chat_id, error = resolve_chat_id(chat_id)
        if error:
            return error
        
        try:
            db = MessagesDB()
            with db.get_db_connection() as conn:
                cursor = conn.cursor()
                messages = db.get_group_chat_messages(actual_chat_id, start_date, end_date)
            
            # Get participants info
            participants = db.get_chat_participants(actual_chat_id)
            
            # Process messages
            transcript_messages = []
            for msg in messages:
                transcript_messages.append({
                    "text": msg["text"],
                    "date": msg["date"],
                    "is_from_me": msg["is_from_me"],
                    "sender_id": msg["sender_id"]
                })
            
            return {
                "chat_id": actual_chat_id,
                "original_query": str(chat_id),
                "messages": transcript_messages,
                "participants": participants,
                "total_count": len(transcript_messages)
            }
        except Exception as e:
            logger.error(f"Error retrieving group chat transcript: {e}")
            logger.error(traceback.format_exc())
            return error_response("ERROR", f"Error retrieving group chat transcript: {str(e)}")

    def extract_group_chat_name(query: str) -> Optional[str]:
        """Extract a group chat name from a natural language query.
        
        Args:
            query: Natural language query
            
        Returns:
            Extracted group chat name or None if not found
        """
        # Patterns to match group chat names in quotes
        patterns = [
            r'group\s+(?:chat|conversation)\s+(?:called|named)\s+"([^"]+)"',
            r'group\s+(?:chat|conversation)\s+"([^"]+)"',
            r'"([^"]+)"\s+group\s+(?:chat|conversation)',
            r'group\s+(?:called|named)\s+"([^"]+)"'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None

    def iter_messages(self, phone_number=None, start_date=None, end_date=None):
        """Iterator that yields messages one at a time to reduce memory usage."""
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
                    date = readable_date.isoformat()
                except:
                    date = None
                
                message_data = {
                    "rowid": rowid,
                    "text": body,
                    "date": date,
                    "is_from_me": bool(is_from_me),
                    "phone_number": handle_id,
                    "group_chat_name": group_chat_name,
                    "has_attachments": False  # We'll add attachment support later if needed
                }
                yield message_data
        finally:
            self.release_connection(conn)
            
    def extract_message_body(self, text, attributed_body):
        """Extract message text from either text or attributedBody."""
        if text is not None:
            return text
        elif attributed_body is None:
            return None
        else:
            # Decode and extract relevant information from attributed_body using string methods
            try:
                attributed_body = attributed_body.decode('utf-8', errors='replace')
                if "NSNumber" in str(attributed_body):
                    attributed_body = str(attributed_body).split("NSNumber")[0]
                    if "NSString" in attributed_body:
                        attributed_body = str(attributed_body).split("NSString")[1]
                        if "NSDictionary" in attributed_body:
                            attributed_body = str(attributed_body).split("NSDictionary")[0]
                            attributed_body = attributed_body[6:-12]
                            return attributed_body
            except:
                pass
            return None
                
    @staticmethod
    def convert_apple_time_to_datetime(apple_time):
        """Convert Apple's date epoch to Python datetime."""
        date_string = '2001-01-01'
        mod_date = datetime.strptime(date_string, '%Y-%m-%d')
        unix_timestamp = int(mod_date.timestamp())*1000000000
        new_date = int((apple_time+unix_timestamp)/1000000000)
        return datetime.fromtimestamp(new_date)
            
    @staticmethod
    def convert_datetime_to_apple_time(dt):
        """Convert Python datetime to Apple's date epoch."""
        base = datetime(2001, 1, 1)
        delta = dt - base
        return int(delta.total_seconds() * 1_000_000_000)

def should_analyze_group_chat(query: str) -> bool:
    """Determine if the query is asking to analyze a group chat.
    
    Args:
        query: Natural language query
        
    Returns:
        Boolean indicating if a group chat should be analyzed
    """
    group_chat_patterns = [
        r'group\s+chat',
        r'group\s+conversation',
        r'group\s+message',
        r'gc\b',
        r'team\s+chat',
        r'family\s+chat',
        r'group\s+thread'
    ]
    
    for pattern in group_chat_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return True
        
        
    return False

@mcp.tool()
def analyze_contact_network_advanced(
    start_date: str = None,
    end_date: str = None,
    min_shared_chats: int = 1,
    include_visualization_data: bool = True
) -> Dict[str, Any]:
    """Perform advanced social network analysis on iMessage contacts using NetworkX.
    
    This function builds a social graph from your iMessage contacts and applies sophisticated
    network analysis algorithms to identify key influencers, communities, and communication patterns.
    
    Args:
        start_date: Optional start date in ISO format (YYYY-MM-DD) to limit analysis timeframe
        end_date: Optional end date in ISO format (YYYY-MM-DD) to limit analysis timeframe
        min_shared_chats: Minimum number of shared chats required to consider contacts connected (default: 1)
        include_visualization_data: Whether to include data formatted for network visualizations (default: True)
        
    Returns:
        Dictionary containing detailed social network metrics and analysis
    """
    logger.info(f"analyze_contact_network_advanced called with start_date={start_date}, end_date={end_date}, min_shared_chats={min_shared_chats}")
    
    # Get basic contact network data using our existing function
    basic_network = analyze_contact_network(start_date, end_date, min_shared_chats)
    
    # Check if there was an error or no data
    if "error" in basic_network:
        return basic_network
    
    if not basic_network.get("connections"):
        return {
            "warning": "Not enough connections to perform network analysis",
            "metrics": {},
            "centrality": [],
            "communities": []
        }
    
    try:
        # Create a NetworkX graph from our connections data
        G = nx.Graph()
        
        # Add nodes (contacts) with attributes
        for node in basic_network["nodes"]:
            G.add_node(
                node["id"],
                display_name=node["display_name"],
                group_count=node["group_count"]
            )
        
        # Add edges (connections) with attributes
        for conn in basic_network["connections"]:
            G.add_edge(
                conn["source"],
                conn["target"],
                weight=conn["shared_chats"],
                shared_chats=conn["shared_chats"]
            )
        
        # Calculate basic network metrics
        metrics = {
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "density": nx.density(G),
            "is_connected": nx.is_connected(G),
            "average_shortest_path_length": nx.average_shortest_path_length(G) if nx.is_connected(G) else None,
            "diameter": nx.diameter(G) if nx.is_connected(G) else None,
            "average_clustering": nx.average_clustering(G),
            "transitivity": nx.transitivity(G)
        }
        
        # Calculate centrality measures
        # Degree centrality - Who has the most connections?
        degree_centrality = nx.degree_centrality(G)
        
        # Betweenness centrality - Who bridges different groups?
        betweenness_centrality = nx.betweenness_centrality(G)
        
        # Closeness centrality - Who can reach others most efficiently?
        closeness_centrality = nx.closeness_centrality(G)
        
        # Eigenvector centrality - Who is connected to other important nodes?
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
        
        # PageRank - Alternative influence measure
        pagerank = nx.pagerank(G)
        
        # Combine centrality measures with contact information
        centrality_data = []
        db = MessagesDB()
        
        for node_id in G.nodes():
            # Get contact display info
            contact_name = db.get_contact_name(node_id) or node_id
            display_name = format_contact_name(node_id, contact_name)
            
            # Create centrality record
            centrality_data.append({
                "id": node_id,
                "display_name": display_name,
                "metrics": {
                    "degree": round(degree_centrality.get(node_id, 0), 4),
                    "betweenness": round(betweenness_centrality.get(node_id, 0), 4),
                    "closeness": round(closeness_centrality.get(node_id, 0), 4),
                    "eigenvector": round(eigenvector_centrality.get(node_id, 0), 4),
                    "pagerank": round(pagerank.get(node_id, 0), 4)
                }
            })
        
        # Sort by degree centrality (most connected first)
        centrality_data.sort(key=lambda x: x["metrics"]["degree"], reverse=True)
        
        # Identify key players
        key_players = {
            "most_connected": centrality_data[0]["display_name"] if centrality_data else None,
            "top_bridger": max(centrality_data, key=lambda x: x["metrics"]["betweenness"])["display_name"] if centrality_data else None,
            "most_central": max(centrality_data, key=lambda x: x["metrics"]["closeness"])["display_name"] if centrality_data else None,
            "most_influential": max(centrality_data, key=lambda x: x["metrics"]["eigenvector"])["display_name"] if centrality_data else None
        }
        
        # Community detection using Louvain method
        communities_dict = best_partition(G)
        
        # Group nodes by community
        community_groups = defaultdict(list)
        for node_id, community_id in communities_dict.items():
            contact_name = db.get_contact_name(node_id) or node_id
            display_name = format_contact_name(node_id, contact_name)
            
            # Get centrality metrics for this contact
            node_metrics = next((item["metrics"] for item in centrality_data if item["id"] == node_id), {})
            
            community_groups[community_id].append({
                "id": node_id,
                "display_name": display_name,
                "centrality": node_metrics
            })
        
        # Format communities data
        communities_data = []
        for community_id, members in community_groups.items():
            # Sort community members by degree centrality
            members.sort(key=lambda x: x.get("centrality", {}).get("degree", 0), reverse=True)
            
            # Identify the leader of this community (most central person)
            community_leader = members[0]["display_name"] if members else "Unknown"
            
            # Calculate community size and density
            community_nodes = [member["id"] for member in members]
            community_subgraph = G.subgraph(community_nodes)
            community_density = nx.density(community_subgraph)
            
            communities_data.append({
                "id": community_id,
                "size": len(members),
                "leader": community_leader,
                "density": round(community_density, 4),
                "cohesion": round(nx.transitivity(community_subgraph) if len(members) > 2 else 0, 4),
                "members": members
            })
        
        # Sort communities by size (largest first)
        communities_data.sort(key=lambda x: x["size"], reverse=True)
        
        # Visualization data (for network graphs)
        visualization_data = None
        if include_visualization_data:
            # Generate positions for nodes using a force-directed layout
            pos = nx.spring_layout(G)
            
            # Prepare nodes data
            nodes_viz = []
            for node_id in G.nodes():
                community_id = communities_dict.get(node_id, 0)
                contact_name = db.get_contact_name(node_id) or node_id
                display_name = format_contact_name(node_id, contact_name)
                
                nodes_viz.append({
                    "id": node_id,
                    "label": display_name,
                    "x": float(pos[node_id][0]),
                    "y": float(pos[node_id][1]),
                    "size": 1 + degree_centrality.get(node_id, 0) * 20,  # Adjust size based on degree
                    "color": f"#{hash(community_id) % 0xFFFFFF:06x}",  # Generate color from community ID
                    "community": community_id
                })
            
            # Prepare edges data
            edges_viz = []
            for source, target, data in G.edges(data=True):
                weight = data.get("weight", 1)
                edges_viz.append({
                    "source": source,
                    "target": target,
                    "weight": weight,
                    "size": 0.5 + (weight / 5),  # Adjust size based on weight
                    "label": f"{weight} shared groups"
                })
            
            visualization_data = {
                "nodes": nodes_viz,
                "edges": edges_viz
            }
        
        # Calculate inter-community connections
        inter_community_connections = []
        for source, target, data in G.edges(data=True):
            source_community = communities_dict.get(source)
            target_community = communities_dict.get(target)
            
            if source_community != target_community:
                source_name = format_contact_name(source, db.get_contact_name(source) or source)
                target_name = format_contact_name(target, db.get_contact_name(target) or target)
                
                inter_community_connections.append({
                    "source": source,
                    "source_name": source_name,
                    "source_community": source_community,
                    "target": target,
                    "target_name": target_name,
                    "target_community": target_community,
                    "weight": data.get("weight", 1)
                })
        
        # Sort by weight
        inter_community_connections.sort(key=lambda x: x["weight"], reverse=True)
        
        # Prepare final analysis result
        result = {
            "metrics": metrics,
            "key_players": key_players,
            "centrality": centrality_data,
            "communities": communities_data,
            "inter_community_connections": inter_community_connections[:20],  # Limit to top 20
            "date_range": basic_network.get("date_range"),
            "raw_network": {
                "nodes": basic_network.get("nodes"),
                "connections": basic_network.get("connections")
            }
        }
        
        # Add visualization data if requested
        if include_visualization_data and visualization_data:
            result["visualization"] = visualization_data
        
        return result
        
    except ImportError as e:
        logger.error(f"Missing required library for network analysis: {e}")
        return error_response(
            "MISSING_DEPENDENCY", 
            "This analysis requires NetworkX and python-louvain libraries. Please install them with: pip install networkx python-louvain"
        )
    except Exception as e:
        logger.error(f"Error performing advanced network analysis: {e}")
        logger.error(traceback.format_exc())
        return error_response("ANALYSIS_ERROR", f"Error performing network analysis: {str(e)}")

@mcp.tool()
def generate_network_visualization(
    network_data: Dict[str, Any],
    visualization_type: str = "force_directed",
    highlight_communities: bool = True,
    include_labels: bool = True
) -> Dict[str, Any]:
    """Generate network visualization data for contact networks.
    
    This function takes the results from analyze_contact_network_advanced and prepares
    visualization data that can be rendered using popular visualization libraries.
    
    Args:
        network_data: Result from analyze_contact_network_advanced function
        visualization_type: Type of layout ('force_directed', 'circular', 'hierarchical')
        highlight_communities: Whether to color nodes by community
        include_labels: Whether to include node labels in visualization
        
    Returns:
        Dictionary containing formatted visualization data
    """
    logger.info(f"generate_network_visualization called with visualization_type={visualization_type}")
    
    # Validate the visualization type
    valid_types = ["force_directed", "circular", "hierarchical", "radial"]
    if visualization_type not in valid_types:
        return error_response(
            "INVALID_PARAMETER", 
            f"Visualization type must be one of: {', '.join(valid_types)}"
        )
    
    # Check if network_data is valid
    if not network_data or not isinstance(network_data, dict):
        return error_response(
            "INVALID_PARAMETER",
            "Network data is required and must be the output from analyze_contact_network_advanced"
        )
    
    # Get nodes and connections from network data
    nodes = network_data.get("raw_network", {}).get("nodes", [])
    connections = network_data.get("raw_network", {}).get("connections", [])
    communities = network_data.get("communities", [])
    
    if not nodes or not connections:
        return error_response(
            "INVALID_DATA",
            "Network data does not contain valid nodes and connections"
        )
    
    try:
        # Create a NetworkX graph from our data
        G = nx.Graph()
        
        # Create mapping from node ID to community
        community_mapping = {}
        if highlight_communities and communities:
            for community in communities:
                for member in community.get("members", []):
                    community_mapping[member.get("id")] = community.get("id", 0)
        
        # Add nodes with attributes
        for node in nodes:
            node_id = node.get("id")
            G.add_node(
                node_id,
                display_name=node.get("display_name", node_id),
                connection_count=node.get("connection_count", 0),
                group_count=node.get("group_count", 0),
                community=community_mapping.get(node_id, 0)
            )
        
        # Add edges with attributes
        for conn in connections:
            G.add_edge(
                conn.get("source"),
                conn.get("target"),
                weight=conn.get("shared_chats", 1)
            )
        
        # Generate positions based on selected layout
        if visualization_type == "force_directed":
            pos = nx.spring_layout(G, k=0.3, iterations=50)
        elif visualization_type == "circular":
            pos = nx.circular_layout(G)
        elif visualization_type == "hierarchical":
            # For hierarchical, use a combination of shell layout with most central nodes in center
            centrality = nx.degree_centrality(G)
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            
            # Create shells based on centrality
            shells = []
            if len(sorted_nodes) > 0:
                # Most central nodes in inner shell
                inner_shell = [node for node, _ in sorted_nodes[:max(1, len(sorted_nodes) // 5)]]
                shells.append(inner_shell)
                
                # Remaining nodes in outer shell
                outer_nodes = [node for node, _ in sorted_nodes[max(1, len(sorted_nodes) // 5):]]
                if outer_nodes:
                    shells.append(outer_nodes)
                    
            pos = nx.shell_layout(G, shells) if shells else nx.shell_layout(G)
        else:  # radial
            # Radial layout with most central node in the center
            centrality = nx.degree_centrality(G)
            central_node = max(centrality.items(), key=lambda x: x[1])[0]
            pos = nx.kamada_kawai_layout(G)
        
        # Generate color palette for communities
        community_colors = {}
        if highlight_communities:
            # Generate distinct colors for each community
            community_ids = set(community_mapping.values())
            for i, comm_id in enumerate(community_ids):
                # Use HSV color space for more distinct colors
                hue = i / max(1, len(community_ids))
                # Convert HSV to RGB (simplified approach)
                r = int(abs(np.sin(hue * 2 * np.pi)) * 200 + 55)
                g = int(abs(np.sin((hue * 2 * np.pi) + 2*np.pi/3)) * 200 + 55)
                b = int(abs(np.sin((hue * 2 * np.pi) + 4*np.pi/3)) * 200 + 55)
                community_colors[comm_id] = f"#{r:02x}{g:02x}{b:02x}"
        
        # Prepare nodes data for visualization
        nodes_viz = []
        for node_id in G.nodes():
            node_data = G.nodes[node_id]
            
            # Set node size based on connection count (with a minimum size)
            size = 5 + (node_data.get("connection_count", 0) * 2)
            
            # Set color based on community if highlighting is enabled
            community_id = node_data.get("community", 0)
            color = community_colors.get(community_id, "#6c757d") if highlight_communities else "#3498db"
            
            nodes_viz.append({
                "id": node_id,
                "label": node_data.get("display_name") if include_labels else "",
                "x": float(pos[node_id][0]),
                "y": float(pos[node_id][1]),
                "size": size,
                "color": color,
                "community": community_id,
                "connection_count": node_data.get("connection_count", 0),
                "group_count": node_data.get("group_count", 0)
            })
        
        # Prepare edges data
        edges_viz = []
        for source, target, data in G.edges(data=True):
            weight = data.get("weight", 1)
            # Scale edge thickness based on weight (shared chats)
            thickness = 0.5 + (weight / 2)
            
            edges_viz.append({
                "source": source,
                "target": target,
                "weight": weight,
                "size": thickness,
                "color": "#aaaaaa",  # Light gray for edges
                "type": "curve"  # Use curved edges for better visualization
            })
        
        # Prepare legend for communities
        legend = []
        if highlight_communities and communities:
            for community in communities:
                comm_id = community.get("id")
                if comm_id in community_colors:
                    legend.append({
                        "label": f"Group {comm_id}: {community.get('leader', 'Unknown')} & {community.get('size', 0) - 1} others",
                        "color": community_colors[comm_id]
                    })
        
        # Prepare the visualization data
        visualization_data = {
            "type": visualization_type,
            "nodes": nodes_viz,
            "edges": edges_viz,
            "legend": legend if highlight_communities else [],
            "stats": {
                "node_count": len(nodes_viz),
                "edge_count": len(edges_viz),
                "community_count": len(set(community_mapping.values())) if highlight_communities else 0
            }
        }
        
        # Add supplementary data for different visualization libraries
        
        # Data for D3.js
        d3_data = {
            "nodes": [{"id": n["id"], "name": n["label"], "group": n["community"], "size": n["size"]} for n in nodes_viz],
            "links": [{"source": e["source"], "target": e["target"], "value": e["weight"]} for e in edges_viz]
        }
        
        # Data for Sigma.js
        sigma_data = {
            "nodes": [{"id": n["id"], "label": n["label"], "x": n["x"], "y": n["y"], "size": n["size"], "color": n["color"]} for n in nodes_viz],
            "edges": [{"id": f"e{i}", "source": e["source"], "target": e["target"], "size": e["size"], "color": e["color"]} for i, e in enumerate(edges_viz)]
        }
        
        # Include library-specific formats
        visualization_data["formats"] = {
            "d3": d3_data,
            "sigma": sigma_data
        }
        
        return visualization_data
        
    except ImportError as e:
        logger.error(f"Missing required library for network visualization: {e}")
        return error_response(
            "MISSING_DEPENDENCY", 
            "This feature requires NetworkX and numpy libraries. Please install them with: pip install networkx numpy"
        )
    except Exception as e:
        logger.error(f"Error generating network visualization: {e}")
        logger.error(traceback.format_exc())
        return error_response("VISUALIZATION_ERROR", f"Error generating network visualization: {str(e)}")

# Start the MCP server - This line is required for the server to respond to requests
if __name__ == "__main__":
    logger.info("Starting MCP server")
    try:
        mcp.run()
    except Exception as e:
        logger.error(f"Unhandled exception in iMessage Query server: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
