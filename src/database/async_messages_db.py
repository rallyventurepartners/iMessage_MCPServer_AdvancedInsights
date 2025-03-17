import os
import logging
import traceback
import asyncio
import aiosqlite
from pathlib import Path
import re
import phonenumbers
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from contextlib import asynccontextmanager

# Import the Redis cache
from src.utils.redis_cache import AsyncRedisCache, cached

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

class AsyncMessagesDB:
    """An asynchronous singleton class for handling message database operations.
    
    This class provides asynchronous access to the iMessage database using aiosqlite.
    It implements a connection pool for better performance and connection management.
    """
    
    _instance = None
    _lock = asyncio.Lock()
    _connection_pool = []
    _max_connections = 10  # Increased from 5 to 10 for better performance
    _init_complete = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AsyncMessagesDB, cls).__new__(cls)
        return cls._instance
        
    def __init__(self, db_path=DB_PATH):
        """Initialize the AsyncMessagesDB with the specified database path.
        
        Note: This is not async because __init__ cannot be async. The actual async
        initialization is done in the initialize() method.
        
        Args:
            db_path: Path to the iMessage database file
        """
        # Only initialize once
        if not hasattr(self, 'initialized'):
            self.initialized = False
            logger.info(f"Creating AsyncMessagesDB instance with path: {db_path}")
            self.db_path = db_path
            self._connection_pool = []
            self._pool_semaphore = asyncio.Semaphore(self._max_connections)
            
            # Initialize cache
            self.cache = AsyncRedisCache()
    
    async def initialize(self):
        """Initialize the database connection pool asynchronously.
        
        This method must be called before any other async methods.
        """
        if self.initialized:
            return
        
        async with self._lock:
            if self.initialized:
                return
                
            # Check if database exists
            if not os.path.exists(self.db_path):
                logger.error(f"Database file not found at {self.db_path}")
                raise FileNotFoundError(f"Database file not found at {self.db_path}")
            
            # Initialize at least one connection to ensure the database is accessible
            try:
                initial_conn = await aiosqlite.connect(self.db_path)
                initial_conn.row_factory = aiosqlite.Row
                self._connection_pool.append(initial_conn)
                logger.info(f"Successfully initialized AsyncMessagesDB with connection pool")
                
                # Initialize cache
                await self.cache.initialize()
                
                # Create database indexes for better performance
                await self._create_indexes()
                
                self.initialized = True
            except Exception as e:
                logger.error(f"Error initializing database connection: {e}")
                logger.error(traceback.format_exc())
                raise
    
    async def _create_indexes(self):
        """Create database indexes to improve query performance."""
        try:
            # Read the index creation SQL script
            script_path = os.path.join(os.path.dirname(__file__), 'db_indexes.sql')
            
            if not os.path.exists(script_path):
                logger.warning(f"Index script not found at {script_path}. Skipping index creation.")
                return
                
            with open(script_path, 'r') as f:
                index_script = f.read()
                
            # Split the script into individual statements
            statements = [stmt.strip() for stmt in index_script.split(';') if stmt.strip()]
            
            # Execute each statement
            async with self.get_db_connection() as conn:
                for stmt in statements:
                    try:
                        await conn.execute(stmt)
                        await conn.commit()
                    except Exception as e:
                        logger.warning(f"Error creating index: {e}")
                        # Continue with other indexes even if one fails
                        
            logger.info("Database indexes created successfully")
        except Exception as e:
            logger.warning(f"Error creating database indexes: {e}")
            logger.warning("The application will continue without optimized indexes")
    
    async def _get_connection(self):
        """Get a connection from the pool or create a new one if pool is not at capacity.
        
        Returns:
            An aiosqlite connection
        """
        async with self._pool_semaphore:
            if not self._connection_pool:
                # Create a new connection if the pool is empty
                conn = await aiosqlite.connect(self.db_path)
                conn.row_factory = aiosqlite.Row
                return conn
            else:
                # Return an existing connection from the pool
                return self._connection_pool.pop()
    
    async def _return_connection(self, conn):
        """Return a connection to the pool or close it if the pool is at capacity.
        
        Args:
            conn: The aiosqlite connection to return
        """
        if len(self._connection_pool) < self._max_connections:
            self._connection_pool.append(conn)
        else:
            await conn.close()
    
    @asynccontextmanager
    async def get_db_connection(self):
        """Get a database connection from the pool and return it when done.
        
        This is an async context manager that should be used with 'async with'.
        
        Example:
            async with db.get_db_connection() as conn:
                # use conn for database operations
        
        Yields:
            An aiosqlite connection
        """
        if not self.initialized:
            await self.initialize()
            
        conn = await self._get_connection()
        try:
            yield conn
        finally:
            await self._return_connection(conn)
    
    @cached(ttl=3600)  # Cache for 1 hour
    async def get_contacts(self) -> List[Dict]:
        """Get all contacts from the database.
        
        Now with caching support for improved performance.
        
        Returns:
            A list of dictionaries containing contact information
        """
        try:
            async with self.get_db_connection() as conn:
                query = """
                SELECT 
                    h.id AS contact_id,
                    IFNULL(h.UNCANONICALIZED_ID, h.id) AS phone_number,
                    IFNULL(h.service, '') AS service,
                    CASE 
                        WHEN c.display_name IS NOT NULL THEN c.display_name
                        WHEN h.UNCANONICALIZED_ID IS NOT NULL THEN h.UNCANONICALIZED_ID
                        ELSE h.id
                    END AS display_name,
                    COUNT(DISTINCT chat_id) AS chat_count,
                    MAX(m.date) AS last_message_date
                FROM chat_handle_join chj
                JOIN handle h ON h.ROWID = chj.handle_id
                JOIN chat c ON c.ROWID = chj.chat_id
                LEFT JOIN message m ON m.handle_id = h.ROWID
                GROUP BY h.id
                ORDER BY last_message_date DESC
                """
                
                cursor = await conn.execute(query)
                rows = await cursor.fetchall()
                
                contacts = []
                for row in rows:
                    contact = dict(row)
                    # Convert date to readable format if it exists
                    if contact.get('last_message_date'):
                        # macOS stores dates as seconds since 2001-01-01
                        mac_epoch = datetime(2001, 1, 1)
                        seconds = contact['last_message_date'] / 1e9  # Convert nanoseconds to seconds
                        date = mac_epoch + timedelta(seconds=seconds)
                        contact['last_message_date'] = date.isoformat()
                    contacts.append(contact)
                
                return contacts
                
        except Exception as e:
            logger.error(f"Error getting contacts: {e}")
            logger.error(traceback.format_exc())
            return []

    async def get_contact_name(self, phone_number):
        """Get a contact's name from their phone number if possible."""
        if not phone_number:
            return "Unknown"
            
        # Check if it's an email or iMessage ID
        if '@' in phone_number or re.match(r'^[A-Za-z]', phone_number):
            # For email addresses, we'll try to find the contact in the database
            async with self.get_db_connection() as connection:
                query = """
                SELECT DISTINCT id, first_name, last_name, display_name
                FROM handle 
                LEFT JOIN chat_handle_join ON handle.ROWID = chat_handle_join.handle_id
                LEFT JOIN chat ON chat_handle_join.chat_id = chat.ROWID
                LEFT JOIN chat_message_join ON chat.ROWID = chat_message_join.chat_id
                LEFT JOIN message ON chat_message_join.message_id = message.ROWID
                WHERE id = ?
                """
                cursor = await connection.execute(query, (phone_number,))
                result = await cursor.fetchone()
                if result:
                    # Use the name from the database if available
                    if result[3]:  # display_name
                        return result[3]
                    elif result[1] and result[2]:  # first_name and last_name
                        return f"{result[1]} {result[2]}"
                    else:
                        return phone_number
                else:
                    return phone_number
        
        # For phone numbers, try to use the Contacts framework if available
        if HAS_CONTACTS_FRAMEWORK:
            try:
                # Try to parse and format the number
                try:
                    parsed_number = phonenumbers.parse(phone_number, "US")
                    formatted_number = phonenumbers.format_number(
                        parsed_number, phonenumbers.PhoneNumberFormat.E164)
                except:
                    formatted_number = phone_number
                
                # Search in Contacts
                people = contacts.CNContactStore().findPeople(phone_number)
                if not people and formatted_number != phone_number:
                    # Try with the formatted number
                    people = contacts.CNContactStore().findPeople(formatted_number)
                
                # Try additional formats if needed
                if not people and '+' not in phone_number:
                    # Try with a '+' prefix
                    people = contacts.CNContactStore().findPeople(f"+{phone_number}")
                
                if people:
                    person = people[0]
                    
                    async def handle_contact(contact, stop):
                        if contact.familyName and contact.givenName:
                            return f"{contact.givenName} {contact.familyName}"
                        elif contact.givenName:
                            return contact.givenName
                        elif contact.familyName:
                            return contact.familyName
                        elif contact.organizationName:
                            return contact.organizationName
                        else:
                            return phone_number
                            
                    return await handle_contact(person, None)
            except Exception as e:
                logger.warning(f"Error searching contacts framework: {e}")
                return phone_number
        else:
            # If we don't have the Contacts framework, try the database
            async with self.get_db_connection() as connection:
                query = """
                SELECT DISTINCT id, first_name, last_name, display_name
                FROM handle 
                LEFT JOIN chat_handle_join ON handle.ROWID = chat_handle_join.handle_id
                LEFT JOIN chat ON chat_handle_join.chat_id = chat.ROWID
                LEFT JOIN chat_message_join ON chat.ROWID = chat_message_join.chat_id
                LEFT JOIN message ON chat_message_join.message_id = message.ROWID
                WHERE id = ?
                """
                cursor = await connection.execute(query, (phone_number,))
                result = await cursor.fetchone()
                if result:
                    # Use the name from the database if available
                    if result[3]:  # display_name
                        return result[3]
                    elif result[1] and result[2]:  # first_name and last_name
                        return f"{result[1]} {result[2]}"
                    else:
                        return phone_number
                else:
                    return phone_number
                    
    @cached(ttl=1800)  # Cache for 30 minutes
    async def get_chat_participants(self, chat_id):
        """Get participants in a chat.
        
        Now with caching support for improved performance.
        
        Args:
            chat_id: The chat ID
            
        Returns:
            list: List of participant dictionaries
        """
        async with self.get_db_connection() as connection:
            query = """
            SELECT DISTINCT handle.id
            FROM handle
            JOIN chat_handle_join ON handle.ROWID = chat_handle_join.handle_id
            WHERE chat_handle_join.chat_id = ?
            """
            cursor = await connection.execute(query, (chat_id,))
            participants = []
            async for row in cursor:
                participant_id = row[0]
                participant_name = await self.get_contact_name(participant_id)
                
                participants.append({
                    "id": participant_id,
                    "name": participant_name
                })
                
            return participants

    async def find_chat_id_by_name(self, chat_identifier):
        """Find a chat ID by name, identifier, or ID."""
        if not chat_identifier:
            return None
            
        # If it's already a number, assume it's a chat ID
        if isinstance(chat_identifier, int) or chat_identifier.isdigit():
            return int(chat_identifier)
            
        async with self.get_db_connection() as connection:
            # Try to find by display name or group name
            query = """
            SELECT DISTINCT chat.ROWID, chat.display_name, chat.group_id
            FROM chat
            WHERE 
                chat.display_name LIKE ? OR
                chat.group_id LIKE ? OR
                chat.chat_identifier LIKE ?
            """
            
            # Add wildcards for partial matching
            search_term = f"%{chat_identifier}%"
            cursor = await connection.execute(query, (search_term, search_term, search_term))
            results = await cursor.fetchall()
            
            if results:
                # If multiple matches, prioritize:
                # 1. Display name exact match
                # 2. Group ID exact match
                # 3. First result
                for row in results:
                    if row[1] and row[1].lower() == chat_identifier.lower():
                        return row[0]  # Exact display_name match
                
                for row in results:
                    if row[2] and row[2].lower() == chat_identifier.lower():
                        return row[0]  # Exact group_id match
                
                # If no exact match found, return the first result
                return results[0][0]
                
            return None
            
    async def get_group_chat_messages(self, chat_id, start_date=None, end_date=None):
        """Get all messages in a group chat within a date range."""
        async with self.get_db_connection() as connection:
            params = [chat_id]
            date_filter = ""
            
            # Add date filters if provided
            if start_date or end_date:
                date_filter = "AND "
                if start_date:
                    start_apple_time = self.convert_datetime_to_apple_time(
                        datetime.fromisoformat(start_date) if isinstance(start_date, str) else start_date
                    )
                    date_filter += "message.date >= ? "
                    params.append(start_apple_time)
                    
                if end_date:
                    if start_date:
                        date_filter += "AND "
                    end_apple_time = self.convert_datetime_to_apple_time(
                        datetime.fromisoformat(end_date) if isinstance(end_date, str) else end_date
                    )
                    date_filter += "message.date <= ? "
                    params.append(end_apple_time)
            
            query = f"""
            SELECT 
                message.ROWID,
                message.date,
                message.text,
                message.is_from_me,
                message.attributedBody,
                message.handle_id,
                handle.id AS sender_id
            FROM 
                chat_message_join
            JOIN message ON chat_message_join.message_id = message.ROWID
            LEFT JOIN handle ON message.handle_id = handle.ROWID
            WHERE 
                chat_message_join.chat_id = ?
                {date_filter}
            ORDER BY 
                message.date ASC
            """
            
            cursor = await connection.execute(query, params)
            messages = []
            async for row in cursor:
                # Extract message data
                message_id = row[0]
                date = self.convert_apple_time_to_datetime(row[1])
                text = row[2]
                is_from_me = bool(row[3])
                attributed_body = row[4]
                handle_id = row[5]
                sender_id = row[6]
                
                # Get sender information
                sender = "You" if is_from_me else await self.get_contact_name(sender_id)
                
                # Extract the actual message body (handling rich text)
                extracted_text = await self.extract_message_body(text, attributed_body)
                
                messages.append({
                    "id": message_id,
                    "date": date.isoformat() if date else None,
                    "text": extracted_text,
                    "is_from_me": is_from_me,
                    "sender": sender,
                    "sender_id": sender_id
                })
                
            return messages
            
    @cached(ttl=3600)  # Cache for 1 hour
    async def get_group_chats(self):
        """Get all group chats from the database.
        
        Now with caching support for improved performance.
        
        Returns:
            list: List of group chat dictionaries
        """
        async with self.get_db_connection() as connection:
            query = """
            SELECT 
                chat.ROWID, 
                chat.display_name,
                chat.group_id,
                MAX(message.date) as last_message_date,
                COUNT(DISTINCT message.ROWID) as message_count,
                COUNT(DISTINCT handle.id) as participant_count
            FROM 
                chat
            LEFT JOIN chat_message_join ON chat.ROWID = chat_message_join.chat_id
            LEFT JOIN message ON chat_message_join.message_id = message.ROWID
            LEFT JOIN chat_handle_join ON chat.ROWID = chat_handle_join.chat_id
            LEFT JOIN handle ON chat_handle_join.handle_id = handle.ROWID
            WHERE 
                chat.style = 43 -- Group chats
            GROUP BY 
                chat.ROWID
            ORDER BY 
                last_message_date DESC
            """
            
            cursor = await connection.execute(query)
            chats = []
            async for row in cursor:
                chat_id = row[0]
                display_name = row[1]
                group_id = row[2]
                last_message_date = row[3]
                message_count = row[4]
                participant_count = row[5]
                
                # Convert Apple time to datetime if available
                last_date = self.convert_apple_time_to_datetime(last_message_date) if last_message_date else None
                
                chats.append({
                    "chat_id": chat_id,
                    "name": display_name or "Unnamed Group",
                    "group_id": group_id,
                    "last_message_date": last_date.isoformat() if last_date else None,
                    "message_count": message_count,
                    "participant_count": participant_count
                })
                
            return chats

    async def iter_messages(self, phone_number=None, chat_id=None, start_date=None, end_date=None):
        """Asynchronously iterate over messages for a contact or chat."""
        if not phone_number and not chat_id:
            return
            
        # Resolve chat_id from phone_number if needed
        if phone_number and not chat_id:
            async with self.get_db_connection() as connection:
                query = """
                SELECT DISTINCT chat_id 
                FROM chat_handle_join
                JOIN handle ON chat_handle_join.handle_id = handle.ROWID
                WHERE handle.id = ?
                """
                cursor = await connection.execute(query, (phone_number,))
                row = await cursor.fetchone()
                if row:
                    chat_id = row[0]
                    
        if not chat_id:
            return
            
        # Build the query
        params = [chat_id]
        date_filter = ""
        
        # Add date filters if provided
        if start_date or end_date:
            date_filter = "AND "
            if start_date:
                start_apple_time = self.convert_datetime_to_apple_time(
                    datetime.fromisoformat(start_date) if isinstance(start_date, str) else start_date
                )
                date_filter += "message.date >= ? "
                params.append(start_apple_time)
                
            if end_date:
                if start_date:
                    date_filter += "AND "
                end_apple_time = self.convert_datetime_to_apple_time(
                    datetime.fromisoformat(end_date) if isinstance(end_date, str) else end_date
                )
                date_filter += "message.date <= ? "
                params.append(end_apple_time)
        
        async with self.get_db_connection() as connection:
            query = f"""
            SELECT 
                message.ROWID,
                message.date,
                message.text,
                message.is_from_me,
                message.attributedBody,
                message.handle_id,
                handle.id AS sender_id
            FROM 
                chat_message_join
            JOIN message ON chat_message_join.message_id = message.ROWID
            LEFT JOIN handle ON message.handle_id = handle.ROWID
            WHERE 
                chat_message_join.chat_id = ?
                {date_filter}
            ORDER BY 
                message.date ASC
            """
            
            async for row in await connection.execute(query, params):
                # Extract message data
                message_id = row[0]
                date = self.convert_apple_time_to_datetime(row[1])
                text = row[2]
                is_from_me = bool(row[3])
                attributed_body = row[4]
                handle_id = row[5]
                sender_id = row[6]
                
                # Get sender information
                sender = "You" if is_from_me else await self.get_contact_name(sender_id)
                
                # Extract the actual message body (handling rich text)
                extracted_text = await self.extract_message_body(text, attributed_body)
                
                yield {
                    "id": message_id,
                    "date": date.isoformat() if date else None,
                    "text": extracted_text,
                    "is_from_me": is_from_me,
                    "sender": sender,
                    "sender_id": sender_id
                }

    async def extract_message_body(self, text, attributed_body):
        """Extract the message body, handling rich text if needed."""
        if text:
            return text
            
        if attributed_body:
            try:
                # For attributed body, it's stored as a binary plist
                # We'll extract just the text content
                if isinstance(attributed_body, bytes):
                    # Look for text content in the binary data
                    text_match = re.search(b'NSString">([^<]+)</string>', attributed_body)
                    if text_match:
                        return text_match.group(1).decode('utf-8', errors='ignore')
            except Exception as e:
                logger.warning(f"Error extracting attributed body: {e}")
                
        return ""

    @staticmethod
    def convert_apple_time_to_datetime(apple_time):
        """Convert Apple's time format to a Python datetime."""
        if not apple_time:
            return None
        # Apple time is nanoseconds since 2001-01-01
        try:
            seconds_since_2001 = apple_time / 1e9
            return datetime(2001, 1, 1) + timedelta(seconds=seconds_since_2001)
        except (ValueError, TypeError, OverflowError):
            return None
            
    @staticmethod
    def convert_datetime_to_apple_time(dt):
        """Convert a Python datetime to Apple's time format."""
        if not dt:
            return None
        # Calculate seconds since 2001-01-01, then convert to nanoseconds
        delta = dt - datetime(2001, 1, 1)
        return int(delta.total_seconds() * 1e9)

    async def get_chat_transcript(self, chat_id=None, phone_number=None, start_date=None, end_date=None):
        """Get a chat transcript with a contact or in a chat."""
        if not chat_id and not phone_number:
            return []
            
        # Resolve chat ID if only phone number is provided
        if phone_number and not chat_id:
            async with self.get_db_connection() as connection:
                query = """
                SELECT chat_id FROM chat_handle_join
                JOIN handle ON chat_handle_join.handle_id = handle.ROWID
                WHERE handle.id = ?
                """
                cursor = await connection.execute(query, (phone_number,))
                result = await cursor.fetchone()
                if result:
                    chat_id = result[0]
                    
        if not chat_id:
            return []
            
        # Get all messages in the chat
        messages = []
        async for message in self.iter_messages(phone_number, chat_id, start_date, end_date):
            messages.append(message)
            
        return messages
        
    async def analyze_contact(self, phone_number, start_date=None, end_date=None, page=1, page_size=100):
        """Analyze messages with a specific contact.
        
        Now supports pagination for better performance with large conversation history.
        
        Args:
            phone_number: The phone number to analyze
            start_date: Start date for filtering messages (ISO format)
            end_date: End date for filtering messages (ISO format) 
            page: The page number (1-indexed)
            page_size: Number of messages per page
            
        Returns:
            dict: Analysis results for the contact
        """
        try:
            # Ensure parameters are valid
            if not phone_number:
                return {"error": "Phone number is required"}
                
            # Normalize page and page_size
            page = max(1, page)
            page_size = max(1, min(1000, page_size))
            
            # Resolve contact
            contact = self.resolve_contact(phone_number)
            
            # Get handle ID
            async with self.get_db_connection() as connection:
                query = "SELECT ROWID FROM handle WHERE id = ?"
                cursor = await connection.execute(query, (phone_number,))
                result = await cursor.fetchone()
                
                if not result:
                    return {"error": f"Contact {phone_number} not found in database"}
                    
                handle_id = result[0]
                
            # Get chat ID(s) for this contact
            async with self.get_db_connection() as connection:
                query = """
                SELECT DISTINCT chat_id 
                FROM chat_handle_join
                WHERE handle_id = ?
                """
                cursor = await connection.execute(query, (handle_id,))
                chat_ids = [row[0] for row in await cursor.fetchall()]
                
                if not chat_ids:
                    return {"error": f"No conversations found with {phone_number}"}
                
            # Get total message count (for pagination)
            async with self.get_db_connection() as connection:
                # Build the query with date filters if needed
                date_conditions = []
                params = []
                
                if start_date:
                    date_conditions.append("message.date >= ?")
                    # Convert ISO date to Apple Epoch time
                    params.append(self.iso_to_apple_time(start_date))
                    
                if end_date:
                    date_conditions.append("message.date <= ?")
                    # Convert ISO date to Apple Epoch time
                    params.append(self.iso_to_apple_time(end_date))
                    
                # Build WHERE clause
                chat_placeholders = ", ".join("?" for _ in chat_ids)
                where_clause = f"chat_message_join.chat_id IN ({chat_placeholders})"
                
                if date_conditions:
                    where_clause += " AND " + " AND ".join(date_conditions)
                    
                # Combine params
                all_params = chat_ids + params
                
                # Count total messages
                count_query = f"""
                SELECT COUNT(*) as count
                FROM message
                JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
                WHERE {where_clause}
                """
                
                cursor = await connection.execute(count_query, all_params)
                result = await cursor.fetchone()
                total_messages = result[0]
                
                # Calculate total pages
                total_pages = (total_messages + page_size - 1) // page_size
                
                # Get paginated messages
                # Add pagination
                offset = (page - 1) * page_size
                limit = page_size
                
                message_query = f"""
                SELECT message.ROWID as id, message.date, message.text, message.is_from_me, 
                       message.associated_message_type, message.service, handle.id as sender
                FROM message
                JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
                LEFT JOIN handle ON message.handle_id = handle.ROWID
                WHERE {where_clause}
                ORDER BY message.date DESC
                LIMIT ? OFFSET ?
                """
                
                # Execute with pagination parameters
                cursor = await connection.execute(message_query, all_params + [limit, offset])
                rows = await cursor.fetchall()
                
                # Process messages
                messages = []
                for row in rows:
                    # Convert Apple epoch to ISO date
                    date = self.apple_time_to_iso(row['date']) if row['date'] else None
                    
                    message = {
                        "id": row['id'],
                        "date": date,
                        "text": row['text'],
                        "is_from_me": bool(row['is_from_me']),
                        "sender": contact['name'] if row['is_from_me'] else "Me",
                        "service": row['service']
                    }
                    messages.append(message)
                    
                # Prepare the result
                result = {
                    "contact": contact,
                    "message_count": total_messages,
                    "messages": messages,
                    "pagination": {
                        "page": page,
                        "page_size": page_size,
                        "total_items": total_messages,
                        "total_pages": total_pages
                    }
                }
                
                return result
        except Exception as e:
            logger.error(f"Error analyzing contact: {e}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}

    async def analyze_group_chat(self, chat_id, start_date=None, end_date=None):
        """Analyze a group chat."""
        from collections import Counter, defaultdict
        import re
        
        if not chat_id:
            return {"error": "Chat ID is required"}
            
        # Resolve chat_id if it's a string
        if isinstance(chat_id, str):
            resolved_chat_id = await self.find_chat_id_by_name(chat_id)
            if not resolved_chat_id:
                return {"error": f"Chat with name '{chat_id}' not found"}
            chat_id = resolved_chat_id
            
        # Get chat information
        async with self.get_db_connection() as connection:
            query = """
            SELECT ROWID, display_name, group_id
            FROM chat
            WHERE ROWID = ?
            """
            cursor = await connection.execute(query, (chat_id,))
            chat_info = await cursor.fetchone()
            
            if not chat_info:
                return {"error": f"Chat with ID {chat_id} not found"}
                
            chat_name = chat_info[1] or "Unnamed Group"
            
            # Get participants
            participants = await self.get_chat_participants(chat_id)
            
            # Get messages
            messages = await self.get_group_chat_messages(chat_id, start_date, end_date)
            
            if not messages:
                return {
                    "chat": {
                        "id": chat_id,
                        "name": chat_name
                    },
                    "participants": participants,
                    "warning": "No messages found in this chat for the specified period"
                }
                
            # Basic statistics
            total_messages = len(messages)
            
            # Messages by participant
            messages_by_participant = defaultdict(int)
            for msg in messages:
                sender = msg["sender"]
                messages_by_participant[sender] += 1
                
            # Sort participants by message count
            participants_stats = [
                {
                    "name": participant,
                    "message_count": messages_by_participant[participant],
                    "percentage": round(messages_by_participant[participant] / total_messages * 100, 1)
                }
                for participant in messages_by_participant
            ]
            participants_stats.sort(key=lambda x: x["message_count"], reverse=True)
            
            # Word frequency analysis
            all_words = []
            for msg in messages:
                if msg["text"]:
                    words = re.findall(r'\b\w+\b', msg["text"].lower())
                    all_words.extend(words)
                    
            word_frequency = Counter(all_words).most_common(20)
            
            # Time analysis
            hour_distribution = defaultdict(int)
            weekday_distribution = defaultdict(int)
            date_distribution = defaultdict(int)
            
            for msg in messages:
                if msg["date"]:
                    date = datetime.fromisoformat(msg["date"])
                    hour_distribution[date.hour] += 1
                    weekday_distribution[date.strftime("%A")] += 1
                    date_distribution[date.strftime("%Y-%m-%d")] += 1
                    
            # Sort weekdays in proper order
            weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            weekday_counts = [weekday_distribution[day] for day in weekday_order]
            
            # Sort dates and get activity over time
            dates = sorted(date_distribution.keys())
            activity_over_time = [
                {"date": date, "count": date_distribution[date]}
                for date in dates
            ]
            
            # Prepare result dictionary
            result = {
                "chat": {
                    "id": chat_id,
                    "name": chat_name
                },
                "message_count": total_messages,
                "participant_stats": participants_stats,
                "time_analysis": {
                    "hour_distribution": {str(h): hour_distribution[h] for h in range(24)},
                    "weekday_distribution": {day: weekday_distribution[day] for day in weekday_order},
                    "activity_over_time": activity_over_time
                },
                "word_frequency": {word: count for word, count in word_frequency}
            }
            
            # Add date range info if provided
            if start_date or end_date:
                result["date_range"] = {
                    "start": start_date,
                    "end": end_date
                }
                
            return result 

    # For methods that modify data, we need to invalidate the cache
    async def invalidate_cache(self, key_prefix=None):
        """Invalidate the cache for a specific key prefix or all keys.
        
        Args:
            key_prefix: The key prefix to invalidate, or None for all keys
        """
        if key_prefix:
            # We would need a more sophisticated invalidation strategy
            # for targeted invalidation (not implemented in this simple example)
            pass
        else:
            # Flush the entire cache
            await self.cache.flush()
            logger.info("Cache invalidated")

    async def get_contact_by_id(self, contact_id):
        """Get contact details by ID.
        
        Args:
            contact_id: The contact ID
            
        Returns:
            dict: Contact details or None if not found
        """
        try:
            async with self.get_db_connection() as connection:
                query = "SELECT id, ROWID FROM handle WHERE id = ?"
                cursor = await connection.execute(query, (contact_id,))
                result = await cursor.fetchone()
                
                if not result:
                    return None
                    
                # Get contact name
                contact_name = self.contact_resolver.resolve_contact(contact_id)['name']
                    
                # Get message count
                query = """
                SELECT COUNT(*) as count
                FROM message
                WHERE handle_id = ?
                """
                cursor = await connection.execute(query, (result['ROWID'],))
                message_count_result = await cursor.fetchone()
                message_count = message_count_result['count'] if message_count_result else 0
                
                return {
                    'id': contact_id,
                    'name': contact_name,
                    'message_count': message_count
                }
        except Exception as e:
            logger.error(f"Error getting contact by ID: {e}")
            return None

    async def get_message_count_since(self, since_datetime):
        """Get the number of new messages since a specific time.
        
        Used for incremental network updates.
        
        Args:
            since_datetime: Datetime object to filter messages from
            
        Returns:
            int: Number of new messages
        """
        try:
            # Convert datetime to Apple epoch time
            since_apple_time = int((since_datetime.timestamp() + 978307200) * 1000000000)
            
            async with self.get_db_connection() as connection:
                query = """
                SELECT COUNT(*) as count
                FROM message
                WHERE date > ?
                """
                cursor = await connection.execute(query, (since_apple_time,))
                result = await cursor.fetchone()
                
                return result['count'] if result else 0
        except Exception as e:
            logger.error(f"Error getting message count since {since_datetime}: {e}")
            return 0
            
    async def get_active_chats_since(self, since_datetime):
        """Get group chats that have had activity since a specific time.
        
        Used for incremental network updates.
        
        Args:
            since_datetime: Datetime object to filter activity from
            
        Returns:
            list: List of active chat dictionaries
        """
        try:
            # Convert datetime to Apple epoch time
            since_apple_time = int((since_datetime.timestamp() + 978307200) * 1000000000)
            
            async with self.get_db_connection() as connection:
                query = """
                SELECT DISTINCT chat.ROWID as chat_id, chat.display_name, 
                       MAX(message.date) as last_message_date
                FROM chat
                JOIN chat_message_join ON chat.ROWID = chat_message_join.chat_id
                JOIN message ON chat_message_join.message_id = message.ROWID
                WHERE message.date > ?
                GROUP BY chat.ROWID
                ORDER BY last_message_date DESC
                """
                cursor = await connection.execute(query, (since_apple_time,))
                rows = await cursor.fetchall()
                
                active_chats = []
                for row in rows:
                    # Convert Apple epoch to ISO date
                    last_message_date = self.apple_time_to_iso(row['last_message_date']) if row['last_message_date'] else None
                    
                    active_chats.append({
                        'chat_id': row['chat_id'],
                        'display_name': row['display_name'],
                        'last_message_date': last_message_date
                    })
                    
                return active_chats
        except Exception as e:
            logger.error(f"Error getting active chats since {since_datetime}: {e}")
            return [] 