"""
Implementation of missing abstract methods for AsyncMessagesDB.

This module adds the missing methods required by the abstract base class.
"""

from typing import Dict, Any, Optional
from datetime import datetime

async def get_contact_by_phone_or_email_impl(self, identifier: str) -> Dict[str, Any]:
    """
    Get contact information by phone number or email.
    
    Args:
        identifier: Phone number or email address
        
    Returns:
        Contact information dictionary
    """
    async with self.get_db_connection() as conn:
        # Normalize the identifier
        normalized = identifier.strip().lower()
        
        # Query to find contact by phone or email
        query = """
            SELECT DISTINCT
                h.ROWID as handle_id,
                h.id as identifier,
                h.service,
                COALESCE(
                    json_extract(h.contact_info, '$.displayName'),
                    json_extract(h.contact_info, '$.firstName') || ' ' || 
                    json_extract(h.contact_info, '$.lastName'),
                    h.id
                ) as display_name,
                NULL as first_name,
                NULL as last_name,
                COUNT(DISTINCT m.ROWID) as message_count,
                MAX(m.date) as last_message_date
            FROM handle h
            LEFT JOIN message m ON h.ROWID = m.handle_id
            WHERE LOWER(h.id) = ? OR LOWER(h.id) LIKE ?
            GROUP BY h.ROWID
            ORDER BY message_count DESC
            LIMIT 1
        """
        
        cursor = await conn.execute(query, [normalized, f"%{normalized}%"])
        row = await cursor.fetchone()
        
        if not row:
            return {
                "success": False,
                "error": f"No contact found for identifier: {identifier}"
            }
            
        # Convert to dictionary
        contact = dict(row)
        
        # Convert Apple timestamp if present
        if contact.get('last_message_date'):
            contact['last_message_date'] = convert_apple_timestamp(contact['last_message_date'])
            
        return {
            "success": True,
            "data": {
                "contact": contact
            }
        }


async def get_messages_from_chat_impl(
    self, 
    chat_id: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 100,
    offset: int = 0
) -> Dict[str, Any]:
    """
    Get messages from a specific chat/conversation.
    
    Args:
        chat_id: Chat identifier
        start_date: Optional start date filter
        end_date: Optional end date filter
        limit: Maximum number of messages to return
        offset: Number of messages to skip
        
    Returns:
        Dictionary with messages and metadata
    """
    async with self.get_db_connection() as conn:
        # Build query with optional date filters
        query = """
            SELECT 
                m.ROWID as message_id,
                m.guid,
                m.text,
                m.date,
                m.is_from_me,
                m.is_read,
                m.is_delivered,
                m.is_sent,
                h.id as sender_id,
                h.id as sender_name,
                c.chat_identifier
            FROM message m
            JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
            JOIN chat c ON cmj.chat_id = c.ROWID
            LEFT JOIN handle h ON m.handle_id = h.ROWID
            WHERE c.chat_identifier = ?
        """
        
        params = [chat_id]
        
        # Add date filters if provided
        if start_date:
            query += " AND datetime(m.date/1000000000 + strftime('%s', '2001-01-01'), 'unixepoch') >= ?"
            params.append(start_date.isoformat())
            
        if end_date:
            query += " AND datetime(m.date/1000000000 + strftime('%s', '2001-01-01'), 'unixepoch') <= ?"
            params.append(end_date.isoformat())
            
        # Add ordering and pagination
        query += " ORDER BY m.date DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor = await conn.execute(query, params)
        rows = await cursor.fetchall()
        
        # Process messages
        messages = []
        for row in rows:
            msg = dict(row)
            
            # Convert Apple timestamp
            if msg.get('date'):
                msg['date'] = convert_apple_timestamp(msg['date'])
                
            # Clean message text
            if msg.get('text'):
                msg['text'] = clean_message_text(msg['text'])
                
            messages.append(msg)
            
        # Get total count
        count_query = """
            SELECT COUNT(*)
            FROM message m
            JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
            JOIN chat c ON cmj.chat_id = c.ROWID
            WHERE c.chat_identifier = ?
        """
        count_params = [chat_id]
        
        if start_date:
            count_query += " AND datetime(m.date/1000000000 + strftime('%s', '2001-01-01'), 'unixepoch') >= ?"
            count_params.append(start_date.isoformat())
            
        if end_date:
            count_query += " AND datetime(m.date/1000000000 + strftime('%s', '2001-01-01'), 'unixepoch') <= ?"
            count_params.append(end_date.isoformat())
            
        cursor = await conn.execute(count_query, count_params)
        total_count = (await cursor.fetchone())[0]
        
        return {
            "success": True,
            "data": {
                "messages": messages,
                "pagination": {
                    "total": total_count,
                    "limit": limit,
                    "offset": offset,
                    "has_more": offset + limit < total_count
                }
            }
        }


async def get_messages_from_contact_impl(
    self, 
    phone_number: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    page: int = 1,
    page_size: int = 50
) -> Dict[str, Any]:
    """
    Get messages exchanged with a specific contact.
    
    Args:
        phone_number: Contact's phone number or identifier
        start_date: Optional start date filter
        end_date: Optional end date filter
        page: Page number (1-based)
        page_size: Number of messages per page
        
    Returns:
        Dictionary with messages and metadata
    """
    async with self.get_db_connection() as conn:
        # Calculate offset from page
        offset = (page - 1) * page_size
        
        # Normalize the identifier properly
        if hasattr(self, 'contact_resolver') and self.contact_resolver:
            normalized_id = self.contact_resolver.normalize_identifier(phone_number)
        else:
            normalized_id = phone_number.strip()
        
        # Query for messages - handle both exact match and flexible matching
        query = """
            SELECT 
                m.ROWID as message_id,
                m.guid,
                m.text,
                m.date,
                m.is_from_me,
                m.is_read,
                m.is_delivered,
                m.is_sent,
                h.id as contact_id,
                h.id as contact_name,
                c.chat_identifier
            FROM message m
            JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
            JOIN chat c ON cmj.chat_id = c.ROWID
            LEFT JOIN handle h ON m.handle_id = h.ROWID
            WHERE (h.id = ? OR h.id LIKE ? OR h.id = ?)
        """
        
        # Try exact match, wildcard match, and original format
        params = [normalized_id, f"%{normalized_id}%", phone_number]
        
        # Add date filters
        if start_date:
            query += " AND datetime(m.date/1000000000 + strftime('%s', '2001-01-01'), 'unixepoch') >= ?"
            params.append(start_date.isoformat())
            
        if end_date:
            query += " AND datetime(m.date/1000000000 + strftime('%s', '2001-01-01'), 'unixepoch') <= ?"
            params.append(end_date.isoformat())
            
        # Add ordering and pagination
        query += " ORDER BY m.date DESC LIMIT ? OFFSET ?"
        params.extend([page_size, offset])
        
        cursor = await conn.execute(query, params)
        rows = await cursor.fetchall()
        
        # Process messages
        messages = []
        for row in rows:
            msg = dict(row)
            
            # Convert timestamp
            if msg.get('date'):
                msg['date'] = convert_apple_timestamp(msg['date'])
                
            # Clean text
            if msg.get('text'):
                msg['text'] = clean_message_text(msg['text'])
                
            messages.append(msg)
            
        # Get total count
        count_query = """
            SELECT COUNT(*)
            FROM message m
            JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
            JOIN chat c ON cmj.chat_id = c.ROWID
            LEFT JOIN handle h ON m.handle_id = h.ROWID
            WHERE (h.id = ? OR LOWER(h.id) LIKE ?)
        """
        count_params = [normalized_phone, f"%{normalized_phone}%"]
        
        if start_date:
            count_query += " AND datetime(m.date/1000000000 + strftime('%s', '2001-01-01'), 'unixepoch') >= ?"
            count_params.append(start_date.isoformat())
            
        if end_date:
            count_query += " AND datetime(m.date/1000000000 + strftime('%s', '2001-01-01'), 'unixepoch') <= ?"
            count_params.append(end_date.isoformat())
            
        cursor = await conn.execute(count_query, count_params)
        total_count = (await cursor.fetchone())[0]
        
        # Build pagination info
        total_pages = (total_count + page_size - 1) // page_size
        
        return {
            "success": True,
            "data": {
                "messages": messages,
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total_messages": total_count,
                    "total_pages": total_pages,
                    "has_more": page < total_pages
                }
            }
        }


async def search_messages_impl(
    self, 
    query: str,
    contact_id: Optional[str] = None,
    chat_id: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 50,
    offset: int = 0
) -> Dict[str, Any]:
    """
    Search messages by content.
    
    Args:
        query: Search query string
        contact_id: Optional contact filter
        chat_id: Optional chat filter
        start_date: Optional start date filter
        end_date: Optional end date filter
        limit: Maximum results
        offset: Results to skip
        
    Returns:
        Search results dictionary
    """
    async with self.get_db_connection() as conn:
        # Base search query
        search_query = """
            SELECT 
                m.ROWID as message_id,
                m.guid,
                m.text,
                m.date,
                m.is_from_me,
                h.id as contact_id,
                h.id as contact_name,
                c.chat_identifier
            FROM message m
            JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
            JOIN chat c ON cmj.chat_id = c.ROWID
            LEFT JOIN handle h ON m.handle_id = h.ROWID
            WHERE m.text LIKE ?
        """
        
        params = [f"%{query}%"]
        
        # Add filters
        if contact_id:
            search_query += " AND (h.id = ? OR LOWER(h.id) LIKE ?)"
            params.extend([contact_id, f"%{contact_id.lower()}%"])
            
        if chat_id:
            search_query += " AND c.chat_identifier = ?"
            params.append(chat_id)
            
        if start_date:
            search_query += " AND datetime(m.date/1000000000 + strftime('%s', '2001-01-01'), 'unixepoch') >= ?"
            params.append(start_date.isoformat())
            
        if end_date:
            search_query += " AND datetime(m.date/1000000000 + strftime('%s', '2001-01-01'), 'unixepoch') <= ?"
            params.append(end_date.isoformat())
            
        # Add ordering and pagination
        search_query += " ORDER BY m.date DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor = await conn.execute(search_query, params)
        rows = await cursor.fetchall()
        
        # Process results
        results = []
        for row in rows:
            msg = dict(row)
            
            # Convert timestamp
            if msg.get('date'):
                msg['date'] = convert_apple_timestamp(msg['date'])
                
            # Clean and highlight text
            if msg.get('text'):
                msg['text'] = clean_message_text(msg['text'])
                # Simple highlighting
                msg['highlighted_text'] = msg['text'].replace(
                    query, f"**{query}**"
                )
                
            results.append(msg)
            
        # Get total count
        count_query = search_query.replace(
            "SELECT m.ROWID as message_id", 
            "SELECT COUNT(*)"
        ).split("ORDER BY")[0]
        
        cursor = await conn.execute(count_query, params[:-2])  # Exclude limit/offset
        count_row = await cursor.fetchone()
        total_count = count_row[0] if count_row else 0
        
        return {
            "success": True,
            "data": {
                "query": query,
                "results": results,
                "pagination": {
                    "total": total_count,
                    "limit": limit,
                    "offset": offset,
                    "has_more": offset + limit < total_count
                }
            }
        }


# Import required utilities
from .db_utils import convert_apple_timestamp, clean_message_text