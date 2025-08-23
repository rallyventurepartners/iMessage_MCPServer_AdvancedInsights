"""
Overview tools for global database statistics and summaries.
"""

import logging
from collections import defaultdict
from typing import Any, Dict

from mcp import Server

from ..config import Config
from ..db import get_database
from ..models import SummaryOverviewInput, SummaryOverviewOutput
from ..privacy import hash_contact_id, sanitize_contact

logger = logging.getLogger(__name__)


def register_overview_tools(server: Server, config: Config) -> None:
    """Register overview tools with the server."""
    
    @server.tool()
    async def imsg_summary_overview(arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Global overview for Claude's kickoff context.
        
        Provides comprehensive statistics about the message database including
        counts, date ranges, platform distribution, and attachment types.
        """
        try:
            # Validate input
            params = SummaryOverviewInput(**arguments)
            
            # Get database connection
            db = await get_database(params.db_path)
            
            # Get total message count
            count_query = "SELECT COUNT(*) as count FROM message"
            count_result = await db.execute_query(count_query)
            total_messages = count_result[0]['count'] if count_result else 0
            
            # Get unique contacts
            contacts_query = """
            SELECT COUNT(DISTINCT handle_id) as count 
            FROM message 
            WHERE handle_id IS NOT NULL
            """
            contacts_result = await db.execute_query(contacts_query)
            unique_contacts = contacts_result[0]['count'] if contacts_result else 0
            
            # Get date range
            date_query = """
            SELECT 
                MIN(date/1000000000 + 978307200) as min_date,
                MAX(date/1000000000 + 978307200) as max_date
            FROM message
            WHERE date IS NOT NULL
            """
            date_result = await db.execute_query(date_query)
            
            date_range = {
                "start": "unknown",
                "end": "unknown"
            }
            
            if date_result and date_result[0]['min_date']:
                from datetime import datetime
                date_range = {
                    "start": datetime.fromtimestamp(date_result[0]['min_date']).strftime("%Y-%m-%d"),
                    "end": datetime.fromtimestamp(date_result[0]['max_date']).strftime("%Y-%m-%d")
                }
            
            # Get message direction counts
            direction_query = """
            SELECT 
                is_from_me,
                COUNT(*) as count
            FROM message
            GROUP BY is_from_me
            """
            direction_result = await db.execute_query(direction_query)
            
            by_direction = {"sent": 0, "received": 0}
            for row in direction_result:
                if row['is_from_me'] == 1:
                    by_direction['sent'] = row['count']
                else:
                    by_direction['received'] = row['count']
            
            # Get platform distribution (iMessage vs SMS)
            platform_query = """
            SELECT 
                CASE 
                    WHEN service = 'iMessage' THEN 'iMessage'
                    ELSE 'SMS'
                END as platform,
                COUNT(*) as count
            FROM message
            GROUP BY platform
            """
            platform_result = await db.execute_query(platform_query)
            
            by_platform = {"iMessage": 0, "SMS": 0}
            for row in platform_result:
                platform = row['platform']
                if platform in by_platform:
                    by_platform[platform] = row['count']
            
            # Get attachment counts by type
            attachment_query = """
            SELECT 
                mime_type,
                COUNT(*) as count
            FROM attachment
            WHERE mime_type IS NOT NULL
            GROUP BY mime_type
            """
            
            attachments = {"images": 0, "videos": 0, "other": 0}
            
            try:
                attachment_result = await db.execute_query(attachment_query)
                
                for row in attachment_result:
                    mime_type = row['mime_type'].lower()
                    count = row['count']
                    
                    if 'image' in mime_type:
                        attachments['images'] += count
                    elif 'video' in mime_type:
                        attachments['videos'] += count
                    else:
                        attachments['other'] += count
                        
            except Exception as e:
                # Attachment table might not exist in older databases
                logger.warning(f"Could not query attachments: {e}")
            
            # Get group chat count
            group_query = """
            SELECT COUNT(DISTINCT chat_id) as count
            FROM chat
            WHERE style = 43 OR group_name IS NOT NULL
            """
            
            notes = []
            try:
                group_result = await db.execute_query(group_query)
                if group_result and group_result[0]['count'] > 0:
                    notes.append(f"{group_result[0]['count']} group chats detected")
            except Exception:
                pass
            
            # Add attachment percentage note
            total_attachments = sum(attachments.values())
            if total_attachments > 0 and total_messages > 0:
                attachment_pct = int((total_attachments / total_messages) * 100)
                notes.append(f"{attachment_pct}% messages have attachments")
            
            # Add activity trend note
            if by_direction['sent'] > 0 and by_direction['received'] > 0:
                sent_ratio = by_direction['sent'] / (by_direction['sent'] + by_direction['received'])
                if sent_ratio > 0.6:
                    notes.append("You send more messages than you receive")
                elif sent_ratio < 0.4:
                    notes.append("You receive more messages than you send")
                else:
                    notes.append("Balanced sending and receiving pattern")
            
            # Build response
            output = SummaryOverviewOutput(
                total_messages=total_messages,
                unique_contacts=unique_contacts,
                date_range=date_range,
                by_direction=by_direction,
                by_platform=by_platform,
                attachments=attachments,
                notes=notes
            )
            
            return output.model_dump()
            
        except Exception as e:
            logger.error(f"Summary overview failed: {e}")
            return {
                "error": str(e),
                "error_type": "overview_failed"
            }
    
    @server.tool()
    async def imsg_contact_resolve(arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve phone/email/handle to pretty name via macOS Contacts.
        
        This tool attempts to resolve a phone number, email, or other identifier
        to a display name using the local Contacts database.
        """
        try:
            # For now, return a simple response
            # TODO: Implement actual macOS Contacts integration
            query = arguments.get('query', '')
            
            # Basic validation
            if not query:
                return {
                    "error": "Query parameter is required",
                    "error_type": "validation_error"
                }
            
            # Hash the contact ID if privacy is enabled
            contact_id = hash_contact_id(query) if config.privacy.hash_identifiers else query
            
            # Determine contact type
            if '@' in query:
                kind = 'email'
            elif query.startswith('+') or any(c.isdigit() for c in query):
                kind = 'phone'
            else:
                kind = 'apple_id'
            
            # Return resolved contact
            # In production, this would query the Contacts database
            return {
                "contact_id": contact_id,
                "display_name": f"Contact {contact_id[:8]}",  # Placeholder
                "kind": kind
            }
            
        except Exception as e:
            logger.error(f"Contact resolution failed: {e}")
            return {
                "error": str(e),
                "error_type": "resolution_failed"
            }