"""
Message retrieval and sampling tools.
"""

import logging
from typing import Any, Dict

from mcp import Server

from ..config import Config
from ..db import get_database
from ..models import SampleMessagesInput, SampleMessagesOutput
from ..privacy import redact_pii, apply_preview_caps, hash_contact_id

logger = logging.getLogger(__name__)


def register_message_tools(server: Server, config: Config) -> None:
    """Register message tools with the server."""
    
    @server.tool()
    async def imsg_sample_messages(arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return small, redacted previews for validation.
        
        This tool provides limited message samples with PII redaction
        and preview caps to prevent bulk data extraction.
        """
        try:
            # Validate input
            params = SampleMessagesInput(**arguments)
            
            # Get database connection
            db = await get_database(params.db_path)
            
            # Build query
            query = """
            SELECT 
                m.text,
                m.date/1000000000 + 978307200 as timestamp,
                m.is_from_me,
                h.id as handle_id
            FROM message m
            LEFT JOIN handle h ON m.handle_id = h.ROWID
            WHERE m.text IS NOT NULL
            """
            
            query_params = []
            
            if params.contact_id:
                # Remove hash prefix if present
                contact_id = params.contact_id
                if contact_id.startswith("hash:"):
                    # TODO: Implement reverse lookup
                    logger.warning("Cannot filter by hashed contact ID")
                else:
                    query += " AND h.id = ?"
                    query_params.append(contact_id)
            
            query += " ORDER BY m.date DESC LIMIT ?"
            query_params.append(params.limit)
            
            results = await db.execute_query(query, tuple(query_params))
            
            # Process messages
            messages = []
            for row in results:
                # Determine direction
                direction = "sent" if row['is_from_me'] else "received"
                
                # Hash contact ID
                contact_id = hash_contact_id(row['handle_id']) if row['handle_id'] else "unknown"
                
                # Redact and truncate text
                text = row['text'] or ""
                if config.privacy.redact_by_default:
                    text = redact_pii(text)
                
                # Apply character limit
                max_chars = config.privacy.preview_caps.get('max_chars', 160)
                if len(text) > max_chars:
                    text = text[:max_chars - 3] + "..."
                
                # Format timestamp
                from datetime import datetime
                ts = datetime.fromtimestamp(row['timestamp']).isoformat()
                
                messages.append({
                    "ts": ts,
                    "direction": direction,
                    "contact_id": contact_id,
                    "preview": text
                })
            
            # Apply preview caps
            messages = apply_preview_caps(
                messages,
                max_messages=params.limit,
                max_chars=160
            )
            
            # Build response
            output = SampleMessagesOutput(messages=messages)
            return output.model_dump()
            
        except Exception as e:
            logger.error(f"Sample messages failed: {e}")
            return {
                "error": str(e),
                "error_type": "retrieval_failed"
            }