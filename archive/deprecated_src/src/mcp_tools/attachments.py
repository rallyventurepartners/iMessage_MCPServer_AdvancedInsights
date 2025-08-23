"""
Attachment analysis tools for the iMessage Advanced Insights server.

This module provides tools for analyzing attachments shared in conversations,
including types, frequency, and patterns.
"""

import logging
import os
import mimetypes
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter
from pathlib import Path

from ..database import get_database
from ..exceptions import DatabaseError, ToolExecutionError
from ..utils.responses import error_response, success_response, paginated_response
from ..utils.decorators import requires_consent, parse_date
from .registry import register_tool

logger = logging.getLogger(__name__)


# Common attachment type categories
ATTACHMENT_CATEGORIES = {
    'image': {
        'extensions': {'.jpg', '.jpeg', '.png', '.gif', '.heic', '.heif', '.webp', '.bmp', '.tiff'},
        'mimetypes': {'image/jpeg', 'image/png', 'image/gif', 'image/heic', 'image/webp'}
    },
    'video': {
        'extensions': {'.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm', '.m4v'},
        'mimetypes': {'video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/webm'}
    },
    'audio': {
        'extensions': {'.mp3', '.m4a', '.wav', '.flac', '.aac', '.ogg', '.wma'},
        'mimetypes': {'audio/mpeg', 'audio/mp4', 'audio/wav', 'audio/flac', 'audio/ogg'}
    },
    'document': {
        'extensions': {'.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt', '.pages'},
        'mimetypes': {'application/pdf', 'application/msword', 'text/plain', 'application/rtf'}
    },
    'spreadsheet': {
        'extensions': {'.xls', '.xlsx', '.csv', '.numbers', '.ods'},
        'mimetypes': {'application/vnd.ms-excel', 'text/csv', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'}
    },
    'presentation': {
        'extensions': {'.ppt', '.pptx', '.key', '.odp'},
        'mimetypes': {'application/vnd.ms-powerpoint', 'application/vnd.openxmlformats-officedocument.presentationml.presentation'}
    },
    'archive': {
        'extensions': {'.zip', '.rar', '.7z', '.tar', '.gz', '.bz2'},
        'mimetypes': {'application/zip', 'application/x-rar-compressed', 'application/x-7z-compressed'}
    },
    'code': {
        'extensions': {'.py', '.js', '.java', '.cpp', '.c', '.html', '.css', '.json', '.xml', '.sh'},
        'mimetypes': {'text/x-python', 'application/javascript', 'text/html', 'application/json'}
    }
}


def categorize_attachment(filename: str, mime_type: Optional[str] = None) -> str:
    """Categorize an attachment based on its filename and MIME type."""
    if not filename:
        return 'unknown'
    
    # Get file extension
    ext = Path(filename).suffix.lower()
    
    # Try to determine from extension first
    for category, info in ATTACHMENT_CATEGORIES.items():
        if ext in info['extensions']:
            return category
    
    # Try MIME type if available
    if mime_type:
        mime_lower = mime_type.lower()
        for category, info in ATTACHMENT_CATEGORIES.items():
            if mime_lower in info['mimetypes']:
                return category
    
    # Try to guess MIME type from filename
    guessed_type, _ = mimetypes.guess_type(filename)
    if guessed_type:
        for category, info in ATTACHMENT_CATEGORIES.items():
            if guessed_type in info['mimetypes']:
                return category
    
    # Special cases
    if ext in {'.log', '.txt', '.md'}:
        return 'document'
    
    return 'other'


@register_tool(
    name="get_attachments",
    description="Get attachments shared in conversations"
)
@requires_consent
async def get_attachments_tool(
    contact_id: Optional[str] = None,
    attachment_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get attachments shared in conversations with filtering options.
    
    Args:
        contact_id: Optional contact to filter by
        attachment_type: Filter by type (image, video, audio, document, etc.)
        limit: Maximum number of attachments to return
        offset: Offset for pagination
        start_date: Start date for filtering
        end_date: End date for filtering
        
    Returns:
        List of attachments with metadata and analysis
    """
    try:
        # Validate parameters
        if limit < 1 or limit > 100:
            return error_response("Limit must be between 1 and 100")
        
        if attachment_type and attachment_type not in ATTACHMENT_CATEGORIES and attachment_type != 'other':
            return error_response(f"Invalid attachment type. Valid types: {', '.join(ATTACHMENT_CATEGORIES.keys())}, other")
        
        # Get database connection
        db = await get_database()
        
        # Parse dates
        start_date_obj = parse_date(start_date) if start_date else None
        end_date_obj = parse_date(end_date) if end_date else None
        
        # Query for messages with attachments
        query = """
            SELECT 
                m.ROWID as message_id,
                m.date as timestamp,
                m.is_from_me,
                m.text,
                a.ROWID as attachment_id,
                a.filename,
                a.mime_type,
                a.total_bytes,
                a.uti,
                a.is_sticker,
                h.id as contact_id,
                COALESCE(
                    
                    h.id
                ) as contact_name
            FROM message m
            JOIN message_attachment_join maj ON m.ROWID = maj.message_id
            JOIN attachment a ON maj.attachment_id = a.ROWID
            LEFT JOIN handle h ON m.handle_id = h.ROWID
            WHERE 1=1
        """
        
        params = []
        
        # Add filters
        if contact_id:
            query += " AND h.id = ?"
            params.append(contact_id)
        
        if start_date_obj:
            query += " AND datetime(m.date/1000000000 + strftime('%s', '2001-01-01'), 'unixepoch') >= ?"
            params.append(start_date_obj.isoformat())
        
        if end_date_obj:
            query += " AND datetime(m.date/1000000000 + strftime('%s', '2001-01-01'), 'unixepoch') <= ?"
            params.append(end_date_obj.isoformat())
        
        # Order by date descending
        query += " ORDER BY m.date DESC"
        
        # Execute query
        if hasattr(db, 'execute_query'):
            result = await db.execute_query(query, params)
            rows = result.get('rows', [])
        else:
            # Fallback for different database interface
            rows = []
        
        # Process attachments
        attachments = []
        attachment_stats = defaultdict(int)
        size_by_type = defaultdict(int)
        contacts_sharing = defaultdict(set)
        
        for row in rows:
            # Parse attachment data
            filename = row.get('filename', 'unknown')
            mime_type = row.get('mime_type')
            size_bytes = row.get('total_bytes', 0)
            
            # Categorize attachment
            category = categorize_attachment(filename, mime_type)
            
            # Apply type filter if specified
            if attachment_type and category != attachment_type:
                continue
            
            # Convert timestamp
            timestamp = row.get('timestamp', 0)
            if timestamp:
                # Convert from Apple's epoch (2001-01-01) to Unix epoch
                date = datetime(2001, 1, 1) + timedelta(seconds=timestamp/1_000_000_000)
            else:
                date = None
            
            attachment = {
                'message_id': row.get('message_id'),
                'attachment_id': row.get('attachment_id'),
                'filename': filename,
                'type': category,
                'mime_type': mime_type,
                'size_bytes': size_bytes,
                'size_readable': _format_file_size(size_bytes),
                'date': date.isoformat() if date else None,
                'is_from_me': bool(row.get('is_from_me')),
                'contact_id': row.get('contact_id'),
                'contact_name': row.get('contact_name'),
                'is_sticker': bool(row.get('is_sticker')),
                'preview_text': row.get('text', '')[:100] if row.get('text') else None
            }
            
            attachments.append(attachment)
            
            # Update statistics
            attachment_stats[category] += 1
            size_by_type[category] += size_bytes
            contacts_sharing[row.get('contact_id', 'unknown')].add(category)
        
        # Apply pagination
        total_count = len(attachments)
        paginated_attachments = attachments[offset:offset + limit]
        
        # Calculate insights
        total_size = sum(a['size_bytes'] for a in attachments)
        
        insights = {
            'total_attachments': total_count,
            'total_size_bytes': total_size,
            'total_size_readable': _format_file_size(total_size),
            'attachments_by_type': dict(attachment_stats),
            'size_by_type': {
                cat: {
                    'bytes': size,
                    'readable': _format_file_size(size),
                    'percentage': round(size / total_size * 100, 1) if total_size > 0 else 0
                }
                for cat, size in size_by_type.items()
            },
            'most_common_type': max(attachment_stats.items(), key=lambda x: x[1])[0] if attachment_stats else None,
            'average_size': {
                'bytes': total_size // total_count if total_count > 0 else 0,
                'readable': _format_file_size(total_size // total_count) if total_count > 0 else '0 B'
            },
            'top_sharers': _get_top_sharers(attachments)[:5],
            'sharing_patterns': {
                'images_mostly_from_me': _analyze_sharing_direction(attachments, 'image'),
                'videos_mostly_from_me': _analyze_sharing_direction(attachments, 'video'),
                'documents_mostly_from_me': _analyze_sharing_direction(attachments, 'document')
            }
        }
        
        # Prepare response
        response_data = {
            'attachments': paginated_attachments,
            'pagination': {
                'offset': offset,
                'limit': limit,
                'total': total_count,
                'has_more': offset + limit < total_count
            },
            'filters_applied': {
                'contact_id': contact_id,
                'attachment_type': attachment_type,
                'date_range': {
                    'start': start_date_obj.isoformat() if start_date_obj else None,
                    'end': end_date_obj.isoformat() if end_date_obj else None
                }
            },
            'insights': insights
        }
        
        return success_response(response_data)
        
    except Exception as e:
        logger.error(f"Error getting attachments: {e}", exc_info=True)
        return error_response(f"Failed to get attachments: {str(e)}")


def _format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def _get_top_sharers(attachments: List[Dict]) -> List[Dict]:
    """Get top contacts by attachment sharing."""
    sharer_stats = defaultdict(lambda: {'count': 0, 'size': 0, 'types': set()})
    
    for att in attachments:
        contact = att.get('contact_name', 'Unknown')
        if not att.get('is_from_me'):
            sharer_stats[contact]['count'] += 1
            sharer_stats[contact]['size'] += att.get('size_bytes', 0)
            sharer_stats[contact]['types'].add(att.get('type'))
    
    top_sharers = []
    for contact, stats in sharer_stats.items():
        top_sharers.append({
            'contact': contact,
            'attachment_count': stats['count'],
            'total_size': _format_file_size(stats['size']),
            'types_shared': list(stats['types'])
        })
    
    return sorted(top_sharers, key=lambda x: x['attachment_count'], reverse=True)


def _analyze_sharing_direction(attachments: List[Dict], attachment_type: str) -> bool:
    """Analyze if a specific type of attachment is mostly sent or received."""
    type_attachments = [a for a in attachments if a.get('type') == attachment_type]
    if not type_attachments:
        return None
    
    from_me_count = sum(1 for a in type_attachments if a.get('is_from_me'))
    return from_me_count > len(type_attachments) / 2