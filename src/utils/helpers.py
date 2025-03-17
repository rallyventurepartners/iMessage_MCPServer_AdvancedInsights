import re
import logging
import dateutil.parser
import phonenumbers
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

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

def error_response(code, message, details=None):
    """Create a standardized error response."""
    error = {
        "code": code,
        "message": message
    }
    if details:
        error["details"] = details
    return {"error": error}

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
    This function is kept for backwards compatibility - new code should
    use the EnhancedContactResolver class instead.
    
    Args:
        phone_number: The phone number
        contact_name: Optional contact name, will be looked up if not provided
        
    Returns:
        A formatted string showing both name and number if available
    """
    if not phone_number:
        return "Unknown"
        
    # Create temporary resolver
    from src.database.messages_db import MessagesDB
    db = MessagesDB()
    
    # Use enhanced resolver if available
    if hasattr(db, 'contact_resolver') and db.contact_resolver:
        contact_info = db.contact_resolver.resolve_contact(phone_number)
        return contact_info["display_name"]
    
    # Fall back to the original method if resolver isn't available
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
        return formatted_number
                
    if contact_name and contact_name != "Unknown":
        # Avoid repetition if the name contains the number
        if phone_number in contact_name or formatted_number in contact_name:
            return contact_name
        else:
            # Always include the formatted number with the contact name
            return f"{contact_name} ({formatted_number})"
    else:
        return formatted_number

def sanitize_name(name):
    """Remove potentially problematic characters from names."""
    if name:
        # Remove control characters and limit length
        return re.sub(r'[\x00-\x1F\x7F]', '', name)[:100]
    return name

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
        from src.database.messages_db import MessagesDB
        db = MessagesDB()
        resolved_id = db.find_chat_id_by_name(chat_id)
        
        if resolved_id is None:
            return None, error_response("CHAT_NOT_FOUND", f"Group chat with ID or name '{chat_id}' not found")
            
        return resolved_id, None
    except Exception as e:
        logger.error(f"Error resolving chat ID: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, error_response("ERROR", f"Error resolving chat ID: {str(e)}")

def parse_natural_language_time_period(query: str) -> Tuple[datetime, datetime]:
    """Extract a natural language time period from a query string.
    
    Args:
        query: The natural language query string
        
    Returns:
        Tuple of (start_date, end_date) as datetime objects
    """
    from datetime import timedelta
    import re
    import dateutil.relativedelta
    
    now = datetime.now()
    
    # Default to 30 days if no time period specified
    start_date = now - timedelta(days=30)
    end_date = now
    
    # Try to extract time periods like "past X days/weeks/months/years"
    past_time_match = re.search(r'past\s+(\d+)\s+(day|days|week|weeks|month|months|year|years)', query, re.IGNORECASE)
    if past_time_match:
        amount = int(past_time_match.group(1))
        unit = past_time_match.group(2).lower()
        
        if unit in ['day', 'days']:
            start_date = now - timedelta(days=amount)
        elif unit in ['week', 'weeks']:
            start_date = now - timedelta(weeks=amount)
        elif unit in ['month', 'months']:
            start_date = now - dateutil.relativedelta.relativedelta(months=amount)
        elif unit in ['year', 'years']:
            start_date = now - dateutil.relativedelta.relativedelta(years=amount)
    
    # Try to extract time periods like "last week/month/year"
    last_period_match = re.search(r'last\s+(week|month|year)', query, re.IGNORECASE)
    if last_period_match:
        unit = last_period_match.group(1).lower()
        
        if unit == 'week':
            # Start from beginning of last week (Sunday)
            last_week_start = now - timedelta(days=now.weekday() + 7)
            start_date = datetime(last_week_start.year, last_week_start.month, last_week_start.day)
            end_date = start_date + timedelta(days=6, hours=23, minutes=59, seconds=59)
        elif unit == 'month':
            # Start from beginning of last month
            if now.month == 1:
                start_date = datetime(now.year - 1, 12, 1)
            else:
                start_date = datetime(now.year, now.month - 1, 1)
            # End at end of last month
            end_date = datetime(now.year, now.month, 1) - timedelta(seconds=1)
        elif unit == 'year':
            # Start from beginning of last year
            start_date = datetime(now.year - 1, 1, 1)
            end_date = datetime(now.year - 1, 12, 31, 23, 59, 59)
    
    return start_date, end_date

def should_analyze_all_contacts(query: str) -> bool:
    """Determine if a query is asking about all contacts.
    
    Args:
        query: The natural language query string
        
    Returns:
        True if query is about all contacts, False otherwise
    """
    all_contacts_patterns = [
        r'\ball\s+contacts\b',
        r'\beveryone\b',
        r'\ball\s+conversations\b',
        r'\ball\s+chats\b',
        r'\ball\s+of\s+my\s+contacts\b',
    ]
    
    for pattern in all_contacts_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return True
            
    return False

def extract_phone_number(query: str) -> Optional[str]:
    """Extract a phone number from a query string.
    
    Args:
        query: The natural language query string
        
    Returns:
        Extracted phone number or None if not found
    """
    # Look for phone numbers with various formats
    phone_patterns = [
        r'(?:^|\s|\(|\[)(\+\d{1,3}\s?)?(\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?:$|\s|\)|\])',  # (123) 456-7890
        r'(?:^|\s|\(|\[)\+\d{1,3}\s\d{1,14}(?:$|\s|\)|\])',  # +1 1234567890
        r'(?:^|\s|\(|\[)\d{7,15}(?:$|\s|\)|\])',  # 1234567890
    ]
    
    for pattern in phone_patterns:
        match = re.search(pattern, query)
        if match:
            # Extract the full match and remove any non-digit or + characters
            phone_number = match.group(0).strip()
            phone_number = re.sub(r'[^\d+]', '', phone_number)
            return phone_number
            
    return None

def should_analyze_group_chat(query: str) -> bool:
    """Determine if a query is asking about a group chat.
    
    Args:
        query: The natural language query string
        
    Returns:
        True if query is about a group chat, False otherwise
    """
    group_chat_patterns = [
        r'\bgroup\s+chat\b',
        r'\bgroup\s+conversation\b',
        r'\bgroup\s+message\b',
        r'\bgroup\b'
    ]
    
    for pattern in group_chat_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return True
            
    return False

def extract_group_chat_name(query: str) -> Optional[str]:
    """Extract a group chat name from a query string.
    
    Args:
        query: The natural language query string
        
    Returns:
        Extracted group chat name or None if not found
    """
    # First, check if query mentions a group chat
    if not should_analyze_group_chat(query):
        return None
        
    # Look for quoted text which might be a group name
    quote_match = re.search(r'"([^"]+)"', query)
    if quote_match:
        return quote_match.group(1).strip()
        
    # Try to extract text following group chat keywords
    keyword_match = re.search(r'\b(?:group|group\s+chat|group\s+conversation)\s+(?:called|named)?\s+([^,.!?]+)', query, re.IGNORECASE)
    if keyword_match:
        return keyword_match.group(1).strip()
        
    return None 