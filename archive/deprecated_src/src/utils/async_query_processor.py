import asyncio
import logging
import re
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import dateutil.parser
import dateutil.relativedelta

from database.async_messages_db import AsyncMessagesDB

# Configure logging
logger = logging.getLogger(__name__)

# Common patterns for extracting time periods
TIME_PATTERNS = {
    "last_n_days": r"(?:last|past)\s+(\d+)\s+days?",
    "last_n_weeks": r"(?:last|past)\s+(\d+)\s+weeks?",
    "last_n_months": r"(?:last|past)\s+(\d+)\s+months?",
    "yesterday": r"yesterday",
    "today": r"today",
    "this_week": r"this\s+week",
    "this_month": r"this\s+month",
    "this_year": r"this\s+year",
    "since_date": r"since\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?",
    "between_dates": r"between\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?\s+and\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?",
}

# Contact and chat patterns
CONTACT_PATTERNS = {
    "with_contact": r"(?:with|from|to)\s+([\"']([^\"']+)[\"']|(\w+(?:\s+\w+)*?))",
    "about_contact": r"(?:about|regarding|concerning)\s+([\"']([^\"']+)[\"']|(\w+(?:\s+\w+)*?))",
    "contact_name": r"(?:contact|person)\s+(?:named|called)\s+([\"']([^\"']+)[\"']|(\w+(?:\s+\w+)*?))",
}

GROUP_CHAT_PATTERNS = {
    "in_group": r"(?:in|within|from)\s+(?:the\s+)?(?:group|chat|conversation)\s+(?:named|called)?\s+([\"']([^\"']+)[\"']|(\w+(?:\s+\w+)*?))",
    "about_group": r"(?:about|regarding|concerning)\s+(?:the\s+)?(?:group|chat|conversation)\s+(?:named|called)?\s+([\"']([^\"']+)[\"']|(\w+(?:\s+\w+)*?))",
    "group_name": r"(?:group|chat|conversation)\s+(?:named|called)\s+([\"']([^\"']+)[\"']|(\w+(?:\s+\w+)*?))",
}

# Analysis type patterns
ANALYSIS_TYPES = {
    "sentiment": r"(?:sentiment|mood|emotion|feeling)s?",
    "network": r"(?:network|connection|relationship)s?",
    "frequency": r"(?:frequency|often|count)",
    "statistics": r"(?:statistics|stats|numbers|data)",
    "visualization": r"(?:visualization|visual|graph|chart)",
}


async def parse_natural_language_time_period(
    query: str,
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Extract a time period from a natural language query.

    Args:
        query: Natural language query

    Returns:
        Tuple of (start_date, end_date) as datetime objects or None if not found
    """
    now = datetime.now()

    # Default to last 7 days if nothing matches
    start_date = now - timedelta(days=7)
    end_date = now

    # Check for "last n days"
    days_match = re.search(TIME_PATTERNS["last_n_days"], query, re.IGNORECASE)
    if days_match:
        days = int(days_match.group(1))
        start_date = now - timedelta(days=days)
        return start_date, end_date

    # Check for "last n weeks"
    weeks_match = re.search(TIME_PATTERNS["last_n_weeks"], query, re.IGNORECASE)
    if weeks_match:
        weeks = int(weeks_match.group(1))
        start_date = now - timedelta(weeks=weeks)
        return start_date, end_date

    # Check for "last n months"
    months_match = re.search(TIME_PATTERNS["last_n_months"], query, re.IGNORECASE)
    if months_match:
        months = int(months_match.group(1))
        start_date = now - dateutil.relativedelta.relativedelta(months=months)
        return start_date, end_date

    # Check for "yesterday"
    if re.search(TIME_PATTERNS["yesterday"], query, re.IGNORECASE):
        start_date = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(
            days=1
        )
        end_date = now.replace(
            hour=23, minute=59, second=59, microsecond=999999
        ) - timedelta(days=1)
        return start_date, end_date

    # Check for "today"
    if re.search(TIME_PATTERNS["today"], query, re.IGNORECASE):
        start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return start_date, end_date

    # Check for "this week"
    if re.search(TIME_PATTERNS["this_week"], query, re.IGNORECASE):
        # Start of the current week (Monday)
        weekday = now.weekday()
        start_date = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(
            days=weekday
        )
        return start_date, end_date

    # Check for "this month"
    if re.search(TIME_PATTERNS["this_month"], query, re.IGNORECASE):
        start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return start_date, end_date

    # Check for "this year"
    if re.search(TIME_PATTERNS["this_year"], query, re.IGNORECASE):
        start_date = now.replace(
            month=1, day=1, hour=0, minute=0, second=0, microsecond=0
        )
        return start_date, end_date

    # Check for "since [date]"
    since_match = re.search(TIME_PATTERNS["since_date"], query, re.IGNORECASE)
    if since_match:
        month = since_match.group(1)
        day = int(since_match.group(2))
        year = int(since_match.group(3)) if since_match.group(3) else now.year

        try:
            start_date = datetime.strptime(f"{month} {day} {year}", "%B %d %Y")
            return start_date, end_date
        except ValueError:
            pass

    # Check for "between [date] and [date]"
    between_match = re.search(TIME_PATTERNS["between_dates"], query, re.IGNORECASE)
    if between_match:
        month1 = between_match.group(1)
        day1 = int(between_match.group(2))
        year1 = int(between_match.group(3)) if between_match.group(3) else now.year

        month2 = between_match.group(4)
        day2 = int(between_match.group(5))
        year2 = int(between_match.group(6)) if between_match.group(6) else now.year

        try:
            start_date = datetime.strptime(f"{month1} {day1} {year1}", "%B %d %Y")
            end_date = datetime.strptime(f"{month2} {day2} {year2}", "%B %d %Y")
            return start_date, end_date
        except ValueError:
            pass

    # If no specific period found, return a default period (last 7 days)
    return start_date, end_date


async def extract_contact_info(query: str) -> Optional[str]:
    """Extract contact information from a query."""
    for pattern_name, pattern in CONTACT_PATTERNS.items():
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            # Return the quoted string if it exists, otherwise the unquoted name
            return (
                match.group(2)
                if match.group(2)
                else match.group(3) if match.group(3) else match.group(1)
            )
    return None


async def extract_group_chat_info(query: str) -> Optional[str]:
    """Extract group chat information from a query."""
    for pattern_name, pattern in GROUP_CHAT_PATTERNS.items():
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            # Return the quoted string if it exists, otherwise the unquoted name
            return (
                match.group(2)
                if match.group(2)
                else match.group(3) if match.group(3) else match.group(1)
            )
    return None


async def determine_analysis_type(query: str) -> str:
    """Determine the type of analysis requested."""
    for analysis_type, pattern in ANALYSIS_TYPES.items():
        if re.search(pattern, query, re.IGNORECASE):
            return analysis_type

    # Default to statistics if no specific analysis type is found
    return "statistics"


async def process_natural_language_query_async(query: str) -> Dict[str, Any]:
    """
    Process a natural language query about message data.

    Args:
        query: The natural language query to process

    Returns:
        Dictionary containing the query results
    """
    try:
        if not query:
            return {"error": "Empty query provided", "query": query}

        logger.info(f"Processing query: {query}")

        # Initialize DB connection
        db = AsyncMessagesDB()

        # Extract time period
        start_date, end_date = await parse_natural_language_time_period(query)

        # Extract contact and group chat info
        contact_info = await extract_contact_info(query)
        group_chat_info = await extract_group_chat_info(query)

        # Determine analysis type
        analysis_type = await determine_analysis_type(query)

        # Build response based on parsed query
        response = {
            "query": query,
            "parsed": {
                "time_period": {
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None,
                },
                "contact": contact_info,
                "group_chat": group_chat_info,
                "analysis_type": analysis_type,
            },
        }

        # Execute the appropriate query based on parsed information
        if analysis_type == "sentiment" and (contact_info or group_chat_info):
            from analysis.async_sentiment_analysis import \
                analyze_sentiment_async

            if contact_info:
                # Get sentiment analysis for contact
                sentiment_results = await analyze_sentiment_async(
                    phone_number=contact_info,
                    start_date=start_date.isoformat() if start_date else None,
                    end_date=end_date.isoformat() if end_date else None,
                )
                response["results"] = sentiment_results
            elif group_chat_info:
                # Get sentiment analysis for group chat
                sentiment_results = await analyze_sentiment_async(
                    chat_id=group_chat_info,
                    start_date=start_date.isoformat() if start_date else None,
                    end_date=end_date.isoformat() if end_date else None,
                )
                response["results"] = sentiment_results

        elif analysis_type == "network":
            from analysis.async_network_analysis import \
                analyze_contact_network_async

            # Get network analysis
            network_results = await analyze_contact_network_async(
                start_date=start_date.isoformat() if start_date else None,
                end_date=end_date.isoformat() if end_date else None,
            )
            response["results"] = network_results

        elif analysis_type == "visualization" and analysis_type == "network":
            from visualization.async_network_viz import \
                generate_network_visualization_async

            # Get network visualization
            viz_results = await generate_network_visualization_async(
                start_date=start_date.isoformat() if start_date else None,
                end_date=end_date.isoformat() if end_date else None,
            )
            response["results"] = viz_results

        elif contact_info:
            # Default analysis for a contact
            contact_results = await db.analyze_contact(
                contact_info,
                start_date=start_date.isoformat() if start_date else None,
                end_date=end_date.isoformat() if end_date else None,
            )
            response["results"] = contact_results

        elif group_chat_info:
            # Default analysis for a group chat
            group_chat_results = await db.analyze_group_chat(
                group_chat_info,
                start_date=start_date.isoformat() if start_date else None,
                end_date=end_date.isoformat() if end_date else None,
            )
            response["results"] = group_chat_results

        else:
            # If no specific entity mentioned, provide a general summary
            # Get counts of messages for the period
            contacts = await db.get_contacts()
            group_chats = await db.get_group_chats()

            active_contacts = [
                contact
                for contact in contacts
                if contact.get("last_message_date")
                and (
                    not start_date
                    or datetime.fromisoformat(contact["last_message_date"])
                    >= start_date
                )
                and (
                    not end_date
                    or datetime.fromisoformat(contact["last_message_date"]) <= end_date
                )
            ]

            active_group_chats = [
                chat
                for chat in group_chats
                if chat.get("last_message_date")
                and (
                    not start_date
                    or datetime.fromisoformat(chat["last_message_date"]) >= start_date
                )
                and (
                    not end_date
                    or datetime.fromisoformat(chat["last_message_date"]) <= end_date
                )
            ]

            response["results"] = {
                "summary": {
                    "active_contacts": len(active_contacts),
                    "active_group_chats": len(active_group_chats),
                    "top_contacts": sorted(
                        active_contacts,
                        key=lambda x: x.get("message_count", 0),
                        reverse=True,
                    )[:5],
                    "top_group_chats": sorted(
                        active_group_chats,
                        key=lambda x: x.get("message_count", 0),
                        reverse=True,
                    )[:5],
                }
            }

        return response

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        logger.error(traceback.format_exc())
        return {"error": f"Error processing query: {str(e)}", "query": query}
