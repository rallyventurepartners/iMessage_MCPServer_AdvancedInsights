"""
Emoji usage analysis tools for the iMessage Advanced Insights server.

This module provides tools for analyzing emoji usage patterns, trends,
and communication styles through emoji usage.
"""

import logging
import re
import unicodedata
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import defaultdict, Counter
import statistics

from ..database import get_database
from ..exceptions import DatabaseError, ToolExecutionError
from ..utils.responses import error_response, success_response
from ..utils.decorators import requires_consent, parse_date
from .registry import register_tool

logger = logging.getLogger(__name__)


# Emoji categories based on Unicode blocks and common usage
EMOJI_CATEGORIES = {
    'faces_positive': {
        '😀', '😃', '😄', '😁', '😆', '😅', '😂', '🤣', '😊', '😇', '🙂', '🙃',
        '😉', '😌', '😍', '🥰', '😘', '😗', '😙', '😚', '😋', '😛', '😜', '🤪',
        '😝', '🤗', '🤭', '🤫', '🤔', '🤠', '🥳', '😎', '🤓', '🧐'
    },
    'faces_negative': {
        '😕', '😟', '🙁', '☹️', '😮', '😯', '😲', '😳', '🥺', '😦', '😧', '😨',
        '😰', '😥', '😢', '😭', '😱', '😖', '😣', '😞', '😓', '😩', '😫', '🥱',
        '😤', '😡', '😠', '🤬', '😈', '👿', '💀', '☠️', '😷', '🤒', '🤕', '🤢',
        '🤮', '🤧', '🥵', '🥶', '🥴', '😵', '🤯'
    },
    'hearts': {
        '❤️', '🧡', '💛', '💚', '💙', '💜', '🖤', '🤍', '🤎', '💔', '❣️', '💕',
        '💞', '💓', '💗', '💖', '💘', '💝', '💟', '♥️'
    },
    'gestures': {
        '👋', '🤚', '🖐️', '✋', '🖖', '👌', '🤌', '🤏', '✌️', '🤞', '🤟', '🤘',
        '🤙', '👈', '👉', '👆', '🖕', '👇', '☝️', '👍', '👎', '👊', '✊', '🤛',
        '🤜', '👏', '🙌', '👐', '🤲', '🤝', '🙏'
    },
    'activities': {
        '🎉', '🎊', '🎈', '🎁', '🎄', '🎃', '🎆', '🎇', '🧨', '✨', '🎪', '🎭',
        '🎨', '🎬', '🎤', '🎧', '🎼', '🎵', '🎶', '🎹', '🥁', '🎷', '🎺', '🎸',
        '🎻', '🎲', '♟️', '🎯', '🎳', '🎮', '🎰', '🧩'
    },
    'food_drink': {
        '🍏', '🍎', '🍐', '🍊', '🍋', '🍌', '🍉', '🍇', '🍓', '🫐', '🍈', '🍒',
        '🍑', '🥭', '🍍', '🥥', '🥝', '🍅', '🍆', '🥑', '🥦', '🥬', '🥒', '🌶️',
        '🫑', '🌽', '🥕', '🫒', '🧄', '🧅', '🥔', '🍠', '🥐', '🥖', '🍞', '🥨',
        '🥯', '🥞', '🧇', '🧀', '🍖', '🍗', '🥩', '🥓', '🍔', '🍟', '🍕', '🌭',
        '🥪', '🌮', '🌯', '🫔', '🥙', '🧆', '🥚', '🍳', '🥘', '🍲', '🫕', '🥣',
        '🥗', '🍿', '🧈', '🧂', '🥫', '🍱', '🍘', '🍙', '🍚', '🍛', '🍜', '🍝',
        '🍠', '🍢', '🍣', '🍤', '🍥', '🥮', '🍡', '🥟', '🥠', '🥡', '🦀', '🦞',
        '🦐', '🦑', '🦪', '🍦', '🍧', '🍨', '🍩', '🍪', '🎂', '🍰', '🧁', '🥧',
        '🍫', '🍬', '🍭', '🍮', '🍯', '☕', '🍵', '🧃', '🥤', '🧋', '🍶', '🍺',
        '🍻', '🥂', '🍷', '🥃', '🍸', '🍹', '🧉', '🍾'
    },
    'animals': {
        '🐶', '🐱', '🐭', '🐹', '🐰', '🦊', '🐻', '🐼', '🐻‍❄️', '🐨', '🐯', '🦁',
        '🐮', '🐷', '🐽', '🐸', '🐵', '🙈', '🙉', '🙊', '🐒', '🐔', '🐧', '🐦',
        '🐤', '🐣', '🐥', '🦆', '🦅', '🦉', '🦇', '🐺', '🐗', '🐴', '🦄', '🐝',
        '🪱', '🐛', '🦋', '🐌', '🐞', '🐜', '🪰', '🪲', '🪳', '🦟', '🦗', '🕷️',
        '🕸️', '🦂', '🐢', '🐍', '🦎', '🦖', '🦕', '🐙', '🦑', '🦐', '🦞', '🦀',
        '🐡', '🐠', '🐟', '🐬', '🐳', '🐋', '🦈', '🐊', '🐅', '🐆', '🦓', '🦍',
        '🦧', '🐘', '🦛', '🦏', '🐪', '🐫', '🦒', '🦘', '🦬', '🐃', '🐂', '🐄',
        '🐎', '🐖', '🐏', '🐑', '🦙', '🐐', '🦌', '🐕', '🐩', '🦮', '🐕‍🦺', '🐈',
        '🐈‍⬛', '🪶', '🐓', '🦃', '🦤', '🦚', '🦜', '🦢', '🦩', '🕊️', '🐇', '🦝',
        '🦨', '🦡', '🦫', '🦦', '🦥', '🐁', '🐀', '🐿️', '🦔'
    },
    'objects': {
        '📱', '💻', '⌨️', '🖥️', '🖨️', '🖱️', '🖲️', '🕹️', '🗜️', '💾', '💿', '📀',
        '📼', '📷', '📸', '📹', '🎥', '📽️', '🎞️', '📞', '☎️', '📟', '📠', '📺',
        '📻', '🎙️', '🎚️', '🎛️', '🧭', '⏱️', '⏲️', '⏰', '🕰️', '⌛', '⏳', '📡',
        '🔋', '🔌', '💡', '🔦', '🕯️', '🪔', '🧯', '🛢️', '💸', '💵', '💴', '💶',
        '💷', '🪙', '💰', '💳', '💎', '⚖️', '🪜', '🧰', '🪛', '🔧', '🔨', '⚒️',
        '🛠️', '⛏️', '🪚', '🔩', '⚙️', '🪤', '🧱', '⛓️', '🧲', '🔫', '💣', '🧨',
        '🪓', '🔪', '🗡️', '⚔️', '🛡️', '🚬', '⚰️', '🪦', '⚱️', '🏺', '🔮', '📿',
        '🧿', '💈', '⚗️', '🔭', '🔬', '🕳️', '🩹', '🩺', '💊', '💉', '🩸', '🧬',
        '🦠', '🧫', '🧪', '🌡️', '🧹', '🪠', '🧺', '🧻', '🚽', '🚰', '🚿', '🛁',
        '🛀', '🧼', '🪥', '🪒', '🧽', '🪣', '🧴', '🛎️', '🔑', '🗝️', '🚪', '🪑',
        '🛋️', '🛏️', '🛌', '🧸', '🖼️', '🪆', '🛍️', '🛒', '🎁', '🎈', '🎏', '🎀',
        '🪄', '🪅', '🎊', '🎉', '🎎', '🏮', '🎐', '🧧', '✉️', '📩', '📨', '📧',
        '💌', '📥', '📤', '📦', '🏷️', '📪', '📫', '📬', '📭', '📮', '📯', '📜',
        '📃', '📄', '📑', '🧾', '📊', '📈', '📉', '🗒️', '🗓️', '📆', '📅', '🗑️',
        '📇', '🗃️', '🗳️', '🗄️', '📋', '📁', '📂', '🗂️', '🗞️', '📰', '📓', '📔',
        '📒', '📕', '📗', '📘', '📙', '📚', '📖', '🔖', '🧷', '🔗', '📎', '🖇️',
        '📐', '📏', '🧮', '📌', '📍', '✂️', '🖊️', '🖋️', '✒️', '🖌️', '🖍️', '📝',
        '✏️', '🔍', '🔎', '🔏', '🔐', '🔒', '🔓'
    },
    'symbols': {
        '❤️', '💔', '❣️', '💕', '💞', '💓', '💗', '💖', '💘', '💝', '❗', '❓',
        '❕', '❔', '❗', '‼️', '⁉️', '💯', '🔴', '🟠', '🟡', '🟢', '🔵', '🟣',
        '⚫', '⚪', '🟤', '🔺', '🔻', '🔸', '🔹', '🔶', '🔷', '🔳', '🔲', '▪️',
        '▫️', '◾', '◽', '◼️', '◻️', '🟥', '🟧', '🟨', '🟩', '🟦', '🟪', '⬛',
        '⬜', '🟫', '🔻', '🔺'
    }
}


def is_emoji(char: str) -> bool:
    """Check if a character is an emoji."""
    return unicodedata.category(char) in ('So', 'Sk') or char in ''.join(EMOJI_CATEGORIES.values())


def extract_emojis(text: str) -> List[str]:
    """Extract all emojis from text."""
    if not text:
        return []
    
    emojis = []
    # Handle multi-character emojis (with modifiers)
    i = 0
    while i < len(text):
        # Check for emoji sequences (base + modifier)
        if i < len(text) - 1:
            two_char = text[i:i+2]
            if is_emoji(two_char[0]) and (two_char[1] in '\u200d\ufe0f' or unicodedata.category(two_char[1]) == 'Mn'):
                # This might be a multi-character emoji
                emoji_sequence = two_char[0]
                j = i + 1
                while j < len(text) and (text[j] in '\u200d\ufe0f' or unicodedata.category(text[j]) in ('Mn', 'So')):
                    emoji_sequence += text[j]
                    j += 1
                emojis.append(emoji_sequence)
                i = j
                continue
        
        # Single character emoji
        if is_emoji(text[i]):
            emojis.append(text[i])
        
        i += 1
    
    return emojis


def categorize_emoji(emoji: str) -> str:
    """Categorize an emoji into predefined categories."""
    for category, emoji_set in EMOJI_CATEGORIES.items():
        if emoji in emoji_set:
            return category
    return 'other'


@register_tool(
    name="analyze_emoji_usage",
    description="Analyze emoji usage patterns"
)
@requires_consent
async def analyze_emoji_usage_tool(
    contact_id: Optional[str] = None,
    time_period: str = "90 days"
) -> Dict[str, Any]:
    """
    Analyze emoji usage patterns in conversations.
    
    Args:
        contact_id: Optional contact to analyze (None = all conversations)
        time_period: Time period to analyze (e.g., "30 days", "3 months")
        
    Returns:
        Comprehensive emoji usage analysis with patterns and insights
    """
    try:
        # Get database connection
        db = await get_database()
        
        # Parse time period
        end_date = datetime.now()
        start_date = parse_date(time_period)
        if not start_date:
            start_date = end_date - timedelta(days=90)
        
        # Get messages with memory-safe pagination
        from ..utils.memory_monitor import MemoryMonitor
        memory_monitor = None
        try:
            memory_monitor = await MemoryMonitor.get_instance()
            memory_stats = memory_monitor.get_memory_stats()
            if memory_stats["process"]["percent"] > 60:
                page_size = 1000
                logger.warning("High memory usage detected, limiting to 1000 messages per page")
            else:
                page_size = 5000  # Reduced from 10000 for safety
        except:
            page_size = 5000  # Default safe size
            
        if contact_id:
            result = await db.get_messages_from_contact(
                phone_number=contact_id,
                start_date=start_date,
                end_date=end_date,
                page=1, page_size=page_size
            )
        else:
            # Analyze all messages - get recent messages without contact filter
            # Note: There's no direct method to get all messages with date filters
            # We'll use get_messages_from_contact with empty contact to get recent messages
            result = await db.get_messages_from_contact(
                phone_number="",  # Empty to get all messages
                start_date=start_date,
                end_date=end_date,
                page=1, page_size=page_size
            )
        
        messages = result.get("messages", [])
        
        if not messages:
            return success_response({
                "time_period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                    "days": (end_date - start_date).days
                },
                "summary": "No messages found in the specified time period",
                "emoji_usage": {},
                "insights": {}
            })
        
        # Analyze emoji usage
        total_emoji_count = 0
        emoji_frequency = Counter()
        emoji_by_category = defaultdict(int)
        emoji_by_contact = defaultdict(Counter)
        emoji_by_time = defaultdict(list)  # For trend analysis
        messages_with_emoji = 0
        emoji_combinations = Counter()
        
        # Analyze each message
        for msg in messages:
            text = msg.get('text', '')
            if not text:
                continue
            
            # Extract emojis
            emojis = extract_emojis(text)
            if emojis:
                messages_with_emoji += 1
                
                # Count individual emojis
                for emoji in emojis:
                    total_emoji_count += 1
                    emoji_frequency[emoji] += 1
                    
                    # Categorize
                    category = categorize_emoji(emoji)
                    emoji_by_category[category] += 1
                    
                    # Track by contact
                    contact = msg.get('contact_name', msg.get('contact_id', 'Me' if msg.get('is_from_me') else 'Unknown'))
                    emoji_by_contact[contact][emoji] += 1
                    
                    # Track by time for trends
                    msg_date = datetime.fromisoformat(msg['date'])
                    date_key = msg_date.strftime('%Y-%m-%d')
                    emoji_by_time[date_key].append(emoji)
                
                # Track emoji combinations (consecutive emojis)
                if len(emojis) > 1:
                    for i in range(len(emojis) - 1):
                        combo = emojis[i] + emojis[i + 1]
                        emoji_combinations[combo] += 1
        
        # Calculate statistics
        total_messages = len(messages)
        emoji_density = total_emoji_count / total_messages if total_messages > 0 else 0
        emoji_message_ratio = messages_with_emoji / total_messages if total_messages > 0 else 0
        
        # Top emojis
        top_emojis = [
            {
                'emoji': emoji,
                'count': count,
                'percentage': round(count / total_emoji_count * 100, 1) if total_emoji_count > 0 else 0,
                'category': categorize_emoji(emoji)
            }
            for emoji, count in emoji_frequency.most_common(20)
        ]
        
        # Category analysis
        category_stats = {}
        for category, count in emoji_by_category.items():
            category_stats[category] = {
                'count': count,
                'percentage': round(count / total_emoji_count * 100, 1) if total_emoji_count > 0 else 0
            }
        
        # Emoji trends over time
        trend_data = []
        for date_key in sorted(emoji_by_time.keys()):
            emojis_on_date = emoji_by_time[date_key]
            if emojis_on_date:
                # Most common emoji categories for this date
                day_categories = Counter(categorize_emoji(e) for e in emojis_on_date)
                trend_data.append({
                    'date': date_key,
                    'emoji_count': len(emojis_on_date),
                    'dominant_category': day_categories.most_common(1)[0][0] if day_categories else None
                })
        
        # Contact-specific insights
        contact_insights = []
        for contact, emoji_counter in emoji_by_contact.items():
            if emoji_counter:
                top_emoji = emoji_counter.most_common(1)[0]
                contact_insights.append({
                    'contact': contact,
                    'total_emojis': sum(emoji_counter.values()),
                    'unique_emojis': len(emoji_counter),
                    'favorite_emoji': top_emoji[0],
                    'favorite_count': top_emoji[1]
                })
        
        contact_insights.sort(key=lambda x: x['total_emojis'], reverse=True)
        
        # Popular emoji combinations
        top_combinations = [
            {
                'combination': combo,
                'count': count
            }
            for combo, count in emoji_combinations.most_common(10)
        ]
        
        # Generate insights
        insights = {
            'emoji_personality': _determine_emoji_personality(emoji_by_category, total_emoji_count),
            'communication_style': _analyze_emoji_communication_style(emoji_density, category_stats),
            'emotional_expression': _analyze_emotional_expression(category_stats),
            'trends': _analyze_emoji_trends(trend_data),
            'recommendations': _generate_emoji_recommendations(
                emoji_density, category_stats, top_emojis
            )
        }
        
        return success_response({
            'time_period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
                'days': (end_date - start_date).days
            },
            'summary': {
                'total_messages': total_messages,
                'messages_with_emoji': messages_with_emoji,
                'emoji_message_ratio': round(emoji_message_ratio * 100, 1),
                'total_emojis': total_emoji_count,
                'unique_emojis': len(emoji_frequency),
                'emoji_density': round(emoji_density, 2)
            },
            'top_emojis': top_emojis[:20],
            'category_distribution': category_stats,
            'top_combinations': top_combinations,
            'contact_analysis': contact_insights[:10],
            'trends': {
                'daily_usage': trend_data[-30:] if len(trend_data) > 30 else trend_data,
                'trend_direction': _calculate_emoji_trend_direction(trend_data)
            },
            'insights': insights
        })
        
    except Exception as e:
        logger.error(f"Error analyzing emoji usage: {e}", exc_info=True)
        return error_response(f"Failed to analyze emoji usage: {str(e)}")


def _determine_emoji_personality(category_counts: Dict[str, int], total: int) -> str:
    """Determine emoji personality based on usage patterns."""
    if total == 0:
        return "minimal_user"
    
    # Calculate category percentages
    percentages = {
        cat: (count / total * 100) 
        for cat, count in category_counts.items()
    }
    
    # Determine personality
    if percentages.get('faces_positive', 0) > 40:
        return "expressive_positive"
    elif percentages.get('faces_negative', 0) > 20:
        return "emotionally_open"
    elif percentages.get('hearts', 0) > 25:
        return "affectionate"
    elif percentages.get('animals', 0) > 20:
        return "playful"
    elif percentages.get('food_drink', 0) > 20:
        return "lifestyle_focused"
    elif percentages.get('activities', 0) > 15:
        return "activity_oriented"
    elif sum(percentages.values()) < 10:
        return "minimal_user"
    else:
        return "balanced_user"


def _analyze_emoji_communication_style(density: float, categories: Dict) -> str:
    """Analyze communication style based on emoji usage."""
    if density < 0.1:
        return "formal"
    elif density < 0.5:
        return "casual"
    elif density < 1.0:
        return "expressive"
    else:
        return "highly_expressive"


def _analyze_emotional_expression(categories: Dict) -> Dict[str, Any]:
    """Analyze emotional expression through emoji usage."""
    total_emotional = (
        categories.get('faces_positive', {}).get('count', 0) +
        categories.get('faces_negative', {}).get('count', 0) +
        categories.get('hearts', {}).get('count', 0)
    )
    
    if total_emotional == 0:
        return {
            'level': 'low',
            'positivity_ratio': 0.5,
            'description': 'Limited emotional expression through emojis'
        }
    
    positive = (
        categories.get('faces_positive', {}).get('count', 0) +
        categories.get('hearts', {}).get('count', 0)
    )
    negative = categories.get('faces_negative', {}).get('count', 0)
    
    positivity_ratio = positive / (positive + negative) if (positive + negative) > 0 else 0.5
    
    return {
        'level': 'high' if total_emotional > 100 else 'moderate',
        'positivity_ratio': round(positivity_ratio, 2),
        'description': f"{'Predominantly positive' if positivity_ratio > 0.7 else 'Balanced'} emotional expression"
    }


def _analyze_emoji_trends(trend_data: List[Dict]) -> str:
    """Analyze trends in emoji usage over time."""
    if len(trend_data) < 7:
        return "insufficient_data"
    
    # Calculate weekly averages
    weekly_averages = []
    for i in range(0, len(trend_data), 7):
        week_data = trend_data[i:i+7]
        avg = sum(d['emoji_count'] for d in week_data) / len(week_data)
        weekly_averages.append(avg)
    
    if len(weekly_averages) < 2:
        return "stable"
    
    # Simple trend analysis
    recent_avg = sum(weekly_averages[-2:]) / 2
    older_avg = sum(weekly_averages[:-2]) / len(weekly_averages[:-2])
    
    if recent_avg > older_avg * 1.2:
        return "increasing"
    elif recent_avg < older_avg * 0.8:
        return "decreasing"
    else:
        return "stable"


def _calculate_emoji_trend_direction(trend_data: List[Dict]) -> str:
    """Calculate the direction of emoji usage trends."""
    return _analyze_emoji_trends(trend_data)


def _generate_emoji_recommendations(
    density: float,
    categories: Dict,
    top_emojis: List[Dict]
) -> List[str]:
    """Generate recommendations based on emoji usage patterns."""
    recommendations = []
    
    if density < 0.1:
        recommendations.append(
            "Consider using more emojis to add warmth and emotion to your messages"
        )
    elif density > 2.0:
        recommendations.append(
            "Your emoji usage is very high - ensure it matches your audience's communication style"
        )
    
    # Category-specific recommendations
    if categories.get('faces_negative', {}).get('percentage', 0) > 30:
        recommendations.append(
            "High usage of negative emojis detected - consider balancing with more positive expressions"
        )
    
    if categories.get('hearts', {}).get('percentage', 0) < 5:
        recommendations.append(
            "Low usage of heart emojis - consider using them to express affection and warmth"
        )
    
    # Diversity recommendation
    unique_count = len(set(e['emoji'] for e in top_emojis[:10]))
    if unique_count < 5:
        recommendations.append(
            "You tend to use the same few emojis - try diversifying for richer expression"
        )
    
    if not recommendations:
        recommendations.append("Your emoji usage appears well-balanced and appropriate")
    
    return recommendations