#!/usr/bin/env python3
"""
Generate synthetic iMessage-like data for documentation and examples.

This script creates realistic but completely fake conversation data
for use in screenshots, examples, and documentation.
"""

import hashlib
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

# Synthetic contact pool
CONTACTS = [
    {"name": "Alice Johnson", "phone": "+1-555-0101", "email": "alice@example.com"},
    {"name": "Bob Smith", "phone": "+1-555-0102", "email": "bob@example.com"},
    {"name": "Carol Davis", "phone": "+1-555-0103", "email": "carol@example.com"},
    {"name": "David Chen", "phone": "+1-555-0104", "email": "david@example.com"},
    {"name": "Emma Wilson", "phone": "+1-555-0105", "email": "emma@example.com"},
    {"name": "Frank Miller", "phone": "+1-555-0106", "email": "frank@example.com"},
    {"name": "Grace Lee", "phone": "+1-555-0107", "email": "grace@example.com"},
    {"name": "Henry Brown", "phone": "+1-555-0108", "email": "henry@example.com"},
]

# Message templates for different conversation types
MESSAGE_TEMPLATES = {
    "casual": [
        "Hey! How's it going?",
        "Pretty good! Just finished work. You?",
        "Want to grab coffee this weekend?",
        "Sure! Saturday morning works for me",
        "Did you see the game last night?",
        "Yeah, what a comeback!",
        "I'm thinking of trying that new restaurant",
        "I heard it's really good!",
        "Thanks for the recommendation!",
        "No problem! Let me know how it is"
    ],
    "work": [
        "Quick question about the project",
        "Sure, what's up?",
        "Can we move the meeting to 3pm?",
        "That works for me",
        "I'll send the updated slides",
        "Great, thanks!",
        "The client loved the proposal",
        "That's fantastic news!",
        "Team lunch tomorrow?",
        "Count me in!"
    ],
    "family": [
        "How was your day?",
        "Really good! Had a productive meeting",
        "Don't forget dinner at 7",
        "I'll be there!",
        "Can you pick up milk on your way home?",
        "Already got it!",
        "Love you!",
        "Love you too!",
        "Safe travels!",
        "Thanks! I'll text when I land"
    ]
}

# Sentiment progression patterns
SENTIMENT_PATTERNS = {
    "improving": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55],
    "stable_positive": [0.6, 0.65, 0.6, 0.65, 0.7, 0.65, 0.6, 0.65, 0.7, 0.65],
    "declining": [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05],
    "volatile": [0.2, 0.6, 0.3, 0.7, 0.1, 0.8, 0.4, 0.5, 0.2, 0.9]
}


def hash_identifier(identifier: str) -> str:
    """Hash an identifier for privacy."""
    return hashlib.sha256(identifier.encode()).hexdigest()


def generate_conversation_data(
    contact: Dict[str, str],
    num_messages: int = 100,
    days_back: int = 90,
    conversation_type: str = "casual"
) -> List[Dict[str, Any]]:
    """Generate synthetic conversation data."""
    messages = []
    templates = MESSAGE_TEMPLATES.get(conversation_type, MESSAGE_TEMPLATES["casual"])

    # Generate messages over time
    current_time = datetime.now()
    for i in range(num_messages):
        # Random time in the past
        days_ago = random.uniform(0, days_back)
        hours_offset = random.uniform(0, 24)
        timestamp = current_time - timedelta(days=days_ago, hours=hours_offset)

        # Alternate between sent and received
        is_from_me = i % 2 == 0

        # Pick a message
        message_text = random.choice(templates)

        # Create message record
        message = {
            "timestamp": timestamp.isoformat(),
            "contact_id": hash_identifier(contact["phone"]),
            "contact_name": contact["name"],
            "is_from_me": is_from_me,
            "text": message_text,
            "conversation_type": conversation_type
        }
        messages.append(message)

    # Sort by timestamp
    messages.sort(key=lambda x: x["timestamp"])
    return messages


def generate_analytics_data() -> Dict[str, Any]:
    """Generate synthetic analytics data for examples."""
    analytics = {
        "relationship_intelligence": {
            "contact_id": hash_identifier("+1-555-0101"),
            "contact_name": "Alice Johnson",
            "messages_total": 842,
            "sent_percentage": 56.2,
            "received_percentage": 43.8,
            "avg_response_time_minutes": 6.8,
            "median_response_time_minutes": 4.2,
            "engagement_score": 0.87,
            "conversation_depth_score": 0.72,
            "topic_diversity": 12,
            "emotional_consistency": 0.81,
            "flags": ["highly-engaged", "quick-responder", "conversation-initiator"],
            "insights": [
                "Response times have improved by 40% over the last month",
                "Conversations show high emotional stability",
                "You typically initiate 65% of conversations"
            ]
        },
        "sentiment_evolution": {
            "contact_id": hash_identifier("+1-555-0102"),
            "series": [
                {"date": "2024-10-01", "score": 0.42, "message_count": 23},
                {"date": "2024-10-08", "score": 0.45, "message_count": 31},
                {"date": "2024-10-15", "score": 0.48, "message_count": 28},
                {"date": "2024-10-22", "score": 0.52, "message_count": 35},
                {"date": "2024-10-29", "score": 0.55, "message_count": 42},
                {"date": "2024-11-05", "score": 0.58, "message_count": 38},
                {"date": "2024-11-12", "score": 0.61, "message_count": 45},
                {"date": "2024-11-19", "score": 0.64, "message_count": 51},
                {"date": "2024-11-26", "score": 0.67, "message_count": 48},
                {"date": "2024-12-03", "score": 0.70, "message_count": 55}
            ],
            "trend": "improving",
            "delta_30d": 0.15,
            "volatility": 0.08
        },
        "communication_heatmap": {
            "contact_id": hash_identifier("+1-555-0103"),
            "data": {
                "Monday": {"9": 12, "10": 8, "14": 15, "15": 18, "19": 22, "20": 25},
                "Tuesday": {"9": 10, "11": 12, "14": 8, "16": 14, "19": 20, "20": 18},
                "Wednesday": {"10": 15, "11": 12, "15": 20, "16": 18, "20": 28, "21": 22},
                "Thursday": {"9": 8, "10": 10, "14": 12, "15": 15, "19": 18, "20": 20},
                "Friday": {"10": 5, "11": 8, "14": 10, "15": 12, "17": 15, "18": 20},
                "Saturday": {"11": 18, "12": 22, "14": 15, "15": 12, "18": 25, "19": 30},
                "Sunday": {"11": 20, "12": 25, "15": 18, "16": 15, "19": 22, "20": 28}
            },
            "peak_hours": ["20:00", "19:00", "21:00"],
            "peak_days": ["Saturday", "Sunday", "Wednesday"]
        },
        "network_analysis": {
            "total_contacts": 45,
            "active_contacts": 12,
            "communities_detected": 3,
            "key_connectors": [
                {"contact_id": hash_identifier("+1-555-0104"), "centrality": 0.82},
                {"contact_id": hash_identifier("+1-555-0105"), "centrality": 0.75},
                {"contact_id": hash_identifier("+1-555-0106"), "centrality": 0.68}
            ],
            "network_density": 0.42,
            "average_degree": 5.2
        }
    }
    return analytics


def save_synthetic_data():
    """Save synthetic data to JSON files for documentation."""
    output_dir = Path("assets/synthetic_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate conversations
    conversations = {}
    for contact in CONTACTS[:3]:  # First 3 contacts
        conv_type = random.choice(["casual", "work", "family"])
        conversations[contact["name"]] = generate_conversation_data(
            contact,
            num_messages=50,
            days_back=60,
            conversation_type=conv_type
        )

    # Save conversations
    with open(output_dir / "conversations.json", "w") as f:
        json.dump(conversations, f, indent=2)

    # Generate and save analytics
    analytics = generate_analytics_data()
    with open(output_dir / "analytics.json", "w") as f:
        json.dump(analytics, f, indent=2)

    # Generate tool outputs
    tool_outputs = {
        "health_check": {
            "db_accessible": True,
            "schema_valid": True,
            "indexes_present": True,
            "read_only": True,
            "stats": {
                "total_messages": 15234,
                "total_contacts": 89,
                "date_range": {
                    "earliest": "2020-01-15",
                    "latest": "2024-12-15"
                },
                "database_size_mb": 128.5
            },
            "status": "healthy"
        },
        "summary_overview": {
            "total_messages": 15234,
            "total_contacts": 89,
            "active_contacts_30d": 12,
            "date_range": {
                "earliest": "2020-01-15T10:30:00",
                "latest": "2024-12-15T18:45:00"
            },
            "top_contacts": [
                {"contact_id": hash_identifier("+1-555-0101")[:8], "messages": 842},
                {"contact_id": hash_identifier("+1-555-0102")[:8], "messages": 756},
                {"contact_id": hash_identifier("+1-555-0103")[:8], "messages": 623}
            ]
        }
    }

    with open(output_dir / "tool_outputs.json", "w") as f:
        json.dump(tool_outputs, f, indent=2)

    print(f"Synthetic data generated in {output_dir}")


if __name__ == "__main__":
    save_synthetic_data()
