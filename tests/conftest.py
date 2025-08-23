"""Pytest configuration and fixtures for iMessage MCP Server tests."""

import random
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database with iMessage schema."""
    temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    db_path = temp_file.name
    temp_file.close()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create minimal iMessage schema
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS handle (
            ROWID INTEGER PRIMARY KEY,
            id TEXT UNIQUE,
            service TEXT
        );
        
        CREATE TABLE IF NOT EXISTS message (
            ROWID INTEGER PRIMARY KEY,
            guid TEXT UNIQUE,
            text TEXT,
            handle_id INTEGER,
            is_from_me INTEGER,
            date INTEGER,
            cache_has_attachments INTEGER DEFAULT 0,
            FOREIGN KEY (handle_id) REFERENCES handle(ROWID)
        );
        
        CREATE TABLE IF NOT EXISTS chat (
            ROWID INTEGER PRIMARY KEY,
            chat_identifier TEXT,
            display_name TEXT,
            style INTEGER
        );
        
        CREATE TABLE IF NOT EXISTS chat_message_join (
            chat_id INTEGER,
            message_id INTEGER,
            PRIMARY KEY (chat_id, message_id),
            FOREIGN KEY (chat_id) REFERENCES chat(ROWID),
            FOREIGN KEY (message_id) REFERENCES message(ROWID)
        );
        
        CREATE TABLE IF NOT EXISTS chat_handle_join (
            chat_id INTEGER,
            handle_id INTEGER,
            PRIMARY KEY (chat_id, handle_id),
            FOREIGN KEY (chat_id) REFERENCES chat(ROWID),
            FOREIGN KEY (handle_id) REFERENCES handle(ROWID)
        );
    """)

    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def populated_db(temp_db):
    """Create a temporary database populated with synthetic data."""
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    # Add handles (contacts)
    contacts = [
        ("+15551234567", "iMessage"),
        ("+15559876543", "iMessage"),
        ("test@example.com", "iMessage"),
        ("+15555555555", "SMS"),
        ("family@group.chat", "iMessage")
    ]

    for contact_id, service in contacts:
        cursor.execute(
            "INSERT INTO handle (id, service) VALUES (?, ?)",
            (contact_id, service)
        )

    # Add messages
    base_date = int((datetime.now() - timedelta(days=180)).timestamp() * 1000000000 - 978307200000000000)

    # Conversation patterns
    patterns = {
        1: {"freq": "daily", "sentiment": "positive"},  # Close friend
        2: {"freq": "weekly", "sentiment": "neutral"},   # Colleague
        3: {"freq": "monthly", "sentiment": "mixed"},    # Acquaintance
        4: {"freq": "rare", "sentiment": "neutral"},     # SMS contact
    }

    message_id = 1
    for handle_id, pattern in patterns.items():
        # Generate messages based on pattern
        if pattern["freq"] == "daily":
            num_messages = 180
            interval = 86400000000000  # 1 day in nanoseconds
        elif pattern["freq"] == "weekly":
            num_messages = 26
            interval = 604800000000000  # 1 week
        elif pattern["freq"] == "monthly":
            num_messages = 6
            interval = 2592000000000000  # 30 days
        else:
            num_messages = 3
            interval = 5184000000000000  # 60 days

        for i in range(num_messages):
            date = base_date + (i * interval) + random.randint(-3600000000000, 3600000000000)

            # Alternate between sent and received
            for is_from_me in [0, 1]:
                text = generate_message_text(pattern["sentiment"], is_from_me)

                cursor.execute(
                    """INSERT INTO message 
                    (ROWID, guid, text, handle_id, is_from_me, date, cache_has_attachments)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        message_id,
                        f"message-{message_id}",
                        text,
                        handle_id,
                        is_from_me,
                        date + (300000000000 if is_from_me else 0),  # 5 min delay for responses
                        random.choice([0, 0, 0, 1])  # 25% chance of attachment
                    )
                )
                message_id += 1

    # Add a group chat
    cursor.execute(
        """INSERT INTO chat (ROWID, chat_identifier, display_name, style)
        VALUES (1, 'chat123456', 'Family Group', 45)"""
    )

    # Add members to group chat
    for handle_id in [1, 2, 3]:
        cursor.execute(
            "INSERT INTO chat_handle_join (chat_id, handle_id) VALUES (1, ?)",
            (handle_id,)
        )

    # Add some group messages
    for i in range(20):
        date = base_date + (i * 86400000000000)
        for handle_id in [1, 2, 3]:
            cursor.execute(
                """INSERT INTO message 
                (ROWID, guid, text, handle_id, is_from_me, date, cache_has_attachments)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    message_id,
                    f"message-{message_id}",
                    "Group chat message",
                    handle_id,
                    1 if handle_id == 1 else 0,
                    date,
                    0
                )
            )

            cursor.execute(
                "INSERT INTO chat_message_join (chat_id, message_id) VALUES (1, ?)",
                (message_id,)
            )
            message_id += 1

    conn.commit()
    conn.close()

    return temp_db


def generate_message_text(sentiment, is_from_me):
    """Generate synthetic message text based on sentiment."""
    positive_messages = [
        "That sounds great!",
        "Looking forward to it",
        "Thanks so much!",
        "Had a wonderful time",
        "This is awesome"
    ]

    neutral_messages = [
        "Got it, thanks",
        "Sure, that works",
        "Let me know",
        "Sounds good",
        "Ok, see you then"
    ]

    mixed_messages = [
        "Not sure about that",
        "Could be better",
        "Let's discuss",
        "I'll think about it",
        "Maybe next time"
    ]

    if sentiment == "positive":
        return random.choice(positive_messages)
    elif sentiment == "neutral":
        return random.choice(neutral_messages)
    else:
        return random.choice(mixed_messages)


@pytest.fixture
def mock_consent_manager():
    """Mock consent manager that always grants consent."""
    class MockConsentManager:
        async def initialize(self):
            pass

        async def check_consent(self):
            return {
                "has_consent": True,
                "expires_at": (datetime.now() + timedelta(hours=24)).isoformat(),
                "remaining_hours": 24
            }

        async def request_consent(self, expiry_hours=24):
            return {
                "consent_granted": True,
                "expires_at": (datetime.now() + timedelta(hours=expiry_hours)).isoformat()
            }

        async def revoke_consent(self):
            return {"consent_revoked": True}

    return MockConsentManager()


@pytest.fixture
def mock_config():
    """Mock configuration object."""
    class MockConfig:
        db_path = "~/Library/Messages/chat.db"
        consent_db_path = "~/.imessage_consent.db"
        privacy_mode = "strict"
        hash_contacts = True
        redact_messages = True

    return MockConfig()
