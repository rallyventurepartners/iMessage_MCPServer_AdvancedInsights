-- SQL script to create indexes for better database query performance
-- These indexes will significantly improve the performance of common queries

-- Index for message dates (for temporal queries)
CREATE INDEX IF NOT EXISTS idx_message_date ON message (date);

-- Index for joining chats and messages
CREATE INDEX IF NOT EXISTS idx_chat_message_join ON chat_message_join (chat_id, message_id);

-- Index for message text (for text search)
CREATE INDEX IF NOT EXISTS idx_message_text ON message (text);

-- Index for handle identification
CREATE INDEX IF NOT EXISTS idx_handle_id ON handle (id);

-- Index for chat handle joins
CREATE INDEX IF NOT EXISTS idx_chat_handle_join ON chat_handle_join (chat_id, handle_id);

-- Index for attachments
CREATE INDEX IF NOT EXISTS idx_attachment_message_id ON attachment (message_id);

-- Index for chat identifiers
CREATE INDEX IF NOT EXISTS idx_chat_identifier ON chat (chat_identifier);

-- Index for message state
CREATE INDEX IF NOT EXISTS idx_message_state ON message (is_from_me, state); 