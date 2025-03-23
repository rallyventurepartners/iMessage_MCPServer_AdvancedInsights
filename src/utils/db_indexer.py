#!/usr/bin/env python3
"""
iMessage Database Indexer

This utility creates custom indexes on the iMessage database to improve query performance.
It attempts to create a copy of the database with indexes if writing to the original is not possible.

Usage:
    python db_indexer.py [options]

Options:
    --db-path PATH     Path to the iMessage database (default: ~/Library/Messages/chat.db)
    --index-db PATH    Path to store the indexed database (default: ~/.imessage_insights/indexed_chat.db)
    --force            Force reindexing even if indexes already exist
    --no-backup        Skip creating a backup before indexing
    --read-only        Only use read-only mode (creates a copy if original isn't writable)
    --analyze-only     Only analyze the database and suggest indexes without creating them
    --fts-only         Only create or rebuild the full-text search index
"""

import os
import sys
import sqlite3
import shutil
import time
import logging
import argparse
from pathlib import Path
import traceback
import json
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Default paths
HOME = os.path.expanduser("~")
DEFAULT_DB_PATH = Path(f"{HOME}/Library/Messages/chat.db")
DEFAULT_INDEX_PATH = Path(f"{HOME}/.imessage_insights/indexed_chat.db")

class DatabaseIndexer:
    """Utility to create and manage indexes for the iMessage database."""
    
    def __init__(self, db_path=None, index_path=None, force=False, make_backup=True, analyze_only=False, fts_only=False):
        """Initialize the indexer with paths and options.
        
        Args:
            db_path: Path to the original iMessage database
            index_path: Path to store the indexed copy (if needed)
            force: Force reindexing even if indexes exist
            make_backup: Whether to create a backup before indexing
            analyze_only: Only analyze the database and suggest indexes
            fts_only: Only create or rebuild the full-text search index
        """
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        self.index_path = Path(index_path or DEFAULT_INDEX_PATH)
        self.force = force
        self.make_backup = make_backup
        self.read_only_mode = False
        self.analyze_only = analyze_only
        self.fts_only = fts_only
        self.total_indexes_created = 0
        self.start_time = 0
        self.slow_queries = []
        self.index_suggestions = []
        
    def check_database(self):
        """Check if the database file exists and is accessible."""
        if not self.db_path.exists():
            logger.error(f"Database file not found: {self.db_path}")
            return False
            
        # Check if we have write access
        if os.access(self.db_path, os.W_OK):
            logger.info(f"Database file is writable: {self.db_path}")
            return True
        else:
            logger.warning(f"Database file is not writable: {self.db_path}")
            logger.warning("Will create a separate indexed copy instead")
            self.read_only_mode = True
            return True
            
    def create_backup(self):
        """Create a backup of the database before indexing."""
        if not self.make_backup:
            logger.info("Backup creation skipped as requested")
            return True
            
        backup_path = self.db_path.with_suffix(".db.backup")
        try:
            logger.info(f"Creating backup at {backup_path}")
            shutil.copy2(self.db_path, backup_path)
            logger.info("Backup created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False
            
    def get_existing_indexes(self, conn):
        """Get a list of existing indexes in the database."""
        cursor = conn.cursor()
        cursor.execute("SELECT name, tbl_name, sql FROM sqlite_master WHERE type='index'")
        indexes = []
        for row in cursor.fetchall():
            indexes.append({
                'name': row[0],
                'table': row[1],
                'sql': row[2]
            })
        return indexes
        
    def analyze_database(self, conn):
        """Analyze the database and suggest indexes for slow queries."""
        cursor = conn.cursor()
        logger.info("Analyzing database for optimization opportunities...")
        
        # Collect table statistics
        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            table_stats = {}
            for table in tables:
                try:
                    cursor.execute(f"SELECT count(*) FROM {table}")
                    row_count = cursor.fetchone()[0]
                    
                    # Get column info
                    cursor.execute(f"PRAGMA table_info({table})")
                    columns = [{'name': col[1], 'type': col[2]} for col in cursor.fetchall()]
                    
                    table_stats[table] = {
                        'row_count': row_count,
                        'columns': columns
                    }
                    
                    logger.info(f"Table {table}: {row_count} rows")
                except Exception as e:
                    logger.warning(f"Error analyzing table {table}: {e}")
        except Exception as e:
            logger.error(f"Error collecting table statistics: {e}")
        
        # Analyze common queries with EXPLAIN QUERY PLAN
        common_queries = [
            # Message retrieval queries
            ("SELECT message.ROWID, message.date, message.text FROM message " 
             "JOIN chat_message_join ON message.ROWID = chat_message_join.message_id " 
             "WHERE chat_message_join.chat_id = 1 ORDER BY message.date DESC LIMIT 100",
             "Recent messages in a chat"),
            
            ("SELECT message.ROWID, message.date, message.text FROM message " 
             "JOIN chat_message_join ON message.ROWID = chat_message_join.message_id " 
             "JOIN handle ON message.handle_id = handle.ROWID " 
             "WHERE handle.id = '+1234567890' ORDER BY message.date DESC LIMIT 100",
             "Messages from a specific contact"),
             
            # Search queries
            ("SELECT message.ROWID, message.date, message.text FROM message " 
             "WHERE message.text LIKE '%important%' ORDER BY message.date DESC LIMIT 100",
             "Text search in messages"),
             
            # Stats and analytics queries
            ("SELECT handle.id, COUNT(message.ROWID) as msg_count FROM message " 
             "JOIN handle ON message.handle_id = handle.ROWID " 
             "WHERE message.is_from_me = 0 GROUP BY handle.id ORDER BY msg_count DESC",
             "Message count per contact"),
             
            # Date range queries
            ("SELECT message.ROWID, message.date, message.text FROM message " 
             "WHERE message.date BETWEEN 600000000000000 AND 700000000000000 " 
             "ORDER BY message.date DESC LIMIT 100",
             "Messages in date range")
        ]
        
        # Analyze each query
        for query, description in common_queries:
            try:
                logger.info(f"Analyzing query: {description}")
                
                # Measure execution time
                start_time = time.time()
                cursor.execute(f"EXPLAIN QUERY PLAN {query}")
                plan = cursor.fetchall()
                
                # Try to execute query with a row limit to get actual timing
                try:
                    cursor.execute(query)
                    # Just fetch a few rows to avoid memory issues
                    rows = cursor.fetchmany(10)
                except Exception as e:
                    logger.warning(f"Error executing query: {e}")
                    continue
                    
                execution_time = time.time() - start_time
                
                # Check if query is slow (>50ms for an explanation is a red flag)
                is_slow = execution_time > 0.05
                
                # Parse the plan
                plan_text = '\n'.join([str(row) for row in plan])
                uses_index = "USING INDEX" in plan_text
                scan_tables = []
                
                for row in plan:
                    if "SCAN TABLE" in row[-1]:
                        table_name = row[-1].split("SCAN TABLE")[1].strip().split(" ")[0]
                        scan_tables.append(table_name)
                
                # Record query information
                query_info = {
                    'description': description,
                    'execution_time': execution_time,
                    'is_slow': is_slow,
                    'plan': plan_text,
                    'uses_index': uses_index,
                    'scan_tables': scan_tables
                }
                
                if is_slow:
                    self.slow_queries.append(query_info)
                    
                    # Suggest indexes
                    if not uses_index:
                        for table in scan_tables:
                            # Check what columns are likely being filtered or joined
                            if "chat_message_join" in query and table == "chat_message_join":
                                self.index_suggestions.append({
                                    'table': 'chat_message_join',
                                    'columns': ['chat_id', 'message_id'],
                                    'reason': f"Improve performance for query: {description}"
                                })
                            elif "handle.id" in query and table == "handle":
                                self.index_suggestions.append({
                                    'table': 'handle',
                                    'columns': ['id'],
                                    'reason': f"Improve performance for query: {description}"
                                })
                            elif "message.date" in query and table == "message":
                                self.index_suggestions.append({
                                    'table': 'message',
                                    'columns': ['date'],
                                    'reason': f"Improve performance for query: {description}"
                                })
                            elif "message.text LIKE" in query and table == "message":
                                self.index_suggestions.append({
                                    'table': 'message',
                                    'columns': ['text'],
                                    'reason': f"Consider FTS for query: {description}"
                                })
                            elif "message.is_from_me" in query and table == "message":
                                self.index_suggestions.append({
                                    'table': 'message',
                                    'columns': ['is_from_me'],
                                    'reason': f"Improve performance for query: {description}"
                                })
                logger.info(f"Query execution time: {execution_time:.4f}s")
                
            except Exception as e:
                logger.error(f"Error analyzing query: {e}")
        
        # Check if FTS is needed based on text searches
        has_fts_queries = any("LIKE" in query for query, _ in common_queries)
        if has_fts_queries:
            # Check if FTS exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='message_fts'")
            has_fts = cursor.fetchone() is not None
            
            if not has_fts:
                self.index_suggestions.append({
                    'type': 'fts',
                    'table': 'message',
                    'columns': ['text'],
                    'reason': "Create FTS5 virtual table for full-text search"
                })
        
        # Get indexing recommendations
        if self.index_suggestions:
            logger.info(f"Found {len(self.index_suggestions)} indexing recommendations:")
            for i, suggestion in enumerate(self.index_suggestions):
                if 'type' in suggestion and suggestion['type'] == 'fts':
                    logger.info(f"  {i+1}. Create FTS virtual table on {suggestion['table']}.{','.join(suggestion['columns'])}")
                else:
                    logger.info(f"  {i+1}. Create index on {suggestion['table']}.{','.join(suggestion['columns'])}")
                logger.info(f"     Reason: {suggestion['reason']}")
        
        # Return analysis results
        return {
            'table_stats': table_stats,
            'slow_queries': self.slow_queries,
            'index_suggestions': self.index_suggestions
        }
    
    def create_materialized_views(self, conn):
        """Create materialized views for common analytics queries."""
        logger.info("Creating materialized views for common analytics...")
        cursor = conn.cursor()
        
        # Define materialized views to create
        mat_views = [
            # Contact message count view
            ("""
            CREATE TABLE IF NOT EXISTS mv_contact_message_counts AS
            SELECT 
                handle.id as contact_id, 
                COUNT(DISTINCT message.ROWID) as total_messages,
                SUM(CASE WHEN message.is_from_me = 1 THEN 1 ELSE 0 END) as sent_messages,
                SUM(CASE WHEN message.is_from_me = 0 THEN 1 ELSE 0 END) as received_messages,
                MIN(message.date) as first_message_date,
                MAX(message.date) as last_message_date
            FROM 
                message 
            JOIN 
                handle ON message.handle_id = handle.ROWID
            GROUP BY 
                handle.id
            """, "Contact message counts"),
            
            # Chat activity view
            ("""
            CREATE TABLE IF NOT EXISTS mv_chat_activity AS
            SELECT 
                cmj.chat_id,
                c.display_name as chat_name,
                COUNT(DISTINCT m.ROWID) as message_count,
                MIN(m.date) as first_message_date,
                MAX(m.date) as last_message_date,
                COUNT(DISTINCT m.handle_id) as participant_count
            FROM 
                message m
            JOIN 
                chat_message_join cmj ON m.ROWID = cmj.message_id
            JOIN
                chat c ON cmj.chat_id = c.ROWID
            GROUP BY 
                cmj.chat_id
            """, "Chat activity summary")
        ]
        
        # Create each materialized view
        for view_sql, description in mat_views:
            try:
                logger.info(f"Creating materialized view: {description}")
                cursor.execute(view_sql)
                logger.info(f"Created materialized view for {description}")
            except Exception as e:
                logger.error(f"Error creating materialized view {description}: {e}")
        
        # Create indexes on materialized views
        view_indexes = [
            ("CREATE INDEX IF NOT EXISTS idx_mv_contact_message_counts_contact ON mv_contact_message_counts(contact_id)", 
             "Contact ID index for message counts view"),
            ("CREATE INDEX IF NOT EXISTS idx_mv_contact_message_counts_total ON mv_contact_message_counts(total_messages DESC)", 
             "Total messages index for sorting"),
            ("CREATE INDEX IF NOT EXISTS idx_mv_chat_activity_chat_id ON mv_chat_activity(chat_id)", 
             "Chat ID index for activity view"),
            ("CREATE INDEX IF NOT EXISTS idx_mv_chat_activity_message_count ON mv_chat_activity(message_count DESC)", 
             "Message count index for sorting")
        ]
        
        # Create each index on the views
        for index_sql, description in view_indexes:
            try:
                logger.info(f"Creating index: {description}")
                cursor.execute(index_sql)
                logger.info(f"Created index {description}")
            except Exception as e:
                logger.error(f"Error creating index on materialized view: {e}")
        
        # Commit the changes
        conn.commit()
        return True
        
    def create_indexes(self):
        """Create indexes on the database to improve query performance."""
        self.start_time = time.time()
        logger.info(f"Starting database indexing for {self.db_path}")
        
        # Check if database exists and is accessible
        if not self.check_database():
            return False
            
        # Determine if we need to work with a copy
        target_path = self.db_path
        if self.read_only_mode:
            # Ensure the directory exists
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy the database if it doesn't exist or force is true
            if not self.index_path.exists() or self.force:
                try:
                    logger.info(f"Creating indexed copy at {self.index_path}")
                    shutil.copy2(self.db_path, self.index_path)
                except Exception as e:
                    logger.error(f"Failed to create indexed copy: {e}")
                    return False
            
            target_path = self.index_path
        else:
            # Create a backup before modifying the original
            if not self.create_backup():
                return False
                
        # Connect to the target database
        try:
            conn = sqlite3.connect(target_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            logger.info(f"Connected to database at {target_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
            
        # Get existing indexes
        existing_indexes = self.get_existing_indexes(conn)
        logger.info(f"Found {len(existing_indexes)} existing indexes")
        existing_index_names = [idx['name'] for idx in existing_indexes]
        
        # If analyze_only, just analyze the database and exit
        if self.analyze_only:
            logger.info("Running in analyze-only mode")
            analysis_results = self.analyze_database(conn)
            
            # Save analysis results to file
            results_path = self.index_path.parent / "index_analysis.json"
            try:
                results_path.parent.mkdir(parents=True, exist_ok=True)
                with open(results_path, 'w') as f:
                    # Convert datetime objects to strings for JSON serialization
                    json.dump(analysis_results, f, indent=2, default=str)
                logger.info(f"Analysis results saved to {results_path}")
            except Exception as e:
                logger.error(f"Failed to save analysis results: {e}")
            
            conn.close()
            return True
        
        # Skip if indexes already exist and force is not set
        if len(existing_indexes) > 10 and not self.force and not self.fts_only:
            logger.info("Database appears to be already indexed. Use --force to reindex.")
            
            # Still run analysis to suggest any missing indexes
            analysis_results = self.analyze_database(conn)
            conn.close()
            return True
            
        # Define standard indexes to create for improved query performance
        standard_indexes = [
            # Message indexes
            ("CREATE INDEX IF NOT EXISTS idx_message_date ON message(date)", 
             "Message date index for time-based queries"),
            ("CREATE INDEX IF NOT EXISTS idx_message_handle_id ON message(handle_id)", 
             "Message handle index for contact-based queries"),
            ("CREATE INDEX IF NOT EXISTS idx_message_text ON message(text COLLATE NOCASE)", 
             "Message text index for basic text search"),
            ("CREATE INDEX IF NOT EXISTS idx_message_is_from_me ON message(is_from_me)", 
             "Message direction index for sent/received filtering"),
            ("CREATE INDEX IF NOT EXISTS idx_message_date_handle ON message(date, handle_id)", 
             "Combined date-handle index for filtered time queries"),
            
            # Chat indexes
            ("CREATE INDEX IF NOT EXISTS idx_chat_style ON chat(style)", 
             "Chat style index for group/individual filtering"),
            ("CREATE INDEX IF NOT EXISTS idx_chat_display_name ON chat(display_name COLLATE NOCASE)", 
             "Chat name index for name-based searches"),
            
            # Join table indexes
            ("CREATE INDEX IF NOT EXISTS idx_chat_message_join_chat_id ON chat_message_join(chat_id)", 
             "Chat-message join index by chat"),
            ("CREATE INDEX IF NOT EXISTS idx_chat_message_join_message_id ON chat_message_join(message_id)", 
             "Chat-message join index by message"),
            ("CREATE INDEX IF NOT EXISTS idx_chat_message_join_combined ON chat_message_join(chat_id, message_id)", 
             "Combined chat-message join index"),
            ("CREATE INDEX IF NOT EXISTS idx_chat_handle_join_chat_id ON chat_handle_join(chat_id)", 
             "Chat-handle join index by chat"),
            ("CREATE INDEX IF NOT EXISTS idx_chat_handle_join_handle_id ON chat_handle_join(handle_id)", 
             "Chat-handle join index by handle"),
            
            # Handle indexes
            ("CREATE INDEX IF NOT EXISTS idx_handle_id ON handle(id COLLATE NOCASE)", 
             "Handle ID index for phone/email searches"),
            ("CREATE INDEX IF NOT EXISTS idx_handle_service ON handle(service)", 
             "Handle service index for filtering by service type")
        ]
        
        # Define advanced composite indexes for better performance
        advanced_indexes = [
            # Advanced indexes for common query patterns
            ("CREATE INDEX IF NOT EXISTS idx_message_date_is_from_me ON message(date, is_from_me)", 
             "Combined date-direction index for filtering sent/received by time"),
            ("CREATE INDEX IF NOT EXISTS idx_message_handle_is_from_me ON message(handle_id, is_from_me)", 
             "Combined handle-direction index for filtering by contact and direction"),
            ("CREATE INDEX IF NOT EXISTS idx_message_date_handle_is_from_me ON message(date, handle_id, is_from_me)",
             "Triple index for time-based contact direction queries"),
            
            # Partial indexes for more efficient filtering
            ("CREATE INDEX IF NOT EXISTS idx_message_sent ON message(date, handle_id) WHERE is_from_me = 1",
             "Partial index for sent messages"),
            ("CREATE INDEX IF NOT EXISTS idx_message_received ON message(date, handle_id) WHERE is_from_me = 0",
             "Partial index for received messages"),
            
            # Additional chat indexes
            ("CREATE INDEX IF NOT EXISTS idx_chat_style_display_name ON chat(style, display_name COLLATE NOCASE)",
             "Combined style-name index for filtered chat searches"),
            
            # Additional handle indexes
            ("CREATE INDEX IF NOT EXISTS idx_handle_id_service ON handle(id COLLATE NOCASE, service)",
             "Combined id-service index for filtered contact searches")
        ]
        
        # Combine all indexes
        indexes = standard_indexes + advanced_indexes
        
        # Try to create FTS virtual table for full-text search
        fts_creation = [
            ("CREATE VIRTUAL TABLE IF NOT EXISTS message_fts USING fts5(text, content='message', content_rowid='ROWID')",
             "FTS5 virtual table for full-text search"),
            ("INSERT OR IGNORE INTO message_fts(rowid, text) SELECT ROWID, text FROM message WHERE text IS NOT NULL",
             "Populate FTS5 table with message text")
        ]
        
        # Transaction for better performance
        try:
            cursor.execute("BEGIN TRANSACTION")
            
            if not self.fts_only:
                # Create standard and advanced indexes
                for idx, (index_sql, description) in enumerate(indexes):
                    try:
                        # Skip if index already exists with same name
                        index_name = index_sql.split("CREATE INDEX IF NOT EXISTS ")[1].split(" ON ")[0]
                        if index_name in existing_index_names and not self.force:
                            logger.info(f"Skipping existing index: {index_name}")
                            continue
                            
                        logger.info(f"Creating index {idx+1}/{len(indexes)}: {description}")
                        cursor.execute(index_sql)
                        self.total_indexes_created += 1
                    except Exception as e:
                        logger.error(f"Error creating index: {e}")
                        logger.error(traceback.format_exc())
            
            # Attempt to create FTS virtual table
            try:
                for idx, (fts_sql, description) in enumerate(fts_creation):
                    logger.info(f"Setting up FTS: {description}")
                    cursor.execute(fts_sql)
                logger.info("FTS setup completed successfully")
            except Exception as e:
                logger.warning(f"FTS setup failed (not critical): {e}")
                logger.warning(traceback.format_exc())
                
            # Create materialized views if not in FTS-only mode
            if not self.fts_only:
                try:
                    self.create_materialized_views(conn)
                except Exception as e:
                    logger.warning(f"Error creating materialized views (not critical): {e}")
                    logger.warning(traceback.format_exc())
                
            # Commit transaction
            conn.commit()
            logger.info(f"Created {self.total_indexes_created} indexes successfully")
            
            # Analyze the database to update statistics
            try:
                logger.info("Running ANALYZE to update statistics...")
                cursor.execute("ANALYZE")
                conn.commit()
            except Exception as e:
                logger.warning(f"Error running ANALYZE (not critical): {e}")
                
            # Try to optimize storage
            try:
                logger.info("Running VACUUM to optimize storage...")
                cursor.execute("VACUUM")
                conn.commit()
            except Exception as e:
                logger.warning(f"Error running VACUUM (not critical): {e}")
                
        except Exception as e:
            logger.error(f"Failed during indexing: {e}")
            logger.error(traceback.format_exc())
            conn.rollback()
            conn.close()
            return False
            
        # Close connection
        conn.close()
        
        # Log completion time
        elapsed = time.time() - self.start_time
        logger.info(f"Indexing completed in {elapsed:.2f} seconds")
        
        # Return indexed database path
        if self.read_only_mode:
            logger.info(f"Indexed database created at: {self.index_path}")
            logger.info(f"To use this indexed database, you can specify:")
            logger.info(f"  --db-path={self.index_path}")
        else:
            logger.info(f"Original database indexed at: {self.db_path}")
            
        return True

def main():
    """Main entry point for the indexer utility."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="iMessage Database Indexer")
    parser.add_argument("--db-path", type=str, help=f"Path to the iMessage database (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--index-db", type=str, help=f"Path to store the indexed database (default: {DEFAULT_INDEX_PATH})")
    parser.add_argument("--force", action="store_true", help="Force reindexing even if indexes already exist")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating a backup before indexing")
    parser.add_argument("--read-only", action="store_true", help="Only use read-only mode (creates a copy)")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze the database and suggest indexes")
    parser.add_argument("--fts-only", action="store_true", help="Only create or rebuild the full-text search index")
    
    args = parser.parse_args()
    
    # Create the indexer with the provided options
    indexer = DatabaseIndexer(
        db_path=args.db_path,
        index_path=args.index_db,
        force=args.force,
        make_backup=not args.no_backup,
        analyze_only=args.analyze_only,
        fts_only=args.fts_only
    )
    
    # Force read-only mode if requested
    if args.read_only:
        indexer.read_only_mode = True
    
    # Run the indexer
    if indexer.create_indexes():
        logger.info("Database indexing completed successfully")
        return 0
    else:
        logger.error("Database indexing failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())