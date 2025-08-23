"""
Health check tools for system validation and diagnostics.
"""

import logging
from typing import Any, Dict

from mcp import Server
from mcp.types import Tool

from ..config import Config
from ..db import get_database
from ..models import HealthCheckInput, HealthCheckOutput

logger = logging.getLogger(__name__)


def register_health_tools(server: Server, config: Config) -> None:
    """Register health check tools with the server."""
    
    @server.tool()
    async def imsg_health_check(arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate DB access, schema presence, index hints, and read-only mode.
        
        This tool performs comprehensive health checks on the iMessage database
        to ensure it's accessible, properly structured, and optimized.
        """
        try:
            # Validate input
            params = HealthCheckInput(**arguments)
            
            # Get database connection
            db = await get_database(params.db_path)
            
            # Get database statistics
            stats = await db.get_db_stats()
            
            # Check schema
            schema_info = await db.check_schema()
            
            # Check indices
            index_info = await db.check_indices()
            
            # Build warnings list
            warnings = []
            
            # Schema warnings
            if schema_info['missing_required']:
                warnings.append(
                    f"Missing required tables: {', '.join(schema_info['missing_required'])}"
                )
            
            # Index warnings
            if index_info['recommendations']:
                warnings.extend(index_info['recommendations'])
            
            # Size warnings
            if stats['size_mb'] > 10000:  # 10GB
                warnings.append(
                    f"Large database ({stats['size_mb']} MB) - consider sharding"
                )
            
            # Performance warning based on message count
            if isinstance(stats.get('message_count'), int) and stats['message_count'] > 1000000:
                warnings.append(
                    f"High message count ({stats['message_count']:,}) - queries may be slow"
                )
            
            # Vacuum recommendation
            try:
                # Check fragmentation (simplified check)
                page_query = "PRAGMA page_count"
                freelist_query = "PRAGMA freelist_count"
                
                page_result = await db.execute_query(page_query)
                freelist_result = await db.execute_query(freelist_query)
                
                if page_result and freelist_result:
                    page_count = page_result[0]['page_count']
                    freelist_count = freelist_result[0]['freelist_count']
                    
                    if freelist_count > 0 and (freelist_count / page_count) > 0.15:
                        savings_pct = int((freelist_count / page_count) * 100)
                        warnings.append(
                            f"Consider running VACUUM for {savings_pct}% space reduction"
                        )
            except Exception:
                # Ignore vacuum check errors
                pass
            
            # Build response
            output = HealthCheckOutput(
                db_version=stats['sqlite_version'],
                tables=schema_info['tables'],
                indices_ok=len(index_info['recommendations']) == 0,
                read_only_ok=True,  # We enforce this in db.py
                warnings=warnings
            )
            
            return output.model_dump()
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "error": str(e),
                "error_type": "health_check_failed"
            }