"""
Predictive analytics and recommendation tools.
"""

import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List

from mcp import Server

from ..config import Config
from ..db import get_database
from ..models import (
    BestContactTimeInput, BestContactTimeOutput,
    AnomalyScanInput, AnomalyScanOutput
)
from ..privacy import hash_contact_id

logger = logging.getLogger(__name__)


def register_prediction_tools(server: Server, config: Config) -> None:
    """Register prediction and recommendation tools with the server."""
    
    @server.tool()
    async def imsg_best_contact_time(arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recommend optimal contact windows based on historic responsiveness.
        
        Analyzes when messages receive the quickest responses to suggest
        the best times to contact someone.
        """
        try:
            # Validate input
            params = BestContactTimeInput(**arguments)
            
            # Get database connection
            db = await get_database(params.db_path)
            
            # Build query for response patterns
            base_query = """
            WITH response_data AS (
                SELECT 
                    strftime('%w', datetime(m1.date/1000000000 + 978307200, 'unixepoch')) as weekday,
                    strftime('%H', datetime(m1.date/1000000000 + 978307200, 'unixepoch')) as hour,
                    MIN((m2.date - m1.date) / 1000000000.0) as response_time_s
                FROM message m1
                JOIN message m2 ON m2.handle_id = m1.handle_id
                WHERE m1.is_from_me = 1 
                AND m2.is_from_me = 0
                AND m2.date > m1.date
                AND m2.date < m1.date + 7200000000000  -- Within 2 hours
            """
            
            if params.contact_id:
                # TODO: Add contact filtering
                pass
            
            query = base_query + """
                GROUP BY weekday, hour
            )
            SELECT 
                weekday,
                hour,
                AVG(response_time_s) as avg_response_time,
                COUNT(*) as sample_count
            FROM response_data
            GROUP BY weekday, hour
            HAVING sample_count >= 3
            ORDER BY avg_response_time ASC
            LIMIT 20
            """
            
            results = await db.execute_query(query)
            
            # Process results into windows
            windows = []
            weekday_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
            
            for row in results[:10]:  # Top 10 windows
                weekday_idx = int(row['weekday'])
                hour = int(row['hour'])
                avg_response = row['avg_response_time']
                
                # Calculate score (inverse of response time, normalized)
                # Lower response time = higher score
                max_response = 3600  # 1 hour
                score = max(0, 1 - (avg_response / max_response))
                
                windows.append({
                    "weekday": weekday_names[weekday_idx],
                    "hour": hour,
                    "score": round(score, 2)
                })
            
            # If no data, provide default recommendations
            if not windows:
                # Default to common communication times
                default_windows = [
                    ("Tuesday", 19, 0.8),
                    ("Wednesday", 20, 0.75),
                    ("Thursday", 19, 0.7),
                    ("Saturday", 11, 0.65),
                    ("Sunday", 14, 0.6)
                ]
                
                windows = [
                    {
                        "weekday": w[0],
                        "hour": w[1],
                        "score": w[2]
                    }
                    for w in default_windows
                ]
            
            # Build response
            output = BestContactTimeOutput(windows=windows)
            return output.model_dump()
            
        except Exception as e:
            logger.error(f"Best contact time failed: {e}")
            return {
                "error": str(e),
                "error_type": "prediction_failed"
            }
    
    @server.tool()
    async def imsg_anomaly_scan(arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect unusual silences or behavior changes relative to baselines.
        
        Identifies anomalies in communication patterns that might indicate
        relationship changes or important life events.
        """
        try:
            # Validate input
            params = AnomalyScanInput(**arguments)
            
            # Get database connection
            db = await get_database(params.db_path)
            
            # Calculate date ranges
            end_date = datetime.now()
            start_date = end_date - timedelta(days=params.lookback_days)
            baseline_start = start_date - timedelta(days=params.lookback_days)
            
            # Convert to iMessage timestamps
            start_ts = int((start_date.timestamp() - 978307200) * 1e9)
            end_ts = int((end_date.timestamp() - 978307200) * 1e9)
            baseline_start_ts = int((baseline_start.timestamp() - 978307200) * 1e9)
            
            # Build query for daily message counts
            base_query = """
            SELECT 
                DATE(date/1000000000 + 978307200, 'unixepoch') as day,
                COUNT(*) as message_count,
                SUM(CASE WHEN is_from_me = 1 THEN 1 ELSE 0 END) as sent_count,
                SUM(CASE WHEN is_from_me = 0 THEN 1 ELSE 0 END) as received_count
            FROM message
            WHERE date >= ? AND date <= ?
            """
            
            if params.contact_id:
                # TODO: Add contact filtering
                pass
            
            query = base_query + " GROUP BY day ORDER BY day"
            
            # Get recent period data
            recent_results = await db.execute_query(query, (start_ts, end_ts))
            
            # Get baseline period data
            baseline_results = await db.execute_query(query, (baseline_start_ts, start_ts))
            
            # Calculate baseline statistics
            if baseline_results:
                baseline_counts = [r['message_count'] for r in baseline_results]
                baseline_mean = sum(baseline_counts) / len(baseline_counts)
                baseline_std = (
                    sum((x - baseline_mean) ** 2 for x in baseline_counts) / len(baseline_counts)
                ) ** 0.5
            else:
                baseline_mean = 10
                baseline_std = 5
            
            # Detect anomalies
            anomalies = []
            
            # Check for silence periods (no messages for multiple days)
            if recent_results:
                last_date = None
                for row in recent_results:
                    current_date = datetime.strptime(row['day'], '%Y-%m-%d')
                    
                    if last_date:
                        gap_days = (current_date - last_date).days
                        if gap_days > 3:  # More than 3 days of silence
                            severity = min(gap_days / 7, 1.0)  # Max severity at 1 week
                            anomalies.append({
                                "ts": last_date.isoformat(),
                                "type": "silence",
                                "severity": round(severity, 2),
                                "note": f"{gap_days} days of silence detected"
                            })
                    
                    last_date = current_date
                    
                    # Check for volume anomalies
                    count = row['message_count']
                    if baseline_std > 0:
                        z_score = abs(count - baseline_mean) / baseline_std
                        if z_score > 2:  # More than 2 standard deviations
                            anomaly_type = "burst" if count > baseline_mean else "drop"
                            severity = min(z_score / 4, 1.0)  # Max severity at 4 std devs
                            anomalies.append({
                                "ts": current_date.isoformat(),
                                "type": anomaly_type,
                                "severity": round(severity, 2),
                                "note": f"Message volume {anomaly_type}: {count} msgs (baseline: {int(baseline_mean)})"
                            })
            
            # Check for pattern changes (simplified)
            if len(recent_results) > 7 and len(baseline_results) > 7:
                # Compare sent/received ratios
                recent_sent_ratio = sum(r['sent_count'] for r in recent_results[-7:]) / max(
                    sum(r['message_count'] for r in recent_results[-7:]), 1
                )
                baseline_sent_ratio = sum(r['sent_count'] for r in baseline_results) / max(
                    sum(r['message_count'] for r in baseline_results), 1
                )
                
                ratio_change = abs(recent_sent_ratio - baseline_sent_ratio)
                if ratio_change > 0.3:  # 30% change in pattern
                    anomalies.append({
                        "ts": datetime.now().isoformat(),
                        "type": "pattern_change",
                        "severity": min(ratio_change * 2, 1.0),
                        "note": f"Communication pattern shifted: {'more sending' if recent_sent_ratio > baseline_sent_ratio else 'more receiving'}"
                    })
            
            # Sort by timestamp
            anomalies.sort(key=lambda x: x['ts'], reverse=True)
            
            # Limit results
            anomalies = anomalies[:20]
            
            # Build response
            output = AnomalyScanOutput(anomalies=anomalies)
            return output.model_dump()
            
        except Exception as e:
            logger.error(f"Anomaly scan failed: {e}")
            return {
                "error": str(e),
                "error_type": "scan_failed"
            }