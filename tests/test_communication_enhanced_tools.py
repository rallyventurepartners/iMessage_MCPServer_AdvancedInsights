"""
Test suite for enhanced communication analysis tools.
"""

import asyncio
import base64
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from imessage_mcp_server.tools.communication_enhanced import (
    enhanced_cadence_calendar_tool,
    enhanced_relationship_intelligence_tool,
    _parse_extended_time_period,
    _auto_select_granularity,
    _calculate_advanced_statistics,
)


# Fixtures
@pytest.fixture
def mock_db():
    """Create a mock database connection."""
    db = Mock()
    db.execute = AsyncMock()
    return db


@pytest.fixture
def sample_time_series_data():
    """Create sample time series data for testing."""
    base_date = datetime(2023, 1, 1)
    data = []
    
    # Generate 365 days of data with patterns
    for day in range(365):
        date = base_date + timedelta(days=day)
        
        # Weekly pattern (more messages on weekdays)
        if date.weekday() < 5:
            base_count = 10
        else:
            base_count = 5
        
        # Add some randomness and trend
        count = base_count + np.random.randint(-3, 4) + (day // 30)  # Slight upward trend
        
        # Create hourly distribution
        for hour in range(24):
            if 9 <= hour <= 21:  # Active hours
                hourly_count = np.random.poisson(count / 10)
                if hourly_count > 0:
                    data.append((
                        date.strftime("%Y-%m-%d"),
                        hourly_count,
                        hourly_count // 2,  # sent
                        hourly_count - hourly_count // 2,  # received
                        str(date.weekday()),
                        str(hour)
                    ))
    
    return data


class TestEnhancedCadenceCalendar:
    """Test enhanced cadence calendar tool."""
    
    @pytest.mark.asyncio
    async def test_enhanced_cadence_extended_period(self, mock_db, sample_time_series_data):
        """Test with extended time periods."""
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=sample_time_series_data)
        mock_db.execute.return_value = mock_cursor
        
        with patch("imessage_mcp_server.tools.communication_enhanced.get_database", 
                   AsyncMock(return_value=mock_db)):
            result = await enhanced_cadence_calendar_tool(
                db_path="test.db",
                contact_id="test@example.com",
                time_period="1y",
                granularity="monthly",
                include_visualizations=True,
                include_time_series=True
            )
        
        assert result["days_analyzed"] == 365
        assert result["granularity"] == "monthly"
        assert "time_series" in result
        assert "visualizations" in result
        assert "statistics" in result
        
        # Check visualizations are base64 encoded
        for chart_name, chart_data in result["visualizations"].items():
            assert chart_data.startswith("data:image/png;base64,")
    
    @pytest.mark.asyncio
    async def test_enhanced_cadence_auto_granularity(self, mock_db):
        """Test automatic granularity selection."""
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=[])
        mock_db.execute.return_value = mock_cursor
        
        test_cases = [
            ("7d", "hourly"),
            ("30d", "daily"),
            ("6m", "weekly"),
            ("2y", "monthly"),
        ]
        
        for time_period, expected_granularity in test_cases:
            with patch("imessage_mcp_server.tools.communication_enhanced.get_database", 
                       AsyncMock(return_value=mock_db)):
                result = await enhanced_cadence_calendar_tool(
                    db_path="test.db",
                    time_period=time_period,
                    granularity="auto"
                )
            
            assert result["granularity"] == expected_granularity
    
    @pytest.mark.asyncio
    async def test_enhanced_cadence_comparison(self, mock_db, sample_time_series_data):
        """Test multi-contact comparison."""
        mock_cursor = AsyncMock()
        # Return different data for different contacts
        mock_cursor.fetchall = AsyncMock(side_effect=[
            sample_time_series_data,  # Primary contact
            sample_time_series_data[:100],  # Comparison 1 (less data)
            sample_time_series_data[:200],  # Comparison 2
        ])
        mock_db.execute.return_value = mock_cursor
        
        with patch("imessage_mcp_server.tools.communication_enhanced.get_database", 
                   AsyncMock(return_value=mock_db)):
            result = await enhanced_cadence_calendar_tool(
                db_path="test.db",
                contact_id="primary@example.com",
                time_period="90d",
                comparison_contacts=["comp1@example.com", "comp2@example.com"],
                include_time_series=True
            )
        
        assert "time_series" in result
        assert "primary" in result["time_series"]
        assert "comparisons" in result["time_series"]
        assert len(result["time_series"]["comparisons"]) == 2
    
    @pytest.mark.asyncio
    async def test_enhanced_cadence_no_visualizations(self, mock_db):
        """Test without visualizations for performance."""
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=[
            ("2023-01-01", 10, 5, 5, "1", "10"),
            ("2023-01-02", 15, 8, 7, "2", "11"),
        ])
        mock_db.execute.return_value = mock_cursor
        
        with patch("imessage_mcp_server.tools.communication_enhanced.get_database", 
                   AsyncMock(return_value=mock_db)):
            result = await enhanced_cadence_calendar_tool(
                db_path="test.db",
                include_visualizations=False,
                include_time_series=True
            )
        
        assert "visualizations" not in result
        assert "time_series" in result


class TestEnhancedRelationshipIntelligence:
    """Test enhanced relationship intelligence tool."""
    
    @pytest.mark.asyncio
    async def test_enhanced_relationship_with_time_series(self, mock_db):
        """Test relationship analysis with time series data."""
        # Mock the base relationship tool
        base_result = {
            "contact_id": "test@example.com",
            "messages_total": 500,
            "engagement_score": 0.85,
        }
        
        # Mock time series data
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=[
            ("2023-01-01", 10, 5, 5, 50.5, 1),
            ("2023-01-02", 12, 6, 6, 55.2, 1),
            ("2023-01-03", 8, 4, 4, 48.0, 1),
        ])
        mock_db.execute.return_value = mock_cursor
        
        with patch("imessage_mcp_server.tools.relationship.relationship_intelligence_tool",
                   AsyncMock(return_value=base_result)):
            with patch("imessage_mcp_server.tools.communication_enhanced.get_database", 
                       AsyncMock(return_value=mock_db)):
                result = await enhanced_relationship_intelligence_tool(
                    db_path="test.db",
                    contact_id="test@example.com",
                    window_days=180,
                    include_time_series=True,
                    include_visualizations=True
                )
        
        assert "time_series" in result
        assert "visualizations" in result
        assert result["messages_total"] == 500  # Base data preserved
        
        # Check visualizations
        assert "dashboard" in result["visualizations"]
        assert "engagement_score" in result["visualizations"]
    
    @pytest.mark.asyncio
    async def test_enhanced_relationship_extended_window(self, mock_db):
        """Test with extended window (3 years)."""
        base_result = {"contact_id": "test", "messages_total": 5000}
        
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=[])
        mock_db.execute.return_value = mock_cursor
        
        with patch("imessage_mcp_server.tools.relationship.relationship_intelligence_tool",
                   AsyncMock(return_value=base_result)):
            with patch("imessage_mcp_server.tools.communication_enhanced.get_database", 
                       AsyncMock(return_value=mock_db)):
                result = await enhanced_relationship_intelligence_tool(
                    db_path="test.db",
                    contact_id="test@example.com",
                    window_days=1095,  # 3 years
                    time_series_granularity="monthly"
                )
        
        # Should handle 3-year window
        assert result["messages_total"] == 5000


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_parse_extended_time_period(self):
        """Test time period parsing."""
        assert _parse_extended_time_period("7d") == 7
        assert _parse_extended_time_period("2w") == 14
        assert _parse_extended_time_period("3m") == 90
        assert _parse_extended_time_period("1y") == 365
        assert _parse_extended_time_period("invalid") == 90  # Default
    
    def test_auto_select_granularity(self):
        """Test granularity auto-selection."""
        assert _auto_select_granularity(5) == "hourly"
        assert _auto_select_granularity(30) == "daily"
        assert _auto_select_granularity(180) == "weekly"
        assert _auto_select_granularity(500) == "monthly"
    
    def test_calculate_advanced_statistics(self):
        """Test advanced statistics calculation."""
        primary_data = {
            "time_series": {
                "2023-01": {"total": 100, "sent": 50, "received": 50},
                "2023-02": {"total": 120, "sent": 60, "received": 60},
                "2023-03": {"total": 110, "sent": 55, "received": 55},
                "2023-04": {"total": 130, "sent": 65, "received": 65},
            },
            "total_messages": 460
        }
        
        stats = _calculate_advanced_statistics(primary_data, {})
        
        assert "mean" in stats
        assert "median" in stats
        assert "std_dev" in stats
        assert "cv" in stats
        assert "trend_direction" in stats
        
        # Check trend detection
        assert stats["trend_direction"] in ["increasing", "decreasing", "stable"]


class TestVisualizationGeneration:
    """Test visualization generation."""
    
    @pytest.mark.asyncio
    async def test_visualization_output_format(self, mock_db):
        """Test that visualizations are properly formatted."""
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=[
            ("2023-01-01", 10, 5, 5, "1", "10"),
            ("2023-01-02", 15, 8, 7, "2", "11"),
            ("2023-01-03", 12, 6, 6, "3", "12"),
        ])
        mock_db.execute.return_value = mock_cursor
        
        with patch("imessage_mcp_server.tools.communication_enhanced.get_database", 
                   AsyncMock(return_value=mock_db)):
            result = await enhanced_cadence_calendar_tool(
                db_path="test.db",
                time_period="7d",
                include_visualizations=True
            )
        
        # Check visualization format
        for viz_name, viz_data in result["visualizations"].items():
            # Should be base64 encoded PNG
            assert viz_data.startswith("data:image/png;base64,")
            
            # Decode and check it's valid base64
            base64_data = viz_data.split(",")[1]
            try:
                decoded = base64.b64decode(base64_data)
                assert len(decoded) > 0  # Should have content
            except:
                pytest.fail(f"Invalid base64 encoding for {viz_name}")


class TestErrorHandling:
    """Test error handling in enhanced tools."""
    
    @pytest.mark.asyncio
    async def test_database_error_handling(self, mock_db):
        """Test handling of database errors."""
        mock_db.execute.side_effect = Exception("Database connection failed")
        
        with patch("imessage_mcp_server.tools.communication_enhanced.get_database", 
                   AsyncMock(return_value=mock_db)):
            result = await enhanced_cadence_calendar_tool(
                db_path="test.db",
                contact_id="test@example.com"
            )
        
        assert "error" in result
        assert result["error_type"] == "analysis_error"
    
    @pytest.mark.asyncio
    async def test_visualization_error_handling(self, mock_db):
        """Test handling of visualization errors."""
        mock_cursor = AsyncMock()
        # Return data that might cause visualization issues
        mock_cursor.fetchall = AsyncMock(return_value=[])
        mock_db.execute.return_value = mock_cursor
        
        with patch("imessage_mcp_server.tools.communication_enhanced.get_database", 
                   AsyncMock(return_value=mock_db)):
            # Should handle empty data gracefully
            result = await enhanced_cadence_calendar_tool(
                db_path="test.db",
                include_visualizations=True
            )
        
        # Should complete without error
        assert "total_messages" in result
        assert result["total_messages"] == 0