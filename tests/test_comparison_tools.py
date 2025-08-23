"""Tests for relationship comparison tools."""

import sys
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from imessage_mcp_server.tools.comparison import relationship_comparison_tool


@pytest.mark.asyncio
async def test_relationship_comparison_basic(populated_db):
    """Test basic relationship comparison."""
    # Use multiple test contacts
    contact_ids = ["+15551234567", "+15559876543", "test@example.com"]
    
    result = await relationship_comparison_tool(
        contact_ids=contact_ids,
        comparison_type="comprehensive",
        db_path=populated_db,
        redact=True
    )
    
    # Check basic structure
    assert "contacts_analyzed" in result
    assert "comparison_type" in result
    assert "overview" in result
    assert "comparative_matrix" in result
    assert "insights" in result
    
    # Check overview
    overview = result["overview"]
    assert "healthiest_relationship" in overview
    assert "highest_quality_score" in overview
    assert "average_quality_score" in overview
    
    # Check matrix
    matrix = result["comparative_matrix"]
    assert "overall_score" in matrix
    assert "depth_score" in matrix
    assert "balance_score" in matrix
    assert "emotion_score" in matrix
    assert "consistency_score" in matrix
    
    # Check each metric has statistics
    for metric_data in matrix.values():
        assert "values" in metric_data
        assert "mean" in metric_data
        assert "std" in metric_data
        assert "min" in metric_data
        assert "max" in metric_data


@pytest.mark.asyncio
async def test_relationship_comparison_clustering(populated_db):
    """Test clustering functionality."""
    # Need at least 3 contacts for clustering
    contact_ids = ["+15551234567", "+15559876543", "test@example.com", "+15551112222"]
    
    result = await relationship_comparison_tool(
        contact_ids=contact_ids,
        comparison_type="comprehensive",
        include_clusters=True,
        db_path=populated_db,
        redact=True
    )
    
    if result.get("contacts_analyzed", 0) >= 3:
        # Should have clusters
        assert "relationship_clusters" in result
        clusters = result["relationship_clusters"]
        
        # Check cluster structure
        for cluster in clusters:
            assert "type" in cluster
            assert "size" in cluster
            assert "members" in cluster
            assert "characteristics" in cluster
            assert "average_scores" in cluster
            
            # Check cluster types
            assert cluster["type"] in ["inner_circle", "regular_contact", "occasional", "dormant"]
            
            # Check members are hashed (due to redact=True)
            for member in cluster["members"]:
                assert member.startswith("hash:")


@pytest.mark.asyncio
async def test_relationship_comparison_no_clustering(populated_db):
    """Test comparison without clustering."""
    contact_ids = ["+15551234567", "+15559876543"]
    
    result = await relationship_comparison_tool(
        contact_ids=contact_ids,
        comparison_type="comprehensive",
        include_clusters=False,
        db_path=populated_db,
        redact=True
    )
    
    # Should not have clusters
    assert "relationship_clusters" not in result


@pytest.mark.asyncio
async def test_relationship_comparison_insights(populated_db):
    """Test insight generation."""
    contact_ids = ["+15551234567", "+15559876543", "test@example.com"]
    
    result = await relationship_comparison_tool(
        contact_ids=contact_ids,
        comparison_type="comprehensive",
        db_path=populated_db,
        redact=True
    )
    
    # Check insights structure
    assert "insights" in result
    insights = result["insights"]
    
    assert "patterns" in insights
    assert "recommendations" in insights
    assert "strengths" in insights
    assert "opportunities" in insights
    
    # All should be lists
    assert isinstance(insights["patterns"], list)
    assert isinstance(insights["recommendations"], list)
    assert isinstance(insights["strengths"], list)
    assert isinstance(insights["opportunities"], list)


@pytest.mark.asyncio
async def test_relationship_comparison_privacy(populated_db):
    """Test privacy features in comparison."""
    contact_ids = ["+15551234567", "+15559876543"]
    
    # Test with redaction
    result_redacted = await relationship_comparison_tool(
        contact_ids=contact_ids,
        db_path=populated_db,
        redact=True
    )
    
    if "overview" in result_redacted:
        # Contact IDs should be hashed
        if "healthiest_relationship" in result_redacted["overview"]:
            assert result_redacted["overview"]["healthiest_relationship"].startswith("hash:")
    
    # Test without redaction
    result_clear = await relationship_comparison_tool(
        contact_ids=contact_ids,
        db_path=populated_db,
        redact=False
    )
    
    if "overview" in result_clear:
        # Contact IDs should be clear
        if "healthiest_relationship" in result_clear["overview"]:
            assert result_clear["overview"]["healthiest_relationship"] in contact_ids


@pytest.mark.asyncio
async def test_relationship_comparison_edge_cases(populated_db):
    """Test edge cases for comparison."""
    # Empty contact list
    result = await relationship_comparison_tool(
        contact_ids=[],
        db_path=populated_db,
        redact=True
    )
    assert "error" in result
    assert result["error"] == "No contacts provided for comparison"
    
    # Too many contacts
    many_contacts = [f"contact{i}@example.com" for i in range(15)]
    result = await relationship_comparison_tool(
        contact_ids=many_contacts,
        db_path=populated_db,
        redact=True
    )
    assert "error" in result
    assert "Maximum 10 contacts" in result["error"]
    
    # Non-existent contacts
    result = await relationship_comparison_tool(
        contact_ids=["nonexistent1@example.com", "nonexistent2@example.com"],
        db_path=populated_db,
        redact=True
    )
    
    # Should handle gracefully
    if "error" not in result:
        assert result["contacts_analyzed"] == 0 or result["contacts_errored"] > 0


@pytest.mark.asyncio
async def test_relationship_comparison_single_contact(populated_db):
    """Test comparison with single contact."""
    result = await relationship_comparison_tool(
        contact_ids=["+15551234567"],
        db_path=populated_db,
        redact=True
    )
    
    # Should work but without clustering
    if "error" not in result:
        assert result["contacts_analyzed"] <= 1
        assert "relationship_clusters" not in result  # Can't cluster single contact


@pytest.mark.asyncio
async def test_comparison_types(populated_db):
    """Test different comparison types."""
    contact_ids = ["+15551234567", "+15559876543"]
    
    for comp_type in ["comprehensive", "quick", "focused"]:
        result = await relationship_comparison_tool(
            contact_ids=contact_ids,
            comparison_type=comp_type,
            db_path=populated_db,
            redact=True
        )
        
        if "error" not in result:
            assert result["comparison_type"] == comp_type