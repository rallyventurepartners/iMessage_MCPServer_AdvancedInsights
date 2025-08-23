#!/usr/bin/env python3
"""Direct test of cloud-aware tools."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from imessage_mcp_server.tools.cloud_aware import (
    cloud_status_tool,
    smart_query_tool,
    progressive_analysis_tool,
)

async def test_cloud_status():
    """Test the cloud status tool."""
    print("\n=== Testing Cloud Status Tool ===")
    
    result = await cloud_status_tool(
        db_path="~/Library/Messages/chat.db",
        check_specific_dates=["2024-01-15", "2024-12-01"]
    )
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    summary = result.get("summary", {})
    print(f"\nTotal messages: {summary.get('total_messages', 0):,}")
    print(f"Local messages: {summary.get('local_messages', 0):,}")
    print(f"Cloud messages: {summary.get('cloud_messages', 0):,}")
    print(f"Local percentage: {summary.get('local_percentage', 0):.1f}%")
    print(f"Cloud percentage: {summary.get('cloud_percentage', 0):.1f}%")
    
    if result.get("availability_gaps"):
        print("\nAvailability gaps (top 5):")
        for gap in result["availability_gaps"][:5]:
            print(f"  - {gap['period']}: {gap['cloud_percentage']:.1f}% in cloud")
    
    if result.get("recommendations"):
        print("\nRecommendations:")
        for rec in result["recommendations"]:
            print(f"  - [{rec['priority']}] {rec.get('description', rec['action'])}")
            if 'command' in rec:
                print(f"    Command: {rec['command']}")

async def test_smart_query():
    """Test the smart query tool."""
    print("\n=== Testing Smart Query Tool ===")
    
    result = await smart_query_tool(
        db_path="~/Library/Messages/chat.db",
        query_type="stats",
        date_range={"start": "2024-01-01", "end": "2024-12-31"},
        auto_download=False
    )
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    metadata = result.get("_metadata", {})
    print(f"\nTotal matching messages: {metadata.get('total_matching_messages', 0):,}")
    print(f"Locally available: {metadata.get('locally_available', 0):,}")
    print(f"Availability: {metadata.get('availability_percentage', 0):.1f}%")
    print(f"Data completeness: {metadata.get('data_completeness', 'unknown')}")

async def test_progressive_analysis():
    """Test the progressive analysis tool."""
    print("\n=== Testing Progressive Analysis Tool ===")
    
    result = await progressive_analysis_tool(
        db_path="~/Library/Messages/chat.db",
        analysis_type="sentiment",
        options={"window": "recent"}
    )
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    print(f"\nAnalysis type: {result.get('analysis_type', 'unknown')}")
    print(f"Time window: {result.get('time_window', 'unknown')}")
    print(f"Overall confidence: {result.get('overall_confidence', 0):.2f}")
    print(f"Data quality: {result.get('data_quality', 'unknown')}")
    
    if result.get("results"):
        print("\nAnalysis results:")
        for chunk in result["results"][:3]:  # Show first 3 chunks
            print(f"  - {chunk['period']}: {chunk['confidence']:.2f} confidence")
            print(f"    Messages: {chunk['messages_analyzed']}/{chunk['messages_total']}")

async def main():
    """Run all tests."""
    print("Testing Cloud-Aware Tools for iMessage MCP Server")
    print("=" * 50)
    
    await test_cloud_status()
    await test_smart_query()
    await test_progressive_analysis()
    
    print("\n" + "=" * 50)
    print("Testing complete!")

if __name__ == "__main__":
    asyncio.run(main())