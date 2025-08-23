"""
MCP Tools for iMessage Advanced Insights.

This package contains all tool implementations for the MCP server.
Each tool is in its own module for better organization and maintainability.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ToolResponse(BaseModel):
    """Standard response format for all tools."""

    success: bool = Field(default=True, description="Whether the tool executed successfully")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Tool response data")
    error: Optional[str] = Field(default=None, description="Error message if any")
    error_type: Optional[str] = Field(default=None, description="Type of error")
    insights: Optional[list[str]] = Field(default=None, description="Human-readable insights")


class ConsentRequest(BaseModel):
    """Schema for consent request."""

    expiry_hours: int = Field(
        default=24, ge=1, le=720, description="Hours until consent expires (1-720)"
    )


class ContactQuery(BaseModel):
    """Schema for contact resolution queries."""

    query: str = Field(description="Phone number, email, or name fragment to search")


class DatabasePath(BaseModel):
    """Schema for database path parameter."""

    db_path: str = Field(
        default="~/Library/Messages/chat.db", description="Path to iMessage database"
    )


__all__ = ["ToolResponse", "ConsentRequest", "ContactQuery", "DatabasePath"]
