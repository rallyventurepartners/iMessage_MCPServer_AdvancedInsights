"""
Pydantic models and schemas for MCP tools.

This module defines all input/output models for tools, ensuring
type safety and automatic JSON schema generation.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ToolName(str, Enum):
    """Enumeration of all available tools."""

    HEALTH_CHECK = "imsg.health_check"
    SUMMARY_OVERVIEW = "imsg.summary_overview"
    CONTACT_RESOLVE = "imsg.contact_resolve"
    RELATIONSHIP_INTELLIGENCE = "imsg.relationship_intelligence"
    CONVERSATION_TOPICS = "imsg.conversation_topics"
    SENTIMENT_EVOLUTION = "imsg.sentiment_evolution"
    RESPONSE_TIME_DISTRIBUTION = "imsg.response_time_distribution"
    CADENCE_CALENDAR = "imsg.cadence_calendar"
    ANOMALY_SCAN = "imsg.anomaly_scan"
    BEST_CONTACT_TIME = "imsg.best_contact_time"
    NETWORK_INTELLIGENCE = "imsg.network_intelligence"
    SAMPLE_MESSAGES = "imsg.sample_messages"


# Input Models


class HealthCheckInput(BaseModel):
    """Input for health check tool."""

    db_path: str = Field(
        default="~/Library/Messages/chat.db", description="Path to iMessage database"
    )


class SummaryOverviewInput(BaseModel):
    """Input for summary overview tool."""

    db_path: str = Field(
        default="~/Library/Messages/chat.db", description="Path to iMessage database"
    )
    redact: bool = Field(default=True, description="Apply PII redaction")


class ContactResolveInput(BaseModel):
    """Input for contact resolution tool."""

    query: str = Field(description="Phone number, email, or contact name to resolve")


class RelationshipIntelligenceInput(BaseModel):
    """Input for relationship intelligence tool."""

    db_path: str = Field(
        default="~/Library/Messages/chat.db", description="Path to iMessage database"
    )
    contact_filters: Optional[List[str]] = Field(
        default=None, description="Filter to specific contacts"
    )
    window_days: int = Field(default=365, ge=1, le=3650, description="Analysis window in days")
    redact: bool = Field(default=True, description="Apply PII redaction")


class ConversationTopicsInput(BaseModel):
    """Input for conversation topics tool."""

    db_path: str = Field(
        default="~/Library/Messages/chat.db", description="Path to iMessage database"
    )
    contact_id: Optional[str] = Field(default=None, description="Specific contact ID")
    since_days: int = Field(default=180, ge=1, le=3650, description="Lookback period in days")
    top_k: int = Field(default=25, ge=1, le=100, description="Number of top topics to return")
    use_transformer: bool = Field(default=False, description="Use transformer models (if enabled)")


class SentimentEvolutionInput(BaseModel):
    """Input for sentiment evolution tool."""

    db_path: str = Field(
        default="~/Library/Messages/chat.db", description="Path to iMessage database"
    )
    contact_id: Optional[str] = Field(default=None, description="Specific contact ID")
    window_days: int = Field(default=30, ge=1, le=365, description="Rolling window size in days")


class ResponseTimeInput(BaseModel):
    """Input for response time distribution tool."""

    db_path: str = Field(
        default="~/Library/Messages/chat.db", description="Path to iMessage database"
    )
    contact_id: Optional[str] = Field(default=None, description="Specific contact ID")


class CadenceCalendarInput(BaseModel):
    """Input for cadence calendar tool."""

    db_path: str = Field(
        default="~/Library/Messages/chat.db", description="Path to iMessage database"
    )
    contact_id: Optional[str] = Field(default=None, description="Specific contact ID")


class AnomalyScanInput(BaseModel):
    """Input for anomaly scan tool."""

    db_path: str = Field(
        default="~/Library/Messages/chat.db", description="Path to iMessage database"
    )
    contact_id: Optional[str] = Field(default=None, description="Specific contact ID")
    lookback_days: int = Field(default=90, ge=7, le=365, description="Analysis period in days")


class BestContactTimeInput(BaseModel):
    """Input for best contact time tool."""

    db_path: str = Field(
        default="~/Library/Messages/chat.db", description="Path to iMessage database"
    )
    contact_id: Optional[str] = Field(default=None, description="Specific contact ID")


class NetworkIntelligenceInput(BaseModel):
    """Input for network intelligence tool."""

    db_path: str = Field(
        default="~/Library/Messages/chat.db", description="Path to iMessage database"
    )
    since_days: int = Field(default=365, ge=1, le=3650, description="Lookback period in days")


class SampleMessagesInput(BaseModel):
    """Input for sample messages tool."""

    db_path: str = Field(
        default="~/Library/Messages/chat.db", description="Path to iMessage database"
    )
    contact_id: Optional[str] = Field(default=None, description="Filter by contact ID")
    limit: int = Field(default=10, ge=1, le=20, description="Maximum messages to return")


# Output Models


class HealthCheckOutput(BaseModel):
    """Output for health check tool."""

    db_version: str
    tables: List[str]
    indices_ok: bool
    read_only_ok: bool
    warnings: List[str]


class DateRange(BaseModel):
    """Date range model."""

    start: str
    end: str


class MessageDirection(BaseModel):
    """Message direction counts."""

    sent: int
    received: int


class MessagePlatform(BaseModel):
    """Message platform counts."""

    iMessage: int = Field(alias="iMessage")
    SMS: int = Field(alias="SMS")


class AttachmentCounts(BaseModel):
    """Attachment type counts."""

    images: int
    videos: int
    other: int


class SummaryOverviewOutput(BaseModel):
    """Output for summary overview tool."""

    total_messages: int
    unique_contacts: int
    date_range: DateRange
    by_direction: MessageDirection
    by_platform: MessagePlatform
    attachments: AttachmentCounts
    notes: List[str]


class ContactResolveOutput(BaseModel):
    """Output for contact resolution tool."""

    contact_id: str
    display_name: str
    kind: str  # "phone", "email", "apple_id"


class ContactIntelligence(BaseModel):
    """Individual contact intelligence data."""

    contact_id: str
    display_name: Optional[str]
    messages_total: int
    sent_pct: float = Field(ge=0, le=100)
    median_response_time_s: float
    avg_daily_msgs: float
    streak_days_max: int
    last_contacted: str
    flags: List[str]


class RelationshipIntelligenceOutput(BaseModel):
    """Output for relationship intelligence tool."""

    contacts: List[ContactIntelligence]


class TopicTerm(BaseModel):
    """Topic term with count."""

    term: str
    count: int


class TopicTrend(BaseModel):
    """Topic with trend sparkline."""

    term: str
    spark: str  # Unicode sparkline


class ConversationTopicsOutput(BaseModel):
    """Output for conversation topics tool."""

    terms: List[TopicTerm]
    trend: List[TopicTrend]
    notes: List[str]


class SentimentPoint(BaseModel):
    """Sentiment data point."""

    ts: str  # ISO timestamp
    score: float = Field(ge=-1.0, le=1.0)


class SentimentSummary(BaseModel):
    """Sentiment summary statistics."""

    mean: float
    delta_30d: float


class SentimentEvolutionOutput(BaseModel):
    """Output for sentiment evolution tool."""

    series: List[SentimentPoint]
    summary: SentimentSummary


class ResponseHistogram(BaseModel):
    """Response time histogram bucket."""

    bucket: str
    count: int


class ResponseTimeOutput(BaseModel):
    """Output for response time distribution tool."""

    p50_s: float
    p90_s: float
    p99_s: float
    histogram: List[ResponseHistogram]
    samples: int


class CadenceCalendarOutput(BaseModel):
    """Output for cadence calendar tool."""

    matrix: List[List[int]]  # 24x7 matrix
    hours: List[int]
    weekdays: List[str]


class Anomaly(BaseModel):
    """Detected anomaly."""

    ts: str  # ISO timestamp
    type: str  # "silence", "burst", "pattern_change"
    severity: float = Field(ge=0.0, le=1.0)
    note: str


class AnomalyScanOutput(BaseModel):
    """Output for anomaly scan tool."""

    anomalies: List[Anomaly]


class ContactWindow(BaseModel):
    """Optimal contact time window."""

    weekday: str
    hour: int = Field(ge=0, le=23)
    score: float = Field(ge=0.0, le=1.0)


class BestContactTimeOutput(BaseModel):
    """Output for best contact time tool."""

    windows: List[ContactWindow]


class NetworkNode(BaseModel):
    """Network graph node."""

    id: str
    label: Optional[str]
    degree: int


class NetworkEdge(BaseModel):
    """Network graph edge."""

    source: str
    target: str
    weight: int


class Community(BaseModel):
    """Network community."""

    community_id: int
    members: List[str]


class KeyConnector(BaseModel):
    """Key network connector."""

    contact_id: str
    score: float


class NetworkIntelligenceOutput(BaseModel):
    """Output for network intelligence tool."""

    nodes: List[NetworkNode]
    edges: List[NetworkEdge]
    communities: List[Community]
    key_connectors: List[KeyConnector]


class MessagePreview(BaseModel):
    """Redacted message preview."""

    ts: str  # ISO timestamp
    direction: str  # "sent" or "received"
    contact_id: str
    preview: str = Field(max_length=160)


class SampleMessagesOutput(BaseModel):
    """Output for sample messages tool."""

    messages: List[MessagePreview]


# Schema generation


def get_tool_input_schema(tool_name: ToolName) -> Dict[str, Any]:
    """Get JSON schema for tool input."""
    input_models = {
        ToolName.HEALTH_CHECK: HealthCheckInput,
        ToolName.SUMMARY_OVERVIEW: SummaryOverviewInput,
        ToolName.CONTACT_RESOLVE: ContactResolveInput,
        ToolName.RELATIONSHIP_INTELLIGENCE: RelationshipIntelligenceInput,
        ToolName.CONVERSATION_TOPICS: ConversationTopicsInput,
        ToolName.SENTIMENT_EVOLUTION: SentimentEvolutionInput,
        ToolName.RESPONSE_TIME_DISTRIBUTION: ResponseTimeInput,
        ToolName.CADENCE_CALENDAR: CadenceCalendarInput,
        ToolName.ANOMALY_SCAN: AnomalyScanInput,
        ToolName.BEST_CONTACT_TIME: BestContactTimeInput,
        ToolName.NETWORK_INTELLIGENCE: NetworkIntelligenceInput,
        ToolName.SAMPLE_MESSAGES: SampleMessagesInput,
    }

    model = input_models.get(tool_name)
    if model:
        return model.model_json_schema()
    return {}


def get_tool_output_model(tool_name: ToolName) -> Optional[type[BaseModel]]:
    """Get output model for a tool."""
    output_models = {
        ToolName.HEALTH_CHECK: HealthCheckOutput,
        ToolName.SUMMARY_OVERVIEW: SummaryOverviewOutput,
        ToolName.CONTACT_RESOLVE: ContactResolveOutput,
        ToolName.RELATIONSHIP_INTELLIGENCE: RelationshipIntelligenceOutput,
        ToolName.CONVERSATION_TOPICS: ConversationTopicsOutput,
        ToolName.SENTIMENT_EVOLUTION: SentimentEvolutionOutput,
        ToolName.RESPONSE_TIME_DISTRIBUTION: ResponseTimeOutput,
        ToolName.CADENCE_CALENDAR: CadenceCalendarOutput,
        ToolName.ANOMALY_SCAN: AnomalyScanOutput,
        ToolName.BEST_CONTACT_TIME: BestContactTimeOutput,
        ToolName.NETWORK_INTELLIGENCE: NetworkIntelligenceOutput,
        ToolName.SAMPLE_MESSAGES: SampleMessagesOutput,
    }

    return output_models.get(tool_name)
