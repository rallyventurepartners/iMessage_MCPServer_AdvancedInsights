#!/usr/bin/env python3
"""
Database Sharding Monitoring Dashboard

This tool provides a web-based dashboard for monitoring the health, performance,
and status of database shards in the iMessage insights system.

Features:
- Real-time metrics visualization
- Shard health monitoring
- Performance tracking across shards
- Administrative controls for optimization and maintenance
- Visual representation of shard date ranges and message distribution

Usage:
    python shard_dashboard.py [--host HOST] [--port PORT] [--debug]
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta

# Web dashboard libraries
import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html

# Add script directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import database classes
from src.database.sharded_async_messages_db import ShardedAsyncMessagesDB

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default paths
HOME = os.path.expanduser("~")
DEFAULT_DB_PATH = os.path.join(HOME, "Library", "Messages", "chat.db")
DEFAULT_SHARDS_DIR = os.path.join(HOME, ".imessage_insights", "shards")

# Global variables to track metrics
sharded_db = None
metrics_history = {
    "timestamp": [],
    "cpu_usage": [],
    "memory_usage": [],
    "query_count": [],
    "avg_query_time": [],
}
shard_metrics = {}
query_history = []

# Maximum history points to keep
MAX_HISTORY_POINTS = 100
MAX_QUERY_HISTORY = 50

# Background data collection task
background_task = None

# Create Dash app
app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    title="iMessage Shard Dashboard",
)

# Define app layout
app.layout = html.Div(
    [
        html.Div(
            [
                html.H1("iMessage Database Shard Dashboard", className="header-title"),
                html.P(
                    "Monitor and manage database shards for large iMessage databases",
                    className="header-description",
                ),
                html.Div(
                    [
                        html.Button(
                            "Refresh Data",
                            id="refresh-button",
                            className="control-button",
                        ),
                        html.Button(
                            "Run Optimization",
                            id="optimize-button",
                            className="control-button",
                        ),
                        html.Div(id="status-message", className="status-message"),
                    ],
                    className="header-controls",
                ),
            ],
            className="header",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H2("System Overview"),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H3(
                                            "Total Shards", className="metric-title"
                                        ),
                                        html.P(
                                            id="total-shards", className="metric-value"
                                        ),
                                    ],
                                    className="metric-card",
                                ),
                                html.Div(
                                    [
                                        html.H3("Total Size", className="metric-title"),
                                        html.P(
                                            id="total-size", className="metric-value"
                                        ),
                                    ],
                                    className="metric-card",
                                ),
                                html.Div(
                                    [
                                        html.H3(
                                            "Total Messages", className="metric-title"
                                        ),
                                        html.P(
                                            id="total-messages",
                                            className="metric-value",
                                        ),
                                    ],
                                    className="metric-card",
                                ),
                                html.Div(
                                    [
                                        html.H3("Date Range", className="metric-title"),
                                        html.P(
                                            id="date-range", className="metric-value"
                                        ),
                                    ],
                                    className="metric-card",
                                ),
                            ],
                            className="metrics-container",
                        ),
                    ],
                    className="card overview-card",
                ),
                html.Div(
                    [
                        html.H2("Performance Metrics"),
                        dcc.Graph(id="performance-graph"),
                        dcc.Interval(
                            id="interval-component",
                            interval=5000,  # in milliseconds
                            n_intervals=0,
                        ),
                    ],
                    className="card performance-card",
                ),
            ],
            className="row",
        ),
        html.Div(
            [
                html.Div(
                    [html.H2("Shard Distribution"), dcc.Graph(id="shard-timeline")],
                    className="card timeline-card",
                ),
                html.Div(
                    [
                        html.H2("Message Distribution"),
                        dcc.Graph(id="message-distribution"),
                    ],
                    className="card distribution-card",
                ),
            ],
            className="row",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H2("Shard Details"),
                        html.Div(id="shard-table", className="table-container"),
                    ],
                    className="card shards-card",
                ),
                html.Div(
                    [
                        html.H2("Recent Queries"),
                        html.Div(id="query-table", className="table-container"),
                    ],
                    className="card queries-card",
                ),
            ],
            className="row",
        ),
        html.Div(
            [
                html.H2("Administrative Tools"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3("Shard Management"),
                                html.Div(
                                    [
                                        html.Label("Shard Size (Months)"),
                                        dcc.Input(
                                            id="shard-size-input",
                                            type="number",
                                            min=1,
                                            max=24,
                                            value=6,
                                        ),
                                        html.Button(
                                            "Create New Shards",
                                            id="create-shards-button",
                                            className="admin-button",
                                        ),
                                        html.Div(
                                            id="shard-creation-status",
                                            className="status-message",
                                        ),
                                    ],
                                    className="admin-control-group",
                                ),
                            ],
                            className="admin-card",
                        ),
                        html.Div(
                            [
                                html.H3("Maintenance"),
                                html.Div(
                                    [
                                        html.Label("Select Shard"),
                                        dcc.Dropdown(
                                            id="shard-dropdown",
                                            options=[],
                                            className="dropdown",
                                        ),
                                        html.Button(
                                            "Optimize Shard",
                                            id="optimize-shard-button",
                                            className="admin-button",
                                        ),
                                        html.Button(
                                            "Rebuild Indexes",
                                            id="rebuild-indexes-button",
                                            className="admin-button",
                                        ),
                                        html.Div(
                                            id="maintenance-status",
                                            className="status-message",
                                        ),
                                    ],
                                    className="admin-control-group",
                                ),
                            ],
                            className="admin-card",
                        ),
                    ],
                    className="admin-controls",
                ),
            ],
            className="card admin-tools-card",
        ),
        # Hidden div for storing the data
        html.Div(id="shards-data-store", style={"display": "none"}),
        html.Div(id="metrics-data-store", style={"display": "none"}),
        html.Div(id="query-data-store", style={"display": "none"}),
    ]
)


async def initialize_database(db_path, shards_dir):
    """Initialize the ShardedAsyncMessagesDB instance."""
    global sharded_db

    try:
        # Initialize database
        db = ShardedAsyncMessagesDB(db_path=db_path, shards_dir=shards_dir)
        await db.initialize()

        sharded_db = db
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


async def collect_metrics():
    """Background task to collect metrics."""
    global metrics_history, shard_metrics, query_history, sharded_db

    if not sharded_db:
        return

    try:
        # Get current metrics
        import psutil

        process = psutil.Process(os.getpid())

        # CPU and memory
        cpu_percent = process.cpu_percent()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)

        # Database metrics
        shards_info = await sharded_db.get_shards_info()

        # Update global metrics
        timestamp = datetime.now().strftime("%H:%M:%S")
        metrics_history["timestamp"].append(timestamp)
        metrics_history["cpu_usage"].append(cpu_percent)
        metrics_history["memory_usage"].append(memory_mb)

        # Query metrics if shard manager exists
        manager = sharded_db.shard_manager if sharded_db.using_shards else None
        if manager:
            query_count = manager._query_stats.get("total_queries", 0)
            avg_time = manager._query_stats.get("total_time_ms", 0) / max(
                1, query_count
            )

            metrics_history["query_count"].append(query_count)
            metrics_history["avg_query_time"].append(avg_time)

            # Update shard-specific metrics
            for shard in manager.shards:
                shard_id = str(shard.shard_path.name)
                if shard_id not in shard_metrics:
                    shard_metrics[shard_id] = {
                        "timestamp": [],
                        "query_count": [],
                        "access_time": [],
                    }

                # For now, we don't have shard-specific metrics in the implementation
                # In a production system, we would track these per shard

            # Collect recent query info
            # This would normally come from a query log that we'd implement in the shard manager
            # For now, we'll simulate with empty data

        # Limit history length
        if len(metrics_history["timestamp"]) > MAX_HISTORY_POINTS:
            for key in metrics_history:
                metrics_history[key] = metrics_history[key][-MAX_HISTORY_POINTS:]

    except Exception as e:
        logger.error(f"Error collecting metrics: {e}")
        import traceback

        logger.error(traceback.format_exc())


async def background_metrics_collection():
    """Run the metrics collection in the background."""
    while True:
        await collect_metrics()
        await asyncio.sleep(5)  # Collect every 5 seconds


@app.callback(
    [
        Output("shards-data-store", "children"),
        Output("total-shards", "children"),
        Output("total-size", "children"),
        Output("total-messages", "children"),
        Output("date-range", "children"),
        Output("shard-table", "children"),
        Output("shard-dropdown", "options"),
    ],
    [Input("refresh-button", "n_clicks"), Input("interval-component", "n_intervals")],
)
def update_shards_data(n_clicks, n_intervals):
    """Update the shards data."""
    if not sharded_db:
        return (
            json.dumps({}),
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            html.Div("Database not initialized"),
            [],
        )

    # Use asyncio to run the async function and get the result
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        shards_info = loop.run_until_complete(sharded_db.get_shards_info())
    finally:
        loop.close()

    # Process data for display
    if not shards_info["using_shards"]:
        return (
            json.dumps(shards_info),
            "1",
            f"{shards_info.get('db_size_gb', 0):.2f} GB",
            "Unknown",
            "Full Database",
            html.Div("Not using sharded database"),
            [],
        )

    # Calculate totals
    total_shards = shards_info["shard_count"]
    total_size_mb = sum(shard.get("size_mb", 0) for shard in shards_info["shards"])
    total_size_gb = total_size_mb / 1024
    total_messages = sum(
        shard.get("message_count", 0) for shard in shards_info["shards"]
    )

    # Determine overall date range
    start_dates = [
        datetime.fromisoformat(shard["start_date"])
        for shard in shards_info["shards"]
        if shard.get("start_date")
    ]
    end_dates = [
        datetime.fromisoformat(shard["end_date"])
        for shard in shards_info["shards"]
        if shard.get("end_date")
    ]

    if start_dates and end_dates:
        min_date = min(start_dates).strftime("%Y-%m-%d")
        max_date = max(end_dates).strftime("%Y-%m-%d")
        date_range = f"{min_date} to {max_date}"
    else:
        date_range = "Unknown"

    # Create shard table
    table_header = html.Thead(
        html.Tr(
            [
                html.Th("Shard"),
                html.Th("Date Range"),
                html.Th("Messages"),
                html.Th("Size"),
                html.Th("Status"),
            ]
        )
    )

    rows = []
    dropdown_options = []

    for shard in shards_info["shards"]:
        # Create row for each shard
        shard_name = os.path.basename(shard["path"])
        date_range = (
            f"{shard.get('start_date', '?')[:10]} to {shard.get('end_date', '?')[:10]}"
        )
        message_count = f"{shard.get('message_count', 0):,}"
        size = f"{shard.get('size_mb', 0):.2f} MB"

        # Determine status (health)
        has_fts = shard.get("has_fts", False)
        index_count = shard.get("index_count", 0)

        if index_count > 5 and has_fts:
            status = html.Span("Healthy", className="status-healthy")
        elif index_count > 0:
            status = html.Span("OK", className="status-ok")
        else:
            status = html.Span("Needs Optimization", className="status-warning")

        row = html.Tr(
            [
                html.Td(shard_name),
                html.Td(date_range),
                html.Td(message_count),
                html.Td(size),
                html.Td(status),
            ]
        )
        rows.append(row)

        # Add to dropdown options
        dropdown_options.append(
            {"label": f"{shard_name} ({date_range})", "value": shard["path"]}
        )

    table_body = html.Tbody(rows)
    table = html.Table([table_header, table_body], className="data-table")

    return (
        json.dumps(shards_info),
        f"{total_shards}",
        f"{total_size_gb:.2f} GB",
        f"{total_messages:,}",
        date_range,
        table,
        dropdown_options,
    )


@app.callback(
    Output("metrics-data-store", "children"),
    [Input("interval-component", "n_intervals")],
)
def update_metrics_data(n_intervals):
    """Update the metrics data."""
    return json.dumps(metrics_history)


@app.callback(
    Output("performance-graph", "figure"), [Input("metrics-data-store", "children")]
)
def update_performance_graph(metrics_json):
    """Update the performance metrics graph."""
    metrics = json.loads(metrics_json)

    if not metrics.get("timestamp"):
        # No data yet
        return go.Figure().update_layout(
            title="Waiting for performance data...",
            xaxis_title="Time",
            yaxis_title="Value",
        )

    # Create figure with secondary y-axis
    fig = go.Figure()

    # Add traces
    fig.add_trace(
        go.Scatter(
            x=metrics["timestamp"], y=metrics["memory_usage"], name="Memory Usage (MB)"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=metrics["timestamp"],
            y=metrics["cpu_usage"],
            name="CPU Usage (%)",
            yaxis="y2",
        )
    )

    # Add query metrics if available
    if "query_count" in metrics and metrics["query_count"]:
        # Normalize query count for display
        max_query = max(metrics["query_count"]) if metrics["query_count"] else 1
        max_mem = max(metrics["memory_usage"]) if metrics["memory_usage"] else 1
        normalized_query = [q * max_mem / max_query for q in metrics["query_count"]]

        fig.add_trace(
            go.Scatter(
                x=metrics["timestamp"],
                y=normalized_query,
                name="Query Count (normalized)",
                line=dict(dash="dash"),
            )
        )

    if "avg_query_time" in metrics and metrics["avg_query_time"]:
        # Normalize query time for display
        max_time = max(metrics["avg_query_time"]) if metrics["avg_query_time"] else 1
        max_cpu = max(metrics["cpu_usage"]) if metrics["cpu_usage"] else 1
        normalized_time = [t * max_cpu / max_time for t in metrics["avg_query_time"]]

        fig.add_trace(
            go.Scatter(
                x=metrics["timestamp"],
                y=normalized_time,
                name="Avg Query Time (normalized)",
                line=dict(dash="dash"),
                yaxis="y2",
            )
        )

    # Create layout with secondary y-axis
    fig.update_layout(
        title="System Performance",
        xaxis=dict(title="Time", titlefont_size=12, tickfont_size=12),
        yaxis=dict(
            title="Memory Usage (MB)",
            titlefont=dict(color="#1f77b4"),
            tickfont=dict(color="#1f77b4"),
            titlefont_size=12,
            tickfont_size=12,
        ),
        yaxis2=dict(
            title="CPU Usage (%)",
            titlefont=dict(color="#ff7f0e"),
            tickfont=dict(color="#ff7f0e"),
            anchor="x",
            overlaying="y",
            side="right",
            titlefont_size=12,
            tickfont_size=12,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=60, t=50, b=50),
        height=300,
    )

    return fig


@app.callback(
    Output("shard-timeline", "figure"), [Input("shards-data-store", "children")]
)
def update_shard_timeline(shards_json):
    """Update the shard timeline visualization."""
    shards_info = json.loads(shards_json)

    if not shards_info.get("using_shards", False):
        return go.Figure().update_layout(title="Not using sharded database")

    # Prepare data for timeline
    shard_data = []

    for shard in shards_info.get("shards", []):
        if not shard.get("start_date") or not shard.get("end_date"):
            continue

        shard_name = os.path.basename(shard["path"])
        start_date = datetime.fromisoformat(shard["start_date"])
        end_date = datetime.fromisoformat(shard["end_date"])

        shard_data.append(
            {
                "Shard": shard_name,
                "Start": start_date,
                "End": end_date,
                "Messages": shard.get("message_count", 0),
                "Size_MB": shard.get("size_mb", 0),
            }
        )

    if not shard_data:
        return go.Figure().update_layout(title="No shard data available")

    # Convert to DataFrame for plotting
    df = pd.DataFrame(shard_data)

    # Create Gantt chart
    fig = px.timeline(
        df,
        x_start="Start",
        x_end="End",
        y="Shard",
        color="Size_MB",
        hover_data=["Messages", "Size_MB"],
        labels={"Size_MB": "Size (MB)"},
    )

    fig.update_layout(
        title="Shard Date Ranges",
        xaxis_title="Time Period",
        yaxis_title="Shard",
        xaxis=dict(title_standoff=25, tickangle=45, tickfont_size=10),
        margin=dict(l=50, r=20, t=50, b=50),
        height=300,
    )

    return fig


@app.callback(
    Output("message-distribution", "figure"), [Input("shards-data-store", "children")]
)
def update_message_distribution(shards_json):
    """Update the message distribution visualization."""
    shards_info = json.loads(shards_json)

    if not shards_info.get("using_shards", False):
        return go.Figure().update_layout(title="Not using sharded database")

    # Prepare data for bar chart
    shard_names = []
    message_counts = []
    shard_sizes = []

    for shard in shards_info.get("shards", []):
        shard_name = os.path.basename(shard["path"])

        # Extract year from shard name if possible, otherwise use full name
        if "_" in shard_name:
            try:
                year = shard_name.split("_")[1][:4]  # Get first part of the date (year)
                shard_names.append(year)
            except:
                shard_names.append(shard_name)
        else:
            shard_names.append(shard_name)

        message_counts.append(shard.get("message_count", 0))
        shard_sizes.append(shard.get("size_mb", 0))

    # Create figure with dual axes
    fig = go.Figure()

    # Add message count bars
    fig.add_trace(
        go.Bar(
            x=shard_names,
            y=message_counts,
            name="Message Count",
            marker_color="royalblue",
        )
    )

    # Add size line
    fig.add_trace(
        go.Scatter(
            x=shard_names,
            y=shard_sizes,
            name="Size (MB)",
            marker_color="firebrick",
            mode="lines+markers",
            yaxis="y2",
        )
    )

    # Set up layout with dual y-axes
    fig.update_layout(
        title="Message and Size Distribution Across Shards",
        xaxis=dict(title="Shard", titlefont_size=12, tickfont_size=12),
        yaxis=dict(
            title="Message Count",
            titlefont=dict(color="royalblue"),
            tickfont=dict(color="royalblue"),
            titlefont_size=12,
            tickfont_size=12,
        ),
        yaxis2=dict(
            title="Size (MB)",
            titlefont=dict(color="firebrick"),
            tickfont=dict(color="firebrick"),
            anchor="x",
            overlaying="y",
            side="right",
            titlefont_size=12,
            tickfont_size=12,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=60, t=50, b=50),
        height=300,
    )

    return fig


@app.callback(
    Output("query-table", "children"), [Input("interval-component", "n_intervals")]
)
def update_query_table(n_intervals):
    """Update the query history table."""
    # In a production system, we'd implement query logging in the shard manager
    # For now, we'll create sample data

    if not query_history:
        for i in range(5):
            query_type = ["get_messages", "search", "count", "recent"][i % 4]
            shard_id = f"shard_202{i}_01.db"
            duration = round(10 + i * 5 + (i % 3) * 15, 2)
            timestamp = (datetime.now() - timedelta(minutes=i * 2)).strftime("%H:%M:%S")

            query_history.append(
                {
                    "query_type": query_type,
                    "shard": shard_id,
                    "duration_ms": duration,
                    "timestamp": timestamp,
                    "status": "success" if i % 5 != 0 else "error",
                }
            )

    # Create table
    table_header = html.Thead(
        html.Tr(
            [
                html.Th("Time"),
                html.Th("Query Type"),
                html.Th("Shard"),
                html.Th("Duration"),
                html.Th("Status"),
            ]
        )
    )

    rows = []
    for query in query_history[:10]:  # Show only the most recent 10
        status_class = (
            "status-error" if query["status"] == "error" else "status-success"
        )
        status = html.Span(query["status"].capitalize(), className=status_class)

        row = html.Tr(
            [
                html.Td(query["timestamp"]),
                html.Td(query["query_type"]),
                html.Td(query["shard"]),
                html.Td(f"{query['duration_ms']} ms"),
                html.Td(status),
            ]
        )
        rows.append(row)

    if not rows:
        return html.Div("No query data available")

    table_body = html.Tbody(rows)
    table = html.Table([table_header, table_body], className="data-table")

    return table


@app.callback(
    Output("create-shards-button", "disabled"), [Input("shard-size-input", "value")]
)
def validate_shard_size(value):
    """Validate shard size input and disable button if invalid."""
    if value is None or value < 1 or value > 24:
        return True
    return False


@app.callback(
    Output("shard-creation-status", "children"),
    [Input("create-shards-button", "n_clicks")],
    [State("shard-size-input", "value")],
)
def handle_create_shards(n_clicks, shard_size):
    """Handle shard creation button click."""
    if not n_clicks:
        return ""

    if not sharded_db:
        return "Database not initialized"

    if shard_size < 1 or shard_size > 24:
        return "Invalid shard size (must be 1-24 months)"

    # In a real implementation, we would call the shard creation function
    # But here we'll just return a message
    return "Shard creation would start here. This would execute in the background."


@app.callback(
    Output("maintenance-status", "children"),
    [
        Input("optimize-shard-button", "n_clicks"),
        Input("rebuild-indexes-button", "n_clicks"),
    ],
    [State("shard-dropdown", "value")],
)
def handle_maintenance(optimize_clicks, rebuild_clicks, selected_shard):
    """Handle maintenance button clicks."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return ""

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if not selected_shard:
        return "Please select a shard first"

    if button_id == "optimize-shard-button":
        # In a real implementation, we would call the optimize function
        return f"Optimization of {os.path.basename(selected_shard)} would start here"
    elif button_id == "rebuild-indexes-button":
        # In a real implementation, we would call the rebuild indexes function
        return (
            f"Rebuilding indexes of {os.path.basename(selected_shard)} would start here"
        )

    return ""


@app.callback(
    Output("status-message", "children"), [Input("optimize-button", "n_clicks")]
)
def handle_optimize(n_clicks):
    """Handle optimize button click."""
    if not n_clicks:
        return ""

    if not sharded_db:
        return "Database not initialized"

    # In a real implementation, we would run optimization on all shards
    return "Global optimization would start here. This would execute in the background."


async def start_background_tasks():
    """Start the background tasks."""
    global background_task
    background_task = asyncio.create_task(background_metrics_collection())


def run_server(host="127.0.0.1", port=8050, debug=False, db_path=None, shards_dir=None):
    """Run the Dash server."""
    global sharded_db

    # Normalize paths
    if db_path:
        db_path = os.path.abspath(os.path.expanduser(db_path))
    else:
        db_path = DEFAULT_DB_PATH

    if shards_dir:
        shards_dir = os.path.abspath(os.path.expanduser(shards_dir))
    else:
        shards_dir = DEFAULT_SHARDS_DIR

    # Initialize database
    logger.info(f"Initializing database: {db_path}")
    logger.info(f"Shards directory: {shards_dir}")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # Initialize database and start background tasks
        loop.run_until_complete(initialize_database(db_path, shards_dir))
        loop.run_until_complete(start_background_tasks())

        # Start server
        logger.info(f"Starting dashboard server at http://{host}:{port}")
        app.run_server(host=host, port=port, debug=debug)
    finally:
        # Clean up
        if background_task:
            background_task.cancel()
        if sharded_db:
            loop.run_until_complete(sharded_db.cleanup())
        loop.close()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Database Sharding Monitoring Dashboard"
    )

    parser.add_argument(
        "--host", default="127.0.0.1", help="Host address to bind the server to"
    )
    parser.add_argument(
        "--port", type=int, default=8050, help="Port to run the server on"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run the server in debug mode"
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help=f"Path to the iMessage database (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--shards-dir",
        default=None,
        help=f"Directory containing database shards (default: {DEFAULT_SHARDS_DIR})",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    print(
        f"""
=======================================================================
iMessage Database Sharding Dashboard
=======================================================================

Starting dashboard server at http://{args.host}:{args.port}

Use Ctrl+C to stop the server
    """
    )

    run_server(
        host=args.host,
        port=args.port,
        debug=args.debug,
        db_path=args.db_path,
        shards_dir=args.shards_dir,
    )
