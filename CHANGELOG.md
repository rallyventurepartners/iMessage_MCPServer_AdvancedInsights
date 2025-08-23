# Changelog

All notable changes to the iMessage Advanced Insights MCP Server will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-23

### Initial Release

First public release of iMessage Advanced Insights - an MCP Server for Claude Desktop that provides AI-powered conversational analytics for macOS iMessage data.

### Features

#### Core Functionality
- **Privacy-First Design**: All analysis happens locally, no data leaves your machine
- **Consent Management**: Explicit user consent required before accessing any data
- **Read-Only Access**: Database operations are strictly read-only for safety

#### Analysis Tools (25 total)
- **Relationship Intelligence**: Deep analysis of communication patterns and dynamics
- **Conversation Quality Scoring**: Multi-dimensional conversation analysis
- **Group Dynamics Analyzer**: Comprehensive group chat analysis with influence networks
- **Predictive Engagement**: ML-powered predictions for response times and activity
- **Communication Patterns**: Cadence analysis, sentiment evolution, response times
- **Advanced Analytics**: Anomaly detection, network intelligence, best contact times
- **Cloud-Aware Tools**: Handle iCloud-stored messages intelligently

#### Enhanced Visualizations
- Time series charts with matplotlib/seaborn
- Communication heatmaps by day/hour
- Balance charts showing conversation dynamics
- Relationship dashboards with multiple metrics
- Support for extended time periods (up to 36 months)
- Multi-contact comparison capabilities

#### Technology Stack
- Python 3.11+ with async/await
- Model Context Protocol (MCP) for Claude integration
- SQLite for local data access
- Machine Learning with scikit-learn
- Visualization with matplotlib and seaborn
- Network analysis with NetworkX

### Created By
David Jelinek / Rally Venture Partners / rvp.io

### License
MIT License - see [LICENSE](LICENSE) for details.