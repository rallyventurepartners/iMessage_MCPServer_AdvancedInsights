# Archive Directory

This directory contains deprecated files from the iMessage Advanced Insights MCP Server that were removed during the cleanup and refactoring process.

## Contents

### deprecated_src/
Contains the old `src/` directory structure that was replaced by the new `mcp_server/` implementation. These files include:
- Old MCP tools implementations that were never used
- Database implementations that were superseded
- Utility functions that were duplicated

### old_docs/
Contains outdated documentation and audit files that are no longer relevant to the current implementation.

## Note
These files are kept for historical reference only. The current implementation is in the `mcp_server/` directory with all tools implemented directly in `main.py`.

**Date Archived**: 2025-08-19