# Claude Desktop Integration Guide

This guide will help you set up the iMessage Advanced Insights server for use with Claude Desktop, allowing Claude to analyze and visualize your iMessage data.

## Prerequisites

- Claude Desktop application installed
- Python 3.8 or higher

## Quick Start

1. **Generate the Claude Desktop configuration**

   ```bash
   python3 generate_claude_config.py
   ```

   This will create a file called `claude_desktop_config.json` in the current directory.

2. **Import the configuration into Claude Desktop**

   - Open Claude Desktop
   - Click on the Settings icon
   - Navigate to the "Advanced" section
   - Click "Import MCP Server Configuration"
   - Select the `claude_desktop_config.json` file

3. **Enable the MCP Server in Claude Desktop**

   - After importing, you should see "iMessage Advanced Insights" in the MCP Servers list
   - Make sure it's enabled by toggling the switch
   - Claude Desktop will start the server automatically when needed

4. **Start a new conversation with Claude**

   You can now ask Claude to analyze your iMessage data. Try questions like:
   - "Show me my most frequent contacts in iMessage"
   - "Analyze the sentiment in my conversations with [contact name]"
   - "What's the pattern of my messaging activity over time?"
   - "Show me a visualization of my messaging network"

## Advanced Configuration

If you need to customize the configuration, you can use these options:

```bash
python3 generate_claude_config.py --port 8080 --server-path /absolute/path/to/main_async.py --output custom_config.json
```

Options:
- `--port`: The port number where the server will run (default: 5001)
- `--server-path`: The absolute path to the main_async.py file (default: automatically detected)
- `--output`: The output file path for the configuration (default: claude_desktop_config.json)

## Troubleshooting

If Claude is unable to access your iMessage data, check the following:

1. **Server Startup**: Check if Claude Desktop successfully started the server. Look for any errors in the Claude Desktop logs.
2. **Port Conflicts**: If port 5001 is already in use, try generating a config with a different port.
3. **File Permissions**: Ensure the main_async.py file has execute permissions.
4. **Python Installation**: Make sure Python 3.8+ is installed and accessible in your PATH.

## Security Considerations

- The server processes your iMessage data locally on your Mac.
- All data stays on your device and is not sent to external servers.
- The server only runs when requested by Claude Desktop and shuts down when not in use.
- For added security, the server only listens on localhost.

## Updating the Configuration

If you make changes to the server or its location, regenerate the configuration file and re-import it into Claude Desktop:

```bash
python3 generate_claude_config.py
```

## Example Questions to Ask Claude

Once the MCP server is set up, you can ask Claude questions like:

1. "Who do I message with most frequently?"
2. "How many group chats am I part of?"
3. "What's the sentiment of my conversations with [name]?"
4. "When am I most active in messaging?"
5. "What does my social network look like based on iMessage?"
6. "How has my messaging pattern changed over the past few months?"
7. "Who are the most active participants in my 'Family' group chat?"
8. "What time of day do I usually message with [name]?"

Claude will analyze your iMessage data and provide insights based on your questions. 