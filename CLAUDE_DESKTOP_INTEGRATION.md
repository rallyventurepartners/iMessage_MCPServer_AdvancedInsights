# Claude Desktop Integration Guide

This guide will help you set up the iMessage Advanced Insights server for use with Claude Desktop, allowing Claude to analyze and visualize your iMessage data.

## Prerequisites

- Claude Desktop application installed
- iMessage Advanced Insights server set up and running
- Python 3.8 or higher

## Quick Start

1. **Start the iMessage Advanced Insights server**

   ```bash
   python3 main_async.py --port 5001
   ```

2. **Generate the Claude Desktop configuration**

   ```bash
   python3 generate_claude_config.py
   ```

   This will create a file called `claude_desktop_config.json` in the current directory.

3. **Import the configuration into Claude Desktop**

   - Open Claude Desktop
   - Click on the Settings icon
   - Navigate to the "Tools" section
   - Click "Import Tools Configuration"
   - Select the `claude_desktop_config.json` file

4. **Start a new conversation with Claude and use the tools**

   You can now ask Claude to analyze your iMessage data, and it will have access to the following tools:
   - iMessage_contacts
   - iMessage_group_chats
   - iMessage_analyze_contact
   - iMessage_analyze_group_chat
   - iMessage_analyze_network
   - iMessage_visualize_network
   - iMessage_analyze_sentiment
   - iMessage_query

## Advanced Configuration

If you're running the server on a different host or port, you can customize the configuration generator:

```bash
python3 generate_claude_config.py --host 192.168.1.100 --port 8080 --output my_config.json
```

Options:
- `--host`: The hostname or IP address where the server is running (default: localhost)
- `--port`: The port number where the server is running (default: 5001)
- `--output`: The output file path for the configuration (default: claude_desktop_config.json)

## Example Usage

Once you've set everything up, you can ask Claude questions like:

1. "Show me my most frequently messaged contacts."
2. "What are my active group chats?"
3. "Analyze my conversations with [contact] over the last month."
4. "Show me the sentiment analysis for my 'Family' group chat."
5. "Generate a visualization of my messaging network."
6. "What days of the week do I message [contact] the most?"

Claude will use the appropriate tool to fetch and analyze your iMessage data.

## Troubleshooting

If Claude is unable to access the tools, check the following:

1. **Server Running**: Make sure the iMessage Advanced Insights server is running.
2. **Correct Host/Port**: Verify that the host and port in the configuration match where the server is running.
3. **Firewall Settings**: Ensure your firewall is not blocking the connection.
4. **Tool Configuration**: Verify that the tools were properly imported in Claude Desktop.

## Security Considerations

- The server processes your iMessage data locally and does not send it to external services.
- Claude will have access to your iMessage data through the API endpoints.
- For added security, run the server on localhost only (the default setting) to prevent network access.

## Updating the Configuration

If you make changes to the server or its endpoints, regenerate the configuration file and re-import it into Claude Desktop.

```bash
python3 generate_claude_config.py
```

## Customizing Tool Descriptions

If you want to customize how Claude uses these tools, you can edit the tool descriptions in the `generate_claude_config.py` script before generating the configuration file. 