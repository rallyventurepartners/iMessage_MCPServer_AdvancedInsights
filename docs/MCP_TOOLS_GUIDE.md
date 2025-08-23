# MCP Tools Guide - For Claude Integration

This guide explains how Claude uses the MCP tools to provide insights about iMessage conversations. Each tool is designed to work seamlessly with Claude's natural language understanding.

## 🎯 Core Philosophy

These tools are designed for Claude to:
1. **Understand Context**: Claude interprets your natural language requests
2. **Select Tools**: Claude automatically chooses the right tools
3. **Combine Insights**: Claude may use multiple tools to answer one question
4. **Provide Synthesis**: Claude adds interpretation and recommendations

## 📚 Tool Categories

### 1. Consent Management Tools

These tools manage privacy and access control.

#### `request_consent`
- **Purpose**: Grant Claude permission to access iMessage data
- **Claude Uses This When**: You ask to analyze messages for the first time
- **Example Interaction**:
  ```
  You: "Analyze my messages"
  Claude: "I need your consent to access iMessage data. May I have permission for 24 hours?"
  ```

#### `check_consent`
- **Purpose**: Verify if Claude has permission to access data
- **Claude Uses This When**: Before any data access operation
- **Automatic**: Claude checks this behind the scenes

#### `get_access_log`
- **Purpose**: Show what data Claude has accessed
- **Example Request**: "What iMessage data have you looked at?"

### 2. Relationship Intelligence Tools

These provide deep insights into specific relationships.

#### `analyze_conversation_intelligence`
- **Purpose**: Comprehensive analysis of conversation quality and dynamics
- **Claude Uses This For**:
  - Relationship health assessments
  - Communication pattern analysis
  - Emotional dynamics evaluation
- **Example Questions**:
  - "How healthy is my relationship with John?"
  - "Analyze the quality of my conversations with my family"
  - "What's the emotional dynamic with my best friend?"

#### `analyze_relationship_trajectory`
- **Purpose**: Track how relationships evolve over time
- **Claude Uses This For**:
  - Identifying relationship changes
  - Predicting future patterns
  - Detecting relationship phases
- **Example Questions**:
  - "How has my friendship with Sarah changed this year?"
  - "Are any of my relationships weakening?"
  - "Show me relationships that are growing stronger"

#### `profile_communication_style`
- **Purpose**: Analyze communication preferences and patterns
- **Claude Uses This For**:
  - Understanding your communication style
  - Comparing styles across relationships
  - Identifying adaptation patterns
- **Example Questions**:
  - "What's my communication style?"
  - "How do I communicate differently with friends vs family?"
  - "Am I adapting too much to others' styles?"

### 3. Message Analysis Tools

These analyze message content and patterns.

#### `get_messages`
- **Purpose**: Retrieve messages for analysis
- **Claude Uses This For**: Getting raw data for deeper analysis
- **Usually Hidden**: Claude uses this behind the scenes

#### `search_messages`
- **Purpose**: Find specific messages or patterns
- **Claude Uses This For**:
  - Finding conversations about specific topics
  - Locating important moments
  - Semantic search across all messages
- **Example Questions**:
  - "When did I last talk about vacation plans?"
  - "Find conversations where I discussed my career"
  - "Show me messages where I was supporting someone"

#### `analyze_conversation_topics`
- **Purpose**: Extract and analyze conversation themes
- **Claude Uses This For**:
  - Understanding what you talk about
  - Tracking topic evolution
  - Identifying conversation patterns
- **Example Questions**:
  - "What do I mainly talk about with my mom?"
  - "How have my conversation topics changed over time?"
  - "What topics dominate my work conversations?"

### 4. Network Intelligence Tools

These analyze your overall communication network.

#### `analyze_social_network`
- **Purpose**: Map and analyze your entire communication network
- **Claude Uses This For**:
  - Understanding social connections
  - Identifying key relationships
  - Finding communication patterns
- **Example Questions**:
  - "Who are the key people in my network?"
  - "How is my social network structured?"
  - "Who connects different parts of my social circle?"

#### `identify_key_connectors`
- **Purpose**: Find people who bridge different groups
- **Claude Uses This For**:
  - Understanding social dynamics
  - Identifying influential connections
- **Example Questions**:
  - "Who connects my work and personal life?"
  - "Which friends know each other through me?"

### 5. Contact Management Tools

These handle contact identification and information retrieval.

#### `get_contacts`
- **Purpose**: List all contacts with real names and activity metrics
- **Claude Uses This For**:
  - Understanding your communication network
  - Identifying active relationships
  - Finding specific people
- **Key Features**:
  - Shows real names from macOS Contacts
  - Displays phone numbers and emails unmasked
  - Includes message counts and last contact date

#### `search_contacts`
- **Purpose**: Find specific contacts by name or identifier
- **Claude Uses This For**:
  - Locating people by partial names
  - Finding contacts when you're unsure of exact spelling
  - Discovering who a phone number belongs to
- **Example Questions**:
  - "Who is 415-555-1234?"
  - "Find contacts named John"
  - "Show me sarah@email.com's info"

#### Contact Identification Features
Claude now intelligently handles contacts:
- **Flexible Formats**: 
  - Names: "John", "Sarah Smith", "Dr. Johnson"
  - Phones: "+1-212-555-1234", "(212) 555-1234", "2125551234", "212.555.1234"
  - Emails: "john@example.com", "work@company.org"
- **Smart Matching**: Finds contacts regardless of format used
- **Real Names**: Shows actual names instead of "Contact for +1234567890"
- **Group Names**: Displays "John, Sarah, Mike +2" instead of "chat123456"

### 6. Predictive Analytics Tools

These forecast future patterns and detect anomalies.

#### `predict_communication_patterns`
- **Purpose**: Forecast future communication needs
- **Claude Uses This For**:
  - Predicting message volumes
  - Identifying upcoming communication needs
  - Suggesting optimal contact times
- **Example Questions**:
  - "When should I reach out to John?"
  - "What are my communication patterns likely to be next month?"
  - "Who might I lose touch with soon?"

#### `detect_anomalies`
- **Purpose**: Identify unusual patterns or changes
- **Claude Uses This For**:
  - Spotting concerning silences
  - Detecting behavior changes
  - Finding communication disruptions
- **Example Questions**:
  - "Are there any unusual patterns in my messages?"
  - "Has anyone gone unusually quiet?"
  - "What communication changes should I be aware of?"

### 6. Life Event Detection Tools

These identify significant events from conversation patterns.

#### `detect_life_events`
- **Purpose**: Identify major life changes from messages
- **Claude Uses This For**:
  - Spotting job changes, moves, relationship changes
  - Understanding life transitions
  - Providing contextual support
- **Example Questions**:
  - "What major events have happened in my network?"
  - "Has anyone mentioned significant life changes?"
  - "Who might be going through a transition?"

#### `track_emotional_milestones`
- **Purpose**: Identify emotional highs and lows
- **Claude Uses This For**:
  - Understanding emotional patterns
  - Identifying support needs
  - Tracking wellbeing
- **Example Questions**:
  - "Who in my network might need support?"
  - "What emotional patterns do you see?"
  - "How is my network's overall emotional health?"

### 7. Insight Generation Tools

These create comprehensive reports and visualizations.

#### `generate_insights_report`
- **Purpose**: Create detailed analysis reports
- **Claude Uses This For**:
  - Providing comprehensive overviews
  - Summarizing patterns over time
  - Creating actionable recommendations
- **Example Questions**:
  - "Give me a monthly summary of my communications"
  - "Create a report on my relationship health"
  - "Summarize my communication patterns this year"

#### `visualize_message_network`
- **Purpose**: Create visual representations of data
- **Claude Uses This For**:
  - Showing network structures
  - Illustrating patterns
  - Making complex data understandable
- **Example Questions**:
  - "Show me a visualization of my social network"
  - "Create a graph of my message frequency"
  - "Visualize how my relationships have changed"

## 🎮 How Claude Combines Tools

Claude often uses multiple tools to answer complex questions:

### Example: "How is my relationship with Mom?"

Claude might use:
1. `check_consent` - Verify permission
2. `get_messages` - Retrieve conversation history
3. `analyze_conversation_intelligence` - Assess relationship quality
4. `analyze_relationship_trajectory` - Check changes over time
5. `analyze_conversation_topics` - Understand themes
6. `track_emotional_milestones` - Identify emotional patterns

Then synthesize all results into a coherent, insightful response.

## 💡 Best Practices for Queries

### Be Natural
```
Good: "How are my friendships doing?"
Also Good: "Analyze friendship health"
```

### Specify Timeframes When Relevant
```
"How have things changed with John this year?"
"Analyze last month's communication patterns"
```

### Ask for Recommendations
```
"What should I do to improve my relationship with Sarah?"
"How can I maintain my long-distance friendships better?"
```

### Request Specific Insights
```
"Focus on emotional support patterns"
"Look for signs of stress in my network"
"Analyze professional relationships only"
```

## 🔐 Privacy Notes

- All tools respect consent boundaries
- Sensitive data is automatically sanitized
- No data is stored beyond the session
- Claude only accesses what's needed for your query

## 🚀 Advanced Usage

### Comparative Analysis
"Compare my communication style between work and personal contexts"

### Trend Analysis
"Show me how my communication patterns have evolved over the past year"

### Predictive Insights
"Based on patterns, what relationships need attention next month?"

### Network Effects
"How do changes in one relationship affect others in my network?"

---

Remember: You don't need to know these tool names. Just ask Claude naturally about your communications, and Claude will use the appropriate tools to provide insights.