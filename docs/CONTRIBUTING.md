# Contributing to iMessage Advanced Insights

Thank you for your interest in contributing to iMessage Advanced Insights! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Coding Standards](#coding-standards)
- [Git Workflow](#git-workflow)
- [Pull Request Process](#pull-request-process)
- [Testing](#testing)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up your development environment (see below)
4. Create a feature branch for your changes
5. Make your changes and test them
6. Push your branch to your fork
7. Open a pull request from your branch to the main repository

## Development Environment

### Prerequisites

- Python 3.10 or higher
- macOS with iMessage enabled (for most functionality)
- Git

### Setup

1. Clone your fork of the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/iMessage_MCPServer_AdvancedInsights.git
   cd iMessage_MCPServer_AdvancedInsights
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

3. Set up pre-commit hooks (recommended):
   ```bash
   pre-commit install
   ```

## Coding Standards

We follow a consistent coding style to maintain code quality and readability:

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use 4 spaces for indentation (no tabs)
- Maximum line length of 100 characters
- Use type hints in all new code
- Use docstrings for all modules, classes, and functions (Google style)

### Linting and Formatting

We use the following tools for code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Code quality checks
- **mypy**: Static type checking

You can run these checks locally:

```bash
# Format code
black .
isort .

# Check code quality
flake8 .
mypy .
```

## Git Workflow

### Branching Model

- `main` branch is the stable branch that always reflects the latest release
- Feature branches should be created from `main` and named descriptively

### Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

Types include:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code
- `refactor`: Code changes that neither fix a bug nor add a feature
- `perf`: Code changes that improve performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools

Example:
```
feat(database): implement time-based sharding for large databases

This adds support for automatically sharding extremely large iMessage
databases by time periods, improving performance for databases over 10GB.

Closes #123
```

## Pull Request Process

1. Ensure your code follows our coding standards and passes all tests
2. Update documentation if necessary
3. Fill out the pull request template completely
4. Request a review from at least one maintainer
5. Address any feedback from reviewers
6. Once approved, a maintainer will merge your PR

### Pull Request Template

When opening a pull request, please use the following template:

```markdown
## Description
[Describe the changes you've made]

## Related Issue
[Link to the issue this PR addresses, if applicable]

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code cleanup or refactor
- [ ] Other (please describe):

## How Has This Been Tested?
[Describe the tests you ran to verify your changes]

## Checklist
- [ ] My code follows the coding standards of this project
- [ ] I have added tests that prove my fix or feature works
- [ ] I have updated the documentation accordingly
- [ ] I have checked that my changes don't introduce new warnings
- [ ] I have added docstrings and type hints for new code
```

## Testing

We use pytest for testing. All new code should include appropriate tests:

- Unit tests for specific functions and classes
- Integration tests for features that interact with multiple components
- End-to-end tests for complete workflows
- MCP tool tests for verifying Model Context Protocol integrations

To run tests:

```bash
# Run all tests
python -m unittest discover

# Run specific test file
python -m unittest test_improvements.py

# Run specific test class
python -m unittest test_improvements.TestIMessageImprovements

# Run specific test method
python -m unittest test_improvements.TestIMessageImprovements.test_message_formatter

# Test database sharding functionality
python -m unittest test_database_sharding.py

# Test MCP prompts and tools
python test_improvements.py TestMCPFunctionality
```

## Documentation

Good documentation is crucial for this project:

- Use clear, descriptive docstrings for all functions, classes, and modules
- Update markdown documentation when adding or changing features
- Add examples for new functionality
- Keep the README up to date

For major features, add or update the relevant documentation files in the repository.

## MCP Development Guidelines

When developing MCP (Model Context Protocol) components:

### MCP Tools

- Each tool function must include comprehensive docstrings
- All tools should have proper error handling with descriptive error messages
- Tools should return structured data (dictionaries) with consistent keys
- Implement parameter validation for all tool functions
- Consider adding the tool to an appropriate prompt's suggested_tools list

Example tool function:

```python
@mcp.tool
def analyze_contact_tool(contact_name: str) -> Dict[str, Any]:
    """
    Analyze communication patterns with a specific contact.
    
    Args:
        contact_name: The name of the contact to analyze
        
    Returns:
        Dict containing analysis results with the following keys:
        - contact_info: Basic information about the contact
        - message_count: Number of messages exchanged
        - sentiment: Overall sentiment analysis
        - common_topics: List of frequently discussed topics
        - response_time: Average response time analysis
    """
    # Implementation...
```

### MCP Prompts

- Organize prompts into logical categories
- Include emoji indicators for visual organization
- Add suggested_tools parameters to all prompts
- Use natural language explanations for all capabilities
- Keep prompts concise but descriptive

Example prompt function:

```python
@mcp.prompt(name="contact_insight")
def contact_insight_prompt() -> Dict[str, Any]:
    """Generate a prompt for analyzing communication with a contact."""
    return {
        "content": """
I can help analyze your communication with a specific contact.

👤 **Contact Analysis**
I can show you message patterns, sentiment trends, and communication insights.

📊 **Communication Metrics**
I can calculate response times, message frequency, and conversation patterns.

💬 **Popular Topics**
I can identify frequently discussed topics and key conversation themes.

To get started, tell me which contact you'd like to analyze.
""",
        "suggested_tools": [
            "analyze_contact_tool",
            "list_recent_contacts_tool",
            "find_contact_by_phone_tool"
        ]
    }
```

### MCP Resources

- Use RESTful-style URIs for all resources
- Implement proper pagination for large result sets
- Include metadata in all resource responses
- Add comprehensive documentation for each resource endpoint
- Use consistent data structures across related resources

Example resource implementation:

```python
@mcp.resource("/contacts/{contact_id}")
async def get_contact_details(contact_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific contact.
    
    Args:
        contact_id: The ID of the contact to retrieve
        
    Returns:
        Dict containing contact details with metadata
    """
    # Implementation...
```

## Community

- Join discussions in the Issues section
- Help answer questions from other users
- Provide feedback on proposed features
- Report bugs when you find them

Thank you for your contributions! Your efforts help make this project better for everyone.