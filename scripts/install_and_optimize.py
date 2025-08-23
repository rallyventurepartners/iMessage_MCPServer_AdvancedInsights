#!/usr/bin/env python3
"""
Installation and Optimization Script for iMessage Advanced Insights MCP Server.

This script:
1. Installs all dependencies
2. Creates the indexed database copy for optimal performance
3. Configures settings for Claude Desktop integration
4. Sets up the server for automatic start

Usage:
    python install_and_optimize.py [options]

Options:
    --skip-indexing     Skip the database indexing step
    --skip-install      Skip dependency installation (use with existing installations)
    --claude-only       Only configure Claude Desktop integration
    --custom-db PATH    Specify a custom database path
    --verbose           Show detailed logging information
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Define default paths
HOME = os.path.expanduser("~")
DEFAULT_DB_PATH = Path(f"{HOME}/Library/Messages/chat.db")
DEFAULT_INDEX_PATH = Path(f"{HOME}/.imessage_insights/indexed_chat.db")


# Set up logging
def setup_logging(verbose=False):
    """Configure logging with appropriate verbosity."""
    log_level = logging.DEBUG if verbose else logging.INFO

    # Ensure log directory exists
    log_dir = Path(f"{HOME}/.imessage_insights")
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{log_dir}/install.log", mode="a"),
        ],
    )

    # Create a logger for this script
    return logging.getLogger("iMessage-Install")


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80))
    print("=" * 80 + "\n")


def install_dependencies(logger):
    """Install required Python dependencies."""
    print_header("Installing Dependencies")
    logger.info("Installing Python dependencies...")

    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            check=True,
            stdout=subprocess.PIPE,
        )

        # Install requirements
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True,
        )

        logger.info("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing dependencies: {e}")
        return False


def optimize_database(logger, db_path=None, skip_prompt=False):
    """Create an optimized indexed copy of the database."""
    print_header("Database Optimization")
    logger.info("Creating optimized database with indexes...")

    try:
        # Build the command with appropriate options
        cmd = [
            sys.executable,
            "index_imessage_db.py",
            "--read-only",  # Create a separate copy
            "--verbose",  # Show detailed logging
        ]

        if db_path:
            cmd.extend(["--db-path", str(db_path)])

        if skip_prompt:
            # We need to provide input for the confirmation prompt
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = process.communicate(input="y\n")
            print(stdout)
            if process.returncode != 0:
                logger.error(f"Error optimizing database: {stderr}")
                return False
        else:
            # Let the user interact with the prompts directly
            subprocess.run(cmd, check=True)

        logger.info("Database optimization completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error optimizing database: {e}")
        return False


def configure_claude_desktop(logger):
    """Configure integration with Claude Desktop."""
    print_header("Claude Desktop Integration")
    logger.info("Configuring Claude Desktop integration...")

    try:
        # Run the configuration script
        subprocess.run([sys.executable, "update_claude_config.py"], check=True)

        logger.info("Claude Desktop integration configured successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error configuring Claude Desktop: {e}")
        return False
    except FileNotFoundError:
        logger.error("update_claude_config.py script not found")
        return False


def create_startup_script(logger):
    """Create a script for easy server startup."""
    print_header("Creating Startup Script")

    startup_script_path = Path(f"{HOME}/.imessage_insights/start_server.sh")
    startup_content = f"""#!/bin/bash
# Auto-generated startup script for iMessage Advanced Insights MCP Server

# Activate environment if available
if [ -d "{os.getcwd()}/venv" ]; then
    source "{os.getcwd()}/venv/bin/activate"
fi

# Start the server with the optimized database
cd "{os.getcwd()}"
python mcp_server_modular.py --db-path "{DEFAULT_INDEX_PATH}"

# Return to the original directory when done
cd - > /dev/null
"""

    try:
        # Create the directory if it doesn't exist
        startup_script_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the script
        with open(startup_script_path, "w") as f:
            f.write(startup_content)

        # Make it executable
        os.chmod(startup_script_path, 0o755)

        logger.info(f"Startup script created at: {startup_script_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating startup script: {e}")
        return False


def main():
    """Main entry point for the installation script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="iMessage Advanced Insights Installation and Optimization"
    )
    parser.add_argument(
        "--skip-indexing", action="store_true", help="Skip the database indexing step"
    )
    parser.add_argument(
        "--skip-install", action="store_true", help="Skip dependency installation"
    )
    parser.add_argument(
        "--claude-only",
        action="store_true",
        help="Only configure Claude Desktop integration",
    )
    parser.add_argument("--custom-db", type=str, help="Specify a custom database path")
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed logging information"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.verbose)

    # Print welcome message
    print_header("iMessage Advanced Insights Installation")
    print(
        "This script will install and optimize the iMessage Advanced Insights MCP Server."
    )
    print("Steps to be performed:")
    if not args.skip_install:
        print("1. Install Python dependencies")
    if not args.skip_indexing and not args.claude_only:
        print("2. Create optimized database indexes (for better performance)")
    if not args.claude_only:
        print("3. Create startup script")
    print("4. Configure Claude Desktop integration")
    print("\n")

    # Confirm proceeding
    if not args.claude_only:
        confirm = input("Proceed with installation? [y/N]: ")
        if confirm.lower() not in ("y", "yes"):
            print("Installation cancelled.")
            return 0

    start_time = time.time()
    success = True

    # Only Claude Desktop configuration
    if args.claude_only:
        success = configure_claude_desktop(logger)
    else:
        # Install dependencies if not skipped
        if not args.skip_install:
            success = install_dependencies(logger) and success

        # Optimize database if not skipped
        if not args.skip_indexing and success:
            success = (
                optimize_database(logger, args.custom_db, skip_prompt=False) and success
            )

        # Create startup script
        if success:
            success = create_startup_script(logger) and success

        # Configure Claude Desktop
        if success:
            success = configure_claude_desktop(logger) and success

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Print completion message
    print_header("Installation " + ("Completed" if success else "Failed"))

    if success:
        print(f"Installation completed successfully in {elapsed_time:.2f} seconds!")
        print("\nTo start the server with optimized database:")
        print(f"  {HOME}/.imessage_insights/start_server.sh")
        print("\nOr manually with:")
        print(f"  python mcp_server_modular.py --db-path {DEFAULT_INDEX_PATH}")
        print("\nTo use with Claude Desktop:")
        print("  1. Open Claude Desktop")
        print("  2. Go to Settings > MCP (Advanced)")
        print("  3. Ensure the iMessage Advanced Insights server is enabled")
    else:
        print("Installation encountered errors. Please check the logs and try again.")
        print(f"Log file: {HOME}/.imessage_insights/install.log")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
