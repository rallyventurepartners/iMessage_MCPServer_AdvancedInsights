#!/usr/bin/env python3
"""
macOS Full Disk Access Permission Checker

This script checks if Terminal/VS Code has the necessary permissions 
to access the iMessage database and provides instructions for granting
Full Disk Access if needed.
"""

import argparse
import os
import platform
import sqlite3
import subprocess
import sys
from pathlib import Path


def print_header(text):
    """Print a formatted header text."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)

def print_step(number, text):
    """Print a step in the instructions."""
    print(f"\n{number}. {text}")

def check_macos_version():
    """Check if the script is running on macOS."""
    if platform.system() != "Darwin":
        print("This script is only for macOS systems.")
        sys.exit(1)

    version = platform.mac_ver()[0]
    print(f"- macOS Version: {version}")

    # Check if macOS version supports the security features we're checking
    major_version = int(version.split('.')[0])
    if major_version < 10:
        print("Your macOS version may not have the Security & Privacy features we're checking.")
        print("Please manually ensure Terminal or VS Code has Full Disk Access.")

    return True

def check_imessage_db():
    """Check if the iMessage database exists."""
    home = os.path.expanduser("~")
    db_path = Path(f"{home}/Library/Messages/chat.db")

    if not os.path.exists(db_path):
        print(f"- iMessage database not found at: {db_path}")
        print("  This path may be different on your system or iMessage might not be set up.")
        return False, db_path

    print(f"- iMessage database found at: {db_path}")
    print(f"- Database size: {os.path.getsize(db_path) / (1024 * 1024):.2f} MB")

    return True, db_path

def check_db_access(db_path):
    """Try to access the database to verify permissions."""
    try:
        # Try to connect with sqlite3
        conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)

        # Try to execute a simple query
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
        row = cursor.fetchone()

        if row:
            print(f"- Successfully accessed database. Found table: {row[0]}")
            conn.close()
            return True
        else:
            print("- Connected to database but found no tables. This is unusual.")
            conn.close()
            return False

    except sqlite3.OperationalError as e:
        if "unable to open database file" in str(e):
            print("- ERROR: Cannot access database file due to permissions.")
            return False
        else:
            print(f"- ERROR: {e}")
            return False
    except Exception as e:
        print(f"- ERROR: Unexpected error: {e}")
        return False

def print_permission_instructions():
    """Print instructions for granting Full Disk Access permission."""
    print_header("INSTRUCTIONS: GRANTING FULL DISK ACCESS")

    print("\nTo grant Full Disk Access to Terminal/VS Code, follow these steps:")

    print_step(1, "Open System Preferences")
    print("   - Click the Apple menu (🍎) → System Preferences/Settings")

    print_step(2, "Open Security & Privacy settings")
    print("   - Click on 'Security & Privacy' or 'Privacy & Security'")

    print_step(3, "Access Privacy settings")
    print("   - Click on the 'Privacy' tab")
    print("   - Scroll down and select 'Full Disk Access' from the left sidebar")

    print_step(4, "Unlock settings if needed")
    print("   - Click the lock icon in the bottom left (if locked)")
    print("   - Enter your administrator password when prompted")

    print_step(5, "Add Terminal/VS Code to the allowed applications")
    print("   - Click the '+' button")
    print("   - Navigate to Applications → Utilities → Terminal (or VS Code)")
    print("   - Click 'Open' to add it to the list")

    print_step(6, "Ensure the checkbox is selected")
    print("   - Make sure the checkbox next to Terminal/VS Code is checked")

    print_step(7, "Restart Terminal/VS Code")
    print("   - Close and reopen Terminal/VS Code for the changes to take effect")

    print("\nAfter completing these steps, run this script again to verify access.")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Check macOS permissions for iMessage database access")
    parser.add_argument('--verbose', action='store_true', help='Show more detailed information')
    args = parser.parse_args()

    print_header("iMessage Database Permission Checker")

    # Check macOS version
    check_macos_version()

    # Check if iMessage database exists
    db_exists, db_path = check_imessage_db()
    if not db_exists:
        print("\nThe iMessage database wasn't found. If you believe this is an error, please check:")
        print("1. If iMessage is set up on this Mac")
        print("2. If the database is in a different location")
        sys.exit(1)

    # Check if we can access the database
    has_access = check_db_access(db_path)

    if has_access:
        print_header("SUCCESS: Full Disk Access Permission Confirmed")
        print("\nYour Terminal/VS Code has the necessary permissions to access the iMessage database.")
        print("You can now run the iMessage MCP Server Advanced Insights tools.")

        # If on a newer macOS, mention potential additional privacy prompts
        if platform.mac_ver()[0].split('.')[0] >= '10':
            print("\nNOTE: When running the application, macOS may still show additional privacy prompts.")
            print("Please accept these prompts to allow full functionality.")
    else:
        print_header("ERROR: Full Disk Access Permission Required")
        print("\nThe application cannot access the iMessage database.")
        print("Terminal/VS Code needs Full Disk Access permission in macOS Security & Privacy settings.")

        # Print instructions
        print_permission_instructions()

    # If verbose, print process info
    if args.verbose and not has_access:
        print_header("VERBOSE: Process Information")
        try:
            process_name = os.path.basename(sys.executable)
            print(f"- Process executable: {sys.executable}")
            print(f"- Process name: {process_name}")
            print(f"- Process ID: {os.getpid()}")
            print(f"- Parent process ID: {os.getppid()}")

            # Try to get the parent process name
            if platform.system() == "Darwin":  # macOS
                try:
                    ps_output = subprocess.check_output(["ps", "-p", str(os.getppid()), "-o", "comm="]).decode().strip()
                    print(f"- Parent process name: {ps_output}")
                except:
                    pass
        except Exception as e:
            print(f"Error getting process info: {e}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
