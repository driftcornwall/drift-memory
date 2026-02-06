#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# ///
"""
Pre-Tool-Use Hook - Safety Checks

Prevents dangerous commands from executing:
- Dangerous rm -rf patterns
- .env file access warnings
"""

import sys
import json
import re


def check_dangerous_rm(command):
    """Check for dangerous rm -rf patterns"""
    # Dangerous patterns
    dangerous_patterns = [
        r'rm\s+-rf\s+/',  # rm -rf /
        r'rm\s+-rf\s+~',  # rm -rf ~
        r'rm\s+-rf\s+\*',  # rm -rf *
        r'rm\s+-rf\s+\.',  # rm -rf .
        r'rm\s+-rf\s+\.\./',  # rm -rf ../
        r'sudo\s+rm\s+-rf',  # sudo rm -rf anything
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, command, re.IGNORECASE):
            return True, f"Dangerous command blocked: {pattern}"

    return False, None


def check_env_access(command):
    """Check for .env file access"""
    env_patterns = [
        r'\.env',
        r'credentials',
        r'secrets',
        r'api[_-]?key',
    ]

    for pattern in env_patterns:
        if re.search(pattern, command, re.IGNORECASE):
            return True, f"Warning: Accessing sensitive file ({pattern})"

    return False, None


def main():
    # Read hook input
    try:
        hook_input = json.loads(sys.stdin.read())
        tool_name = hook_input.get("toolName", "")
        tool_parameters = hook_input.get("toolParameters", {})
    except:
        print(json.dumps({"hookSpecificOutput": {"status": "parse_error"}}))
        sys.exit(0)

    # Only check Bash tool
    if tool_name != "Bash":
        sys.exit(0)

    command = tool_parameters.get("command", "")
    if not command:
        sys.exit(0)

    # Check for dangerous rm
    is_dangerous, msg = check_dangerous_rm(command)
    if is_dangerous:
        output = {
            "hookSpecificOutput": {
                "status": "blocked",
                "reason": msg,
                "command": command
            }
        }
        print(json.dumps(output), file=sys.stderr)
        sys.exit(1)  # Block execution

    # Check for .env access
    is_sensitive, msg = check_env_access(command)
    if is_sensitive:
        output = {
            "hookSpecificOutput": {
                "status": "warning",
                "reason": msg,
                "command": command
            }
        }
        print(json.dumps(output), file=sys.stderr)
        # Don't block, just warn
        sys.exit(0)

    # Safe command
    sys.exit(0)


if __name__ == "__main__":
    main()
