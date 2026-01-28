#!/usr/bin/env python3
"""
Environment Setup and Validation Script

This script helps validate that all required environment variables are properly configured
before running the service. It also provides helpful error messages if anything is missing.

Usage:
    python setup_env.py           # Validate environment
    python setup_env.py --check   # Check and exit (for CI/CD)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ANSI color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{BLUE}{'=' * 70}{RESET}")
    print(f"{BLUE}{text.center(70)}{RESET}")
    print(f"{BLUE}{'=' * 70}{RESET}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{GREEN}✓{RESET} {text}")


def print_error(text: str):
    """Print error message."""
    print(f"{RED}✗{RESET} {text}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{YELLOW}⚠{RESET} {text}")


def check_env_file() -> bool:
    """Check if .env file exists."""
    env_file = Path(".env")
    env_example = Path("env.example")
    
    if not env_file.exists():
        print_error(".env file not found!")
        print("\n  To create .env file:")
        print("  1. Copy the template: cp env.example .env")
        print("  2. Edit .env and fill in your credentials")
        return False
    
    print_success(".env file found")
    
    # Check if env.example exists
    if not env_example.exists():
        print_warning("env.example template not found (optional)")
    
    return True


def load_env_file():
    """Load environment variables from .env file."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print_success("Environment variables loaded from .env")
    except ImportError:
        print_error("python-dotenv not installed!")
        print("  Install it with: pip install python-dotenv")
        sys.exit(1)


def validate_required_vars() -> Tuple[bool, List[str]]:
    """Validate all required environment variables."""
    required_vars = {
        "FR24_API_TOKEN": "FlightRadar24 API token",
        "OPENAI_API_KEY": "OpenAI API key",
        "GEMINI_API_KEY": "Google Gemini API key",
        "POSTGRES_DSN": "PostgreSQL connection string",
        "AVIATION_EDGE_API_KEY": "Aviation Edge API key"
    }
    
    missing_vars = []
    
    for var_name, description in required_vars.items():
        value = os.getenv(var_name)
        if not value:
            print_error(f"{var_name}: NOT SET ({description})")
            missing_vars.append(var_name)
        else:
            # Mask sensitive values in output
            masked_value = value[:8] + "..." if len(value) > 8 else "***"
            print_success(f"{var_name}: {masked_value}")
    
    return len(missing_vars) == 0, missing_vars


def validate_optional_vars():
    """Validate optional environment variables and show defaults."""
    optional_vars = {
        "API_HOST": ("0.0.0.0", "API server host"),
        "API_PORT": ("8000", "API server port"),
        "PG_POOL_MIN_CONNECTIONS": ("2", "Min PostgreSQL connections"),
        "PG_POOL_MAX_CONNECTIONS": ("10", "Max PostgreSQL connections"),
        "PG_CONNECT_TIMEOUT": ("10", "Connection timeout (seconds)"),
        "PG_STATEMENT_TIMEOUT": ("30000", "Query timeout (milliseconds)")
    }
    
    for var_name, (default, description) in optional_vars.items():
        value = os.getenv(var_name, default)
        if os.getenv(var_name):
            print_success(f"{var_name}: {value} ({description})")
        else:
            print_warning(f"{var_name}: {value} (using default - {description})")


def validate_postgres_dsn():
    """Validate PostgreSQL connection string format."""
    dsn = os.getenv("POSTGRES_DSN")
    if not dsn:
        return False
    
    if not dsn.startswith("postgresql://"):
        print_error("POSTGRES_DSN should start with 'postgresql://'")
        return False
    
    # Try to parse DSN
    try:
        from urllib.parse import urlparse
        result = urlparse(dsn)
        
        if not result.hostname:
            print_error("POSTGRES_DSN: Missing hostname")
            return False
        if not result.username:
            print_error("POSTGRES_DSN: Missing username")
            return False
        if not result.password:
            print_error("POSTGRES_DSN: Missing password")
            return False
        
        print_success(f"PostgreSQL DSN format valid (host: {result.hostname})")
        return True
    except Exception as e:
        print_error(f"POSTGRES_DSN: Invalid format - {str(e)}")
        return False


def test_database_connection():
    """Test PostgreSQL database connection."""
    try:
        import psycopg2
        dsn = os.getenv("POSTGRES_DSN")
        
        print("\n  Testing database connection...")
        conn = psycopg2.connect(dsn, connect_timeout=5)
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        
        print_success(f"Database connection successful!")
        print(f"    PostgreSQL version: {version.split(',')[0]}")
        return True
    except ImportError:
        print_warning("psycopg2 not installed - skipping connection test")
        return True
    except Exception as e:
        print_error(f"Database connection failed: {str(e)}")
        print("\n  Troubleshooting:")
        print("  1. Check if database is accessible from your network")
        print("  2. Verify credentials are correct")
        print("  3. Check security group rules (for AWS RDS)")
        return False


def check_dependencies():
    """Check if all required Python packages are installed."""
    required_packages = [
        ("fastapi", "FastAPI web framework"),
        ("uvicorn", "ASGI web server"),
        ("psycopg2", "PostgreSQL adapter"),
        ("openai", "OpenAI API client"),
        ("dotenv", "Environment variable loader"),
    ]
    
    missing_packages = []
    
    for package, description in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print_success(f"{package}: Installed ({description})")
        except ImportError:
            print_error(f"{package}: NOT INSTALLED ({description})")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n  Install missing packages:")
        print(f"  pip install -r requirements.txt")
        return False
    
    return True


def main():
    """Main validation function."""
    print_header("Environment Setup Validation")
    
    # Check for --check flag (non-interactive mode for CI/CD)
    check_only = "--check" in sys.argv
    
    all_valid = True
    
    # Step 1: Check .env file
    print(f"\n{BLUE}[1/6] Checking .env file...{RESET}")
    if not check_env_file():
        all_valid = False
        if check_only:
            sys.exit(1)
    
    # Step 2: Load environment variables
    print(f"\n{BLUE}[2/6] Loading environment variables...{RESET}")
    load_env_file()
    
    # Step 3: Validate required variables
    print(f"\n{BLUE}[3/6] Validating required environment variables...{RESET}")
    vars_valid, missing_vars = validate_required_vars()
    if not vars_valid:
        all_valid = False
        print(f"\n  Missing {len(missing_vars)} required variable(s):")
        for var in missing_vars:
            print(f"    - {var}")
        print("\n  Edit your .env file and add the missing variables")
    
    # Step 4: Validate optional variables
    print(f"\n{BLUE}[4/6] Checking optional environment variables...{RESET}")
    validate_optional_vars()
    
    # Step 5: Validate PostgreSQL DSN
    print(f"\n{BLUE}[5/6] Validating PostgreSQL connection...{RESET}")
    if not validate_postgres_dsn():
        all_valid = False
    else:
        # Test actual connection (optional, can be slow)
        if not check_only:
            if not test_database_connection():
                all_valid = False
    
    # Step 6: Check dependencies
    print(f"\n{BLUE}[6/6] Checking Python dependencies...{RESET}")
    if not check_dependencies():
        all_valid = False
    
    # Final result
    print_header("Validation Results")
    
    if all_valid:
        print_success("All checks passed! ✓")
        print(f"\n{GREEN}Your environment is properly configured.{RESET}")
        print(f"\nYou can now start the service with:")
        print(f"  python app.py")
        return 0
    else:
        print_error("Some checks failed! ✗")
        print(f"\n{RED}Please fix the issues above before starting the service.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
