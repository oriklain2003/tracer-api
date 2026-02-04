#!/usr/bin/env python3
"""
Create Admin User Script
Creates a new admin user from the command line
"""
import argparse
import getpass
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from service.pg_provider import get_pool
from service.user_service import UserService
from service.auth_service import AuthService, AuthConfig
from core.auth_models import UserType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_username(username: str) -> bool:
    """Validate username format"""
    if len(username) < 3 or len(username) > 50:
        logger.error("Username must be between 3 and 50 characters")
        return False
    
    if not username.replace('_', '').replace('-', '').isalnum():
        logger.error("Username must be alphanumeric (can include _ and -)")
        return False
    
    return True


def validate_password(password: str) -> bool:
    """Validate password strength"""
    if len(password) < 8:
        logger.error("Password must be at least 8 characters")
        return False
    
    if len(password) > 100:
        logger.error("Password must be less than 100 characters")
        return False
    
    return True


def create_admin_user(
    username: str,
    password: str,
    email: str = None,
    user_type: UserType = UserType.ADMIN,
    max_active_sessions: int = 10
) -> bool:
    """Create an admin user"""
    try:
        # Initialize services
        pg_pool = get_pool()
        user_service = UserService(pg_pool)
        
        auth_config = AuthConfig(
            jwt_secret_key=os.getenv("JWT_SECRET_KEY", "CHANGE_ME_IN_PRODUCTION")
        )
        auth_service = AuthService(auth_config)
        
        # Check if user already exists
        existing_user = user_service.get_user_by_username(username)
        if existing_user:
            logger.error(f"User '{username}' already exists")
            return False
        
        # Hash password
        password_hash = auth_service.hash_password(password)
        
        # Create user
        user = user_service.create_user(
            username=username,
            email=email,
            password_hash=password_hash,
            user_type=user_type,
            max_active_sessions=max_active_sessions
        )
        
        if user:
            logger.info(f"✓ Admin user created successfully!")
            logger.info(f"  Username: {user.username}")
            logger.info(f"  User Type: {user.user_type.value}")
            logger.info(f"  Email: {user.email or 'N/A'}")
            logger.info(f"  Max Active Sessions: {user.max_active_sessions}")
            return True
        else:
            logger.error("Failed to create user")
            return False
            
    except Exception as e:
        logger.error(f"Error creating admin user: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Create a new admin user for the Tracer API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (prompts for password)
  python scripts/create_admin_user.py --username admin --email admin@example.com
  
  # With password (not recommended for production)
  python scripts/create_admin_user.py --username admin --password MySecurePass123
  
  # Create super_user instead of admin
  python scripts/create_admin_user.py --username superuser --user-type super_user
        """
    )
    
    parser.add_argument(
        '--username',
        type=str,
        required=True,
        help='Username for the admin user (3-50 alphanumeric characters)'
    )
    
    parser.add_argument(
        '--password',
        type=str,
        help='Password for the admin user (8-100 characters). If not provided, will prompt securely.'
    )
    
    parser.add_argument(
        '--email',
        type=str,
        help='Email address for the admin user (optional)'
    )
    
    parser.add_argument(
        '--user-type',
        type=str,
        choices=['admin', 'super_user', 'user'],
        default='admin',
        help='User type (default: admin)'
    )
    
    parser.add_argument(
        '--max-sessions',
        type=int,
        default=10,
        help='Maximum number of active sessions (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Validate username
    if not validate_username(args.username):
        sys.exit(1)
    
    # Get password
    password = args.password
    if not password:
        print("\nPassword not provided. Please enter password securely:")
        password = getpass.getpass("Password: ")
        password_confirm = getpass.getpass("Confirm password: ")
        
        if password != password_confirm:
            logger.error("Passwords do not match")
            sys.exit(1)
    
    # Validate password
    if not validate_password(password):
        sys.exit(1)
    
    # Convert user type string to enum
    user_type = UserType(args.user_type)
    
    # Create user
    logger.info(f"Creating {user_type.value} user: {args.username}")
    
    success = create_admin_user(
        username=args.username,
        password=password,
        email=args.email,
        user_type=user_type,
        max_active_sessions=args.max_sessions
    )
    
    if success:
        print("\n✓ User created successfully!")
        print(f"\nYou can now login with:")
        print(f"  Username: {args.username}")
        print(f"  Password: <the password you entered>")
        sys.exit(0)
    else:
        print("\n✗ Failed to create user. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
