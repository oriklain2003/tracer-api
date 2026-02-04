"""
Authentication Service
Handles password hashing, JWT token generation, and session management
"""
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from uuid import UUID

import bcrypt
from jose import JWTError, jwt

from core.auth_models import UserType, AuthenticatedUser


class AuthConfig:
    """Authentication configuration"""
    def __init__(
        self,
        jwt_secret_key: str = "CHANGE_ME_IN_PRODUCTION",
        jwt_algorithm: str = "HS256",
        access_token_expire_minutes: int = 15,
        refresh_token_expire_days: int = 7,
        session_expire_hours: int = 24,
        session_inactivity_timeout_minutes: int = 120,
    ):
        self.jwt_secret_key = jwt_secret_key
        self.jwt_algorithm = jwt_algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        self.session_expire_hours = session_expire_hours
        self.session_inactivity_timeout_minutes = session_inactivity_timeout_minutes


class AuthService:
    """Service for authentication operations"""

    def __init__(self, config: AuthConfig):
        self.config = config

    # ========================================================================
    # Password Management
    # ========================================================================

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt"""
        password_bytes = password.encode('utf-8')
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password_bytes, salt)
        return hashed.decode('utf-8')

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        password_bytes = plain_password.encode('utf-8')
        hashed_bytes = hashed_password.encode('utf-8')
        return bcrypt.checkpw(password_bytes, hashed_bytes)

    # ========================================================================
    # JWT Token Management
    # ========================================================================

    def create_access_token(
        self,
        user_id: UUID,
        username: str,
        user_type: UserType,
        permissions: list[str],
        session_id: Optional[UUID] = None,
    ) -> str:
        """Create a JWT access token. If session_id is set, the token is bound to that session."""
        expire = datetime.utcnow() + timedelta(minutes=self.config.access_token_expire_minutes)

        payload = {
            "sub": str(user_id),
            "username": username,
            "user_type": user_type.value,
            "permissions": permissions,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access",
        }
        if session_id is not None:
            payload["session_id"] = str(session_id)

        return jwt.encode(payload, self.config.jwt_secret_key, algorithm=self.config.jwt_algorithm)

    def create_refresh_token(self, user_id: UUID) -> tuple[str, str]:
        """
        Create a refresh token and return (token, token_hash)
        The token is given to the user, token_hash is stored in DB
        """
        expire = datetime.utcnow() + timedelta(days=self.config.refresh_token_expire_days)
        
        # Generate a random token
        raw_token = secrets.token_urlsafe(32)
        
        # Create JWT with the random token as subject
        payload = {
            "sub": str(user_id),
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh",
            "jti": raw_token  # JWT ID
        }
        
        token = jwt.encode(payload, self.config.jwt_secret_key, algorithm=self.config.jwt_algorithm)
        
        # Hash the token for storage
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        
        return token, token_hash

    def verify_access_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode an access token
        Returns payload if valid, None otherwise
        """
        try:
            payload = jwt.decode(
                token, 
                self.config.jwt_secret_key, 
                algorithms=[self.config.jwt_algorithm]
            )
            
            # Verify token type
            if payload.get("type") != "access":
                return None
            
            return payload
        except JWTError:
            return None

    def verify_refresh_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode a refresh token
        Returns payload if valid, None otherwise
        """
        try:
            payload = jwt.decode(
                token, 
                self.config.jwt_secret_key, 
                algorithms=[self.config.jwt_algorithm]
            )
            
            # Verify token type
            if payload.get("type") != "refresh":
                return None
            
            return payload
        except JWTError:
            return None

    @staticmethod
    def hash_refresh_token(token: str) -> str:
        """Hash a refresh token for storage"""
        return hashlib.sha256(token.encode()).hexdigest()

    # ========================================================================
    # Session Token Management
    # ========================================================================

    @staticmethod
    def generate_session_token() -> str:
        """Generate a secure random session token"""
        return secrets.token_urlsafe(48)

    def calculate_session_expiry(self) -> datetime:
        """Calculate session expiration time"""
        return datetime.utcnow() + timedelta(hours=self.config.session_expire_hours)

    def is_session_expired(self, expires_at: datetime, last_activity: datetime) -> bool:
        """
        Check if a session is expired based on:
        1. Absolute expiration time
        2. Inactivity timeout
        """
        now = datetime.now(timezone.utc)
        # Normalize DB datetimes to UTC-aware for comparison (if naive, treat as UTC)
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if last_activity.tzinfo is None:
            last_activity = last_activity.replace(tzinfo=timezone.utc)

        # Check absolute expiration
        if now > expires_at:
            return True
        
        # Check inactivity timeout
        inactivity_limit = timedelta(minutes=self.config.session_inactivity_timeout_minutes)
        if now - last_activity > inactivity_limit:
            return True
        
        return False

    # ========================================================================
    # Token Extraction
    # ========================================================================

    @staticmethod
    def extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
        """Extract token from Authorization header"""
        if not authorization:
            return None
        
        parts = authorization.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None
        
        return parts[1]

    # ========================================================================
    # User Authentication Helper
    # ========================================================================

    def create_authenticated_user(
        self,
        user_id: UUID,
        username: str,
        email: Optional[str],
        user_type: UserType,
        is_blocked: bool,
        permissions: list[str]
    ) -> AuthenticatedUser:
        """Create an AuthenticatedUser instance"""
        return AuthenticatedUser(
            id=user_id,
            username=username,
            email=email,
            user_type=user_type,
            is_blocked=is_blocked,
            permissions=permissions
        )

    # ========================================================================
    # Token Expiry Info
    # ========================================================================

    def get_access_token_expiry_seconds(self) -> int:
        """Get access token expiry in seconds"""
        return self.config.access_token_expire_minutes * 60

    def get_refresh_token_expiry(self) -> datetime:
        """Get refresh token expiry datetime"""
        return datetime.utcnow() + timedelta(days=self.config.refresh_token_expire_days)
