"""
Authentication Middleware
FastAPI dependencies for JWT and session-based authentication
"""
import logging
from typing import Optional
from uuid import UUID

from fastapi import Depends, HTTPException, status, Header, Cookie
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from core.auth_models import AuthenticatedUser, UserType
from service.auth_service import AuthService
from service.user_service import UserService

logger = logging.getLogger(__name__)

# HTTP Bearer token scheme
bearer_scheme = HTTPBearer(auto_error=False)


class AuthMiddleware:
    """Authentication middleware with dependency injection"""

    def __init__(self, auth_service: AuthService, user_service: UserService):
        self.auth_service = auth_service
        self.user_service = user_service

    # ========================================================================
    # JWT Authentication
    # ========================================================================

    async def get_current_user_jwt(
        self,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)
    ) -> AuthenticatedUser:
        """
        Dependency to get current user from JWT token
        Validates JWT from Authorization: Bearer <token> header
        """
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = credentials.credentials

        # Verify JWT token
        payload = self.auth_service.verify_access_token(token)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Extract user_id from payload
        user_id_str = payload.get("sub")
        if not user_id_str:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
                headers={"WWW-Authenticate": "Bearer"},
            )

        try:
            user_id = UUID(user_id_str)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid user ID in token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # If JWT is bound to a session, require that session to still be active
        # (so evicted sessions get 401 on next request and user is sent to login)
        session_id_str = payload.get("session_id")
        if session_id_str:
            try:
                session_id = UUID(session_id_str)
                if not self.user_service.is_session_active(session_id):
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Session was closed (e.g. logged in elsewhere). Please sign in again.",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
            except (ValueError, TypeError):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid session in token",
                    headers={"WWW-Authenticate": "Bearer"},
                )

        # Get authenticated user with permissions
        auth_user = self.user_service.get_authenticated_user(user_id)
        if not auth_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Check if user is blocked
        if auth_user.is_blocked:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is blocked"
            )

        return auth_user

    # ========================================================================
    # Session Authentication
    # ========================================================================

    async def get_current_user_session(
        self,
        session_token: Optional[str] = Cookie(None, alias="session_token"),
        x_session_token: Optional[str] = Header(None, alias="X-Session-Token")
    ) -> AuthenticatedUser:
        """
        Dependency to get current user from session token
        Checks cookie or X-Session-Token header
        """
        # Try cookie first, then header
        token = session_token or x_session_token

        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing session token"
            )

        # Get session from database
        session = self.user_service.get_session_by_token(token)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired session"
            )

        # Check if session is expired
        if self.auth_service.is_session_expired(session.expires_at, session.last_activity):
            # Invalidate expired session
            self.user_service.invalidate_session(session.id)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session expired"
            )

        # Update session activity
        self.user_service.update_session_activity(session.id)

        # Get authenticated user
        auth_user = self.user_service.get_authenticated_user(session.user_id)
        if not auth_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )

        # Check if user is blocked
        if auth_user.is_blocked:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is blocked"
            )

        return auth_user

    # ========================================================================
    # Flexible Authentication (JWT or Session)
    # ========================================================================

    async def get_authenticated_user(
        self,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
        session_token: Optional[str] = Cookie(None, alias="session_token"),
        x_session_token: Optional[str] = Header(None, alias="X-Session-Token")
    ) -> AuthenticatedUser:
        """
        Dependency that accepts either JWT or session authentication
        Tries JWT first, then falls back to session
        """
        # Try JWT first
        if credentials:
            try:
                return await self.get_current_user_jwt(credentials)
            except HTTPException:
                pass  # Fall through to session auth

        # Try session auth
        token = session_token or x_session_token
        if token:
            try:
                return await self.get_current_user_session(session_token=session_token, x_session_token=x_session_token)
            except HTTPException:
                pass

        # No valid authentication found
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # ========================================================================
    # Permission Checking
    # ========================================================================

    def require_permission(self, permission: str):
        """
        Dependency factory to check if user has a specific permission
        Usage: Depends(auth_middleware.require_permission("feedback.create"))
        """
        async def permission_checker(
            user: AuthenticatedUser = Depends(self.get_authenticated_user)
        ) -> AuthenticatedUser:
            if not user.has_permission(permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission denied: {permission} required"
                )
            return user

        return permission_checker

    def require_role(self, min_role: UserType):
        """
        Dependency factory to check if user meets minimum role requirement
        Usage: Depends(auth_middleware.require_role(UserType.ADMIN))
        """
        async def role_checker(
            user: AuthenticatedUser = Depends(self.get_authenticated_user)
        ) -> AuthenticatedUser:
            if not user.has_role(min_role):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role {min_role.value} or higher required"
                )
            return user

        return role_checker

    # ========================================================================
    # Optional Authentication
    # ========================================================================

    async def get_current_user_optional(
        self,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
        session_token: Optional[str] = Cookie(None, alias="session_token"),
        x_session_token: Optional[str] = Header(None, alias="X-Session-Token")
    ) -> Optional[AuthenticatedUser]:
        """
        Dependency for optional authentication
        Returns user if authenticated, None otherwise (no exception raised)
        """
        try:
            return await self.get_authenticated_user(credentials, session_token, x_session_token)
        except HTTPException:
            return None
