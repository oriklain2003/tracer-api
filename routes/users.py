"""
User Management and Authentication Router
Handles user registration, login, logout, and user management
"""
import logging
import os
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Cookie, Depends, Header, HTTPException, Request, status, Response
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials
import psycopg2.extras

from core.auth_models import (
    UserCreate, UserUpdate, UserResponse, LoginRequest, LoginResponse,
    TokenRefreshRequest, TokenRefreshResponse, PermissionGrant,
    PermissionResponse, SessionResponse, UserListResponse,
    UserType, AuthenticatedUser, ChangePasswordRequest
)
from service.auth_service import AuthService, AuthConfig
from service.user_service import UserService
from service.pg_provider import PostgreSQLConnectionPool
from api.middleware.auth import AuthMiddleware, bearer_scheme

logger = logging.getLogger(__name__)

# Module-level variables for dependency injection
_auth_service: Optional[AuthService] = None
_user_service: Optional[UserService] = None
_auth_middleware: Optional[AuthMiddleware] = None

# Create router
router = APIRouter(prefix="/api", tags=["authentication", "users"])


def configure(pg_pool: PostgreSQLConnectionPool):
    """Configure the router with required dependencies"""
    global _auth_service, _user_service, _auth_middleware

    # Initialize auth config from environment
    auth_config = AuthConfig(
        jwt_secret_key=os.getenv("JWT_SECRET_KEY", "CHANGE_ME_IN_PRODUCTION"),
        jwt_algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
        access_token_expire_minutes=int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "15")),
        refresh_token_expire_days=int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "7")),
        session_expire_hours=int(os.getenv("SESSION_EXPIRE_HOURS", "24")),
        session_inactivity_timeout_minutes=int(os.getenv("SESSION_INACTIVITY_TIMEOUT_MINUTES", "120")),
    )

    _auth_service = AuthService(auth_config)
    _user_service = UserService(pg_pool)
    _auth_middleware = AuthMiddleware(_auth_service, _user_service)

    logger.info("Users router configured")


def get_auth_service() -> AuthService:
    """Dependency to get auth service"""
    if _auth_service is None:
        raise RuntimeError("Router not configured. Call configure() first.")
    return _auth_service


def get_user_service() -> UserService:
    """Dependency to get user service"""
    if _user_service is None:
        raise RuntimeError("Router not configured. Call configure() first.")
    return _user_service


def get_auth_middleware() -> AuthMiddleware:
    """Dependency to get auth middleware"""
    if _auth_middleware is None:
        raise RuntimeError("Router not configured. Call configure() first.")
    return _auth_middleware


async def get_current_user(
    auth_middleware: AuthMiddleware = Depends(get_auth_middleware),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    session_token: Optional[str] = Cookie(None, alias="session_token"),
    x_session_token: Optional[str] = Header(None, alias="X-Session-Token"),
) -> AuthenticatedUser:
    """Dependency that returns the authenticated user (JWT or session)."""
    return await auth_middleware.get_authenticated_user(
        credentials, session_token, x_session_token
    )


def require_role(min_role: UserType):
    """Dependency factory: require current user to have at least min_role."""

    async def _check(
        current_user: AuthenticatedUser = Depends(get_current_user),
    ) -> AuthenticatedUser:
        has_it = current_user.has_role(min_role)
        logger.info(
            "require_role: user_id=%s username=%s user_type=%s min_role=%s has_role=%s",
            current_user.id,
            current_user.username,
            current_user.user_type.value,
            min_role.value,
            has_it,
        )
        if not has_it:
            logger.warning(
                "403 Forbidden: user %s (type=%s) does not have required role %s or higher",
                current_user.username,
                current_user.user_type.value,
                min_role.value,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role {min_role.value} or higher required",
            )
        return current_user

    return _check


# ============================================================================
# Public Authentication Endpoints
# ============================================================================

@router.post("/auth/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserCreate,
    auth_service: AuthService = Depends(get_auth_service),
    user_service: UserService = Depends(get_user_service)
):
    """
    Register a new user
    Note: In production, you may want to restrict this to admin-only
    """
    # Hash password
    password_hash = auth_service.hash_password(user_data.password)

    # Create user
    user = user_service.create_user(
        username=user_data.username,
        email=user_data.email,
        password_hash=password_hash,
        user_type=user_data.user_type,
        max_active_sessions=user_data.max_active_sessions
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already exists"
        )

    logger.info(f"User registered: {user.username} (id: {user.id})")

    # Get permissions for response
    permissions = user_service.get_user_permissions(user.id)

    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        user_type=user.user_type,
        is_blocked=user.is_blocked,
        max_active_sessions=user.max_active_sessions,
        created_at=user.created_at,
        updated_at=user.updated_at,
        last_login_at=user.last_login_at,
        permissions=permissions
    )


def _client_ip_from_request(request: Request) -> str:
    """Get client IP from request (X-Forwarded-For when behind proxy, else request.client)."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host or ""
    return ""


@router.post("/auth/login", response_model=LoginResponse)
async def login(
    request: Request,
    login_data: LoginRequest,
    response: Response,
    auth_service: AuthService = Depends(get_auth_service),
    user_service: UserService = Depends(get_user_service)
):
    """Login with username and password. Client IP is taken from the request (not from body) for accuracy."""
    # Get user
    user = user_service.get_user_by_username(login_data.username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )

    # Verify password
    if not auth_service.verify_password(login_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )

    # Check if user is blocked
    if user.is_blocked:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is blocked"
        )

    # Get user permissions
    permissions = user_service.get_user_permissions(user.id)

    # Enforce session limit: invalidate oldest sessions so we stay at or below max_active_sessions
    user_service.enforce_session_limit(user.id, user.max_active_sessions)
    # Revoke oldest refresh tokens so evicted sessions cannot refresh and get sent to login
    user_service.revoke_excess_refresh_tokens(user.id, user.max_active_sessions)

    # Create session (IP from request so we don't trust client-sent data)
    client_ip = _client_ip_from_request(request)
    session_token = auth_service.generate_session_token()
    session_expires_at = auth_service.calculate_session_expiry()
    session = user_service.create_session(
        user_id=user.id,
        token=session_token,
        expires_at=session_expires_at,
        ip_address=client_ip or None,
        user_agent=login_data.user_agent
    )

    if not session:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create session"
        )

    # Create JWT tokens (bind access token to this session so evicted clients get 401)
    access_token = auth_service.create_access_token(
        user_id=user.id,
        username=user.username,
        user_type=user.user_type,
        permissions=permissions,
        session_id=session.id,
    )

    refresh_token, refresh_token_hash = auth_service.create_refresh_token(user.id)

    # Store refresh token
    user_service.store_refresh_token(
        user_id=user.id,
        token_hash=refresh_token_hash,
        expires_at=auth_service.get_refresh_token_expiry()
    )

    # Update last login
    user_service.update_last_login(user.id)

    # Set session cookie
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        max_age=auth_service.config.session_expire_hours * 3600,
        samesite="lax"
    )

    logger.info(f"User logged in: {user.username} (session: {session.id})")

    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=auth_service.get_access_token_expiry_seconds(),
        user=UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            user_type=user.user_type,
            is_blocked=user.is_blocked,
            max_active_sessions=user.max_active_sessions,
            created_at=user.created_at,
            updated_at=user.updated_at,
            last_login_at=user.last_login_at,
            permissions=permissions
        )
    )


@router.post("/auth/refresh", response_model=TokenRefreshResponse)
async def refresh_access_token(
    refresh_data: TokenRefreshRequest,
    auth_service: AuthService = Depends(get_auth_service),
    user_service: UserService = Depends(get_user_service)
):
    """Refresh access token using refresh token"""
    # Verify refresh token
    payload = auth_service.verify_refresh_token(refresh_data.refresh_token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token"
        )

    # Get token hash
    token_hash = auth_service.hash_refresh_token(refresh_data.refresh_token)

    # Verify token in database
    user_id = user_service.verify_refresh_token(token_hash)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked refresh token"
        )

    # Get user
    user = user_service.get_user_by_id(user_id)
    if not user or user.is_blocked:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or blocked"
        )

    # Get permissions
    permissions = user_service.get_user_permissions(user.id)

    # Create new access token
    access_token = auth_service.create_access_token(
        user_id=user.id,
        username=user.username,
        user_type=user.user_type,
        permissions=permissions
    )

    return TokenRefreshResponse(
        access_token=access_token,
        expires_in=auth_service.get_access_token_expiry_seconds()
    )


# ============================================================================
# Authenticated User Endpoints
# ============================================================================

@router.post("/auth/logout")
async def logout(
    response: Response,
    auth_middleware: AuthMiddleware = Depends(get_auth_middleware),
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """Logout current user (invalidate session and refresh tokens)"""
    user_service = _user_service
    auth_service = _auth_service

    # Invalidate all user's sessions (or just current one - adjust as needed)
    sessions = user_service.list_user_sessions(current_user.id, active_only=True)
    for session in sessions:
        user_service.invalidate_session(session.id)

    # Clear session cookie
    response.delete_cookie("session_token")

    logger.info(f"User logged out: {current_user.username}")

    return {"message": "Logged out successfully"}


@router.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(
    auth_middleware: AuthMiddleware = Depends(get_auth_middleware),
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """Get current user information"""
    user_service = _user_service
    user = user_service.get_user_by_id(current_user.id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        user_type=user.user_type,
        is_blocked=user.is_blocked,
        max_active_sessions=user.max_active_sessions,
        created_at=user.created_at,
        updated_at=user.updated_at,
        last_login_at=user.last_login_at,
        permissions=current_user.permissions
    )


@router.get("/auth/sessions", response_model=List[SessionResponse])
async def get_current_user_sessions(
    auth_middleware: AuthMiddleware = Depends(get_auth_middleware),
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """Get current user's active sessions"""
    user_service = _user_service
    sessions = user_service.list_user_sessions(current_user.id, active_only=True)

    return [
        SessionResponse(
            id=session.id,
            user_id=session.user_id,
            ip_address=session.ip_address,
            user_agent=session.user_agent,
            created_at=session.created_at,
            expires_at=session.expires_at,
            last_activity=session.last_activity,
            is_active=session.is_active
        )
        for session in sessions
    ]


@router.delete("/auth/sessions/{session_id}")
async def delete_current_user_session(
    session_id: UUID,
    user_service: UserService = Depends(get_user_service),
    auth_middleware: AuthMiddleware = Depends(get_auth_middleware),
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """Delete a specific session of the current user. Idempotent: returns 200 if already inactive."""
    # Include inactive so we can return 200 for already-deleted (idempotent)
    all_sessions = user_service.list_user_sessions(current_user.id, active_only=False)
    session_by_id = {str(s.id): s for s in all_sessions}

    if str(session_id) not in session_by_id:
        logger.info(
            "DELETE /auth/sessions/%s: 404 session not found for user_id=%s (user has %s sessions)",
            session_id,
            current_user.id,
            len(all_sessions),
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found. To terminate another user's session use DELETE /api/users/{user_id}/sessions/{session_id}."
        )

    session = session_by_id[str(session_id)]
    if session.is_active:
        user_service.invalidate_session(session_id)
    else:
        logger.debug("DELETE /auth/sessions/%s: already inactive, returning 200", session_id)
    return {"message": "Session deleted successfully"}


@router.post("/auth/change-password")
async def change_password(
    password_data: ChangePasswordRequest,
    auth_service: AuthService = Depends(get_auth_service),
    user_service: UserService = Depends(get_user_service),
    auth_middleware: AuthMiddleware = Depends(get_auth_middleware),
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """Change current user's password"""
    # Get user with password hash
    user = user_service.get_user_by_id(current_user.id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Verify current password
    if not auth_service.verify_password(password_data.current_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    # Hash new password
    new_password_hash = auth_service.hash_password(password_data.new_password)

    # Update password (need to add method to user_service)
    # For now, we'll do it directly
    try:
        with user_service.pg_pool.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE auth.users SET password_hash = %s WHERE id = %s",
                    (new_password_hash, current_user.id)
                )
                conn.commit()
    except Exception as e:
        logger.error(f"Error changing password: {e}")
        raise HTTPException(status_code=500, detail="Failed to change password")

    return {"message": "Password changed successfully"}


# ============================================================================
# Admin User Management Endpoints
# ============================================================================

@router.get("/users", response_model=UserListResponse)
async def list_users(
    page: int = 1,
    page_size: int = 50,
    user_type: Optional[UserType] = None,
    is_blocked: Optional[bool] = None,
    user_service: UserService = Depends(get_user_service),
    auth_middleware: AuthMiddleware = Depends(get_auth_middleware),
    current_user: AuthenticatedUser = Depends(require_role(UserType.SUPER_USER))
):
    """List all users (requires super_user or admin role)"""
    logger.info(
        "GET /api/users: caller=%s (user_type=%s) page=%s page_size=%s",
        current_user.username,
        current_user.user_type.value,
        page,
        page_size,
    )
    users, total = user_service.list_users(page, page_size, user_type, is_blocked)

    user_responses = []
    for user in users:
        permissions = user_service.get_user_permissions(user.id)
        user_responses.append(UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            user_type=user.user_type,
            is_blocked=user.is_blocked,
            max_active_sessions=user.max_active_sessions,
            created_at=user.created_at,
            updated_at=user.updated_at,
            last_login_at=user.last_login_at,
            permissions=permissions
        ))

    return UserListResponse(
        users=user_responses,
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: UUID,
    user_service: UserService = Depends(get_user_service),
    auth_middleware: AuthMiddleware = Depends(get_auth_middleware),
    current_user: AuthenticatedUser = Depends(require_role(UserType.SUPER_USER))
):
    """Get user by ID (requires super_user or admin role)"""
    user = user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    permissions = user_service.get_user_permissions(user.id)

    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        user_type=user.user_type,
        is_blocked=user.is_blocked,
        max_active_sessions=user.max_active_sessions,
        created_at=user.created_at,
        updated_at=user.updated_at,
        last_login_at=user.last_login_at,
        permissions=permissions
    )


@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: UUID,
    user_data: UserUpdate,
    user_service: UserService = Depends(get_user_service),
    auth_middleware: AuthMiddleware = Depends(get_auth_middleware),
    current_user: AuthenticatedUser = Depends(require_role(UserType.ADMIN))
):
    """Update user (requires admin role)"""
    user = user_service.update_user(
        user_id=user_id,
        username=user_data.username,
        email=user_data.email,
        user_type=user_data.user_type,
        max_active_sessions=user_data.max_active_sessions
    )

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    permissions = user_service.get_user_permissions(user.id)

    logger.info(f"User updated: {user.username} by {current_user.username}")

    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        user_type=user.user_type,
        is_blocked=user.is_blocked,
        max_active_sessions=user.max_active_sessions,
        created_at=user.created_at,
        updated_at=user.updated_at,
        last_login_at=user.last_login_at,
        permissions=permissions
    )


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: UUID,
    user_service: UserService = Depends(get_user_service),
    current_user: AuthenticatedUser = Depends(require_role(UserType.ADMIN)),
):
    """Delete a user (requires admin). Cannot delete yourself."""
    if str(user_id) == str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account",
        )
    success = user_service.delete_user(user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    logger.info(f"User {user_id} deleted by {current_user.username}")


@router.post("/users/{user_id}/block")
async def block_user(
    user_id: UUID,
    user_service: UserService = Depends(get_user_service),
    auth_middleware: AuthMiddleware = Depends(get_auth_middleware),
    current_user: AuthenticatedUser = Depends(require_role(UserType.ADMIN))
):
    """Block a user (requires admin role)"""
    success = user_service.block_user(user_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to block user")

    logger.warning(f"User {user_id} blocked by {current_user.username}")
    return {"message": "User blocked successfully"}


@router.post("/users/{user_id}/unblock")
async def unblock_user(
    user_id: UUID,
    user_service: UserService = Depends(get_user_service),
    auth_middleware: AuthMiddleware = Depends(get_auth_middleware),
    current_user: AuthenticatedUser = Depends(require_role(UserType.ADMIN))
):
    """Unblock a user (requires admin role)"""
    success = user_service.unblock_user(user_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to unblock user")

    logger.info(f"User {user_id} unblocked by {current_user.username}")
    return {"message": "User unblocked successfully"}


@router.post("/users/{user_id}/permissions", response_model=PermissionResponse)
async def grant_permission(
    user_id: UUID,
    permission_data: PermissionGrant,
    user_service: UserService = Depends(get_user_service),
    auth_middleware: AuthMiddleware = Depends(get_auth_middleware),
    current_user: AuthenticatedUser = Depends(require_role(UserType.ADMIN))
):
    """Grant permission to user (requires admin role)"""
    permission = user_service.grant_permission(
        user_id=user_id,
        permission=permission_data.permission,
        granted_by=current_user.id
    )

    if not permission:
        raise HTTPException(status_code=500, detail="Failed to grant permission")

    logger.info(f"Permission '{permission_data.permission}' granted to user {user_id} by {current_user.username}")

    return PermissionResponse(
        id=permission.id,
        user_id=permission.user_id,
        permission=permission.permission,
        granted_at=permission.granted_at,
        granted_by=permission.granted_by
    )


@router.delete("/users/{user_id}/permissions/{permission}")
async def revoke_permission(
    user_id: UUID,
    permission: str,
    user_service: UserService = Depends(get_user_service),
    auth_middleware: AuthMiddleware = Depends(get_auth_middleware),
    current_user: AuthenticatedUser = Depends(require_role(UserType.ADMIN))
):
    """Revoke permission from user (requires admin role)"""
    success = user_service.revoke_permission(user_id, permission)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to revoke permission")

    logger.info(f"Permission '{permission}' revoked from user {user_id} by {current_user.username}")
    return {"message": "Permission revoked successfully"}


@router.get("/users/{user_id}/sessions", response_model=List[SessionResponse])
async def get_user_sessions(
    user_id: UUID,
    active_only: bool = True,
    user_service: UserService = Depends(get_user_service),
    auth_middleware: AuthMiddleware = Depends(get_auth_middleware),
    current_user: AuthenticatedUser = Depends(require_role(UserType.ADMIN))
):
    """Get user's sessions (requires admin role)"""
    sessions = user_service.list_user_sessions(user_id, active_only)

    return [
        SessionResponse(
            id=session.id,
            user_id=session.user_id,
            ip_address=session.ip_address,
            user_agent=session.user_agent,
            created_at=session.created_at,
            expires_at=session.expires_at,
            last_activity=session.last_activity,
            is_active=session.is_active
        )
        for session in sessions
    ]


@router.delete("/users/{user_id}/sessions/{session_id}")
async def delete_user_session(
    user_id: UUID,
    session_id: UUID,
    user_service: UserService = Depends(get_user_service),
    auth_middleware: AuthMiddleware = Depends(get_auth_middleware),
    current_user: AuthenticatedUser = Depends(require_role(UserType.ADMIN))
):
    """Terminate user session (requires admin role)"""
    # Verify session belongs to user (normalize to UUID for comparison)
    sessions = user_service.list_user_sessions(user_id, active_only=True)
    session_ids_normalized = {str(s.id) for s in sessions}

    if str(session_id) not in session_ids_normalized:
        raise HTTPException(status_code=404, detail="Session not found")

    user_service.invalidate_session(session_id)
    logger.info(f"Session {session_id} of user {user_id} terminated by {current_user.username}")
    return {"message": "Session terminated successfully"}


@router.get("/admin/sessions")
async def get_all_sessions(
    page: int = 1,
    page_size: int = 50,
    active_only: bool = True,
    user_service: UserService = Depends(get_user_service),
    auth_middleware: AuthMiddleware = Depends(get_auth_middleware),
    current_user: AuthenticatedUser = Depends(require_role(UserType.ADMIN))
):
    """
    Get all sessions across all users (requires admin role)
    Useful for system-wide session monitoring
    """
    try:
        with user_service.pg_pool.get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # Build where clause
                where_clause = "WHERE s.is_active = true" if active_only else ""
                
                # Get total count
                cursor.execute(
                    f"SELECT COUNT(*) as total FROM auth.sessions s {where_clause}"
                )
                total = cursor.fetchone()['total']
                
                # Get paginated results with user info
                offset = (page - 1) * page_size
                cursor.execute(
                    f"""
                    SELECT 
                        s.id,
                        s.user_id,
                        s.token,
                        s.ip_address,
                        s.user_agent,
                        s.created_at,
                        s.expires_at,
                        s.last_activity,
                        s.is_active,
                        u.username,
                        u.email,
                        u.user_type,
                        u.is_blocked
                    FROM auth.sessions s
                    JOIN auth.users u ON s.user_id = u.id
                    {where_clause}
                    ORDER BY s.last_activity DESC
                    LIMIT %s OFFSET %s
                    """,
                    (page_size, offset)
                )
                rows = cursor.fetchall()
                
                sessions = [
                    {
                        "id": row['id'],
                        "user_id": row['user_id'],
                        "username": row['username'],
                        "email": row['email'],
                        "user_type": row['user_type'],
                        "is_user_blocked": row['is_blocked'],
                        "ip_address": row['ip_address'],
                        "user_agent": row['user_agent'],
                        "created_at": row['created_at'],
                        "expires_at": row['expires_at'],
                        "last_activity": row['last_activity'],
                        "is_active": row['is_active']
                    }
                    for row in rows
                ]
                
                return {
                    "sessions": sessions,
                    "total": total,
                    "page": page,
                    "page_size": page_size
                }
                
    except Exception as e:
        logger.error(f"Error fetching all sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch sessions")
