"""
User Service
Handles user CRUD operations, permissions management, and session management
"""
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID
import psycopg2.extras

from core.auth_models import UserType, User, Session, Permission, AuthenticatedUser
from service.pg_provider import PostgreSQLConnectionPool

logger = logging.getLogger(__name__)


class UserService:
    """Service for user management operations"""

    def __init__(self, pg_pool: PostgreSQLConnectionPool):
        self.pg_pool = pg_pool

    # ========================================================================
    # User CRUD Operations
    # ========================================================================

    def create_user(
        self,
        username: str,
        email: Optional[str],
        password_hash: str,
        user_type: UserType = UserType.USER,
        max_active_sessions: int = 5
    ) -> Optional[User]:
        """
        Create a new user
        Returns User object if successful, None otherwise
        """
        try:
            with self.pg_pool.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute(
                        """
                        INSERT INTO auth.users (
                            username, email, password_hash, user_type, max_active_sessions
                        ) VALUES (%s, %s, %s, %s, %s)
                        RETURNING id, username, email, password_hash, user_type, 
                                  is_blocked, max_active_sessions, created_at, 
                                  updated_at, last_login_at
                        """,
                        (username, email, password_hash, user_type.value, max_active_sessions)
                    )
                    row = cursor.fetchone()
                    conn.commit()

                    if row:
                        return User(
                            id=row['id'],
                            username=row['username'],
                            email=row['email'],
                            password_hash=row['password_hash'],
                            user_type=UserType(row['user_type']),
                            is_blocked=row['is_blocked'],
                            max_active_sessions=row['max_active_sessions'],
                            created_at=row['created_at'],
                            updated_at=row['updated_at'],
                            last_login_at=row['last_login_at']
                        )
                    return None

        except psycopg2.errors.UniqueViolation:
            logger.warning(f"User with username '{username}' or email '{email}' already exists")
            return None
        except Exception as e:
            logger.error(f"Error creating user: {e}", exc_info=True)
            return None

    def get_user_by_id(self, user_id: UUID) -> Optional[User]:
        """Get user by ID"""
        try:
            with self.pg_pool.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute(
                        """
                        SELECT id, username, email, password_hash, user_type,
                               is_blocked, max_active_sessions, created_at,
                               updated_at, last_login_at
                        FROM auth.users
                        WHERE id = %s
                        """,
                        (str(user_id),)
                    )
                    row = cursor.fetchone()

                    if row:
                        return User(
                            id=row['id'],
                            username=row['username'],
                            email=row['email'],
                            password_hash=row['password_hash'],
                            user_type=UserType(row['user_type']),
                            is_blocked=row['is_blocked'],
                            max_active_sessions=row['max_active_sessions'],
                            created_at=row['created_at'],
                            updated_at=row['updated_at'],
                            last_login_at=row['last_login_at']
                        )
                    return None

        except Exception as e:
            logger.error(f"Error fetching user by ID: {e}", exc_info=True)
            return None

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        try:
            with self.pg_pool.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute(
                        """
                        SELECT id, username, email, password_hash, user_type,
                               is_blocked, max_active_sessions, created_at,
                               updated_at, last_login_at
                        FROM auth.users
                        WHERE username = %s
                        """,
                        (username,)
                    )
                    row = cursor.fetchone()

                    if row:
                        return User(
                            id=row['id'],
                            username=row['username'],
                            email=row['email'],
                            password_hash=row['password_hash'],
                            user_type=UserType(row['user_type']),
                            is_blocked=row['is_blocked'],
                            max_active_sessions=row['max_active_sessions'],
                            created_at=row['created_at'],
                            updated_at=row['updated_at'],
                            last_login_at=row['last_login_at']
                        )
                    return None

        except Exception as e:
            logger.error(f"Error fetching user by username: {e}", exc_info=True)
            return None

    def update_user(
        self,
        user_id: UUID,
        username: Optional[str] = None,
        email: Optional[str] = None,
        user_type: Optional[UserType] = None,
        max_active_sessions: Optional[int] = None
    ) -> Optional[User]:
        """Update user fields"""
        try:
            updates = []
            params = []

            if username is not None:
                updates.append("username = %s")
                params.append(username)
            if email is not None:
                updates.append("email = %s")
                params.append(email)
            if user_type is not None:
                updates.append("user_type = %s")
                params.append(user_type.value)
            if max_active_sessions is not None:
                updates.append("max_active_sessions = %s")
                params.append(max_active_sessions)

            if not updates:
                return self.get_user_by_id(user_id)

            params.append(str(user_id))

            with self.pg_pool.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute(
                        f"""
                        UPDATE auth.users
                        SET {', '.join(updates)}
                        WHERE id = %s
                        RETURNING id, username, email, password_hash, user_type,
                                  is_blocked, max_active_sessions, created_at,
                                  updated_at, last_login_at
                        """,
                        params
                    )
                    row = cursor.fetchone()
                    conn.commit()

                    if row:
                        return User(
                            id=row['id'],
                            username=row['username'],
                            email=row['email'],
                            password_hash=row['password_hash'],
                            user_type=UserType(row['user_type']),
                            is_blocked=row['is_blocked'],
                            max_active_sessions=row['max_active_sessions'],
                            created_at=row['created_at'],
                            updated_at=row['updated_at'],
                            last_login_at=row['last_login_at']
                        )
                    return None

        except Exception as e:
            logger.error(f"Error updating user: {e}", exc_info=True)
            return None

    def delete_user(self, user_id: UUID) -> bool:
        """Delete a user. Cascades to sessions, permissions, refresh_tokens."""
        try:
            with self.pg_pool.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("DELETE FROM auth.users WHERE id = %s", (str(user_id),))
                    conn.commit()
                    return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error deleting user: {e}", exc_info=True)
            return False

    def update_last_login(self, user_id: UUID) -> bool:
        """Update user's last login timestamp"""
        try:
            with self.pg_pool.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "UPDATE auth.users SET last_login_at = NOW() WHERE id = %s",
                        (str(user_id),)
                    )
                    conn.commit()
                    return True
        except Exception as e:
            logger.error(f"Error updating last login: {e}", exc_info=True)
            return False

    def list_users(
        self,
        page: int = 1,
        page_size: int = 50,
        user_type: Optional[UserType] = None,
        is_blocked: Optional[bool] = None
    ) -> tuple[List[User], int]:
        """
        List users with pagination and filters
        Returns (users_list, total_count)
        """
        try:
            offset = (page - 1) * page_size
            conditions = []
            params = []

            if user_type is not None:
                conditions.append("user_type = %s")
                params.append(user_type.value)
            if is_blocked is not None:
                conditions.append("is_blocked = %s")
                params.append(is_blocked)

            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

            with self.pg_pool.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    # Get total count
                    cursor.execute(
                        f"SELECT COUNT(*) as total FROM auth.users {where_clause}",
                        params
                    )
                    total = cursor.fetchone()['total']

                    # Get paginated results
                    cursor.execute(
                        f"""
                        SELECT id, username, email, password_hash, user_type,
                               is_blocked, max_active_sessions, created_at,
                               updated_at, last_login_at
                        FROM auth.users
                        {where_clause}
                        ORDER BY created_at DESC
                        LIMIT %s OFFSET %s
                        """,
                        params + [page_size, offset]
                    )
                    rows = cursor.fetchall()

                    users = [
                        User(
                            id=row['id'],
                            username=row['username'],
                            email=row['email'],
                            password_hash=row['password_hash'],
                            user_type=UserType(row['user_type']),
                            is_blocked=row['is_blocked'],
                            max_active_sessions=row['max_active_sessions'],
                            created_at=row['created_at'],
                            updated_at=row['updated_at'],
                            last_login_at=row['last_login_at']
                        )
                        for row in rows
                    ]

                    return users, total

        except Exception as e:
            logger.error(f"Error listing users: {e}", exc_info=True)
            return [], 0

    def block_user(self, user_id: UUID) -> bool:
        """Block a user and invalidate all their sessions"""
        try:
            with self.pg_pool.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Block user
                    cursor.execute(
                        "UPDATE auth.users SET is_blocked = true WHERE id = %s",
                        (str(user_id),)
                    )
                    # Invalidate all sessions
                    cursor.execute(
                        "UPDATE auth.sessions SET is_active = false WHERE user_id = %s",
                        (str(user_id),)
                    )
                    # Revoke all refresh tokens
                    cursor.execute(
                        "UPDATE auth.refresh_tokens SET is_revoked = true WHERE user_id = %s",
                        (str(user_id),)
                    )
                    conn.commit()
                    logger.info(f"User {user_id} blocked and sessions invalidated")
                    return True

        except Exception as e:
            logger.error(f"Error blocking user: {e}", exc_info=True)
            return False

    def unblock_user(self, user_id: UUID) -> bool:
        """Unblock a user"""
        try:
            with self.pg_pool.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "UPDATE auth.users SET is_blocked = false WHERE id = %s",
                        (str(user_id),)
                    )
                    conn.commit()
                    logger.info(f"User {user_id} unblocked")
                    return True

        except Exception as e:
            logger.error(f"Error unblocking user: {e}", exc_info=True)
            return False

    # ========================================================================
    # Permission Management
    # ========================================================================

    def grant_permission(
        self,
        user_id: UUID,
        permission: str,
        granted_by: Optional[UUID] = None
    ) -> Optional[Permission]:
        """Grant a permission to a user"""
        try:
            with self.pg_pool.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute(
                        """
                        INSERT INTO auth.user_permissions (user_id, permission, granted_by)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (user_id, permission) DO UPDATE
                        SET granted_at = NOW(), granted_by = EXCLUDED.granted_by
                        RETURNING id, user_id, permission, granted_at, granted_by
                        """,
                        (str(user_id), permission, str(granted_by) if granted_by else None)
                    )
                    row = cursor.fetchone()
                    conn.commit()

                    if row:
                        logger.info(f"Granted permission '{permission}' to user {user_id}")
                        return Permission(
                            id=row['id'],
                            user_id=row['user_id'],
                            permission=row['permission'],
                            granted_at=row['granted_at'],
                            granted_by=row['granted_by']
                        )
                    return None

        except Exception as e:
            logger.error(f"Error granting permission: {e}", exc_info=True)
            return None

    def revoke_permission(self, user_id: UUID, permission: str) -> bool:
        """Revoke a permission from a user"""
        try:
            with self.pg_pool.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "DELETE FROM auth.user_permissions WHERE user_id = %s AND permission = %s",
                        (str(user_id), permission)
                    )
                    conn.commit()
                    logger.info(f"Revoked permission '{permission}' from user {user_id}")
                    return True

        except Exception as e:
            logger.error(f"Error revoking permission: {e}", exc_info=True)
            return False

    def get_user_permissions(self, user_id: UUID) -> List[str]:
        """Get all permissions for a user"""
        try:
            with self.pg_pool.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute(
                        "SELECT permission FROM auth.user_permissions WHERE user_id = %s",
                        (str(user_id),)
                    )
                    rows = cursor.fetchall()
                    return [row['permission'] for row in rows]

        except Exception as e:
            logger.error(f"Error fetching user permissions: {e}", exc_info=True)
            return []

    def has_permission(self, user_id: UUID, user_type: UserType, permission: str) -> bool:
        """Check if user has a specific permission"""
        # Admins have all permissions
        if user_type == UserType.ADMIN:
            return True

        # Check explicit permissions
        permissions = self.get_user_permissions(user_id)
        return permission in permissions

    # ========================================================================
    # Session Management
    # ========================================================================

    def create_session(
        self,
        user_id: UUID,
        token: str,
        expires_at: datetime,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Optional[Session]:
        """Create a new session"""
        try:
            with self.pg_pool.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute(
                        """
                        INSERT INTO auth.sessions (
                            user_id, token, expires_at, ip_address, user_agent
                        ) VALUES (%s, %s, %s, %s, %s)
                        RETURNING id, user_id, token, ip_address, user_agent,
                                  created_at, expires_at, last_activity, is_active
                        """,
                        (str(user_id), token, expires_at, ip_address, user_agent)
                    )
                    row = cursor.fetchone()
                    conn.commit()

                    if row:
                        return Session(
                            id=row['id'],
                            user_id=row['user_id'],
                            token=row['token'],
                            ip_address=row['ip_address'],
                            user_agent=row['user_agent'],
                            created_at=row['created_at'],
                            expires_at=row['expires_at'],
                            last_activity=row['last_activity'],
                            is_active=row['is_active']
                        )
                    return None

        except Exception as e:
            logger.error(f"Error creating session: {e}", exc_info=True)
            return None

    def is_session_active(self, session_id: UUID) -> bool:
        """Check if a session is still active (exists and is_active=true)."""
        try:
            with self.pg_pool.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT 1 FROM auth.sessions WHERE id = %s AND is_active = true",
                        (str(session_id),),
                    )
                    return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Error checking session: {e}", exc_info=True)
            return False

    def get_session_by_token(self, token: str) -> Optional[Session]:
        """Get session by token"""
        try:
            with self.pg_pool.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute(
                        """
                        SELECT id, user_id, token, ip_address, user_agent,
                               created_at, expires_at, last_activity, is_active
                        FROM auth.sessions
                        WHERE token = %s AND is_active = true
                        """,
                        (token,)
                    )
                    row = cursor.fetchone()

                    if row:
                        return Session(
                            id=row['id'],
                            user_id=row['user_id'],
                            token=row['token'],
                            ip_address=row['ip_address'],
                            user_agent=row['user_agent'],
                            created_at=row['created_at'],
                            expires_at=row['expires_at'],
                            last_activity=row['last_activity'],
                            is_active=row['is_active']
                        )
                    return None

        except Exception as e:
            logger.error(f"Error fetching session: {e}", exc_info=True)
            return None

    def update_session_activity(self, session_id: UUID) -> bool:
        """Update session last activity timestamp"""
        try:
            with self.pg_pool.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "UPDATE auth.sessions SET last_activity = NOW() WHERE id = %s",
                        (str(session_id),)
                    )
                    conn.commit()
                    return True

        except Exception as e:
            logger.error(f"Error updating session activity: {e}", exc_info=True)
            return False

    def invalidate_session(self, session_id: UUID) -> bool:
        """Invalidate a session"""
        try:
            with self.pg_pool.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "UPDATE auth.sessions SET is_active = false WHERE id = %s",
                        (str(session_id),)
                    )
                    conn.commit()
                    return True

        except Exception as e:
            logger.error(f"Error invalidating session: {e}", exc_info=True)
            return False

    def list_user_sessions(self, user_id: UUID, active_only: bool = True) -> List[Session]:
        """List all sessions for a user"""
        try:
            with self.pg_pool.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    query = """
                        SELECT id, user_id, token, ip_address, user_agent,
                               created_at, expires_at, last_activity, is_active
                        FROM auth.sessions
                        WHERE user_id = %s
                    """
                    if active_only:
                        query += " AND is_active = true"
                    query += " ORDER BY created_at DESC"

                    cursor.execute(query, (str(user_id),))
                    rows = cursor.fetchall()

                    return [
                        Session(
                            id=row['id'],
                            user_id=row['user_id'],
                            token=row['token'],
                            ip_address=row['ip_address'],
                            user_agent=row['user_agent'],
                            created_at=row['created_at'],
                            expires_at=row['expires_at'],
                            last_activity=row['last_activity'],
                            is_active=row['is_active']
                        )
                        for row in rows
                    ]

        except Exception as e:
            logger.error(f"Error listing user sessions: {e}", exc_info=True)
            return []

    def enforce_session_limit(self, user_id: UUID, max_sessions: int) -> bool:
        """Delete oldest sessions if user exceeds limit"""
        try:
            with self.pg_pool.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Delete oldest sessions beyond the limit
                    cursor.execute(
                        """
                        UPDATE auth.sessions
                        SET is_active = false
                        WHERE id IN (
                            SELECT id FROM auth.sessions
                            WHERE user_id = %s AND is_active = true
                            ORDER BY created_at ASC
                            LIMIT GREATEST(0, (
                                SELECT COUNT(*) FROM auth.sessions
                                WHERE user_id = %s AND is_active = true
                            ) - %s + 1)
                        )
                        """,
                        (str(user_id), str(user_id), max_sessions)
                    )
                    conn.commit()
                    return True

        except Exception as e:
            logger.error(f"Error enforcing session limit: {e}", exc_info=True)
            return False

    # ========================================================================
    # Refresh Token Management
    # ========================================================================

    def store_refresh_token(
        self,
        user_id: UUID,
        token_hash: str,
        expires_at: datetime
    ) -> bool:
        """Store a refresh token"""
        try:
            with self.pg_pool.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO auth.refresh_tokens (user_id, token_hash, expires_at)
                        VALUES (%s, %s, %s)
                        """,
                        (str(user_id), token_hash, expires_at)
                    )
                    conn.commit()
                    return True

        except Exception as e:
            logger.error(f"Error storing refresh token: {e}", exc_info=True)
            return False

    def verify_refresh_token(self, token_hash: str) -> Optional[UUID]:
        """Verify refresh token and return user_id if valid"""
        try:
            with self.pg_pool.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute(
                        """
                        SELECT user_id FROM auth.refresh_tokens
                        WHERE token_hash = %s
                          AND is_revoked = false
                          AND expires_at > NOW()
                        """,
                        (token_hash,)
                    )
                    row = cursor.fetchone()
                    return row['user_id'] if row else None

        except Exception as e:
            logger.error(f"Error verifying refresh token: {e}", exc_info=True)
            return None

    def revoke_refresh_token(self, token_hash: str) -> bool:
        """Revoke a refresh token"""
        try:
            with self.pg_pool.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "UPDATE auth.refresh_tokens SET is_revoked = true WHERE token_hash = %s",
                        (token_hash,)
                    )
                    conn.commit()
                    return True

        except Exception as e:
            logger.error(f"Error revoking refresh token: {e}", exc_info=True)
            return False

    def revoke_excess_refresh_tokens(self, user_id: UUID, max_sessions: int) -> None:
        """
        Revoke oldest refresh tokens so that the user has at most (max_sessions - 1)
        valid (non-revoked, not expired) refresh tokens. Called before adding a new
        one on login so evicted sessions cannot refresh and will be sent to login.
        """
        try:
            with self.pg_pool.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Revoke oldest valid refresh tokens so only (max_sessions - 1) remain
                    cursor.execute(
                        """
                        UPDATE auth.refresh_tokens
                        SET is_revoked = true
                        WHERE id IN (
                            SELECT id FROM (
                                SELECT id,
                                    ROW_NUMBER() OVER (ORDER BY created_at ASC) AS rn,
                                    (SELECT COUNT(*) FROM auth.refresh_tokens
                                     WHERE user_id = %s AND is_revoked = false AND expires_at > NOW()) AS cnt
                                FROM auth.refresh_tokens
                                WHERE user_id = %s AND is_revoked = false AND expires_at > NOW()
                            ) t
                            WHERE rn <= cnt - %s
                        )
                        """,
                        (str(user_id), str(user_id), max_sessions - 1)
                    )
                    conn.commit()
        except Exception as e:
            logger.error(f"Error revoking excess refresh tokens: {e}", exc_info=True)

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def get_authenticated_user(self, user_id: UUID) -> Optional[AuthenticatedUser]:
        """Get AuthenticatedUser with permissions for use in auth middleware"""
        user = self.get_user_by_id(user_id)
        if not user:
            return None

        permissions = self.get_user_permissions(user_id)

        return AuthenticatedUser(
            id=user.id,
            username=user.username,
            email=user.email,
            user_type=user.user_type,
            is_blocked=user.is_blocked,
            permissions=permissions
        )
