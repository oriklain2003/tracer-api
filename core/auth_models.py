"""
Authentication and User Management Models
"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, List
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, validator

# Role hierarchy for UserType (module-level to avoid Enum/str attribute shadowing)
_USER_TYPE_ORDER = {"user": 0, "super_user": 1, "admin": 2}


class UserType(str, Enum):
    """User type enumeration with hierarchy: admin > super_user > user"""
    ADMIN = "admin"
    SUPER_USER = "super_user"
    USER = "user"

    def __lt__(self, other):
        """Define ordering for permission hierarchy"""
        if isinstance(other, UserType):
            return _USER_TYPE_ORDER[self.value] < _USER_TYPE_ORDER[other.value]
        return NotImplemented

    def __le__(self, other):
        """Define less-than-or-equal for permission hierarchy"""
        return self == other or self < other

    def __ge__(self, other):
        """Define greater-than-or-equal so 'admin' >= 'super_user' (hierarchy), not lexicographic."""
        if isinstance(other, UserType):
            return _USER_TYPE_ORDER[self.value] >= _USER_TYPE_ORDER[other.value]
        return NotImplemented


# ============================================================================
# Pydantic Request/Response Models
# ============================================================================

class UserCreate(BaseModel):
    """Request model for creating a new user"""
    username: str = Field(..., min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    password: str = Field(..., min_length=8, max_length=100)
    user_type: UserType = UserType.USER
    max_active_sessions: int = Field(default=5, ge=1, le=100)

    @validator('username')
    def username_alphanumeric(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username must be alphanumeric (can include _ and -)')
        return v


class UserUpdate(BaseModel):
    """Request model for updating a user"""
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    user_type: Optional[UserType] = None
    max_active_sessions: Optional[int] = Field(None, ge=1, le=100)

    @validator('username')
    def username_alphanumeric(cls, v):
        if v is not None and not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username must be alphanumeric (can include _ and -)')
        return v


class UserResponse(BaseModel):
    """Response model for user data"""
    id: UUID
    username: str
    email: Optional[str]
    user_type: UserType
    is_blocked: bool
    max_active_sessions: int
    created_at: datetime
    updated_at: datetime
    last_login_at: Optional[datetime]
    permissions: List[str] = []

    class Config:
        from_attributes = True


class LoginRequest(BaseModel):
    """Request model for user login"""
    username: str
    password: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class LoginResponse(BaseModel):
    """Response model for successful login"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    user: UserResponse


class TokenRefreshRequest(BaseModel):
    """Request model for refreshing access token"""
    refresh_token: str


class TokenRefreshResponse(BaseModel):
    """Response model for token refresh"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class PermissionGrant(BaseModel):
    """Request model for granting permission to a user"""
    permission: str = Field(..., min_length=1, max_length=255)

    @validator('permission')
    def validate_permission_format(cls, v):
        """Validate permission format (e.g., 'feedback.create', 'analytics.write')"""
        if '.' not in v:
            raise ValueError('Permission must be in format: resource.action')
        parts = v.split('.')
        if len(parts) != 2 or not all(parts):
            raise ValueError('Permission must be in format: resource.action')
        return v


class PermissionResponse(BaseModel):
    """Response model for permission data"""
    id: UUID
    user_id: UUID
    permission: str
    granted_at: datetime
    granted_by: Optional[UUID]


class SessionResponse(BaseModel):
    """Response model for session data"""
    id: UUID
    user_id: UUID
    ip_address: Optional[str]
    user_agent: Optional[str]
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    is_active: bool

    class Config:
        from_attributes = True


class UserListResponse(BaseModel):
    """Response model for paginated user list"""
    users: List[UserResponse]
    total: int
    page: int
    page_size: int


class ChangePasswordRequest(BaseModel):
    """Request model for changing password"""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=100)


# ============================================================================
# Internal Dataclasses (for service layer)
# ============================================================================

@dataclass
class User:
    """Internal user representation"""
    id: UUID
    username: str
    email: Optional[str]
    password_hash: str
    user_type: UserType
    is_blocked: bool
    max_active_sessions: int
    created_at: datetime
    updated_at: datetime
    last_login_at: Optional[datetime]


@dataclass
class Session:
    """Internal session representation"""
    id: UUID
    user_id: UUID
    token: str
    ip_address: Optional[str]
    user_agent: Optional[str]
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    is_active: bool


@dataclass
class Permission:
    """Internal permission representation"""
    id: UUID
    user_id: UUID
    permission: str
    granted_at: datetime
    granted_by: Optional[UUID]


@dataclass
class RefreshToken:
    """Internal refresh token representation"""
    id: UUID
    user_id: UUID
    token_hash: str
    expires_at: datetime
    created_at: datetime
    is_revoked: bool


@dataclass
class AuthenticatedUser:
    """User data after authentication with permissions"""
    id: UUID
    username: str
    email: Optional[str]
    user_type: UserType
    is_blocked: bool
    permissions: List[str]

    def has_permission(self, permission: str) -> bool:
        """Check if user has a specific permission"""
        # Admins have all permissions
        if self.user_type == UserType.ADMIN:
            return True
        return permission in self.permissions

    def has_role(self, min_role: UserType) -> bool:
        """Check if user meets minimum role requirement"""
        return self.user_type >= min_role
