# User Management & Authentication System

Complete user authentication and session management system with JWT and session-based authentication.

## Features

- **Dual Authentication**: JWT tokens + Session-based authentication
- **Role-Based Access Control**: Admin, Super User, and User roles
- **Fine-Grained Permissions**: User-specific permissions for granular access control
- **Session Management**: Track and manage active sessions with configurable limits
- **User Blocking**: Block/unblock users and automatically invalidate their sessions
- **Secure Password Storage**: bcrypt hashing with cost factor 12
- **Refresh Tokens**: Long-lived refresh tokens for seamless authentication

## Quick Start

### 1. Run Database Migration

Execute the SQL migration to create the required tables:

```bash
psql -h your_host -U your_user -d your_database -f migrations/001_create_users_and_sessions.sql
```

Or using the PostgreSQL DSN:

```bash
psql $POSTGRES_DSN -f migrations/001_create_users_and_sessions.sql
```

### 2. Configure Environment Variables

Add these to your `.env` file:

```env
# Authentication & Security
JWT_SECRET_KEY=your_secure_random_key_here
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=15
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7
SESSION_EXPIRE_HOURS=24
SESSION_INACTIVITY_TIMEOUT_MINUTES=120

# Default Admin User (optional)
DEFAULT_ADMIN_USERNAME=admin
DEFAULT_ADMIN_PASSWORD=change_me_immediately
```

**IMPORTANT**: Generate a secure JWT_SECRET_KEY:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Create Your First Admin User

**Option A: Use the script (recommended)**

```bash
python scripts/create_admin_user.py --username admin --email admin@example.com
```

This will prompt you to enter a password securely.

**Option B: Let the system create a default admin**

On first startup, if no users exist, the system will automatically create a default admin user using the credentials from `.env`:
- Username: `DEFAULT_ADMIN_USERNAME` (default: "admin")
- Password: `DEFAULT_ADMIN_PASSWORD` (default: "admin123")

**⚠️ CHANGE THE DEFAULT PASSWORD IMMEDIATELY!**

### 5. Start the API

```bash
python app.py
```

## Database Schema

### Tables

**`users`**
- Stores user account information
- Fields: id, username, email, password_hash, user_type, is_blocked, max_active_sessions
- Indexes on: username, email, user_type, is_blocked

**`user_permissions`**
- Stores user-specific permissions
- Fields: id, user_id, permission, granted_at, granted_by
- Unique constraint on (user_id, permission)

**`sessions`**
- Stores active user sessions
- Fields: id, user_id, token, ip_address, user_agent, created_at, expires_at, last_activity, is_active
- Indexes on: user_id, token, expires_at

**`refresh_tokens`**
- Stores JWT refresh tokens
- Fields: id, user_id, token_hash, expires_at, created_at, is_revoked
- Index on: token_hash, user_id

## User Types & Hierarchy

1. **Admin** - Full access to all features and management capabilities
2. **Super User** - Elevated privileges, can view and manage users
3. **User** - Standard user with limited access

Admins automatically have all permissions. Super users and regular users require explicit permission grants.

## API Endpoints

### Public Endpoints

#### Register User
```http
POST /api/auth/register
Content-Type: application/json

{
  "username": "johndoe",
  "email": "john@example.com",
  "password": "SecurePass123",
  "user_type": "user",
  "max_active_sessions": 5
}
```

**Note**: In production, you may want to restrict registration to admin-only.

#### Login
```http
POST /api/auth/login
Content-Type: application/json

{
  "username": "johndoe",
  "password": "SecurePass123",
  "ip_address": "192.168.1.1",
  "user_agent": "Mozilla/5.0..."
}
```

Response:
```json
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 900,
  "user": {
    "id": "uuid",
    "username": "johndoe",
    "email": "john@example.com",
    "user_type": "user",
    "is_blocked": false,
    "permissions": []
  }
}
```

#### Refresh Token
```http
POST /api/auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJ..."
}
```

### Authenticated Endpoints

All authenticated endpoints require either:
- JWT: `Authorization: Bearer <access_token>` header
- Session: `session_token` cookie or `X-Session-Token` header

#### Logout
```http
POST /api/auth/logout
Authorization: Bearer <access_token>
```

#### Get Current User
```http
GET /api/auth/me
Authorization: Bearer <access_token>
```

#### Get Active Sessions
```http
GET /api/auth/sessions
Authorization: Bearer <access_token>
```

#### Delete Session
```http
DELETE /api/auth/sessions/{session_id}
Authorization: Bearer <access_token>
```

#### Change Password
```http
POST /api/auth/change-password
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "current_password": "OldPass123",
  "new_password": "NewSecurePass456"
}
```

### Admin Endpoints

Require `admin` or `super_user` role.

#### List Users
```http
GET /api/users?page=1&page_size=50&user_type=user&is_blocked=false
Authorization: Bearer <admin_access_token>
```

#### Get User
```http
GET /api/users/{user_id}
Authorization: Bearer <admin_access_token>
```

#### Update User (Admin only)
```http
PUT /api/users/{user_id}
Authorization: Bearer <admin_access_token>
Content-Type: application/json

{
  "username": "newusername",
  "email": "newemail@example.com",
  "user_type": "super_user",
  "max_active_sessions": 10
}
```

#### Block User (Admin only)
```http
POST /api/users/{user_id}/block
Authorization: Bearer <admin_access_token>
```

#### Unblock User (Admin only)
```http
POST /api/users/{user_id}/unblock
Authorization: Bearer <admin_access_token>
```

#### Grant Permission (Admin only)
```http
POST /api/users/{user_id}/permissions
Authorization: Bearer <admin_access_token>
Content-Type: application/json

{
  "permission": "feedback.create"
}
```

Permission format: `resource.action` (e.g., `feedback.create`, `analytics.write`)

#### Revoke Permission (Admin only)
```http
DELETE /api/users/{user_id}/permissions/feedback.create
Authorization: Bearer <admin_access_token>
```

#### Get User Sessions (Admin only)
```http
GET /api/users/{user_id}/sessions?active_only=true
Authorization: Bearer <admin_access_token>
```

#### Terminate User Session (Admin only)
```http
DELETE /api/users/{user_id}/sessions/{session_id}
Authorization: Bearer <admin_access_token>
```

## Permission System

### Current Permissions

- `feedback.create` - Required to submit anomaly feedback

### Granting Permissions

Only admins can grant permissions:

```bash
# Using curl
curl -X POST http://localhost:8000/api/users/{user_id}/permissions \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"permission": "feedback.create"}'
```

### Permission Hierarchy

1. **Admin users** - Have ALL permissions by default (no explicit grants needed)
2. **Super users** - Can be granted specific permissions
3. **Regular users** - Need explicit permission grants for any protected actions

## Protected Routes

The following routes now require authentication and specific permissions:

- `POST /api/feedback` - Requires `feedback.create` permission
- `POST /api/feedback/reanalyze/{flight_id}` - Requires `feedback.create` permission

## Session Management

### Session Limits

Each user has a `max_active_sessions` limit (default: 5). When this limit is reached:
- New login automatically invalidates the oldest session
- User can manually manage sessions via the API

### Session Expiration

Sessions expire based on:
1. **Absolute timeout**: Default 24 hours
2. **Inactivity timeout**: Default 120 minutes (2 hours)

Configure via environment variables:
- `SESSION_EXPIRE_HOURS`
- `SESSION_INACTIVITY_TIMEOUT_MINUTES`

## Security Best Practices

1. **JWT Secret Key**: Use a strong, randomly generated key
2. **HTTPS**: Always use HTTPS in production
3. **Password Policy**: Enforce strong passwords (min 8 characters)
4. **Default Credentials**: Change default admin password immediately
5. **Session Limits**: Set appropriate max_active_sessions per user
6. **Token Expiry**: Keep access token expiry short (15 minutes recommended)
7. **Refresh Tokens**: Store refresh tokens securely on client side
8. **CORS**: Configure proper CORS settings for production

## Authentication Flow

### JWT + Session Flow

1. User logs in with username/password
2. Server creates:
   - Access token (15 min expiry)
   - Refresh token (7 days expiry)
   - Session record in database
3. Client stores both tokens
4. Client includes access token in requests
5. When access token expires, use refresh token to get new access token
6. Session tracks activity and enforces limits

### Dual Authentication Support

Endpoints accept either:
- **JWT**: Fast, stateless, ideal for APIs
- **Session**: Stateful, trackable, ideal for web apps

Both can be used simultaneously for maximum flexibility.

## Troubleshooting

### Default admin not created

Check:
1. Database connection is working
2. Tables were created successfully
3. Check application logs for errors

### Cannot login

Verify:
1. User exists and is not blocked
2. Password is correct
3. Database is accessible

### Permission denied errors

Check:
1. User has required permission (or is admin)
2. Access token is valid and not expired
3. User is not blocked

### Session issues

1. Check session hasn't expired (absolute or inactivity timeout)
2. Verify session limit hasn't been exceeded
3. Ensure session token is being sent correctly

## Examples

### Full Authentication Flow (Python)

```python
import requests

API_URL = "http://localhost:8000"

# Login
response = requests.post(f"{API_URL}/api/auth/login", json={
    "username": "admin",
    "password": "SecurePass123"
})
auth_data = response.json()
access_token = auth_data["access_token"]
refresh_token = auth_data["refresh_token"]

# Make authenticated request
headers = {"Authorization": f"Bearer {access_token}"}
response = requests.get(f"{API_URL}/api/auth/me", headers=headers)
print(response.json())

# Submit feedback (requires permission)
response = requests.post(
    f"{API_URL}/api/feedback",
    headers=headers,
    json={
        "flight_id": "ABC123",
        "is_anomaly": True,
        "comments": "Unusual flight pattern"
    }
)

# Refresh token when access token expires
response = requests.post(f"{API_URL}/api/auth/refresh", json={
    "refresh_token": refresh_token
})
new_access_token = response.json()["access_token"]
```

### Managing Users (Admin)

```python
# List all users
response = requests.get(
    f"{API_URL}/api/users?page=1&page_size=50",
    headers=headers
)
users = response.json()

# Grant permission to user
user_id = "some-user-uuid"
response = requests.post(
    f"{API_URL}/api/users/{user_id}/permissions",
    headers=headers,
    json={"permission": "feedback.create"}
)

# Block a user
response = requests.post(
    f"{API_URL}/api/users/{user_id}/block",
    headers=headers
)
```

## Migration from No-Auth System

If you're adding this to an existing system:

1. Run the migration script to create tables
2. Create an admin user using the script
3. Gradually add permission checks to routes that need protection
4. Update frontend to handle authentication
5. Test thoroughly before deploying to production

## Support

For issues or questions:
1. Check application logs
2. Verify database connection and tables
3. Ensure environment variables are set correctly
4. Review this documentation for proper usage
