# Frontend Integration Guide - User Management

Complete guide for integrating the user management system into your frontend application.

## Table of Contents
1. [Authentication Flow](#authentication-flow)
2. [User Sessions Management](#user-sessions-management)
3. [User Management (Admin)](#user-management-admin)
4. [Permission Management](#permission-management)
5. [Complete UI Workflows](#complete-ui-workflows)

---

## Authentication Flow

### 1. Login

**Endpoint:** `POST /api/auth/login`

**Request:**
```javascript
const response = await fetch('http://localhost:8000/api/auth/login', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    username: 'admin',
    password: 'your_password',
    ip_address: window.location.hostname, // Optional
    user_agent: navigator.userAgent        // Optional
  })
});

const data = await response.json();
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 900,
  "user": {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "username": "admin",
    "email": "admin@example.com",
    "user_type": "admin",
    "is_blocked": false,
    "max_active_sessions": 10,
    "permissions": ["feedback.create"],
    "created_at": "2026-02-04T12:00:00Z",
    "updated_at": "2026-02-04T12:00:00Z",
    "last_login_at": "2026-02-04T15:30:00Z"
  }
}
```

**Store Tokens:**
```javascript
// Store in localStorage or secure cookie
localStorage.setItem('access_token', data.access_token);
localStorage.setItem('refresh_token', data.refresh_token);
localStorage.setItem('user', JSON.stringify(data.user));
```

### 2. Using Authenticated Endpoints

**Add Authorization Header:**
```javascript
const token = localStorage.getItem('access_token');

const response = await fetch('http://localhost:8000/api/auth/me', {
  method: 'GET',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  }
});
```

### 3. Refresh Token When Expired

**Endpoint:** `POST /api/auth/refresh`

```javascript
async function refreshAccessToken() {
  const refreshToken = localStorage.getItem('refresh_token');
  
  const response = await fetch('http://localhost:8000/api/auth/refresh', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      refresh_token: refreshToken
    })
  });
  
  const data = await response.json();
  localStorage.setItem('access_token', data.access_token);
  
  return data.access_token;
}

// Auto-refresh on 401 errors
async function fetchWithAuth(url, options = {}) {
  let token = localStorage.getItem('access_token');
  
  let response = await fetch(url, {
    ...options,
    headers: {
      ...options.headers,
      'Authorization': `Bearer ${token}`
    }
  });
  
  // If token expired, refresh and retry
  if (response.status === 401) {
    token = await refreshAccessToken();
    response = await fetch(url, {
      ...options,
      headers: {
        ...options.headers,
        'Authorization': `Bearer ${token}`
      }
    });
  }
  
  return response;
}
```

### 4. Logout

**Endpoint:** `POST /api/auth/logout`

```javascript
async function logout() {
  const token = localStorage.getItem('access_token');
  
  await fetch('http://localhost:8000/api/auth/logout', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`
    }
  });
  
  // Clear local storage
  localStorage.removeItem('access_token');
  localStorage.removeItem('refresh_token');
  localStorage.removeItem('user');
  
  // Redirect to login
  window.location.href = '/login';
}
```

---

## User Sessions Management

### View Current User's Sessions

**Endpoint:** `GET /api/auth/sessions`

Shows all active sessions for the logged-in user with IP addresses and device information.

```javascript
async function getCurrentUserSessions() {
  const response = await fetchWithAuth('http://localhost:8000/api/auth/sessions');
  const sessions = await response.json();
  return sessions;
}
```

**Response:**
```json
[
  {
    "id": "session-uuid-1",
    "user_id": "user-uuid",
    "ip_address": "192.168.1.100",
    "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)...",
    "created_at": "2026-02-04T12:00:00Z",
    "expires_at": "2026-02-05T12:00:00Z",
    "last_activity": "2026-02-04T15:30:00Z",
    "is_active": true
  },
  {
    "id": "session-uuid-2",
    "user_id": "user-uuid",
    "ip_address": "10.0.0.50",
    "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0...)...",
    "created_at": "2026-02-03T08:00:00Z",
    "expires_at": "2026-02-04T08:00:00Z",
    "last_activity": "2026-02-04T14:00:00Z",
    "is_active": true
  }
]
```

**UI Display:**
```javascript
// Example React component
function SessionsList({ sessions }) {
  return (
    <div className="sessions-list">
      <h2>Active Sessions</h2>
      {sessions.map(session => (
        <div key={session.id} className="session-card">
          <div className="session-info">
            <p><strong>IP:</strong> {session.ip_address}</p>
            <p><strong>Device:</strong> {parseUserAgent(session.user_agent)}</p>
            <p><strong>Last Active:</strong> {formatDate(session.last_activity)}</p>
            <p><strong>Expires:</strong> {formatDate(session.expires_at)}</p>
          </div>
          <button onClick={() => terminateSession(session.id)}>
            Terminate
          </button>
        </div>
      ))}
    </div>
  );
}
```

### Terminate a Session

**Endpoint:** `DELETE /api/auth/sessions/{session_id}`

```javascript
async function terminateSession(sessionId) {
  const response = await fetchWithAuth(
    `http://localhost:8000/api/auth/sessions/${sessionId}`,
    { method: 'DELETE' }
  );
  
  if (response.ok) {
    alert('Session terminated successfully');
    // Refresh session list
  }
}
```

---

## User Management (Admin)

### List All Users

**Endpoint:** `GET /api/users`

**Requires:** Admin or Super User role

**Query Parameters:**
- `page` (default: 1)
- `page_size` (default: 50)
- `user_type` (optional: admin, super_user, user)
- `is_blocked` (optional: true, false)

```javascript
async function listUsers(page = 1, filters = {}) {
  const params = new URLSearchParams({
    page: page,
    page_size: 50,
    ...filters
  });
  
  const response = await fetchWithAuth(
    `http://localhost:8000/api/users?${params}`
  );
  
  return await response.json();
}
```

**Response:**
```json
{
  "users": [
    {
      "id": "user-uuid-1",
      "username": "john_doe",
      "email": "john@example.com",
      "user_type": "user",
      "is_blocked": false,
      "max_active_sessions": 5,
      "permissions": ["feedback.create"],
      "created_at": "2026-01-15T10:00:00Z",
      "updated_at": "2026-02-01T14:30:00Z",
      "last_login_at": "2026-02-04T09:15:00Z"
    },
    {
      "id": "user-uuid-2",
      "username": "jane_admin",
      "email": "jane@example.com",
      "user_type": "admin",
      "is_blocked": false,
      "max_active_sessions": 10,
      "permissions": [],
      "created_at": "2026-01-10T08:00:00Z",
      "updated_at": "2026-02-04T15:00:00Z",
      "last_login_at": "2026-02-04T15:30:00Z"
    }
  ],
  "total": 25,
  "page": 1,
  "page_size": 50
}
```

**UI Component:**
```javascript
function UsersTable({ users, onUserClick }) {
  return (
    <table className="users-table">
      <thead>
        <tr>
          <th>Username</th>
          <th>Email</th>
          <th>Type</th>
          <th>Status</th>
          <th>Last Login</th>
          <th>Permissions</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody>
        {users.map(user => (
          <tr key={user.id} onClick={() => onUserClick(user)}>
            <td>{user.username}</td>
            <td>{user.email || 'N/A'}</td>
            <td>
              <span className={`badge ${user.user_type}`}>
                {user.user_type}
              </span>
            </td>
            <td>
              <span className={user.is_blocked ? 'blocked' : 'active'}>
                {user.is_blocked ? '🔒 Blocked' : '✓ Active'}
              </span>
            </td>
            <td>{formatDate(user.last_login_at)}</td>
            <td>
              <span className="permission-count">
                {user.permissions.length} permissions
              </span>
            </td>
            <td>
              <button onClick={(e) => {
                e.stopPropagation();
                viewUserDetails(user.id);
              }}>
                Details
              </button>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
```

### View User Details

**Endpoint:** `GET /api/users/{user_id}`

```javascript
async function getUserDetails(userId) {
  const response = await fetchWithAuth(
    `http://localhost:8000/api/users/${userId}`
  );
  return await response.json();
}
```

### View All Sessions (System-Wide Admin View)

**Endpoint:** `GET /api/admin/sessions`

Shows all active sessions across all users (admin only). Perfect for system monitoring.

**Query Parameters:**
- `page` (default: 1)
- `page_size` (default: 50)
- `active_only` (default: true)

```javascript
async function getAllSessions(page = 1, activeOnly = true) {
  const params = new URLSearchParams({
    page: page,
    page_size: 50,
    active_only: activeOnly
  });
  
  const response = await fetchWithAuth(
    `http://localhost:8000/api/admin/sessions?${params}`
  );
  return await response.json();
}
```

**Response:**
```json
{
  "sessions": [
    {
      "id": "session-uuid-1",
      "user_id": "user-uuid-1",
      "username": "john_doe",
      "email": "john@example.com",
      "user_type": "user",
      "is_user_blocked": false,
      "ip_address": "192.168.1.100",
      "user_agent": "Mozilla/5.0...",
      "created_at": "2026-02-04T12:00:00Z",
      "expires_at": "2026-02-05T12:00:00Z",
      "last_activity": "2026-02-04T15:30:00Z",
      "is_active": true
    },
    {
      "id": "session-uuid-2",
      "user_id": "user-uuid-2",
      "username": "jane_admin",
      "email": "jane@example.com",
      "user_type": "admin",
      "is_user_blocked": false,
      "ip_address": "10.0.0.50",
      "user_agent": "Mozilla/5.0...",
      "created_at": "2026-02-04T10:00:00Z",
      "expires_at": "2026-02-05T10:00:00Z",
      "last_activity": "2026-02-04T15:25:00Z",
      "is_active": true
    }
  ],
  "total": 45,
  "page": 1,
  "page_size": 50
}
```

**UI Component:**
```javascript
function AllSessionsMonitor() {
  const [sessions, setSessions] = useState([]);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  
  useEffect(() => {
    loadSessions();
  }, [page]);
  
  const loadSessions = async () => {
    const data = await getAllSessions(page);
    setSessions(data.sessions);
    setTotal(data.total);
  };
  
  return (
    <div className="sessions-monitor">
      <h2>All Active Sessions ({total})</h2>
      
      <table className="sessions-table">
        <thead>
          <tr>
            <th>User</th>
            <th>Type</th>
            <th>IP Address</th>
            <th>Device</th>
            <th>Last Active</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {sessions.map(session => (
            <tr key={session.id}>
              <td>
                <div>
                  <strong>{session.username}</strong>
                  {session.is_user_blocked && (
                    <span className="badge blocked">Blocked</span>
                  )}
                </div>
                <small>{session.email}</small>
              </td>
              <td>
                <span className={`badge ${session.user_type}`}>
                  {session.user_type}
                </span>
              </td>
              <td>{session.ip_address}</td>
              <td>{parseUserAgent(session.user_agent)}</td>
              <td>{formatRelativeTime(session.last_activity)}</td>
              <td>
                <button 
                  onClick={() => terminateUserSession(session.user_id, session.id)}
                  className="btn-sm btn-danger"
                >
                  Terminate
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      
      <Pagination 
        currentPage={page}
        totalPages={Math.ceil(total / 50)}
        onPageChange={setPage}
      />
    </div>
  );
}
```

### View User's Sessions (Admin)

**Endpoint:** `GET /api/users/{user_id}/sessions`

Shows all sessions for a specific user (admin only).

```javascript
async function getUserSessions(userId) {
  const response = await fetchWithAuth(
    `http://localhost:8000/api/users/${userId}/sessions?active_only=true`
  );
  return await response.json();
}
```

**UI Component:**
```javascript
function UserSessionsModal({ userId, username }) {
  const [sessions, setSessions] = useState([]);
  
  useEffect(() => {
    getUserSessions(userId).then(setSessions);
  }, [userId]);
  
  return (
    <div className="modal">
      <h2>Sessions for {username}</h2>
      <table>
        <thead>
          <tr>
            <th>IP Address</th>
            <th>Device</th>
            <th>Created</th>
            <th>Last Active</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {sessions.map(session => (
            <tr key={session.id}>
              <td>{session.ip_address}</td>
              <td>{parseUserAgent(session.user_agent)}</td>
              <td>{formatDate(session.created_at)}</td>
              <td>{formatDate(session.last_activity)}</td>
              <td>
                <button onClick={() => terminateUserSession(userId, session.id)}>
                  Terminate
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

### Create New User

**Endpoint:** `POST /api/auth/register`

```javascript
async function createUser(userData) {
  const response = await fetchWithAuth('http://localhost:8000/api/auth/register', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      username: userData.username,
      email: userData.email,
      password: userData.password,
      user_type: userData.user_type,      // 'admin', 'super_user', or 'user'
      max_active_sessions: userData.maxSessions || 5
    })
  });
  
  return await response.json();
}
```

**UI Form:**
```javascript
function CreateUserForm({ onSuccess }) {
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    user_type: 'user',
    maxSessions: 5
  });
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const newUser = await createUser(formData);
      alert('User created successfully!');
      onSuccess(newUser);
    } catch (error) {
      alert('Failed to create user: ' + error.message);
    }
  };
  
  return (
    <form onSubmit={handleSubmit}>
      <input
        type="text"
        placeholder="Username"
        value={formData.username}
        onChange={(e) => setFormData({...formData, username: e.target.value})}
        required
      />
      
      <input
        type="email"
        placeholder="Email (optional)"
        value={formData.email}
        onChange={(e) => setFormData({...formData, email: e.target.value})}
      />
      
      <input
        type="password"
        placeholder="Password (min 8 characters)"
        value={formData.password}
        onChange={(e) => setFormData({...formData, password: e.target.value})}
        required
        minLength={8}
      />
      
      <select
        value={formData.user_type}
        onChange={(e) => setFormData({...formData, user_type: e.target.value})}
      >
        <option value="user">User</option>
        <option value="super_user">Super User</option>
        <option value="admin">Admin</option>
      </select>
      
      <input
        type="number"
        placeholder="Max Active Sessions"
        value={formData.maxSessions}
        onChange={(e) => setFormData({...formData, maxSessions: parseInt(e.target.value)})}
        min={1}
        max={100}
      />
      
      <button type="submit">Create User</button>
    </form>
  );
}
```

### Update User

**Endpoint:** `PUT /api/users/{user_id}`

**Requires:** Admin role

```javascript
async function updateUser(userId, updates) {
  const response = await fetchWithAuth(
    `http://localhost:8000/api/users/${userId}`,
    {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(updates)
    }
  );
  
  return await response.json();
}

// Example: Change user type
await updateUser('user-uuid', {
  user_type: 'super_user',
  max_active_sessions: 10
});
```

### Block User

**Endpoint:** `POST /api/users/{user_id}/block`

When you block a user:
- They cannot login
- All their active sessions are immediately terminated
- All refresh tokens are revoked

```javascript
async function blockUser(userId) {
  const confirmed = confirm('Are you sure you want to block this user? All their sessions will be terminated.');
  
  if (!confirmed) return;
  
  const response = await fetchWithAuth(
    `http://localhost:8000/api/users/${userId}/block`,
    { method: 'POST' }
  );
  
  if (response.ok) {
    alert('User blocked successfully');
  }
}
```

### Unblock User

**Endpoint:** `POST /api/users/{user_id}/unblock`

```javascript
async function unblockUser(userId) {
  const response = await fetchWithAuth(
    `http://localhost:8000/api/users/${userId}/unblock`,
    { method: 'POST' }
  );
  
  if (response.ok) {
    alert('User unblocked successfully');
  }
}
```

---

## Permission Management

### Understanding Permissions

**Permission Format:** `resource.action`

Examples:
- `feedback.create` - Can submit anomaly feedback
- `analytics.write` - Can modify analytics settings
- `users.manage` - Can manage other users

**User Type Hierarchy:**
1. **Admin** - Has ALL permissions automatically (no grants needed)
2. **Super User** - Can be granted specific permissions
3. **User** - Needs explicit permission grants for protected actions

### Grant Permission

**Endpoint:** `POST /api/users/{user_id}/permissions`

```javascript
async function grantPermission(userId, permission) {
  const response = await fetchWithAuth(
    `http://localhost:8000/api/users/${userId}/permissions`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        permission: permission
      })
    }
  );
  
  return await response.json();
}

// Example: Grant feedback permission
await grantPermission('user-uuid', 'feedback.create');
```

### Revoke Permission

**Endpoint:** `DELETE /api/users/{user_id}/permissions/{permission}`

```javascript
async function revokePermission(userId, permission) {
  const response = await fetchWithAuth(
    `http://localhost:8000/api/users/${userId}/permissions/${permission}`,
    { method: 'DELETE' }
  );
  
  if (response.ok) {
    alert('Permission revoked');
  }
}
```

### Permission Management UI

```javascript
function PermissionManager({ user, onUpdate }) {
  const availablePermissions = [
    { id: 'feedback.create', name: 'Submit Feedback', description: 'Can submit anomaly feedback' },
    { id: 'analytics.write', name: 'Modify Analytics', description: 'Can change analytics settings' },
    { id: 'users.manage', name: 'Manage Users', description: 'Can manage other users' }
  ];
  
  const handleTogglePermission = async (permission) => {
    const hasPermission = user.permissions.includes(permission.id);
    
    if (hasPermission) {
      await revokePermission(user.id, permission.id);
    } else {
      await grantPermission(user.id, permission.id);
    }
    
    onUpdate(); // Refresh user data
  };
  
  return (
    <div className="permission-manager">
      <h3>Permissions for {user.username}</h3>
      
      {user.user_type === 'admin' && (
        <div className="info-box">
          ℹ️ Admins have all permissions automatically
        </div>
      )}
      
      <div className="permissions-list">
        {availablePermissions.map(permission => {
          const hasPermission = user.user_type === 'admin' || 
                                user.permissions.includes(permission.id);
          const canToggle = user.user_type !== 'admin';
          
          return (
            <div key={permission.id} className="permission-item">
              <div className="permission-info">
                <strong>{permission.name}</strong>
                <p>{permission.description}</p>
              </div>
              
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={hasPermission}
                  disabled={!canToggle}
                  onChange={() => handleTogglePermission(permission)}
                />
                <span className="slider"></span>
              </label>
            </div>
          );
        })}
      </div>
    </div>
  );
}
```

---

## Complete UI Workflows

### Admin Dashboard

```javascript
function AdminDashboard() {
  const [users, setUsers] = useState([]);
  const [selectedUser, setSelectedUser] = useState(null);
  const [filter, setFilter] = useState({});
  
  useEffect(() => {
    loadUsers();
  }, [filter]);
  
  const loadUsers = async () => {
    const data = await listUsers(1, filter);
    setUsers(data.users);
  };
  
  return (
    <div className="admin-dashboard">
      <div className="header">
        <h1>User Management</h1>
        <button onClick={() => setShowCreateUser(true)}>
          Create New User
        </button>
      </div>
      
      <div className="filters">
        <select onChange={(e) => setFilter({...filter, user_type: e.target.value})}>
          <option value="">All Types</option>
          <option value="admin">Admin</option>
          <option value="super_user">Super User</option>
          <option value="user">User</option>
        </select>
        
        <select onChange={(e) => setFilter({...filter, is_blocked: e.target.value})}>
          <option value="">All Statuses</option>
          <option value="false">Active</option>
          <option value="true">Blocked</option>
        </select>
      </div>
      
      <UsersTable users={users} onUserClick={setSelectedUser} />
      
      {selectedUser && (
        <UserDetailsModal
          user={selectedUser}
          onClose={() => setSelectedUser(null)}
          onUpdate={loadUsers}
        />
      )}
    </div>
  );
}
```

### User Details Modal

```javascript
function UserDetailsModal({ user, onClose, onUpdate }) {
  const [sessions, setSessions] = useState([]);
  const [activeTab, setActiveTab] = useState('info');
  
  useEffect(() => {
    if (activeTab === 'sessions') {
      getUserSessions(user.id).then(setSessions);
    }
  }, [activeTab, user.id]);
  
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>{user.username}</h2>
          <button onClick={onClose}>×</button>
        </div>
        
        <div className="tabs">
          <button 
            className={activeTab === 'info' ? 'active' : ''}
            onClick={() => setActiveTab('info')}
          >
            User Info
          </button>
          <button 
            className={activeTab === 'permissions' ? 'active' : ''}
            onClick={() => setActiveTab('permissions')}
          >
            Permissions
          </button>
          <button 
            className={activeTab === 'sessions' ? 'active' : ''}
            onClick={() => setActiveTab('sessions')}
          >
            Sessions ({sessions.length})
          </button>
        </div>
        
        <div className="tab-content">
          {activeTab === 'info' && (
            <div className="user-info">
              <p><strong>User ID:</strong> {user.id}</p>
              <p><strong>Email:</strong> {user.email || 'N/A'}</p>
              <p><strong>Type:</strong> {user.user_type}</p>
              <p><strong>Status:</strong> {user.is_blocked ? 'Blocked' : 'Active'}</p>
              <p><strong>Max Sessions:</strong> {user.max_active_sessions}</p>
              <p><strong>Last Login:</strong> {formatDate(user.last_login_at)}</p>
              
              <div className="actions">
                {user.is_blocked ? (
                  <button onClick={() => {
                    unblockUser(user.id).then(() => {
                      onUpdate();
                      onClose();
                    });
                  }}>
                    Unblock User
                  </button>
                ) : (
                  <button className="danger" onClick={() => {
                    blockUser(user.id).then(() => {
                      onUpdate();
                      onClose();
                    });
                  }}>
                    Block User
                  </button>
                )}
              </div>
            </div>
          )}
          
          {activeTab === 'permissions' && (
            <PermissionManager user={user} onUpdate={onUpdate} />
          )}
          
          {activeTab === 'sessions' && (
            <SessionsList 
              sessions={sessions} 
              onTerminate={(sessionId) => {
                terminateUserSession(user.id, sessionId);
                setSessions(sessions.filter(s => s.id !== sessionId));
              }}
            />
          )}
        </div>
      </div>
    </div>
  );
}
```

### My Sessions Page (User View)

```javascript
function MySessionsPage() {
  const [sessions, setSessions] = useState([]);
  
  useEffect(() => {
    loadSessions();
  }, []);
  
  const loadSessions = async () => {
    const data = await getCurrentUserSessions();
    setSessions(data);
  };
  
  const handleTerminate = async (sessionId) => {
    await terminateSession(sessionId);
    loadSessions();
  };
  
  return (
    <div className="my-sessions-page">
      <h1>My Active Sessions</h1>
      <p>Manage your active sessions across different devices</p>
      
      <div className="sessions-grid">
        {sessions.map(session => (
          <div key={session.id} className="session-card">
            <div className="session-icon">
              {getDeviceIcon(session.user_agent)}
            </div>
            
            <div className="session-details">
              <h3>{getDeviceName(session.user_agent)}</h3>
              <p><strong>IP:</strong> {session.ip_address}</p>
              <p><strong>Location:</strong> {session.ip_address} (estimate)</p>
              <p><strong>Started:</strong> {formatDate(session.created_at)}</p>
              <p><strong>Last Active:</strong> {formatRelativeTime(session.last_activity)}</p>
            </div>
            
            <div className="session-actions">
              {isCurrentSession(session) ? (
                <span className="badge current">Current Session</span>
              ) : (
                <button 
                  className="btn-danger"
                  onClick={() => handleTerminate(session.id)}
                >
                  Terminate
                </button>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
```

---

## Helper Functions

```javascript
// Parse user agent to readable device name
function parseUserAgent(userAgent) {
  if (!userAgent) return 'Unknown Device';
  
  if (userAgent.includes('iPhone')) return 'iPhone';
  if (userAgent.includes('iPad')) return 'iPad';
  if (userAgent.includes('Android')) return 'Android Device';
  if (userAgent.includes('Mac OS')) return 'Mac';
  if (userAgent.includes('Windows')) return 'Windows PC';
  if (userAgent.includes('Linux')) return 'Linux';
  
  return 'Unknown Device';
}

// Get device icon
function getDeviceIcon(userAgent) {
  const device = parseUserAgent(userAgent);
  
  const icons = {
    'iPhone': '📱',
    'iPad': '📱',
    'Android Device': '📱',
    'Mac': '💻',
    'Windows PC': '💻',
    'Linux': '💻',
    'Unknown Device': '🖥️'
  };
  
  return icons[device] || '🖥️';
}

// Format date
function formatDate(dateString) {
  if (!dateString) return 'Never';
  return new Date(dateString).toLocaleString();
}

// Format relative time
function formatRelativeTime(dateString) {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now - date;
  const diffMins = Math.floor(diffMs / 60000);
  
  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins} minutes ago`;
  
  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours} hours ago`;
  
  const diffDays = Math.floor(diffHours / 24);
  return `${diffDays} days ago`;
}

// Check if current session
function isCurrentSession(session) {
  // You could store current session ID in localStorage during login
  const currentSessionId = localStorage.getItem('current_session_id');
  return session.id === currentSessionId;
}
```

---

## Quick Reference

### Common Permission Checks

```javascript
// Check if user can submit feedback
function canSubmitFeedback(user) {
  return user.user_type === 'admin' || 
         user.permissions.includes('feedback.create');
}

// Hide/show UI elements based on permissions
{canSubmitFeedback(currentUser) && (
  <button onClick={submitFeedback}>Submit Feedback</button>
)}
```

### Error Handling

```javascript
async function fetchWithAuth(url, options = {}) {
  try {
    const response = await fetch(url, options);
    
    if (response.status === 401) {
      // Unauthorized - redirect to login
      window.location.href = '/login';
      return;
    }
    
    if (response.status === 403) {
      // Forbidden - no permission
      alert('You do not have permission to perform this action');
      return;
    }
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Request failed');
    }
    
    return response;
  } catch (error) {
    console.error('Request failed:', error);
    throw error;
  }
}
```

---

## Summary

**Key Endpoints:**
- `POST /api/auth/login` - Login
- `POST /api/auth/logout` - Logout  
- `GET /api/auth/me` - Get current user
- `GET /api/auth/sessions` - View my sessions
- `GET /api/users` - List all users (admin)
- `GET /api/admin/sessions` - View ALL sessions system-wide (admin)
- `GET /api/users/{id}/sessions` - View specific user sessions (admin)
- `POST /api/users/{id}/block` - Block user (admin)
- `POST /api/users/{id}/unblock` - Unblock user (admin)
- `POST /api/users/{id}/permissions` - Grant permission (admin)
- `DELETE /api/users/{id}/permissions/{permission}` - Revoke permission (admin)

**User Types:**
- Admin: Full access
- Super User: Can be granted permissions
- User: Needs explicit permissions

**Remember:**
- Store tokens securely
- Handle token refresh automatically
- Show appropriate UI based on user permissions
- Provide clear feedback for blocked users
