-- Migration: Create Users and Sessions Tables
-- Description: Create tables for user management, authentication, sessions, and permissions

-- Create auth schema
CREATE SCHEMA IF NOT EXISTS auth;

-- Create user_type enum in auth schema
CREATE TYPE auth.user_type AS ENUM ('admin', 'super_user', 'user');

-- Create users table
CREATE TABLE IF NOT EXISTS auth.users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(255) NOT NULL UNIQUE,
    email VARCHAR(255) UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    user_type auth.user_type NOT NULL DEFAULT 'user',
    is_blocked BOOLEAN NOT NULL DEFAULT false,
    max_active_sessions INTEGER NOT NULL DEFAULT 5,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    last_login_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes for users table
CREATE INDEX idx_users_username ON auth.users(username);
CREATE INDEX idx_users_email ON auth.users(email) WHERE email IS NOT NULL;
CREATE INDEX idx_users_user_type ON auth.users(user_type);
CREATE INDEX idx_users_is_blocked ON auth.users(is_blocked);

-- Create user_permissions table
CREATE TABLE IF NOT EXISTS auth.user_permissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    permission VARCHAR(255) NOT NULL,
    granted_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    granted_by UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    UNIQUE(user_id, permission)
);

-- Create indexes for user_permissions table
CREATE INDEX idx_user_permissions_user_id ON auth.user_permissions(user_id);
CREATE INDEX idx_user_permissions_permission ON auth.user_permissions(permission);

-- Create sessions table
CREATE TABLE IF NOT EXISTS auth.sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    token VARCHAR(255) NOT NULL UNIQUE,
    ip_address VARCHAR(45),
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    last_activity TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    is_active BOOLEAN NOT NULL DEFAULT true
);

-- Create indexes for sessions table
CREATE INDEX idx_sessions_user_id ON auth.sessions(user_id);
CREATE INDEX idx_sessions_token ON auth.sessions(token);
CREATE INDEX idx_sessions_expires_at ON auth.sessions(expires_at);
CREATE INDEX idx_sessions_user_id_is_active ON auth.sessions(user_id, is_active);

-- Create refresh_tokens table
CREATE TABLE IF NOT EXISTS auth.refresh_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) NOT NULL UNIQUE,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    is_revoked BOOLEAN NOT NULL DEFAULT false
);

-- Create indexes for refresh_tokens table
CREATE INDEX idx_refresh_tokens_token_hash ON auth.refresh_tokens(token_hash);
CREATE INDEX idx_refresh_tokens_user_id ON auth.refresh_tokens(user_id);
CREATE INDEX idx_refresh_tokens_expires_at ON auth.refresh_tokens(expires_at);

-- Create function to update updated_at timestamp in auth schema
CREATE OR REPLACE FUNCTION auth.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to automatically update updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON auth.users
    FOR EACH ROW EXECUTE FUNCTION auth.update_updated_at_column();

-- Comments for documentation
COMMENT ON SCHEMA auth IS 'User authentication and authorization schema';
COMMENT ON TABLE auth.users IS 'Stores user account information';
COMMENT ON TABLE auth.user_permissions IS 'Stores user-specific permissions (overrides role defaults)';
COMMENT ON TABLE auth.sessions IS 'Stores active user sessions for session-based authentication';
COMMENT ON TABLE auth.refresh_tokens IS 'Stores JWT refresh tokens';
COMMENT ON COLUMN auth.users.max_active_sessions IS 'Maximum number of concurrent active sessions allowed';
COMMENT ON COLUMN auth.sessions.token IS 'Session token hash for validation';
COMMENT ON COLUMN auth.refresh_tokens.token_hash IS 'SHA256 hash of refresh token';
