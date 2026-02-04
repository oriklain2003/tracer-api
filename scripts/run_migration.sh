#!/bin/bash
# Run database migration for user management system

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check if POSTGRES_DSN is set
if [ -z "$POSTGRES_DSN" ]; then
    echo "Error: POSTGRES_DSN not set in .env file"
    exit 1
fi

echo "Running user management migration..."
psql "$POSTGRES_DSN" -f migrations/001_create_users_and_sessions.sql

if [ $? -eq 0 ]; then
    echo "✓ Migration completed successfully!"
else
    echo "✗ Migration failed. Check the error messages above."
    exit 1
fi
