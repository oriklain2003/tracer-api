# Database Connection Pool Exhaustion Fix

## Problem
The application was experiencing PostgreSQL connection errors:
```
PostgreSQL operational error: server closed the connection unexpectedly
This probably means the server terminated abnormally before or while processing the request.
```

This happened because:
1. **Dead connections returned to pool**: When a connection died during a query, it was being returned to the pool instead of being closed
2. **Double putconn calls**: In some error scenarios, connections were returned to the pool twice
3. **Uncommitted transactions**: Connections with open transactions were returned to pool, holding locks
4. **No connection validation**: Dead connections could accumulate in the pool

## Solution

### 1. Enhanced Connection Context Manager (`pg_provider.py`)

**Before:**
- Basic error handling
- Potential double-return of connections
- No tracking of whether connection was already returned

**After:**
- Added `connection_returned` flag to prevent double-returns
- Proper handling of `OperationalError` - connections are closed, not returned to pool
- Always rollback uncommitted transactions before returning to pool
- Check if connection is closed before attempting rollback
- Better error logging at each stage

### 2. Connection Lifecycle

**Success Path:**
1. Get connection from pool
2. Validate it's not closed
3. Yield to user code
4. On success: rollback any uncommitted transaction
5. Return to pool

**Error Path (OperationalError):**
1. Connection is dead
2. Close connection (putconn with close=True)
3. Mark as returned
4. Raise exception

**Error Path (Other):**
1. Attempt rollback if connection alive
2. Return to pool if successful
3. Close connection if dead
4. Mark as returned
5. Raise exception

### 3. Pool Health Monitoring

Added health check to `get_pool_status()`:
```python
{
    "status": "active" | "unhealthy",
    "connection_healthy": true | false,
    "min_connections": 2,
    "max_connections": 10,
    "dsn": "hidden"
}
```

## Configuration

Pool can be configured via environment variables:
```bash
PG_POOL_MIN_CONNECTIONS=5   # Default: 2
PG_POOL_MAX_CONNECTIONS=20  # Default: 10
```

**Recommendations:**
- Low traffic: MIN=2, MAX=5
- Medium traffic: MIN=5, MAX=15
- High traffic: MIN=10, MAX=30

## Testing

To test the connection pool health:
```bash
curl http://localhost:5555/api/health
```

Monitor for:
- `connection_healthy: true`
- No "PostgreSQL operational error" in logs
- No connection exhaustion over time

## Additional Notes

- All write operations in codebase already use `conn.commit()`
- All connection uses properly use `with` context manager
- Connection timeouts: 10s connect, 30s statement
- ThreadedConnectionPool is thread-safe
