-- ============================================================================
-- Test: Index Performance on Time Fields
-- This demonstrates why indexes on timestamp fields are beneficial
-- ============================================================================

\echo ''
\echo '=========================================================================='
\echo 'Testing Query Performance: WITH vs WITHOUT Time Index'
\echo '=========================================================================='
\echo ''

-- First, check current table size
\echo 'Current table size:'
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size,
    n_live_tup AS row_count
FROM pg_stat_user_tables 
WHERE schemaname = 'feedback' AND tablename = 'user_feedback';

\echo ''
\echo '--------------------------------------------------------------------------'
\echo 'Test 1: Query WITHOUT index (current state)'
\echo '--------------------------------------------------------------------------'
\echo ''

-- Drop index temporarily if it exists (to test without it)
DROP INDEX IF EXISTS feedback.idx_user_feedback_timestamp;

-- Clear cache to ensure fair test
SELECT pg_sleep(0.5);

-- Test query (same as your actual query)
EXPLAIN (ANALYZE, BUFFERS, TIMING ON) 
SELECT 
    uf.flight_id,
    uf.first_seen_ts,
    uf.user_label,
    fm.callsign
FROM feedback.user_feedback uf
LEFT JOIN feedback.flight_metadata fm ON uf.flight_id = fm.flight_id
WHERE COALESCE(uf.first_seen_ts, uf.tagged_at) >= EXTRACT(EPOCH FROM NOW() - INTERVAL '30 days')::INTEGER
ORDER BY COALESCE(uf.first_seen_ts, uf.tagged_at) DESC
LIMIT 100;

\echo ''
\echo '--------------------------------------------------------------------------'
\echo 'Test 2: Query WITH index'
\echo '--------------------------------------------------------------------------'
\echo ''

-- Create the index
CREATE INDEX idx_user_feedback_timestamp 
    ON feedback.user_feedback(first_seen_ts DESC NULLS LAST);

-- Analyze table to update statistics
ANALYZE feedback.user_feedback;

-- Clear cache
SELECT pg_sleep(0.5);

-- Same query - should be much faster now
EXPLAIN (ANALYZE, BUFFERS, TIMING ON) 
SELECT 
    uf.flight_id,
    uf.first_seen_ts,
    uf.user_label,
    fm.callsign
FROM feedback.user_feedback uf
LEFT JOIN feedback.flight_metadata fm ON uf.flight_id = fm.flight_id
WHERE COALESCE(uf.first_seen_ts, uf.tagged_at) >= EXTRACT(EPOCH FROM NOW() - INTERVAL '30 days')::INTEGER
ORDER BY COALESCE(uf.first_seen_ts, uf.tagged_at) DESC
LIMIT 100;

\echo ''
\echo '--------------------------------------------------------------------------'
\echo 'Index Information'
\echo '--------------------------------------------------------------------------'
\echo ''

-- Show index size and statistics
SELECT 
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size,
    idx_scan AS times_used,
    idx_tup_read AS tuples_read,
    idx_tup_fetch AS tuples_fetched
FROM pg_stat_user_indexes 
WHERE schemaname = 'feedback' 
  AND tablename = 'user_feedback'
  AND indexname = 'idx_user_feedback_timestamp';

\echo ''
\echo '=========================================================================='
\echo 'Analysis Summary'
\echo '=========================================================================='
\echo ''
\echo 'Look for these improvements in Test 2:'
\echo '  1. "Index Scan" instead of "Seq Scan" - using index!'
\echo '  2. Lower "actual time" - faster execution'
\echo '  3. Fewer "Buffers: shared hit/read" - less I/O'
\echo '  4. No "Sort" node - index already sorted!'
\echo ''
\echo 'Expected improvements:'
\echo '  - Small table (<10k rows):  2-10x faster'
\echo '  - Medium table (10k-100k):  10-50x faster'  
\echo '  - Large table (>100k rows): 50-1000x faster'
\echo ''
\echo 'Index overhead:'
\echo '  - Storage: ~500KB - 5MB (negligible)'
\echo '  - INSERT impact: +5-10% slower (acceptable for read-heavy workload)'
\echo '  - Maintenance: Automatic (PostgreSQL handles it)'
\echo ''
