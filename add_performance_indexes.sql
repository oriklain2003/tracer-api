-- ============================================================================
-- Performance Optimization Indexes
-- Add indexes to improve query performance for slow API routes
-- ============================================================================

\echo 'Creating performance indexes...'

-- ============================================================================
-- FEEDBACK SCHEMA INDEXES
-- ============================================================================

-- Primary time-based index for feedback history queries
-- This index handles: WHERE timestamp BETWEEN x AND y ORDER BY timestamp DESC LIMIT n
-- Using NULLS LAST because we COALESCE(first_seen_ts, tagged_at) in queries
CREATE INDEX IF NOT EXISTS idx_user_feedback_timestamp 
    ON feedback.user_feedback(first_seen_ts DESC NULLS LAST);

-- Fallback timestamp index (for when first_seen_ts is NULL)
CREATE INDEX IF NOT EXISTS idx_user_feedback_tagged_at 
    ON feedback.user_feedback(tagged_at DESC NULLS LAST);

-- Index on flight_id for joins with flight_metadata and anomaly_reports
CREATE INDEX IF NOT EXISTS idx_user_feedback_flight_id 
    ON feedback.user_feedback(flight_id);

-- Composite index for filtered queries (anomalies only vs all)
-- This covers: WHERE timestamp BETWEEN x AND y AND user_label = 1 ORDER BY timestamp DESC
-- PostgreSQL can use this for both filtered and unfiltered queries
CREATE INDEX IF NOT EXISTS idx_user_feedback_timestamp_label 
    ON feedback.user_feedback(first_seen_ts DESC NULLS LAST, user_label) 
    WHERE first_seen_ts IS NOT NULL;

-- Partial index for active anomalies (user_label = 1)
-- Smaller index, faster for "anomalies only" queries
CREATE INDEX IF NOT EXISTS idx_user_feedback_anomalies_only
    ON feedback.user_feedback(first_seen_ts DESC NULLS LAST)
    WHERE user_label = 1 AND first_seen_ts IS NOT NULL;

-- Flight tracks - critical for track queries
CREATE INDEX IF NOT EXISTS idx_flight_tracks_flight_id 
    ON feedback.flight_tracks(flight_id);

CREATE INDEX IF NOT EXISTS idx_flight_tracks_flight_timestamp 
    ON feedback.flight_tracks(flight_id, timestamp);

-- Flight metadata - used in joins
CREATE INDEX IF NOT EXISTS idx_flight_metadata_flight_id 
    ON feedback.flight_metadata(flight_id);

-- Anomaly reports - used in feedback history joins  
CREATE INDEX IF NOT EXISTS idx_anomaly_reports_flight_id 
    ON feedback.anomaly_reports(flight_id);

-- ============================================================================
-- RESEARCH SCHEMA INDEXES
-- ============================================================================

-- Normal tracks - used in unified track route
CREATE INDEX IF NOT EXISTS idx_normal_tracks_flight_id 
    ON research.normal_tracks(flight_id);

CREATE INDEX IF NOT EXISTS idx_normal_tracks_flight_timestamp 
    ON research.normal_tracks(flight_id, timestamp);

-- Spatial lookup for lat/lon bounding box (flights_near_point, geo queries)
CREATE INDEX IF NOT EXISTS idx_normal_tracks_lat_lon 
    ON research.normal_tracks(lat, lon);

-- Anomaly tracks - used in unified track route
CREATE INDEX IF NOT EXISTS idx_anomalies_tracks_flight_id 
    ON research.anomalies_tracks(flight_id);

CREATE INDEX IF NOT EXISTS idx_anomalies_tracks_flight_timestamp 
    ON research.anomalies_tracks(flight_id, timestamp);

-- Flight metadata
CREATE INDEX IF NOT EXISTS idx_research_metadata_flight_id 
    ON research.flight_metadata(flight_id);

-- ============================================================================
-- LEARNED TUBES (public schema) - composite indexes for get_all_tubes
-- ============================================================================

-- Filtered by origin + destination + member_count (union-tubes with both params)
CREATE INDEX IF NOT EXISTS idx_learned_tubes_origin_dest_members 
    ON learned_tubes(origin, destination, member_count DESC);

-- Filtered by origin only
CREATE INDEX IF NOT EXISTS idx_learned_tubes_origin_members 
    ON learned_tubes(origin, member_count DESC);

-- Filtered by destination only
CREATE INDEX IF NOT EXISTS idx_learned_tubes_dest_members 
    ON learned_tubes(destination, member_count DESC);

-- ============================================================================
-- ANALYZE TABLES
-- ============================================================================

\echo 'Analyzing tables to update statistics...'

ANALYZE feedback.user_feedback;
ANALYZE feedback.flight_tracks;
ANALYZE feedback.flight_metadata;
ANALYZE feedback.anomaly_reports;
ANALYZE research.normal_tracks;
ANALYZE research.anomalies_tracks;
ANALYZE research.flight_metadata;
ANALYZE learned_tubes;

-- ============================================================================
-- VERIFICATION
-- ============================================================================

\echo ''
\echo 'Indexes created successfully!'
\echo ''
\echo 'Verify indexes with:'
\echo '  SELECT schemaname, tablename, indexname FROM pg_indexes WHERE schemaname IN (''feedback'', ''research'') ORDER BY tablename, indexname;'
\echo ''
\echo 'Check index usage with:'
\echo '  SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch FROM pg_stat_user_indexes WHERE schemaname IN (''feedback'', ''research'') ORDER BY idx_scan DESC;'
