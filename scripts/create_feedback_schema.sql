-- Migration script to create feedback schema in PostgreSQL
-- This replaces the SQLite training_ops databases

-- Create feedback schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS feedback;

-- 1. User Feedback Table (Main feedback metadata)
CREATE TABLE IF NOT EXISTS feedback.user_feedback (
    id SERIAL PRIMARY KEY,
    flight_id TEXT UNIQUE NOT NULL,
    tagged_at INTEGER NOT NULL,
    first_seen_ts INTEGER,
    last_seen_ts INTEGER,
    user_label INTEGER NOT NULL,  -- 0: Normal, 1: Anomaly
    comments TEXT,
    rule_id INTEGER,
    rule_ids INTEGER[],  -- Array of rule IDs for multiple selection
    rule_name TEXT,
    rule_names TEXT[],  -- Array of rule names
    other_details TEXT,
    callsign TEXT,
    
    -- Indexes
    CONSTRAINT user_feedback_flight_id_unique UNIQUE (flight_id)
);

-- Create indexes for user_feedback
CREATE INDEX IF NOT EXISTS idx_feedback_user_feedback_flight_id ON feedback.user_feedback(flight_id);
CREATE INDEX IF NOT EXISTS idx_feedback_user_feedback_tagged_at ON feedback.user_feedback(tagged_at);
CREATE INDEX IF NOT EXISTS idx_feedback_user_feedback_rule_id ON feedback.user_feedback(rule_id);
CREATE INDEX IF NOT EXISTS idx_feedback_user_feedback_first_seen_ts ON feedback.user_feedback(first_seen_ts);
CREATE INDEX IF NOT EXISTS idx_feedback_user_feedback_callsign ON feedback.user_feedback(callsign);

-- 2. Flight Metadata Table (Comprehensive flight details)
CREATE TABLE IF NOT EXISTS feedback.flight_metadata (
    flight_id TEXT PRIMARY KEY,
    callsign TEXT,
    flight_number TEXT,
    
    -- Origin & Destination
    origin_airport TEXT,
    destination_airport TEXT,
    
    -- Airline & Aircraft
    airline TEXT,
    aircraft_type TEXT,
    aircraft_model TEXT,
    aircraft_registration TEXT,
    
    -- Flight category
    category TEXT,
    is_military BOOLEAN DEFAULT FALSE,
    
    -- Flight statistics
    first_seen_ts INTEGER,
    last_seen_ts INTEGER,
    flight_duration_sec INTEGER,
    total_points INTEGER,
    
    -- Altitude stats
    min_altitude_ft REAL,
    max_altitude_ft REAL,
    avg_altitude_ft REAL,
    
    -- Speed stats
    min_speed_kts REAL,
    max_speed_kts REAL,
    avg_speed_kts REAL,
    
    -- Position
    start_lat REAL,
    start_lon REAL,
    end_lat REAL,
    end_lon REAL,
    total_distance_nm REAL,
    
    -- Other
    squawk_codes TEXT,  -- JSON array
    emergency_squawk_detected BOOLEAN DEFAULT FALSE,
    
    -- Schedule
    scheduled_departure TEXT,
    scheduled_arrival TEXT,
    actual_departure TEXT,
    actual_arrival TEXT
);

-- Create indexes for flight_metadata
CREATE INDEX IF NOT EXISTS idx_feedback_flight_metadata_flight_id ON feedback.flight_metadata(flight_id);
CREATE INDEX IF NOT EXISTS idx_feedback_flight_metadata_callsign ON feedback.flight_metadata(callsign);
CREATE INDEX IF NOT EXISTS idx_feedback_flight_metadata_origin ON feedback.flight_metadata(origin_airport);
CREATE INDEX IF NOT EXISTS idx_feedback_flight_metadata_dest ON feedback.flight_metadata(destination_airport);
CREATE INDEX IF NOT EXISTS idx_feedback_flight_metadata_airline ON feedback.flight_metadata(airline);
CREATE INDEX IF NOT EXISTS idx_feedback_flight_metadata_first_seen ON feedback.flight_metadata(first_seen_ts);
CREATE INDEX IF NOT EXISTS idx_feedback_flight_metadata_is_military ON feedback.flight_metadata(is_military);

-- 3. Flight Tracks Table (Track points)
CREATE TABLE IF NOT EXISTS feedback.flight_tracks (
    id SERIAL PRIMARY KEY,
    flight_id TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    alt REAL,
    gspeed REAL,
    vspeed REAL,
    track REAL,
    squawk TEXT,
    callsign TEXT,
    source TEXT DEFAULT 'feedback'
);

-- Create indexes for flight_tracks
CREATE INDEX IF NOT EXISTS idx_feedback_flight_tracks_flight_id ON feedback.flight_tracks(flight_id);
CREATE INDEX IF NOT EXISTS idx_feedback_flight_tracks_timestamp ON feedback.flight_tracks(timestamp);
CREATE INDEX IF NOT EXISTS idx_feedback_flight_tracks_flight_ts ON feedback.flight_tracks(flight_id, timestamp);

-- 4. Anomaly Reports Table (Full anomaly report data)
CREATE TABLE IF NOT EXISTS feedback.anomaly_reports (
    id SERIAL PRIMARY KEY,
    flight_id TEXT UNIQUE NOT NULL,
    timestamp INTEGER NOT NULL,
    full_report JSONB,
    
    -- Summary fields
    severity_cnn REAL,
    severity_dense REAL,
    
    -- Rule matches
    matched_rule_ids TEXT,  -- JSON array
    matched_rule_names TEXT,  -- JSON array
    matched_rule_categories TEXT,  -- JSON array
    
    -- Quick reference fields
    callsign TEXT,
    origin_airport TEXT,
    destination_airport TEXT,
    
    CONSTRAINT anomaly_reports_flight_id_unique UNIQUE (flight_id)
);

-- Create indexes for anomaly_reports
CREATE INDEX IF NOT EXISTS idx_feedback_anomaly_reports_flight_id ON feedback.anomaly_reports(flight_id);
CREATE INDEX IF NOT EXISTS idx_feedback_anomaly_reports_timestamp ON feedback.anomaly_reports(timestamp);
CREATE INDEX IF NOT EXISTS idx_feedback_anomaly_reports_severity_cnn ON feedback.anomaly_reports(severity_cnn);
CREATE INDEX IF NOT EXISTS idx_feedback_anomaly_reports_callsign ON feedback.anomaly_reports(callsign);

-- Grant necessary permissions (adjust based on your PostgreSQL user)
-- GRANT ALL PRIVILEGES ON SCHEMA feedback TO your_db_user;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA feedback TO your_db_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA feedback TO your_db_user;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Feedback schema created successfully';
END $$;
