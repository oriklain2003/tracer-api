#!/usr/bin/env python3
"""
Upload Learned Paths, Tubes, SIDs, and STARs to PostgreSQL

This script reads the learned behavior JSON files and uploads them to PostgreSQL
with optimized table structures and indexes for fast querying.

Usage:
    python scripts/upload_learned_data_to_postgres.py [--drop-tables]

Options:
    --drop-tables    Drop existing tables before creating new ones (WARNING: deletes all data)
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
import argparse
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import psycopg2
    from psycopg2.extras import execute_batch
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
except ImportError:
    print("ERROR: psycopg2 is required. Install it with: pip install psycopg2-binary")
    sys.exit(1)


# File paths
RULES_DIR = Path(__file__).resolve().parent.parent / "rules"
PATHS_FILE = RULES_DIR / "learned_paths.json"
TUBES_FILE = RULES_DIR / "learned_tubes.json"
SID_FILE = RULES_DIR / "learned_sid.json"
STAR_FILE = RULES_DIR / "learned_star.json"


def get_db_connection():
    """Get PostgreSQL connection from environment variables."""
    # dsn = os.getenv("POSTGRES_DSN")
    dsn = "postgresql://postgres:Warqi4-sywsow-zozfyc@tracer-db.cb80eku2emy0.eu-north-1.rds.amazonaws.com:5432/tracer"
    if not dsn:
        raise ValueError("POSTGRES_DSN environment variable is required")
    
    conn = psycopg2.connect(dsn)
    return conn


def create_tables(conn, drop_existing: bool = False):
    """
    Create optimized PostgreSQL tables for learned data.
    
    Tables are optimized with:
    - Proper indexes on frequently queried columns (origin, destination, airport)
    - JSONB for flexible geometry storage with GiST indexes
    - Composite indexes for common query patterns
    """
    cursor = conn.cursor()
    
    if drop_existing:
        print("⚠️  Dropping existing tables...")
        cursor.execute("DROP TABLE IF EXISTS learned_paths CASCADE")
        cursor.execute("DROP TABLE IF EXISTS learned_tubes CASCADE")
        cursor.execute("DROP TABLE IF EXISTS learned_sids CASCADE")
        cursor.execute("DROP TABLE IF EXISTS learned_stars CASCADE")
        conn.commit()
        print("✓ Existing tables dropped")
    
    print("Creating tables...")
    
    # =========================================================================
    # LEARNED PATHS TABLE
    # =========================================================================
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS learned_paths (
            id VARCHAR(100) PRIMARY KEY,
            origin VARCHAR(10) NOT NULL,
            destination VARCHAR(10) NOT NULL,
            centerline JSONB NOT NULL,
            width_nm FLOAT DEFAULT 4.0,
            member_count INTEGER DEFAULT 0,
            min_alt_ft FLOAT,
            max_alt_ft FLOAT,
            created_at TIMESTAMP DEFAULT NOW(),
            
            -- Composite index for origin-destination queries
            CONSTRAINT valid_origin_dest CHECK (origin != '' AND destination != '')
        )
    """)
    
    # Indexes for paths
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_paths_origin 
        ON learned_paths(origin)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_paths_destination 
        ON learned_paths(destination)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_paths_od_pair 
        ON learned_paths(origin, destination)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_paths_member_count 
        ON learned_paths(member_count DESC)
    """)
    
    # GiST index for JSONB centerline (for spatial queries if needed)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_paths_centerline_gin 
        ON learned_paths USING GIN(centerline)
    """)
    
    # =========================================================================
    # LEARNED TUBES TABLE
    # =========================================================================
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS learned_tubes (
            id VARCHAR(100) PRIMARY KEY,
            origin VARCHAR(10) NOT NULL,
            destination VARCHAR(10) NOT NULL,
            geometry JSONB NOT NULL,
            min_alt_ft FLOAT,
            max_alt_ft FLOAT,
            member_count INTEGER DEFAULT 0,
            buffer_nm FLOAT DEFAULT 1.0,
            alpha FLOAT DEFAULT 0.5,
            created_at TIMESTAMP DEFAULT NOW(),
            
            CONSTRAINT valid_tube_origin_dest CHECK (origin != '' AND destination != '')
        )
    """)
    
    # Indexes for tubes
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_tubes_origin 
        ON learned_tubes(origin)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_tubes_destination 
        ON learned_tubes(destination)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_tubes_od_pair 
        ON learned_tubes(origin, destination)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_tubes_member_count 
        ON learned_tubes(member_count DESC)
    """)
    
    # GiST index for JSONB geometry
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_tubes_geometry_gin 
        ON learned_tubes USING GIN(geometry)
    """)
    
    # =========================================================================
    # LEARNED SIDS TABLE
    # =========================================================================
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS learned_sids (
            id VARCHAR(100) PRIMARY KEY,
            airport VARCHAR(10) NOT NULL,
            type VARCHAR(10) DEFAULT 'SID',
            centerline JSONB NOT NULL,
            width_nm FLOAT DEFAULT 6.0,
            member_count INTEGER DEFAULT 0,
            runway VARCHAR(10),
            created_at TIMESTAMP DEFAULT NOW(),
            
            CONSTRAINT valid_sid_airport CHECK (airport != '')
        )
    """)
    
    # Indexes for SIDs
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_sids_airport 
        ON learned_sids(airport)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_sids_member_count 
        ON learned_sids(member_count DESC)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_sids_centerline_gin 
        ON learned_sids USING GIN(centerline)
    """)
    
    # =========================================================================
    # LEARNED STARS TABLE
    # =========================================================================
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS learned_stars (
            id VARCHAR(100) PRIMARY KEY,
            airport VARCHAR(10) NOT NULL,
            type VARCHAR(10) DEFAULT 'STAR',
            centerline JSONB NOT NULL,
            width_nm FLOAT DEFAULT 6.0,
            member_count INTEGER DEFAULT 0,
            runway VARCHAR(10),
            created_at TIMESTAMP DEFAULT NOW(),
            
            CONSTRAINT valid_star_airport CHECK (airport != '')
        )
    """)
    
    # Indexes for STARs
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_stars_airport 
        ON learned_stars(airport)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_stars_member_count 
        ON learned_stars(member_count DESC)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_stars_centerline_gin 
        ON learned_stars USING GIN(centerline)
    """)
    
    conn.commit()
    cursor.close()
    print("✓ Tables and indexes created successfully")


def upload_paths(conn):
    """Upload learned paths to PostgreSQL."""
    if not PATHS_FILE.exists():
        print(f"⚠️  Paths file not found: {PATHS_FILE}")
        return 0
    
    print(f"\n📂 Reading paths from {PATHS_FILE}...")
    with open(PATHS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    paths = data.get("paths", [])
    if not paths:
        print("⚠️  No paths found in file")
        return 0
    
    print(f"Found {len(paths)} paths")
    
    cursor = conn.cursor()
    
    # Prepare batch insert
    insert_query = """
        INSERT INTO learned_paths 
        (id, origin, destination, centerline, width_nm, member_count, min_alt_ft, max_alt_ft)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            origin = EXCLUDED.origin,
            destination = EXCLUDED.destination,
            centerline = EXCLUDED.centerline,
            width_nm = EXCLUDED.width_nm,
            member_count = EXCLUDED.member_count,
            min_alt_ft = EXCLUDED.min_alt_ft,
            max_alt_ft = EXCLUDED.max_alt_ft
    """
    
    batch_data = []
    for path in paths:
        batch_data.append((
            path.get("id", "unknown"),
            path.get("origin", ""),
            path.get("destination", ""),
            json.dumps(path.get("centerline", [])),  # Store as JSONB
            path.get("width_nm", 4.0),
            path.get("member_count", 0),
            path.get("min_alt_ft"),
            path.get("max_alt_ft")
        ))
    
    # Batch insert for performance
    execute_batch(cursor, insert_query, batch_data, page_size=1000)
    conn.commit()
    cursor.close()
    
    print(f"✓ Uploaded {len(batch_data)} paths")
    return len(batch_data)


def upload_tubes(conn):
    """Upload learned tubes to PostgreSQL."""
    if not TUBES_FILE.exists():
        print(f"⚠️  Tubes file not found: {TUBES_FILE}")
        return 0
    
    print(f"\n📂 Reading tubes from {TUBES_FILE}...")
    with open(TUBES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    tubes = data.get("tubes", [])
    if not tubes:
        print("⚠️  No tubes found in file")
        return 0
    
    print(f"Found {len(tubes)} tubes")
    
    cursor = conn.cursor()
    
    insert_query = """
        INSERT INTO learned_tubes 
        (id, origin, destination, geometry, min_alt_ft, max_alt_ft, member_count, buffer_nm, alpha)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            origin = EXCLUDED.origin,
            destination = EXCLUDED.destination,
            geometry = EXCLUDED.geometry,
            min_alt_ft = EXCLUDED.min_alt_ft,
            max_alt_ft = EXCLUDED.max_alt_ft,
            member_count = EXCLUDED.member_count,
            buffer_nm = EXCLUDED.buffer_nm,
            alpha = EXCLUDED.alpha
    """
    
    batch_data = []
    for tube in tubes:
        batch_data.append((
            tube.get("id", "unknown"),
            tube.get("origin", ""),
            tube.get("destination", ""),
            json.dumps(tube.get("geometry", [])),  # Store as JSONB
            tube.get("min_alt_ft"),
            tube.get("max_alt_ft"),
            tube.get("member_count", 0),
            tube.get("buffer_nm", 1.0),
            tube.get("alpha", 0.5)
        ))
    
    execute_batch(cursor, insert_query, batch_data, page_size=1000)
    conn.commit()
    cursor.close()
    
    print(f"✓ Uploaded {len(batch_data)} tubes")
    return len(batch_data)


def upload_sids(conn):
    """Upload learned SIDs to PostgreSQL."""
    if not SID_FILE.exists():
        print(f"⚠️  SIDs file not found: {SID_FILE}")
        return 0
    
    print(f"\n📂 Reading SIDs from {SID_FILE}...")
    with open(SID_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    sids = data.get("procedures", [])
    if not sids:
        print("⚠️  No SIDs found in file")
        return 0
    
    print(f"Found {len(sids)} SIDs")
    
    cursor = conn.cursor()
    
    insert_query = """
        INSERT INTO learned_sids 
        (id, airport, type, centerline, width_nm, member_count, runway)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            airport = EXCLUDED.airport,
            type = EXCLUDED.type,
            centerline = EXCLUDED.centerline,
            width_nm = EXCLUDED.width_nm,
            member_count = EXCLUDED.member_count,
            runway = EXCLUDED.runway
    """
    
    batch_data = []
    for sid in sids:
        batch_data.append((
            sid.get("id", "unknown"),
            sid.get("airport", ""),
            sid.get("type", "SID"),
            json.dumps(sid.get("centerline", [])),
            sid.get("width_nm", 6.0),
            sid.get("member_count", 0),
            sid.get("runway")
        ))
    
    execute_batch(cursor, insert_query, batch_data, page_size=1000)
    conn.commit()
    cursor.close()
    
    print(f"✓ Uploaded {len(batch_data)} SIDs")
    return len(batch_data)


def upload_stars(conn):
    """Upload learned STARs to PostgreSQL."""
    if not STAR_FILE.exists():
        print(f"⚠️  STARs file not found: {STAR_FILE}")
        return 0
    
    print(f"\n📂 Reading STARs from {STAR_FILE}...")
    with open(STAR_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    stars = data.get("procedures", [])
    if not stars:
        print("⚠️  No STARs found in file")
        return 0
    
    print(f"Found {len(stars)} STARs")
    
    cursor = conn.cursor()
    
    insert_query = """
        INSERT INTO learned_stars 
        (id, airport, type, centerline, width_nm, member_count, runway)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            airport = EXCLUDED.airport,
            type = EXCLUDED.type,
            centerline = EXCLUDED.centerline,
            width_nm = EXCLUDED.width_nm,
            member_count = EXCLUDED.member_count,
            runway = EXCLUDED.runway
    """
    
    batch_data = []
    for star in stars:
        batch_data.append((
            star.get("id", "unknown"),
            star.get("airport", ""),
            star.get("type", "STAR"),
            json.dumps(star.get("centerline", [])),
            star.get("width_nm", 6.0),
            star.get("member_count", 0),
            star.get("runway")
        ))
    
    execute_batch(cursor, insert_query, batch_data, page_size=1000)
    conn.commit()
    cursor.close()
    
    print(f"✓ Uploaded {len(batch_data)} STARs")
    return len(batch_data)


def print_statistics(conn):
    """Print statistics about uploaded data."""
    cursor = conn.cursor()
    
    print("\n" + "="*60)
    print("📊 DATABASE STATISTICS")
    print("="*60)
    
    # Paths statistics
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(DISTINCT origin) as unique_origins,
            COUNT(DISTINCT destination) as unique_destinations,
            AVG(member_count) as avg_member_count,
            MAX(member_count) as max_member_count
        FROM learned_paths
    """)
    stats = cursor.fetchone()
    print(f"\n📍 PATHS:")
    print(f"  Total: {stats[0]}")
    print(f"  Unique Origins: {stats[1]}")
    print(f"  Unique Destinations: {stats[2]}")
    print(f"  Avg Member Count: {stats[3]:.1f}")
    print(f"  Max Member Count: {stats[4]}")
    
    # Tubes statistics
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(DISTINCT origin) as unique_origins,
            COUNT(DISTINCT destination) as unique_destinations,
            AVG(member_count) as avg_member_count
        FROM learned_tubes
    """)
    stats = cursor.fetchone()
    print(f"\n🔵 TUBES:")
    print(f"  Total: {stats[0]}")
    print(f"  Unique Origins: {stats[1]}")
    print(f"  Unique Destinations: {stats[2]}")
    print(f"  Avg Member Count: {stats[3]:.1f}")
    
    # SIDs statistics
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(DISTINCT airport) as unique_airports,
            AVG(member_count) as avg_member_count
        FROM learned_sids
    """)
    stats = cursor.fetchone()
    print(f"\n🛫 SIDS:")
    print(f"  Total: {stats[0]}")
    print(f"  Unique Airports: {stats[1]}")
    print(f"  Avg Member Count: {stats[3]:.1f}")
    
    # STARs statistics
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(DISTINCT airport) as unique_airports,
            AVG(member_count) as avg_member_count
        FROM learned_stars
    """)
    stats = cursor.fetchone()
    print(f"\n🛬 STARS:")
    print(f"  Total: {stats[0]}")
    print(f"  Unique Airports: {stats[1]}")
    print(f"  Avg Member Count: {stats[3]:.1f}")
    
    cursor.close()
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Upload learned paths, tubes, SIDs, and STARs to PostgreSQL"
    )
    parser.add_argument(
        "--drop-tables",
        action="store_true",
        help="Drop existing tables before uploading (WARNING: deletes all data)"
    )
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 LEARNED DATA UPLOAD TO POSTGRESQL")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.drop_tables:
        print("\n⚠️  WARNING: --drop-tables flag is set")
        print("This will delete all existing learned data!")
        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted.")
            return
    
    try:
        # Connect to database
        print("\n🔌 Connecting to PostgreSQL...")
        conn = get_db_connection()
        print("✓ Connected successfully")
        
        # Create tables
        create_tables(conn, drop_existing=args.drop_tables)
        
        # Upload data
        total_uploaded = 0
        total_uploaded += upload_paths(conn)
        total_uploaded += upload_tubes(conn)
        total_uploaded += upload_sids(conn)
        total_uploaded += upload_stars(conn)
        
        # Print statistics
        print_statistics(conn)
        
        # Close connection
        conn.close()
        
        print(f"\n✅ SUCCESS! Uploaded {total_uploaded} total records")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
