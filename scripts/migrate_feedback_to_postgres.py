"""
Migration script to set up PostgreSQL feedback schema.

This script:
1. Creates the feedback schema in PostgreSQL
2. Creates all necessary tables (user_feedback, flight_metadata, flight_tracks, anomaly_reports)
3. Creates indexes for optimal query performance

Run this script once to set up the PostgreSQL feedback system.
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_migration():
    """Run the feedback schema migration."""
    
    # Get PostgreSQL connection string from environment
    pg_dsn = os.getenv(
        "POSTGRES_DSN",
        "postgresql://postgres:Warqi4-sywsow-zozfyc@tracer-db.cb80eku2emy0.eu-north-1.rds.amazonaws.com:5432/tracer"
    )
    
    if not pg_dsn:
        logger.error("POSTGRES_DSN environment variable not set")
        return False
    
    # Read SQL migration script
    sql_file = Path(__file__).parent / "create_feedback_schema.sql"
    if not sql_file.exists():
        logger.error(f"Migration SQL file not found: {sql_file}")
        return False
    
    with open(sql_file, 'r') as f:
        migration_sql = f.read()
    
    try:
        # Connect to PostgreSQL
        logger.info("Connecting to PostgreSQL...")
        conn = psycopg2.connect(pg_dsn)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Run migration
        logger.info("Running migration...")
        cursor.execute(migration_sql)
        
        # Verify tables were created
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'feedback'
            ORDER BY table_name
        """)
        tables = cursor.fetchall()
        
        logger.info("Migration completed successfully!")
        logger.info(f"Created {len(tables)} tables in feedback schema:")
        for table in tables:
            logger.info(f"  - feedback.{table[0]}")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = run_migration()
    sys.exit(0 if success else 1)
