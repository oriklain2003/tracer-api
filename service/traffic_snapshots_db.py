"""
Traffic Snapshots Database
Stores the last 3 traffic data snapshots for route planning
"""
import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class TrafficSnapshotsDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize the traffic snapshots database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS traffic_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                aircraft_count INTEGER NOT NULL,
                data JSON NOT NULL,
                created_at INTEGER NOT NULL
            )
        """)
        
        # Create index on timestamp for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON traffic_snapshots(timestamp DESC)
        """)
        
        conn.commit()
        conn.close()
    
    def save_snapshot(self, name: str, timestamp: int, traffic_data: List[Dict[str, Any]]) -> int:
        """
        Save a traffic snapshot and maintain only the last 3
        Returns the snapshot ID
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            aircraft_count = len(traffic_data)
            created_at = int(datetime.utcnow().timestamp())
            
            # Insert new snapshot
            cursor.execute("""
                INSERT INTO traffic_snapshots (name, timestamp, aircraft_count, data, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (name, timestamp, aircraft_count, json.dumps(traffic_data), created_at))
            
            snapshot_id = cursor.lastrowid
            
            # Keep only the last 3 snapshots
            cursor.execute("""
                DELETE FROM traffic_snapshots
                WHERE id NOT IN (
                    SELECT id FROM traffic_snapshots
                    ORDER BY created_at DESC
                    LIMIT 3
                )
            """)
            
            conn.commit()
            logger.info(f"Saved traffic snapshot '{name}' with {aircraft_count} aircraft")
            return snapshot_id
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to save traffic snapshot: {e}")
            raise
        finally:
            conn.close()
    
    def get_snapshots(self) -> List[Dict[str, Any]]:
        """Get list of all snapshots (metadata only, no full data)"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT id, name, timestamp, aircraft_count, created_at
                FROM traffic_snapshots
                ORDER BY created_at DESC
                LIMIT 3
            """)
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
        finally:
            conn.close()
    
    def get_snapshot(self, snapshot_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific snapshot with full data"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT id, name, timestamp, aircraft_count, data, created_at
                FROM traffic_snapshots
                WHERE id = ?
            """, (snapshot_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            snapshot = dict(row)
            snapshot['data'] = json.loads(snapshot['data'])
            return snapshot
            
        finally:
            conn.close()
    
    def delete_snapshot(self, snapshot_id: int) -> bool:
        """Delete a specific snapshot"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            cursor.execute("DELETE FROM traffic_snapshots WHERE id = ?", (snapshot_id,))
            conn.commit()
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info(f"Deleted traffic snapshot {snapshot_id}")
            return deleted
            
        finally:
            conn.close()
