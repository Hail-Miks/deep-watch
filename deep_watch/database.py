"""
Database module for Deep Watch incident logging.
Manages SQL Server connections and incident records.
"""

import os
import pyodbc
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Database connection parameters
DB_SERVER = os.environ.get("DB_SERVER", "localhost\\SQLEXPRESS")
DB_NAME = os.environ.get("DB_NAME", "deepwatch")
DB_DRIVER = os.environ.get("DB_DRIVER", "ODBC Driver 17 for SQL Server")
DB_TRUSTED = os.environ.get("DB_TRUSTED", "yes").lower() in {
    "1", "true", "yes"}
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")

# Connection string with TrustServerCertificate
_conn_parts = [
    f"Driver={{{DB_DRIVER}}}",
    f"Server={DB_SERVER}",
    f"Database={DB_NAME}",
    "TrustServerCertificate=yes",
    "Encrypt=yes",
]

if DB_TRUSTED or (not DB_USER and not DB_PASSWORD):
    _conn_parts.append("Trusted_Connection=yes")
else:
    _conn_parts.append(f"UID={DB_USER}")
    _conn_parts.append(f"PWD={DB_PASSWORD}")

CONNECTION_STRING = ";".join(_conn_parts) + ";"


class DatabaseConnection:
    """Manages SQL Server database connections."""

    _instance = None
    _connection = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseConnection, cls).__new__(cls)
        return cls._instance

    def get_connection(self) -> Optional[pyodbc.Connection]:
        """Get or create a database connection."""
        try:
            if self._connection is None or getattr(self._connection, "closed", False):
                self._connection = pyodbc.connect(CONNECTION_STRING)
                logger.info(f"Connected to SQL Server: {DB_SERVER}")
            return self._connection
        except pyodbc.Error as e:
            logger.error(f"Database connection error: {e}")
            return None

    def close(self):
        """Close the database connection."""
        if self._connection and not getattr(self._connection, "closed", False):
            self._connection.close()
            logger.info("Database connection closed")


class IncidentLogger:
    """Logs drowning incidents to the database."""

    def __init__(self):
        self.db = DatabaseConnection()
        self.last_logged_ids = set()  # Track IDs already logged in current alert

    def log_incident(
        self,
        track_ids: List[int],
        description: str = "Drowning Detected",
    ) -> Optional[int]:
        """
        Log a drowning incident to the database.

        Args:
            track_ids: List of person tracking IDs
            description: Description of the incident

        Returns:
            Incident ID if successful, None otherwise
        """
        try:
            conn = self.db.get_connection()
            if conn is None:
                logger.error("Cannot log incident: no database connection")
                return None

            cursor = conn.cursor()

            # Convert track IDs to comma-separated string
            track_ids_str = ",".join(str(tid) for tid in track_ids)

            # Insert incident record
            query = """
            INSERT INTO Incidents 
            (Timestamp, TrackIDs, Description)
            VALUES (?, ?, ?)
            """

            cursor.execute(
                query,
                (
                    datetime.now(timezone(timedelta(hours=8))),
                    track_ids_str,
                    description,
                ),
            )

            conn.commit()

            # Get the inserted ID
            cursor.execute("SELECT @@IDENTITY")
            incident_id = cursor.fetchone()[0]

            logger.info(
                f"Logged incident #{incident_id}: TrackIDs={track_ids_str}, Description={description}"
            )

            return int(incident_id)

        except pyodbc.Error as e:
            logger.error(f"Error logging incident: {e}")
            return None

    def get_all_incidents(self, limit: int = 100) -> List[Dict]:
        """
        Retrieve all incidents from the database.

        Args:
            limit: Maximum number of incidents to retrieve

        Returns:
            List of incident dictionaries
        """
        try:
            conn = self.db.get_connection()
            if conn is None:
                logger.error("Cannot fetch incidents: no database connection")
                return []

            cursor = conn.cursor()

            query = """
            SELECT TOP (?)
                ID, Timestamp, TrackIDs, Description, CreatedAt
            FROM Incidents
            ORDER BY Timestamp DESC
            """

            cursor.execute(query, (limit,))
            rows = cursor.fetchall()

            incidents = []
            for row in rows:
                incidents.append({
                    "id": row[0],
                    "timestamp": row[1].isoformat() if row[1] else None,
                    "track_ids": row[2],
                    "description": row[3],
                    "created_at": row[4].isoformat() if row[4] else None,
                })

            return incidents

        except pyodbc.Error as e:
            logger.error(f"Error fetching incidents: {e}")
            return []

    def get_incident_by_id(self, incident_id: int) -> Optional[Dict]:
        """
        Retrieve a specific incident by ID.

        Args:
            incident_id: The incident ID

        Returns:
            Incident dictionary or None if not found
        """
        try:
            conn = self.db.get_connection()
            if conn is None:
                logger.error("Cannot fetch incident: no database connection")
                return None

            cursor = conn.cursor()

            query = """
            SELECT ID, Timestamp, TrackIDs, Description, CreatedAt
            FROM Incidents
            WHERE ID = ?
            """

            cursor.execute(query, (incident_id,))
            row = cursor.fetchone()

            if row:
                return {
                    "id": row[0],
                    "timestamp": row[1].isoformat() if row[1] else None,
                    "track_ids": row[2],
                    "description": row[3],
                    "created_at": row[5].isoformat() if row[5] else None,
                }

            return None

        except pyodbc.Error as e:
            logger.error(f"Error fetching incident {incident_id}: {e}")
            return None

    def delete_incident(self, incident_id: int) -> bool:
        """
        Delete an incident from the database.

        Args:
            incident_id: The incident ID to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            conn = self.db.get_connection()
            if conn is None:
                logger.error("Cannot delete incident: no database connection")
                return False

            cursor = conn.cursor()

            query = "DELETE FROM Incidents WHERE ID = ?"
            cursor.execute(query, (incident_id,))
            conn.commit()

            logger.info(f"Deleted incident #{incident_id}")
            return True

        except pyodbc.Error as e:
            logger.error(f"Error deleting incident {incident_id}: {e}")
            return False


# Global instance
_incident_logger = None


def get_incident_logger() -> IncidentLogger:
    """Get the global incident logger instance."""
    global _incident_logger
    if _incident_logger is None:
        _incident_logger = IncidentLogger()
    return _incident_logger
