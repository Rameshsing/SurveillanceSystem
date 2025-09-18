# db.py
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any

# Defaults
DEFAULT_DIR = Path(__file__).resolve().parent / "logs"
DEFAULT_DB = DEFAULT_DIR / "analytics.db"

# DDL for the canonical table used by the app
DDL_TABLE = """
CREATE TABLE IF NOT EXISTS traffic_logs (
    time TEXT,
    camera_id TEXT,
    in_count INTEGER,
    out_count INTEGER,
    posture TEXT,
    alert TEXT
);
"""

# A compatibility view so dashboards can SELECT * FROM TrafficLog
# and get columns: time, "in", "out", camera_id, posture, alert
DDL_VIEW = """
CREATE VIEW IF NOT EXISTS TrafficLog AS
SELECT
    time,
    camera_id,
    in_count AS "in",
    out_count AS "out",
    posture,
    alert
FROM traffic_logs;
"""

def get_db_path(db_path: Optional[str | Path] = None) -> Path:
    """
    Resolve the SQLite file path; ensure the parent directory exists.
    """
    path = Path(db_path) if db_path else DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)  # make sure folder exists
    return path

def init_db(db_path: Optional[str | Path] = None) -> sqlite3.Connection:
    """
    Open the SQLite database, ensure schema exists, and return a connection.
    """
    path = get_db_path(db_path)
    # check_same_thread=False allows reuse across threads (e.g., Streamlit/Flask)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    cur = conn.cursor()
    cur.execute(DDL_TABLE)
    cur.execute(DDL_VIEW)
    conn.commit()
    return conn

def insert_log(conn: sqlite3.Connection, log_entry: Dict[str, Any]) -> None:
    """
    Insert a single log row. Accepts either:
      - in_count/out_count keys, or
      - in/out keys (mapped to in_count/out_count).
    """
    # Map keys for compatibility
    in_count = log_entry.get("in_count", log_entry.get("in", 0))
    out_count = log_entry.get("out_count", log_entry.get("out", 0))

    params = (
        log_entry.get("time"),
        log_entry.get("camera_id"),
        int(in_count) if in_count is not None else 0,
        int(out_count) if out_count is not None else 0,
        log_entry.get("posture"),
        log_entry.get("alert"),
    )
    conn.execute(
        """
        INSERT INTO traffic_logs (time, camera_id, in_count, out_count, posture, alert)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        params,
    )
    conn.commit()
