# init_trafficlog.py
# Place this file at the project root (same folder as main.py).
# Run once: python init_trafficlog.py

from db import init_db

DDL = """
CREATE TABLE IF NOT EXISTS TrafficLog (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT,
    "in" INTEGER,
    "out" INTEGER,
    camera_id TEXT,
    posture TEXT,
    alert TEXT
);
"""

def main():
    conn = init_db()  # uses the app's configured SQLite file
    cur = conn.cursor()
    cur.execute(DDL)
    conn.commit()
    conn.close()
    print("TrafficLog table is ready.")

if __name__ == "__main__":
    main()
