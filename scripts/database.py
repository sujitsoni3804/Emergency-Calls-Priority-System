import sqlite3
from datetime import datetime
from contextlib import contextmanager

DB_PATH = "transcriptions.db"

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transcriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                upload_date TEXT NOT NULL,
                summary TEXT,
                urgency_rating INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending',
                duration INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        try:
            conn.execute("ALTER TABLE transcriptions ADD COLUMN duration INTEGER DEFAULT 0")
        except:
            pass

def add_transcription(job_id, filename):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO transcriptions (job_id, filename, upload_date) VALUES (?, ?, ?)",
            (job_id, filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )

def update_summary(job_id, summary, urgency_rating):
    with get_db() as conn:
        conn.execute(
            "UPDATE transcriptions SET summary = ?, urgency_rating = ? WHERE job_id = ?",
            (summary, urgency_rating, job_id)
        )

def update_duration(job_id, duration):
    with get_db() as conn:
        conn.execute(
            "UPDATE transcriptions SET duration = ? WHERE job_id = ?",
            (duration, job_id)
        )

def update_status(job_id, status):
    with get_db() as conn:
        conn.execute(
            "UPDATE transcriptions SET status = ? WHERE job_id = ?",
            (status, job_id)
        )

def get_all_transcriptions():
    with get_db() as conn:
        cursor = conn.execute("""
            SELECT * FROM transcriptions 
            ORDER BY 
                CASE WHEN status = 'completed' THEN 1 ELSE 0 END,
                urgency_rating DESC,
                upload_date DESC
        """)
        return [dict(row) for row in cursor.fetchall()]

def get_transcription(job_id):
    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM transcriptions WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def delete_transcription(job_id):
    with get_db() as conn:
        conn.execute("DELETE FROM transcriptions WHERE job_id = ?", (job_id,))