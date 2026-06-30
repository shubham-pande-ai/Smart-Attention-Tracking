import sqlite3
import secrets
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

DEFAULT_DB = Path("data/attention_tracker.sqlite3")

def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 150_000)
    return salt, digest.hex()

def verify_password(password: str, salt: str, stored_hash: str) -> bool:
    _, candidate_hash = hash_password(password, salt)
    return secrets.compare_digest(candidate_hash, stored_hash)

class Database:
    """
    Abstracted Database layer. 
    Currently uses SQLite, but all methods return standard Python dictionaries.
    To migrate to MongoDB in the future, simply replace the internals of these 
    methods with PyMongo queries; the rest of the application won't notice a difference!
    """
    def __init__(self, path: Path = DEFAULT_DB):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self) -> None:
        with self.get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_salt TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('admin', 'teacher', 'user')),
                    email TEXT,
                    display_name TEXT,
                    verified BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS subjects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    teacher_id INTEGER,
                    FOREIGN KEY(teacher_id) REFERENCES users(id)
                );
                
                CREATE TABLE IF NOT EXISTS enrollments (
                    user_id INTEGER,
                    subject_id INTEGER,
                    FOREIGN KEY(user_id) REFERENCES users(id),
                    FOREIGN KEY(subject_id) REFERENCES subjects(id),
                    PRIMARY KEY(user_id, subject_id)
                );
                
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject_id INTEGER,
                    teacher_id INTEGER,
                    status TEXT NOT NULL CHECK(status IN ('waiting', 'active', 'ended')),
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    FOREIGN KEY(subject_id) REFERENCES subjects(id),
                    FOREIGN KEY(teacher_id) REFERENCES users(id)
                );
                
                CREATE TABLE IF NOT EXISTS telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    user_id INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    state TEXT NOT NULL,
                    boredom INTEGER DEFAULT 0,
                    engagement INTEGER DEFAULT 0,
                    confusion INTEGER DEFAULT 0,
                    frustration INTEGER DEFAULT 0,
                    FOREIGN KEY(session_id) REFERENCES sessions(id),
                    FOREIGN KEY(user_id) REFERENCES users(id)
                );
            """)
            
            # Ensure default accounts exist for testing
            admin = conn.execute("SELECT id FROM users WHERE role = 'admin'").fetchone()
            if not admin:
                salt, pw_hash = hash_password("admin")
                conn.execute(
                    "INSERT INTO users (username, password_salt, password_hash, role, display_name, verified) VALUES (?, ?, ?, ?, ?, ?)",
                    ("admin", salt, pw_hash, "admin", "System Admin", 1)
                )
                
            teacher = conn.execute("SELECT id FROM users WHERE role = 'teacher'").fetchone()
            if not teacher:
                salt, pw_hash = hash_password("teacher")
                conn.execute(
                    "INSERT INTO users (username, password_salt, password_hash, role, display_name, verified) VALUES (?, ?, ?, ?, ?, ?)",
                    ("teacher", salt, pw_hash, "teacher", "Test Teacher", 1)
                )
                
            test_user = conn.execute("SELECT id FROM users WHERE username = 'student1'").fetchone()
            if not test_user:
                salt, pw_hash = hash_password("student")
                conn.execute(
                    "INSERT INTO users (username, password_salt, password_hash, role, display_name, verified) VALUES (?, ?, ?, ?, ?, ?)",
                    ("student1", salt, pw_hash, "user", "Test Student", 1)
                )

    # --- USER MANAGEMENT ---
    
    def create_user(self, username: str, password: str, role: str, email: str = "", display_name: str = "") -> Optional[Dict[str, Any]]:
        salt, pw_hash = hash_password(password)
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    "INSERT INTO users (username, password_salt, password_hash, role, email, display_name) VALUES (?, ?, ?, ?, ?, ?)",
                    (username, salt, pw_hash, role, email, display_name or username)
                )
                user_id = cursor.lastrowid
                return self.get_user_by_id(user_id)
        except sqlite3.IntegrityError:
            return None # Username exists

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            row = conn.execute("SELECT id, username, role, email, display_name, verified FROM users WHERE id = ?", (user_id,)).fetchone()
            return dict(row) if row else None

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            return dict(row) if row else None
            
    def update_password(self, user_id: int, new_password: str) -> bool:
        salt, pw_hash = hash_password(new_password)
        with self.get_connection() as conn:
            conn.execute("UPDATE users SET password_salt = ?, password_hash = ? WHERE id = ?", (salt, pw_hash, user_id))
        return True
        
    def verify_email(self, user_id: int) -> bool:
        with self.get_connection() as conn:
            conn.execute("UPDATE users SET verified = 1 WHERE id = ?", (user_id,))
        return True

    def list_users_by_role(self, role: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            rows = conn.execute("SELECT id, username, email, display_name FROM users WHERE role = ?", (role,)).fetchall()
            return [dict(r) for r in rows]

    # --- SUBJECTS & ENROLLMENT ---
    
    def create_subject(self, name: str, description: str = "", teacher_id: int = None) -> Optional[Dict[str, Any]]:
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    "INSERT INTO subjects (name, description, teacher_id) VALUES (?, ?, ?)",
                    (name, description, teacher_id)
                )
                return {"id": cursor.lastrowid, "name": name, "teacher_id": teacher_id}
        except sqlite3.IntegrityError:
            return None

    def enroll_user(self, user_id: int, subject_id: int) -> bool:
        try:
            with self.get_connection() as conn:
                conn.execute("INSERT INTO enrollments (user_id, subject_id) VALUES (?, ?)", (user_id, subject_id))
            return True
        except sqlite3.IntegrityError:
            return False
            
    # --- SESSIONS ---
    
    def create_session(self, subject_id: int, teacher_id: int) -> int:
        with self.get_connection() as conn:
            cursor = conn.execute(
                "INSERT INTO sessions (subject_id, teacher_id, status) VALUES (?, ?, 'waiting')",
                (subject_id, teacher_id)
            )
            return cursor.lastrowid
            
    def update_session_status(self, session_id: int, status: str) -> None:
        with self.get_connection() as conn:
            conn.execute("UPDATE sessions SET status = ? WHERE id = ?", (status, session_id))
            
    def get_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            return dict(row) if row else None
            
    # --- TELEMETRY ---
    
    def add_telemetry(self, session_id: int, user_id: int, state: str, emotions: dict) -> None:
        boredom = emotions.get("boredom", 0)
        engagement = emotions.get("engagement", 0)
        confusion = emotions.get("confusion", 0)
        frustration = emotions.get("frustration", 0)
        
        with self.get_connection() as conn:
            conn.execute(
                """INSERT INTO telemetry (session_id, user_id, state, boredom, engagement, confusion, frustration) 
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (session_id, user_id, state, boredom, engagement, confusion, frustration)
            )

    def get_historical_reports(self, timeframe: str) -> List[Dict[str, Any]]:
        # timeframe: 'daily', 'weekly', 'monthly'
        if timeframe == 'daily':
            group_by = "date(timestamp)"
        elif timeframe == 'weekly':
            group_by = "strftime('%Y-%W', timestamp)"
        elif timeframe == 'monthly':
            group_by = "strftime('%Y-%m', timestamp)"
        else:
            group_by = "date(timestamp)"
            
        query = f"""
            SELECT 
                {group_by} as period,
                users.display_name as student_name,
                AVG(engagement) as avg_engagement,
                AVG(frustration) as avg_frustration,
                AVG(boredom) as avg_boredom
            FROM telemetry
            JOIN users ON telemetry.user_id = users.id
            GROUP BY period, telemetry.user_id
            ORDER BY period DESC
        """
        
        with self.get_connection() as conn:
            rows = conn.execute(query).fetchall()
            return [dict(r) for r in rows]
