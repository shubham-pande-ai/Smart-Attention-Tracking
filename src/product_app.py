from __future__ import annotations

import csv
import hashlib
import os
import secrets
import smtplib
import sqlite3
import threading
import time
from dataclasses import dataclass
from dataclasses import dataclass, field
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from textwrap import wrap

# Tkinter imports removed
# Tkinter imports removed
# Tkinter imports removed
# Tkinter imports removed
# Tkinter imports removed
# Tkinter imports removed
# Tkinter imports removed
# Tkinter imports removed
# Tkinter imports removed

IMG_SIZE = (160, 160)
CLASS_NAMES = ["eyes_closed", "eyes_open"]
SCORE_MAP = {
    "eyes_open": 1.0,
    "eyes_closed": 0.0,
    "uncertain": 0.5,
}
ALERT_LABELS = {"eyes_closed"}
DEFAULT_DB = Path("data/attention_tracker.sqlite3")
_cv2 = None
_np = None
_image = None
_image_tk = None
_torch = None


def get_cv2():
    global _cv2
    if _cv2 is None:
        import cv2

        _cv2 = cv2
    return _cv2


def get_np():
    global _np
    if _np is None:
        import numpy as np

        _np = np
    return _np


def get_pil():
    global _image, _image_tk
    if _image is None or _image_tk is None:
        from PIL import Image, ImageTk

        _image = Image
        _image_tk = ImageTk
    return _image, _image_tk


def get_torch():
    global _torch
    if _torch is None:
        import torch

        _torch = torch
    return _torch


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def readable_label(label: str) -> str:
    return label.replace("_", " ").title()


def format_duration(seconds: int) -> str:
    minutes, remaining = divmod(max(seconds, 0), 60)
    if minutes:
        return f"{minutes}m {remaining:02d}s"
    return f"{remaining}s"


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 150_000)
    return salt, digest.hex()


def verify_password(password: str, salt: str, stored_hash: str) -> bool:
    _, candidate_hash = hash_password(password, salt)
    return secrets.compare_digest(candidate_hash, stored_hash)


class TrackerDatabase:
    def __init__(self, path: Path = DEFAULT_DB) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.migrate_roles()
        self.repair_user_foreign_keys()
        self.init_schema()
        self.seed_defaults()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def migrate_roles(self) -> None:
        if not self.path.exists():
            return
        with self.connect() as conn:
            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'users'"
            ).fetchone()
            if not row or "main_admin" in row["sql"]:
                return
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute("ALTER TABLE users RENAME TO users_old")
            conn.execute(
                """
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'admin', 'main_admin')),
                    display_name TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                INSERT INTO users (id, username, password_hash, salt, role, display_name, active, created_at)
                SELECT id, username, password_hash, salt, role, display_name, active, created_at
                FROM users_old
                """
            )
            conn.execute("DROP TABLE users_old")
            conn.execute("PRAGMA foreign_keys = ON")

    def repair_user_foreign_keys(self) -> None:
        if not self.path.exists():
            return
        with self.connect() as conn:
            broken = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE sql LIKE '%users_old%'"
            ).fetchone()[0]
            if not broken:
                return
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute("PRAGMA legacy_alter_table = ON")
            rebuilds = {
                "students": (
                    """
                    CREATE TABLE students (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL UNIQUE,
                        guardian_email TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (user_id) REFERENCES users(id)
                    )
                    """,
                    "id, user_id, guardian_email, created_at",
                ),
                "teachers": (
                    """
                    CREATE TABLE teachers (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL UNIQUE,
                        title TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (user_id) REFERENCES users(id)
                    )
                    """,
                    "id, user_id, title, created_at",
                ),
                "attention_sessions": (
                    """
                    CREATE TABLE attention_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        class_id INTEGER,
                        session_name TEXT NOT NULL,
                        started_at TEXT NOT NULL,
                        ended_at TEXT,
                        average_attention REAL NOT NULL DEFAULT 0,
                        looking_away_seconds INTEGER NOT NULL DEFAULT 0,
                        eyes_closed_seconds INTEGER NOT NULL DEFAULT 0,
                        alert_count INTEGER NOT NULL DEFAULT 0,
                        last_status TEXT NOT NULL DEFAULT 'Idle',
                        active INTEGER NOT NULL DEFAULT 1,
                        FOREIGN KEY (user_id) REFERENCES users(id),
                        FOREIGN KEY (class_id) REFERENCES classes(id)
                    )
                    """,
                    (
                        "id, user_id, class_id, session_name, started_at, ended_at, average_attention, "
                        "looking_away_seconds, eyes_closed_seconds, alert_count, last_status, active"
                    ),
                ),
                "alerts": (
                    """
                    CREATE TABLE alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id INTEGER,
                        user_id INTEGER NOT NULL,
                        class_id INTEGER,
                        alert_time TEXT NOT NULL,
                        label TEXT NOT NULL,
                        duration_seconds INTEGER NOT NULL,
                        attention_score REAL NOT NULL,
                        delivery_mode TEXT NOT NULL,
                        message TEXT NOT NULL,
                        acknowledged INTEGER NOT NULL DEFAULT 0,
                        FOREIGN KEY (session_id) REFERENCES attention_sessions(id),
                        FOREIGN KEY (user_id) REFERENCES users(id),
                        FOREIGN KEY (class_id) REFERENCES classes(id)
                    )
                    """,
                    (
                        "id, session_id, user_id, class_id, alert_time, label, duration_seconds, "
                        "attention_score, delivery_mode, message, acknowledged"
                    ),
                ),
                "alert_settings": (
                    """
                    CREATE TABLE alert_settings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        class_id INTEGER,
                        note_taking_grace INTEGER NOT NULL DEFAULT 180,
                        eyes_closed_after INTEGER NOT NULL DEFAULT 20,
                        alert_cooldown INTEGER NOT NULL DEFAULT 300,
                        confidence_threshold REAL NOT NULL DEFAULT 0.55,
                        teacher_email TEXT,
                        UNIQUE (user_id, class_id),
                        FOREIGN KEY (user_id) REFERENCES users(id),
                        FOREIGN KEY (class_id) REFERENCES classes(id)
                    )
                    """,
                    (
                        "id, user_id, class_id, note_taking_grace, eyes_closed_after, alert_cooldown, "
                        "confidence_threshold, teacher_email"
                    ),
                ),
            }
            for table, (create_sql, columns) in rebuilds.items():
                exists = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                    (table,),
                ).fetchone()
                if not exists:
                    continue
                old_table = f"{table}_fk_old"
                conn.execute(f"DROP TABLE IF EXISTS {old_table}")
                conn.execute(f"ALTER TABLE {table} RENAME TO {old_table}")
                conn.execute(create_sql)
                conn.execute(f"INSERT INTO {table} ({columns}) SELECT {columns} FROM {old_table}")
                conn.execute(f"DROP TABLE {old_table}")
            conn.execute("PRAGMA legacy_alter_table = OFF")
            conn.execute("PRAGMA foreign_keys = ON")

    def init_schema(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('user', 'admin', 'main_admin')),
            display_name TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL UNIQUE,
            guardian_email TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS teachers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL UNIQUE,
            title TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS classes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            description TEXT,
            active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS enrollments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_id INTEGER NOT NULL,
            student_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE (class_id, student_id),
            FOREIGN KEY (class_id) REFERENCES classes(id),
            FOREIGN KEY (student_id) REFERENCES students(id)
        );

        CREATE TABLE IF NOT EXISTS attention_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            class_id INTEGER,
            session_name TEXT NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            average_attention REAL NOT NULL DEFAULT 0,
            looking_away_seconds INTEGER NOT NULL DEFAULT 0,
            eyes_closed_seconds INTEGER NOT NULL DEFAULT 0,
            alert_count INTEGER NOT NULL DEFAULT 0,
            last_status TEXT NOT NULL DEFAULT 'Idle',
            active INTEGER NOT NULL DEFAULT 1,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (class_id) REFERENCES classes(id)
        );

        CREATE TABLE IF NOT EXISTS attention_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            event_time TEXT NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            attention_score REAL NOT NULL,
            face_detected INTEGER NOT NULL,
            FOREIGN KEY (session_id) REFERENCES attention_sessions(id)
        );

        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            user_id INTEGER NOT NULL,
            class_id INTEGER,
            alert_time TEXT NOT NULL,
            label TEXT NOT NULL,
            duration_seconds INTEGER NOT NULL,
            attention_score REAL NOT NULL,
            delivery_mode TEXT NOT NULL,
            message TEXT NOT NULL,
            acknowledged INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (session_id) REFERENCES attention_sessions(id),
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (class_id) REFERENCES classes(id)
        );

        CREATE TABLE IF NOT EXISTS alert_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            class_id INTEGER,
            note_taking_grace INTEGER NOT NULL DEFAULT 180,
            eyes_closed_after INTEGER NOT NULL DEFAULT 20,
            alert_cooldown INTEGER NOT NULL DEFAULT 300,
            confidence_threshold REAL NOT NULL DEFAULT 0.55,
            teacher_email TEXT,
            UNIQUE (user_id, class_id),
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (class_id) REFERENCES classes(id)
        );

        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
        with self.connect() as conn:
            conn.executescript(schema)

    def seed_defaults(self) -> None:
        defaults = {
            "model_path": "models/attention_cnn.pt",
            "note_taking_grace": "180",
            "eyes_closed_after": "20",
            "alert_cooldown": "300",
            "confidence_threshold": "0.55",
            "no_face_after": "60",
            "teacher_email": os.getenv("TEACHER_EMAIL", ""),
            "smtp_host": os.getenv("SMTP_HOST", ""),
            "smtp_port": os.getenv("SMTP_PORT", "587"),
            "smtp_user": os.getenv("SMTP_USER", ""),
            "smtp_password": os.getenv("SMTP_PASSWORD", ""),
            "alert_from": os.getenv("ALERT_FROM", ""),
            "email_alerts": "0",
        }
        with self.connect() as conn:
            for key, value in defaults.items():
                conn.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (key, value))
            conn.execute("UPDATE settings SET value = 'models/attention_cnn.pt' WHERE key = 'model_path' AND value LIKE '%.keras'")
            user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            if user_count == 0:
                main_admin_id = self.create_account("mainadmin", "mainadmin123", "main_admin", "Main Admin", conn)
                admin_id = self.create_account("admin", "admin123", "admin", "Admin", conn)
                user_id = self.create_account("user", "user123", "user", "Demo User", conn)
                class_id = self.create_class("Demo Class", "Default class for local testing.", conn)
                self.enroll_user(class_id, user_id, conn)
                conn.execute(
                    "INSERT OR IGNORE INTO alert_settings (user_id, class_id) VALUES (?, ?)",
                    (user_id, class_id),
                )
                conn.execute("UPDATE teachers SET title = ? WHERE user_id = ?", ("Main Admin", main_admin_id))
                conn.execute("UPDATE teachers SET title = ? WHERE user_id = ?", ("Admin", admin_id))
            else:
                main_admin = conn.execute("SELECT id FROM users WHERE role = 'main_admin' LIMIT 1").fetchone()
                if main_admin is None:
                    main_admin_id = self.create_account("mainadmin", "mainadmin123", "main_admin", "Main Admin", conn)
                    conn.execute("UPDATE teachers SET title = ? WHERE user_id = ?", ("Main Admin", main_admin_id))

    def create_account(
        self,
        username: str,
        password: str,
        role: str,
        display_name: str,
        conn: sqlite3.Connection | None = None,
    ) -> int:
        owns_connection = conn is None
        conn = conn or self.connect()
        try:
            salt, password_hash = hash_password(password)
            cursor = conn.execute(
                """
                INSERT INTO users (username, password_hash, salt, role, display_name, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (username.strip(), password_hash, salt, role, display_name.strip(), now_text()),
            )
            user_id = int(cursor.lastrowid)
            if role == "user":
                conn.execute("INSERT INTO students (user_id, created_at) VALUES (?, ?)", (user_id, now_text()))
            else:
                conn.execute("INSERT INTO teachers (user_id, title, created_at) VALUES (?, ?, ?)", (user_id, "Admin", now_text()))
            if owns_connection:
                conn.commit()
            return user_id
        finally:
            if owns_connection:
                conn.close()

    def authenticate(self, username: str, password: str, role: str) -> sqlite3.Row | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ? AND role = ? AND active = 1",
                (username.strip(), role),
            ).fetchone()
        if row and verify_password(password, row["salt"], row["password_hash"]):
            return row
        return None

    def get_settings(self) -> dict[str, str]:
        with self.connect() as conn:
            rows = conn.execute("SELECT key, value FROM settings").fetchall()
        return {row["key"]: row["value"] for row in rows}

    def save_settings(self, values: dict[str, str]) -> None:
        with self.connect() as conn:
            for key, value in values.items():
                conn.execute(
                    "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (key, value),
                )

    def create_class(self, name: str, description: str = "", conn: sqlite3.Connection | None = None) -> int:
        owns_connection = conn is None
        conn = conn or self.connect()
        try:
            cursor = conn.execute(
                "INSERT INTO classes (name, description, created_at) VALUES (?, ?, ?)",
                (name.strip(), description.strip(), now_text()),
            )
            if owns_connection:
                conn.commit()
            return int(cursor.lastrowid)
        finally:
            if owns_connection:
                conn.close()

    def enroll_user(self, class_id: int, user_id: int, conn: sqlite3.Connection | None = None) -> None:
        owns_connection = conn is None
        conn = conn or self.connect()
        try:
            student = conn.execute("SELECT id FROM students WHERE user_id = ?", (user_id,)).fetchone()
            if not student:
                raise ValueError("Only user accounts can be enrolled in classes.")
            conn.execute(
                "INSERT OR IGNORE INTO enrollments (class_id, student_id, created_at) VALUES (?, ?, ?)",
                (class_id, student["id"], now_text()),
            )
            if owns_connection:
                conn.commit()
        finally:
            if owns_connection:
                conn.close()

    def list_accounts(self) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT id, username, display_name, role, active, created_at FROM users ORDER BY role, display_name"
            ).fetchall()

    def list_classes(self) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM classes WHERE active = 1 ORDER BY name").fetchall()

    def list_user_classes(self, user_id: int) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT c.id, c.name
                FROM classes c
                JOIN enrollments e ON e.class_id = c.id
                JOIN students s ON s.id = e.student_id
                WHERE s.user_id = ? AND c.active = 1
                ORDER BY c.name
                """,
                (user_id,),
            ).fetchall()

    def list_roster(self) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT c.name AS class_name, u.display_name, u.username,
                       COALESCE(s.last_status, 'No session') AS last_status,
                       COALESCE(s.average_attention, 0) AS average_attention,
                       s.started_at,
                       s.ended_at
                FROM classes c
                LEFT JOIN enrollments e ON e.class_id = c.id
                LEFT JOIN students st ON st.id = e.student_id
                LEFT JOIN users u ON u.id = st.user_id
                LEFT JOIN attention_sessions s ON s.id = (
                    SELECT id FROM attention_sessions
                    WHERE user_id = u.id
                    ORDER BY started_at DESC
                    LIMIT 1
                )
                WHERE c.active = 1
                ORDER BY c.name, u.display_name
                """
            ).fetchall()

    def start_session(self, user_id: int, class_id: int | None, session_name: str) -> int:
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO attention_sessions (user_id, class_id, session_name, started_at)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, class_id, session_name.strip() or "Attention Session", now_text()),
            )
            return int(cursor.lastrowid)

    def add_event(
        self,
        session_id: int,
        label: str,
        confidence: float,
        attention_score: float,
        face_detected: bool,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO attention_events
                (session_id, event_time, label, confidence, attention_score, face_detected)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, now_text(), label, confidence, attention_score, int(face_detected)),
            )
            conn.execute(
                "UPDATE attention_sessions SET last_status = ?, average_attention = ? WHERE id = ?",
                (label, attention_score, session_id),
            )

    def add_alert(
        self,
        session_id: int,
        user_id: int,
        class_id: int | None,
        label: str,
        duration_seconds: int,
        attention_score: float,
        delivery_mode: str,
        message: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO alerts
                (session_id, user_id, class_id, alert_time, label, duration_seconds, attention_score, delivery_mode, message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, user_id, class_id, now_text(), label, duration_seconds, attention_score, delivery_mode, message),
            )
            conn.execute(
                "UPDATE attention_sessions SET alert_count = alert_count + 1 WHERE id = ?",
                (session_id,),
            )

    def end_session(
        self,
        session_id: int,
        average_attention: float,
        looking_away_seconds: int,
        eyes_closed_seconds: int,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE attention_sessions
                SET ended_at = ?, active = 0, average_attention = ?,
                    looking_away_seconds = ?, eyes_closed_seconds = ?
                WHERE id = ?
                """,
                (now_text(), average_attention, looking_away_seconds, eyes_closed_seconds, session_id),
            )

    def list_sessions(self, user_id: int | None = None) -> list[sqlite3.Row]:
        query = """
            SELECT s.id, u.display_name, c.name AS class_name, s.session_name,
                   s.started_at, s.ended_at, s.average_attention,
                   s.looking_away_seconds, s.eyes_closed_seconds, s.alert_count, s.last_status
            FROM attention_sessions s
            JOIN users u ON u.id = s.user_id
            LEFT JOIN classes c ON c.id = s.class_id
        """
        params: tuple[int, ...] = ()
        if user_id is not None:
            query += " WHERE s.user_id = ?"
            params = (user_id,)
        query += " ORDER BY s.started_at DESC LIMIT 250"
        with self.connect() as conn:
            return conn.execute(query, params).fetchall()

    def list_alerts(self) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT a.id, a.alert_time, u.display_name, c.name AS class_name, a.label,
                       a.duration_seconds, a.attention_score, a.delivery_mode, a.acknowledged, a.message
                FROM alerts a
                JOIN users u ON u.id = a.user_id
                LEFT JOIN classes c ON c.id = a.class_id
                ORDER BY a.alert_time DESC
                LIMIT 250
                """
            ).fetchall()

    def set_account_active(self, user_id: int, active: bool) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE users SET active = ? WHERE id = ?", (int(active), user_id))

    def acknowledge_alert(self, alert_id: int) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE alerts SET acknowledged = 1 WHERE id = ?", (alert_id,))

    def delete_history(self, user_id: int | None = None) -> None:
        with self.connect() as conn:
            if user_id is None:
                session_ids = [row["id"] for row in conn.execute("SELECT id FROM attention_sessions").fetchall()]
            else:
                session_ids = [
                    row["id"] for row in conn.execute("SELECT id FROM attention_sessions WHERE user_id = ?", (user_id,)).fetchall()
                ]
            for session_id in session_ids:
                conn.execute("DELETE FROM attention_events WHERE session_id = ?", (session_id,))
                conn.execute("DELETE FROM alerts WHERE session_id = ?", (session_id,))
            if user_id is None:
                conn.execute("DELETE FROM attention_sessions")
            else:
                conn.execute("DELETE FROM attention_sessions WHERE user_id = ?", (user_id,))


@dataclass
class Prediction:
    label: str
    confidence: float
    face_detected: bool
    box: tuple[int, int, int, int] | None = None
    emotions: dict[str, float] = field(default_factory=dict)


class AttentionEngine:
    def __init__(self, model_path: Path, confidence_threshold: float) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.model_error = ""
        cv2 = get_cv2()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        self.load_model()

    def load_model(self) -> None:
        if not self.model_path.exists():
            self.model_error = f"Model not found: {self.model_path}"
            return
        try:
            torch = get_torch()
            self.model = torch.jit.load(str(self.model_path))
            self.model.eval()
            self.model_error = ""
        except Exception as exc:
            self.model_error = str(exc)

    def crop_eye_region(self, frame: np.ndarray) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
        cv2 = get_cv2()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            return None, None

        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        upper_face_gray = gray[y : y + h // 2, x : x + w]
        eyes = self.eye_cascade.detectMultiScale(upper_face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
        if len(eyes) > 0:
            ex, ey, ew, eh = max(eyes, key=lambda box: box[2] * box[3])
            pad = 12
            x1 = max(x + ex - pad, 0)
            y1 = max(y + ey - pad, 0)
            x2 = min(x + ex + ew + pad, frame.shape[1])
            y2 = min(y + ey + eh + pad, frame.shape[0])
        else:
            # When eyes are closed, Haar cascade fails. We must fallback to a single eye 
            # region so it matches the tightly-cropped training data.
            ew = int(w * 0.25)
            eh = int(h * 0.20)
            ex = int(w * 0.25) # Approximate right eye offset
            ey = int(h * 0.25)
            
            pad = 12
            x1 = max(x + ex - pad, 0)
            y1 = max(y + ey - pad, 0)
            x2 = min(x + ex + ew + pad, frame.shape[1])
            y2 = min(y + ey + eh + pad, frame.shape[0])
            
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

    def predict(self, frame: np.ndarray) -> Prediction:
        cv2 = get_cv2()
        np = get_np()
        torch = get_torch()
        if self.model is None:
            return Prediction("model_unavailable", 0.0, False, None)
        eye_crop, box = self.crop_eye_region(frame)
        if eye_crop is None or eye_crop.size == 0:
            return Prediction("no_face", 0.0, False, None)

        rgb = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, IMG_SIZE)
        
        # PyTorch preprocess
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        batch = tensor.unsqueeze(0)
        
        device = next(self.model.parameters()).device if hasattr(self.model, "parameters") else torch.device("cpu")
        batch = batch.to(device)
        
        with torch.no_grad():
            outputs = self.model(batch)
            probs_eye = torch.nn.functional.softmax(outputs["eye_state"][0], dim=0).cpu().numpy()
            
        index = int(np.argmax(probs_eye))
        confidence = float(probs_eye[index])
        print(f"Debug Pred: {CLASS_NAMES[index]} | Probs: {probs_eye} | Conf: {confidence}")
        
        # We drop the confidence_threshold for binary eye_state because 
        # argmax is mathematically always >= 0.5. Forcing a higher threshold 
        # just causes it to incorrectly report 'uncertain'.
        label = CLASS_NAMES[index]
        
        emotions = {}
        for em in ["boredom", "engagement", "confusion", "frustration"]:
            probs_em = torch.nn.functional.softmax(outputs[em][0], dim=0).cpu().numpy()
            emotions[em] = float(np.argmax(probs_em))
            
        # Hard override: If eyes are closed, the person cannot be engaged or happy.
        if label == "eyes_closed":
            emotions["boredom"] = 3.0
            emotions["engagement"] = 0.0
            emotions["confusion"] = 0.0
            emotions["frustration"] = 0.0
            
        return Prediction(label, confidence, True, box, emotions=emotions)


# ProductApp Tkinter GUI class removed











































































































































































































































































































































































































































































































































































def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_simple_pdf(path: Path, lines: list[str]) -> None:
    content_lines = ["BT", "/F1 10 Tf", "50 780 Td", "14 TL"]
    for line in lines:
        for wrapped in wrap(line, width=95) or [""]:
            content_lines.append(f"({pdf_escape(wrapped)}) Tj")
            content_lines.append("T*")
    content_lines.append("ET")
    content = "\n".join(content_lines).encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"\nendstream",
    ]
    output = [b"%PDF-1.4\n"]
    offsets = []
    for index, obj in enumerate(objects, start=1):
        offsets.append(sum(len(part) for part in output))
        output.append(f"{index} 0 obj\n".encode("ascii") + obj + b"\nendobj\n")
    xref_offset = sum(len(part) for part in output)
    output.append(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets:
        output.append(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.append(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    path.write_bytes(b"".join(output))


def main() -> None:
    print("The desktop Tkinter application has been removed.")
    print("Starting the web portals instead...")
    import web_app
    web_app.main()


if __name__ == "__main__":
    main()
