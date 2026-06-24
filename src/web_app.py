from __future__ import annotations

import argparse
import base64
import csv
import json
import threading
import time
from dataclasses import dataclass, field
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from product_app import (
    AttentionEngine,
    Prediction,
    SCORE_MAP,
    TrackerDatabase,
    format_duration,
    get_cv2,
    get_np,
    now_text,
    readable_label,
    write_simple_pdf,
)


PORTALS = {
    "user": {"port": 8010, "label": "User Portal", "path": "/user"},
    "admin": {"port": 8020, "label": "Admin Portal", "path": "/admin"},
}
DB = TrackerDatabase()
TOKENS: dict[str, dict] = {}
LIVE_SESSIONS: dict[int, "LiveSession"] = {}
STATE_LOCK = threading.Lock()
ENGINE: AttentionEngine | None = None
ENGINE_KEY: tuple[str, float] | None = None


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


@dataclass
class LiveSession:
    session_id: int
    user_id: int
    display_name: str
    class_id: int | None
    class_name: str
    session_name: str
    started_at: float
    scores: list[float] = field(default_factory=list)
    looking_away_seconds: int = 0
    eyes_closed_seconds: int = 0
    last_event_at: float = field(default_factory=time.time)
    current_alert_label: str | None = None
    alert_started_at: float = 0.0
    last_alert_at: float = 0.0
    label: str = "Idle"
    confidence: float = 0.0
    face_detected: bool = True
    average_attention: int = 0
    emotions: dict[str, float] = field(default_factory=dict)


def html_escape(value: object) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def make_token() -> str:
    raw = f"{time.time()}-{threading.get_ident()}-{len(TOKENS)}".encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def cookie_name(portal: str) -> str:
    return f"attention_session_{portal}"


def get_engine() -> AttentionEngine:
    global ENGINE, ENGINE_KEY
    settings = DB.get_settings()
    model_path = settings.get("model_path", "models/attention_cnn.pt")
    confidence = float(settings.get("confidence_threshold", "0.55"))
    key = (model_path, confidence)
    if ENGINE is None or ENGINE_KEY != key:
        ENGINE = AttentionEngine(Path(model_path), confidence)
        ENGINE_KEY = key
    return ENGINE


def class_name_for(class_id: int | None) -> str:
    if class_id is None:
        return "No Class"
    for row in DB.list_classes():
        if int(row["id"]) == int(class_id):
            return row["name"]
    return "No Class"


SESSION_TIMEOUT_SECONDS = 300


def cleanup_expired_sessions() -> None:
    now = time.time()
    expired_sessions = []
    with STATE_LOCK:
        for sid, s in list(LIVE_SESSIONS.items()):
            last_active = s.last_event_at if s.last_event_at > 0 else s.started_at
            if now - last_active > SESSION_TIMEOUT_SECONDS:
                expired_sessions.append(s)
                LIVE_SESSIONS.pop(sid, None)
    for s in expired_sessions:
        DB.end_session(s.session_id, s.average_attention, s.looking_away_seconds, s.eyes_closed_seconds)


def session_to_dict(session: LiveSession) -> dict:
    elapsed = int(time.time() - session.started_at)
    return {
        "session_id": session.session_id,
        "display_name": session.display_name,
        "class_name": session.class_name,
        "session_name": session.session_name,
        "elapsed": format_duration(elapsed),
        "label": readable_label(session.label),
        "confidence": round(session.confidence * 100),
        "face_detected": session.face_detected,
        "average_attention": session.average_attention,
        "looking_away_seconds": session.looking_away_seconds,
        "eyes_closed_seconds": session.eyes_closed_seconds,
        "emotions": session.emotions,
    }


def decode_frame(data_url: str):
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    cv2 = get_cv2()
    np = get_np()
    encoded = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode frame.")
    return frame


def maybe_send_email(message: str) -> None:
    settings = DB.get_settings()
    if settings.get("email_alerts", "0") != "1":
        return
    if not settings.get("teacher_email") or not settings.get("smtp_host") or not settings.get("smtp_user"):
        return

    def worker() -> None:
        from email.message import EmailMessage
        import smtplib

        email = EmailMessage()
        email["Subject"] = "Soft attention alert"
        email["From"] = settings.get("alert_from") or settings.get("smtp_user")
        email["To"] = settings["teacher_email"]
        email.set_content(message)
        with smtplib.SMTP(settings["smtp_host"], int(settings.get("smtp_port", "587"))) as server:
            server.starttls()
            server.login(settings["smtp_user"], settings.get("smtp_password", ""))
            server.send_message(email)

    threading.Thread(target=worker, daemon=True).start()


def apply_prediction(session: LiveSession, prediction: Prediction) -> dict:
    now = time.time()
    delta = 1 if session.last_event_at == 0 else max(1, min(5, int(round(now - session.last_event_at))))
    session.last_event_at = now
    label = prediction.label
    score_label = label if label in SCORE_MAP else "uncertain"
    score = SCORE_MAP.get(score_label, 0.5)
    if label in SCORE_MAP:
        session.scores.append(score)

    if label == "eyes_closed":
        session.eyes_closed_seconds += delta

    session.label = label
    session.confidence = prediction.confidence
    session.face_detected = prediction.face_detected
    session.emotions = prediction.emotions
    session.average_attention = int((sum(session.scores) / len(session.scores)) * 100) if session.scores else 0
    DB.add_event(session.session_id, label, prediction.confidence, session.average_attention, prediction.face_detected)

    alert_status = update_alert_state(session)
    return {
        "ok": True,
        "state": readable_label(label),
        "confidence": round(prediction.confidence * 100),
        "attention": session.average_attention,
        "face_detected": prediction.face_detected,
        "alert": alert_status,
        "emotions": session.emotions,
    }


def update_alert_state(session: LiveSession) -> str:
    settings = DB.get_settings()
    thresholds = {
        "eyes_closed": int(settings.get("eyes_closed_after", "20")),
    }
    no_face_threshold = int(settings.get("no_face_after", "60"))
    cooldown = int(settings.get("alert_cooldown", "300"))
    label = session.label
    now = time.time()

    if label == "eyes_open":
        session.current_alert_label = None
        session.alert_started_at = 0.0
        return "Attentive"

    if label == "no_face":
        if session.current_alert_label != "no_face":
            session.current_alert_label = "no_face"
            session.alert_started_at = now
        duration = int(now - session.alert_started_at)
        if duration < no_face_threshold:
            return f"No face detected {format_duration(duration)}/{format_duration(no_face_threshold)}"
        if now - session.last_alert_at >= cooldown:
            session.last_alert_at = now
            message = (
                f"{session.display_name} has no visible face for {format_duration(duration)} "
                f"during {session.session_name}. Attention score: {session.average_attention}%."
            )
            DB.add_alert(
                session.session_id,
                session.user_id,
                session.class_id,
                label,
                duration,
                session.average_attention,
                "in_app",
                message,
            )
            maybe_send_email(message)
        return f"Soft alert: No face for {format_duration(duration)}"

    if label == "uncertain":
        if session.current_alert_label != "uncertain":
            session.current_alert_label = "uncertain"
            session.alert_started_at = now
        duration = int(now - session.alert_started_at)
        if duration < 2:
            return "Checking gaze..."
        return f"Uncertain gaze for {format_duration(duration)}"

    if label in thresholds:
        if session.current_alert_label != label:
            session.current_alert_label = label
            session.alert_started_at = now
        duration = int(now - session.alert_started_at)
        threshold = thresholds[label]
        if duration < 2:
            return f"Brief {readable_label(label).lower()}"
        if duration < threshold:
            return f"Possible note-taking {format_duration(duration)}/{format_duration(threshold)}"
        if now - session.last_alert_at >= cooldown:
            session.last_alert_at = now
            message = (
                f"{session.display_name} is {readable_label(label)} for {format_duration(duration)} "
                f"during {session.session_name}. Attention score: {session.average_attention}%."
            )
            DB.add_alert(
                session.session_id,
                session.user_id,
                session.class_id,
                label,
                duration,
                session.average_attention,
                "in_app",
                message,
            )
            maybe_send_email(message)
        return f"Soft alert: {readable_label(label)} for {format_duration(duration)}"

    if session.current_alert_label:
        duration = int(now - session.alert_started_at)
        return f"Soft alert: {readable_label(session.current_alert_label)} for {format_duration(duration)}"

    return "Attentive"


class WebHandler(BaseHTTPRequestHandler):
    server_version = "AttentionTrackerWeb/1.0"

    def log_message(self, format: str, *args) -> None:
        return

    def current_user(self) -> dict | None:
        header = self.headers.get("Cookie", "")
        jar = cookies.SimpleCookie(header)
        for portal in PORTALS:
            name = cookie_name(portal)
            if name in jar:
                return TOKENS.get(jar[name].value)
        return None

    def require_user(self, roles: tuple[str, ...]) -> dict | None:
        user = self.current_user()
        if not user or user["role"] not in roles:
            self.redirect(f"/login?role={roles[0]}")
            return None
        return user

    def read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(length) if length else b""

    def form(self) -> dict[str, str]:
        body = self.read_body().decode("utf-8")
        return {key: values[0] for key, values in parse_qs(body).items()}

    def json_body(self) -> dict:
        body = self.read_body().decode("utf-8")
        return json.loads(body or "{}")

    def send_html(self, content: str, status: int = 200) -> None:
        data = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            return

    def send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            return

    def redirect(self, location: str) -> None:
        self.send_response(303)
        self.send_header("Location", location)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            portal = getattr(self.server, "portal", "user")
            self.redirect(PORTALS[portal]["path"])
        elif path == "/login":
            role = parse_qs(parsed.query).get("role", [getattr(self.server, "portal", "user")])[0]
            self.login_page(role)
        elif path == "/logout":
            self.logout()
        elif path == "/user":
            self.user_page()
        elif path == "/admin":
            self.admin_page()
        elif path == "/api/live":
            self.api_live()
        elif path == "/api/sessions":
            self.api_sessions()
        elif path == "/export/sessions.csv":
            self.export_csv()
        elif path == "/export/sessions.pdf":
            self.export_pdf()
        else:
            self.send_html(layout("Not Found", "<h1>Not Found</h1>"), 404)

    def do_HEAD(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/login":
            self.login()
        elif path == "/api/session/start":
            self.api_session_start()
        elif path == "/api/session/end":
            self.api_session_end()
        elif path == "/api/frame":
            self.api_frame()
        elif path == "/api/demo-event":
            self.api_demo_event()
        elif path == "/api/create-account":
            self.create_account()
        elif path == "/api/create-class":
            self.create_class()
        elif path == "/api/assign-class":
            self.assign_class()
        elif path == "/api/save-settings":
            self.save_settings()
        elif path == "/api/ack-alert":
            self.ack_alert()
        elif path == "/api/delete-history":
            self.delete_history()
        else:
            self.send_json({"ok": False, "error": "Unknown endpoint"}, 404)

    def login_page(self, role: str) -> None:
        role = role if role in ("user", "admin") else "user"
        labels = {
            "user": "User Login",
            "admin": "Admin Login",
        }
        body = f"""
        <section class="login">
          <div>
            <p class="eyebrow">Attention Tracker</p>
            <h1>{html_escape(labels.get(role, "Login"))}</h1>
            <p class="muted">Local classroom attention monitoring with soft alerts and privacy-first session storage.</p>
          </div>
          <form class="panel login-panel" method="post" action="/login">
            <input type="hidden" name="role" value="{html_escape(role)}">
            <label>Username<input name="username" autocomplete="username" required></label>
            <label>Password<input name="password" type="password" autocomplete="current-password" required></label>
            <button class="primary" type="submit">Sign In</button>
            <p class="tiny">Demo: user/user123, admin/admin123</p>
          </form>
        </section>
        """
        self.send_html(layout(labels.get(role, "Login"), body))

    def login(self) -> None:
        data = self.form()
        role = data.get("role", "user")
        row = DB.authenticate(data.get("username", ""), data.get("password", ""), role)
        if not row:
            self.send_html(layout("Login Failed", "<main class='center'><h1>Login failed</h1><a href='/'>Try again</a></main>"), 401)
            return
        token = make_token()
        TOKENS[token] = {"id": row["id"], "username": row["username"], "display_name": row["display_name"], "role": row["role"]}
        target = "/admin" if row["role"] in ("admin", "main_admin") else "/user"
        self.send_response(303)
        self.send_header("Location", target)
        name = cookie_name("admin") if row["role"] in ("admin", "main_admin") else cookie_name("user")
        self.send_header("Set-Cookie", f"{name}={token}; Path=/; HttpOnly; SameSite=Lax")
        self.end_headers()

    def logout(self) -> None:
        user = self.current_user()
        if user:
            for token, value in list(TOKENS.items()):
                if value == user:
                    TOKENS.pop(token, None)
        self.send_response(303)
        self.send_header("Location", "/")
        for portal in PORTALS:
            name = cookie_name(portal)
            self.send_header("Set-Cookie", f"{name}=; Path=/; Max-Age=0")
        self.end_headers()

    def user_page(self) -> None:
        user = self.require_user(("user",))
        if not user:
            return
        classes = DB.list_user_classes(int(user["id"]))
        options = "".join(f"<option value='{row['id']}'>{html_escape(row['name'])}</option>" for row in classes)
        body = f"""
        <header class="topbar">
          <div><p class="eyebrow">User Portal</p><h1>{html_escape(user['display_name'])}</h1></div>
          <a class="ghost" href="/logout">Sign Out</a>
        </header>
        <main class="grid two">
          <section class="panel">
            <video id="video" autoplay muted playsinline></video>
            <canvas id="canvas" width="384" height="288"></canvas>
          </section>
          <section class="panel stack">
            <h2>Session</h2>
            <label>Class<select id="classId">{options}</select></label>
            <label>Session name<input id="sessionName" value="Lecture Session"></label>
            <label class="check"><input id="consent" type="checkbox"> I consent to local webcam processing. Raw video is not saved.</label>
            <div class="buttons">
              <button class="primary" id="startBtn">Start Tracking</button>
              <button id="endBtn">End Session</button>
            </div>
            <div class="metrics">
              <div><span id="state">Idle</span><small>State</small></div>
              <div><span id="attention">0%</span><small>Attention</small></div>
              <div><span id="confidence">0%</span><small>Confidence</small></div>
            </div>
            
            <h3 style="margin-top:20px;margin-bottom:8px;font-size:14px;color:var(--muted)">High-Level Analysis</h3>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px;">
              <div class="premium-card"><span id="happy" class="big-value">-</span><small>Is Happy?</small></div>
              <div class="premium-card"><span id="understanding" class="big-value">-</span><small>Is Understanding?</small></div>
            </div>

            <h3 style="margin-bottom:8px;font-size:14px;color:var(--muted)">Raw DAiSEE Emotions</h3>
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;">
              <div class="premium-card"><span id="boredom" class="med-value">-</span><small>Boredom</small></div>
              <div class="premium-card"><span id="engagement" class="med-value">-</span><small>Engaged</small></div>
              <div class="premium-card"><span id="confusion" class="med-value">-</span><small>Confused</small></div>
              <div class="premium-card"><span id="frustration" class="med-value">-</span><small>Frustrated</small></div>
            </div>
            <p id="alert" class="notice">Soft alert: waiting</p>
            <div class="demo">
              <select id="demoLabel">
                <option value="looking_screen">Looking Screen</option>
                <option value="looking_away">Looking Away</option>
                <option value="eyes_closed">Eyes Closed</option>
                <option value="no_face">No Face</option>
              </select>
              <button id="demoBtn">Send Demo Signal</button>
            </div>
          </section>
        </main>
        <script>{USER_JS}</script>
        """
        self.send_html(layout("User Portal", body))

    def admin_page(self) -> None:
        user = self.require_user(("admin", "main_admin"))
        if not user:
            return
        settings = DB.get_settings()
        accounts = DB.list_accounts()
        classes = DB.list_classes()
        account_options = "".join(
            f"<option value='{row['id']}'>{html_escape(row['display_name'])} ({html_escape(row['role'])})</option>"
            for row in accounts
            if row["role"] == "user"
        )
        class_options = "".join(f"<option value='{row['id']}'>{html_escape(row['name'])}</option>" for row in classes)
        rows = "".join(
            f"<tr><td>{row['id']}</td><td>{html_escape(row['username'])}</td><td>{html_escape(row['display_name'])}</td>"
            f"<td>{html_escape(row['role'])}</td><td>{row['active']}</td></tr>"
            for row in accounts
        )
        class_rows = "".join(
            f"<tr><td>{row['id']}</td><td>{html_escape(row['name'])}</td><td>{html_escape(row['description'])}</td></tr>"
            for row in classes
        )
        body = f"""
        <header class="topbar">
          <div><p class="eyebrow">Admin Portal</p><h1>Lecture Monitor</h1></div>
          <nav><a class="ghost" href="/logout">Sign Out</a></nav>
        </header>
        <main class="grid two">
          <section class="panel">
            <h2>Live Students</h2>
            <table id="liveTable"><thead><tr><th>User</th><th>Class</th><th>State</th><th>Attention</th><th>Time</th></tr></thead><tbody></tbody></table>
          </section>
          <section class="panel">
            <h2>Teacher Alerts</h2>
            <table id="alertTable"><thead><tr><th>Time</th><th>User</th><th>Status</th><th>Message</th><th></th></tr></thead><tbody></tbody></table>
          </section>
        </main>
        <section class="panel wide">
          <h2>Recent Sessions</h2>
          <table id="sessionTable"><thead><tr><th>User</th><th>Class</th><th>Session</th><th>Average</th><th>Away</th><th>Closed</th><th>Alerts</th></tr></thead><tbody></tbody></table>
        </section>
        <main class="grid three">
          <section class="panel stack">
            <h2>Add Account</h2>
            <form method="post" action="/api/create-account">
              <input name="username" placeholder="username" required>
              <input name="display_name" placeholder="display name" required>
              <input name="password" placeholder="password" type="password" required>
              <select name="role"><option value="user">User</option><option value="admin">Admin</option></select>
              <button class="primary">Create</button>
            </form>
          </section>
          <section class="panel stack">
            <h2>Create Class</h2>
            <form method="post" action="/api/create-class">
              <input name="name" placeholder="class name" required>
              <input name="description" placeholder="description">
              <button class="primary">Create</button>
            </form>
          </section>
          <section class="panel stack">
            <h2>Assign User</h2>
            <form method="post" action="/api/assign-class">
              <select name="user_id">{account_options}</select>
              <select name="class_id">{class_options}</select>
              <button class="primary">Assign</button>
            </form>
          </section>
        </main>
        <section class="panel wide">
          <h2>Settings</h2>
          <form class="settings" method="post" action="/api/save-settings">
            {setting_input("model_path", settings)}
            {setting_input("confidence_threshold", settings)}
            {setting_input("note_taking_grace", settings)}
            {setting_input("eyes_closed_after", settings)}
            {setting_input("no_face_after", settings)}
            {setting_input("alert_cooldown", settings)}
            {setting_input("teacher_email", settings)}
            {setting_input("smtp_host", settings)}
            {setting_input("smtp_port", settings)}
            {setting_input("smtp_user", settings)}
            {setting_input("smtp_password", settings, "password")}
            {setting_input("alert_from", settings)}
            {setting_input("email_alerts", settings)}
            <button class="primary">Save Settings</button>
          </form>
        </section>
        <main class="grid two">
          <section class="panel"><h2>Accounts</h2><table><tbody>{rows}</tbody></table></section>
          <section class="panel"><h2>Classes</h2><table><tbody>{class_rows}</tbody></table></section>
        </main>
        <section class="panel wide">
          <h2>Reports</h2>
          <div class="buttons"><a class="button" href="/export/sessions.csv">Export CSV</a><a class="button" href="/export/sessions.pdf">Export PDF</a></div>
          <form method="post" action="/api/delete-history"><button class="danger">Delete All Session History</button></form>
        </section>
        <script>{ADMIN_JS}</script>
        """
        self.send_html(layout("Admin Portal", body))


    def api_session_start(self) -> None:
        user = self.require_user(("user",))
        if not user:
            return
        data = self.json_body()
        class_id = int(data.get("class_id") or 0) or None
        session_name = data.get("session_name") or "Lecture Session"

        # Terminate any orphaned live sessions for the same user to avoid duplicate listings
        uid = int(user["id"])
        with STATE_LOCK:
            old_session_ids = [sid for sid, s in LIVE_SESSIONS.items() if s.user_id == uid]
            old_sessions = [LIVE_SESSIONS.pop(sid) for sid in old_session_ids]
        for s in old_sessions:
            DB.end_session(s.session_id, s.average_attention, s.looking_away_seconds, s.eyes_closed_seconds)

        session_id = DB.start_session(uid, class_id, session_name)
        live = LiveSession(
            session_id=session_id,
            user_id=uid,
            display_name=user["display_name"],
            class_id=class_id,
            class_name=class_name_for(class_id),
            session_name=session_name,
            started_at=time.time(),
        )
        with STATE_LOCK:
            LIVE_SESSIONS[session_id] = live
        self.send_json({"ok": True, "session_id": session_id})

    def api_session_end(self) -> None:
        user = self.require_user(("user",))
        if not user:
            return
        data = self.json_body()
        session_id = int(data.get("session_id") or 0)
        with STATE_LOCK:
            session = LIVE_SESSIONS.pop(session_id, None)
        if session:
            DB.end_session(session_id, session.average_attention, session.looking_away_seconds, session.eyes_closed_seconds)
        self.send_json({"ok": True})

    def api_frame(self) -> None:
        user = self.require_user(("user",))
        if not user:
            return
        data = self.json_body()
        session_id = int(data.get("session_id") or 0)
        with STATE_LOCK:
            session = LIVE_SESSIONS.get(session_id)
        if not session or session.user_id != int(user["id"]):
            self.send_json({"ok": False, "error": "No active session"}, 400)
            return
        try:
            frame = decode_frame(data.get("image", ""))
            if frame is None:
                raise ValueError("Invalid camera frame")
            prediction = get_engine().predict(frame)
            result = apply_prediction(session, prediction)
            self.send_json(result)
        except Exception as exc:
            error_message = str(exc)
            try:
                self.send_json({"ok": False, "error": error_message}, 500)
            except Exception:
                pass

    def api_demo_event(self) -> None:
        user = self.require_user(("user",))
        if not user:
            return
        data = self.json_body()
        session_id = int(data.get("session_id") or 0)
        label = data.get("label", "looking_screen")
        with STATE_LOCK:
            session = LIVE_SESSIONS.get(session_id)
        if not session or session.user_id != int(user["id"]):
            self.send_json({"ok": False, "error": "No active session"}, 400)
            return
        prediction = Prediction(label, 0.99 if label != "no_face" else 0.0, label != "no_face", None)
        self.send_json(apply_prediction(session, prediction))

    def api_live(self) -> None:
        user = self.require_user(("admin", "main_admin"))
        if not user:
            return
        cleanup_expired_sessions()
        with STATE_LOCK:
            sessions = [session_to_dict(session) for session in LIVE_SESSIONS.values()]
        alerts = [
            {
                "id": row["id"],
                "time": row["alert_time"],
                "display_name": row["display_name"],
                "label": readable_label(row["label"]),
                "message": row["message"],
                "acknowledged": row["acknowledged"],
            }
            for row in DB.list_alerts()[:50]
            if not row["acknowledged"]
        ]
        self.send_json({"ok": True, "sessions": sessions, "alerts": alerts})

    def api_sessions(self) -> None:
        user = self.require_user(("admin", "main_admin"))
        if not user:
            return
        rows = [
            {key: row[key] for key in row.keys()}
            for row in DB.list_sessions()
        ]
        self.send_json({"ok": True, "sessions": rows})

    def create_account(self) -> None:
        if not self.require_user(("admin", "main_admin")):
            return
        data = self.form()
        DB.create_account(data["username"], data["password"], data["role"], data["display_name"])
        self.redirect("/admin")

    def create_class(self) -> None:
        if not self.require_user(("admin", "main_admin")):
            return
        data = self.form()
        DB.create_class(data["name"], data.get("description", ""))
        self.redirect("/admin")

    def assign_class(self) -> None:
        if not self.require_user(("admin", "main_admin")):
            return
        data = self.form()
        DB.enroll_user(int(data["class_id"]), int(data["user_id"]))
        self.redirect("/admin")

    def save_settings(self) -> None:
        if not self.require_user(("admin", "main_admin")):
            return
        DB.save_settings(self.form())
        self.redirect("/admin")

    def ack_alert(self) -> None:
        if not self.require_user(("admin", "main_admin")):
            return
        data = self.json_body()
        DB.acknowledge_alert(int(data["alert_id"]))
        self.send_json({"ok": True})

    def delete_history(self) -> None:
        if not self.require_user(("admin", "main_admin")):
            return
        DB.delete_history()
        self.redirect("/admin")

    def export_csv(self) -> None:
        if not self.require_user(("admin", "main_admin")):
            return
        rows = DB.list_sessions()
        columns = (
            "id",
            "display_name",
            "class_name",
            "session_name",
            "started_at",
            "ended_at",
            "average_attention",
            "looking_away_seconds",
            "eyes_closed_seconds",
            "alert_count",
        )
        from io import StringIO

        buffer = StringIO()
        writer = csv.writer(buffer)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row[col] if row[col] is not None else "" for col in columns])
        data = buffer.getvalue().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/csv")
        self.send_header("Content-Disposition", "attachment; filename=attention_sessions.csv")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def export_pdf(self) -> None:
        if not self.require_user(("admin", "main_admin")):
            return
        rows = DB.list_sessions()
        path = Path("/private/tmp/attention_sessions_report.pdf")
        lines = ["Attention Session Report", f"Generated: {now_text()}", ""]
        for row in rows[:80]:
            lines.append(
                f"{row['display_name']} | {row['session_name']} | avg {row['average_attention']}% | "
                f"away {row['looking_away_seconds']}s | closed {row['eyes_closed_seconds']}s | alerts {row['alert_count']}"
            )
        write_simple_pdf(path, lines)
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Content-Disposition", "attachment; filename=attention_sessions_report.pdf")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def setting_input(key: str, settings: dict[str, str], input_type: str = "text") -> str:
    return (
        f"<label>{html_escape(key.replace('_', ' ').title())}"
        f"<input name='{html_escape(key)}' type='{input_type}' value='{html_escape(settings.get(key, ''))}'></label>"
    )


def layout(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html_escape(title)}</title>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
  <style>{CSS}</style>
</head>
<body>{body}</body>
</html>"""


CSS = """
:root{--bg:#0f172a;--panel:rgba(30,41,59,0.6);--ink:#f8fafc;--muted:#94a3b8;--line:rgba(255,255,255,0.08);--blue:#3b82f6;--red:#ef4444;--green:#10b981;--glow:rgba(59,130,246,0.3)}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 'Outfit',sans-serif;background-image:radial-gradient(circle at 50% 0%, #1e293b 0%, #0f172a 100%);min-height:100vh}a{color:inherit;text-decoration:none}
h1,h2,h3,p{margin:0}h1{font-size:32px;letter-spacing:-0.5px;font-weight:800}h2{font-size:20px;margin-bottom:16px;font-weight:600}.eyebrow{text-transform:uppercase;font-size:12px;letter-spacing:0.1em;color:var(--blue);font-weight:800}.muted,.tiny{color:var(--muted)}.tiny{font-size:12px}
.topbar{display:flex;align-items:center;justify-content:space-between;padding:20px 32px;border-bottom:1px solid var(--line);background:rgba(15,23,42,0.8);backdrop-filter:blur(12px);position:sticky;top:0;z-index:10}.topbar nav{display:flex;gap:12px}
.login{min-height:100vh;display:grid;grid-template-columns:1fr 420px;gap:30px;align-items:center;padding:48px;max-width:1120px;margin:auto}.login h1{font-size:48px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:24px;box-shadow:0 8px 32px rgba(0,0,0,0.3);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px)}
.grid{display:grid;gap:24px;padding:24px 32px}.grid.two{grid-template-columns:minmax(0,1.4fr) minmax(400px,.8fr)}.grid.three{grid-template-columns:repeat(3,minmax(0,1fr))}.wide{margin:0 32px 24px}.stack{display:flex;flex-direction:column;gap:16px}
label{display:grid;gap:8px;color:var(--muted);font-weight:600}input,select{width:100%;border:1px solid var(--line);border-radius:10px;background:rgba(0,0,0,0.2);padding:12px 14px;color:#fff;font:inherit;transition:border-color 0.2s;backdrop-filter:blur(4px)}input:focus,select:focus{outline:none;border-color:var(--blue);box-shadow:0 0 0 3px var(--glow)}
.check{display:flex;grid-template-columns:auto 1fr;align-items:center;gap:10px;cursor:pointer}.check input{width:auto;margin:0}
button,.button{border:none;border-radius:10px;background:rgba(255,255,255,0.05);color:var(--ink);padding:12px 18px;font:inherit;font-weight:600;cursor:pointer;transition:all 0.2s;border:1px solid var(--line)}
.primary{background:linear-gradient(135deg, #3b82f6, #2563eb);border:none;box-shadow:0 4px 12px var(--glow)}.primary:hover{transform:translateY(-1px);box-shadow:0 6px 16px var(--glow);background:linear-gradient(135deg, #60a5fa, #3b82f6)}
.ghost{background:transparent}.ghost:hover{background:rgba(255,255,255,0.1)}
.buttons{display:flex;gap:12px;flex-wrap:wrap}
video{display:block;width:100%;aspect-ratio:16/10;background:#000;border-radius:12px;object-fit:cover;box-shadow:0 8px 24px rgba(0,0,0,0.4);border:1px solid var(--line)}canvas{display:none}
.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.premium-card, .metrics div{border:1px solid var(--line);border-radius:12px;padding:16px;background:rgba(255,255,255,0.02);box-shadow:inset 0 1px 1px rgba(255,255,255,0.05);transition:transform 0.2s, border-color 0.2s;text-align:center}
.premium-card:hover, .metrics div:hover{transform:translateY(-2px);border-color:rgba(59,130,246,0.5);box-shadow:0 4px 20px var(--glow)}
.metrics span{display:block;font-size:28px;font-weight:800;background:linear-gradient(to right, #fff, #cbd5e1);-webkit-background-clip:text;-webkit-text-fill-color:transparent}.metrics small, .premium-card small{color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:0.05em;font-size:11px}
.big-value{display:block;font-size:24px;font-weight:800;color:#fff;margin-bottom:4px}
.med-value{display:block;font-size:16px;font-weight:800;color:#e2e8f0;margin-bottom:2px}
.notice{padding:14px 16px;border-radius:10px;background:rgba(59,130,246,0.1);color:#60a5fa;border:1px solid rgba(59,130,246,0.2);font-weight:600}.demo{display:flex;gap:12px}
table{width:100%;border-collapse:collapse}th,td{border-bottom:1px solid var(--line);padding:14px 12px;text-align:left;vertical-align:top}th{font-size:12px;text-transform:uppercase;color:var(--muted);letter-spacing:0.06em;font-weight:800}td{font-size:14px}
.settings{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.settings button{align-self:end}
@media(max-width:900px){.login,.grid.two,.grid.three,.settings{grid-template-columns:1fr}.login{padding:24px}.topbar{padding:16px 20px}.grid,.wide{padding:20px;margin:0}}
"""


USER_JS = """
let sessionId=null, video=null, timer=null;
const $=id=>document.getElementById(id);
async function post(url,payload){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});return await r.json();}
function setResult(d){
    if(!d.ok){$('alert').textContent=d.error||'Error';return;}
    $('state').textContent=d.state;
    $('attention').textContent=d.attention+'%';
    $('confidence').textContent=d.confidence+'%';
    $('alert').textContent='Soft alert: '+d.alert;
    if(d.emotions){
        const isUnderstanding = (d.emotions.engagement >= 2 && d.emotions.confusion <= 1);
        const isHappy = (d.emotions.frustration === 0 && d.emotions.boredom <= 1);
        
        $('understanding').textContent = isUnderstanding ? 'Yes 🧠' : 'No ❌';
        $('understanding').style.color = isUnderstanding ? '#10b981' : '#ef4444';
        
        $('happy').textContent = isHappy ? 'Yes 😊' : 'No 😐';
        $('happy').style.color = isHappy ? '#10b981' : '#94a3b8';

        const lvls=['None','Low','High','Max'];
        $('boredom').textContent=lvls[d.emotions.boredom]||'-';
        $('engagement').textContent=lvls[d.emotions.engagement]||'-';
        $('confusion').textContent=lvls[d.emotions.confusion]||'-';
        $('frustration').textContent=lvls[d.emotions.frustration]||'-';
    }
}
async function start(){if(!$('consent').checked){alert('Consent is required before starting.');return;}let started=await post('/api/session/start',{class_id:$('classId').value,session_name:$('sessionName').value});if(!started.ok)return;sessionId=started.session_id;try{video=$('video');video.srcObject=await navigator.mediaDevices.getUserMedia({video:true,audio:false});}catch(e){$('alert').textContent='Webcam unavailable. Demo signal controls are active.';}timer=setInterval(sendFrame,1000);}
async function sendFrame(){if(!sessionId||!video||!video.videoWidth)return;let c=$('canvas'),ctx=c.getContext('2d');ctx.drawImage(video,0,0,c.width,c.height);let image=c.toDataURL('image/jpeg',0.72);setResult(await post('/api/frame',{session_id:sessionId,image}));}
async function end(){if(timer)clearInterval(timer);timer=null;if(video&&video.srcObject){video.srcObject.getTracks().forEach(t=>t.stop());}if(sessionId){await post('/api/session/end',{session_id:sessionId});sessionId=null;}$('alert').textContent='Soft alert: session saved';}
$('startBtn').onclick=start;$('endBtn').onclick=end;$('demoBtn').onclick=async()=>{if(!sessionId){alert('Start a session first.');return;}setResult(await post('/api/demo-event',{session_id:sessionId,label:$('demoLabel').value}));};
"""


ADMIN_JS = """
async function post(url,payload){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});return await r.json();}
function td(v){return '<td>'+String(v??'').replace(/[&<>]/g,s=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[s]))+'</td>'}
async function refreshLive(){let r=await fetch('/api/live');let d=await r.json();if(!d.ok)return;document.querySelector('#liveTable tbody').innerHTML=d.sessions.map(s=>'<tr>'+td(s.display_name)+td(s.class_name)+td(s.label)+td(s.average_attention+'%')+td(s.elapsed)+'</tr>').join('')||'<tr><td colspan=5>No active sessions</td></tr>';document.querySelector('#alertTable tbody').innerHTML=d.alerts.map(a=>'<tr>'+td(a.time)+td(a.display_name)+td(a.label)+td(a.message)+'<td><button onclick=\"ack('+a.id+')\">Ack</button></td></tr>').join('')||'<tr><td colspan=5>No alerts</td></tr>';}
async function refreshSessions(){let r=await fetch('/api/sessions');let d=await r.json();if(!d.ok)return;document.querySelector('#sessionTable tbody').innerHTML=d.sessions.map(s=>'<tr>'+td(s.display_name)+td(s.class_name)+td(s.session_name)+td(s.average_attention+'%')+td(s.looking_away_seconds+'s')+td(s.eyes_closed_seconds+'s')+td(s.alert_count)+'</tr>').join('')||'<tr><td colspan=7>No sessions yet</td></tr>';}
async function ack(id){await post('/api/ack-alert',{alert_id:id});refreshLive();}
setInterval(refreshLive,2000);setInterval(refreshSessions,5000);refreshLive();refreshSessions();
"""


def start_server(port: int, portal: str) -> ThreadingHTTPServer:
    original_port = port
    for candidate in (port, port + 100, port + 200):
        try:
            server = ReusableThreadingHTTPServer(("127.0.0.1", candidate), WebHandler)
            port = candidate
            break
        except OSError:
            server = None
    if server is None:
        raise OSError(f"Could not bind a localhost port for {portal} near {original_port}")
    server.portal = portal
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"{PORTALS[portal]['label']}: http://127.0.0.1:{port}")
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local web portals for Attention Tracker.")
    parser.add_argument("--single", choices=tuple(PORTALS), help="Run only one portal.")
    parser.add_argument("--port", type=int, help="Port for --single.")
    args = parser.parse_args()

    # Pre-load/warmup PyTorch model in a background thread
    print("Pre-loading PyTorch model in the background...")
    threading.Thread(target=get_engine, daemon=True).start()

    if args.single:
        start_server(args.port or PORTALS[args.single]["port"], args.single)
    else:
        for portal, config in PORTALS.items():
            start_server(config["port"], portal)

    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("Stopping.")


if __name__ == "__main__":
    main()
