from pathlib import Path
import sqlite3
from uuid import uuid4

from app.schemas import ChatMessage, ChatSession


class ChatHistoryStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    document_id TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, id)"
            )

    def create_session(self, title: str = "Chat mới", document_id: str | None = None) -> ChatSession:
        session_id = uuid4().hex
        clean_title = title.strip()[:80] or "Chat mới"
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO chat_sessions (id, title, document_id) VALUES (?, ?, ?)",
                (session_id, clean_title, document_id),
            )
        return self.get_session(session_id)

    def get_or_create_session(self, session_id: str | None, title: str, document_id: str | None) -> ChatSession:
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
        return self.create_session(title=title, document_id=document_id)

    def list_sessions(self) -> list[ChatSession]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, title, document_id, created_at, updated_at
                FROM chat_sessions
                ORDER BY updated_at DESC, created_at DESC
                """
            ).fetchall()
        return [ChatSession(**dict(row)) for row in rows]

    def get_session(self, session_id: str) -> ChatSession | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, title, document_id, created_at, updated_at FROM chat_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        return ChatSession(**dict(row)) if row else None

    def delete_session(self, session_id: str) -> bool:
        with self._connect() as connection:
            connection.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
            cursor = connection.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
        return cursor.rowcount > 0

    def add_message(self, session_id: str, role: str, content: str) -> ChatMessage:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO chat_messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content),
            )
            connection.execute(
                "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (session_id,),
            )
            message_id = int(cursor.lastrowid)
        return self.get_message(message_id)

    def list_messages(self, session_id: str) -> list[ChatMessage]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, session_id, role, content, created_at
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
        return [ChatMessage(**dict(row)) for row in rows]

    def get_message(self, message_id: int) -> ChatMessage:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, session_id, role, content, created_at FROM chat_messages WHERE id = ?",
                (message_id,),
            ).fetchone()
        return ChatMessage(**dict(row))

    def update_title_from_question(self, session_id: str, question: str) -> None:
        title = question.strip().replace("\n", " ")[:60]
        if not title:
            return
        with self._connect() as connection:
            current = connection.execute(
                "SELECT title FROM chat_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if current and current["title"] == "Chat mới":
                connection.execute(
                    "UPDATE chat_sessions SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (title, session_id),
                )
