"""SQLite store shared by all agents. Tables: notes, agent runs, and trials
(one row per score_model call, so a result can be compared with the number of
models tried).

Agents access the store only through the tools in arh.tools. No tool returns
the holdout column.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

KINDS = ("summary", "finding", "idea", "question", "forecast")
MAX_TEXT = 4000

SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY, kind TEXT NOT NULL, author TEXT NOT NULL, run TEXT,
    text TEXT NOT NULL, refs TEXT NOT NULL DEFAULT '[]', artifact TEXT,
    cv REAL, holdout REAL, ts TEXT NOT NULL);
CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(text, kind UNINDEXED);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, agent TEXT, task TEXT, model TEXT, started TEXT, ended TEXT,
    cost REAL, turns INTEGER, error TEXT);
CREATE TABLE IF NOT EXISTS trials (
    id INTEGER PRIMARY KEY, run TEXT, artifact_sha TEXT, cv REAL, error TEXT, ts TEXT);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        with self._db() as db:
            db.executescript(SCHEMA)

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        return db

    # notes

    def add(self, kind: str, author: str, text: str, refs: list[int] = (), artifact: str | None = None,
            run: str | None = None, cv: float | None = None) -> int:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if not text.strip() or len(text) > MAX_TEXT:
            raise ValueError(f"text must be nonempty and at most {MAX_TEXT} characters")
        refs = sorted({int(r) for r in refs})
        with self._db() as db:
            known = {r[0] for r in db.execute(
                f"SELECT id FROM notes WHERE id IN ({','.join('?' * len(refs))})", refs)} if refs else set()
            if missing := set(refs) - known:
                raise ValueError(f"refs to unknown notes: {sorted(missing)}")
            cur = db.execute(
                "INSERT INTO notes (kind, author, run, text, refs, artifact, cv, ts) VALUES (?,?,?,?,?,?,?,?)",
                (kind, author, run, text, json.dumps(refs), artifact, cv, now()))
            db.execute("INSERT INTO notes_fts (rowid, text, kind) VALUES (?,?,?)", (cur.lastrowid, text, kind))
            return cur.lastrowid

    def set_holdout(self, note_id: int, holdout: float) -> None:
        with self._db() as db:
            db.execute("UPDATE notes SET holdout = ? WHERE id = ?", (holdout, note_id))

    def get(self, note_id: int) -> dict | None:
        with self._db() as db:
            row = db.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
            if row is None:
                return None
            note = dict(row)
            note["refs"] = json.loads(note["refs"])
            note["cited_by"] = [r[0] for r in db.execute(
                "SELECT n.id FROM notes n, json_each(n.refs) j WHERE j.value = ? ORDER BY n.id", (note_id,))]
            return note

    def search(self, query: str = "", kind: str | None = None, open_only: bool = False, k: int = 8) -> list[dict]:
        """Best FTS matches for any word of ``query`` (newest first when empty).
        ``open_only``: questions no finding refs, ideas no forecast refs."""
        words = re.findall(r"\w+", query)
        sql, args = "SELECT n.* FROM notes n", []
        where = []
        if words:
            sql += " JOIN notes_fts f ON f.rowid = n.id"
            where.append("notes_fts MATCH ?")
            args.append(" OR ".join(f'"{w}"' for w in words))
        if kind:
            where.append("n.kind = ?")
            args.append(kind)
        if open_only:
            where.append("n.kind IN ('question', 'idea') AND NOT EXISTS (SELECT 1 FROM notes m, json_each(m.refs) j "
                         "WHERE j.value = n.id AND m.kind = CASE n.kind WHEN 'question' THEN 'finding' "
                         "ELSE 'forecast' END)")
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY " + ("bm25(notes_fts)" if words else "n.id DESC") + " LIMIT ?"
        args.append(k)
        with self._db() as db:
            rows = [dict(r) for r in db.execute(sql, args)]
        for r in rows:
            r["refs"] = json.loads(r["refs"])
        return rows

    def counts(self) -> dict[str, int]:
        with self._db() as db:
            return dict(db.execute("SELECT kind, COUNT(*) FROM notes GROUP BY kind").fetchall())

    def forecasts(self) -> list[dict]:
        with self._db() as db:
            return [dict(r) for r in db.execute(
                "SELECT id, run, cv, holdout, substr(text, 1, 100) AS text FROM notes "
                "WHERE kind = 'forecast' ORDER BY id")]

    # runs and trials

    def start_run(self, run: str, agent: str, task: str, model: str | None) -> None:
        with self._db() as db:
            db.execute("INSERT INTO runs (id, agent, task, model, started) VALUES (?,?,?,?,?)",
                       (run, agent, task, model, now()))

    def end_run(self, run: str, cost: float | None, turns: int | None, error: str | None) -> None:
        with self._db() as db:
            db.execute("UPDATE runs SET ended = ?, cost = ?, turns = ?, error = ? WHERE id = ?",
                       (now(), cost, turns, error, run))

    def runs(self) -> list[dict]:
        with self._db() as db:
            return [dict(r) for r in db.execute("SELECT * FROM runs ORDER BY started")]

    def add_trial(self, run: str, artifact_sha: str, cv: float | None, error: str | None) -> int:
        with self._db() as db:
            db.execute("INSERT INTO trials (run, artifact_sha, cv, error, ts) VALUES (?,?,?,?,?)",
                       (run, artifact_sha, cv, error, now()))
            return db.execute("SELECT COUNT(DISTINCT artifact_sha) FROM trials").fetchone()[0]


def render(note: dict, full: bool = False) -> str:
    """Format a note as shown to agents. The holdout score is not included."""
    head = f"#{note['id']} {note['kind']} by {note['author']}"
    if note.get("cv") is not None:
        head += f" (cv R2 {note['cv']:.4f})"
    if note["refs"]:
        head += f" refs {note['refs']}"
    if note.get("cited_by"):
        head += f" cited by {note['cited_by']}"
    text = note["text"] if full or len(note["text"]) <= 400 else note["text"][:400] + " ..."
    tail = f"\n  artifact: {note['artifact']}" if note.get("artifact") else ""
    return f"{head}\n  {text}{tail}"
