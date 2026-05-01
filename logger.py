"""
logger.py
Records trial events to a local SQLite database and exports CSV on session end.

Tables:
- sessions:  one row per participant session
- trials:    one row per response cycle
"""

import sqlite3
import csv
import time
from pathlib import Path
from datetime import datetime


# ─── Schema ───────────────────────────────────────────────────────────────────

CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    participant  TEXT,
    group_num    INTEGER,      -- 2 or 3 (number of lists)
    started_at   TEXT,
    ended_at     TEXT
);
"""

CREATE_TRIALS = """
CREATE TABLE IF NOT EXISTS trials (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT,
    phase           INTEGER,   -- 1, 2, or 3
    phase_attempt   INTEGER DEFAULT 1,
    cycle           INTEGER,   -- cycle number within session (1-indexed)
    timestamp       REAL,      -- Unix time when cycle started
    response_raw    TEXT,      -- exactly what Whisper heard (or NULL if no response)
    response_word   TEXT,      -- canonical matched word (or NULL)
    matched_list    INTEGER,   -- which list it matched (or NULL)
    is_novel        INTEGER,   -- 1 = not on any list
    is_repeat       INTEGER,   -- 1 = word already used this session
    reinforced      INTEGER,   -- 1 = pleasant sound played
    response_time   REAL,      -- seconds from cycle start to response (or NULL)
    discarded       INTEGER DEFAULT 0,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
"""


# ─── Logger ───────────────────────────────────────────────────────────────────

class SessionLogger:

    def __init__(self, db_path: str, participant_id: str, group_num: int):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.participant_id = participant_id
        self.group_num = group_num

        # session_id = participant + timestamp, unique per run
        self.session_id = f"{participant_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.started_at = datetime.now().isoformat()

        self._init_db()
        self._create_session()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(CREATE_SESSIONS)
            conn.execute(CREATE_TRIALS)
            self._migrate_trials(conn)

    def _migrate_trials(self, conn):
        cols = {row[1] for row in conn.execute("PRAGMA table_info(trials)").fetchall()}
        if "phase_attempt" not in cols:
            conn.execute("ALTER TABLE trials ADD COLUMN phase_attempt INTEGER DEFAULT 1")
        if "discarded" not in cols:
            conn.execute("ALTER TABLE trials ADD COLUMN discarded INTEGER DEFAULT 0")

    def _create_session(self):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, participant, group_num, started_at) VALUES (?,?,?,?)",
                (self.session_id, self.participant_id, self.group_num, self.started_at)
            )

    def log_trial(
        self,
        phase: int,
        cycle: int,
        timestamp: float,
        match_result: dict,     # output from matcher.match()
        reinforced: bool,
        response_time: float | None,
        phase_attempt: int = 1,
        discarded: bool = False,
    ):
        """Log one completed cycle."""
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO trials
                   (session_id, phase, phase_attempt, cycle, timestamp,
                    response_raw, response_word, matched_list,
                    is_novel, is_repeat, reinforced, response_time, discarded)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    self.session_id,
                    phase,
                    phase_attempt,
                    cycle,
                    timestamp,
                    match_result.get("raw"),
                    match_result.get("word"),
                    match_result.get("list"),
                    int(match_result.get("novel", False)),
                    int(match_result.get("repeat", False)),
                    int(reinforced),
                    response_time,
                    int(discarded),
                )
            )

    def log_no_response(self, phase: int, cycle: int, timestamp: float, phase_attempt: int = 1):
        """Log a cycle where participant said nothing."""
        self.log_trial(
            phase=phase,
            cycle=cycle,
            timestamp=timestamp,
            match_result={"raw": None, "word": None, "list": None, "novel": False, "repeat": False},
            reinforced=False,
            response_time=None,
            phase_attempt=phase_attempt,
        )

    def mark_phase_attempt_discarded(self, phase: int, phase_attempt: int):
        """Mark all trials from one phase attempt as excluded from analysis."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE trials SET discarded=1 WHERE session_id=? AND phase=? AND phase_attempt=?",
                (self.session_id, phase, phase_attempt),
            )

    def close(self, export_csv: bool = True) -> str | None:
        """Mark session as ended and optionally export CSV. Returns CSV path."""
        ended_at = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET ended_at=? WHERE session_id=?",
                (ended_at, self.session_id)
            )

        if export_csv:
            return self._export_csv()
        return None

    def get_summary(self) -> dict:
        """
        Return a summary dict for the completed session:
        {
          "phases": {
            1: {"cycles": int, "responses": int, "response_rate": float},
            2: ...,
            3: ...,
          },
          "resurgence_count": int,   # List-1 novel hits in Phase 3
          "total_cycles": int,
        }
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT phase, response_word, matched_list, is_repeat "
                "FROM trials WHERE session_id=? AND discarded=0 ORDER BY cycle",
                (self.session_id,)
            ).fetchall()

        from collections import defaultdict
        phases = defaultdict(lambda: {"cycles": 0, "responses": 0})
        resurgence_count = 0

        for phase, word, matched_list, is_repeat in rows:
            phases[phase]["cycles"] += 1
            if word is not None:
                phases[phase]["responses"] += 1
            # Resurgence: List-1 response in Phase 3 that hasn't been said before
            if phase == 3 and matched_list == 1 and not is_repeat:
                resurgence_count += 1

        result = {}
        for p, d in sorted(phases.items()):
            c = d["cycles"]
            r = d["responses"]
            result[p] = {
                "cycles": c,
                "responses": r,
                "response_rate": round(r / c, 3) if c > 0 else 0.0,
            }

        return {
            "phases": result,
            "resurgence_count": resurgence_count,
            "total_cycles": sum(d["cycles"] for d in result.values()),
        }

    def _export_csv(self) -> str:
        csv_path = str(Path(self.db_path).parent / f"{self.session_id}.csv")
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT * FROM trials WHERE session_id=? ORDER BY cycle",
                (self.session_id,)
            )
            rows = cursor.fetchall()
            cols = [d[0] for d in cursor.description]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(cols)
            writer.writerows(rows)

        return csv_path


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, tempfile

    db = os.path.join(tempfile.gettempdir(), "test_resurgence.db")
    logger = SessionLogger(db_path=db, participant_id="P001", group_num=2)

    # Simulate a few cycles
    logger.log_trial(
        phase=1, cycle=1, timestamp=time.time(),
        match_result={"raw": "cat", "word": "cat", "list": 1, "novel": False, "repeat": False},
        reinforced=True, response_time=1.2
    )
    logger.log_trial(
        phase=1, cycle=2, timestamp=time.time(),
        match_result={"raw": "kitten", "word": "kitten", "list": None, "novel": True, "repeat": False},
        reinforced=False, response_time=2.1
    )
    logger.log_no_response(phase=1, cycle=3, timestamp=time.time())

    csv_path = logger.close(export_csv=True)
    print(f"Session ID : {logger.session_id}")
    print(f"CSV saved  : {csv_path}")

    # Print CSV contents
    with open(csv_path) as f:
        print(f.read())
