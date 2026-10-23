"""In-memory session leaderboard.

Stores evaluation results for the current Streamlit session only. No database,
no file persistence - the leaderboard resets when the app is reloaded, as
required by the project brief.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LeaderboardEntry:
    """A single leaderboard row."""

    name: str
    job_role: str
    score: int
    level: str


def _get(session_state, key, default):
    """Read a value from session_state via attribute or item access."""
    try:
        return getattr(session_state, key)
    except AttributeError:
        return session_state.get(key, default)


def _set(session_state, key, value):
    """Write a value to session_state via attribute or item access."""
    try:
        setattr(session_state, key, value)
    except AttributeError:
        session_state[key] = value


def _ensure_board(session_state) -> list[LeaderboardEntry]:
    """Return the board list from session_state, creating it if missing."""
    try:
        has = "leaderboard" in session_state
    except TypeError:
        has = hasattr(session_state, "leaderboard")
    if not has:
        _set(session_state, "leaderboard", [])
    return _get(session_state, "leaderboard", [])


def add_to_leaderboard(session_state, entry: LeaderboardEntry) -> None:
    """Add an entry and keep the board sorted by score descending."""
    board = _ensure_board(session_state)
    board.append(entry)
    board.sort(key=lambda e: e.score, reverse=True)


def get_leaderboard(session_state) -> list[LeaderboardEntry]:
    """Return the current leaderboard (sorted, highest first)."""
    return list(_ensure_board(session_state))


def clear_leaderboard(session_state) -> None:
    """Remove all entries from the leaderboard."""
    _set(session_state, "leaderboard", [])


def medal_for_rank(rank: int) -> str:
    """Return a medal emoji string for top-3 ranks, empty string otherwise."""
    return {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "")
