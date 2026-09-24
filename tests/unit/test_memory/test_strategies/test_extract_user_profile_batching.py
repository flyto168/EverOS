"""Multi-run batch progression for :func:`extract_user_profile`.

Uses a **real** ``_MemcellRepo`` (SQLite-backed, tmp DB) so
``memcell_repo.select_oldest_batch`` runs its actual query instead of a
mock. Only the cluster/episode lookup, LLM extractor, and markdown
reader/writer are mocked. This exercises the actual interaction between
bounded batch selection and profile-timestamp advancement across
sequential runs against a backlog larger than ``PROFILE_MAX_MEMCELLS``:
each run must pick up exactly where the previous one left off, with no
memcell skipped and none processed twice.
"""

from __future__ import annotations

import importlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from everalgo.types import ChatMessage as AlgoChatMessage
from everalgo.types import MemCell as AlgoMemCell
from everalgo.types import Profile as AlgoProfile
from sqlmodel import SQLModel

from everos.component.utils.datetime import to_timestamp_ms
from everos.config import SqliteSettings
from everos.core.persistence import (
    MemoryRoot,
    create_session_factory,
    create_system_engine,
)
from everos.infra.ome.testing import FakeStrategyContext
from everos.infra.persistence.markdown import UserProfileFrontmatter
from everos.infra.persistence.sqlite.repos.memcell import _MemcellRepo
from everos.infra.persistence.sqlite.tables import Memcell
from everos.memory._partition_locks import _reset_for_tests
from everos.memory.events import EpisodeExtracted
from everos.memory.strategies.extract_user_profile import (
    PROFILE_MAX_MEMCELLS,
    extract_user_profile,
)

OWNER_ID = "u_alice"


@pytest.fixture(autouse=True)
def _isolate_partition_locks() -> None:
    _reset_for_tests()


@pytest.fixture
async def repo(tmp_path: Path) -> _MemcellRepo:
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    engine = create_system_engine(mr.system_db, SqliteSettings())
    factory = create_session_factory(engine)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    return _MemcellRepo(session_factory=factory)


def _seed_backlog(count: int) -> list[tuple[str, int, Memcell]]:
    """Build ``count`` memcells one second apart, oldest first.

    Returns ``(memcell_id, timestamp_ms, row)`` triples; ``timestamp_ms``
    is the value embedded in the algo payload — kept equal to the SQLite
    ``timestamp`` column so ``select_oldest_batch`` (SQLite side) and
    ``ProfileExtractor``'s ``memcells[-1].timestamp`` (algo side) advance
    in lockstep, matching the real write path.
    """
    base = datetime(2026, 1, 1, tzinfo=UTC)
    out: list[tuple[str, int, Memcell]] = []
    for i in range(count):
        memcell_id = f"mc_{i:04d}"
        ts_dt = base + timedelta(seconds=i)
        ts_ms = to_timestamp_ms(ts_dt)
        cell = AlgoMemCell(
            items=[
                AlgoChatMessage(
                    id=f"{memcell_id}_m1",
                    role="user",
                    content=f"message {i}",
                    timestamp=ts_ms,
                    sender_id=OWNER_ID,
                )
            ],
            timestamp=ts_ms,
        )
        row = Memcell(
            memcell_id=memcell_id,
            app_id="default",
            project_id="default",
            session_id="s_test",
            track="user_memory",
            raw_type="chat",
            message_ids_json="[]",
            sender_ids_json="[]",
            payload_json=cell.model_dump_json(),
            timestamp=ts_dt,
        )
        out.append((memcell_id, ts_ms, row))
    return out


class _FakeProfileStore:
    """Stand-in for the ProfileReader/ProfileWriter pair, sharing state
    across sequential ``extract_user_profile`` runs the way the real
    ``users/<user_id>/user.md`` file would."""

    def __init__(self) -> None:
        self._profiles: dict[str, UserProfileFrontmatter] = {}

    async def read(self, owner_id: str, *, schema, app_id, project_id):
        fm = self._profiles.get(owner_id)
        return None if fm is None else (fm, fm.summary)

    async def write(self, owner_id: str, *, frontmatter, body, app_id, project_id):
        self._profiles[owner_id] = frontmatter


async def test_backlog_worked_off_across_sequential_runs_with_no_skip_or_repeat(
    repo: _MemcellRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """100 memcells, cap 50/run → 2 runs fully drain the backlog.

    Each run's processed set must be disjoint from the other's, their
    union must cover every seeded memcell exactly once, and the final
    profile timestamp must equal the newest memcell's timestamp — proving
    ``last_profile_ts`` advances to the bounded batch's own max, not to
    "now" or the full unbounded candidate set's max.
    """
    backlog_size = 2 * PROFILE_MAX_MEMCELLS
    backlog = _seed_backlog(backlog_size)
    await repo.insert_many([row for _, _, row in backlog])
    id_by_ts = {ts_ms: mid for mid, ts_ms, _ in backlog}
    all_ts = set(id_by_ts)
    newest_id, newest_ts = backlog[-1][0], backlog[-1][1]

    store = _FakeProfileStore()
    captured_batches: list[list[int]] = []

    async def fake_list_by_owner_after_ts(
        *, owner_id, after_ts, parent_type, app_id, project_id, columns
    ):
        return [{"parent_id": mid} for mid, ts_ms, _ in backlog if ts_ms > after_ts]

    async def fake_aextract(memcells, *, sender_id, old_profile=None, **_kwargs):
        batch_ts = [mc.timestamp for mc in memcells]
        captured_batches.append(batch_ts)
        return AlgoProfile(
            owner_id=sender_id,
            summary=f"profile through {max(batch_ts)}",
            timestamp=max(batch_ts),
            explicit_info=[],
            implicit_traits=[],
        )

    with (
        patch("everos.memory.strategies.extract_user_profile.memcell_repo", repo),
        patch(
            "everos.memory.strategies.extract_user_profile.episode_repo"
        ) as mock_episode_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileExtractor"
        ) as mock_extractor_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileReader",
            return_value=store,
        ),
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileWriter",
            return_value=store,
        ),
    ):
        mock_episode_repo.list_by_owner_after_ts = AsyncMock(
            side_effect=fake_list_by_owner_after_ts
        )
        mock_episode_repo.count_by_owner = AsyncMock(return_value=1)
        mock_extractor_cls.return_value.aextract = fake_aextract
        mod = importlib.import_module("everos.memory.strategies.extract_user_profile")
        monkeypatch.setattr(mod, "_writer", None, raising=False)
        monkeypatch.setattr(mod, "_reader", None, raising=False)

        event = EpisodeExtracted(
            memcell_id=newest_id,
            episode_entry_id="ep_1",
            episode_text="text",
            episode_timestamp_ms=newest_ts,
            owner_id=OWNER_ID,
            session_id="s_test",
            source="pipeline",
        )

        await extract_user_profile(event, FakeStrategyContext())
        await extract_user_profile(event, FakeStrategyContext())

    assert len(captured_batches) == 2, "expected exactly one batch per run"
    run1_ts, run2_ts = (set(b) for b in captured_batches)

    assert len(captured_batches[0]) == PROFILE_MAX_MEMCELLS
    assert len(captured_batches[1]) == PROFILE_MAX_MEMCELLS
    assert run1_ts.isdisjoint(run2_ts), "same memcell processed twice across runs"
    assert run1_ts | run2_ts == all_ts, "a backlog memcell was never processed"
    assert max(run1_ts) < min(run2_ts), "run 2 must pick up strictly after run 1"

    final_profile = store._profiles[OWNER_ID]
    assert final_profile.profile_timestamp_ms == newest_ts, (
        "profile timestamp must advance to the last processed memcell, "
        "not to 'now' or the full unbounded candidate set's max"
    )
