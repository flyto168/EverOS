"""Repository for ``memcell`` table — singleton bound to ``sqlite_manager``.

Pure persistence: callers build the SQLModel ``Memcell`` rows (including
``message_ids_json`` / ``sender_ids_json``) and hand them in. The pipeline
is responsible for mapping algo-side messages back to everos
``message_id`` because algo's ``Message`` does not carry per-message
identifiers.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from everos.core.persistence.sqlite import RepoBase, session_scope

from ..sqlite_manager import get_session_factory
from ..tables import Memcell

_SQLITE_IN_CHUNK = 500
"""SQL ``IN (?, ?, ...)`` parameter cap for :meth:`_MemcellRepo.find_by_ids`.

SQLite's ``SQLITE_MAX_VARIABLE_NUMBER`` defaults to 999 on the shipped
amalgamation and rises to 32766 on ≥3.32; 500 stays well under the older
cap and keeps the query planner's cost estimate cheap. Callers passing
more ``memcell_ids`` than this fan out into multiple SELECTs, merged
in-memory — see the direct-path selector in
``memory/strategies/extract_user_profile.py`` where
``last_profile_ts=0`` can pull an owner's entire episode history."""


class _MemcellRepo(RepoBase[Memcell]):
    model = Memcell

    def _factory_lookup(self) -> async_sessionmaker[AsyncSession]:
        return get_session_factory()

    async def insert_many(self, rows: list[Memcell]) -> list[Memcell]:
        """Insert MemCell rows in one transaction; rows are constructed by caller."""
        async with session_scope(self._factory) as s:
            s.add_all(rows)
            await s.commit()
            for r in rows:
                await s.refresh(r)
        return rows

    async def find_by_ids(self, memcell_ids: list[str]) -> list[Memcell]:
        """Bulk fetch rows by primary key list — preserves caller order.

        Chunked at ``_SQLITE_IN_CHUNK`` to protect against
        ``SQLITE_MAX_VARIABLE_NUMBER`` blowups on large owner histories
        (e.g. ``extract_user_profile``'s direct path when ``user.md``
        doesn't exist yet — ``last_profile_ts=0`` fetches every memcell
        the owner has ever produced).

        Used by offline strategies that pull every memcell in a cluster
        (membership lives in :class:`ClusterMember` and is supplied to
        the strategy via :class:`everalgo.clustering.Cluster.members`).
        """
        if not memcell_ids:
            return []
        by_id: dict[str, Memcell] = {}
        async with session_scope(self._factory) as s:
            for start in range(0, len(memcell_ids), _SQLITE_IN_CHUNK):
                chunk = memcell_ids[start : start + _SQLITE_IN_CHUNK]
                stmt = select(Memcell).where(Memcell.memcell_id.in_(chunk))
                rows = (await s.execute(stmt)).scalars().all()
                by_id.update((r.memcell_id, r) for r in rows)
        return [by_id[mid] for mid in memcell_ids if mid in by_id]

    async def select_oldest_batch(
        self,
        memcell_ids: list[str],
        *,
        after: datetime | None,
        max_count: int,
        max_bytes: int,
    ) -> list[str]:
        """Oldest-first ids newer than ``after``, bounded by count and payload bytes.

        Reads only ``memcell_id`` / ``timestamp`` / ``length(payload_json)``, so
        an arbitrarily large candidate list never pulls payloads into memory.
        The first row is always kept, even if it alone exceeds ``max_bytes`` —
        otherwise a single oversized memcell would wedge the backlog forever
        (every run selects it, then rejects it before including anything).

        Used by ``extract_user_profile`` to work an unbounded candidate set
        off in bounded batches instead of feeding it to the LLM in one shot.
        """
        if not memcell_ids:
            return []
        candidates: list[tuple[datetime, str, int]] = []
        async with session_scope(self._factory) as s:
            for start in range(0, len(memcell_ids), _SQLITE_IN_CHUNK):
                chunk = memcell_ids[start : start + _SQLITE_IN_CHUNK]
                stmt = select(
                    Memcell.memcell_id,
                    Memcell.timestamp,
                    func.length(Memcell.payload_json),
                ).where(Memcell.memcell_id.in_(chunk))
                if after is not None:
                    stmt = stmt.where(Memcell.timestamp > after)
                rows = (await s.execute(stmt)).all()
                candidates.extend((ts, mid, size) for mid, ts, size in rows)
        candidates.sort()
        picked: list[str] = []
        total = 0
        for _, mid, size in candidates[:max_count]:
            if picked and total + size > max_bytes:
                break
            picked.append(mid)
            total += size
        return picked


memcell_repo = _MemcellRepo()
