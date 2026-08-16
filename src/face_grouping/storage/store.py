"""SQLite-backed persistence for photos, faces, clusters, and suggestions."""
from contextlib import contextmanager
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid
from face_grouping.time_utils import utcnow_naive
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from face_grouping.clustering.data_types import (
    Cluster,
    Face,
    Photo,
    PhotoProcessingStatus,
)
from face_grouping.clustering.merge_rules import Suggestion, SuggestionStatus, SuggestionType
from face_grouping.matching.assignment import AssignmentState
from face_grouping.matching.exemplars import Exemplar, ExemplarSet
from face_grouping.storage.schema import LEGACY_USER_ID, SCHEMA_STATEMENTS, SCHEMA_VERSION
from face_grouping.config import normalize_image_path

LEGACY_VERSION = "legacy_unknown"
MEMBER_STATES = (AssignmentState.CONFIRMED.value, AssignmentState.MANUAL.value)


@dataclass(frozen=True)
class PhotoProcessingClaim:
    status: str
    token: Optional[str]
    retry_after_seconds: Optional[int] = None


@dataclass(frozen=True)
class LifecycleState:
    photos_since_consolidation: int
    last_photo_completed_at: Optional[datetime]
    last_consolidated_at: Optional[datetime]
    consolidation_due: bool
    due_reason: Optional[str]
    due_since: Optional[datetime]
    consolidation_in_progress: bool
    last_consolidation_error: Optional[str]


@dataclass(frozen=True)
class ConsolidationClaim:
    status: str
    token: Optional[str]
    retry_after_seconds: Optional[int] = None


def _embedding_to_blob(embedding: np.ndarray) -> bytes:
    return np.asarray(embedding, dtype=np.float32).tobytes()


def _blob_to_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


def embedding_versions_compatible(left: str, right: str) -> bool:
    left = left or LEGACY_VERSION
    right = right or LEGACY_VERSION
    return left == LEGACY_VERSION or right == LEGACY_VERSION or left == right


class FaceGroupingStore:
    """Tenant-scoped SQLite repository.

    One store instance is bound to exactly one ``user_id``. All reads and
    writes are scoped to that tenant, while multiple tenants may safely share
    the same SQLite database file.
    """

    def __init__(self, db_path: str, user_id: str = LEGACY_USER_ID):
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        self.db_path = db_path
        self.user_id = user_id.strip()
        # A short-lived store/connection is created per tenant operation.
        # WAL allows readers to proceed while another request is writing, and
        # the timeout absorbs normal short write contention instead of failing
        # immediately with "database is locked".
        self._conn = sqlite3.connect(db_path, isolation_level=None, timeout=5.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        # WAL is persistent once enabled. Concurrent startup can momentarily
        # lock this PRAGMA, so do not make every request fail just because
        # another worker is currently establishing the same persistent mode.
        try:
            mode = str(self._conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
            if mode != "wal":
                self._conn.execute("PRAGMA journal_mode = WAL")
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower():
                raise
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._in_transaction = False
        self._initialize_schema()

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @contextmanager
    def transaction(self):
        """One real SQLite transaction; nested callers join the outer one."""
        if self._in_transaction:
            yield
            return

        self._conn.execute("BEGIN IMMEDIATE")
        self._in_transaction = True
        try:
            yield
        except Exception:
            self._conn.rollback()
            raise
        else:
            self._conn.commit()
        finally:
            self._in_transaction = False

    def _commit_if_needed(self) -> None:
        if not self._in_transaction:
            self._conn.commit()

    def _table_exists(self, table_name: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        return row is not None

    def _columns(self, table_name: str) -> set:
        if not self._table_exists(table_name):
            return set()
        return {
            row["name"]
            for row in self._conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        }

    def _initialize_schema(self) -> None:
        """Create/migrate the schema with a no-write fast path for v7.

        Production requests open short-lived tenant connections. Once the
        database is current, startup must not take an EXCLUSIVE lock on every
        request. Only a fresh/legacy database enters the serialized migration
        path.
        """
        version = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
        if (
            version == SCHEMA_VERSION
            and self._table_exists("user_lifecycle")
            and {"processing_started_at", "processing_token", "processing_attempts"}.issubset(
                self._columns("photos")
            )
        ):
            return

        self._conn.execute("PRAGMA foreign_keys = OFF")
        try:
            self._conn.execute("BEGIN EXCLUSIVE")
            version = int(self._conn.execute("PRAGMA user_version").fetchone()[0])

            legacy_tables = [
                name
                for name in (
                    "clusters",
                    "photos",
                    "faces",
                    "exemplars",
                    "suggestions",
                    "cluster_cannot_links",
                )
                if self._table_exists(name)
            ]
            needs_tenant_migration = any(
                "user_id" not in self._columns(name) for name in legacy_tables
            )
            if needs_tenant_migration:
                self._migrate_v5_to_v6_locked()
                version = 6

            if version == 6:
                self._migrate_v6_to_v7_locked()
            elif version not in (0, SCHEMA_VERSION):
                raise RuntimeError(
                    f"Unsupported face-grouping schema version {version}; "
                    "upgrade from v5/v6 or start with a fresh production database."
                )

            for statement in SCHEMA_STATEMENTS:
                self._conn.execute(statement)
            self._recompute_all_face_counts_sql()
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._conn.execute("PRAGMA foreign_keys = ON")

    def _migrate_v6_to_v7_locked(self) -> None:
        """Add processing leases and per-user lifecycle state to schema v6."""
        columns = self._columns("photos")
        if "processing_started_at" not in columns:
            self._conn.execute("ALTER TABLE photos ADD COLUMN processing_started_at TEXT")
        if "processing_token" not in columns:
            self._conn.execute("ALTER TABLE photos ADD COLUMN processing_token TEXT")
        if "processing_attempts" not in columns:
            self._conn.execute(
                "ALTER TABLE photos ADD COLUMN processing_attempts INTEGER NOT NULL DEFAULT 0"
            )
        for statement in SCHEMA_STATEMENTS:
            self._conn.execute(statement)


    def _migrate_v5_to_v6_locked(self) -> None:
        """Rebuild v5 as tenant-aware schema inside the initialization lock."""
        version = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
        if version not in (0, 5):
            raise RuntimeError(
                f"Unsupported face-grouping schema migration from version {version}; "
                "upgrade to schema v5 first or start with a fresh production database."
            )

        required = {
            "clusters",
            "photos",
            "faces",
            "exemplars",
            "suggestions",
            "cluster_cannot_links",
        }
        present = {name for name in required if self._table_exists(name)}
        if present and present != required:
            missing = sorted(required - present)
            raise RuntimeError(
                "Incomplete legacy database; cannot safely migrate to multi-user schema. "
                f"Missing tables: {missing}"
            )
        if not present:
            return

        # Rebuilding is required because v6 changes primary/unique keys.
        for table in required:
            self._conn.execute(f"ALTER TABLE {table} RENAME TO {table}__v5")

        for statement in SCHEMA_STATEMENTS:
            self._conn.execute(statement)

        uid = LEGACY_USER_ID
        self._conn.execute(
            """
            INSERT INTO clusters
                (user_id, cluster_id, face_count, is_user_confirmed,
                 has_manual_correction, created_at, last_updated_at, merged_into)
            SELECT ?, cluster_id, face_count, is_user_confirmed,
                   has_manual_correction, created_at, last_updated_at, merged_into
            FROM clusters__v5
            """,
            (uid,),
        )
        self._conn.execute(
            """
            INSERT INTO photos
                (user_id, photo_id, image_path, image_width, image_height,
                 processing_status, processed_at, embedding_model_version,
                 config_version, error_message)
            SELECT ?, photo_id, image_path, image_width, image_height,
                   processing_status, processed_at, embedding_model_version,
                   config_version, error_message
            FROM photos__v5
            """,
            (uid,),
        )
        self._conn.execute(
            """
            INSERT INTO faces
                (user_id, face_id, embedding, quality_score, yaw_ratio, cluster_id,
                 is_manually_corrected, created_at, assignment_state,
                 candidate_cluster_id, best_match_score, second_best_cluster_id,
                 second_best_score, score_margin, decision_threshold, decision_reason,
                 photo_id, face_index, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                 detection_score, embedding_model_version, config_version,
                 recognition_restricted, recognition_restriction_reason)
            SELECT ?, face_id, embedding, quality_score, yaw_ratio, cluster_id,
                   is_manually_corrected, created_at, assignment_state,
                   candidate_cluster_id, best_match_score, second_best_cluster_id,
                   second_best_score, score_margin, decision_threshold, decision_reason,
                   photo_id, face_index, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                   detection_score, embedding_model_version, config_version,
                   recognition_restricted, recognition_restriction_reason
            FROM faces__v5
            """,
            (uid,),
        )
        self._conn.execute(
            """
            INSERT INTO exemplars
                (id, user_id, cluster_id, bucket, face_id, embedding,
                 quality_score, yaw_ratio, embedding_model_version)
            SELECT id, ?, cluster_id, bucket, face_id, embedding,
                   quality_score, yaw_ratio, embedding_model_version
            FROM exemplars__v5
            """,
            (uid,),
        )
        self._conn.execute(
            """
            INSERT INTO suggestions
                (user_id, suggestion_id, suggestion_type, cluster_ids, status,
                 created_at, payload_json, evidence_json, updated_at, resolved_at)
            SELECT ?, suggestion_id, suggestion_type, cluster_ids, status,
                   created_at, payload_json, evidence_json, updated_at, resolved_at
            FROM suggestions__v5
            """,
            (uid,),
        )
        self._conn.execute(
            """
            INSERT INTO cluster_cannot_links
                (user_id, cluster_a_id, cluster_b_id, reason, created_at)
            SELECT ?, cluster_a_id, cluster_b_id, reason, created_at
            FROM cluster_cannot_links__v5
            """,
            (uid,),
        )

        # Drop in dependency order.
        for table in (
            "exemplars__v5",
            "cluster_cannot_links__v5",
            "faces__v5",
            "photos__v5",
            "suggestions__v5",
            "clusters__v5",
        ):
            self._conn.execute(f"DROP TABLE {table}")


    # ------------------------------------------------------------------
    # Photo
    # ------------------------------------------------------------------

    def _ensure_lifecycle_row(self) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO user_lifecycle(user_id) VALUES (?)",
            (self.user_id,),
        )

    def claim_photo_processing(
        self,
        *,
        photo_id: str,
        image_path: str,
        embedding_model_version: str,
        config_version: str,
        lease_seconds: int = 300,
    ) -> PhotoProcessingClaim:
        """Atomically claim one photo before expensive inference.

        Completed photos are cache hits. A fresh processing lease blocks a
        duplicate worker; stale leases may be reclaimed. The opaque token is
        checked again before writes so a stale worker cannot commit after it
        lost ownership.
        """
        if lease_seconds < 0:
            raise ValueError("lease_seconds must be >= 0")
        now = utcnow_naive()
        normalized_path = normalize_image_path(image_path)
        with self.transaction():
            by_id = self._conn.execute(
                "SELECT * FROM photos WHERE user_id=? AND photo_id=?",
                (self.user_id, photo_id),
            ).fetchone()
            by_path = self._conn.execute(
                "SELECT * FROM photos WHERE user_id=? AND image_path=?",
                (self.user_id, normalized_path),
            ).fetchone()
            if by_id and by_id["image_path"] != normalized_path:
                raise ValueError(
                    f"photo_id {photo_id!r} is already bound to a different image path"
                )
            if by_path and by_path["photo_id"] != photo_id:
                raise ValueError(
                    f"image path is already bound to photo_id {by_path['photo_id']!r}"
                )
            row = by_id or by_path
            if row and row["processing_status"] == PhotoProcessingStatus.COMPLETED.value:
                return PhotoProcessingClaim(status="completed", token=None)

            if row and row["processing_status"] == PhotoProcessingStatus.PROCESSING.value:
                started_raw = row["processing_started_at"]
                if started_raw:
                    started = datetime.fromisoformat(started_raw)
                    age = max(0.0, (now - started).total_seconds())
                    if age < lease_seconds:
                        retry_after = max(1, int(lease_seconds - age + 0.999))
                        return PhotoProcessingClaim(
                            status="in_progress",
                            token=None,
                            retry_after_seconds=retry_after,
                        )

            token = str(uuid.uuid4())
            if row:
                self._conn.execute(
                    """
                    UPDATE photos
                    SET processing_status='processing', processing_started_at=?,
                        processing_token=?, processing_attempts=processing_attempts+1,
                        processed_at=NULL, error_message=NULL,
                        embedding_model_version=?, config_version=?
                    WHERE user_id=? AND photo_id=?
                    """,
                    (
                        now.isoformat(), token, embedding_model_version, config_version,
                        self.user_id, photo_id,
                    ),
                )
            else:
                self._conn.execute(
                    """
                    INSERT INTO photos(
                        user_id, photo_id, image_path, image_width, image_height,
                        processing_status, embedding_model_version, config_version,
                        processing_started_at, processing_token, processing_attempts
                    ) VALUES (?, ?, ?, 0, 0, 'processing', ?, ?, ?, ?, 1)
                    """,
                    (
                        self.user_id, photo_id, normalized_path, embedding_model_version,
                        config_version, now.isoformat(), token,
                    ),
                )
            return PhotoProcessingClaim(status="claimed", token=token)

    def assert_photo_processing_claim(self, photo_id: str, token: str) -> None:
        row = self._conn.execute(
            "SELECT processing_status, processing_token FROM photos WHERE user_id=? AND photo_id=?",
            (self.user_id, photo_id),
        ).fetchone()
        if (
            row is None
            or row["processing_status"] != PhotoProcessingStatus.PROCESSING.value
            or row["processing_token"] != token
        ):
            raise RuntimeError(f"Processing lease lost for photo_id {photo_id!r}")

    def complete_photo_processing_claim(self, photo_id: str, token: str) -> None:
        """Finalize a successful photo and increment lifecycle exactly once."""
        row = self._conn.execute(
            "SELECT processing_status, processing_token FROM photos WHERE user_id=? AND photo_id=?",
            (self.user_id, photo_id),
        ).fetchone()
        if (
            row is None
            or row["processing_status"] != PhotoProcessingStatus.COMPLETED.value
            or row["processing_token"] != token
        ):
            raise RuntimeError(f"Processing lease lost for photo_id {photo_id!r}")
        now = utcnow_naive()
        self._conn.execute(
            """
            UPDATE photos
            SET processing_started_at=NULL, processing_token=NULL
            WHERE user_id=? AND photo_id=?
            """,
            (self.user_id, photo_id),
        )
        self._ensure_lifecycle_row()
        self._conn.execute(
            """
            UPDATE user_lifecycle
            SET photos_since_consolidation=photos_since_consolidation+1,
                last_photo_completed_at=?
            WHERE user_id=?
            """,
            (now.isoformat(), self.user_id),
        )

    def fail_photo_processing_claim(self, photo_id: str, token: str, error_message: str) -> bool:
        """Mark the current owner failed; a stale/lost owner cannot overwrite state."""
        now = utcnow_naive().isoformat()
        with self.transaction():
            result = self._conn.execute(
                """
                UPDATE photos
                SET processing_status='failed', processed_at=?, error_message=?,
                    processing_started_at=NULL, processing_token=NULL
                WHERE user_id=? AND photo_id=? AND processing_token=?
                  AND processing_status='processing'
                """,
                (now, error_message[:500], self.user_id, photo_id, token),
            )
            return result.rowcount == 1

    def get_lifecycle_state(
        self,
        *,
        photo_threshold: int = 50,
        idle_hours: float = 24.0,
    ) -> LifecycleState:
        if photo_threshold <= 0:
            raise ValueError("photo_threshold must be positive")
        if idle_hours <= 0:
            raise ValueError("idle_hours must be positive")
        self._ensure_lifecycle_row()
        self._commit_if_needed()
        row = self._conn.execute(
            "SELECT * FROM user_lifecycle WHERE user_id=?", (self.user_id,)
        ).fetchone()
        now = utcnow_naive()
        last_photo = datetime.fromisoformat(row["last_photo_completed_at"]) if row["last_photo_completed_at"] else None
        last_consolidated = datetime.fromisoformat(row["last_consolidated_at"]) if row["last_consolidated_at"] else None
        count_due_since = datetime.fromisoformat(row["count_due_since"]) if row["count_due_since"] else None
        count = int(row["photos_since_consolidation"] or 0)

        count_due = count >= photo_threshold
        idle_due_since = (last_photo + timedelta(hours=idle_hours)) if last_photo and count > 0 else None
        idle_due = bool(idle_due_since and now >= idle_due_since)
        if count_due:
            reason = "photo_count"
            due_since = count_due_since or last_photo or now
        elif idle_due:
            reason = "idle_timeout"
            due_since = idle_due_since
        else:
            reason = None
            due_since = None

        started = datetime.fromisoformat(row["consolidation_started_at"]) if row["consolidation_started_at"] else None
        return LifecycleState(
            photos_since_consolidation=count,
            last_photo_completed_at=last_photo,
            last_consolidated_at=last_consolidated,
            consolidation_due=bool(count_due or idle_due),
            due_reason=reason,
            due_since=due_since,
            consolidation_in_progress=started is not None,
            last_consolidation_error=row["last_consolidation_error"],
        )

    def mark_count_due_if_needed(self, *, photo_threshold: int) -> None:
        """Persist the first time the count threshold becomes due."""
        self._ensure_lifecycle_row()
        self._conn.execute(
            """
            UPDATE user_lifecycle
            SET count_due_since=COALESCE(count_due_since, ?)
            WHERE user_id=? AND photos_since_consolidation>=?
            """,
            (utcnow_naive().isoformat(), self.user_id, photo_threshold),
        )
        self._commit_if_needed()

    def claim_consolidation(
        self,
        *,
        force: bool,
        photo_threshold: int,
        idle_hours: float,
        lease_seconds: int = 1800,
    ) -> ConsolidationClaim:
        if lease_seconds < 0:
            raise ValueError("lease_seconds must be >= 0")
        with self.transaction():
            self._ensure_lifecycle_row()
            state = self.get_lifecycle_state(
                photo_threshold=photo_threshold, idle_hours=idle_hours
            )
            if not force and not state.consolidation_due:
                return ConsolidationClaim(status="not_due", token=None)
            row = self._conn.execute(
                "SELECT consolidation_started_at FROM user_lifecycle WHERE user_id=?",
                (self.user_id,),
            ).fetchone()
            now = utcnow_naive()
            if row["consolidation_started_at"]:
                started = datetime.fromisoformat(row["consolidation_started_at"])
                age = max(0.0, (now - started).total_seconds())
                if age < lease_seconds:
                    return ConsolidationClaim(
                        status="in_progress", token=None,
                        retry_after_seconds=max(1, int(lease_seconds - age + 0.999)),
                    )
            token = str(uuid.uuid4())
            self._conn.execute(
                """
                UPDATE user_lifecycle
                SET consolidation_started_at=?, consolidation_token=?,
                    consolidation_attempts=consolidation_attempts+1,
                    last_consolidation_error=NULL, last_consolidation_error_at=NULL
                WHERE user_id=?
                """,
                (now.isoformat(), token, self.user_id),
            )
            return ConsolidationClaim(status="claimed", token=token)

    def complete_consolidation_claim(self, token: str) -> None:
        now = utcnow_naive().isoformat()
        with self.transaction():
            row = self._conn.execute(
                "SELECT consolidation_token FROM user_lifecycle WHERE user_id=?",
                (self.user_id,),
            ).fetchone()
            if row is None or row["consolidation_token"] != token:
                raise RuntimeError("Consolidation lease lost")
            self._conn.execute(
                """
                UPDATE user_lifecycle
                SET photos_since_consolidation=0, last_consolidated_at=?,
                    count_due_since=NULL, consolidation_started_at=NULL,
                    consolidation_token=NULL, last_consolidation_error=NULL,
                    last_consolidation_error_at=NULL
                WHERE user_id=?
                """,
                (now, self.user_id),
            )

    def fail_consolidation_claim(self, token: str, error_message: str) -> bool:
        now = utcnow_naive().isoformat()
        with self.transaction():
            result = self._conn.execute(
                """
                UPDATE user_lifecycle
                SET consolidation_started_at=NULL, consolidation_token=NULL,
                    last_consolidation_error=?, last_consolidation_error_at=?
                WHERE user_id=? AND consolidation_token=?
                """,
                (error_message[:500], now, self.user_id, token),
            )
            return result.rowcount == 1

    def save_photo(self, photo: Photo) -> None:
        self._conn.execute(
            """
            INSERT INTO photos
                (user_id, photo_id, image_path, image_width, image_height,
                 processing_status, processed_at, embedding_model_version,
                 config_version, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, photo_id) DO UPDATE SET
                image_path=excluded.image_path,
                image_width=excluded.image_width,
                image_height=excluded.image_height,
                processing_status=excluded.processing_status,
                processed_at=excluded.processed_at,
                embedding_model_version=excluded.embedding_model_version,
                config_version=excluded.config_version,
                error_message=excluded.error_message
            """,
            (
                self.user_id,
                photo.photo_id,
                photo.image_path,
                photo.image_width,
                photo.image_height,
                photo.processing_status.value,
                photo.processed_at.isoformat() if photo.processed_at else None,
                photo.embedding_model_version,
                photo.config_version,
                photo.error_message,
            ),
        )
        self._commit_if_needed()

    def load_photo(self, photo_id: str) -> Optional[Photo]:
        row = self._conn.execute(
            "SELECT * FROM photos WHERE user_id = ? AND photo_id = ?",
            (self.user_id, photo_id),
        ).fetchone()
        return self._row_to_photo(row) if row else None

    def get_photo_by_path(self, image_path: str) -> Optional[Photo]:
        """Load a photo by normalized path inside this tenant only."""
        normalized_path = normalize_image_path(image_path)
        row = self._conn.execute(
            "SELECT * FROM photos WHERE user_id = ? AND image_path = ?",
            (self.user_id, normalized_path),
        ).fetchone()
        return self._row_to_photo(row) if row else None

    def load_all_photos(self) -> List[Photo]:
        rows = self._conn.execute(
            "SELECT * FROM photos WHERE user_id = ?", (self.user_id,)
        ).fetchall()
        return [self._row_to_photo(row) for row in rows]

    def load_photos_by_cluster(self, cluster_id: str) -> List[Photo]:
        rows = self._conn.execute(
            """
            SELECT DISTINCT p.*
            FROM photos p
            JOIN faces f ON f.user_id = p.user_id AND f.photo_id = p.photo_id
            WHERE p.user_id = ?
              AND f.cluster_id = ?
              AND f.assignment_state IN ('confirmed', 'manual')
            ORDER BY p.processed_at, p.photo_id
            """,
            (self.user_id, cluster_id),
        ).fetchall()
        return [self._row_to_photo(row) for row in rows]

    def _row_to_photo(self, row: sqlite3.Row) -> Photo:
        return Photo(
            photo_id=row["photo_id"],
            image_path=row["image_path"],
            image_width=row["image_width"],
            image_height=row["image_height"],
            processing_status=PhotoProcessingStatus(row["processing_status"]),
            processed_at=datetime.fromisoformat(row["processed_at"]) if row["processed_at"] else None,
            embedding_model_version=row["embedding_model_version"],
            config_version=row["config_version"],
            error_message=row["error_message"],
        )

    # ------------------------------------------------------------------
    # Face
    # ------------------------------------------------------------------

    def save_face(self, face: Face) -> None:
        self._conn.execute(
            """
            INSERT INTO faces
                (user_id, face_id, embedding, quality_score, yaw_ratio, cluster_id,
                 is_manually_corrected, created_at, assignment_state,
                 candidate_cluster_id, best_match_score,
                 second_best_cluster_id, second_best_score, score_margin,
                 decision_threshold, decision_reason, photo_id, face_index,
                 bbox_x1, bbox_y1, bbox_x2, bbox_y2, detection_score,
                 embedding_model_version, config_version, recognition_restricted,
                 recognition_restriction_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, face_id) DO UPDATE SET
                embedding=excluded.embedding,
                quality_score=excluded.quality_score,
                yaw_ratio=excluded.yaw_ratio,
                cluster_id=excluded.cluster_id,
                is_manually_corrected=excluded.is_manually_corrected,
                assignment_state=excluded.assignment_state,
                candidate_cluster_id=excluded.candidate_cluster_id,
                best_match_score=excluded.best_match_score,
                second_best_cluster_id=excluded.second_best_cluster_id,
                second_best_score=excluded.second_best_score,
                score_margin=excluded.score_margin,
                decision_threshold=excluded.decision_threshold,
                decision_reason=excluded.decision_reason,
                photo_id=excluded.photo_id,
                face_index=excluded.face_index,
                bbox_x1=excluded.bbox_x1,
                bbox_y1=excluded.bbox_y1,
                bbox_x2=excluded.bbox_x2,
                bbox_y2=excluded.bbox_y2,
                detection_score=excluded.detection_score,
                embedding_model_version=excluded.embedding_model_version,
                config_version=excluded.config_version,
                recognition_restricted=excluded.recognition_restricted,
                recognition_restriction_reason=excluded.recognition_restriction_reason
            """,
            (
                self.user_id,
                face.face_id,
                _embedding_to_blob(face.embedding),
                face.quality_score,
                face.yaw_ratio,
                face.cluster_id,
                int(face.is_manually_corrected),
                face.created_at.isoformat(),
                face.assignment_state.value,
                face.candidate_cluster_id,
                face.best_match_score,
                face.second_best_cluster_id,
                face.second_best_score,
                face.score_margin,
                face.decision_threshold,
                face.decision_reason,
                face.photo_id,
                face.face_index,
                face.bbox_x1,
                face.bbox_y1,
                face.bbox_x2,
                face.bbox_y2,
                face.detection_score,
                face.embedding_model_version,
                face.config_version,
                int(face.recognition_restricted),
                face.recognition_restriction_reason,
            ),
        )
        self._commit_if_needed()

    def load_face(self, face_id: str) -> Optional[Face]:
        row = self._conn.execute(
            "SELECT * FROM faces WHERE user_id = ? AND face_id = ?",
            (self.user_id, face_id),
        ).fetchone()
        return self._row_to_face(row) if row else None

    def load_faces_by_cluster(self, cluster_id: str) -> List[Face]:
        rows = self._conn.execute(
            "SELECT * FROM faces WHERE user_id = ? AND cluster_id = ?",
            (self.user_id, cluster_id),
        ).fetchall()
        return [self._row_to_face(row) for row in rows]

    def load_faces_by_photo(self, photo_id: str) -> List[Face]:
        rows = self._conn.execute(
            "SELECT * FROM faces WHERE user_id = ? AND photo_id = ? ORDER BY face_index, face_id",
            (self.user_id, photo_id),
        ).fetchall()
        return [self._row_to_face(row) for row in rows]

    def load_faces_by_assignment_state(self, state: AssignmentState) -> List[Face]:
        rows = self._conn.execute(
            "SELECT * FROM faces WHERE user_id = ? AND assignment_state = ?",
            (self.user_id, state.value),
        ).fetchall()
        return [self._row_to_face(row) for row in rows]

    def delete_face(self, face_id: str) -> None:
        self.delete_face_atomic(face_id)

    def delete_face_atomic(
        self,
        face_id: str,
        *,
        exemplar_quality_threshold: float = 0.7,
    ) -> None:
        with self.transaction():
            face = self.load_face(face_id)
            if face is None:
                return
            affected = face.cluster_id
            self._conn.execute("DELETE FROM exemplars WHERE user_id = ? AND face_id = ?", (self.user_id, face_id))
            self._conn.execute("DELETE FROM faces WHERE user_id = ? AND face_id = ?", (self.user_id, face_id))
            if affected:
                self._repair_cluster_after_membership_change(
                    affected,
                    exemplar_quality_threshold=exemplar_quality_threshold,
                )

    def delete_photo_atomic(
        self,
        photo_id: str,
        *,
        exemplar_quality_threshold: float = 0.7,
    ) -> None:
        with self.transaction():
            faces = self.load_faces_by_photo(photo_id)
            affected = {face.cluster_id for face in faces if face.cluster_id}
            face_ids = [face.face_id for face in faces]
            if face_ids:
                placeholders = ",".join("?" for _ in face_ids)
                self._conn.execute(
                    f"DELETE FROM exemplars WHERE user_id = ? AND face_id IN ({placeholders})",
                    [self.user_id, *face_ids],
                )
                self._conn.execute("DELETE FROM faces WHERE user_id = ? AND photo_id = ?", (self.user_id, photo_id))
            self._conn.execute("DELETE FROM photos WHERE user_id = ? AND photo_id = ?", (self.user_id, photo_id))
            for cluster_id in affected:
                self._repair_cluster_after_membership_change(
                    cluster_id,
                    exemplar_quality_threshold=exemplar_quality_threshold,
                )

    def move_face_atomic(
        self,
        face_id: str,
        to_cluster_id: str,
        *,
        exemplar_quality_threshold: float = 0.7,
    ) -> None:
        with self.transaction():
            face = self.load_face(face_id)
            if face is None:
                raise ValueError(f"No face with id {face_id}")
            target = self.load_cluster(to_cluster_id)
            if target is None:
                raise ValueError(f"No cluster with id {to_cluster_id}")
            source_id = face.cluster_id
            for exemplar in target.exemplar_set.all_exemplars():
                if not embedding_versions_compatible(
                    face.embedding_model_version, exemplar.embedding_model_version
                ):
                    raise ValueError(
                        f"Cannot move face model {face.embedding_model_version!r} "
                        f"into cluster {to_cluster_id} model {exemplar.embedding_model_version!r}"
                    )

            if source_id:
                self._conn.execute("DELETE FROM exemplars WHERE user_id = ? AND face_id = ?", (self.user_id, face.face_id))

            face.cluster_id = to_cluster_id
            face.assignment_state = AssignmentState.MANUAL
            face.is_manually_corrected = True
            face.candidate_cluster_id = to_cluster_id
            face.best_match_score = None
            face.second_best_cluster_id = None
            face.second_best_score = None
            face.score_margin = None
            face.decision_threshold = None
            face.decision_reason = "manual_correction"
            self.save_face(face)

            if (
                face.quality_score >= exemplar_quality_threshold
                and not face.recognition_restricted
            ):
                target.exemplar_set.try_add(
                    Exemplar(
                        embedding=face.embedding,
                        quality_score=face.quality_score,
                        yaw_ratio=face.yaw_ratio,
                        face_id=face.face_id,
                        embedding_model_version=face.embedding_model_version,
                    )
                )
                target.last_updated_at = utcnow_naive()
                self.save_cluster(target)

            if source_id and source_id != to_cluster_id:
                self._repair_cluster_after_membership_change(
                    source_id,
                    exemplar_quality_threshold=exemplar_quality_threshold,
                )
            self.recompute_cluster_face_count(to_cluster_id)

    def _row_to_face(self, row: sqlite3.Row) -> Face:
        return Face(
            face_id=row["face_id"],
            embedding=_blob_to_embedding(row["embedding"]),
            quality_score=row["quality_score"],
            yaw_ratio=row["yaw_ratio"],
            cluster_id=row["cluster_id"],
            is_manually_corrected=bool(row["is_manually_corrected"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            assignment_state=AssignmentState(row["assignment_state"]),
            candidate_cluster_id=row["candidate_cluster_id"],
            best_match_score=row["best_match_score"],
            second_best_cluster_id=row["second_best_cluster_id"],
            second_best_score=row["second_best_score"],
            score_margin=row["score_margin"],
            decision_threshold=row["decision_threshold"],
            decision_reason=row["decision_reason"],
            photo_id=row["photo_id"],
            face_index=row["face_index"],
            bbox_x1=row["bbox_x1"],
            bbox_y1=row["bbox_y1"],
            bbox_x2=row["bbox_x2"],
            bbox_y2=row["bbox_y2"],
            detection_score=row["detection_score"],
            embedding_model_version=row["embedding_model_version"],
            config_version=row["config_version"],
            recognition_restricted=bool(row["recognition_restricted"]),
            recognition_restriction_reason=row["recognition_restriction_reason"],
        )

    # ------------------------------------------------------------------
    # Cluster (+ exemplars)
    # ------------------------------------------------------------------

    def save_cluster(self, cluster: Cluster) -> None:
        if not self._in_transaction:
            with self.transaction():
                self.save_cluster(cluster)
            return
        self._conn.execute(
            """
            INSERT INTO clusters
                (user_id, cluster_id, face_count, is_user_confirmed,
                 has_manual_correction, created_at, last_updated_at,
                 merged_into)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, cluster_id) DO UPDATE SET
                face_count=excluded.face_count,
                is_user_confirmed=excluded.is_user_confirmed,
                has_manual_correction=excluded.has_manual_correction,
                last_updated_at=excluded.last_updated_at,
                merged_into=excluded.merged_into
            """,
            (
                self.user_id,
                cluster.cluster_id,
                cluster.face_count,
                int(cluster.is_user_confirmed),
                int(cluster.has_manual_correction),
                cluster.created_at.isoformat(),
                cluster.last_updated_at.isoformat(),
                cluster.merged_into,
            ),
        )

        self._conn.execute("DELETE FROM exemplars WHERE user_id = ? AND cluster_id = ?", (self.user_id, cluster.cluster_id))
        for bucket_name, bucket in (
            ("quality", cluster.exemplar_set.quality_bucket),
            ("pose", cluster.exemplar_set.pose_bucket),
        ):
            for exemplar in bucket:
                # Every exemplar must be backed by a real face row. This also
                # upgrades old/test clusters that stored only exemplar data.
                if exemplar.face_id is not None:
                    self._conn.execute(
                        """
                        INSERT OR IGNORE INTO faces
                            (user_id, face_id, embedding, quality_score, yaw_ratio, cluster_id,
                             is_manually_corrected, created_at, assignment_state,
                             candidate_cluster_id, decision_reason,
                             embedding_model_version, config_version)
                        VALUES (?, ?, ?, ?, ?, ?, 0, ?, 'confirmed', ?,
                                'exemplar_face_backfill', ?, 'legacy_unknown')
                        """,
                        (
                            self.user_id,
                            exemplar.face_id,
                            _embedding_to_blob(exemplar.embedding),
                            exemplar.quality_score,
                            exemplar.yaw_ratio,
                            cluster.cluster_id,
                            utcnow_naive().isoformat(),
                            cluster.cluster_id,
                            exemplar.embedding_model_version,
                        ),
                    )
                self._conn.execute(
                    """
                    INSERT INTO exemplars
                        (user_id, cluster_id, bucket, face_id, embedding,
                         quality_score, yaw_ratio, embedding_model_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.user_id,
                        cluster.cluster_id,
                        bucket_name,
                        exemplar.face_id,
                        _embedding_to_blob(exemplar.embedding),
                        exemplar.quality_score,
                        exemplar.yaw_ratio,
                        exemplar.embedding_model_version,
                    ),
                )
        self._commit_if_needed()

    def load_cluster(self, cluster_id: str) -> Optional[Cluster]:
        row = self._conn.execute(
            "SELECT * FROM clusters WHERE user_id = ? AND cluster_id = ?", (self.user_id, cluster_id)
        ).fetchone()
        return self._row_to_cluster(row) if row else None

    def load_all_clusters(self, include_merged: bool = False) -> List[Cluster]:
        if include_merged:
            rows = self._conn.execute("SELECT * FROM clusters WHERE user_id = ?", (self.user_id,)).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM clusters WHERE user_id = ? AND merged_into IS NULL",
                (self.user_id,),
            ).fetchall()
        return [self._row_to_cluster(row) for row in rows]

    def delete_cluster(self, cluster_id: str) -> None:
        with self.transaction():
            self._conn.execute("DELETE FROM exemplars WHERE user_id = ? AND cluster_id = ?", (self.user_id, cluster_id))
            self._conn.execute("DELETE FROM faces WHERE user_id = ? AND cluster_id = ?", (self.user_id, cluster_id))
            self._conn.execute("DELETE FROM clusters WHERE user_id = ? AND cluster_id = ?", (self.user_id, cluster_id))

    def recompute_cluster_face_count(self, cluster_id: str) -> int:
        row = self._conn.execute(
            """
            SELECT COUNT(*) AS n
            FROM faces
            WHERE user_id = ?
              AND cluster_id = ?
              AND assignment_state IN ('confirmed', 'manual')
            """,
            (self.user_id, cluster_id),
        ).fetchone()
        count = int(row["n"])
        self._conn.execute(
            "UPDATE clusters SET face_count = ? WHERE user_id = ? AND cluster_id = ?",
            (count, self.user_id, cluster_id),
        )
        self._commit_if_needed()
        return count

    def _recompute_all_face_counts_sql(self) -> None:
        if not self._table_exists("clusters") or not self._table_exists("faces"):
            return
        columns = self._columns("faces")
        if "assignment_state" not in columns:
            self._conn.execute(
                """
                UPDATE clusters
                SET face_count = (
                    SELECT COUNT(*) FROM faces f WHERE f.user_id = clusters.user_id
                      AND f.cluster_id = clusters.cluster_id
                )
                """
            )
            return
        self._conn.execute(
            """
            UPDATE clusters
            SET face_count = (
                SELECT COUNT(*)
                FROM faces f
                WHERE f.user_id = clusters.user_id
                  AND f.cluster_id = clusters.cluster_id
                  AND f.assignment_state IN ('confirmed', 'manual')
            )
            """
        )

    def _repair_cluster_after_membership_change(
        self,
        cluster_id: str,
        *,
        exemplar_quality_threshold: float,
    ) -> None:
        cluster = self.load_cluster(cluster_id)
        if cluster is None:
            return

        self._conn.execute(
            """
            DELETE FROM exemplars
            WHERE user_id = ? AND cluster_id = ?
              AND (
                  face_id IS NULL
                  OR NOT EXISTS (
                      SELECT 1 FROM faces f
                      WHERE f.user_id = exemplars.user_id
                        AND f.face_id = exemplars.face_id
                        AND f.cluster_id = exemplars.cluster_id
                        AND f.assignment_state IN ('confirmed', 'manual')
                  )
              )
            """,
            (self.user_id, cluster_id),
        )
        faces = [
            face for face in self.load_faces_by_cluster(cluster_id)
            if face.assignment_state in (AssignmentState.CONFIRMED, AssignmentState.MANUAL)
        ]
        if not faces:
            self._conn.execute("DELETE FROM exemplars WHERE user_id = ? AND cluster_id = ?", (self.user_id, cluster_id))
            self._conn.execute("DELETE FROM clusters WHERE user_id = ? AND cluster_id = ?", (self.user_id, cluster_id))
            return

        cluster = self.load_cluster(cluster_id)
        if cluster is not None and len(cluster.exemplar_set) == 0:
            eligible = sorted(
                (
                    face for face in faces
                    if face.quality_score >= exemplar_quality_threshold
                    and not face.recognition_restricted
                ),
                key=lambda face: face.quality_score,
                reverse=True,
            )
            if not eligible:
                self._conn.execute(
                    """
                    UPDATE faces
                    SET cluster_id = NULL,
                        assignment_state = 'unassigned',
                        candidate_cluster_id = NULL,
                        decision_reason = 'cluster_lost_last_eligible_exemplar'
                    WHERE user_id = ? AND cluster_id = ?
                    """,
                    (self.user_id, cluster_id),
                )
                self._conn.execute("DELETE FROM clusters WHERE user_id = ? AND cluster_id = ?", (self.user_id, cluster_id))
                return
            seed = eligible[0]
            cluster.exemplar_set.try_add(
                Exemplar(
                    embedding=seed.embedding,
                    quality_score=seed.quality_score,
                    yaw_ratio=seed.yaw_ratio,
                    face_id=seed.face_id,
                    embedding_model_version=seed.embedding_model_version,
                )
            )
            cluster.last_updated_at = utcnow_naive()
            self.save_cluster(cluster)

        self.recompute_cluster_face_count(cluster_id)

    def repair_empty_active_clusters(self) -> Dict[str, int]:
        rows = self._conn.execute(
            """
            SELECT c.cluster_id
            FROM clusters c
            WHERE c.user_id = ?
              AND c.merged_into IS NULL
              AND NOT EXISTS (
                  SELECT 1 FROM exemplars e
                  WHERE e.user_id = c.user_id AND e.cluster_id = c.cluster_id
              )
            """,
            (self.user_id,),
        ).fetchall()
        cluster_ids = [row["cluster_id"] for row in rows]
        repaired_faces = 0
        with self.transaction():
            for cluster_id in cluster_ids:
                count_row = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM faces WHERE user_id = ? AND cluster_id = ?", (self.user_id, cluster_id)
                ).fetchone()
                repaired_faces += count_row["n"]
                self._conn.execute(
                    """
                    UPDATE faces
                    SET cluster_id = NULL,
                        assignment_state = 'unassigned',
                        candidate_cluster_id = NULL,
                        decision_reason = 'legacy_empty_cluster_repaired'
                    WHERE user_id = ? AND cluster_id = ?
                    """,
                    (self.user_id, cluster_id),
                )
                self._conn.execute("DELETE FROM clusters WHERE user_id = ? AND cluster_id = ?", (self.user_id, cluster_id))
        return {"clusters_repaired": len(cluster_ids), "faces_unassigned": repaired_faces}

    def _row_to_cluster(self, row: sqlite3.Row) -> Cluster:
        exemplar_set = ExemplarSet()
        for exemplar_row in self._conn.execute(
            "SELECT * FROM exemplars WHERE user_id = ? AND cluster_id = ?", (self.user_id, row["cluster_id"])
        ).fetchall():
            exemplar = Exemplar(
                embedding=_blob_to_embedding(exemplar_row["embedding"]),
                quality_score=exemplar_row["quality_score"],
                yaw_ratio=exemplar_row["yaw_ratio"],
                face_id=exemplar_row["face_id"],
                embedding_model_version=exemplar_row["embedding_model_version"],
            )
            bucket = (
                exemplar_set.quality_bucket
                if exemplar_row["bucket"] == "quality"
                else exemplar_set.pose_bucket
            )
            bucket.append(exemplar)

        return Cluster(
            cluster_id=row["cluster_id"],
            exemplar_set=exemplar_set,
            face_count=row["face_count"],
            is_user_confirmed=bool(row["is_user_confirmed"]),
            has_manual_correction=bool(row["has_manual_correction"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            last_updated_at=datetime.fromisoformat(row["last_updated_at"]),
            merged_into=row["merged_into"],
        )

    # ------------------------------------------------------------------
    # Stage 4: user corrections / merge-split safety
    # ------------------------------------------------------------------

    @staticmethod
    def _cluster_pair(cluster_a_id: str, cluster_b_id: str) -> Tuple[str, str]:
        if cluster_a_id == cluster_b_id:
            raise ValueError("A cannot-link requires two different clusters")
        return tuple(sorted((cluster_a_id, cluster_b_id)))

    def add_cannot_link(self, cluster_a_id: str, cluster_b_id: str, *, reason: str) -> None:
        a, b = self._cluster_pair(cluster_a_id, cluster_b_id)
        if self.load_cluster(a) is None or self.load_cluster(b) is None:
            raise ValueError("Cannot-link clusters must exist")
        self._conn.execute(
            """
            INSERT INTO cluster_cannot_links(user_id, cluster_a_id, cluster_b_id, reason, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id, cluster_a_id, cluster_b_id) DO UPDATE SET reason=excluded.reason
            """,
            (self.user_id, a, b, reason, utcnow_naive().isoformat()),
        )
        self._commit_if_needed()

    def has_cannot_link(self, cluster_a_id: str, cluster_b_id: str) -> bool:
        if cluster_a_id == cluster_b_id:
            return False
        a, b = self._cluster_pair(cluster_a_id, cluster_b_id)
        return self._conn.execute(
            "SELECT 1 FROM cluster_cannot_links WHERE user_id=? AND cluster_a_id=? AND cluster_b_id=?",
            (self.user_id, a, b),
        ).fetchone() is not None

    def clusters_share_photo_conflict(self, cluster_a_id: str, cluster_b_id: str) -> bool:
        """True when two clusters contain distinct faces from the same photo."""
        row = self._conn.execute(
            """
            SELECT 1
            FROM faces a
            JOIN faces b ON b.user_id = a.user_id AND a.photo_id = b.photo_id AND a.face_id <> b.face_id
            WHERE a.user_id = ?
              AND a.cluster_id = ? AND b.cluster_id = ?
              AND a.photo_id IS NOT NULL
              AND a.assignment_state IN ('confirmed','manual')
              AND b.assignment_state IN ('confirmed','manual')
            LIMIT 1
            """,
            (self.user_id, cluster_a_id, cluster_b_id),
        ).fetchone()
        return row is not None

    def _assert_cluster_model_compatible(self, cluster_ids: Sequence[str]) -> None:
        versions = set()
        for cluster_id in cluster_ids:
            for face in self.load_faces_by_cluster(cluster_id):
                if face.assignment_state not in (AssignmentState.CONFIRMED, AssignmentState.MANUAL):
                    continue
                if face.embedding_model_version != LEGACY_VERSION:
                    versions.add(face.embedding_model_version)
        if len(versions) > 1:
            raise ValueError(f"Cannot combine incompatible embedding models: {sorted(versions)}")

    def _rebuild_cluster_exemplars(
        self,
        cluster_id: str,
        *,
        exemplar_quality_threshold: float,
        allow_low_quality_seed: bool = False,
    ) -> None:
        cluster = self.load_cluster(cluster_id)
        if cluster is None:
            return
        faces = [
            face for face in self.load_faces_by_cluster(cluster_id)
            if face.assignment_state in (AssignmentState.CONFIRMED, AssignmentState.MANUAL)
        ]
        if not faces:
            self._conn.execute("DELETE FROM exemplars WHERE user_id=? AND cluster_id=?", (self.user_id, cluster_id))
            self._conn.execute("DELETE FROM clusters WHERE user_id=? AND cluster_id=?", (self.user_id, cluster_id))
            return
        eligible = [
            f for f in faces
            if f.quality_score >= exemplar_quality_threshold and not f.recognition_restricted
        ]
        if not eligible and allow_low_quality_seed:
            eligible = [max(faces, key=lambda f: f.quality_score)]
        if not eligible:
            raise ValueError(f"Cluster {cluster_id} would have no exemplar-eligible face")

        exemplar_set = ExemplarSet()
        for face in sorted(eligible, key=lambda f: f.quality_score, reverse=True):
            exemplar_set.try_add(Exemplar(
                embedding=face.embedding,
                quality_score=face.quality_score,
                yaw_ratio=face.yaw_ratio,
                face_id=face.face_id,
                embedding_model_version=face.embedding_model_version,
            ))
        cluster.exemplar_set = exemplar_set
        cluster.face_count = len(faces)
        cluster.last_updated_at = utcnow_naive()
        self.save_cluster(cluster)
        self.recompute_cluster_face_count(cluster_id)

    def _invalidate_suggestions_for_clusters(
        self,
        cluster_ids: Iterable[str],
        *,
        except_suggestion_id: Optional[str] = None,
    ) -> None:
        targets = set(cluster_ids)
        rows = self._conn.execute(
            "SELECT suggestion_id, cluster_ids, status FROM suggestions WHERE user_id=? AND status='pending'"
            , (self.user_id,)
        ).fetchall()
        now = utcnow_naive().isoformat()
        for row in rows:
            if row["suggestion_id"] == except_suggestion_id:
                continue
            if targets.intersection(json.loads(row["cluster_ids"])):
                self._conn.execute(
                    "UPDATE suggestions SET status='stale', updated_at=?, resolved_at=? WHERE user_id=? AND suggestion_id=?",
                    (now, now, self.user_id, row["suggestion_id"]),
                )

    def execute_merge_atomic(
        self,
        cluster_a_id: str,
        cluster_b_id: str,
        *,
        exemplar_quality_threshold: float = 0.7,
        suggestion_id: Optional[str] = None,
    ) -> str:
        """Merge two user-approved clusters as one atomic operation."""
        with self.transaction():
            a = self.load_cluster(cluster_a_id)
            b = self.load_cluster(cluster_b_id)
            if a is None or b is None or a.merged_into or b.merged_into:
                raise ValueError("STALE: merge clusters are no longer active")
            if self.has_cannot_link(a.cluster_id, b.cluster_id):
                raise ValueError("Merge blocked by a stored cannot-link")
            if self.clusters_share_photo_conflict(a.cluster_id, b.cluster_id):
                raise ValueError("Merge blocked by same-photo cannot-link")
            self._assert_cluster_model_compatible([a.cluster_id, b.cluster_id])

            # Confirmed cluster, then larger cluster, then older/id for stable tie-breaking.
            survivor, loser = sorted(
                (a, b),
                key=lambda c: (-int(c.is_user_confirmed), -c.face_count, c.created_at, c.cluster_id),
            )

            self._conn.execute(
                """
                UPDATE faces
                SET cluster_id=?,
                    candidate_cluster_id=CASE WHEN candidate_cluster_id=? THEN ? ELSE candidate_cluster_id END
                WHERE user_id=? AND cluster_id=?
                """,
                (survivor.cluster_id, loser.cluster_id, survivor.cluster_id, self.user_id, loser.cluster_id),
            )
            survivor.has_manual_correction = survivor.has_manual_correction or loser.has_manual_correction
            survivor.is_user_confirmed = survivor.is_user_confirmed or loser.is_user_confirmed
            survivor.last_updated_at = utcnow_naive()
            loser.merged_into = survivor.cluster_id
            loser.face_count = 0
            loser.last_updated_at = survivor.last_updated_at
            # Clear stale exemplar objects before persisting cluster metadata;
            # the survivor is rebuilt only from its new real membership.
            survivor.exemplar_set = ExemplarSet()
            loser.exemplar_set = ExemplarSet()
            self._conn.execute("DELETE FROM exemplars WHERE user_id=? AND cluster_id IN (?,?)", (self.user_id, survivor.cluster_id, loser.cluster_id))
            self.save_cluster(survivor)
            self.save_cluster(loser)
            self._rebuild_cluster_exemplars(
                survivor.cluster_id,
                exemplar_quality_threshold=exemplar_quality_threshold,
            )

            # Inherit every cannot-link of the retired cluster.
            links = self._conn.execute(
                """
                SELECT cluster_a_id, cluster_b_id, reason FROM cluster_cannot_links
                WHERE user_id=? AND (cluster_a_id=? OR cluster_b_id=?)
                """,
                (self.user_id, loser.cluster_id, loser.cluster_id),
            ).fetchall()
            self._conn.execute(
                "DELETE FROM cluster_cannot_links WHERE user_id=? AND (cluster_a_id=? OR cluster_b_id=?)",
                (self.user_id, loser.cluster_id, loser.cluster_id),
            )
            for row in links:
                other = row["cluster_b_id"] if row["cluster_a_id"] == loser.cluster_id else row["cluster_a_id"]
                if other != survivor.cluster_id and self.load_cluster(other) is not None:
                    self.add_cannot_link(survivor.cluster_id, other, reason=row["reason"])

            self._invalidate_suggestions_for_clusters(
                {survivor.cluster_id, loser.cluster_id}, except_suggestion_id=suggestion_id
            )
            return survivor.cluster_id

    def execute_split_atomic(
        self,
        source_cluster_id: str,
        groups: Sequence[Sequence[str]],
        *,
        exemplar_quality_threshold: float = 0.7,
        suggestion_id: Optional[str] = None,
        mark_manual_correction: bool = True,
        cannot_link_reason: str = "accepted_split",
    ) -> List[str]:
        """Execute an exact two-way split as one atomic operation.

        User-approved splits keep the historical default of marking both
        clusters as manually corrected. High-confidence automatic splits pass
        ``mark_manual_correction=False`` but still receive a cannot-link so a
        later merge pass cannot immediately undo the structural correction.
        """
        import uuid

        with self.transaction():
            source = self.load_cluster(source_cluster_id)
            if source is None or source.merged_into is not None:
                raise ValueError("STALE: split source cluster is no longer active")
            normalized = [sorted(set(group)) for group in groups]
            if len(normalized) != 2 or any(len(g) < 2 for g in normalized):
                raise ValueError("Split requires exactly two groups with at least two faces each")
            flat = [face_id for group in normalized for face_id in group]
            if len(flat) != len(set(flat)):
                raise ValueError("Split payload contains duplicate face IDs")

            current_faces = [
                face for face in self.load_faces_by_cluster(source_cluster_id)
                if face.assignment_state in (AssignmentState.CONFIRMED, AssignmentState.MANUAL)
            ]
            restricted_faces = [f for f in current_faces if f.recognition_restricted]
            authoritative_faces = [f for f in current_faces if not f.recognition_restricted]
            current_ids = {f.face_id for f in authoritative_faces}
            if set(flat) != current_ids:
                raise ValueError("STALE: split authoritative membership changed after suggestion creation")
            by_id = {f.face_id: f for f in authoritative_faces}
            if any(f.is_manually_corrected for f in current_faces):
                raise ValueError("Split blocked because the cluster contains manual corrections")
            for group in normalized:
                if not any(
                    by_id[fid].quality_score >= exemplar_quality_threshold
                    and not by_id[fid].recognition_restricted
                    for fid in group
                ):
                    raise ValueError("Split group has no exemplar-eligible face")

            retained = sorted(normalized, key=lambda g: (-len(g), g))[0]
            new_group = next(group for group in normalized if group != retained)
            new_cluster_id = str(uuid.uuid4())
            new_cluster = Cluster(
                cluster_id=new_cluster_id,
                exemplar_set=ExemplarSet(),
                face_count=0,
                has_manual_correction=bool(mark_manual_correction),
            )
            if mark_manual_correction:
                source.has_manual_correction = True
            source.last_updated_at = utcnow_naive()
            self.save_cluster(new_cluster)
            self.save_cluster(source)

            # Restricted-pose members are deliberately non-authoritative. A
            # structural split invalidates their prior recognition context, so
            # detach them and let the next restricted-pose recovery pass decide
            # which resulting mature identity, if any, they belong to.
            for restricted in restricted_faces:
                restricted.cluster_id = None
                restricted.assignment_state = AssignmentState.UNASSIGNED
                restricted.candidate_cluster_id = None
                restricted.best_match_score = None
                restricted.second_best_cluster_id = None
                restricted.second_best_score = None
                restricted.score_margin = None
                restricted.decision_threshold = None
                restricted.decision_reason = "restricted_pose_recheck_after_split"
                self.save_face(restricted)

            placeholders = ",".join("?" for _ in new_group)
            self._conn.execute(
                f"UPDATE faces SET cluster_id=?, candidate_cluster_id=? WHERE user_id=? AND face_id IN ({placeholders})",
                [new_cluster_id, new_cluster_id, self.user_id, *new_group],
            )
            self._conn.execute("DELETE FROM exemplars WHERE user_id=? AND cluster_id IN (?,?)", (self.user_id, source_cluster_id, new_cluster_id))
            self._rebuild_cluster_exemplars(source_cluster_id, exemplar_quality_threshold=exemplar_quality_threshold)
            self._rebuild_cluster_exemplars(new_cluster_id, exemplar_quality_threshold=exemplar_quality_threshold)
            self.add_cannot_link(source_cluster_id, new_cluster_id, reason=cannot_link_reason)
            self._invalidate_suggestions_for_clusters(
                {source_cluster_id, new_cluster_id}, except_suggestion_id=suggestion_id
            )
            return [source_cluster_id, new_cluster_id]

    def manual_move_faces_atomic(
        self,
        face_ids: Sequence[str],
        to_cluster_id: str,
        *,
        from_cluster_id: Optional[str] = None,
        exemplar_quality_threshold: float = 0.7,
    ) -> int:
        """Apply an explicit user move. No automatic caller should use this."""
        unique_ids = list(dict.fromkeys(face_ids))
        if not unique_ids:
            return 0
        with self.transaction():
            target = self.load_cluster(to_cluster_id)
            if target is None or target.merged_into is not None:
                raise ValueError(f"No active cluster with id {to_cluster_id}")
            faces = []
            for face_id in unique_ids:
                face = self.load_face(face_id)
                if face is None:
                    raise ValueError(f"No face with id {face_id}")
                if face.cluster_id is None:
                    raise ValueError(f"Face {face_id} is not currently assigned")
                if from_cluster_id is not None and face.cluster_id != from_cluster_id:
                    raise ValueError(f"Face {face_id} is no longer in source cluster {from_cluster_id}")
                faces.append(face)
            source_ids = {f.cluster_id for f in faces}
            if to_cluster_id in source_ids and len(source_ids) == 1:
                return 0
            self._assert_cluster_model_compatible(list(source_ids | {to_cluster_id}))

            moving_ids = {f.face_id for f in faces}
            moving_photo_ids = [f.photo_id for f in faces if f.photo_id is not None]
            if len(moving_photo_ids) != len(set(moving_photo_ids)):
                raise ValueError("Manual move would place two faces from the same photo in one cluster")
            for face in faces:
                if face.photo_id is None:
                    continue
                conflict = self._conn.execute(
                    """
                    SELECT face_id FROM faces
                    WHERE user_id=? AND photo_id=? AND cluster_id=?
                      AND assignment_state IN ('confirmed','manual')
                    LIMIT 1
                    """,
                    (self.user_id, face.photo_id, to_cluster_id),
                ).fetchone()
                if conflict is not None and conflict["face_id"] not in moving_ids:
                    raise ValueError(
                        f"Manual move blocked by same-photo cannot-link for photo {face.photo_id}"
                    )

            self._conn.executemany("DELETE FROM exemplars WHERE user_id=? AND face_id=?", [(self.user_id, fid) for fid in moving_ids])
            now = utcnow_naive().isoformat()
            for face in faces:
                self._conn.execute(
                    """
                    UPDATE faces
                    SET cluster_id=?, assignment_state='manual', is_manually_corrected=1,
                        candidate_cluster_id=?, best_match_score=NULL,
                        second_best_cluster_id=NULL, second_best_score=NULL,
                        score_margin=NULL, decision_threshold=NULL,
                        decision_reason='manual_correction'
                    WHERE user_id=? AND face_id=?
                    """,
                    (to_cluster_id, to_cluster_id, self.user_id, face.face_id),
                )
            for cluster_id in source_ids | {to_cluster_id}:
                self._conn.execute(
                    "UPDATE clusters SET has_manual_correction=1, last_updated_at=? WHERE user_id=? AND cluster_id=?",
                    (now, self.user_id, cluster_id),
                )

            for source_id in source_ids - {to_cluster_id}:
                if self.load_faces_by_cluster(source_id):
                    self._rebuild_cluster_exemplars(
                        source_id,
                        exemplar_quality_threshold=exemplar_quality_threshold,
                        allow_low_quality_seed=True,
                    )
                else:
                    self._conn.execute("DELETE FROM clusters WHERE user_id=? AND cluster_id=?", (self.user_id, source_id))
            self._rebuild_cluster_exemplars(
                to_cluster_id,
                exemplar_quality_threshold=exemplar_quality_threshold,
                allow_low_quality_seed=True,
            )
            self._invalidate_suggestions_for_clusters(source_ids | {to_cluster_id})
            return len(faces)

    def manual_move_photos_atomic(
        self,
        photo_ids: Sequence[str],
        from_cluster_id: str,
        to_cluster_id: str,
        *,
        exemplar_quality_threshold: float = 0.7,
    ) -> int:
        face_ids = []
        for photo_id in dict.fromkeys(photo_ids):
            rows = self._conn.execute(
                """
                SELECT face_id FROM faces
                WHERE user_id=? AND photo_id=? AND cluster_id=?
                  AND assignment_state IN ('confirmed','manual')
                """,
                (self.user_id, photo_id, from_cluster_id),
            ).fetchall()
            if len(rows) != 1:
                raise ValueError(
                    f"Expected exactly one face for photo {photo_id} in cluster {from_cluster_id}; found {len(rows)}"
                )
            face_ids.append(rows[0]["face_id"])
        return self.manual_move_faces_atomic(
            face_ids,
            to_cluster_id,
            from_cluster_id=from_cluster_id,
            exemplar_quality_threshold=exemplar_quality_threshold,
        )

    # ------------------------------------------------------------------
    # Suggestions
    # ------------------------------------------------------------------

    def remove_legacy_pending_suggestions(self) -> int:
        rows = self._conn.execute(
            "SELECT suggestion_id FROM suggestions WHERE user_id=? AND status = 'pending'"
            , (self.user_id,)
        ).fetchall()
        legacy_ids = [
            row["suggestion_id"]
            for row in rows
            if re.fullmatch(r"(?:merge|split)_\d+", row["suggestion_id"])
        ]
        if legacy_ids:
            self._conn.executemany(
                "DELETE FROM suggestions WHERE user_id=? AND suggestion_id = ?",
                [(self.user_id, suggestion_id) for suggestion_id in legacy_ids],
            )
            self._commit_if_needed()
        return len(legacy_ids)

    def save_suggestion(self, suggestion: Suggestion) -> bool:
        """Insert a suggestion once. Rejected/uncertain/stale IDs never resurface."""
        cursor = self._conn.execute(
            """
            INSERT INTO suggestions
                (user_id, suggestion_id, suggestion_type, cluster_ids, status, created_at,
                 payload_json, evidence_json, updated_at, resolved_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, suggestion_id) DO NOTHING
            """,
            (
                self.user_id,
                suggestion.suggestion_id,
                suggestion.suggestion_type.value,
                json.dumps(suggestion.cluster_ids),
                suggestion.status.value,
                suggestion.created_at.isoformat(),
                json.dumps(suggestion.payload, sort_keys=True),
                json.dumps(suggestion.evidence, sort_keys=True),
                suggestion.updated_at.isoformat(),
                suggestion.resolved_at.isoformat() if suggestion.resolved_at else None,
            ),
        )
        self._commit_if_needed()
        return cursor.rowcount > 0

    def update_suggestion_status(self, suggestion_id: str, status: SuggestionStatus) -> None:
        now = utcnow_naive()
        resolved_at = now.isoformat() if status != SuggestionStatus.PENDING else None
        cursor = self._conn.execute(
            "UPDATE suggestions SET status=?, updated_at=?, resolved_at=? WHERE user_id=? AND suggestion_id=?",
            (status.value, now.isoformat(), resolved_at, self.user_id, suggestion_id),
        )
        if cursor.rowcount == 0:
            raise ValueError(f"Unknown suggestion_id: {suggestion_id}")
        self._commit_if_needed()

    def load_suggestion(self, suggestion_id: str) -> Optional[Suggestion]:
        row = self._conn.execute(
            "SELECT * FROM suggestions WHERE user_id=? AND suggestion_id=?", (self.user_id, suggestion_id)
        ).fetchone()
        return self._row_to_suggestion(row) if row else None

    def load_pending_suggestions(self) -> List[Suggestion]:
        rows = self._conn.execute(
            "SELECT * FROM suggestions WHERE user_id=? AND status = 'pending' ORDER BY created_at, suggestion_id"
            , (self.user_id,)
        ).fetchall()
        return [self._row_to_suggestion(row) for row in rows]

    def _row_to_suggestion(self, row: sqlite3.Row) -> Suggestion:
        columns = set(row.keys())
        created_at = datetime.fromisoformat(row["created_at"])
        updated_at = (
            datetime.fromisoformat(row["updated_at"])
            if "updated_at" in columns and row["updated_at"]
            else created_at
        )
        return Suggestion(
            suggestion_id=row["suggestion_id"],
            suggestion_type=SuggestionType(row["suggestion_type"]),
            cluster_ids=json.loads(row["cluster_ids"]),
            status=SuggestionStatus(row["status"]),
            created_at=created_at,
            payload=json.loads(row["payload_json"] or "{}") if "payload_json" in columns else {},
            evidence=json.loads(row["evidence_json"] or "{}") if "evidence_json" in columns else {},
            updated_at=updated_at,
            resolved_at=(
                datetime.fromisoformat(row["resolved_at"])
                if "resolved_at" in columns and row["resolved_at"] else None
            ),
        )

    def delete_user_data(self) -> Dict[str, int]:
        """Delete every AI-owned row for the bound tenant atomically."""
        tables = (
            "suggestions",
            "cluster_cannot_links",
            "exemplars",
            "faces",
            "photos",
            "clusters",
            "user_lifecycle",
        )
        counts: Dict[str, int] = {}
        with self.transaction():
            for table in tables:
                row = self._conn.execute(
                    f"SELECT COUNT(*) AS n FROM {table} WHERE user_id=?",
                    (self.user_id,),
                ).fetchone()
                counts[table] = int(row["n"])
            for table in tables:
                self._conn.execute(
                    f"DELETE FROM {table} WHERE user_id=?",
                    (self.user_id,),
                )
        return counts

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def validate_integrity(self, expected_embedding_dim: Optional[int] = None) -> List[str]:
        """Validate only the currently bound tenant's repository state."""
        errors: List[str] = []

        for row in self._conn.execute(
            """
            SELECT c.cluster_id, c.face_count,
                   SUM(CASE WHEN f.assignment_state IN ('confirmed','manual') THEN 1 ELSE 0 END) AS actual
            FROM clusters c
            LEFT JOIN faces f
              ON f.user_id = c.user_id AND f.cluster_id = c.cluster_id
            WHERE c.user_id = ?
            GROUP BY c.user_id, c.cluster_id
            """,
            (self.user_id,),
        ).fetchall():
            actual = int(row["actual"] or 0)
            if row["face_count"] != actual:
                errors.append(
                    f"cluster {row['cluster_id']} face_count={row['face_count']} actual={actual}"
                )

        same_photo_conflicts = self._conn.execute(
            """
            SELECT cluster_id, photo_id, COUNT(*) AS n
            FROM faces
            WHERE user_id = ?
              AND cluster_id IS NOT NULL
              AND photo_id IS NOT NULL
              AND assignment_state IN ('confirmed','manual')
            GROUP BY cluster_id, photo_id
            HAVING COUNT(*) > 1
            """,
            (self.user_id,),
        ).fetchall()
        for row in same_photo_conflicts:
            errors.append(
                f"cluster {row['cluster_id']} contains {row['n']} faces from photo {row['photo_id']}"
            )

        invalid_members = self._conn.execute(
            """
            SELECT face_id, assignment_state, cluster_id FROM faces
            WHERE user_id = ?
              AND (
                  (assignment_state IN ('confirmed','manual') AND cluster_id IS NULL)
                  OR (assignment_state IN ('ambiguous','unassigned') AND cluster_id IS NOT NULL)
              )
            """,
            (self.user_id,),
        ).fetchall()
        for row in invalid_members:
            errors.append(
                f"face {row['face_id']} has inconsistent state={row['assignment_state']} cluster={row['cluster_id']}"
            )

        for row in self._conn.execute(
            """
            SELECT e.id, e.face_id, e.cluster_id
            FROM exemplars e
            LEFT JOIN faces f
              ON f.user_id = e.user_id AND f.face_id = e.face_id
            WHERE e.user_id = ?
              AND (
                  e.face_id IS NULL
                  OR f.face_id IS NULL
                  OR f.cluster_id != e.cluster_id
                  OR f.assignment_state NOT IN ('confirmed','manual')
                  OR COALESCE(f.recognition_restricted, 0) = 1
              )
            """,
            (self.user_id,),
        ).fetchall():
            errors.append(
                f"orphan/invalid exemplar id={row['id']} face={row['face_id']} cluster={row['cluster_id']}"
            )

        for row in self._conn.execute(
            """
            SELECT c.cluster_id FROM clusters c
            WHERE c.user_id = ?
              AND c.merged_into IS NULL
              AND NOT EXISTS (
                  SELECT 1 FROM exemplars e
                  WHERE e.user_id = c.user_id AND e.cluster_id = c.cluster_id
              )
            """,
            (self.user_id,),
        ).fetchall():
            errors.append(f"active cluster {row['cluster_id']} has no exemplar")

        rows = self._conn.execute(
            """
            SELECT face_id, embedding, embedding_model_version, cluster_id
            FROM faces
            WHERE user_id = ?
            """,
            (self.user_id,),
        ).fetchall()
        versions_by_cluster: Dict[str, set] = {}
        for row in rows:
            dim = len(_blob_to_embedding(row["embedding"]))
            if expected_embedding_dim is not None and dim != expected_embedding_dim:
                errors.append(f"face {row['face_id']} embedding_dim={dim}")
            version = row["embedding_model_version"] or LEGACY_VERSION
            if row["cluster_id"] and version != LEGACY_VERSION:
                versions_by_cluster.setdefault(row["cluster_id"], set()).add(version)
        for cluster_id, versions in versions_by_cluster.items():
            if len(versions) > 1:
                errors.append(f"cluster {cluster_id} mixes embedding versions {sorted(versions)}")

        return errors
