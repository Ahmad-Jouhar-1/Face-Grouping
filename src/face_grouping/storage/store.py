"""SQLite-backed persistence for photos, faces, clusters, and suggestions."""
from contextlib import contextmanager
import json
import re
import sqlite3
from datetime import datetime
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
from face_grouping.storage.schema import SCHEMA_STATEMENTS, SCHEMA_VERSION
from face_grouping.config import normalize_image_path

LEGACY_VERSION = "legacy_unknown"
MEMBER_STATES = (AssignmentState.CONFIRMED.value, AssignmentState.MANUAL.value)


def _embedding_to_blob(embedding: np.ndarray) -> bytes:
    return np.asarray(embedding, dtype=np.float32).tobytes()


def _blob_to_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


def embedding_versions_compatible(left: str, right: str) -> bool:
    left = left or LEGACY_VERSION
    right = right or LEGACY_VERSION
    return left == LEGACY_VERSION or right == LEGACY_VERSION or left == right


class FaceGroupingStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
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
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            if self._table_exists("faces"):
                self._migrate_faces_table()
            if self._table_exists("exemplars"):
                self._migrate_exemplars_table()
            if self._table_exists("suggestions"):
                self._migrate_suggestions_table()

            for statement in SCHEMA_STATEMENTS:
                self._conn.execute(statement)

            self._recompute_all_face_counts_sql()
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def _migrate_faces_table(self) -> None:
        """Idempotently upgrade old face rows through Stage 3."""
        existing = self._columns("faces")
        additions = {
            # Stage 1
            "assignment_state": "TEXT NOT NULL DEFAULT 'confirmed'",
            "candidate_cluster_id": "TEXT",
            "best_match_score": "REAL",
            "second_best_cluster_id": "TEXT",
            "second_best_score": "REAL",
            "score_margin": "REAL",
            "decision_threshold": "REAL",
            "decision_reason": "TEXT",
            # Stage 3
            "photo_id": "TEXT REFERENCES photos(photo_id)",
            "face_index": "INTEGER",
            "bbox_x1": "REAL",
            "bbox_y1": "REAL",
            "bbox_x2": "REAL",
            "bbox_y2": "REAL",
            "detection_score": "REAL",
            "embedding_model_version": "TEXT NOT NULL DEFAULT 'legacy_unknown'",
            "config_version": "TEXT NOT NULL DEFAULT 'legacy_unknown'",
            # Stage 5: pose-only recognition-restricted faces.
            "recognition_restricted": "INTEGER NOT NULL DEFAULT 0",
            "recognition_restriction_reason": "TEXT",
        }
        assignment_state_was_added = "assignment_state" not in existing
        for column, definition in additions.items():
            if column not in existing:
                self._conn.execute(f"ALTER TABLE faces ADD COLUMN {column} {definition}")

        if assignment_state_was_added:
            self._conn.execute(
                """
                UPDATE faces
                SET assignment_state = CASE
                    WHEN cluster_id IS NULL THEN 'unassigned'
                    ELSE 'confirmed'
                END
                """
            )
        else:
            self._conn.execute(
                """
                UPDATE faces
                SET assignment_state = CASE
                    WHEN cluster_id IS NULL THEN 'unassigned'
                    ELSE 'confirmed'
                END
                WHERE assignment_state IS NULL
                   OR assignment_state NOT IN ('confirmed', 'ambiguous', 'unassigned', 'manual')
                """
            )
        self._conn.execute(
            "UPDATE faces SET decision_reason = COALESCE(decision_reason, 'legacy_assignment')"
        )
        self._conn.execute(
            "UPDATE faces SET embedding_model_version = COALESCE(embedding_model_version, ?), config_version = COALESCE(config_version, ?)",
            (LEGACY_VERSION, LEGACY_VERSION),
        )

    def _migrate_exemplars_table(self) -> None:
        existing = self._columns("exemplars")
        if "embedding_model_version" not in existing:
            self._conn.execute(
                "ALTER TABLE exemplars ADD COLUMN embedding_model_version TEXT NOT NULL DEFAULT 'legacy_unknown'"
            )

    def _migrate_suggestions_table(self) -> None:
        """Idempotently upgrade Stage-1/2 suggestion rows through Stage 4."""
        existing = self._columns("suggestions")
        additions = {
            "payload_json": "TEXT NOT NULL DEFAULT '{}'",
            "evidence_json": "TEXT NOT NULL DEFAULT '{}'",
            "updated_at": "TEXT",
            "resolved_at": "TEXT",
        }
        for column, definition in additions.items():
            if column not in existing:
                self._conn.execute(f"ALTER TABLE suggestions ADD COLUMN {column} {definition}")
        self._conn.execute(
            "UPDATE suggestions SET updated_at = COALESCE(updated_at, created_at)"
        )

    # ------------------------------------------------------------------
    # Photo
    # ------------------------------------------------------------------

    def save_photo(self, photo: Photo) -> None:
        self._conn.execute(
            """
            INSERT INTO photos
                (photo_id, image_path, image_width, image_height,
                 processing_status, processed_at, embedding_model_version,
                 config_version, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(photo_id) DO UPDATE SET
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
            "SELECT * FROM photos WHERE photo_id = ?", (photo_id,)
        ).fetchone()
        return self._row_to_photo(row) if row else None

    def get_photo_by_path(self, image_path: str) -> Optional[Photo]:
        """Load a photo using any equivalent path spelling.

        Photo ingestion stores normalized paths. Normalize lookups here too so
        callers do not need to remember Windows-specific ``normcase`` rules.
        """
        normalized_path = normalize_image_path(image_path)
        row = self._conn.execute(
            "SELECT * FROM photos WHERE image_path = ?", (normalized_path,)
        ).fetchone()
        return self._row_to_photo(row) if row else None

    def load_all_photos(self) -> List[Photo]:
        return [self._row_to_photo(row) for row in self._conn.execute("SELECT * FROM photos").fetchall()]

    def load_photos_by_cluster(self, cluster_id: str) -> List[Photo]:
        rows = self._conn.execute(
            """
            SELECT DISTINCT p.*
            FROM photos p
            JOIN faces f ON f.photo_id = p.photo_id
            WHERE f.cluster_id = ?
              AND f.assignment_state IN ('confirmed', 'manual')
            ORDER BY p.processed_at, p.photo_id
            """,
            (cluster_id,),
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
                (face_id, embedding, quality_score, yaw_ratio, cluster_id,
                 is_manually_corrected, created_at, assignment_state,
                 candidate_cluster_id, best_match_score,
                 second_best_cluster_id, second_best_score, score_margin,
                 decision_threshold, decision_reason, photo_id, face_index,
                 bbox_x1, bbox_y1, bbox_x2, bbox_y2, detection_score,
                 embedding_model_version, config_version, recognition_restricted,
                 recognition_restriction_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(face_id) DO UPDATE SET
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
            "SELECT * FROM faces WHERE face_id = ?", (face_id,)
        ).fetchone()
        return self._row_to_face(row) if row else None

    def load_faces_by_cluster(self, cluster_id: str) -> List[Face]:
        rows = self._conn.execute(
            "SELECT * FROM faces WHERE cluster_id = ?", (cluster_id,)
        ).fetchall()
        return [self._row_to_face(row) for row in rows]

    def load_faces_by_photo(self, photo_id: str) -> List[Face]:
        rows = self._conn.execute(
            "SELECT * FROM faces WHERE photo_id = ? ORDER BY face_index, face_id",
            (photo_id,),
        ).fetchall()
        return [self._row_to_face(row) for row in rows]

    def load_faces_by_assignment_state(self, state: AssignmentState) -> List[Face]:
        rows = self._conn.execute(
            "SELECT * FROM faces WHERE assignment_state = ?", (state.value,)
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
            self._conn.execute("DELETE FROM exemplars WHERE face_id = ?", (face_id,))
            self._conn.execute("DELETE FROM faces WHERE face_id = ?", (face_id,))
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
                    f"DELETE FROM exemplars WHERE face_id IN ({placeholders})",
                    face_ids,
                )
                self._conn.execute("DELETE FROM faces WHERE photo_id = ?", (photo_id,))
            self._conn.execute("DELETE FROM photos WHERE photo_id = ?", (photo_id,))
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
                self._conn.execute("DELETE FROM exemplars WHERE face_id = ?", (face.face_id,))

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
                target.last_updated_at = datetime.utcnow()
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
                (cluster_id, face_count, is_user_confirmed,
                 has_manual_correction, created_at, last_updated_at,
                 merged_into)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(cluster_id) DO UPDATE SET
                face_count=excluded.face_count,
                is_user_confirmed=excluded.is_user_confirmed,
                has_manual_correction=excluded.has_manual_correction,
                last_updated_at=excluded.last_updated_at,
                merged_into=excluded.merged_into
            """,
            (
                cluster.cluster_id,
                cluster.face_count,
                int(cluster.is_user_confirmed),
                int(cluster.has_manual_correction),
                cluster.created_at.isoformat(),
                cluster.last_updated_at.isoformat(),
                cluster.merged_into,
            ),
        )

        self._conn.execute("DELETE FROM exemplars WHERE cluster_id = ?", (cluster.cluster_id,))
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
                            (face_id, embedding, quality_score, yaw_ratio, cluster_id,
                             is_manually_corrected, created_at, assignment_state,
                             candidate_cluster_id, decision_reason,
                             embedding_model_version, config_version)
                        VALUES (?, ?, ?, ?, ?, 0, ?, 'confirmed', ?,
                                'exemplar_face_backfill', ?, 'legacy_unknown')
                        """,
                        (
                            exemplar.face_id,
                            _embedding_to_blob(exemplar.embedding),
                            exemplar.quality_score,
                            exemplar.yaw_ratio,
                            cluster.cluster_id,
                            datetime.utcnow().isoformat(),
                            cluster.cluster_id,
                            exemplar.embedding_model_version,
                        ),
                    )
                self._conn.execute(
                    """
                    INSERT INTO exemplars
                        (cluster_id, bucket, face_id, embedding,
                         quality_score, yaw_ratio, embedding_model_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
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
            "SELECT * FROM clusters WHERE cluster_id = ?", (cluster_id,)
        ).fetchone()
        return self._row_to_cluster(row) if row else None

    def load_all_clusters(self, include_merged: bool = False) -> List[Cluster]:
        if include_merged:
            rows = self._conn.execute("SELECT * FROM clusters").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM clusters WHERE merged_into IS NULL"
            ).fetchall()
        return [self._row_to_cluster(row) for row in rows]

    def delete_cluster(self, cluster_id: str) -> None:
        with self.transaction():
            self._conn.execute("DELETE FROM exemplars WHERE cluster_id = ?", (cluster_id,))
            self._conn.execute("DELETE FROM faces WHERE cluster_id = ?", (cluster_id,))
            self._conn.execute("DELETE FROM clusters WHERE cluster_id = ?", (cluster_id,))

    def recompute_cluster_face_count(self, cluster_id: str) -> int:
        row = self._conn.execute(
            """
            SELECT COUNT(*) AS n
            FROM faces
            WHERE cluster_id = ?
              AND assignment_state IN ('confirmed', 'manual')
            """,
            (cluster_id,),
        ).fetchone()
        count = int(row["n"])
        self._conn.execute(
            "UPDATE clusters SET face_count = ? WHERE cluster_id = ?",
            (count, cluster_id),
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
                    SELECT COUNT(*) FROM faces f WHERE f.cluster_id = clusters.cluster_id
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
                WHERE f.cluster_id = clusters.cluster_id
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
            WHERE cluster_id = ?
              AND (
                  face_id IS NULL
                  OR NOT EXISTS (
                      SELECT 1 FROM faces f
                      WHERE f.face_id = exemplars.face_id
                        AND f.cluster_id = exemplars.cluster_id
                        AND f.assignment_state IN ('confirmed', 'manual')
                  )
              )
            """,
            (cluster_id,),
        )
        faces = [
            face for face in self.load_faces_by_cluster(cluster_id)
            if face.assignment_state in (AssignmentState.CONFIRMED, AssignmentState.MANUAL)
        ]
        if not faces:
            self._conn.execute("DELETE FROM exemplars WHERE cluster_id = ?", (cluster_id,))
            self._conn.execute("DELETE FROM clusters WHERE cluster_id = ?", (cluster_id,))
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
                    WHERE cluster_id = ?
                    """,
                    (cluster_id,),
                )
                self._conn.execute("DELETE FROM clusters WHERE cluster_id = ?", (cluster_id,))
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
            cluster.last_updated_at = datetime.utcnow()
            self.save_cluster(cluster)

        self.recompute_cluster_face_count(cluster_id)

    def repair_empty_active_clusters(self) -> Dict[str, int]:
        rows = self._conn.execute(
            """
            SELECT c.cluster_id
            FROM clusters c
            WHERE c.merged_into IS NULL
              AND NOT EXISTS (
                  SELECT 1 FROM exemplars e WHERE e.cluster_id = c.cluster_id
              )
            """
        ).fetchall()
        cluster_ids = [row["cluster_id"] for row in rows]
        repaired_faces = 0
        with self.transaction():
            for cluster_id in cluster_ids:
                count_row = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM faces WHERE cluster_id = ?", (cluster_id,)
                ).fetchone()
                repaired_faces += count_row["n"]
                self._conn.execute(
                    """
                    UPDATE faces
                    SET cluster_id = NULL,
                        assignment_state = 'unassigned',
                        candidate_cluster_id = NULL,
                        decision_reason = 'legacy_empty_cluster_repaired'
                    WHERE cluster_id = ?
                    """,
                    (cluster_id,),
                )
                self._conn.execute("DELETE FROM clusters WHERE cluster_id = ?", (cluster_id,))
        return {"clusters_repaired": len(cluster_ids), "faces_unassigned": repaired_faces}

    def _row_to_cluster(self, row: sqlite3.Row) -> Cluster:
        exemplar_set = ExemplarSet()
        for exemplar_row in self._conn.execute(
            "SELECT * FROM exemplars WHERE cluster_id = ?", (row["cluster_id"],)
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
            INSERT INTO cluster_cannot_links(cluster_a_id, cluster_b_id, reason, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(cluster_a_id, cluster_b_id) DO UPDATE SET reason=excluded.reason
            """,
            (a, b, reason, datetime.utcnow().isoformat()),
        )
        self._commit_if_needed()

    def has_cannot_link(self, cluster_a_id: str, cluster_b_id: str) -> bool:
        if cluster_a_id == cluster_b_id:
            return False
        a, b = self._cluster_pair(cluster_a_id, cluster_b_id)
        return self._conn.execute(
            "SELECT 1 FROM cluster_cannot_links WHERE cluster_a_id=? AND cluster_b_id=?",
            (a, b),
        ).fetchone() is not None

    def clusters_share_photo_conflict(self, cluster_a_id: str, cluster_b_id: str) -> bool:
        """True when two clusters contain distinct faces from the same photo."""
        row = self._conn.execute(
            """
            SELECT 1
            FROM faces a
            JOIN faces b ON a.photo_id = b.photo_id AND a.face_id <> b.face_id
            WHERE a.cluster_id = ? AND b.cluster_id = ?
              AND a.photo_id IS NOT NULL
              AND a.assignment_state IN ('confirmed','manual')
              AND b.assignment_state IN ('confirmed','manual')
            LIMIT 1
            """,
            (cluster_a_id, cluster_b_id),
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
            self._conn.execute("DELETE FROM exemplars WHERE cluster_id=?", (cluster_id,))
            self._conn.execute("DELETE FROM clusters WHERE cluster_id=?", (cluster_id,))
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
        cluster.last_updated_at = datetime.utcnow()
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
            "SELECT suggestion_id, cluster_ids, status FROM suggestions WHERE status='pending'"
        ).fetchall()
        now = datetime.utcnow().isoformat()
        for row in rows:
            if row["suggestion_id"] == except_suggestion_id:
                continue
            if targets.intersection(json.loads(row["cluster_ids"])):
                self._conn.execute(
                    "UPDATE suggestions SET status='stale', updated_at=?, resolved_at=? WHERE suggestion_id=?",
                    (now, now, row["suggestion_id"]),
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
                WHERE cluster_id=?
                """,
                (survivor.cluster_id, loser.cluster_id, survivor.cluster_id, loser.cluster_id),
            )
            survivor.has_manual_correction = survivor.has_manual_correction or loser.has_manual_correction
            survivor.is_user_confirmed = survivor.is_user_confirmed or loser.is_user_confirmed
            survivor.last_updated_at = datetime.utcnow()
            loser.merged_into = survivor.cluster_id
            loser.face_count = 0
            loser.last_updated_at = survivor.last_updated_at
            # Clear stale exemplar objects before persisting cluster metadata;
            # the survivor is rebuilt only from its new real membership.
            survivor.exemplar_set = ExemplarSet()
            loser.exemplar_set = ExemplarSet()
            self._conn.execute("DELETE FROM exemplars WHERE cluster_id IN (?,?)", (survivor.cluster_id, loser.cluster_id))
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
                WHERE cluster_a_id=? OR cluster_b_id=?
                """,
                (loser.cluster_id, loser.cluster_id),
            ).fetchall()
            self._conn.execute(
                "DELETE FROM cluster_cannot_links WHERE cluster_a_id=? OR cluster_b_id=?",
                (loser.cluster_id, loser.cluster_id),
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
            source.last_updated_at = datetime.utcnow()
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
                f"UPDATE faces SET cluster_id=?, candidate_cluster_id=? WHERE face_id IN ({placeholders})",
                [new_cluster_id, new_cluster_id, *new_group],
            )
            self._conn.execute("DELETE FROM exemplars WHERE cluster_id IN (?,?)", (source_cluster_id, new_cluster_id))
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
                    WHERE photo_id=? AND cluster_id=?
                      AND assignment_state IN ('confirmed','manual')
                    LIMIT 1
                    """,
                    (face.photo_id, to_cluster_id),
                ).fetchone()
                if conflict is not None and conflict["face_id"] not in moving_ids:
                    raise ValueError(
                        f"Manual move blocked by same-photo cannot-link for photo {face.photo_id}"
                    )

            self._conn.executemany("DELETE FROM exemplars WHERE face_id=?", [(fid,) for fid in moving_ids])
            now = datetime.utcnow().isoformat()
            for face in faces:
                self._conn.execute(
                    """
                    UPDATE faces
                    SET cluster_id=?, assignment_state='manual', is_manually_corrected=1,
                        candidate_cluster_id=?, best_match_score=NULL,
                        second_best_cluster_id=NULL, second_best_score=NULL,
                        score_margin=NULL, decision_threshold=NULL,
                        decision_reason='manual_correction'
                    WHERE face_id=?
                    """,
                    (to_cluster_id, to_cluster_id, face.face_id),
                )
            for cluster_id in source_ids | {to_cluster_id}:
                self._conn.execute(
                    "UPDATE clusters SET has_manual_correction=1, last_updated_at=? WHERE cluster_id=?",
                    (now, cluster_id),
                )

            for source_id in source_ids - {to_cluster_id}:
                if self.load_faces_by_cluster(source_id):
                    self._rebuild_cluster_exemplars(
                        source_id,
                        exemplar_quality_threshold=exemplar_quality_threshold,
                        allow_low_quality_seed=True,
                    )
                else:
                    self._conn.execute("DELETE FROM clusters WHERE cluster_id=?", (source_id,))
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
                WHERE photo_id=? AND cluster_id=?
                  AND assignment_state IN ('confirmed','manual')
                """,
                (photo_id, from_cluster_id),
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
            "SELECT suggestion_id FROM suggestions WHERE status = 'pending'"
        ).fetchall()
        legacy_ids = [
            row["suggestion_id"]
            for row in rows
            if re.fullmatch(r"(?:merge|split)_\d+", row["suggestion_id"])
        ]
        if legacy_ids:
            self._conn.executemany(
                "DELETE FROM suggestions WHERE suggestion_id = ?",
                [(suggestion_id,) for suggestion_id in legacy_ids],
            )
            self._commit_if_needed()
        return len(legacy_ids)

    def save_suggestion(self, suggestion: Suggestion) -> bool:
        """Insert a suggestion once. Rejected/uncertain/stale IDs never resurface."""
        cursor = self._conn.execute(
            """
            INSERT INTO suggestions
                (suggestion_id, suggestion_type, cluster_ids, status, created_at,
                 payload_json, evidence_json, updated_at, resolved_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(suggestion_id) DO NOTHING
            """,
            (
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
        now = datetime.utcnow()
        resolved_at = now.isoformat() if status != SuggestionStatus.PENDING else None
        cursor = self._conn.execute(
            "UPDATE suggestions SET status=?, updated_at=?, resolved_at=? WHERE suggestion_id=?",
            (status.value, now.isoformat(), resolved_at, suggestion_id),
        )
        if cursor.rowcount == 0:
            raise ValueError(f"Unknown suggestion_id: {suggestion_id}")
        self._commit_if_needed()

    def load_suggestion(self, suggestion_id: str) -> Optional[Suggestion]:
        row = self._conn.execute(
            "SELECT * FROM suggestions WHERE suggestion_id=?", (suggestion_id,)
        ).fetchone()
        return self._row_to_suggestion(row) if row else None

    def load_pending_suggestions(self) -> List[Suggestion]:
        rows = self._conn.execute(
            "SELECT * FROM suggestions WHERE status = 'pending' ORDER BY created_at, suggestion_id"
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

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def validate_integrity(self, expected_embedding_dim: Optional[int] = None) -> List[str]:
        errors: List[str] = []

        for row in self._conn.execute(
            """
            SELECT c.cluster_id, c.face_count,
                   SUM(CASE WHEN f.assignment_state IN ('confirmed','manual') THEN 1 ELSE 0 END) AS actual
            FROM clusters c
            LEFT JOIN faces f ON f.cluster_id = c.cluster_id
            GROUP BY c.cluster_id
            """
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
            WHERE cluster_id IS NOT NULL
              AND photo_id IS NOT NULL
              AND assignment_state IN ('confirmed','manual')
            GROUP BY cluster_id, photo_id
            HAVING COUNT(*) > 1
            """
        ).fetchall()
        for row in same_photo_conflicts:
            errors.append(
                f"cluster {row['cluster_id']} contains {row['n']} faces from photo {row['photo_id']}"
            )

        invalid_members = self._conn.execute(
            """
            SELECT face_id, assignment_state, cluster_id FROM faces
            WHERE (assignment_state IN ('confirmed','manual') AND cluster_id IS NULL)
               OR (assignment_state IN ('ambiguous','unassigned') AND cluster_id IS NOT NULL)
            """
        ).fetchall()
        for row in invalid_members:
            errors.append(
                f"face {row['face_id']} has inconsistent state={row['assignment_state']} cluster={row['cluster_id']}"
            )

        for row in self._conn.execute(
            """
            SELECT e.id, e.face_id, e.cluster_id
            FROM exemplars e
            LEFT JOIN faces f ON f.face_id = e.face_id
            WHERE e.face_id IS NULL
               OR f.face_id IS NULL
               OR f.cluster_id != e.cluster_id
               OR f.assignment_state NOT IN ('confirmed','manual')
               OR COALESCE(f.recognition_restricted, 0) = 1
            """
        ).fetchall():
            errors.append(
                f"orphan/invalid exemplar id={row['id']} face={row['face_id']} cluster={row['cluster_id']}"
            )

        for row in self._conn.execute(
            """
            SELECT cluster_id FROM clusters
            WHERE merged_into IS NULL
              AND NOT EXISTS (SELECT 1 FROM exemplars e WHERE e.cluster_id = clusters.cluster_id)
            """
        ).fetchall():
            errors.append(f"active cluster {row['cluster_id']} has no exemplar")

        foreign_key_errors = self._conn.execute("PRAGMA foreign_key_check").fetchall()
        for row in foreign_key_errors:
            errors.append(f"foreign key error: {tuple(row)}")

        rows = self._conn.execute(
            "SELECT face_id, embedding, embedding_model_version, cluster_id FROM faces"
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
