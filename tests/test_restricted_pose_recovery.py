from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from face_grouping.clustering.consolidation import ConsolidationEngine
from face_grouping.clustering.data_types import Cluster, Face, Photo, PhotoProcessingStatus
from face_grouping.matching.assignment import AssignmentState
from face_grouping.matching.exemplars import Exemplar, ExemplarSet
from face_grouping.matching.incremental import IncrementalAssigner
from face_grouping.storage.store import FaceGroupingStore


def _unit(values):
    arr = np.asarray(values, dtype=np.float32)
    return arr / np.linalg.norm(arr)



def _ensure_photo(store: FaceGroupingStore, photo_id: str):
    if store.load_photo(photo_id) is None:
        store.save_photo(Photo(
            photo_id=photo_id,
            image_path=f"/tmp/{photo_id}.jpg",
            image_width=100,
            image_height=100,
            processing_status=PhotoProcessingStatus.COMPLETED,
            embedding_model_version="test_v1",
            config_version="test_cfg",
        ))

def _face(face_id: str, emb, *, cluster_id=None, state=AssignmentState.CONFIRMED,
          photo_id=None, face_index=0, restricted=False, quality=0.9):
    return Face(
        face_id=face_id,
        embedding=_unit(emb),
        quality_score=quality,
        yaw_ratio=0.8 if restricted else 0.0,
        cluster_id=cluster_id,
        assignment_state=state,
        photo_id=photo_id,
        face_index=face_index,
        embedding_model_version="test_v1",
        config_version="test_cfg",
        recognition_restricted=restricted,
        recognition_restriction_reason="pose_floor_only" if restricted else None,
        created_at=datetime.utcnow(),
    )


def _add_cluster(store: FaceGroupingStore, cluster_id: str, emb, *, members=8):
    emb = _unit(emb)
    exset = ExemplarSet(quality_bucket_size=3, pose_bucket_size=2)
    for i in range(2):
        exset.try_add(Exemplar(
            embedding=emb,
            quality_score=0.95 - i * 0.01,
            yaw_ratio=0.0,
            face_id=f"{cluster_id}_ex{i}",
            embedding_model_version="test_v1",
        ))
    cluster = Cluster(cluster_id=cluster_id, exemplar_set=exset, face_count=members)
    store.save_cluster(cluster)
    # save_cluster backfills the two exemplar faces. Add remaining authoritative members.
    for i in range(2, members):
        store.save_face(_face(f"{cluster_id}_m{i}", emb, cluster_id=cluster_id))
    store.recompute_cluster_face_count(cluster_id)
    return store.load_cluster(cluster_id)


def _engine(store: FaceGroupingStore, *, mature_min=8):
    assigner = IncrementalAssigner(
        store=store,
        t_match=0.41,
        band_width=0.08,
        top_k=2,
        sparse_cluster_margin=0.05,
        exemplar_admission_margin=0.10,
        min_cluster_margin=0.05,
        exemplar_quality_bucket_size=3,
        exemplar_pose_bucket_size=2,
    )
    return ConsolidationEngine(
        store=store,
        assigner=assigner,
        t_match=0.41,
        band_width=0.08,
        top_k=2,
        sparse_cluster_margin=0.05,
        min_cluster_margin=0.05,
        exemplar_admission_margin=0.10,
        exemplar_quality_threshold=0.70,
        exemplar_quality_bucket_size=3,
        exemplar_pose_bucket_size=2,
        restricted_pose_recovery_enabled=True,
        restricted_pose_mature_cluster_min_faces=mature_min,
        auto_correction_enabled=False,
    )


def test_restricted_pose_recovers_only_into_mature_cluster_and_never_becomes_exemplar(tmp_path: Path):
    store = FaceGroupingStore(str(tmp_path / "db.sqlite"))
    try:
        _add_cluster(store, "target", [1, 0, 0], members=8)
        _add_cluster(store, "other", [0, 1, 0], members=8)
        before = len(store.load_cluster("target").exemplar_set)
        restricted = _face(
            "restricted",
            [1, 0.02, 0],
            state=AssignmentState.UNASSIGNED,
            restricted=True,
            photo_id="photo_restricted",
        )
        _ensure_photo(store, "photo_restricted")
        _ensure_photo(store, "photo_r")
        store.save_face(restricted)

        result = _engine(store).recover_restricted_pose_faces()

        saved = store.load_face("restricted")
        assert result["restricted_pose_checked"] == 1
        assert result["restricted_pose_recovered_confirmed"] == 1
        assert saved.assignment_state == AssignmentState.CONFIRMED
        assert saved.cluster_id == "target"
        assert saved.recognition_restricted is True
        assert abs(saved.decision_threshold - 0.46) < 1e-9
        assert saved.best_match_score > 0.99
        assert saved.score_margin > 0.05
        assert saved.decision_reason.startswith("restricted_pose_recovery:")
        assert len(store.load_cluster("target").exemplar_set) == before
        assert all(
            ex.face_id != "restricted"
            for ex in store.load_cluster("target").exemplar_set.all_exemplars()
        )
        assert store.validate_integrity() == []
    finally:
        store.close()


def test_restricted_pose_does_not_use_ordinary_deferred_recovery_or_seed_hdbscan(tmp_path: Path):
    store = FaceGroupingStore(str(tmp_path / "db.sqlite"))
    try:
        # Only 3 authoritative members: not mature enough for restricted recognition.
        _add_cluster(store, "small", [1, 0, 0], members=3)
        restricted = _face(
            "restricted",
            [1, 0, 0],
            state=AssignmentState.UNASSIGNED,
            restricted=True,
            photo_id="photo_r",
        )
        _ensure_photo(store, "photo_r")
        store.save_face(restricted)
        engine = _engine(store, mature_min=8)

        ordinary = engine.recover_deferred_faces()
        discovery = engine.create_clusters_from_unassigned()
        restricted_result = engine.recover_restricted_pose_faces()

        saved = store.load_face("restricted")
        assert ordinary["deferred_checked"] == 0
        assert discovery["unassigned_hdbscan_points"] == 0
        assert discovery["new_clusters"] == 0
        assert restricted_result["restricted_pose_recovered_confirmed"] == 0
        assert saved.cluster_id is None
        assert saved.assignment_state == AssignmentState.AMBIGUOUS
        assert "best_cluster_not_mature" in saved.decision_reason
    finally:
        store.close()


def test_restricted_pose_requires_existing_margin_and_high_confidence_floor(tmp_path: Path):
    store = FaceGroupingStore(str(tmp_path / "db.sqlite"))
    try:
        # Two mature clusters intentionally close to the restricted face.
        _add_cluster(store, "a", [1.0, 0.0, 0.0], members=8)
        _add_cluster(store, "b", [0.995, 0.1, 0.0], members=8)
        restricted = _face(
            "restricted",
            [1.0, 0.05, 0.0],
            state=AssignmentState.UNASSIGNED,
            restricted=True,
            photo_id="photo_r",
        )
        _ensure_photo(store, "photo_r")
        store.save_face(restricted)

        result = _engine(store).recover_restricted_pose_faces()
        saved = store.load_face("restricted")
        assert result["restricted_pose_recovered_confirmed"] == 0
        assert saved.assignment_state == AssignmentState.AMBIGUOUS
        assert saved.cluster_id is None
        assert saved.best_match_score >= 0.46
        assert saved.score_margin < 0.05
        assert "insufficient_margin" in saved.decision_reason
    finally:
        store.close()


def test_restricted_pose_same_photo_cannot_link_is_preserved(tmp_path: Path):
    store = FaceGroupingStore(str(tmp_path / "db.sqlite"))
    try:
        _add_cluster(store, "target", [1, 0, 0], members=8)
        # Existing confirmed member in the same photo already owns target.
        existing = _face(
            "same_photo_member", [1, 0, 0], cluster_id="target",
            state=AssignmentState.CONFIRMED, photo_id="group_photo", face_index=0
        )
        _ensure_photo(store, "group_photo")
        store.save_face(existing)
        store.recompute_cluster_face_count("target")

        restricted = _face(
            "restricted", [1, 0, 0], state=AssignmentState.UNASSIGNED,
            restricted=True, photo_id="group_photo", face_index=1
        )
        store.save_face(restricted)

        result = _engine(store).recover_restricted_pose_faces()
        saved = store.load_face("restricted")
        assert result["restricted_pose_recovered_confirmed"] == 0
        assert saved.cluster_id is None
        assert saved.assignment_state == AssignmentState.UNASSIGNED
    finally:
        store.close()


def test_structural_split_detaches_restricted_members_for_recovery(tmp_path: Path):
    store = FaceGroupingStore(str(tmp_path / "db.sqlite"))
    try:
        _add_cluster(store, "source", [1, 0, 0], members=6)
        _ensure_photo(store, "restricted_split_photo")
        restricted = _face(
            "restricted_split", [1, 0, 0], cluster_id="source",
            state=AssignmentState.CONFIRMED, restricted=True,
            photo_id="restricted_split_photo", face_index=0,
        )
        store.save_face(restricted)
        store.recompute_cluster_face_count("source")

        authoritative = [
            f.face_id for f in store.load_faces_by_cluster("source")
            if f.assignment_state in (AssignmentState.CONFIRMED, AssignmentState.MANUAL)
            and not f.recognition_restricted
        ]
        assert len(authoritative) == 6
        result_clusters = store.execute_split_atomic(
            "source",
            [authoritative[:3], authoritative[3:]],
            exemplar_quality_threshold=0.7,
            mark_manual_correction=False,
            cannot_link_reason="test_split",
        )

        saved = store.load_face("restricted_split")
        assert len(result_clusters) == 2
        assert saved.recognition_restricted is True
        assert saved.cluster_id is None
        assert saved.assignment_state == AssignmentState.UNASSIGNED
        assert saved.decision_reason == "restricted_pose_recheck_after_split"
        assert store.validate_integrity() == []
    finally:
        store.close()
