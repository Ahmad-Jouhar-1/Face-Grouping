"""Conservative ground-truth resolver for folder-only private validation.

The resolver intentionally separates *exact* labels from labels inferred through
cluster identity. Exact face-level metrics use only:

1. a single-person benchmark photo that produced exactly one accepted face; or
2. deterministic within-photo elimination after other identities were resolved.

Cluster-to-identity mapping is learned only from (1). It is used for photo-level
set metrics and to enable elimination, but a face directly labelled by that
mapping is not counted as independent face-level ground truth. This prevents the
validator from simply grading the pipeline using the pipeline's own answer.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set


@dataclass
class FaceView:
    face_id: str
    photo_code: str
    face_index: int
    cluster_id: Optional[str]
    assignment_state: str


@dataclass
class Resolution:
    exact_face_gt: Dict[str, str]
    exact_source: Dict[str, str]
    cluster_anchor_counts: Dict[str, Dict[str, int]]
    cluster_identity_map: Dict[str, str]
    conflicted_anchor_clusters: Dict[str, Dict[str, int]]
    predicted_identity_sets: Dict[str, List[str]]
    unresolved_face_ids: List[str]
    photo_resolution: Dict[str, dict]


def resolve_folder_ground_truth(
    *,
    expected_identities_by_photo: Mapping[str, Sequence[str]],
    faces_by_photo: Mapping[str, Sequence[FaceView]],
) -> Resolution:
    exact: Dict[str, str] = {}
    source: Dict[str, str] = {}

    # 1) Independent exact anchors.
    for photo_code, expected_seq in expected_identities_by_photo.items():
        expected = list(expected_seq)
        faces = list(faces_by_photo.get(photo_code, ()))
        if len(expected) == 1 and len(faces) == 1:
            exact[faces[0].face_id] = expected[0]
            source[faces[0].face_id] = "single_person_single_face"

    # 2) Fixed cluster map from exact single-person anchors only.
    cluster_counts: Dict[str, Counter] = defaultdict(Counter)
    face_lookup = {face.face_id: face for faces in faces_by_photo.values() for face in faces}
    for face_id, identity in exact.items():
        face = face_lookup[face_id]
        if face.cluster_id:
            cluster_counts[face.cluster_id][identity] += 1

    cluster_map: Dict[str, str] = {}
    conflicted: Dict[str, Dict[str, int]] = {}
    for cluster_id, counts in cluster_counts.items():
        if len(counts) == 1:
            cluster_map[cluster_id] = next(iter(counts))
        else:
            conflicted[cluster_id] = dict(sorted(counts.items()))

    predicted_sets: Dict[str, List[str]] = {}
    photo_resolution: Dict[str, dict] = {}

    # 3) Resolve group photos conservatively. Direct cluster-map assignments
    # are useful for set evaluation but are not independent face GT. Elimination
    # can create an independent exact label for the last remaining face.
    for photo_code, expected_seq in expected_identities_by_photo.items():
        expected = set(expected_seq)
        faces = list(faces_by_photo.get(photo_code, ()))
        resolved_in_photo: Dict[str, str] = {
            face.face_id: exact[face.face_id]
            for face in faces
            if face.face_id in exact
        }
        direct_inferred: Dict[str, str] = {}

        # Only accept an identity if exactly one face in this photo points to it
        # via an anchor-clean cluster map. This avoids arbitrary choice when
        # fragmented clusters of the same person co-occur.
        mapped_candidates: Dict[str, List[FaceView]] = defaultdict(list)
        for face in faces:
            if face.face_id in resolved_in_photo or not face.cluster_id:
                continue
            identity = cluster_map.get(face.cluster_id)
            if identity in expected:
                mapped_candidates[identity].append(face)

        used_identities = set(resolved_in_photo.values())
        for identity, candidates in mapped_candidates.items():
            if identity not in used_identities and len(candidates) == 1:
                direct_inferred[candidates[0].face_id] = identity
                used_identities.add(identity)

        # Deterministic elimination: after exact/direct identities are accounted
        # for, one remaining face + one remaining identity has a unique answer.
        remaining_faces = [
            face for face in faces
            if face.face_id not in resolved_in_photo and face.face_id not in direct_inferred
        ]
        remaining_identities = expected - set(resolved_in_photo.values()) - set(direct_inferred.values())
        if len(remaining_faces) == 1 and len(remaining_identities) == 1:
            face = remaining_faces[0]
            identity = next(iter(remaining_identities))
            exact[face.face_id] = identity
            source[face.face_id] = "within_photo_elimination"
            resolved_in_photo[face.face_id] = identity
            remaining_faces = []
            remaining_identities = set()

        predicted = set(resolved_in_photo.values()) | set(direct_inferred.values())
        # Also include mapped cluster identities even when not face-unique; this
        # is only a photo-level set prediction and never exact face GT.
        for face in faces:
            if face.cluster_id and face.cluster_id in cluster_map:
                predicted.add(cluster_map[face.cluster_id])
        predicted_sets[photo_code] = sorted(predicted)

        photo_resolution[photo_code] = {
            "expected_identity_count": len(expected),
            "accepted_face_count": len(faces),
            "exact_face_labels": len(resolved_in_photo),
            "cluster_inferred_face_labels": len(direct_inferred),
            "predicted_identity_count": len(predicted),
            "unresolved_face_count": max(0, len(faces) - len(resolved_in_photo) - len(direct_inferred)),
            "missing_expected_identities": sorted(expected - predicted),
            "extra_predicted_identities": sorted(predicted - expected),
        }

    unresolved = [
        face.face_id
        for faces in faces_by_photo.values()
        for face in faces
        if face.face_id not in exact
    ]

    return Resolution(
        exact_face_gt=exact,
        exact_source=source,
        cluster_anchor_counts={cid: dict(sorted(counts.items())) for cid, counts in cluster_counts.items()},
        cluster_identity_map=cluster_map,
        conflicted_anchor_clusters=conflicted,
        predicted_identity_sets=predicted_sets,
        unresolved_face_ids=unresolved,
        photo_resolution=photo_resolution,
    )
