import tempfile
import unittest
from pathlib import Path

import numpy as np

from face_grouping.clustering.candidates import SplitCandidate
from face_grouping.clustering.consolidation import ConsolidationEngine
from face_grouping.clustering.data_types import Cluster, Face, Photo, PhotoProcessingStatus
from face_grouping.matching.assignment import AssignmentState
from face_grouping.matching.exemplars import Exemplar, ExemplarSet
from face_grouping.storage.store import FaceGroupingStore


def unit(values):
    arr = np.asarray(values, dtype=np.float32)
    return arr / np.linalg.norm(arr)


class ConsolidationAutoCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = FaceGroupingStore(str(Path(self.tmp.name) / "test.db"))
        self.engine = ConsolidationEngine(
            store=self.store,
            assigner=None,
            t_match=0.41,
            band_width=0.08,
            top_k=2,
            sparse_cluster_margin=0.05,
            min_cluster_margin=0.05,
            exemplar_admission_margin=0.10,
            exemplar_quality_threshold=0.70,
            exemplar_quality_bucket_size=3,
            exemplar_pose_bucket_size=2,
            auto_correction_enabled=True,
            auto_correction_max_actions=12,
            small_fragment_max_faces=4,
            mature_cluster_min_faces=8,
            auto_split_min_group_faces=3,
        )

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def add_cluster(self, cluster_id, embeddings, *, protected=False):
        cluster = Cluster(
            cluster_id=cluster_id,
            exemplar_set=ExemplarSet(quality_bucket_size=3, pose_bucket_size=2),
            is_user_confirmed=protected,
        )
        self.store.save_cluster(cluster)

        faces = []
        for index, embedding in enumerate(embeddings):
            photo_id = f"{cluster_id}_p{index}"
            self.store.save_photo(Photo(
                photo_id=photo_id,
                image_path=f"/{photo_id}.jpg",
                image_width=100,
                image_height=100,
                processing_status=PhotoProcessingStatus.COMPLETED,
                embedding_model_version="irse50_test",
                config_version="test",
            ))
            face = Face(
                face_id=f"{cluster_id}_f{index}",
                embedding=np.asarray(embedding, dtype=np.float32),
                quality_score=0.95 - index * 0.001,
                yaw_ratio=0.0,
                cluster_id=cluster_id,
                assignment_state=AssignmentState.CONFIRMED,
                candidate_cluster_id=cluster_id,
                photo_id=photo_id,
                face_index=0,
                embedding_model_version="irse50_test",
                config_version="test",
            )
            self.store.save_face(face)
            faces.append(face)

        exemplar_set = ExemplarSet(quality_bucket_size=3, pose_bucket_size=2)
        for face in faces:
            exemplar_set.try_add(
                Exemplar(
                    embedding=face.embedding,
                    quality_score=face.quality_score,
                    yaw_ratio=face.yaw_ratio,
                    face_id=face.face_id,
                    embedding_model_version=face.embedding_model_version,
                )
            )
        cluster.exemplar_set = exemplar_set
        cluster.face_count = len(faces)
        self.store.save_cluster(cluster)
        self.store.recompute_cluster_face_count(cluster_id)
        return cluster

    def test_auto_merge_reconciles_tiny_fragment_into_unique_mature_target(self):
        identity = unit([1.0, 0.0, 0.0])
        other = unit([0.0, 1.0, 0.0])
        self.add_cluster("target", [identity] * 8)
        self.add_cluster("fragment", [identity] * 2)
        self.add_cluster("competitor", [other] * 8)

        audit = self.engine.audit_confirmed_clusters()
        matches = [
            c for c in audit.auto_merge_candidates
            if c.source_cluster_id == "fragment" and c.target_cluster_id == "target"
        ]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].source_coverage, 1.0)
        self.assertGreaterEqual(matches[0].strong_anchor_count, 1)

        result = self.engine.apply_high_confidence_auto_corrections()
        self.assertEqual(result["auto_merges"], 1)
        self.assertEqual(result["auto_splits"], 0)
        self.assertEqual(len(self.store.load_all_clusters(include_merged=False)), 2)
        retired = self.store.load_cluster("fragment")
        self.assertEqual(retired.merged_into, "target")
        self.assertEqual(self.store.load_cluster("target").face_count, 10)

    def test_auto_merge_is_blocked_when_target_is_not_unique_best_alternative(self):
        identity = unit([1.0, 0.0, 0.0])
        self.add_cluster("target", [identity] * 8)
        self.add_cluster("fragment", [identity] * 2)
        # An equally good third-person representation makes the evidence unsafe.
        self.add_cluster("competitor", [identity] * 8)

        audit = self.engine.audit_confirmed_clusters()
        pairs = {(c.source_cluster_id, c.target_cluster_id) for c in audit.auto_merge_candidates}
        self.assertNotIn(("fragment", "target"), pairs)
        self.assertNotIn(("fragment", "competitor"), pairs)

    def test_auto_merge_never_overrides_user_confirmed_cluster(self):
        identity = unit([1.0, 0.0, 0.0])
        other = unit([0.0, 1.0, 0.0])
        self.add_cluster("target", [identity] * 8, protected=True)
        self.add_cluster("fragment", [identity] * 2)
        self.add_cluster("competitor", [other] * 8)

        audit = self.engine.audit_confirmed_clusters()
        pairs = {(c.source_cluster_id, c.target_cluster_id) for c in audit.auto_merge_candidates}
        self.assertNotIn(("fragment", "target"), pairs)


    def test_singleton_cluster_is_never_auto_merged_even_with_perfect_similarity(self):
        identity = unit([1.0, 0.0, 0.0, 0.0])
        other = unit([0.0, 1.0, 0.0, 0.0])
        self.add_cluster("target", [identity] * 8)
        self.add_cluster("one_photo_person", [identity])
        self.add_cluster("competitor", [other] * 8)

        audit = self.engine.audit_confirmed_clusters()
        pairs = {(c.source_cluster_id, c.target_cluster_id) for c in audit.auto_merge_candidates}
        self.assertNotIn(("one_photo_person", "target"), pairs)

    def test_member_bridge_recovers_appearance_fragment_missing_from_current_exemplars(self):
        # Source appearance is x-axis. The target's first five members become
        # its 3-quality + 2-pose exemplars and are intentionally weak (0.30)
        # to that appearance. Two later, lower-priority target members are
        # strong bridge images (0.70 and 0.55), reproducing the wedding/makeup
        # blind spot found in the Gallery diagnostics.
        source_mode = unit([1.0, 0.0, 0.0, 0.0])
        weak_mode = unit([0.30, np.sqrt(1.0 - 0.30**2), 0.0, 0.0])
        bridge_a = unit([0.70, np.sqrt(1.0 - 0.70**2), 0.0, 0.0])
        bridge_b = unit([0.55, np.sqrt(1.0 - 0.55**2), 0.0, 0.0])
        other = unit([0.0, 0.0, 1.0, 0.0])

        target_embeddings = [weak_mode] * 5 + [bridge_a, bridge_b] + [weak_mode] * 5
        target = self.add_cluster("target", target_embeddings)
        self.add_cluster("appearance_fragment", [source_mode] * 4)
        self.add_cluster("competitor", [other] * 8)

        # Prove the current exemplar representation alone cannot explain it.
        exemplar_scores = [
            float(np.dot(source_mode, ex.embedding))
            for ex in target.exemplar_set.all_exemplars()
        ]
        self.assertLess(max(exemplar_scores), self.engine.t_match)

        audit = self.engine.audit_confirmed_clusters()
        matches = [
            c for c in audit.auto_merge_candidates
            if c.source_cluster_id == "appearance_fragment"
            and c.target_cluster_id == "target"
        ]
        self.assertEqual(len(matches), 1)
        candidate = matches[0]
        self.assertEqual(candidate.mode, "fragment_member_bridge")
        self.assertEqual(candidate.member_bridge_source_coverage, 1.0)
        self.assertGreaterEqual(candidate.distinct_bridge_target_faces, 2)
        self.assertGreaterEqual(candidate.distinct_bridge_target_photos, 0)
        self.assertGreaterEqual(candidate.strong_bridge_source_count, 2)

    def test_normal_90_percent_merge_suggestion_does_not_become_auto_merge(self):
        identity = unit([1.0, 0.0, 0.0, 0.0])
        outlier = unit([0.0, 1.0, 0.0, 0.0])
        competitor = unit([0.0, 0.0, 1.0, 0.0])
        self.add_cluster("a", [identity] * 9 + [outlier])
        self.add_cluster("b", [identity] * 10)
        self.add_cluster("competitor", [competitor] * 8)

        audit = self.engine.audit_confirmed_clusters()
        normal_pairs = {tuple(sorted(c.contributing_cluster_ids)) for c in audit.merge_candidates}
        auto_pairs = {tuple(sorted((c.source_cluster_id, c.target_cluster_id))) for c in audit.auto_merge_candidates}
        self.assertIn(("a", "b"), normal_pairs)
        self.assertNotIn(("a", "b"), auto_pairs)

    def test_member_bridge_is_blocked_by_close_third_person_competition(self):
        source_mode = unit([1.0, 0.0, 0.0, 0.0])
        weak_mode = unit([0.30, np.sqrt(1.0 - 0.30**2), 0.0, 0.0])
        bridge_a = unit([0.70, np.sqrt(1.0 - 0.70**2), 0.0, 0.0])
        bridge_b = unit([0.55, np.sqrt(1.0 - 0.55**2), 0.0, 0.0])
        close_competitor = unit([0.68, 0.0, np.sqrt(1.0 - 0.68**2), 0.0])

        self.add_cluster("target", [weak_mode] * 5 + [bridge_a, bridge_b] + [weak_mode] * 5)
        self.add_cluster("appearance_fragment", [source_mode] * 4)
        self.add_cluster("competitor", [close_competitor] * 8)

        audit = self.engine.audit_confirmed_clusters()
        pairs = {(c.source_cluster_id, c.target_cluster_id) for c in audit.auto_merge_candidates}
        self.assertNotIn(("appearance_fragment", "target"), pairs)

    def test_auto_split_safety_requires_three_faces_per_group_and_high_confidence_cohesion(self):
        left = unit([1.0, 0.0, 0.0])
        right = unit([-1.0, 0.0, 0.0])
        cluster = self.add_cluster("mixed", [left] * 3 + [right] * 3)
        faces = self.store.load_faces_by_cluster("mixed")
        faces_by_cluster = {"mixed": faces}
        groups = {
            0: [f.face_id for f in faces[:3]],
            1: [f.face_id for f in faces[3:]],
        }
        candidate = SplitCandidate(
            existing_cluster_id="mixed",
            receiving_new_labels=[0, 1],
            fractions={0: 0.5, 1: 0.5},
            face_groups=groups,
        )
        self.assertTrue(
            self.engine._split_candidate_is_auto_safe(candidate, [cluster], faces_by_cluster)
        )

        too_small = SplitCandidate(
            existing_cluster_id="mixed",
            receiving_new_labels=[0, 1],
            fractions={0: 2 / 6, 1: 4 / 6},
            face_groups={
                0: [f.face_id for f in faces[:2]],
                1: [f.face_id for f in faces[2:]],
            },
        )
        self.assertFalse(
            self.engine._split_candidate_is_auto_safe(too_small, [cluster], faces_by_cluster)
        )

    def test_automatic_split_execution_does_not_mark_manual_but_adds_cannot_link(self):
        left = unit([1.0, 0.0, 0.0])
        right = unit([-1.0, 0.0, 0.0])
        self.add_cluster("mixed", [left] * 3 + [right] * 3)
        faces = self.store.load_faces_by_cluster("mixed")
        groups = [
            [f.face_id for f in faces[:3]],
            [f.face_id for f in faces[3:]],
        ]
        result_ids = self.store.execute_split_atomic(
            "mixed",
            groups,
            exemplar_quality_threshold=0.7,
            mark_manual_correction=False,
            cannot_link_reason="auto_split_high_confidence",
        )
        self.assertEqual(len(result_ids), 2)
        a, b = result_ids
        self.assertFalse(self.store.load_cluster(a).has_manual_correction)
        self.assertFalse(self.store.load_cluster(b).has_manual_correction)
        self.assertTrue(self.store.has_cannot_link(a, b))


if __name__ == "__main__":
    unittest.main()
