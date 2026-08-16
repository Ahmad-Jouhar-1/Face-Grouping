"""Dependency-light clustering metrics used only by the final validation tool."""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def weighted_purity(gt: Mapping[str, str], pred: Mapping[str, str]) -> dict:
    groups = defaultdict(list)
    for sample_id, gt_label in gt.items():
        if sample_id in pred:
            groups[pred[sample_id]].append(gt_label)
    correct = sum(max(Counter(labels).values()) for labels in groups.values() if labels)
    n = sum(len(labels) for labels in groups.values())
    return {"samples": n, "correct_majority_faces": correct, "purity": correct / n if n else 0.0}


def bcubed(gt: Mapping[str, str], pred: Mapping[str, str]) -> dict:
    sample_ids = [sample_id for sample_id in gt if sample_id in pred]
    if not sample_ids:
        return {"samples": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    gt_groups = defaultdict(set)
    pred_groups = defaultdict(set)
    for sample_id in sample_ids:
        gt_groups[gt[sample_id]].add(sample_id)
        pred_groups[pred[sample_id]].add(sample_id)

    precision_sum = 0.0
    recall_sum = 0.0
    for sample_id in sample_ids:
        same_gt = gt_groups[gt[sample_id]]
        same_pred = pred_groups[pred[sample_id]]
        overlap = len(same_gt & same_pred)
        precision_sum += overlap / len(same_pred)
        recall_sum += overlap / len(same_gt)

    precision = precision_sum / len(sample_ids)
    recall = recall_sum / len(sample_ids)
    return {"samples": len(sample_ids), "precision": precision, "recall": recall, "f1": _f1(precision, recall)}


def _pair_count(labels: Iterable[str]) -> int:
    counts = Counter(labels)
    return sum(count * (count - 1) // 2 for count in counts.values())


def pairwise(gt: Mapping[str, str], pred: Mapping[str, str]) -> dict:
    sample_ids = [sample_id for sample_id in gt if sample_id in pred]
    if not sample_ids:
        return {
            "samples": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "true_pairs": 0,
            "predicted_pairs": 0,
            "true_positive_pairs": 0,
        }

    true_pairs = _pair_count(gt[s] for s in sample_ids)
    predicted_pairs = _pair_count(pred[s] for s in sample_ids)
    joint = Counter((gt[s], pred[s]) for s in sample_ids)
    true_positive_pairs = sum(count * (count - 1) // 2 for count in joint.values())
    precision = true_positive_pairs / predicted_pairs if predicted_pairs else (1.0 if true_pairs == 0 else 0.0)
    recall = true_positive_pairs / true_pairs if true_pairs else 1.0
    return {
        "samples": len(sample_ids),
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "true_pairs": true_pairs,
        "predicted_pairs": predicted_pairs,
        "true_positive_pairs": true_positive_pairs,
    }


def contamination(gt: Mapping[str, str], pred: Mapping[str, str]) -> dict:
    cluster_counts = defaultdict(Counter)
    for sample_id, gt_label in gt.items():
        cluster_id = pred.get(sample_id)
        if cluster_id is not None:
            cluster_counts[cluster_id][gt_label] += 1

    details = {}
    minority_total = 0
    assigned_total = 0
    for cluster_id, counts in cluster_counts.items():
        known = sum(counts.values())
        assigned_total += known
        minority = known - max(counts.values())
        if minority:
            minority_total += minority
            details[cluster_id] = {
                "known_gt_faces": known,
                "minority_faces": minority,
                "identity_counts": dict(sorted(counts.items())),
            }

    return {
        "known_gt_assigned_faces": assigned_total,
        "contaminated_clusters": len(details),
        "minority_faces_in_contaminated_clusters": minority_total,
        "false_merge_face_rate": minority_total / assigned_total if assigned_total else 0.0,
        "contaminated_cluster_details": details,
    }


def fragmentation(gt: Mapping[str, str], pred: Mapping[str, str]) -> dict:
    gt_totals = Counter(gt.values())
    identity_clusters = defaultdict(Counter)
    for sample_id, identity in gt.items():
        cluster_id = pred.get(sample_id)
        if cluster_id is not None:
            identity_clusters[identity][cluster_id] += 1

    per_identity = {}
    fragmented = {}
    largest_total = 0
    for identity in sorted(gt_totals):
        counts = identity_clusters.get(identity, Counter())
        sizes = sorted(counts.values(), reverse=True)
        largest = sizes[0] if sizes else 0
        largest_total += largest
        row = {
            "total_evaluable_faces": gt_totals[identity],
            "assigned_faces": sum(sizes),
            "cluster_count": len(sizes),
            "cluster_sizes": sizes,
            "largest_cluster_faces": largest,
            "largest_cluster_recall": largest / gt_totals[identity] if gt_totals[identity] else 0.0,
        }
        per_identity[identity] = row
        if len(sizes) > 1:
            fragmented[identity] = row

    macro = (
        sum(row["largest_cluster_recall"] for row in per_identity.values()) / len(per_identity)
        if per_identity else 0.0
    )
    micro = largest_total / sum(gt_totals.values()) if gt_totals else 0.0
    return {
        "identities": len(gt_totals),
        "identities_with_any_cluster": sum(1 for row in per_identity.values() if row["cluster_count"] > 0),
        "fragmented_identities": len(fragmented),
        "micro_identity_recall": micro,
        "macro_identity_recall": macro,
        "fragmented_identity_details": fragmented,
        "per_identity": per_identity,
    }


def clustering_bundle(gt: Mapping[str, str], pred: Mapping[str, str]) -> dict:
    return {
        "purity": weighted_purity(gt, pred),
        "bcubed": bcubed(gt, pred),
        "pairwise": pairwise(gt, pred),
    }


def photo_set_metrics(expected_by_photo: Mapping[str, Sequence[str]], predicted_by_photo: Mapping[str, Sequence[str]]) -> dict:
    rows = []
    for photo_id, expected_seq in expected_by_photo.items():
        expected = set(expected_seq)
        predicted = set(predicted_by_photo.get(photo_id, ()))
        tp = len(expected & predicted)
        precision = tp / len(predicted) if predicted else (1.0 if not expected else 0.0)
        recall = tp / len(expected) if expected else 1.0
        union = len(expected | predicted)
        jaccard = tp / union if union else 1.0
        rows.append((precision, recall, jaccard, expected == predicted, len(expected) > 1))

    if not rows:
        return {"photos": 0, "exact_set_match_rate": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "macro_jaccard": 0.0}

    def summarize(selected):
        if not selected:
            return {"photos": 0, "exact_set_match_rate": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "macro_jaccard": 0.0}
        return {
            "photos": len(selected),
            "exact_set_match_rate": sum(1 for r in selected if r[3]) / len(selected),
            "macro_precision": sum(r[0] for r in selected) / len(selected),
            "macro_recall": sum(r[1] for r in selected) / len(selected),
            "macro_jaccard": sum(r[2] for r in selected) / len(selected),
        }

    return {
        "all": summarize(rows),
        "multi_person_only": summarize([r for r in rows if r[4]]),
    }
