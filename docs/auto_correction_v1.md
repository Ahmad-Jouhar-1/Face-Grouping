# Consolidation Auto-Correction v1

## Goal
Reduce arrival-order sensitivity without changing the calibrated IR-SE50 matching threshold (`t_match = 0.41`). The incremental matcher remains precision-first; late consolidation corrects a narrow class of history-created confirmed fragments after more gallery evidence exists.

## Execution order
1. Deferred recovery (`AMBIGUOUS` / `UNASSIGNED`) on an immutable cluster snapshot.
2. HDBSCAN only on still-`UNASSIGNED` faces.
3. High-confidence structural auto-correction.
4. If an auto-correction occurred, rebuild affected exemplars (inside the atomic merge/split) and re-run deferred recovery once against the corrected representation.
5. Final confirmed-cluster audit.
6. Borderline Merge/Split evidence remains a user suggestion.

## Auto-Merge v1: small confirmed fragment -> mature cluster
Automation is intentionally limited to a source cluster of 2-4 faces and a target with at least 8 confirmed faces and at least `top_k` exemplars.

A pair is auto-merged only when all of the following hold:
- neither cluster is user-confirmed or manually corrected;
- no stored cannot-link exists;
- the clusters do not contain different faces from the same photo;
- every source face passes the existing high-confidence floor `max(effective_threshold, t_match + min_cluster_margin)` against the target;
- for every source face, the target is the unique best non-source alternative by at least the existing `min_cluster_margin`;
- at least one source face clears the existing strong exemplar-admission floor `max(effective_threshold, t_match + exemplar_admission_margin)`.

No new similarity threshold is introduced.

Each merge is applied atomically and then all evidence is recomputed before another automatic action. This prevents stale evidence from cascading.

## Auto-Split v1
The existing conservative split detector is retained as the first gate:
- exactly two HDBSCAN groups;
- no noise;
- each group has an exemplar-eligible face;
- every group is internally cohesive;
- maximum cross-group similarity remains below the existing lower ambiguous boundary (`t_match - band_width`).

Automatic execution adds stricter gates:
- at least 3 faces in each group;
- every member of each group must match that group's eligible medoid at `t_match + min_cluster_margin`;
- no user/manual correction may be present.

An automatic split does **not** mark the clusters as manually corrected, but it stores a cannot-link between the resulting groups to prevent a later merge pass from immediately undoing the split.

## Safety / auditability
- Auto-correction can be disabled in `configs/thresholds.yaml`.
- Maximum automatic actions per consolidation run are bounded.
- Every automatic action is returned in `run_consolidation()` as an evidence-bearing event and is stored in the Gallery runner's `summary.json` consolidation history.
- Pending user suggestions are created only after the automatic tier reaches a stable state.

## Stability validation
Use `tools/gallery_grouping/compare_seed_stability.py` after running the same exact Gallery under multiple seeds. It compares faces by `(photo, face_index)` and partitions with label-invariant ARI/Rand metrics, so changing `person_xxx` or cluster UUIDs does not affect the comparison.
