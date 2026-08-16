# Auto-Correction v2 — Arrival-Order Stabilization

## Goal

Reduce history-dependent fragmentation during consolidation without changing the calibrated face-matching threshold (`t_match = 0.41`) and without treating legitimate one-photo people as errors.

## Key product rule: singleton protection

A cluster containing exactly one confirmed face may simply represent a real person who appears once in the gallery. Therefore **singleton clusters are never auto-merged**, even if a similarity happens to be high. They also remain excluded from the normal mutual merge-suggestion rule.

The multi-seed stability evaluator now reports singleton and non-singleton cluster counts separately. For the 495-photo baseline (seeds 7, 42, 21, 77, 123):

- total clusters: 29..35
- singleton clusters: 8..12
- non-singleton clusters: 20..24
- mean ARI: 0.972086
- minimum ARI: 0.963351

The correction target is therefore not “minimize all clusters”; it is to reduce unstable **multi-face fragments** while preserving valid one-photo identities.

## Auto-Merge Tier A — Full Mutual Evidence

The normal human-review merge suggestion remains unchanged at 90% mutual high-confidence coverage.

Automatic merge is stricter:

- both clusters contain at least two faces;
- no user-confirmed/manual cluster is touched;
- no stored cannot-link;
- no same-photo conflict;
- 100% of cluster A members pass the existing high-confidence match floor against cluster B;
- 100% of cluster B members pass the same floor against cluster A;
- at least two strong anchors in each direction (or all members when the cluster has only two faces), using the existing exemplar-admission-strength floor;
- every member must prefer the proposed partner over any third cluster by the existing `min_cluster_margin`.

A pair that is 90%/90% remains a suggestion; it is not promoted to automatic merge.

## Auto-Merge Tier B — Member-Bridge Fragment Reconciliation

This tier addresses appearance modes that are real members of a mature identity but are absent from its current five exemplars (for example strong wedding makeup, age changes, hair changes, or other appearance regimes).

A fragment is defined **relatively**, not by a fixed 2–4 face limit:

- source has at least two faces (singleton protection);
- target is mature (default at least 8 faces);
- source is smaller than target;
- source/target size ratio is at most 0.35;
- source is internally connected at the existing lower ambiguous boundary;
- every source face has repeated support from at least two distinct target members at `t_match`;
- at least two source faces reach the existing high-confidence floor;
- at least two source faces reach the existing exemplar-admission-strength floor;
- bridge evidence spans at least two target faces and two target photos;
- target must be the best member-level alternative for every source face;
- high-confidence bridge faces must beat third-person alternatives by `min_cluster_margin`;
- normal safety blockers (user intent, cannot-link, same-photo) still apply.

This is designed to catch the observed wedding/makeup blind spot without lowering the production matching threshold.

## Auto-Split

Auto-split remains the stricter v1 policy: exactly two clear HDBSCAN groups, at least three faces per group, strong internal cohesion, weak cross-group similarity, and no user/manual protection conflict.

## Re-audit after every mutation

Only one structural correction is applied per audit snapshot. The engine rebuilds the merged/split cluster representation, re-audits from fresh state, and only then considers another correction. Deferred AMBIGUOUS/UNASSIGNED faces get another recovery pass after structural correction.

## Diagnostics

Every Gallery run now writes:

- `auto_corrections.csv` — every automatic merge/split and its evidence.
- `auto_merge_policy_diagnostics.json` — final pairwise policy evaluations, including rejection reasons.
- `summary.json` — total auto-merge/auto-split counts plus per-consolidation events.

The multi-seed evaluator also reports singleton and non-singleton cluster ranges separately.
