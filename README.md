# Face Grouping AI

Precision-first, multi-user face-grouping service designed to run independently from the product backend.

This README is the **main integration contract for backend developers**. A backend developer should be able to deploy the AI service and integrate with it without reading the internal clustering code or accessing the AI database directly.

---

## 1. Responsibility boundary

The system is intentionally split into two independent services:

```text
Mobile / Web Client
        |
        v
Product Backend (for example FastAPI)
        |
        | Internal HTTP
        v
Face Grouping AI API
        |
        v
FaceGroupingService
        |
   +----+--------------------+
   |                         |
Shared AI Runtime      Tenant-scoped session
(models once/process)   (user_id + AI SQLite)
   |                         |
   +----------> Face Grouping Core
                            |
                            v
                       AI-only SQLite
```

### Product backend owns

- Authentication and authorization
- User accounts
- Original photos / object storage
- Product database and gallery metadata
- Stable backend `photo_id`
- Background jobs and retry scheduling
- User-facing API and UI

### Face Grouping AI owns

- Face detection and landmarks
- Face alignment and quality assessment
- IR-SE50 embeddings
- Incremental face matching
- Person clusters and exemplars
- Deferred recovery
- HDBSCAN new-person discovery
- Merge / Split suggestions
- Manual Move execution
- AI lifecycle state
- Its own SQLite persistence

**The backend must never read or modify the AI SQLite database directly.** All integration must go through the HTTP API documented below.

---

## 2. Installation

The project has exactly **one dependency file** and exactly **one installation command**:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` installs the AI runtime, FastAPI HTTP layer, test tooling, and the local `face_grouping` package itself. No second requirements file, optional API extra, or additional package-install command is required.

`pyproject.toml` contains package/build metadata and exposes the `face-grouping-api` console entry point. It intentionally contains **no dependency list**; all Python dependencies remain in the single `requirements.txt` file.

The `tests/` directory is kept for AI regression/CI verification only. Backend integration does not depend on the test files and they are not installed as part of the `face_grouping` package.

To verify the AI project during development or CI:

```bash
pytest -q
```

---

## 3. Required model assets

The service expects:

```text
models/mediapipe/face_detector.tflite
models/mediapipe/face_landmarker.task
models/irse50/irse50.pth
```

The approved IR-SE50 checkpoint may be deployed separately because of its file size. It must be placed at the configured path before the AI service is started.

The project root can be configured with:

```text
FACE_GROUPING_ROOT=/opt/face-grouping-ai
```

Optional configuration overrides:

```text
FACE_GROUPING_THRESHOLDS_PATH
FACE_GROUPING_MODEL_PATHS_PATH
```

---

## 4. Runtime configuration

Recommended production environment variables:

```text
FACE_GROUPING_DB_PATH=/var/lib/face-grouping/face_grouping.db
FACE_GROUPING_ROOT=/opt/face-grouping-ai
FACE_GROUPING_INTERNAL_API_KEY=<strong-internal-secret>
FACE_GROUPING_MAX_UPLOAD_MB=25

FACE_GROUPING_CONSOLIDATE_EVERY_PHOTOS=50
FACE_GROUPING_CONSOLIDATE_IDLE_HOURS=24
FACE_GROUPING_PHOTO_LEASE_SECONDS=300
FACE_GROUPING_CONSOLIDATION_LEASE_SECONDS=1800

FACE_GROUPING_API_HOST=127.0.0.1
FACE_GROUPING_API_PORT=8001
```

See `.env.example` for the available settings.

---

## 5. Start the AI service

Using the installed command:

```bash
face-grouping-api
```

or directly with Uvicorn:

```bash
uvicorn face_grouping.api.app:app --host 127.0.0.1 --port 8001
```

Example internal base URL:

```text
http://127.0.0.1:8001
```

The backend should keep this URL in server-side configuration, for example:

```text
FACE_GROUPING_AI_URL=http://127.0.0.1:8001
```

---

# Backend Integration Contract

## 6. Authentication and user isolation

Every request that operates on one user's AI state requires:

```http
X-User-ID: <authenticated-backend-user-id>
```

Example:

```http
X-User-ID: 52
```

or:

```http
X-User-ID: 9c60bca7-3df9-4ccd-a603-462c048de445
```

The AI treats this value as an opaque string.

### Important security rule

`X-User-ID` must be derived by the backend from its authenticated principal.

Correct flow:

```text
Client request
    |
    v
Backend authenticates client
    |
    v
Backend resolves authenticated user ID
    |
    v
Backend sends X-User-ID to AI
```

Do **not** trust an arbitrary `user_id` sent by the mobile/web client and forward it directly to the AI service.

All persisted AI entities are tenant-scoped by `user_id`. Matching, consolidation, HDBSCAN, Merge, Split, Manual Move, suggestions, and deletion only see the current user's data.

---

## 7. Internal service authentication

The AI service can additionally require an internal API key.

AI configuration:

```text
FACE_GROUPING_INTERNAL_API_KEY=<secret>
```

Backend request header:

```http
X-Internal-API-Key: <secret>
```

This secret must stay server-side and must never be sent to the mobile/web client.

---

## 8. API endpoint summary

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/v1/photos/{photo_id}/process` | Process one photo idempotently |
| `GET` | `/v1/people` | List discovered people for the current user |
| `GET` | `/v1/people/{cluster_id}/photos` | List backend photo IDs belonging to one person |
| `GET` | `/v1/lifecycle` | Read consolidation lifecycle state |
| `POST` | `/v1/consolidation?force=false` | Run consolidation when it is due |
| `POST` | `/v1/consolidation?force=true` | Explicitly force consolidation |
| `GET` | `/v1/suggestions` | List pending Merge / Split suggestions |
| `POST` | `/v1/suggestions/{suggestion_id}/resolve` | Accept, reject, or mark a suggestion uncertain |
| `POST` | `/v1/people/move-photos` | Execute an explicit user-driven Manual Move |
| `DELETE` | `/v1/photos/{photo_id}` | Delete one photo's AI state |
| `DELETE` | `/v1/users/me/data` | Delete all AI state for the current user |
| `GET` | `/v1/storage/integrity` | Run a tenant-local storage diagnostic |
| `GET` | `/health/live` | Process liveness |
| `GET` | `/health/ready` | AI runtime/startup readiness |

---

## 9. Process a photo

### Request

```http
POST /v1/photos/{photo_id}/process
Content-Type: multipart/form-data
X-User-ID: <authenticated-user-id>
X-Internal-API-Key: <internal-secret>
```

The multipart field name is:

```text
image
```

Example:

```http
POST /v1/photos/photo_123/process
```

The backend sends the original image bytes. The AI stages the upload only temporarily for inference and deletes the temporary file afterward.

The AI SQLite database stores a stable internal reference such as:

```text
backend-photo://photo_123
```

It does **not** depend on a shared filesystem path from the backend.

### `photo_id` ownership

`photo_id` is created and owned by the product backend.

It may be an integer, UUID, or other stable string representation. The important requirement is:

> The same backend photo must always use the same `photo_id` for the same user.

The same `photo_id` may exist for different users because all AI state is also scoped by `user_id`.

### Example response

```json
{
  "photo_id": "photo_123",
  "faces": [
    {
      "face_id": "...",
      "photo_id": "photo_123",
      "cluster_id": "...",
      "assignment_state": "confirmed",
      "quality_score": 0.91,
      "bbox": [120.0, 45.0, 248.0, 198.0],
      "recognition_restricted": false
    }
  ],
  "cached": false,
  "lifecycle": {
    "photos_since_consolidation": 18,
    "consolidation_due": false,
    "due_reason": null,
    "due_since": null,
    "last_photo_completed_at": "2026-08-16T10:30:00",
    "last_consolidated_at": "2026-08-15T20:00:00",
    "consolidation_in_progress": false,
    "last_consolidation_error": null
  }
}
```

### Important response fields

#### `face_id`

Internal stable identifier for one detected face.

#### `cluster_id`

Identifier of the discovered person inside Face Grouping AI.

Faces/photos associated with the same `cluster_id` belong to the same current person cluster.

The backend should treat `cluster_id` as an opaque AI-owned identifier. It should not generate or modify it.

#### `assignment_state`

Typical states include:

```text
confirmed
ambiguous
unassigned
manual
```

#### `bbox`

Detected face bounding box inside the source photo.

#### `cached`

If:

```json
{"cached": true}
```

then this `(user_id, photo_id)` was already completed. The AI returned the stored result without running inference again.

---

## 10. Idempotency and duplicate requests

The tuple:

```text
(user_id, photo_id)
```

is the idempotency identity for photo processing.

If the backend loses the network response after sending a photo, it should retry with the **same `photo_id`**.

A completed retry returns:

```text
200 OK
cached=true
```

and does not:

- rerun face inference
- create duplicate faces
- increment the consolidation cadence again

Never generate a new `photo_id` just because an HTTP retry is required.

---

## 11. Concurrent processing of the same photo

The AI protects photo processing with an SQLite lease.

If two workers send the same `(user_id, photo_id)` at the same time:

```text
Request A -> owns processing lease -> inference runs
Request B -> 409 Conflict + Retry-After
```

The second backend job should wait according to `Retry-After`, then resend the same request using the same `photo_id`.

If an AI worker dies, the lease expires and another request may reclaim the photo. A stale worker cannot later commit its old result.

---

## 12. Consolidation lifecycle

The approved default lifecycle is:

```text
50 successfully processed photos
OR
24 hours of inactivity
```

Configuration:

```text
FACE_GROUPING_CONSOLIDATE_EVERY_PHOTOS=50
FACE_GROUPING_CONSOLIDATE_IDLE_HOURS=24
```

Photo processing itself does **not** automatically run consolidation inside the user-facing request.

After each successful photo, inspect:

```json
"consolidation_due": true
```

If true, the backend should enqueue a background job that calls:

```http
POST /v1/consolidation?force=false
```

Recommended flow:

```text
User uploads photo
    |
    v
Backend stores original photo
    |
    v
POST /v1/photos/{photo_id}/process
    |
    v
Return normal photo request
    |
    +---- if consolidation_due=true ---->
              enqueue background job
                      |
                      v
            POST /v1/consolidation?force=false
```

This keeps normal photo-upload latency predictable.

### 24-hour idle trigger

If no new photo arrives after the idle period, there is no photo response available to notify the backend.

A backend scheduler can periodically call:

```http
GET /v1/lifecycle
```

for users that have Face Grouping state. If consolidation is due, enqueue the consolidation job.

### Consolidation response when not due

A `force=false` call may return:

```json
{
  "ran": false,
  "summary": {},
  "skipped_reason": "not_due",
  "lifecycle": {}
}
```

### Concurrent consolidation jobs

Only one consolidation should run for a user at a time.

A duplicate job receives:

```text
409 Conflict
Retry-After: ...
```

---

## 13. List discovered people

```http
GET /v1/people
X-User-ID: <authenticated-user-id>
```

The response provides the current person clusters for this user.

The backend should treat each returned `cluster_id` as the AI's identifier for one person.

---

## 14. Get photos for a person

```http
GET /v1/people/{cluster_id}/photos
X-User-ID: <authenticated-user-id>
```

The AI returns backend `photo_id` values associated with that person.

The original images remain owned by the backend. Therefore the normal gallery flow is:

```text
GET /v1/people/{cluster_id}/photos
        |
        v
AI returns photo_id values
        |
        v
Backend resolves those IDs in its own DB/storage
        |
        v
Backend returns actual gallery photos to the client
```

The AI service is the source of the relationship:

```text
Person <-> Photo
```

not the source of the original image files.

---

## 15. Merge / Split suggestions

The AI may create conservative high-confidence Merge or Split suggestions.

List pending suggestions:

```http
GET /v1/suggestions
```

Resolve one suggestion:

```http
POST /v1/suggestions/{suggestion_id}/resolve
Content-Type: application/json
```

Body:

```json
{
  "status": "accepted"
}
```

Allowed statuses:

```text
accepted
rejected
uncertain
```

A rejected Merge creates a persistent cannot-link inside the AI state so the same inappropriate merge is not repeatedly proposed.

---

## 16. Manual Move

Manual Move is intentionally user-driven. The AI does not generate automatic Move suggestions.

Endpoint:

```http
POST /v1/people/move-photos
Content-Type: application/json
```

Example body:

```json
{
  "photo_ids": ["p1", "p2"],
  "from_cluster_id": "person_a",
  "to_cluster_id": "person_b"
}
```

The backend should call this only after an explicit user action in the product UI.

---

## 17. Delete a photo

When a photo is permanently deleted from the product account, the backend should also call:

```http
DELETE /v1/photos/{photo_id}
```

This removes the photo's AI state, including affected faces/exemplars, and repairs affected clusters atomically.

If inference for that photo is still running, deletion invalidates its processing lease so the stale inference worker cannot commit afterward.

---

## 18. Delete all AI data for a user

When the product account is deleted, call:

```http
DELETE /v1/users/me/data
X-User-ID: <authenticated-user-id>
```

This removes the user's AI-owned state, including:

- Photos
- Faces
- Clusters
- Exemplars
- Suggestions
- Cannot-links
- Lifecycle state

It does not affect any other tenant.

The backend database and AI SQLite intentionally do not participate in one distributed transaction. For account deletion, the backend should keep a tombstone/retryable cleanup job until the AI deletion succeeds.

---

## 19. Storage integrity diagnostic

```http
GET /v1/storage/integrity
```

This is a tenant-local diagnostic endpoint intended for administration/debugging. It is not required for the normal user-facing flow.

---

## 20. Health checks

### Liveness

```http
GET /health/live
```

Confirms that the API process is alive.

### Readiness

```http
GET /health/ready
```

Confirms that the service has completed startup and the AI runtime is ready to accept work.

Use these endpoints for container orchestration, service monitoring, or load-balancer health checks.

---

## 21. Backend HTTP example with FastAPI/httpx

The product backend can keep one long-lived `httpx.AsyncClient` for calls to the AI service.

Example:

```python
import httpx


class FaceGroupingAIClient:
    def __init__(self, base_url: str, internal_api_key: str) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(120.0),
        )
        self._internal_api_key = internal_api_key

    def _headers(self, user_id: str) -> dict[str, str]:
        return {
            "X-User-ID": str(user_id),
            "X-Internal-API-Key": self._internal_api_key,
        }

    async def process_photo(
        self,
        *,
        user_id: str,
        photo_id: str,
        filename: str,
        image_bytes: bytes,
        content_type: str = "image/jpeg",
    ) -> dict:
        response = await self._client.post(
            f"/v1/photos/{photo_id}/process",
            headers=self._headers(user_id),
            files={
                "image": (
                    filename,
                    image_bytes,
                    content_type,
                )
            },
        )
        response.raise_for_status()
        return response.json()

    async def consolidate(self, *, user_id: str, force: bool = False) -> dict:
        response = await self._client.post(
            "/v1/consolidation",
            params={"force": str(force).lower()},
            headers=self._headers(user_id),
        )
        response.raise_for_status()
        return response.json()
```

The exact backend client implementation is up to the backend team. The important contract is the HTTP behavior documented in this README.

---

## 22. HTTP errors and retry policy

Recommended backend behavior:

| Status | Meaning | Backend action |
|---|---|---|
| `200` | Success; completed duplicates may return `cached=true` | Continue normally |
| `400` | Invalid request/domain input | Do not retry unchanged |
| `401` | Wrong internal API key | Fix service configuration |
| `404` | Resource not found for this tenant | Do not retry unchanged |
| `409` | Same photo/consolidation currently owned, or conflicting operation | Honor `Retry-After` when present |
| `413` | Uploaded image exceeds configured limit | Do not retry same file unchanged |
| `415` | Unsupported/non-image Content-Type | Correct the upload |
| `422` | Image bytes could not be decoded/read | Refetch/re-encode, then reuse same `photo_id` |
| `503` | SQLite/service temporarily busy | Retry with backoff and honor `Retry-After` |
| `5xx` | Unexpected transient server failure may have occurred | Bounded retry with same IDs |

### Suggested retry pattern

For retryable operations use bounded exponential backoff, for example:

```text
Attempt 1 -> 1 s
Attempt 2 -> 2 s
Attempt 3 -> 4 s
```

If the AI response contains:

```http
Retry-After: N
```

that value should take precedence over the backend's default delay.

Always preserve the same `(user_id, photo_id)` across photo retries.

For `422`, do not blindly resend the same invalid bytes. Refetch or re-encode the image first, then retry with the same `photo_id`.

---

## 23. Recommended backend event flows

### New photo

```text
1. Authenticate user.
2. Store the original image in backend/object storage.
3. Create or obtain stable photo_id.
4. POST image to /v1/photos/{photo_id}/process.
5. Store/use returned AI IDs as required by the product.
6. Return the user-facing request without waiting for consolidation.
7. If lifecycle.consolidation_due == true, enqueue consolidation.
```

### Show People gallery

```text
1. GET /v1/people.
2. Select a cluster_id/person.
3. GET /v1/people/{cluster_id}/photos.
4. Resolve returned photo_id values in the backend DB/storage.
5. Return actual images to the client.
```

### User manually corrects a person

```text
Explicit UI action
    |
    +--> Manual Move -> POST /v1/people/move-photos
    |
    +--> Merge/Split suggestion decision
            -> POST /v1/suggestions/{id}/resolve
```

### Delete photo

```text
1. Delete/tombstone photo according to backend policy.
2. DELETE /v1/photos/{photo_id} in AI.
3. Retry cleanup if a transient AI error occurs.
```

### Delete account

```text
1. Disable/tombstone user account.
2. DELETE /v1/users/me/data in AI.
3. Keep retryable cleanup state until AI confirms deletion.
4. Complete backend-specific account cleanup.
```

---

## 24. SQLite deployment boundary

The AI database uses:

- WAL mode
- foreign keys
- busy timeout
- explicit transactions
- tenant-scoped keys/relationships

Current schema version:

```text
v7
```

The service can automatically upgrade legacy v5 and production v6 databases to v7.

Supported deployment model:

```text
Many users
    |
    v
One AI node
    |
    v
Local AI SQLite file
```

Multiple worker processes on the same AI node may share the local SQLite file, although writes remain serialized by SQLite.

Do **not** put the SQLite file on generic NFS/network storage and share it between independent AI hosts.

If horizontal multi-host scale is required later, use user sharding/sticky routing with separate databases or replace the persistence adapter with a server database. This can be done without changing the AI core/API contract.

The MediaPipe/IR-SE50 runtime is shared per process, not per user. Additional worker processes duplicate model memory, so start with one worker and scale only after measuring throughput and memory usage.

---

## 25. Integration rules that must not be violated

1. **Never access AI SQLite directly from the product backend.**
2. **Always derive `X-User-ID` from the authenticated backend user.**
3. **Keep `photo_id` stable across retries.**
4. **Treat `cluster_id`, `face_id`, and suggestion IDs as opaque AI-owned identifiers.**
5. **Keep original image storage in the product backend/object store.**
6. **Run consolidation as a background operation, not inside the user's upload request.**
7. **Notify the AI service when a photo or account is permanently deleted.**
8. **Honor `Retry-After` for transient ownership/busy responses.**
9. **Do not share the AI SQLite file across independent hosts via generic network storage.**

If these rules are followed, the product backend remains fully decoupled from the implementation details of the Face Grouping pipeline.

---

## 26. Repository layout relevant to backend integration

The repository intentionally keeps the integration surface small:

```text
README.md                  # Main and authoritative backend integration contract
requirements.txt           # Single dependency/install entry point
pyproject.toml              # Package/build metadata only; no dependency list
.env.example               # Runtime environment configuration template
configs/                   # AI model and threshold configuration
src/face_grouping/         # AI core, application service, persistence, and HTTP API
tests/                     # AI regression/CI tests; not required by the backend at runtime
models/                    # Required model assets; IR-SE50 weights may be deployed separately
```

There is intentionally no separate `docs/` integration guide and there is exactly **one** dependency file: `requirements.txt`. `pyproject.toml` contains package/build metadata only and no dependencies. **When the AI API contract, lifecycle behavior, configuration, dependencies, or deployment requirements change, this README and `requirements.txt` (when dependencies change) must be updated in the same change.**

For the backend team, this README is the only integration document that should be treated as authoritative.
