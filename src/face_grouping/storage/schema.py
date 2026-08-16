"""SQLite schema for the local face-grouping persistence layer."""

SCHEMA_VERSION = 4

SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS clusters (
        cluster_id TEXT PRIMARY KEY,
        face_count INTEGER NOT NULL DEFAULT 0,
        is_user_confirmed INTEGER NOT NULL DEFAULT 0,
        has_manual_correction INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        last_updated_at TEXT NOT NULL,
        merged_into TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS photos (
        photo_id TEXT PRIMARY KEY,
        image_path TEXT NOT NULL UNIQUE,
        image_width INTEGER NOT NULL,
        image_height INTEGER NOT NULL,
        processing_status TEXT NOT NULL
            CHECK(processing_status IN ('processing', 'completed', 'failed')),
        processed_at TEXT,
        embedding_model_version TEXT NOT NULL DEFAULT 'legacy_unknown',
        config_version TEXT NOT NULL DEFAULT 'legacy_unknown',
        error_message TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS faces (
        face_id TEXT PRIMARY KEY,
        embedding BLOB NOT NULL,
        quality_score REAL NOT NULL,
        yaw_ratio REAL NOT NULL,
        cluster_id TEXT,
        is_manually_corrected INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        assignment_state TEXT NOT NULL DEFAULT 'confirmed'
            CHECK(assignment_state IN ('confirmed', 'ambiguous', 'unassigned', 'manual')),
        candidate_cluster_id TEXT,
        best_match_score REAL,
        second_best_cluster_id TEXT,
        second_best_score REAL,
        score_margin REAL,
        decision_threshold REAL,
        decision_reason TEXT,
        photo_id TEXT,
        face_index INTEGER,
        bbox_x1 REAL,
        bbox_y1 REAL,
        bbox_x2 REAL,
        bbox_y2 REAL,
        detection_score REAL,
        embedding_model_version TEXT NOT NULL DEFAULT 'legacy_unknown',
        config_version TEXT NOT NULL DEFAULT 'legacy_unknown',
        FOREIGN KEY (cluster_id) REFERENCES clusters(cluster_id),
        FOREIGN KEY (photo_id) REFERENCES photos(photo_id),
        UNIQUE(photo_id, face_index)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS exemplars (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cluster_id TEXT NOT NULL,
        bucket TEXT NOT NULL CHECK(bucket IN ('quality', 'pose')),
        face_id TEXT,
        embedding BLOB NOT NULL,
        quality_score REAL NOT NULL,
        yaw_ratio REAL NOT NULL,
        embedding_model_version TEXT NOT NULL DEFAULT 'legacy_unknown',
        FOREIGN KEY (cluster_id) REFERENCES clusters(cluster_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS suggestions (
        suggestion_id TEXT PRIMARY KEY,
        suggestion_type TEXT NOT NULL CHECK(suggestion_type IN ('merge', 'split')),
        cluster_ids TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        created_at TEXT NOT NULL,
        payload_json TEXT NOT NULL DEFAULT '{}',
        evidence_json TEXT NOT NULL DEFAULT '{}',
        updated_at TEXT,
        resolved_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cluster_cannot_links (
        cluster_a_id TEXT NOT NULL,
        cluster_b_id TEXT NOT NULL,
        reason TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (cluster_a_id, cluster_b_id),
        CHECK(cluster_a_id <> cluster_b_id),
        FOREIGN KEY (cluster_a_id) REFERENCES clusters(cluster_id) ON DELETE CASCADE,
        FOREIGN KEY (cluster_b_id) REFERENCES clusters(cluster_id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_faces_cluster_id ON faces(cluster_id)",
    "CREATE INDEX IF NOT EXISTS idx_faces_assignment_state ON faces(assignment_state)",
    "CREATE INDEX IF NOT EXISTS idx_faces_candidate_cluster_id ON faces(candidate_cluster_id)",
    "CREATE INDEX IF NOT EXISTS idx_faces_photo_id ON faces(photo_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_faces_photo_face_index ON faces(photo_id, face_index) WHERE photo_id IS NOT NULL AND face_index IS NOT NULL",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_photos_image_path ON photos(image_path)",
    "CREATE INDEX IF NOT EXISTS idx_exemplars_cluster_id ON exemplars(cluster_id)",
    "CREATE INDEX IF NOT EXISTS idx_suggestions_status ON suggestions(status)",
    "CREATE INDEX IF NOT EXISTS idx_cannot_links_b ON cluster_cannot_links(cluster_b_id)",
]
