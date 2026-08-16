"""SQLite schema for tenant-isolated face-grouping persistence."""

SCHEMA_VERSION = 7
LEGACY_USER_ID = "__legacy__"

SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS clusters (
        user_id TEXT NOT NULL,
        cluster_id TEXT NOT NULL,
        face_count INTEGER NOT NULL DEFAULT 0,
        is_user_confirmed INTEGER NOT NULL DEFAULT 0,
        has_manual_correction INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        last_updated_at TEXT NOT NULL,
        merged_into TEXT,
        PRIMARY KEY (user_id, cluster_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS photos (
        user_id TEXT NOT NULL,
        photo_id TEXT NOT NULL,
        image_path TEXT NOT NULL,
        image_width INTEGER NOT NULL,
        image_height INTEGER NOT NULL,
        processing_status TEXT NOT NULL
            CHECK(processing_status IN ('processing', 'completed', 'failed')),
        processed_at TEXT,
        embedding_model_version TEXT NOT NULL DEFAULT 'legacy_unknown',
        config_version TEXT NOT NULL DEFAULT 'legacy_unknown',
        error_message TEXT,
        processing_started_at TEXT,
        processing_token TEXT,
        processing_attempts INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (user_id, photo_id),
        UNIQUE(user_id, image_path)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS faces (
        user_id TEXT NOT NULL,
        face_id TEXT NOT NULL,
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
        recognition_restricted INTEGER NOT NULL DEFAULT 0,
        recognition_restriction_reason TEXT,
        PRIMARY KEY (user_id, face_id),
        FOREIGN KEY (user_id, cluster_id)
            REFERENCES clusters(user_id, cluster_id),
        FOREIGN KEY (user_id, photo_id)
            REFERENCES photos(user_id, photo_id),
        UNIQUE(user_id, photo_id, face_index)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS exemplars (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        cluster_id TEXT NOT NULL,
        bucket TEXT NOT NULL CHECK(bucket IN ('quality', 'pose')),
        face_id TEXT,
        embedding BLOB NOT NULL,
        quality_score REAL NOT NULL,
        yaw_ratio REAL NOT NULL,
        embedding_model_version TEXT NOT NULL DEFAULT 'legacy_unknown',
        FOREIGN KEY (user_id, cluster_id)
            REFERENCES clusters(user_id, cluster_id) ON DELETE CASCADE,
        FOREIGN KEY (user_id, face_id)
            REFERENCES faces(user_id, face_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS suggestions (
        user_id TEXT NOT NULL,
        suggestion_id TEXT NOT NULL,
        suggestion_type TEXT NOT NULL CHECK(suggestion_type IN ('merge', 'split')),
        cluster_ids TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        created_at TEXT NOT NULL,
        payload_json TEXT NOT NULL DEFAULT '{}',
        evidence_json TEXT NOT NULL DEFAULT '{}',
        updated_at TEXT,
        resolved_at TEXT,
        PRIMARY KEY (user_id, suggestion_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cluster_cannot_links (
        user_id TEXT NOT NULL,
        cluster_a_id TEXT NOT NULL,
        cluster_b_id TEXT NOT NULL,
        reason TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (user_id, cluster_a_id, cluster_b_id),
        CHECK(cluster_a_id <> cluster_b_id),
        FOREIGN KEY (user_id, cluster_a_id)
            REFERENCES clusters(user_id, cluster_id) ON DELETE CASCADE,
        FOREIGN KEY (user_id, cluster_b_id)
            REFERENCES clusters(user_id, cluster_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS user_lifecycle (
        user_id TEXT PRIMARY KEY,
        photos_since_consolidation INTEGER NOT NULL DEFAULT 0,
        last_photo_completed_at TEXT,
        last_consolidated_at TEXT,
        count_due_since TEXT,
        consolidation_started_at TEXT,
        consolidation_token TEXT,
        consolidation_attempts INTEGER NOT NULL DEFAULT 0,
        last_consolidation_error TEXT,
        last_consolidation_error_at TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_clusters_user_active ON clusters(user_id, merged_into)",
    "CREATE INDEX IF NOT EXISTS idx_faces_user_cluster_id ON faces(user_id, cluster_id)",
    "CREATE INDEX IF NOT EXISTS idx_faces_user_assignment_state ON faces(user_id, assignment_state)",
    "CREATE INDEX IF NOT EXISTS idx_faces_user_candidate_cluster_id ON faces(user_id, candidate_cluster_id)",
    "CREATE INDEX IF NOT EXISTS idx_faces_user_photo_id ON faces(user_id, photo_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_faces_user_photo_face_index ON faces(user_id, photo_id, face_index) WHERE photo_id IS NOT NULL AND face_index IS NOT NULL",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_photos_user_image_path ON photos(user_id, image_path)",
    "CREATE INDEX IF NOT EXISTS idx_photos_user_processing_status ON photos(user_id, processing_status)",
    "CREATE INDEX IF NOT EXISTS idx_exemplars_user_cluster_id ON exemplars(user_id, cluster_id)",
    "CREATE INDEX IF NOT EXISTS idx_suggestions_user_status ON suggestions(user_id, status)",
    "CREATE INDEX IF NOT EXISTS idx_cannot_links_user_b ON cluster_cannot_links(user_id, cluster_b_id)",
]
