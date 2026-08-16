"""Dataset scanning for the private final face-grouping validation.

Ground-truth contract
---------------------
``data/Gallery/<identity>/...`` is the only required input structure.

* A single-person photo appears in exactly one identity folder.
* A multi-person photo must be copied byte-for-byte into every visible
  benchmark identity's folder. The scanner SHA-256 hashes files, collapses
  those copies into one canonical gallery photo, and unions the folder labels.
* The production pipeline processes that canonical photo once, exactly like a
  real gallery would. Duplicate copies exist only to express ground truth.

No image bytes or raw identity names are written to the shareable report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Dict, Iterable, List, Set

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class SourceCopy:
    identity: str
    path: Path


@dataclass
class CanonicalPhoto:
    content_hash: str
    canonical_path: Path
    identities: Set[str] = field(default_factory=set)
    source_copies: List[SourceCopy] = field(default_factory=list)

    @property
    def photo_code(self) -> str:
        return f"IMG_{self.content_hash[:12].upper()}"


@dataclass
class GalleryIndex:
    gallery_dir: Path
    identities: List[str]
    identity_codes: Dict[str, str]
    photos: List[CanonicalPhoto]
    ignored_files: List[str]
    warnings: List[str]

    @property
    def photo_by_hash(self) -> Dict[str, CanonicalPhoto]:
        return {photo.content_hash: photo for photo in self.photos}


def _hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _identity_dirs(gallery_dir: Path) -> List[Path]:
    return sorted(
        [p for p in gallery_dir.iterdir() if p.is_dir() and not p.name.startswith(".")],
        key=lambda p: p.name.lower(),
    )


def scan_gallery(gallery_dir: str | Path) -> GalleryIndex:
    root = Path(gallery_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Gallery directory does not exist: {root}")

    dirs = _identity_dirs(root)
    if not dirs:
        raise ValueError(
            f"No identity folders found under {root}. Expected data/Gallery/<person>/images."
        )

    identities = [p.name for p in dirs]
    identity_codes = {identity: f"P{i:03d}" for i, identity in enumerate(identities, start=1)}

    grouped: Dict[str, CanonicalPhoto] = {}
    ignored_files: List[str] = []
    warnings: List[str] = []
    identity_image_counts: Dict[str, int] = {name: 0 for name in identities}

    for identity_dir in dirs:
        identity = identity_dir.name
        for path in sorted(identity_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.name.startswith("."):
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                ignored_files.append(str(path.relative_to(root)))
                continue

            identity_image_counts[identity] += 1
            digest = _hash_file(path)
            photo = grouped.get(digest)
            if photo is None:
                photo = CanonicalPhoto(
                    content_hash=digest,
                    canonical_path=path.resolve(),
                )
                grouped[digest] = photo
            photo.identities.add(identity)
            photo.source_copies.append(SourceCopy(identity=identity, path=path.resolve()))

    empty = [identity for identity, count in identity_image_counts.items() if count == 0]
    if empty:
        raise ValueError(f"Identity folder(s) contain no supported images: {', '.join(empty)}")

    photos = sorted(grouped.values(), key=lambda p: p.content_hash)
    if not photos:
        raise ValueError(f"No supported images found under {root}")

    # Helpful benchmark-quality warnings; these never change pipeline behavior.
    single_counts = {identity: 0 for identity in identities}
    for photo in photos:
        if len(photo.identities) == 1:
            single_counts[next(iter(photo.identities))] += 1

    low_anchor = [identity for identity, count in single_counts.items() if count < 2]
    if low_anchor:
        warnings.append(
            "Some identities have fewer than 2 single-person photos. Folder-only exact "
            "face-level metrics may have limited coverage for: "
            + ", ".join(identity_codes[i] for i in low_anchor)
        )

    duplicate_same_identity = []
    for photo in photos:
        by_identity: Dict[str, int] = {}
        for source in photo.source_copies:
            by_identity[source.identity] = by_identity.get(source.identity, 0) + 1
        if any(count > 1 for count in by_identity.values()):
            duplicate_same_identity.append(photo.photo_code)
    if duplicate_same_identity:
        warnings.append(
            f"{len(duplicate_same_identity)} canonical photo(s) are duplicated more than once "
            "inside the same identity folder; they are safely deduplicated by SHA-256."
        )

    return GalleryIndex(
        gallery_dir=root,
        identities=identities,
        identity_codes=identity_codes,
        photos=photos,
        ignored_files=ignored_files,
        warnings=warnings,
    )


def anonymized_identities(photo: CanonicalPhoto, identity_codes: Dict[str, str]) -> List[str]:
    return sorted(identity_codes[name] for name in photo.identities)
