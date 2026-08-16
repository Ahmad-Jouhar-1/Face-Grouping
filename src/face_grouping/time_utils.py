"""Time helpers used by the persistence/domain layer.

The existing database stores naive ISO-8601 timestamps that semantically mean
UTC.  Keep that representation for backward compatibility while avoiding the
deprecated ``datetime.utcnow()`` API.
"""
from datetime import datetime, timezone


def utcnow_naive() -> datetime:
    """Return the current UTC time as a naive ``datetime``.

    This preserves the timestamp semantics/schema used by earlier versions of
    the project without relying on ``datetime.utcnow()``.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)
