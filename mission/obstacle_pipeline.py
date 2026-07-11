"""Mission pipeline placeholder preserving the Phase 1 router boundary."""

from __future__ import annotations


def run(*args, **kwargs):
    """Run this mission pipeline.

    Full mission execution remains in the existing legacy modules until later
    migration phases wire those implementations into this architecture.
    """

    raise NotImplementedError("Pipeline execution is not wired in this migration phase")
