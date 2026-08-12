"""What a fetched job posting carries back from the network.

The pipeline only needs the posting's text. The rest of these fields exist for
provenance: when a generated resume looks wrong, the first question is what was
actually read, and the second is where it came from. ``final_url`` answers the second
after redirects, which is not always the URL that was pasted.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class PostingBoard(StrEnum):
    """Which adapter produced a posting, recorded so a bad extraction is traceable."""

    GREENHOUSE = "greenhouse"
    LEVER = "lever"
    ASHBY = "ashby"
    WORKDAY = "workday"
    GENERIC = "generic"


class FetchedPosting(BaseModel):
    """A job posting retrieved from a URL, reduced to text.

    Frozen: the text handed to :class:`~resumelab.models.job.JobDescription` is the
    text that was fetched, so the run artifacts and the network response agree.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    """The posting body as plain text, assembled by the adapter that read it."""

    board: PostingBoard
    requested_url: str
    final_url: str
    """Where the content was actually read from, after redirects."""

    title: str | None = None
    company: str | None = None
    location: str | None = None

    @property
    def label(self) -> str:
        """A short human-readable name for this posting, for run directory names.

        Falls back through what the adapter managed to extract; a posting with no
        title at all still names its board rather than nothing.
        """
        parts = [part for part in (self.company, self.title) if part]
        return " ".join(parts) if parts else self.board.value
