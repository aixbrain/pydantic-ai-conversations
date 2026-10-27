"""The feedback add-on's wire model, apart from the add-on and its routes so both can import it."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class TurnFeedbackEntryOut(BaseModel):
    """One author's entry on one turn. `data` is declared open here and narrowed to the app's
    payload model per add-on, so the class parses any add-on's entry and OpenAPI shows the
    app's fields."""

    run_id: str
    author_user_id: str
    created_at: datetime
    updated_at: datetime
    data: Any
