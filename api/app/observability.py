from __future__ import annotations

import json
import logging
from typing import Any


logger = logging.getLogger("agent_boundary")
logging.basicConfig(level=logging.INFO)


def log_event(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    logger.info(json.dumps(payload, sort_keys=True, default=str))
