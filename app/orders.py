from __future__ import annotations

import json
import re
from pathlib import Path
from .models import PublicOrderStatus

ORDER_ID = re.compile(r"^ORD[- ]?(\d{4})$", re.I)


class OrderStore:
    def __init__(self, path: Path) -> None:
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.snapshot_at = raw["snapshot_at"]
        self._orders = {row["order_id"]: row for row in raw["orders"]}

    @staticmethod
    def normalize(order_id: str) -> str | None:
        cleaned = order_id.strip().upper().strip(".,!?:;()[]{}")
        match = ORDER_ID.fullmatch(cleaned)
        return f"ORD-{match.group(1)}" if match else None

    def lookup_order(self, order_id: str) -> PublicOrderStatus | None:
        normalized = self.normalize(order_id)
        if not normalized or normalized not in self._orders:
            return None
        row = self._orders[normalized]
        # Whitelist explicitly; neither customer nor internal objects can leak by schema evolution.
        safe = {key: row.get(key) for key in PublicOrderStatus.model_fields}
        safe["items"] = [{key: item[key] for key in ("name", "quantity", "final_sale")} for item in row["items"]]
        if row["status"] in {"cancelled", "returned"}:
            for key in ("shipped_at", "delivered_at", "carrier", "tracking_number", "estimated_delivery"):
                safe[key] = None
        return PublicOrderStatus.model_validate(safe)
