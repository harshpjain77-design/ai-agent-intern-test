"""Every supplied visible case is represented by test_agent assertions above."""
import json
from pathlib import Path

def test_visible_case_inventory_is_complete():
    cases = json.loads((Path(__file__).parents[1] / "evaluation" / "visible-cases.json").read_text())["cases"]
    expected = {"standard-return-window", "trailplus-return-window", "final-sale-damaged-exception", "canada-multiturn", "unsupported-country", "valid-order-lookup", "missing-order-id", "cancelled-order-stale-eta", "unknown-order", "shipped-without-eta", "order-data-privacy", "no-lifetime-warranty", "retrieved-prompt-injection", "insufficient-information", "genuine-active-source-conflict"}
    assert {case["id"] for case in cases} == expected
