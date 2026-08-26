from app.agent import SupportAgent
from app.models import DecisionState

def test_cancellation_is_explained_but_never_claimed_completed():
    r = SupportAgent().respond("Please cancel ORD-1001")
    assert r.state == DecisionState.HANDOFF
    assert "within the 30-minute request window" in r.answer
    assert "can’t complete" in r.answer
    assert r.tool_calls[0].arguments == {"order_id": "ORD-1001"}
