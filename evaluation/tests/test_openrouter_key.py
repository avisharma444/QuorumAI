from councileval.openrouter_key import evaluate


def test_key_limit_rules():
    assert evaluate({"limit": 2.0, "limit_remaining": 2.0}, 2.0, 2.0) == []
    assert "no spending limit" in evaluate({"limit": None}, 2.0, 2.0)[0]
    assert "above the approved" in evaluate({"limit": 5.0, "limit_remaining": 5.0}, 2.0, 2.0)[0]
    assert "remains" in evaluate({"limit": 2.0, "limit_remaining": 0.5}, 2.0, 2.0)[0]
