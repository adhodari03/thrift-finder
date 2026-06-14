"""
tests/test_agent.py

Tests for the run_agent() planning loop in agent.py.
Focused on the branch-path guarantee: suggest_outfit must never be called
when search_listings returns an empty list.
"""

from unittest.mock import patch

from agent import run_agent
from utils.data_loader import get_empty_wardrobe


def test_no_results_halts_before_suggest_outfit():
    """
    When search_listings returns nothing, run_agent must:
      - set session["error"] to a non-empty string
      - leave session["fit_card"] as None
      - never call suggest_outfit
    """
    with patch("agent.search_listings", return_value=[]):
        with patch("agent.suggest_outfit") as mock_suggest:
            session = run_agent("xyzzy impossible query", get_empty_wardrobe())

    assert session["error"] is not None
    assert session["fit_card"] is None
    mock_suggest.assert_not_called()
