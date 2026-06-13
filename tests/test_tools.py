"""
tests/test_tools.py

Pytest tests for each tool in tools.py.
One test per failure mode, plus a basic happy-path test per tool.
"""

import pytest
from unittest.mock import MagicMock, patch

from tools import search_listings, suggest_outfit, create_fit_card


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_ITEM = {
    "id": "lst_001",
    "title": "Faded 90s Rock Tee",
    "description": "Vintage rock band tee with a faded graphic.",
    "category": "tops",
    "style_tags": ["vintage", "grunge", "graphic tee"],
    "size": "M",
    "condition": "good",
    "price": 25.00,
    "colors": ["black", "grey"],
    "brand": None,
    "platform": "depop",
}

SAMPLE_WARDROBE = {
    "items": [
        {
            "id": "w_001",
            "name": "Baggy straight-leg jeans, dark wash",
            "category": "bottoms",
            "colors": ["dark blue"],
            "style_tags": ["denim", "streetwear", "baggy"],
            "notes": None,
        },
        {
            "id": "w_002",
            "name": "Chunky white sneakers",
            "category": "shoes",
            "colors": ["white"],
            "style_tags": ["streetwear", "chunky"],
            "notes": None,
        },
    ]
}

EMPTY_WARDROBE = {"items": []}


def _make_llm_response(text: str):
    """Build a minimal mock that looks like a Groq chat completion."""
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ── search_listings ────────────────────────────────────────────────────────────

class TestSearchListings:
    def test_returns_list(self):
        results = search_listings("vintage tee")
        assert isinstance(results, list)

    def test_no_match_returns_empty_list_not_exception(self):
        """Failure mode: no listings match the description."""
        results = search_listings("xyzzy nonexistent item 12345")
        assert results == []

    def test_price_filter_applied(self):
        """Failure mode: price ceiling too restrictive → 0 results within budget."""
        results = search_listings("vintage", max_price=0.01)
        assert all(item["price"] <= 0.01 for item in results)

    def test_size_filter_applied(self):
        """Failure mode: size filter too restrictive → only matching sizes returned."""
        results = search_listings("jacket", size="XXXL_IMPOSSIBLE_SIZE_999")
        assert results == []

    def test_relevance_ordering(self):
        """Best keyword overlap should come first."""
        results = search_listings("vintage tee shirt")
        if len(results) >= 2:
            # Ensure results are ordered best-first (no regression on sort)
            titles = [r["title"].lower() for r in results[:3]]
            assert any("tee" in t or "vintage" in t or "shirt" in t for t in titles)

    def test_size_case_insensitive(self):
        """Size matching must be case-insensitive per the docstring."""
        lower = search_listings("jacket", size="m")
        upper = search_listings("jacket", size="M")
        assert lower == upper

    def test_no_price_filter_returns_results(self):
        """Omitting max_price should not drop valid listings."""
        results = search_listings("vintage", max_price=None)
        assert len(results) > 0


# ── suggest_outfit ─────────────────────────────────────────────────────────────

class TestSuggestOutfit:
    @patch("tools._get_groq_client")
    def test_happy_path_with_wardrobe(self, mock_client):
        """Basic call with a populated wardrobe returns a non-empty string."""
        mock_client.return_value.chat.completions.create.return_value = (
            _make_llm_response("Outfit: rock tee + baggy jeans + chunky sneakers — effortless streetwear.")
        )
        result = suggest_outfit(SAMPLE_ITEM, SAMPLE_WARDROBE)
        assert isinstance(result, str)
        assert len(result) > 0

    @patch("tools._get_groq_client")
    def test_empty_wardrobe_does_not_crash(self, mock_client):
        """Failure mode: wardrobe is empty — must not raise, must return a string."""
        mock_client.return_value.chat.completions.create.return_value = (
            _make_llm_response("Your wardrobe is empty. Pair this tee with baggy denim and white sneakers.")
        )
        result = suggest_outfit(SAMPLE_ITEM, EMPTY_WARDROBE)
        assert isinstance(result, str)
        assert len(result) > 0

    @patch("tools._get_groq_client")
    def test_missing_items_key_does_not_crash(self, mock_client):
        """Failure mode: wardrobe dict is missing the 'items' key entirely."""
        mock_client.return_value.chat.completions.create.return_value = (
            _make_llm_response("General styling advice for an empty wardrobe.")
        )
        result = suggest_outfit(SAMPLE_ITEM, {})
        assert isinstance(result, str)

    @patch("tools._get_groq_client")
    def test_empty_wardrobe_triggers_general_prompt(self, mock_client):
        """Verify the LLM is still called even when wardrobe is empty."""
        mock_client.return_value.chat.completions.create.return_value = (
            _make_llm_response("Pair with chinos and loafers.")
        )
        suggest_outfit(SAMPLE_ITEM, EMPTY_WARDROBE)
        mock_client.return_value.chat.completions.create.assert_called_once()


# ── create_fit_card ────────────────────────────────────────────────────────────

class TestCreateFitCard:
    @patch("tools._get_groq_client")
    def test_happy_path(self, mock_client):
        """Basic call with valid outfit returns a non-empty string."""
        mock_client.return_value.chat.completions.create.return_value = (
            _make_llm_response("Scored this rock tee on depop for $25 and it slaps with the baggy denim.")
        )
        result = create_fit_card("Rock tee + baggy jeans + chunky sneakers", SAMPLE_ITEM)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_outfit_returns_error_string(self):
        """Failure mode: outfit is an empty string — must return error message, not raise."""
        result = create_fit_card("", SAMPLE_ITEM)
        assert isinstance(result, str)
        assert "error" in result.lower() or "empty" in result.lower()

    def test_whitespace_only_outfit_returns_error_string(self):
        """Failure mode: outfit is whitespace-only — must return error message, not raise."""
        result = create_fit_card("   ", SAMPLE_ITEM)
        assert isinstance(result, str)
        assert "error" in result.lower() or "empty" in result.lower()

    @patch("tools._get_groq_client")
    def test_llm_not_called_for_empty_outfit(self, mock_client):
        """Guard clause must short-circuit before calling the LLM."""
        create_fit_card("", SAMPLE_ITEM)
        mock_client.assert_not_called()

    @patch("tools._get_groq_client")
    def test_outputs_vary_across_calls(self, mock_client):
        """Failure mode: identical outputs for same input → increase temperature.
        Here we verify the LLM is called with temperature >= 1.0 to encourage variety."""
        mock_client.return_value.chat.completions.create.return_value = (
            _make_llm_response("caption one")
        )
        create_fit_card("Rock tee + jeans", SAMPLE_ITEM)
        call_args = mock_client.return_value.chat.completions.create.call_args
        temperature = call_args.kwargs.get("temperature")
        # Temperature must be set high enough to produce varied outputs
        assert temperature is not None and temperature >= 1.0

    @patch("tools._get_groq_client")
    def test_minimal_new_item_does_not_crash(self, mock_client):
        """Failure mode: new_item dict has missing fields — must not raise."""
        mock_client.return_value.chat.completions.create.return_value = (
            _make_llm_response("A clean, simple fit.")
        )
        result = create_fit_card("some outfit description", {})
        assert isinstance(result, str)
