"""
tools.py

The three required FitFindr tools. Each tool is a standalone function that
can be called and tested independently before being wired into the agent loop.

Complete and test each tool before moving to agent.py.

Tools:
    search_listings(description, size, max_price)  → list[dict]
    suggest_outfit(new_item, wardrobe)              → str
    create_fit_card(outfit, new_item)               → str
"""

import os

from dotenv import load_dotenv
from groq import Groq

from utils.data_loader import load_listings

load_dotenv()


# ── Groq client ───────────────────────────────────────────────────────────────

def _get_groq_client():
    """Initialize and return a Groq client using GROQ_API_KEY from .env."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set. Add it to a .env file in the project root."
        )
    return Groq(api_key=api_key)


# ── Tool 1: search_listings ───────────────────────────────────────────────────

def search_listings(
    description: str,
    size: str | None = None,
    max_price: float | None = None,
) -> list[dict]:
    """
    Search the mock listings dataset for items matching the description,
    optional size, and optional price ceiling.

    Args:
        description: Keywords describing what the user is looking for
                     (e.g., "vintage graphic tee").
        size:        Size string to filter by, or None to skip size filtering.
                     Matching is case-insensitive (e.g., "M" matches "S/M").
        max_price:   Maximum price (inclusive), or None to skip price filtering.

    Returns:
        A list of matching listing dicts, sorted by relevance (best match first).
        Returns an empty list if nothing matches — does NOT raise an exception.

    Each listing dict has the following fields:
        id, title, description, category, style_tags (list), size,
        condition, price (float), colors (list), brand, platform

    TODO:
        1. Load all listings with load_listings().
        2. Filter by max_price and size (if provided).
        3. Score each remaining listing by keyword overlap with `description`.
        4. Drop any listings with a score of 0 (no relevant matches).
        5. Sort by score, highest first, and return the listing dicts.

    Before writing code, fill in the Tool 1 section of planning.md.
    """
    # Step 1: Load all listings from the dataset
    all_listings = load_listings()
    
    # Step 2: Filter by max_price (if provided)
    if max_price is not None:
        all_listings = [item for item in all_listings if item.get("price", float("inf")) <= max_price]
    
    # Step 3: Filter by size (if provided)
    # Size matching is case-insensitive and allows partial matches (e.g., "M" matches "S/M")
    if size is not None:
        size_upper = size.upper()
        all_listings = [
            item for item in all_listings 
            if size_upper in item.get("size", "").upper()
        ]
    
    # Step 4: Score each remaining listing by keyword overlap with description
    # Keywords come from title, description, style_tags, and category
    description_keywords = set(description.lower().split())
    
    scored_listings = []
    for item in all_listings:
        # Combine searchable fields
        searchable_text = " ".join([
            item.get("title", "").lower(),
            item.get("description", "").lower(),
            item.get("category", "").lower(),
            " ".join(item.get("style_tags", [])).lower(),
        ])
        
        # Count keyword matches
        searchable_keywords = set(searchable_text.split())
        score = len(description_keywords & searchable_keywords)
        
        # Step 5: Drop any listings with a score of 0 (no relevant matches)
        if score > 0:
            scored_listings.append((score, item))
    
    # Step 6: Sort by score (highest first) and extract just the listing dicts
    scored_listings.sort(key=lambda x: x[0], reverse=True)
    results = [item for score, item in scored_listings]
    
    return results

# ── Tool 2: suggest_outfit ────────────────────────────────────────────────────

def suggest_outfit(new_item: dict, wardrobe: dict) -> str:
    """
    Given a thrifted item and the user's wardrobe, suggest 1–2 complete outfits.

    Args:
        new_item: A listing dict (the item the user is considering buying).
        wardrobe: A wardrobe dict with an 'items' key containing a list of
                  wardrobe item dicts. May be empty — handle this gracefully.

    Returns:
        A non-empty string with outfit suggestions.
        If the wardrobe is empty, offer general styling advice for the item
        rather than raising an exception or returning an empty string.

    TODO:
        1. Check whether wardrobe['items'] is empty.
        2. If empty: call the LLM with a prompt for general styling ideas
           (what kinds of items pair well, what vibe it suits, etc.).
        3. If not empty: format the wardrobe items into a prompt and ask
           the LLM to suggest specific outfit combinations using the new item
           and named pieces from the wardrobe.
        4. Return the LLM's response as a string.

    Before writing code, fill in the Tool 2 section of planning.md.
    """
    client = _get_groq_client()

    item_summary = (
        f"{new_item.get('title', 'Unknown item')} "
        f"(${new_item.get('price', '?')}, {new_item.get('condition', '?')} condition, "
        f"colors: {', '.join(new_item.get('colors', []))}, "
        f"style: {', '.join(new_item.get('style_tags', []))})"
    )

    wardrobe_items = wardrobe.get("items", [])

    if not wardrobe_items:
        prompt = (
            f"A user just found this thrifted item: {item_summary}.\n\n"
            "Their wardrobe is empty. Suggest 1-2 complete outfits they could build "
            "around this piece. Recommend specific complementary items (bottoms, shoes, "
            "accessories) they should look for. Keep the advice practical, specific, and "
            "style-forward. No filler."
        )
    else:
        wardrobe_lines = "\n".join(
            f"- {w.get('name', 'unknown')} ({w.get('category', '?')}, "
            f"colors: {', '.join(w.get('colors', []))}, "
            f"style: {', '.join(w.get('style_tags', []))})"
            for w in wardrobe_items
        )
        prompt = (
            f"A user just found this thrifted item: {item_summary}.\n\n"
            f"Their existing wardrobe:\n{wardrobe_lines}\n\n"
            "Suggest 1-2 complete outfit combinations using the new item with specific "
            "pieces from their wardrobe. Name the exact wardrobe pieces you're pairing. "
            "Include the style vibe and a brief explanation of why each outfit works. "
            "Be specific and practical, not generic."
        )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content


# ── Tool 3: create_fit_card ───────────────────────────────────────────────────

def create_fit_card(outfit: str, new_item: dict) -> str:
    """
    Generate a short, shareable outfit caption for the thrifted find.

    Args:
        outfit:   The outfit suggestion string from suggest_outfit().
        new_item: The listing dict for the thrifted item.

    Returns:
        A 2–4 sentence string usable as an Instagram/TikTok caption.
        If outfit is empty or missing, return a descriptive error message
        string — do NOT raise an exception.

    The caption should:
    - Feel casual and authentic (like a real OOTD post, not a product description)
    - Mention the item name, price, and platform naturally (once each)
    - Capture the outfit vibe in specific terms
    - Sound different each time for different inputs (use higher LLM temperature)

    TODO:
        1. Guard against an empty or whitespace-only outfit string.
        2. Build a prompt that gives the LLM the item details and the outfit,
           and asks for a caption matching the style guidelines above.
        3. Call the LLM and return the response.

    Before writing code, fill in the Tool 3 section of planning.md.
    """
    if not outfit or not outfit.strip():
        return (
            "Error: outfit description is empty — cannot generate a fit card. "
            "Run suggest_outfit first to build an outfit, then pass the result here."
        )

    item_name = new_item.get("title", "this piece")
    item_price = new_item.get("price", "?")
    item_platform = new_item.get("platform", "a thrift app")

    prompt = (
        f"Write a 2-4 sentence Instagram/TikTok caption for this outfit:\n\n"
        f"Featured thrift find: {item_name} for ${item_price} from {item_platform}\n"
        f"Outfit: {outfit}\n\n"
        "Rules:\n"
        "- Casual and authentic — sounds like a real OOTD post, not an ad\n"
        "- Mention the item name, price, and platform naturally (once each)\n"
        "- Capture the specific vibe of the outfit\n"
        "- No hashtags, no emojis unless they feel natural\n"
        "- No marketing language or overselling\n"
        "Just write the caption, nothing else."
    )

    client = _get_groq_client()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=1.2,
    )
    return response.choices[0].message.content
