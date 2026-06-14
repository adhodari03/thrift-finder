# FitFindr

FitFindr is a thrift-shopping agent that searches a secondhand marketplace dataset, suggests outfits using your wardrobe, and generates a shareable caption — all from a single natural language query.

## Setup

```bash
pip install -r requirements.txt
```

Add your Groq API key to a `.env` file (free key at [console.groq.com](https://console.groq.com)):
```
GROQ_API_KEY=your_key_here
```

Run the Gradio UI:
```bash
python app.py
```

Or test the agent directly:
```bash
python agent.py
```

---

## Tool Inventory

### `search_listings(description: str, size: str | None, max_price: float | None) → list[dict]`

Searches `data/listings.json` for items matching the description. Filters by `max_price` and `size` first (size matching is case-insensitive and supports partials — `"M"` matches `"S/M"`), then scores each remaining listing by keyword overlap across its title, description, category, and style tags. Listings with zero overlap are dropped. Returns results sorted best-match first, or an empty list if nothing matches — never raises.

### `suggest_outfit(new_item: dict, wardrobe: dict) → str`

Calls Groq (`llama-3.3-70b-versatile`) to generate 1–2 outfit combinations. If `wardrobe["items"]` is empty (or missing), the prompt shifts to general styling advice — what to look for to complement the item. If the wardrobe has items, the prompt names them explicitly and asks the LLM to pair specific pieces. Returns the LLM response as a string.

### `create_fit_card(outfit: str, new_item: dict) → str`

Calls Groq to write a 2–4 sentence OOTD caption. Guards first: if `outfit` is empty or whitespace, returns an error string immediately without touching the LLM. Otherwise builds a prompt with the item name, price, platform, and the outfit description, instructing the model to write casually — no hashtags, no marketing language. Uses `temperature=1.2` so captions vary meaningfully across runs.

---

## How the Planning Loop Works

`run_agent(query, wardrobe)` in `agent.py` runs four sequential steps, with a hard stop after step 2 if search comes up empty.

1. **Parse** — `_parse_query()` uses regex to extract `max_price` (e.g. `under $30`), `size` (e.g. `size M`, `XL`), and `description` (the remainder after stripping both). Falls back to the full query string if stripping removes everything.

2. **Search** — calls `search_listings()` with the parsed params. If the result list is empty, sets `session["error"]` with a message like `No listings found for "designer ballgown", size XXS under $5.0 — try broadening your search`, and **returns immediately**. `suggest_outfit` and `create_fit_card` are not called.

3. **Suggest** — takes the top-ranked listing as `selected_item` and calls `suggest_outfit()` with it and the user's wardrobe.

4. **Fit card** — passes the outfit string and selected item into `create_fit_card()`, stores the caption, and returns the complete session dict.

The conditional in step 2 is the only branch point. The agent never calls all three tools unconditionally.

---

## State Management

Each call to `run_agent()` initializes a fresh session dict:

```python
{
    "query":             # original user input
    "parsed":            # {description, size, max_price} from _parse_query()
    "search_results":    # full list returned by search_listings()
    "selected_item":     # search_results[0] — passed directly to suggest_outfit
    "wardrobe":          # the wardrobe dict passed in by the caller
    "outfit_suggestion": # string from suggest_outfit — passed directly to create_fit_card
    "fit_card":          # string from create_fit_card
    "error":             # non-None only if the loop halted early
}
```

Nothing is re-fetched or re-prompted between steps. `selected_item` is written once in step 2 and read in steps 3 and 4 from the same dict. `outfit_suggestion` is written in step 3 and read in step 4. Callers (e.g. `handle_query()` in `app.py`) inspect the returned session dict to decide what to show.

---

## Error Handling

| Tool | Failure mode | Behavior |
|---|---|---|
| `search_listings` | No keyword overlap | Returns `[]`; loop sets `session["error"]` and halts |
| `search_listings` | Price/size too restrictive | Same — empty list triggers the same halt with a specific message |
| `suggest_outfit` | Empty wardrobe | Switches to a "build around this piece" prompt; still returns a string |
| `suggest_outfit` | Missing `"items"` key | `wardrobe.get("items", [])` defaults to `[]`; same empty-wardrobe path |
| `create_fit_card` | Empty or whitespace outfit | Returns an error string before calling the LLM — no exception |

**Concrete example from testing:** Running `run_agent("designer ballgown size XXS under $5", ...)` produces `session["error"] = 'No listings found for "designer ballgown", size XXS under $5.0...'` and leaves `fit_card`, `outfit_suggestion`, and `selected_item` all as `None`. A pytest test in `tests/test_agent.py` patches `search_listings` to return `[]` and asserts `suggest_outfit` is never called.

---

## Spec Reflection

**Where the spec helped:** The error-handling table in `planning.md` was the most directly useful artifact. Having a row for each tool × failure mode meant I didn't have to reason about edge cases during implementation — I just translated each row into a guard clause or a branch condition.

**Where implementation diverged:** The spec described query parsing as "extract search criteria from the user's query" without committing to a method. The planning doc mentioned using the LLM as one option. In practice, a regex parser (`_parse_query()`) was faster, cheaper, and fully testable without mocking — the LLM approach would have added latency and a new failure surface just to extract two fields. The tradeoff is that the regex won't catch unusual phrasings like "no more than thirty dollars," but that's an acceptable gap for this dataset.

---

## AI Usage

**1. Implementing `suggest_outfit` and `create_fit_card`**

I directed Claude to implement both LLM-calling tools given the full spec blocks from `planning.md` (inputs, return values, failure modes) and the constraint that `suggest_outfit` must not crash on an empty wardrobe. Claude generated prompts that branched on wardrobe state and set `temperature=1.2` on `create_fit_card`. I reviewed and kept both decisions — the branching prompt structure matched the spec exactly, and the higher temperature was the right call for caption variety. I didn't need to revise either.

**2. Writing the test suite**

I directed Claude to write one pytest test per failure mode for all three tools, using `unittest.mock` to avoid real API calls. The generated tests were structurally correct but one (`test_outputs_vary_across_calls`) had a brittle kwarg extraction: it used `call_args.args[1]` as a fallback, which returned `None` on the actual call signature. I caught the failure in the pytest run and fixed it to read `call_args.kwargs.get("temperature")` directly — the correct way to inspect keyword-only arguments in a mock call.
