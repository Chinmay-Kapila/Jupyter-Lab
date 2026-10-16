# Gadgets360 AI Assistant Backend

Production-ready FastAPI backend powering an AI assistant scoped strictly
to Gadgets360 content: phones, laptops, tablets, smartwatches, TVs, AI &
technology, reviews, comparisons, buying guides, news, and finance topics
(crypto, gold, silver, petrol, diesel, stocks, loans, banking).

## Architecture

```
User
  -> FastAPI Backend (/api/chat)
    -> Pre-filter (keyword-based, rejects obviously out-of-scope requests
       BEFORE any LLM call)
    -> LLM Query Parser (Gemini) - returns ONLY structured JSON, never a
       direct answer to the user
    -> Domain Validation Layer - accepts ONLY Gadgets360-supported topics;
       rejected requests never reach Gemini for a response
    -> API Orchestrator - routes to Products / Reviews / News / Price /
       Search API clients based on parsed intent
    -> Optimization - if the upstream API data is sufficient on its own
       (e.g. "what's the current gold price"), the answer is formatted
       directly WITHOUT calling Gemini again
    -> Prompt Builder - assembles a clean, minimal, API-data-only context
    -> Response Generator (Gemini) - generates a concise markdown answer
       grounded strictly in the provided API data (only when reasoning /
       summarization / comparison is actually required)
  -> Response: { answer, product_cards, article_cards, related_links, metadata }
```

The backend NEVER lets Gemini invent product cards or links -- those are
always attached separately from raw API data. Gemini only ever writes the
narrative markdown answer.

## Project Structure

```
backend/
  app/
    config/          # Settings and domain constants
    models/          # Pydantic schemas
    routes/          # FastAPI routers (chat, health)
    services/        # Business logic (LLM, orchestrator, cache, session, validator)
    api_clients/      # Upstream Gadgets360 API clients (products, reviews, news, price, search)
    utils/           # Logger, helper functions
    prompts/         # Prompt templates for the parser and response generator
  main.py            # FastAPI app entrypoint
  requirements.txt
  .env.example
  .gitignore
```

## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Copy the example environment file and fill in your keys:

   ```bash
   cp .env.example .env
   ```

   At minimum, set `GEMINI_API_KEY` to a valid Gemini API key for
   development/testing. If `GADGETS360_*_API_BASE` upstream endpoints are
   unreachable (e.g. no real credentials yet), the API clients gracefully
   fall back to a small local sample dataset so the full pipeline remains
   testable end-to-end.

3. Run the development server:

   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

4. Open the interactive API docs at `http://localhost:8000/docs`.

## API

### `POST /api/chat`

Request body:

```json
{
  "session_id": "optional-existing-session-id",
  "message": "Suggest 5 phones under 40000 with a good camera"
}
```

Response body:

```json
{
  "session_id": "sess_...",
  "answer": "Markdown-formatted answer text",
  "format": "markdown",
  "product_cards": [ { "id": "...", "name": "...", "price": 24999, "...": "..." } ],
  "article_cards": [],
  "related_links": [ { "title": "...", "url": "..." } ],
  "metadata": {
    "intent": "recommendation",
    "entity": "phone",
    "used_gemini": true,
    "source_apis": ["products"],
    "cached": false,
    "generated_at": "2026-07-08T00:00:00Z"
  }
}
```

Out-of-scope requests (essays, homework, coding help, translation, story
writing, personal advice, resumes, emails, etc.) are rejected before any
Gemini call is made and return:

```json
{
  "session_id": "sess_...",
  "answer": "I'm the Gadgets360 AI Assistant, so I can only help with ...",
  "format": "markdown",
  "rejected": true,
  "reason": "..."
}
```

### `GET /api/health`

Simple liveness/readiness check.

## Session Handling

There is no login system and no database. Each conversation is tracked
by a `session_id` returned in the first response, kept in an in-memory
store, and holds only the last 5 messages. Sessions are purged
automatically after a period of inactivity by a background cleanup task.

## Caching

- Parser JSON output is cached (per message + history hash) to avoid
  re-parsing identical queries.
- Common, non-personalized API responses (product searches, reviews,
  news, finance rates) are cached briefly to reduce upstream load.
- Personalized responses are never cached.

## Extending to a New LLM Provider

`LLMService` (in `app/services/llm_service.py`) is an abstract interface
with `parse_query()` and `generate_response()`. `GeminiService` is the
current implementation. To add a new provider, implement the same
interface and swap the instantiation in `app/routes/chat.py` and
`app/services/orchestrator.py`.
