# TSN Web Portal

This directory hosts the FastAPI-powered web experience that replaces the legacy
`kk7nqn.net` dashboard. It reuses the core TSN database so repeater activity is
visible through the same MySQL/MariaDB instance that drives ingestion.

## Features
- Public overview landing page with repeater info, activity snapshots, and vLLM
  summaries.
- Deep-dive pages for callsigns, club profiles, net sessions, transcriptions,
  and health metrics.
- JSON API endpoints mirroring every page to power future clients.
- User registration/login with hashed passwords, session cookies, and a private
  dashboard for saved filters (room for expansion).
- Shared SQLAlchemy models via `tsn_common` so all analytics stay consistent.

## Running the server
1. Ensure the main TSN dependencies are installed (FastAPI/uvicorn already live
   in `pyproject.toml`). Add `passlib[bcrypt]` if not yet installed.
2. Export environment variables (or copy `.env.example`):
   ```bash
   export TSN_DB_ENGINE=mysql
   export TSN_DB_HOST=...
   export TSN_WEB_SESSION_SECRET="super-secret"
   ```
3. Launch the app from the repo root:
   ```bash
   uvicorn web.main:app --reload
   ```
4. Visit http://localhost:8000/ to explore the dashboards.

## Directory layout
```
web/
  main.py            # FastAPI entrypoint
  config.py          # Web-specific settings
  dependencies.py    # Shared dependency wiring
  models/            # Portal-only models (PortalUser)
  services/          # Database query helpers + auth helpers
  routes/            # Page+API routers
  templates/         # Jinja2 templates for HTML pages
  static/            # CSS/JS assets
```

Future enhancements can add React/htmx interactivity or integrate websockets
for live net updates, but the current structure already exposes the entire TSN
knowledge graph through a modern portal.
