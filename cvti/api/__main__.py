"""Run the Argus Engine API.

  python -m cvti.api --db runs/site/events.db --site configs/site_live.json
  python -m cvti.api --mock          # canned data for the frontend, no engine
"""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description="Argus Engine API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--db", default="runs/site/events.db",
                   help="path to the feed's events.db (its parent holds gate_health.json)")
    p.add_argument("--site", default="configs/site_live.json",
                   help="path to the site config")
    p.add_argument("--mock", action="store_true",
                   help="serve canned data — no engine, no cameras")
    p.add_argument("--webui-dir", default="",
                   help="serve a built web UI (Frontend/dist) at / , same-origin "
                        "with the API; also settable via ARGUS_WEBUI_DIR")
    args = p.parse_args()

    import uvicorn

    if args.mock:
        from cvti.api.mock import create_mock_app
        app = create_mock_app()
        print(f"[argus-api] MOCK server on http://{args.host}:{args.port}{'/api/v1'} "
              "— any username/password works")
    else:
        from cvti.api.app import create_app
        app = create_app(db_path=args.db, site_path=args.site,
                         webui_dir=args.webui_dir)
        print(f"[argus-api] on http://{args.host}:{args.port}/api/v1  "
              f"(db={args.db}, site={args.site})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
