"""`python3 -m patentdb3` — the front door. See `cli.py`.

Nothing lives here. Importing `cli` lazily inside `main` would buy nothing:
this module only ever runs as `__main__`, so the import always happens.
"""
from .cli import main

raise SystemExit(main())
