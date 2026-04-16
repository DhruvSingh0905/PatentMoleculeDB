# Patent Compound Extraction — Claude Context

## Obsidian Vault Sync

After every meaningful `git commit`, update the Obsidian vault at `~/Main/Projects/Patent Compound Extraction/`:

1. **Decision Log.md** — append a row if the commit represents an architectural decision
2. **Benchmarks.md** — update tables if recall/precision/cost numbers changed
3. **Open Problems.md** — update if blockers shifted or were resolved
4. **Concept pages** — update gotchas if a new bug was found in that area

Do NOT create new files for every commit. Keep pages concise and searchable.

## Project Structure
- `patent_extraction/` — all pipeline modules
- `output/` — extraction results, caches, benchmarks
- `test_set/` — mini benchmark + Markush test sets
- `venv/` — Python 3.11 with rdkit-pypi

## Key Commands
```bash
source venv/bin/activate
python3 run_markush_enum.py      # Markush enumeration + BDB benchmark
python3 run_mini_bench.py        # Fast benchmark (<2 min)
python3 export_csv.py            # Generate Jie's CSV format
pytest tests/                    # Run test suite
```
