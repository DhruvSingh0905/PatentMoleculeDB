"""Remove stale output directories from pre-versioning era.

Safe to run after migrating to output/results/ canonical layout.
Preserves: cache/, gpatents_cache/, bindingdb/, checkpoints/, images/, logs/, results/
"""
import shutil
from pathlib import Path

OUTPUT = Path("output")

STALE_DIRS = [
    *[f"v{i}" for i in range(1, 16)],
    "final",
    "final_v2",
    "google_patents_v1",
    "decimer_v1",
    "scaffold_cache",
]

KEEP = {"cache", "gpatents_cache", "bindingdb", "checkpoints", "images",
        "logs", "results", "bigquery", "fragment_vocab.json"}

removed = 0
for name in STALE_DIRS:
    d = OUTPUT / name
    if d.exists():
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        print(f"  Removing {d} ({size/1024:.0f} KB)")
        shutil.rmtree(d)
        removed += 1

# Also remove stale root-level CSV exports (they'll be regenerated)
for csv in OUTPUT.glob("PatentMoleculeDB_*.csv"):
    print(f"  Removing {csv.name}")
    csv.unlink()
    removed += 1

for csv in OUTPUT.glob("ic50_*.csv"):
    print(f"  Removing {csv.name}")
    csv.unlink()
    removed += 1

if removed:
    print(f"\nCleaned {removed} stale artifacts.")
else:
    print("Nothing to clean.")

# Verify KEEP directories exist
print("\nActive directories:")
for name in sorted(KEEP):
    d = OUTPUT / name
    if d.exists():
        print(f"  ✓ {d}")
    elif name.endswith('.json'):
        f = OUTPUT / name
        if f.exists():
            print(f"  ✓ {f}")
