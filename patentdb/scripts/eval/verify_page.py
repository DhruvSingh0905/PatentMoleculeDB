"""Put the patent's own XML next to what we made of it, for hand-verification.

    python3 -m patentdb.scripts.eval.verify_page --patents US11649247,US9302989
    python3 -m patentdb.scripts.eval.verify_page --all --out docs/reports/verify.html

Every automated check here compares a number to another number. None of them
can look at a table and say "that is not what the patent means" — which is how
a dimensionless ratio spent a week recorded as a nanomolar potency while every
count rose. This renders the two things a person needs side by side: the raw
OASIS/CALS the patent published, and the records we derived from it, per table.

The page is self-contained (no network, no CDN) and shows, for each block:

  * the source XML, verbatim, with `colspec` widths visible — they state how a
    header tgroup maps onto a body tgroup of a different width, and getting
    that wrong cost US10376513 all 348 of its compounds
  * how we classified every column, since a wrong `kind` is invisible in the
    output but explains most of it
  * the records themselves, with the raw cell text each value came from

Records from the repair loop are marked, because they reached the page by a
different route than the deterministic parser and are worth more scrutiny.
"""
from __future__ import annotations

import argparse
import json
import sys

from ...core import config
from ...sources import uspto_assays as A
from ...sources import uspto_xml

# Per-table cap. A 699-row block runs to 428 KB and no one hand-verifies that;
# the shape is legible in the first rows and the closing tags.
_XML_CAP = 30_000
_ROW_CAP = 60


def _blocks(pid: str, xml: str, only_gaps: bool = False) -> list[dict]:
    raw = uspto_xml.parse_tables(xml)
    tables = {t.table_id: t for t in A._best_per_block(raw)}
    records = A.extract_from_patent(xml)
    try:
        from ...repair.loop import repair_patent
        # This returns baseline ∪ rules. The page marks the rule-recovered rows
        # specially, so subtract the baseline back out by (cid, assay, value)
        # rather than by identity — the loop re-extracts its own copies.
        all_records, _ = repair_patent(pid, xml, max_calls=0)
        base_keys = {(r.cid, r.assay_name, r.value_numeric) for r in records}
        repaired = [r for r in all_records
                    if (r.cid, r.assay_name, r.value_numeric) not in base_keys]
    except Exception:
        repaired = []
    by_block: dict[str, list] = {}
    for r in list(records) + list(repaired):
        by_block.setdefault(r.table_id, []).append(r)
    repaired_ids = {id(r) for r in repaired}

    from ...repair.gap import raw_block

    out = []
    for tid, t in tables.items():
        src = raw_block(xml, tid)
        truncated = len(src) > _XML_CAP
        if truncated:
            src = src[: _XML_CAP * 2 // 3] + \
                f"\n\n<!-- {len(src) - _XML_CAP} characters omitted -->\n\n" + \
                src[-(_XML_CAP // 3):]
        recs = by_block.get(tid, [])
        if only_gaps and recs:
            continue
        cols = A.build_columns(t)
        out.append({
            "id": tid,
            "xml": src,
            "header": A.merge_header(t),
            "columns": [{"i": c.index, "header": c.header, "kind": c.kind,
                         "unit": c.unit} for c in cols],
            "rows": [[c.text for c in r] for r in t.body_rows[:_ROW_CAP]],
            "n_rows": len(t.body_rows),
            "records": [{
                "cid": r.cid, "assay": r.assay_name, "value": r.value_numeric,
                "unit": r.unit, "qual": r.qualifier, "n": r.n_runs,
                "lo": r.range_lo, "hi": r.range_hi, "grade": r.letter_grade,
                "text": r.value_text, "src": r.source,
                "repaired": id(r) in repaired_ids,
            } for r in recs[:400]],
            "n_records": len(recs),
        })
    return out


_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hand-verify: source XML vs what we extracted</title><style>
:root{--bg:#fbfaf8;--surface:#fff;--ink:#1a1a1a;--ink2:#4a4a4a;--ink3:#767676;
--line:#e4e0d9;--accent:#1f6f6b;--warn:#b06a12;--bad:#a8322a;--good:#2f7d4f;
--mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,monospace}
@media(prefers-color-scheme:dark){:root{--bg:#141414;--surface:#1c1c1c;--ink:#ececec;
--ink2:#b9b9b9;--ink3:#8a8a8a;--line:#2f2f2f;--accent:#4fb3ad;--warn:#d69647;
--bad:#e0776d;--good:#5fbd84}}
:root[data-theme=dark]{--bg:#141414;--surface:#1c1c1c;--ink:#ececec;--ink2:#b9b9b9;
--ink3:#8a8a8a;--line:#2f2f2f;--accent:#4fb3ad;--warn:#d69647;--bad:#e0776d;--good:#5fbd84}
:root[data-theme=light]{--bg:#fbfaf8;--surface:#fff;--ink:#1a1a1a;--ink2:#4a4a4a;
--ink3:#767676;--line:#e4e0d9;--accent:#1f6f6b;--warn:#b06a12;--bad:#a8322a;--good:#2f7d4f}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 ui-sans-serif,system-ui,sans-serif;overflow-x:hidden}
header{position:sticky;top:0;z-index:9;background:var(--bg);border-bottom:1px solid var(--line);padding:14px 20px}
h1{font-size:17px;margin:0 0 10px;letter-spacing:-.01em}
.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
select,input{font:13px var(--mono);background:var(--surface);color:var(--ink);
border:1px solid var(--line);border-radius:6px;padding:6px 9px;max-width:100%}
.hint{color:var(--ink3);font-size:12.5px;font-family:var(--mono)}
main{padding:18px 20px 80px}
.block{border:1px solid var(--line);border-radius:10px;margin-bottom:22px;background:var(--surface);overflow:hidden}
.bhead{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:baseline;padding:11px 16px;
border-bottom:1px solid var(--line);font-family:var(--mono);font-size:13px}
.bhead b{font-size:14px}
.tag{font-size:11.5px;padding:2px 7px;border-radius:20px;border:1px solid var(--line);color:var(--ink2)}
.tag.none{color:var(--bad);border-color:var(--bad)}
.tag.rep{color:var(--warn);border-color:var(--warn)}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:0}
@media(max-width:1000px){.pair{grid-template-columns:1fr}}
.pane{padding:12px 16px;min-width:0}
.pane+.pane{border-left:1px solid var(--line)}
@media(max-width:1000px){.pane+.pane{border-left:none;border-top:1px solid var(--line)}}
.pane h3{font:600 11px var(--mono);letter-spacing:.08em;text-transform:uppercase;
color:var(--accent);margin:0 0 8px}
pre{margin:0;font-family:var(--mono);font-size:11.5px;line-height:1.5;color:var(--ink2);
white-space:pre-wrap;word-break:break-word;max-height:460px;overflow:auto}
table{border-collapse:collapse;width:100%;font-family:var(--mono);font-size:12px}
th{text-align:left;color:var(--ink3);font-size:10.5px;letter-spacing:.05em;text-transform:uppercase;
padding:0 10px 6px 0;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:5px 10px 5px 0;border-bottom:1px solid var(--line);color:var(--ink2);
vertical-align:top;word-break:break-word}
tr.rep td{color:var(--warn)}
.k{font-size:10.5px;padding:1px 6px;border-radius:20px;border:1px solid var(--line)}
.k.assay{color:var(--good);border-color:var(--good)}
.k.cid{color:var(--accent);border-color:var(--accent)}
.k.unknown{color:var(--bad);border-color:var(--bad)}
.scroll{overflow-x:auto}
mark{background:#ffe58a;color:#000;border-radius:2px}
</style></head><body>
<header>
<h1>Hand-verify — the patent's own XML beside what we extracted</h1>
<div class="controls">
<select id="pat"></select>
<input id="q" placeholder="filter blocks / compound id / assay" size="34">
<label class="hint"><input type="checkbox" id="only"> only blocks with NO records</label>
<span class="hint" id="count"></span>
</div>
</header>
<main id="out"></main>
<script>
const DATA = __DATA__;
const out=document.getElementById('out'), pat=document.getElementById('pat'),
      q=document.getElementById('q'), only=document.getElementById('only'),
      cnt=document.getElementById('count');
Object.keys(DATA).sort().forEach(p=>{const o=document.createElement('option');
  o.value=p;o.textContent=`${p}  (${DATA[p].length} blocks)`;pat.appendChild(o);});
const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function hl(s,term){if(!term)return esc(s);
  return esc(s).replace(new RegExp(term.replace(/[.*+?^${}()|[\\]\\\\]/g,'\\\\$&'),'gi'),
    m=>`<mark>${m}</mark>`);}
function render(){
  const p=pat.value, term=q.value.trim(); out.innerHTML='';
  let blocks=DATA[p]||[];
  if(only.checked) blocks=blocks.filter(b=>b.n_records===0);
  if(term){const t=term.toLowerCase();
    blocks=blocks.filter(b=>b.id.toLowerCase().includes(t)
      ||(b.header||[]).join(' ').toLowerCase().includes(t)
      ||b.records.some(r=>String(r.cid).toLowerCase()===t
        ||String(r.assay||'').toLowerCase().includes(t)));}
  cnt.textContent=`${blocks.length} block(s) shown`;
  blocks.forEach(b=>{
    const div=document.createElement('div'); div.className='block';
    const nrec=b.n_records, rep=b.records.filter(r=>r.repaired).length;
    div.innerHTML=`<div class="bhead"><b>${b.id}</b>
      <span>${b.n_rows} data rows</span>
      <span class="tag ${nrec?'':'none'}">${nrec} record${nrec===1?'':'s'}</span>
      ${rep?`<span class="tag rep">${rep} from repair rules</span>`:''}
      <span class="hint">${esc((b.header||[]).join(' | '))}</span></div>
    <div class="pair">
      <div class="pane"><h3>Source — the patent's CALS XML</h3>
        <pre>${hl(b.xml,term)}</pre></div>
      <div class="pane"><h3>Ours — columns, then records</h3>
        <div class="scroll"><table><thead><tr><th>#</th><th>header</th><th>kind</th><th>unit</th></tr></thead>
        <tbody>${b.columns.map(c=>`<tr><td>${c.i}</td><td>${esc(c.header)}</td>
          <td><span class="k ${c.kind}">${c.kind}</span></td><td>${esc(c.unit??'')}</td></tr>`).join('')}
        </tbody></table></div>
        ${nrec?`<div class="scroll" style="margin-top:10px"><table>
        <thead><tr><th>cid</th><th>assay</th><th>value</th><th>unit</th><th>n</th><th>range</th><th>from cell</th></tr></thead>
        <tbody>${b.records.map(r=>`<tr class="${r.repaired?'rep':''}">
          <td>${hl(r.cid,term)}</td><td>${hl(r.assay,term)}</td>
          <td>${r.qual??''}${r.value??''}</td><td>${esc(r.unit??'')}</td>
          <td>${r.n??''}</td><td>${r.lo??r.hi!=null?`${r.lo??''}-${r.hi??''}`:''}</td>
          <td>${esc(r.text??'')}</td></tr>`).join('')}</tbody></table></div>`
        :'<p class="hint" style="margin-top:10px">No records from this block.</p>'}
      </div></div>`;
    out.appendChild(div);
  });
}
pat.onchange=render; q.oninput=render; only.onchange=render;
pat.value=Object.keys(DATA).sort()[0]; render();
</script></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patents", default=None, help="comma-separated ids")
    ap.add_argument("--all", action="store_true", help="every cached patent")
    ap.add_argument("--only-gaps", action="store_true",
                    help="only blocks that produced no records")
    ap.add_argument("--out", default="docs/reports/verify_source_vs_extraction.html")
    a = ap.parse_args()

    xml_dir = config.OUTPUT_DIR / "uspto_xml"
    if a.patents:
        pids = a.patents.split(",")
    elif a.all:
        pids = sorted(p.stem for p in xml_dir.glob("*.xml"))
    else:
        pids = sorted(p.stem for p in xml_dir.glob("*.xml"))[:8]

    data = {}
    for pid in pids:
        f = xml_dir / f"{pid}.xml"
        if not f.exists():
            print(f"  ! {pid}: not cached", file=sys.stderr)
            continue
        data[pid] = _blocks(pid, f.read_text(errors="ignore"), a.only_gaps)
        print(f"  {pid}: {len(data[pid])} blocks, "
              f"{sum(b['n_records'] for b in data[pid])} records")
    if not data:
        print("nothing to render", file=sys.stderr)
        return 1

    page = _PAGE.replace("__DATA__", json.dumps(data))
    dest = config.REPO_ROOT / a.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(page, encoding="utf-8")
    print(f"\nwrote {dest}  ({len(page) // 1024} KB, {len(data)} patents)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
