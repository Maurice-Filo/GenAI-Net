from pathlib import Path
from collections import Counter
import json,sqlite3
OUT=Path(__file__).resolve().parent
BASE=OUT.parents[2]/'comparisons/search_exploration/reports/dose-ultrasensitive-full-20seed-20261001'
rows=[]
for seed in range(1,21):
 run=BASE/'runs'/f'dose-ultrasensitive-full-seed{seed:02}'
 f=run/'direct_llm_proposals.jsonl';direct=[json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []
 valid=[x for x in direct if x['valid']]
 expected=Counter((x['record']['candidate_hash'],x['completed_epoch']) for x in valid)
 c=sqlite3.connect(f'file:{run}/exploration.sqlite?mode=ro&immutable=1',uri=True)
 events=[json.loads(x[0]) for x in c.execute("SELECT payload FROM events WHERE kind='archive_insertion' AND json_extract(payload,'$.source')='llm' AND COALESCE(json_extract(payload,'$.stream'),'')!='optimization'")]
 actual=Counter((x['candidate_hash'],x['epoch']) for x in events)
 fits=c.execute("SELECT count(*) FROM events WHERE kind='archive_insertion' AND json_extract(payload,'$.origin')='llm' AND json_extract(payload,'$.stream')='optimization'").fetchone()[0]
 c.close()
 d=json.loads((run/'completed.json').read_text())['metrics']
 h=json.loads((run/'llm_history.json').read_text())
 row=dict(seed=seed,direct_records=len(direct),valid=len(valid),invalid_reasons=dict(Counter(x['message'] for x in direct if not x['valid'])),
   insertion_events=len(events),merged_counter=d['LLM/merged_candidates'],missing_insertions=sum((expected-actual).values()),extra_insertions=sum((actual-expected).values()),
   admitted=sum(bool(x['admitted']) for x in events),admitted_before_end=sum(bool(x['admitted']) and x['epoch']<100 for x in events),
   arrivals_at_end=sum(x['epoch']==100 for x in events),llm_fit_merges=fits,
   rounds=len(h),failed_rounds=sum(bool(x.get('error')) for x in h),pending_at_end=d['LLM/pending_rounds'])
 rows.append(row);print(row,flush=True)
summary={k:sum(r[k] for r in rows) for k in ['valid','insertion_events','missing_insertions','extra_insertions','admitted','admitted_before_end','arrivals_at_end','llm_fit_merges','rounds','failed_rounds','pending_at_end']}
(OUT/'insertion_audit.json').write_text(json.dumps(dict(summary=summary,runs=rows),indent=2)+'\n');print('TOTAL',summary)
