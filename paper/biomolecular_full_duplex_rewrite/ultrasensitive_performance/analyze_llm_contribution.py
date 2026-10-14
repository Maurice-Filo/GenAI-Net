"""Event-time provenance audit for completed Ultrasensitive runs."""
from pathlib import Path
import json, math
from collections import Counter
from comparisons.search_exploration.analyze_rpa_llm_provenance import extract, GROUPS
OUT=Path(__file__).resolve().parent
BASE=OUT.parents[2]/'comparisons/search_exploration/reports/dose-ultrasensitive-full-20seed-20261001'
rows=[];first=Counter();winners=Counter();successful_winners=Counter()
for seed in range(1,21):
 r=extract(BASE/'runs'/f'dose-ultrasensitive-full-seed{seed:02}',seed)
 minimum=min(g['best_loss'] for g in r['groups'])
 tied=[g['group'] for g in r['groups'] if math.isclose(g['best_loss'],minimum,rel_tol=1e-9,abs_tol=1e-12)]
 for g in tied:
  winners[g]+=1/len(tied)
  if minimum<.05:successful_winners[g]+=1/len(tied)
 for g in r['groups']:
  first[g['group']]+=g['first_qualified']
  if not math.isfinite(g['best_loss']):g['best_loss']=None
 rows.append(dict(seed=seed,winning_routes=tied,groups=r['groups'],qualified_total=r['qualified_total'],interactions=r['interactions']))
 print('seed',seed,'winner',tied,'qualified',r['qualified_total'],flush=True)
summary=dict(winning_routes=dict(winners),successful_winning_routes=dict(successful_winners),
 first_qualifying_routes=dict(first),runs=rows,
 interpretation='Event-time provenance; ties split equally. Per-run topology credits, not pooled unique mechanisms. Numerical policy may already have learned from LLM suggestions.')
(OUT/'llm_contribution.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='runs'},indent=2))
