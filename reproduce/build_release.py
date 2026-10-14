"""Create a compact, non-destructive scientific-data release from local records.

The destination must be new or contain this builder's release marker. SQLite
sources are read-only; large diagnostic blobs and checkpoints are not copied.
"""
import argparse, base64, csv, gzip, hashlib, json, os, shutil, sqlite3, subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO/'paper/biomolecular_full_duplex_rewrite'
SMALL = ['manifest.json','completed.json','terminal_union.json','metrics.jsonl',
         'direct_llm_proposals.jsonl','llm_history.json','optimization_results.json',
         'progress.csv','source.tar.gz','diagnostic_downsampling.json',
         'best_network.json','best_network.txt','trainer_history.csv','config.json']

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(x,indent=2)+'\n')

def strings(x):
 if isinstance(x,str):yield x
 elif isinstance(x,list):
  for v in x:yield from strings(v)
 elif isinstance(x,dict):
  for k,v in x.items():yield str(k);yield from strings(v)

def inventory():
 jobs=json.loads((PAPER/'token_costs/summary.json').read_text())['selected_runs']
 runs={};aliases={};members=[]
 def add(p,group=None,seed=None):
  p=Path(p)
  if not p.is_dir():return
  if not any((p/n).exists() for n in ['completed.json','progress.csv']):return
  real=str(p.resolve());rid=hashlib.sha256(real.encode()).hexdigest()[:16]
  runs.setdefault(rid,{'id':rid,'source':real,'aliases':[],'memberships':[]})
  aliases[str(p)]=rid;aliases[real]=rid
  if str(p) not in runs[rid]['aliases']:runs[rid]['aliases'].append(str(p))
  if group: runs[rid]['memberships'].append({'group':group,'seed':seed})
 for j in jobs:add(j['run'],j['group'],j['seed'])
 evidence=[PAPER/'genainet_comparison/audit.json',PAPER/'numerical_comparison/frozen_runs.json',
           PAPER.parent/'jmlr_hybrid_crn/generated/channels.json',
           PAPER/'qwen_comparison/qwen_rpa_timeout3600_snapshot.json']
 for f in evidence:
  for s in strings(json.loads(f.read_text())):
   if s.startswith('/local'):
    p=Path(s)
    if p.is_file():p=p.parent
    add(p)
 plan=Path('/local1/rossin/search-exploration/rpa-statistical-20seed-20260915/active_plan.json')
 for s in strings(json.loads(plan.read_text())):
  if s.startswith('/local'):add(s)
 return list(runs.values()),aliases

def copy_file(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True)
 shutil.copyfile(src,dst)
 return {'file':str(dst),'bytes':dst.stat().st_size,'sha256':sha(dst)}

def export_run(item,root):
 src=Path(item['source']);dst=root/'data/runs'/item['id'];dst.mkdir(parents=True,exist_ok=True)
 complete=dst/'export.json'
 if complete.exists():return json.loads(complete.read_text())
 if shutil.disk_usage(root).free<30*1024**3:raise RuntimeError('Export reserve below 30 GiB')
 meta=dict(item,tables={},omitted=['diagnostics/evaluations blobs','checkpoints','Julia scratch jobs','duplicated workspace context'])
 for name in SMALL:
  if (src/name).is_file():copy_file(src/name,dst/name)
 dbpath=src/'exploration.sqlite'
 if dbpath.exists():
  wal=Path(str(dbpath)+'-wal')
  if wal.exists() and wal.stat().st_size:raise RuntimeError(f'Nonempty WAL: {src}')
  before=dbpath.stat()
  with sqlite3.connect(dbpath.as_uri()+'?mode=ro&immutable=1',uri=True) as db:
   schemas=db.execute("SELECT name,sql FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
   for name,sql in schemas:
    if name in ['diagnostics','evaluations']:continue
    assert name.replace('_','').isalnum()
    cur=db.execute(f'SELECT * FROM "{name}"')
    columns=[x[0] for x in cur.description];count=0;digest=hashlib.sha256()
    target=dst/(name+'.jsonl.gz');tmp=target.with_suffix('.partial')
    with tmp.open('wb') as raw, gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0,compresslevel=5) as out:
     for row in cur:
      vals=[{'base64':base64.b64encode(v).decode()} if isinstance(v,bytes) else v for v in row]
      line=(json.dumps(vals,separators=(',',':'),ensure_ascii=True)+'\n').encode()
      out.write(line);digest.update(line);count+=1
    tmp.replace(target)
    assert count==db.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
    meta['tables'][name]={'columns':columns,'schema':sql,'rows':count,'rows_sha256':digest.hexdigest(),'bytes':target.stat().st_size}
   if any(n=='designs' for n,_ in schemas):
    manifest=json.loads((src/'manifest.json').read_text()) if (src/'manifest.json').exists() else {}
    threshold=manifest.get('config',{}).get('quality_threshold')
    meta['threshold']=threshold
    best=db.execute("SELECT min(json_extract(payload,'$.loss')) FROM designs WHERE json_extract(payload,'$.simulation_complete')=1").fetchone()[0]
    meta['best_design_loss']=best
    if threshold is not None:
     keys=set();rows=0
     for payload, in db.execute("SELECT payload FROM designs WHERE json_extract(payload,'$.simulation_complete')=1 AND json_extract(payload,'$.loss') < ?",(threshold,)):
      d=json.loads(payload);keys.add(tuple(sorted(d['reaction_ids'])));rows+=1
     meta['qualifying_design_rows']=rows;meta['qualifying_topologies_in_designs']=len(keys)
  after=dbpath.stat();assert(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
  meta['source_database_bytes']=before.st_size
 dump(complete,meta)
 return meta

def stage_assets(root):
 # Retain source data unchanged: embedded source hashes remain verifiable.
 for srcroot,rel in [(PAPER,Path('analysis/biomolecular_full_duplex_rewrite')),
                     (PAPER.parent/'jmlr_hybrid_crn/generated',Path('analysis/jmlr_hybrid_crn/generated'))]:
  for p in srcroot.rglob('*'):
   if not p.is_file() or '__pycache__' in p.parts or 'tmlr_template' in p.parts:continue
   if p.suffix not in ['.py','.csv','.json','.jsonl','.npz','.txt'] and p.name!='selection.json.gz':continue
   if p.stat().st_size>150*1024**2:raise RuntimeError(f'Unexpected asset size: {p}')
   copy_file(p,root/rel/p.relative_to(srcroot))
 # Exact figure references, independent of generated outputs.
 import re
 figs=set()
 for p in [PAPER/n for n in ['introduction.tex','results.tex','supplementary_results.tex','supplementary_prompts.tex']]:
  figs.update(re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',p.read_text()))
 for f in sorted(figs):copy_file(PAPER/f,root/'expected/figures'/f)
 dump(root/'manifest/figures.json',sorted(figs))
 # Plot/diagnostic code dependencies; no data, credentials, VCS or environments.
 for directory in ['RL4CRN','comparisons','tools']:
  for parent,dirs,files in os.walk(REPO/directory):
   dirs[:]=[d for d in dirs if d not in ['reports','runs','data','generated','node_modules','__pycache__','.git','harness_runs','results','outputs'] and not Path(parent,d).is_symlink()]
   for n in files:
    p=Path(parent,n)
    if p.suffix in ['.py','.jl','.sh','.toml','.mjs','.js'] or n in ['pnpm-lock.yaml','package.json','pnpm-workspace.yaml']:
     copy_file(p,root/p.relative_to(REPO))
 for n in ['LICENSE','setup.py']:copy_file(REPO/n,root/n)
 for n in ['reproduce.py','scientific_checks.py','README.txt','reproduce.sh','requirements-analysis.txt']:
  if (REPO/'reproduce'/n).exists():copy_file(REPO/'reproduce'/n,root/n)
 # Full measured analysis dependencies; numerical/LLM dependencies are separately recorded.
 import importlib.metadata
 names=['numpy','scipy','matplotlib','pillow','seaborn','networkx']
 if not (REPO/'reproduce/requirements-analysis.txt').exists():
  (root/'requirements-analysis.txt').write_text('\n'.join(n+'=='+importlib.metadata.version(n) for n in names)+'\n')
 freeze=subprocess.check_output([os.sys.executable,'-m','pip','freeze'],text=True)
 (root/'manifest/python-environment.txt').write_text(freeze)


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--limit',type=int);ap.add_argument('--assets-only',action='store_true');args=ap.parse_args()
 root=args.output.resolve()
 if root.exists() and not(root/'RELEASE_BUILD').exists():raise RuntimeError('Refusing an unrelated destination')
 root.mkdir(parents=True,exist_ok=True);(root/'RELEASE_BUILD').write_text('compact release v1\n')
 (root/'manifest').mkdir(exist_ok=True)
 stage_assets(root)
 jobs,aliases=inventory();dump(root/'manifest/runs.json',jobs);dump(root/'manifest/path_map.json',aliases)
 print(f'Selected {len(jobs)} unique runs',flush=True)
 if args.assets_only:return
 if args.limit:jobs=jobs[:args.limit]
 results=[];errors=[]
 with ProcessPoolExecutor(max_workers=args.workers) as pool:
  pending={pool.submit(export_run,j,root):j for j in jobs}
  for i,f in enumerate(as_completed(pending),1):
   try:results.append(f.result())
   except Exception as e:errors.append({'run':pending[f]['id'],'error':repr(e)})
   if i%10==0 or i==len(jobs):print(f'{i}/{len(jobs)} exported; errors={len(errors)}',flush=True)
 dump(root/'manifest/export_status.json',{'completed':len(results),'requested':len(jobs),'errors':errors})
 if errors:raise RuntimeError(f'{len(errors)} exports failed; inspect export_status.json')

if __name__=='__main__':main()
