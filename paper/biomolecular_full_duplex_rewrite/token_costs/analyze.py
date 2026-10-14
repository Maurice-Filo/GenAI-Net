"""Read-only accounting of recorded LLM usage in the final manuscript cohorts.

Run from repository root with .venv/bin/python. --cached replots saved aggregates.
No inference, billing API requests, or modifications to source campaigns.
"""
from pathlib import Path
import argparse, csv, json, re, subprocess, hashlib
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT=Path(__file__).resolve().parent
PAPER=OUT.parent
REPO=PAPER.parents[1]
ROOT=Path('/local1/rossin/search-exploration')
SESSIONS=Path('/local0/home/rossin/ai-workspaces/deepseek-test/.dsh-home/sessions')
RATE={'inputTokens':0.15,'cacheReadTokens':0.003,'outputTokens':0.60}
TOKEN_KEYS=tuple(RATE)+('reasoningTokens','cacheWriteTokens')

def read(p):return json.loads(Path(p).read_text())
def write_csv(p,rows):
    if not rows:return
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def select_runs():
    jobs=[]
    def add(group,label,task,seed,path,kind='flash'):
        path=Path(path)
        assert read(path/'completed.json')['status']=='complete',path
        jobs.append(dict(group=group,label=label,task=task,seed=seed,run=str(path),kind=kind))
    channels=read(REPO/'paper/jmlr_hybrid_crn/generated/channels.json')['rows']
    labels={'llm-to-rl':'Insertion only','rl-to-llm':'Return only','full':'Full duplex','none':'No communication'}
    for task in ['rpa','logic']:
        arms=['llm-to-rl','rl-to-llm','full'] if task=='rpa' else ['none','llm-to-rl','rl-to-llm','full']
        for arm in arms:
            for row in sorted([r for r in channels if r['task']==task and r['arm']==arm],key=lambda r:r['seed']):
                add(f'{task}_{arm}',f'{task.upper() if task=="rpa" else "Logic"} · {labels[arm]}',task,row['seed'],row['folder'])
    # Use the manuscript's explicit outage-replacement selection for breadth runs.
    import sys
    sys.path.insert(0,str(PAPER))
    from generate_dose_overviews import select_runs as breadth
    for task,label in [('hill','Hill dose response'),('ultrasensitive','Ultrasensitivity'),('biphasic','Biphasic response'),('classifier','Classification'),('parity','Four-input parity')]:
        chosen,excluded=breadth(task)
        assert len(chosen)==20 and not excluded,(task,len(chosen),excluded)
        for seed,path,_,replacement in chosen:add(task,label+' · Full duplex',task,seed,path)
    for campaign,group,label,kind in [
        ('rpa-flash-mechanism-unspecified-20seed-20261009-r2','rpa_prompt','RPA · No mechanism hints','flash'),
        ('logic-flash-full-eil0-20seed-20261012-r2','logic_eil0','Logic · EIL = 0','flash'),
        ('qwen38-matched-rpa-logic-20seed-20261006-r2','qwen','Qwen','local'),
        ('qwen38-rpa-timeout3600-20seed-20261007','qwen60','RPA · Qwen, 60 min','local')]:
        for j in read(ROOT/campaign/'plan.json')['jobs']:
            task=j.get('task','logic' if group=='logic_eil0' else 'rpa')
            if group=='qwen': g=group+'_'+task;lab=('RPA · Qwen, 15 min' if task=='rpa' else 'Logic · Qwen, 60 min')
            else:g=group;lab=label
            add(g,lab,task,j['seed'],j['output'],kind)
    assert len(jobs)==340,len(jobs)
    assert len({str(Path(j['run']).resolve()) for j in jobs})==len(jobs),'Duplicate source run'
    return jobs

def index_sessions():
    index=defaultdict(list)
    for path in SESSIONS.iterdir():
        m=re.search(r'(\d{8}T\d{6}Z-crn-decider-writer-[0-9a-f]{8})',path.name)
        if m:index[m.group(1)].append(path)
    return index

def parse_session(path):
    files=[p for p in [path/'session.jsonl',path/'session.jsonl.zstd',path/'session.jsonl.zst'] if p.is_file()]
    if not files:return None
    # Prefer plain if both present. Compressed files are archived copies, not extra calls.
    f=files[0];proc=None
    if f.suffix=='.jsonl':stream=f.open('rb')
    else:proc=subprocess.Popen(['zstd','-dc',str(f)],stdout=subprocess.PIPE,stderr=subprocess.PIPE);stream=proc.stdout
    messages={};chunks={};header=None;models=set();starts=set();errors=0;title_calls=0;parse_errors=0;missing_message_usage=0;usage_fields=Counter()
    with stream:
        for line in stream:
            if not any(x in line for x in [b'"usage"',b'"type":"session"',b'"type": "session"',b'"step/start"',b'"session/title-llm-request"',b'"request/error"',b'"assistant/message"']):continue
            try:d=json.loads(line)
            except json.JSONDecodeError:parse_errors+=1;continue
            typ=d.get('type');a=d.get('data',{});key=(a.get('turn'),a.get('step'))
            if typ=='session':header=d
            if typ=='step/start':starts.add(key)
            if typ=='session/title-llm-request':title_calls+=1
            if typ=='request/error':errors+=1
            if typ=='assistant/message':
                usage=a.get('usage');source=a.get('message',{}).get('source',{})
                if source.get('model'):models.add(source['model'])
                if usage:
                    ident=(key,tuple(sorted(usage.items())))
                    messages[ident]=usage
                    usage_fields.update(usage)
                else:missing_message_usage+=1
            elif typ=='assistant/chunk' and a.get('chunk',{}).get('type')=='usage':
                usage=a['chunk'].get('usage')
                if usage:chunks[(key,tuple(sorted(usage.items())))]=usage
    if proc is not None:
        err=proc.stderr.read();assert proc.wait()==0,(f,err)
    usage=dict(chunks);usage.update(messages)
    totals={k:sum(u.get(k,0) or 0 for u in usage.values()) for k in TOKEN_KEYS}
    assert totals['outputTokens']>=totals['reasoningTokens'],(f,totals)
    return dict(session=path.name,file=str(f),bytes=f.stat().st_size,cwd=(header or {}).get('cwd'),
        models='|'.join(sorted(models)),steps=len(starts),usage_records=len(usage),message_usage_records=len(messages),
        chunk_only_records=len(set(chunks)-set(messages)),steps_without_usage=len(starts-{k[0] for k in usage}),
        title_calls_without_usage=title_calls,parse_errors=parse_errors,request_errors=errors,
        missing_message_usage=missing_message_usage,**totals)

def extract_run(job,index):
    run=Path(job['run']);hist=read(run/'llm_history.json');workspaces=list((run/'harness-workspaces').iterdir())
    sessions=[];missing=[];seen=set()
    for w in workspaces:
        candidates=index.get(w.name,[])
        if not candidates:missing.append(w.name);continue
        for directory in candidates:
            for session in directory.iterdir():
                if not session.is_dir() or session.name in seen:continue
                seen.add(session.name)
                parsed=parse_session(session)
                if parsed: sessions.append({**job,'workspace':w.name,**parsed})
                else:missing.append(str(session))
    keys=TOKEN_KEYS+('usage_records','steps','steps_without_usage','title_calls_without_usage','parse_errors','request_errors','chunk_only_records','missing_message_usage')
    row={**job,'batches':len(hist),'failed_batches':sum(bool(h.get('error')) for h in hist),
        'workspaces':len(workspaces),'missing_workspaces':len(missing),'sessions':len(sessions),
        'sessions_without_usage':sum(s['usage_records']==0 for s in sessions),
        **{k:sum(s[k] for s in sessions) for k in keys}}
    row['total_tokens']=row['inputTokens']+row['cacheReadTokens']+row['outputTokens']+row['cacheWriteTokens']
    row['cost_offpeak_usd']=sum(row[k]*rate/1e6 for k,rate in RATE.items()) if job['kind']=='flash' else None
    row['cost_peak_usd']=2*row['cost_offpeak_usd'] if row['cost_offpeak_usd'] is not None else None
    return row,sessions,missing

def aggregate(rows):
    groups={}
    for r in rows:groups.setdefault(r['group'],[]).append(r)
    out=[]
    for key,rr in groups.items():
        assert len(rr)==20,(key,len(rr))
        fields=['batches','failed_batches','workspaces','missing_workspaces','sessions','sessions_without_usage','total_tokens',*TOKEN_KEYS,'usage_records','steps','steps_without_usage','title_calls_without_usage','parse_errors','request_errors','chunk_only_records','missing_message_usage']
        out.append(dict(group=key,label=rr[0]['label'],kind=rr[0]['kind'],runs=len(rr),
          **{k:sum(r[k] for r in rr) for k in fields},
          cost_offpeak_usd=sum(r['cost_offpeak_usd'] for r in rr) if rr[0]['kind']=='flash' else None,
          cost_peak_usd=sum(r['cost_peak_usd'] for r in rr) if rr[0]['kind']=='flash' else None))
    return out

def render(data,per_run=False):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(12.8,9.1),gridspec_kw={'width_ratios':[1.35,1]},sharey=True)
    fig.subplots_adjust(left=.255,right=.98,bottom=.12,top=.885,wspace=.12)
    colors=['#287da8','#b1d4df','#943c87'];fields=['inputTokens','cacheReadTokens','outputTokens'];y=np.arange(len(data))
    factor=np.array([d['runs'] if per_run else 1 for d in data]);left=np.zeros(len(data))
    for k,c in zip(fields,colors):
        vals=np.array([d[k] for d in data])/factor/1e6
        axs[0].barh(y,vals,left=left,color=c,height=.66,edgecolor='white',linewidth=.3);left+=vals
    for i,value in enumerate(left):axs[0].text(value+max(left)*.014,i,f'{value:.2f}',va='center',fontsize=8)
    axs[0].set_xlim(0,max(left)*1.18);axs[0].set_yticks(y,[d['label'] for d in data]);axs[0].invert_yaxis()
    axs[0].set_xlabel('Recorded tokens per run (millions)' if per_run else 'Recorded tokens per 20-run campaign (millions)')
    axs[0].set_title('A  Token use',loc='left',fontweight='bold',fontsize=11)
    uppermax=max(d['cost_peak_usd']/factor[i] for i,d in enumerate(data) if d['cost_peak_usd'] is not None)
    for i,d in enumerate(data):
        if d['kind']=='local':
            axs[1].text(.015*uppermax,i,'Local compute not priced',va='center',fontsize=8,color='#5b6367');continue
        lower=d['cost_offpeak_usd']/factor[i];upper=d['cost_peak_usd']/factor[i]
        axs[1].barh(i,lower,color='#ba803b',height=.66)
        axs[1].barh(i,upper-lower,left=lower,color='#f0e1cb',hatch='///',edgecolor='#ba803b',linewidth=.3,height=.66)
        axs[1].text(upper+uppermax*.016,i,f'{lower:.2f}–{upper:.2f}',va='center',fontsize=8)
    axs[1].set_xlim(0,uppermax*1.31);axs[1].set_xlabel('Estimated API cost per run (USD)' if per_run else 'Estimated API cost per 20-run campaign (USD)')
    axs[1].set_title('B  Repriced at current Flash rates',loc='left',fontweight='bold',fontsize=11)
    for ax in axs:
        ax.set_axisbelow(True);ax.grid(axis='x',alpha=.18);ax.tick_params(axis='y',length=0)
        for edge in [2.5,6.5,11.5,13.5]:ax.axhline(edge,color='.8',lw=.6)
    handles=[Patch(facecolor=c,label=l) for c,l in zip(colors,['Other input','Reported cached input','Output (includes reasoning)'])]
    handles += [Patch(facecolor='#ba803b',label='All off-peak'),Patch(facecolor='#f0e1cb',edgecolor='#ba803b',hatch='///',label='Up to all peak')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.57,.965),ncol=3,frameon=False,fontsize=9)
    fig.text(.255,.045,'Final study cohorts: 20 completed runs each. Includes recorded usage within failed batches.\nCosts use 14 Oct 2026 rates, not invoices; title calls and requests lacking usage are uncounted.',fontsize=8,color='.35')
    name='tokens_and_cost_per_run' if per_run else 'tokens_and_cost_by_campaign'
    for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{name}.{ext}',dpi=180)
    plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--cached',action='store_true');args=p.parse_args()
    if args.cached:data=read(OUT/'summary.json')['campaigns']
    else:
        jobs=select_runs();index=index_sessions();print(f'{len(jobs)} runs selected; {len(index)} workspace names indexed',flush=True)
        ordered={j['run']:i for i,j in enumerate(jobs)};rows=[];sessions=[];missing={}
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures={pool.submit(extract_run,j,index):j for j in jobs}
            for n,f in enumerate(as_completed(futures),1):
                r,ss,mm=f.result();rows.append(r);sessions.extend(ss)
                if mm:missing[r['run']]=mm
                if n%20==0:print(f'{n}/{len(jobs)} runs processed',flush=True)
        rows.sort(key=lambda r:ordered[r['run']]);data=aggregate(rows)
        assert len({s['session'] for s in sessions})==len(sessions),'Duplicate session attributed across cohorts'
        write_csv(OUT/'runs.csv',rows);write_csv(OUT/'sessions.csv',sessions);write_csv(OUT/'campaigns.csv',data)
        meta=dict(as_of='2026-10-14',campaigns=data,missing=missing,pricing=dict(url='https://api-docs.deepseek.com/quick_start/pricing/',usd_per_million_offpeak=RATE,peak_multiplier=2),
            notes=[
                'Selected final 20-seed cohorts used in the manuscript; excludes abandoned/replaced runs and older pilots. Not an account-wide spending audit.',
                'RPA independent union is reconstructed from reused cohorts and is not charged again. No-LLM numerical runs have zero LLM tokens and are omitted.',
                'Each assistant usage event is counted once per session, turn, step and usage tuple, deduplicating stream and final-message representations. Chunk-only usage retained.',
                'Harness inputTokens excludes cache hits; outputTokens includes reasoningTokens. Reasoning is not added again. Verified in installed dsh-llm-deepseek mapUsage.',
                'Costs reprice recorded usage at the published 2026-10-14 Flash tariff: cache miss $0.15/$0.30, cache hit $0.003/$0.006, output $0.60/$1.20 per million off-peak/peak. Range is pricing sensitivity, not a confidence interval or historical invoice.',
                'Current pricing documentation states deepseek-v4-flash is now an alias served by V4.1-Flash; logs identify the requested route, not immutable provider weights.',
                'Local Qwen compute/hardware/energy costs are unavailable and shown as unpriced, not zero.',
                'Local Qwen does not separately report cache-hit tokens; Other input includes all input not separately reported as cached. Model-specific tokenizers differ.',
                'Automatic title-generation calls lack saved usage; cancelled/failed requests may also lack final usage. Totals are observed reported usage, not complete billed consumption.'],
            selected_runs=jobs)
        (OUT/'summary.json').write_text(json.dumps(meta,indent=2)+'\n')
        print('Total recorded tokens',sum(d['total_tokens'] for d in data),flush=True)
        print('Missing workspace/session mappings',sum(len(m) for m in missing.values()),flush=True)
    render(data);render(data,True)
    for d in data:print(d['label'],round(d['total_tokens']/1e6,2),'M tokens',d['cost_offpeak_usd'],d['cost_peak_usd'],'missing',d['missing_workspaces'],flush=True)

if __name__=='__main__':main()
