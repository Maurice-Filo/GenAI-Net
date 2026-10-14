"""Frequent added reactions and best parity network from the frozen 20-run cohort."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from proposed_results_figures import style, axes_style, PURPLE


def extract():
    path=HERE/'qualification_snapshot.json'
    cohort=json.loads(path.read_text())
    assert len(cohort['runs'])==20
    winner=min(cohort['topologies'],key=lambda d:(d['loss'],d['seed'],d['candidate_hash']))
    run=Path(next(r['folder'] for r in cohort['runs'] if r['seed']==winner['seed']))
    contract_path=next(p/'OUTPUT_CONTRACT.json' for p in sorted((run/'harness-workspaces').iterdir())
                       if (p/'OUTPUT_CONTRACT.json').exists())
    contract=json.loads(contract_path.read_text())
    catalog={r['id']:r for r in contract['reaction_library']}
    library_digest=hashlib.sha256(json.dumps(contract['reaction_library'],sort_keys=True).encode()).hexdigest()
    for r in cohort['runs']:
        workspace=Path(r['folder'])/'harness-workspaces'
        other=next(p/'OUTPUT_CONTRACT.json' for p in sorted(workspace.iterdir()) if (p/'OUTPUT_CONTRACT.json').exists())
        c=json.loads(other.read_text())
        assert hashlib.sha256(json.dumps(c['reaction_library'],sort_keys=True).encode()).hexdigest()==library_digest
        assert c['template_reaction_ids']==[1,2,3,4]
    counts=Counter(rid for d in cohort['topologies'] for rid in d['reaction_ids'])
    assert not set(counts).intersection(contract['forbidden_reaction_ids'])
    assert sum(counts.values())==12*len(cohort['topologies'])
    ranked=[dict(reaction_id=rid,display=catalog[rid]['display'],count=n,
                 fraction=n/len(cohort['topologies']))
            for rid,n in sorted(counts.items(),key=lambda item:(-item[1],item[0]))]
    db_path=(run/'exploration.sqlite').resolve()
    assert not Path(str(db_path)+'-wal').exists() or Path(str(db_path)+'-wal').stat().st_size==0
    with sqlite3.connect(db_path.as_uri()+'?mode=ro&immutable=1',uri=True) as db:
        raw=db.execute('SELECT payload FROM designs WHERE key=?',(winner['candidate_hash'],)).fetchone()[0]
    design=json.loads(raw)
    assert design['loss']==winner['loss'] and design['all_rows_correct']
    assert sorted(design['reaction_ids'])==winner['reaction_ids']
    reactions=[dict(reaction_id=rid,display=catalog[rid]['display'],rate=rate)
               for rid,rate in zip(design['reaction_ids'],design['rates'],strict=True)]
    result=dict(denominator=len(cohort['topologies']),ranking=ranked,winner=design,
                winner_seed=winner['seed'],winner_reactions=reactions,
                definition='Each qualifying labelled topology counts once per run, including terminal synchronization; fixed input reactions excluded from frequencies. Winner minimizes recorded canonical loss over all 20 runs.',
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                winner_record_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                contract_path=str(contract_path),library_sha256=library_digest)
    (HERE/'reaction_network_snapshot.json').write_text(json.dumps(result,indent=2)+'\n')
    with (HERE/'reaction_frequencies.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(ranked[0]));writer.writeheader();writer.writerows(ranked)
    return result


def side_math(side):
    if side.strip()=='∅':return r'\varnothing'
    terms=[]
    for name,count in Counter(side.strip().split(' + ')).items():
        symbol=r'\mathrm{OUT}' if name=='OUT' else name.split('_')[0]+'_{'+name.split('_')[1]+'}'
        terms.append((str(count) if count>1 else '')+symbol)
    return ' + '.join(terms)


def equation(display,rate=None):
    left,right=display.split(';')[0].strip().split(' ----> ')
    arrow=r'\longrightarrow' if rate is None else r'\overset{'+format(rate,'.6g')+r'}{\longrightarrow}'
    return '$'+side_math(left)+r'\;'+arrow+r'\;'+side_math(right)+'$'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--refresh',action='store_true');args=parser.parse_args()
    data=extract() if args.refresh else json.loads((HERE/'reaction_network_snapshot.json').read_text())
    style()
    fig,axes=plt.subplots(1,3,figsize=(12.5,4),gridspec_kw={'width_ratios':[1,1.15,1.5]},constrained_layout=True)
    top=data['ranking'][:5];ax=axes[0]
    bars=ax.barh(range(5),[r['count'] for r in top],color=PURPLE,height=.55)
    ax.bar_label(bars,labels=[str(r['count']) for r in top],padding=4,fontsize=9)
    ax.set(yticks=range(5),yticklabels=[equation(r['display']) for r in top],
           xlim=(0,data['denominator']*1.17),xlabel='Qualifying topologies',ylim=(4.6,-.6))
    axes_style(ax,'A  Frequent reactions')
    ax=axes[1];ax.set_axis_off();ax.set_title('B  Best network',loc='left')
    for i,r in enumerate(data['winner_reactions']):
        col,row=divmod(i,6)
        ax.text(.02+col*.50,.9-row*.128,equation(r['display'],r['rate']),
                transform=ax.transAxes,fontsize=11,va='center')
    ax.text(.02,.045,r'$\varnothing\;\overset{u_i}{\longrightarrow}\;X_i,\quad i=1,2,3,4$',
            transform=ax.transAxes,fontsize=11,va='center')
    path=HERE/'qualification_snapshot.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==data['source_sha256']
    cohort=json.loads(path.read_text())
    winner=next(r for r in cohort['topologies']
                if r['seed']==data['winner_seed'] and r['candidate_hash']==data['winner']['candidate_hash'])
    assert winner['loss']==data['winner']['loss'] and winner['diagnostics_available']
    responses=sorted(zip(winner['inputs'],winner['targets'],winner['terminal_outputs'],strict=True))
    inputs=np.array([r[0] for r in responses]);targets=np.array([r[1] for r in responses])
    outputs=np.array([r[2] for r in responses])
    assert inputs.shape==(16,4) and np.isin(inputs,[0,1]).all()
    np.testing.assert_array_equal(inputs.sum(axis=1)%2,targets)
    assert np.isfinite(outputs).all() and np.all((2*targets-1)*(outputs-.5)>1e-6)
    ax=axes[2];x=np.arange(16)
    ax.bar(x,outputs,color=PURPLE,width=.7,label='Network')
    ax.hlines(targets,x-.45,x+.45,color='#222222',ls='--',lw=1.3,label='Target',zorder=3)
    ax.set(xticks=x,xticklabels=[''.join(str(int(u)) for u in r) for r in inputs],
           xlabel=r'Input $(u_1,u_2,u_3,u_4)$',ylabel='Terminal output',
           xlim=(-.7,15.7),ylim=(-.04,max(1.22,float(outputs.max())*1.2)))
    ax.tick_params(axis='x',labelrotation=90,labelsize=8)
    ax.legend(frameon=False,fontsize=8,loc='upper right',ncol=2)
    axes_style(ax,'C  Input–output response')
    with (HERE/'best_network_response.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['u1','u2','u3','u4','target','terminal_output'])
        writer.writerows([*map(int,u),t,y] for u,t,y in responses)
    for ext in ['pdf','png','svg']:fig.savefig(HERE/f'parity_reactions_and_network.{ext}',dpi=180)
    plt.close(fig)
    print(json.dumps(dict(top5=top,winner_seed=data['winner_seed'],loss=data['winner']['loss']),indent=2))


if __name__=='__main__':main()
