"""One-shot completion trigger for the authorized RPA prompt-ablation analysis."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path, value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(path)


def main(root, output, state):
    state.mkdir(parents=True,exist_ok=True)
    lock=(state/'watcher.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    repo=Path(__file__).resolve().parents[2]
    scripts=[repo/'comparisons/search_exploration'/name for name in (
        'analyze_rpa_prompt_ablation.py','analyze_rpa_mechanisms.py','analyze_rpa_communication_families.py')]
    if (state/'registration.json').exists():
        raise RuntimeError('Watcher already registered; reconcile before restarting')
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts}
    command=[sys.executable,'-m','comparisons.search_exploration.analyze_rpa_prompt_ablation',
             '--root',str(root),'--output',str(output),'--final']
    write(state/'registration.json',dict(pid=os.getpid(),registered=time.time(),campaign=str(root),
        output=str(output),command=command,source_hashes=hashes,
        trigger='finished.json plus all 20 seeds complete, terminal draining and diagnostic compaction complete'))
    while True:
        progress=json.loads((root/'progress.json').read_text())
        jobs=progress['jobs']
        complete=sum(j['status']=='complete' for j in jobs)
        compacted=sum(j.get('storage_exit_code')==0 for j in jobs)
        failed=[j['seed'] for j in jobs if j['status']=='failed']
        write(state/'status.json',dict(status='waiting',updated=time.time(),completed=complete,
            compacted=compacted,failed_seeds=failed,controller_guard=progress.get('guard')))
        if (root/'finished.json').exists():
            if len(jobs)!=20 or complete!=20 or compacted!=20:
                write(state/'status.json',dict(status='needs_attention',updated=time.time(),
                    reason='Campaign ended without 20 completed and compacted seeds; no incomplete result labelled final',failed_seeds=failed))
                return
            changed=[name for name,digest in hashes.items() if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest]
            if changed:
                write(state/'status.json',dict(status='needs_attention',reason='Analysis source changed',files=changed))
                return
            write(state/'status.json',dict(status='analyzing',started=time.time()))
            env=os.environ.copy()
            env.update(MPLCONFIGDIR='/tmp/crn-mpl',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            with (state/'analysis.log').open('x') as log:
                result=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(state/'status.json',dict(status='complete' if result.returncode==0 else 'failed',
                finished=time.time(),exit_code=result.returncode,output=str(output)))
            return
        time.sleep(45)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--state',type=Path,required=True)
    a=p.parse_args()
    main(a.root.resolve(),a.output.resolve(),a.state.resolve())
