"""Run a frozen pilot, including explicitly adopted live runs, and preserve diagnostics."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('campaign',type=Path);a=p.parse_args()
    root=a.campaign.resolve();meta=json.loads((root/'campaign.json').read_text())
    lock=(root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=Path(meta['source_root'])
    for rel,h in json.loads((root/'source_hashes.json').read_text()).items():
        if hashlib.sha256((source/rel).read_bytes()).hexdigest()!=h:raise RuntimeError('Frozen source changed: '+rel)
    for rel,h in json.loads((root/'configuration_hashes.json').read_text()).items():
        if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=h:raise RuntimeError('Frozen config changed: '+rel)
    for name,h in json.loads((root/'julia_source_hashes.json').read_text()).items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=h:raise RuntimeError('Julia dependency changed: '+name)
    env=dict(os.environ,PYTHONPATH=str(source),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1',JULIA_NUM_THREADS='1',MPLCONFIGDIR=str(root/'mpl'),
        PILOT_NETWORK_AUDIT=str(root/'network_attempts.jsonl'))
    semaphores={task:threading.Semaphore(limit) for task,limit in meta["task_concurrency"].items()}
    def run(job):
        with semaphores[job["task"]]:
            return run_one(job)
    def run_one(job):
        target=root/'runs'/job['name']
        if job.get('reuse_existing'):
            expected=job['reused_manifest_sha256']
            if hashlib.sha256((target/'manifest.json').read_bytes()).hexdigest()!=expected:
                raise RuntimeError('Adopted run manifest changed')
            while not (target/'completed.json').exists() and not (target/'failed.json').exists():
                time.sleep(5)
            return {'name':job['name'],'exit_code':0 if (target/'completed.json').exists() else 1,
                'complete':(target/'completed.json').exists(),'reused_live_from':job['reused_live_from']}
        if job['task']=='rpa':
            while any(not Path(f).exists() for f in meta.get('wait_for_prior_rpa_exits',[])):
                time.sleep(5)
        if target.exists():
            raise RuntimeError('Refusing to overwrite/retry existing run: '+str(target))
        start=time.time()
        with (root/'logs'/(job['name']+'.log')).open('x') as log:
            proc=subprocess.Popen(job['command'],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
            (root/'logs'/(job['name']+'.process.json')).write_text(json.dumps({'pid':proc.pid,'started':start,'command':job['command']}))
            code=proc.wait()
        result={'name':job['name'],'exit_code':code,'elapsed':time.time()-start,'complete':(target/'completed.json').exists()}
        (root/'logs'/(job['name']+'.exit.json')).write_text(json.dumps(result))
        return result
    (root/'queue_started.json').write_text(json.dumps({'pid':os.getpid(),'started':time.time()}))
    results=[]
    with ThreadPoolExecutor(max_workers=meta['concurrency']) as pool:
        # All Logic seeds run concurrently; one RPA run at a time bounds peak workers.
        jobs=sorted(meta['jobs'],key=lambda j:(j['task']!='logic',j['seed'],j['condition']))
        futures={pool.submit(run,job):job for job in jobs}
        for future in as_completed(futures):
            results.append(future.result())
            (root/'queue_status.json').write_text(json.dumps({'results':results,'updated':time.time()},indent=2))
            analysis=subprocess.run([jobs[0]['command'][0],'-m','comparisons.search_exploration.category_pilot.analyze',str(root)],cwd=source,env=env)
            if analysis.returncode:
                (root/'analysis_error.json').write_text(json.dumps({'exit_code':analysis.returncode,'time':time.time()}))
    if any(r['exit_code'] or not r['complete'] for r in results):
        raise RuntimeError('One or more runs failed; evidence retained and no run retried.')
    (root/'queue_completed.json').write_text(json.dumps({'results':results,'finished':time.time()},indent=2))

if __name__=='__main__':main()
