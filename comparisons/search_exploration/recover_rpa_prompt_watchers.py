"""Separate prompt-ablation recovery; use file polling and require a live role round."""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]


def read(p):
    return json.loads(p.read_text())


def write(p, value):
    p.write_text(json.dumps(value, indent=2) + '\n')


def prepare(previous, root):
    for name, digest in read(previous / 'input_hashes.json').items():
        assert hashlib.sha256((previous / name).read_bytes()).hexdigest() == digest, name
    root.mkdir(parents=True, exist_ok=False)
    for name in ('source', 'configs', 'controller'):
        shutil.copytree(previous / name, root / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('logs', 'runs', 'tmp', 'mpl'):
        (root / name).mkdir()
    for name in ('TASK.md', 'reference_TASK.md', 'prompt.diff', 'configuration_matches.json'):
        shutil.copy2(previous / name, root / name)
    path = root / 'source/RL4CRN/search_streams/harness.py'
    old = path.read_text()
    match = 'model=model, provider="deepseek-official", timeout_seconds=request_timeout,'
    assert old.count(match) == 1
    new = old.replace(match, match + '\n            extra_environment={"CHOKIDAR_USEPOLLING": "1", "CHOKIDAR_INTERVAL": "1000"},')
    path.write_text(new)
    (root / 'transport.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True))))
    for path in ('controller/run_rpa_prompt_ablation.py', 'source/comparisons/search_exploration/run_rpa_prompt_ablation.py'):
        shutil.copy2(REPO / 'comparisons/search_exploration/run_rpa_prompt_ablation.py', root / path)
    shutil.copy2(__file__, root / 'controller' / Path(__file__).name)
    plan = read(previous / 'plan.json')
    plan['root'] = str(root)
    plan['infrastructure_recovery_of'] = str(previous)
    plan['execution_changes'] += ['Harness file watching uses polling (1000 ms) instead of exhausted inotify watchers',
                                 'Mandatory successful live Decider/Writer preflight before dispatch',
                                 'Hold dispatch on repeated LLM errors before any valid proposal']
    for j in plan['jobs']:
        j['command'] = [v.replace(str(previous), str(root), 1) for v in j['command']]
        j['output'] = j['output'].replace(str(previous), str(root), 1)
        j['status'] = 'queued'
    write(root / 'plan.json', plan)
    write(root / 'progress.json', dict(status='awaiting_live_preflight', jobs=plan['jobs'], issues=[]))
    write(root / 'input_hashes.json', {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob('*') if p.is_file() and p.name != 'progress.json'})


def preflight(root):
    os.chdir(root / 'source')
    sys.path.insert(0, str(root / 'source'))
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'JULIA_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['MPLCONFIGDIR'] = str(root / 'mpl')
    from RL4CRN.search_exploration.evaluation import build
    from RL4CRN.search_exploration.harness import ExplorationHarnessClient
    from RL4CRN.llm.harness_runner import HarnessDeciderWriterCRNGraph
    from RL4CRN.llm.workspace_loss_protocol import PROTOCOL
    evaluator, _, _ = build('rpa', read(root / 'configs/task-01.json'), .001, 100.)
    client = ExplorationHarnessClient(workspace_root=root / 'preflight/workspaces',
        dsh_home=Path('/local0/home/rossin/ai-workspaces/deepseek-test/.dsh-home'),
        model='deepseek-v4-flash', provider='deepseek-official', timeout_seconds=900,
        candidate_validation_policy='independent-members', harness_protocol=PROTOCOL,
        extra_environment={'CHOKIDAR_USEPOLLING': '1', 'CHOKIDAR_INTERVAL': '1000'})
    graph = HarnessDeciderWriterCRNGraph(client=client, evaluator=evaluator, writer_retry_limit=0,
                                        max_workspace_evaluations=0, max_literature_searches=0)
    started = time.time()
    result = graph.run_round(task_description=(root / 'TASK.md').read_text(), num_candidates=10,
        step=0, hall_of_fame_iter=None, forbidden_topologies_text='No topology exclusions supplied.',
        sil_feedback_text='Initial request: RL/HOF/EIL information intentionally withheld.',
        jsonl_path=root / 'preflight/evaluations.jsonl')
    valid = sum(bool(e.valid) for e in result.evaluations)
    assert valid > 0, 'No valid evaluated LLM proposals'
    assert client.last_workspace.call_count == 2
    assert not read(client.last_workspace.path / 'CONTEXT/HALL_OF_FAME.json')['entries']
    write(root / 'live_preflight.json', dict(status='passed', valid=valid, roles=2,
        elapsed_seconds=time.time()-started, initial_hof_empty=True,
        prompt_sha256=hashlib.sha256((root / 'TASK.md').read_bytes()).hexdigest(),
        note='Preflight candidates are not inserted into any scientific run.'))


def start(root):
    try:
        preflight(root)
    except Exception as exc:
        write(root / 'live_preflight_failed.json', dict(type=type(exc).__name__, error=str(exc), time=time.time()))
        raise
    command = [sys.executable, str(root / 'controller/run_rpa_prompt_ablation.py'), '--root', str(root)]
    subprocess.run(command + ['--check'], check=True)
    with (root / 'logs/controller.log').open('x') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (root / 'controller.pid').write_text(str(process.pid) + '\n')
    print(json.dumps(dict(controller_pid=process.pid, status='dispatched_after_live_preflight')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--start', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare(args.previous.resolve(), args.root.resolve())
    if args.start:
        start(args.root.resolve())
