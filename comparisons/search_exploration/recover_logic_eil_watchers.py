"""Recover the EIL ablation after Harness watcher exhaustion, with a live preflight."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from comparisons.search_exploration.recover_rpa_prompt_watchers import preflight, read, write


def prepare(previous, root):
    assert read(previous / 'progress.json')['status'] == 'stopped_infrastructure_failure'
    for name, expected in read(previous / 'input_hashes.json').items():
        assert hashlib.sha256((previous / name).read_bytes()).hexdigest() == expected, name
    root.mkdir(parents=True, exist_ok=False)
    for name in ('source', 'configs', 'controller'):
        shutil.copytree(previous / name, root / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('logs', 'runs', 'tmp', 'mpl'):
        (root / name).mkdir()
    for name in ('TASK.md', 'configuration_matches.json', 'preflight.json'):
        shutil.copy2(previous / name, root / name)
    path = root / 'source/RL4CRN/search_streams/harness.py'
    old = path.read_text()
    target = 'model=model, provider="deepseek-official", timeout_seconds=request_timeout,'
    assert old.count(target) == 1
    new = old.replace(target, target + '\n            extra_environment={"CHOKIDAR_USEPOLLING": "1", "CHOKIDAR_INTERVAL": "1000"},')
    path.write_text(new)
    (root / 'transport.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True))))
    shutil.copy2(__file__, root / 'controller' / Path(__file__).name)
    helper = Path(__file__).with_name('recover_rpa_prompt_watchers.py')
    shutil.copy2(helper, root / 'controller' / helper.name)
    plan = read(previous / 'plan.json')
    plan.update(root=str(root), infrastructure_recovery_of=str(previous))
    plan['execution_changes'] += ['Harness file watching uses polling at 1000 ms; no scientific algorithm change',
                                 'Require a successful live Decider/Writer round before launch; do not reuse preflight proposals']
    for job in plan['jobs']:
        job['command'] = [v.replace(str(previous), str(root), 1) for v in job['command']]
        job['output'] = job['output'].replace(str(previous), str(root), 1)
        job['status'] = 'queued'
    # The controller independently enforces the live preflight, including manual starts.
    path = root / 'controller/run_logic_eil_ablation.py'
    controller = path.read_text()
    target = "    lock = (root / 'controller.lock').open('a')"
    assert controller.count(target) == 1
    controller = controller.replace(target, "    assert read(root / 'live_preflight.json')['status'] == 'passed'\n" + target)
    path.write_text(controller)
    write(root / 'plan.json', plan)
    write(root / 'progress.json', dict(status='awaiting_live_preflight', jobs=plan['jobs'], issues=[]))
    paths = [p for p in root.rglob('*') if p.is_file() and p.name != 'progress.json']
    write(root / 'input_hashes.json', {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    print(json.dumps(dict(prepared=str(root), jobs=20)), flush=True)


def start(root):
    if (root / 'started.json').exists():
        raise RuntimeError('Campaign already started; refusing a second launch')
    plan = read(root / 'plan.json')
    deadlines = {float(j['command'][j['command'].index('--llm-timeout') + 1]) for j in plan['jobs']}
    if len(deadlines) != 1:
        raise ValueError('Expected a shared production role deadline')
    timeout_seconds = deadlines.pop()
    write(root / 'progress.json', dict(status='live_preflight_running', updated=time.time(),
        role_timeout_seconds=timeout_seconds, jobs=plan['jobs'], issues=[]))
    print(json.dumps(dict(status='live_preflight_running', model=plan['model'],
                         role_timeout_seconds=timeout_seconds)), flush=True)
    try:
        preflight(root, task='logic', timeout_seconds=timeout_seconds)
    except Exception as exc:
        write(root / 'live_preflight_failed.json', dict(type=type(exc).__name__, error=str(exc), time=time.time()))
        write(root / 'progress.json', dict(status='live_preflight_failed', updated=time.time(),
            role_timeout_seconds=timeout_seconds, jobs=plan['jobs'],
            issues=[dict(type=type(exc).__name__, error=str(exc))]))
        raise
    command = [sys.executable, str(root / 'controller/run_logic_eil_ablation.py'), '--root', str(root)]
    subprocess.run(command + ['--check'], check=True)
    with (root / 'logs/controller.log').open('x') as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (root / 'controller_pid.txt').write_text(str(process.pid) + '\n')
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
