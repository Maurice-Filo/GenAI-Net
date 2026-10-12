"""Relocate verified closed runs, then apply the already-approved diagnostic policy."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from downsample_closed_diagnostics import assert_no_open_database, migrate, atomic_json


def read(path):
    return json.loads(path.read_text())


def inventory(root):
    result = {}
    for path in root.rglob('*'):
        if path.is_symlink():
            result[str(path.relative_to(root))] = ['link', os.readlink(path)]
        elif path.is_file():
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 2**20), b''):
                    h.update(block)
            result[str(path.relative_to(root))] = [path.stat().st_size, h.hexdigest()]
    return result


def relocate(source, dest, state):
    assert source.is_dir() and not source.is_symlink() and not dest.exists()
    assert read(source / 'completed.json')['status'] == 'complete'
    assert read(source / 'status.json')['status'] == 'complete'
    assert not (source / 'failed.json').exists()
    with (source / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for db in source.glob('*.sqlite'):
            assert_no_open_database(db)
            assert not Path(str(db) + '-wal').exists() or Path(str(db) + '-wal').stat().st_size == 0
        before = inventory(source)
        shutil.copytree(source, dest, symlinks=True)
        assert inventory(dest) == before, 'Copied bytes differ'
        assert inventory(source) == before, 'Source changed during copy'
        for db in source.glob('*.sqlite'):
            assert_no_open_database(db)
        report = dict(source=str(source), destination=str(dest), files=before,
                      status='verified_copy', time=time.time())
        report_path = state / (source.name + '-relocation.json')
        atomic_json(report_path, report)
        shutil.rmtree(source)
        source.symlink_to(dest, target_is_directory=True)
        report['status'] = 'relocated_original_path_linked'
        atomic_json(report_path, report)
        print(json.dumps(dict(relocated=str(source), destination=str(dest),
                              local1_free_GiB=shutil.disk_usage('/local1').free / 2**30)), flush=True)


def main(candidates, state):
    state.mkdir(parents=True, exist_ok=True)
    with (state / 'operation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        selected = sorted(read(candidates), reverse=True)
        if (state / 'started.json').exists():
            raise RuntimeError('Already started; reconcile before restarting')
        atomic_json(state / 'started.json', dict(time=time.time(), pid=os.getpid(), selected=selected))
        (state / 'relocated').mkdir(exist_ok=False)
        for _, run in selected[:3]:
            relocate(Path(run), state / 'relocated' / Path(run).name, state)
        issues = []
        for _, run in selected[3:]:
            if shutil.disk_usage('/local1').free >= 150 * 2**30:
                break
            try:
                result = migrate(Path(run), workers=8)
                assert result['status'] == 'complete'
            except Exception as exc:
                issues.append(dict(run=run, error=str(exc)))
            atomic_json(state / 'progress.json', dict(time=time.time(),
                local1_free_GiB=shutil.disk_usage('/local1').free / 2**30, issues=issues))
        atomic_json(state / 'finished.json', dict(time=time.time(), issues=issues,
                    local1_free_GiB=shutil.disk_usage('/local1').free / 2**30))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    args = parser.parse_args()
    main(args.candidates, args.state)
