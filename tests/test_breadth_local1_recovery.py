import json
from pathlib import Path

import pytest
from comparisons.search_exploration import recover_breadth_local1 as recovery


def test_preserves_interrupted_attempts_and_keeps_completed_runs(tmp_path, monkeypatch):
    root=tmp_path/'campaign';(root/'runs').mkdir(parents=True);(root/'logs').mkdir()
    complete=root/'runs'/'done';complete.mkdir()
    for name in ('completed','status'):(complete/f'{name}.json').write_text('{"status":"complete"}')
    failed=root/'runs'/'interrupted';failed.mkdir();(failed/'evidence').write_text('original')
    (root/'logs/interrupted.log').write_text('disk full')
    original_resolve=Path.resolve
    def resolve(path,*a,**k):
        resolved=original_resolve(path,*a,**k)
        return Path('/local1/rossin')/resolved.relative_to(tmp_path) if resolved.is_relative_to(tmp_path) else resolved
    monkeypatch.setattr(Path,'resolve',resolve)
    plan={'jobs':[{'name':n,'status':'queued','seed':i} for i,n in enumerate(['done','interrupted','new'])]}
    jobs=recovery.prepare_jobs(root,plan)
    assert [j['status'] for j in jobs]==['complete','queued','queued']
    assert complete.exists() and not failed.exists()
    assert (root/'interrupted_attempts_20261005/interrupted/evidence').read_text()=='original'
    assert (root/'interrupted_attempts_20261005/interrupted.log').read_text()=='disk full'
    assert jobs[1]['restart_reason'].startswith('Disk-full')
    assert not (root/'runs/new').exists()


def test_refuses_local0_output_even_if_not_started(tmp_path):
    root=tmp_path/'campaign';(root/'runs').mkdir(parents=True)
    with pytest.raises(AssertionError):
        recovery.prepare_jobs(root,{'jobs':[{'name':'seed01'}]})
