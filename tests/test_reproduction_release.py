"""Check relocation without requiring the experiment archive or manuscript."""
import importlib.util
import json
from pathlib import Path

import pytest


def runner():
    path = Path(__file__).resolve().parents[1] / 'reproduce/reproduce.py'
    spec = importlib.util.spec_from_file_location('release_runner_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_analysis_uses_checkout_code_and_leaves_release_unchanged(tmp_path):
    r = runner()
    r.ROOT = tmp_path / 'release'
    r.CODE_ROOT = tmp_path / 'code'
    frozen = r.ROOT / 'analysis/task'
    code = r.CODE_ROOT / 'analysis/task'
    frozen.mkdir(parents=True)
    code.mkdir(parents=True)
    (frozen / 'plot.py').write_text('old source')
    (frozen / 'unused.py').write_text('unneeded source')
    (frozen / 'inputs.json').write_text('{"loss": 1}')
    (code / 'plot.py').write_text('checkout source')
    work = r.prepare_analysis(tmp_path / 'output')
    assert (work / 'analysis/task/plot.py').read_text() == 'checkout source'
    assert not (work / 'analysis/task/unused.py').exists()
    assert (work / 'analysis/task/inputs.json').read_text() == '{"loss": 1}'
    assert (frozen / 'plot.py').read_text() == 'old source'
    assert not (work / 'paper').exists()
    assert r.resolve(work / 'analysis/task/inputs.json').is_file()


def test_recorded_paths_require_released_mapping(tmp_path):
    r = runner()
    r.ROOT = tmp_path / 'release'
    (r.ROOT / 'manifest').mkdir(parents=True)
    (r.ROOT / 'manifest/path_map.json').write_text(json.dumps({'/old/run': 'run-id'}))
    assert r.resolve('/old/run/metrics.jsonl') == r.ROOT / 'data/runs/run-id/metrics.jsonl'
    with pytest.raises(FileNotFoundError, match='No released mapping'):
        r.resolve('/old/unreleased/metrics.jsonl')
