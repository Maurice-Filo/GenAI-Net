"""Setup must reject incompatible runtimes and protect the scientific lockfiles."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('setup_julia', Path(__file__).resolve().parents[1] / 'tools/setup_julia.py')
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


def inputs(tmp_path):
    project = tmp_path / 'reference'
    (project / 'src').mkdir(parents=True)
    (project / 'src/CRNOptim.jl').write_text('# fixture')
    (project / 'Project.toml').write_text('name = "fixture"\n')
    (project / 'Manifest.toml').write_text('julia_version = "1.12.6"\n')
    binary = tmp_path / 'julia'
    binary.touch()
    return SimpleNamespace(project=project, prefix=tmp_path/'installation', julia=str(binary),
                           install_julia=False, depot=None, precompile_tasks=1)


def test_version_mismatch_fails_before_package_installation(tmp_path, monkeypatch):
    args = inputs(tmp_path)
    monkeypatch.setattr(setup, 'binary_version', lambda binary: '1.12.1')
    monkeypatch.setattr(setup.subprocess, 'run', lambda *a, **kw: pytest.fail('Must not instantiate'))
    with pytest.raises(ValueError, match='--install-julia'):
        setup.setup(args)
    assert not (args.prefix / 'env.sh').exists()


def test_package_rewrite_does_not_touch_original_lockfiles(tmp_path, monkeypatch):
    args = inputs(tmp_path)
    before = (args.project / 'Manifest.toml').read_bytes()
    monkeypatch.setattr(setup, 'binary_version', lambda binary: '1.12.6')
    def rewrite_copy(command, **kw):
        copy = Path(next(x.split('=', 1)[1] for x in command if x.startswith('--project=')))
        assert copy != args.project
        (copy / 'Manifest.toml').write_text('changed by Pkg')
    monkeypatch.setattr(setup.subprocess, 'run', rewrite_copy)
    # setup assigns process environment; restore it after this test.
    for key in ('JULIA_NUM_THREADS', 'JULIA_NUM_PRECOMPILE_TASKS', 'JULIA_PKG_PRECOMPILE_AUTO',
                'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
        monkeypatch.setenv(key, '1')
    with pytest.raises(RuntimeError, match='original files are untouched'):
        setup.setup(args)
    assert (args.project / 'Manifest.toml').read_bytes() == before
    assert not (args.prefix / 'env.sh').exists()
