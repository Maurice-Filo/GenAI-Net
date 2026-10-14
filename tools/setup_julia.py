#!/usr/bin/env python3
"""Install/check the manifest's Julia version and exercise the rate-fitting worker."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = 'https://julialang-s3.julialang.org/'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def manifest_version(project):
    for name in ('Project.toml', 'Manifest.toml', 'src/CRNOptim.jl'):
        if not (project / name).is_file():
            raise ValueError(f'Missing {project / name}; supply the prepared SynthEvo project.')
    match = re.search(r'^julia_version\s*=\s*"(\d+\.\d+\.\d+)"',
                      (project / 'Manifest.toml').read_text(), re.MULTILINE)
    if not match:
        raise ValueError('Manifest.toml must specify a stable julia_version.')
    return match.group(1)


def binary_version(binary):
    result = subprocess.run([str(binary), '--startup-file=no', '-e', 'print(VERSION)'],
                            capture_output=True, text=True, timeout=30, check=True)
    return result.stdout.strip()


def select_archive(metadata, version, machine):
    triplet = {'x86_64': 'x86_64-linux-gnu', 'aarch64': 'aarch64-linux-gnu'}.get(machine)
    if not triplet or platform.system() != 'Linux':
        raise ValueError('Automatic Julia installation supports Linux x86_64/aarch64.')
    matches = [f for f in metadata.get(version, {}).get('files', [])
               if f.get('triplet') == triplet and f.get('extension') == 'tar.gz']
    if len(matches) != 1:
        raise ValueError(f'No unique official Julia {version} archive for {triplet}.')
    item = matches[0]
    if not item['url'].startswith(OFFICIAL) or not re.fullmatch(r'[0-9a-f]{64}', item['sha256']):
        raise ValueError('Unexpected Julia download URL or checksum.')
    return item


def install_julia(prefix, version):
    target = prefix / f'julia-{version}'
    if target.exists():
        raise ValueError(f'{target} exists but has no usable matching Julia; inspect it before retrying.')
    with urllib.request.urlopen(OFFICIAL + 'bin/versions.json', timeout=60) as response:
        item = select_archive(json.load(response), version, platform.machine())
    with tempfile.TemporaryDirectory(prefix='download-', dir=prefix) as work:
        work = Path(work)
        archive = work / 'julia.tar.gz'
        print(f'Downloading Julia {version} from the official distribution…', flush=True)
        with urllib.request.urlopen(item['url'], timeout=120) as response, archive.open('wb') as out:
            shutil.copyfileobj(response, out)
        if sha256(archive) != item['sha256']:
            raise ValueError('Julia archive SHA-256 mismatch; installation aborted.')
        subprocess.run(['tar', '-xzf', str(archive), '--no-same-owner', '-C', str(work)], check=True)
        extracted = work / f'julia-{version}'
        if binary_version(extracted / 'bin/julia') != version:
            raise ValueError('Downloaded Julia reports the wrong version.')
        extracted.rename(target)
    return target / 'bin/julia'


def smoke_test(project, binary):
    # Check the actual Python-to-Julia worker, analytic sensitivities and AdaGrad.
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from RL4CRN.parameter_optimization import CRNSpec, Experiment, JuliaOptimizer, Reaction
    times = np.linspace(0, 3, 61)
    crn = CRNSpec(('X',), (), (Reaction((1,), (0,)),))
    experiment = Experiment(times, [[1.]], [[]], [[1.]],
                            np.exp(-.7 * times)[None, None, :], loss='l2')
    with JuliaOptimizer(project=project, julia=str(binary), startup_timeout=600) as worker:
        model = worker.prepare(crn, experiment)
        score = worker.evaluate(model, [.3], trajectories=True)
        output = np.exp(-.3 * times)
        error = output - np.exp(-.7 * times)
        np.testing.assert_allclose(np.asarray(score['outputs'])[0, 0], output, atol=2e-7, rtol=0)
        np.testing.assert_allclose(score['loss'], np.mean(error**2), atol=2e-8, rtol=0)
        np.testing.assert_allclose(score['gradient'][0], np.mean(2 * error * -times * output),
                                   atol=2e-7, rtol=0)
        fit = worker.optimize(model, [[.3]], steps=3, method='adagrad', learning_rate=.1)['results'][0]
        if fit['status'] != 'ok' or not np.isfinite(fit['best_loss']) or fit['best_loss'] >= score['loss']:
            raise RuntimeError(f'AdaGrad smoke test failed: {fit}')
        return dict(worker=worker.info, initial_loss=score['loss'], best_loss=fit['best_loss'],
                    evaluations=fit['evaluations'], analytic_gradient_checked=True)


def setup(args):
    project = args.project.expanduser().resolve()
    version = manifest_version(project)
    # Fail early before downloads if the selected Python environment is incomplete.
    sys.path.insert(0, str(ROOT))
    from RL4CRN.parameter_optimization import JuliaOptimizer  # noqa: F401
    prefix = args.prefix.expanduser().resolve()
    prefix.mkdir(parents=True, exist_ok=True)
    with (prefix / '.setup.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(f'[1/3] Required Julia version: {version}', flush=True)
        installed = prefix / f'julia-{version}/bin/julia'
        candidate = args.julia or (str(installed) if installed.is_file() else shutil.which('julia'))
        binary = Path(shutil.which(candidate) or candidate).absolute() if candidate else None
        actual = binary_version(binary) if binary and binary.is_file() else None
        if actual != version:
            if not args.install_julia:
                raise ValueError(f'Found Julia {actual or "not installed"}; need {version}. '
                                 'Run again with --install-julia for a local installation.')
            binary = install_julia(prefix, version)
        depot = args.depot.expanduser().resolve() if args.depot else None
        if depot:
            depot.mkdir(parents=True, exist_ok=True)
            os.environ['JULIA_DEPOT_PATH'] = str(depot)
        os.environ.update(JULIA_NUM_THREADS='1', JULIA_NUM_PRECOMPILE_TASKS=str(args.precompile_tasks),
                          JULIA_PKG_PRECOMPILE_AUTO='0', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        before = {name: sha256(project / name) for name in ('Project.toml', 'Manifest.toml')}
        print('[2/3] Installing and precompiling locked dependencies; first setup can take several minutes.',
              flush=True)
        # Instantiate a copy: Pkg never writes to the user's frozen input project.
        with tempfile.TemporaryDirectory(prefix='project-', dir=prefix) as work:
            copied = Path(work) / 'SynthEvo'
            shutil.copytree(project, copied)
            subprocess.run([str(binary), '--startup-file=no', f'--project={copied}', '-e',
                            'using Pkg; Pkg.instantiate(); Pkg.precompile()'], check=True)
            if any(sha256(copied / name) != value for name, value in before.items()):
                raise RuntimeError('Pkg changed the copied lockfiles; original files are untouched. '
                                   'Inspect the Julia/project compatibility before running searches.')
            print('[3/3] Checking dynamics, analytic gradient and a three-step AdaGrad fit…', flush=True)
            result = smoke_test(copied, binary)
        if any(sha256(project / name) != value for name, value in before.items()):
            raise RuntimeError('Input project changed during setup.')
        result.update(project=str(project), lockfile_sha256=before, python=sys.executable,
                      julia=str(binary), depot=os.environ.get('JULIA_DEPOT_PATH', 'Julia default'))
        (prefix / 'setup_report.json').write_text(json.dumps(result, indent=2) + '\n')
        env = f'export PATH={shlex.quote(str(binary.parent))}:"$PATH"\n'
        if os.environ.get('JULIA_DEPOT_PATH'):
            env += f'export JULIA_DEPOT_PATH={shlex.quote(os.environ["JULIA_DEPOT_PATH"])}\n'
        (prefix / 'env.sh').write_text('# Generated by tools/setup_julia.py\n' + env)
        print(f'Passed: loss {result["initial_loss"]:.6g} → {result["best_loss"]:.6g}. '
              'Original lockfiles unchanged.\n'
              f'Load Julia for search commands:\n  source {shlex.quote(str(prefix / "env.sh"))}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True, help='Prepared numerical_runtime/SynthEvo directory')
    parser.add_argument('--prefix', type=Path, default=Path.home() / '.local/share/fullduplex-julia')
    parser.add_argument('--install-julia', action='store_true', help='Download the exact manifest version if needed')
    parser.add_argument('--julia', help='Existing Julia executable (must match the manifest)')
    parser.add_argument('--depot', type=Path, help='Optional separate Julia package cache; default: existing Julia depot')
    parser.add_argument('--precompile-tasks', type=int, default=2, help='Parallel precompilation tasks (default: 2)')
    args = parser.parse_args()
    if args.precompile_tasks < 1:
        parser.error('--precompile-tasks must be positive')
    try:
        setup(args)
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(1, f'Julia setup failed: {exc}\n')


if __name__ == '__main__':
    main()
