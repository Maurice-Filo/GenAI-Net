"""Execute one prepared, hash-checked reference runner in its own process."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    job_path = Path(sys.argv[1]).resolve()
    job = json.loads(job_path.read_text())
    base = job_path.parent
    source = base / 'source'
    for name, expected in job['source_hashes'].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Prepared source changed: {name}')
    for name, expected in job['input_hashes'].items():
        if hashlib.sha256((base / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Prepared input changed: {name}; prepare a new campaign to change settings')
    os.chdir(source)
    sys.path.insert(0, str(source))
    if job['harness']:
        from RL4CRN.llm.harness_client import HarnessLLMClient
        from search_transport import install, install_manifest_metadata
        from RL4CRN.search_exploration import store
        install(HarnessLLMClient, job['provider'], job['model'])
        install_manifest_metadata(store, job['provider'], job['model'])
    # Reference metadata remains in reference_manifest.json. New run identities
    # include the effective routing, with an additional execution sidecar.
    sys.argv = [job['module'], *job['argv']]
    try:
        runpy.run_module(job['module'], run_name='__main__')
    finally:
        output = base / 'output'
        if output.is_dir():
            (output / 'effective_execution.json').write_text(json.dumps({
                'reference_run': job['reference_run'], 'experiment': job['experiment'],
                'seed': job['seed'], 'provider': job['provider'], 'model': job['model'],
                'overrides': job['overrides'], 'source_hashes': job['source_hashes'],
                'routing': 'Harness headless configuration; archived provider/endpoint overridden',
            }, indent=2) + '\n')


if __name__ == '__main__':
    main()
