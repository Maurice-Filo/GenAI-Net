"""Portable, opt-in live check of the paper's Harness Decider/Writer protocol.

Default mode builds the RPA evaluator and emits a configuration patch, without
starting Harness or contacting an LLM. --live explicitly requests one batch.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dsh-home', type=Path, required=True)
    p.add_argument('--backend', choices=('deepseek', 'local'), default='deepseek')
    p.add_argument('--model', help='Defaults to deepseek-v4-flash; required for local serving.')
    p.add_argument('--endpoint', help='Local OpenAI-compatible base URL, including /v1.')
    p.add_argument('--context-window', type=int, default=32768)
    p.add_argument('--max-output-tokens', type=int, default=4096)
    p.add_argument('--reasoning', choices=('low', 'high'), default='high')
    p.add_argument('--role-timeout', type=float, default=900)
    p.add_argument('--candidates', type=int, default=1)
    p.add_argument('--output', type=Path, help='New directory; defaults to a fresh temporary directory.')
    p.add_argument('--live', action='store_true', help='Send one Decider/Writer batch to the selected model.')
    a = p.parse_args()
    if a.backend == 'local' and (not a.endpoint or not a.model):
        p.error('Local serving requires --endpoint and --model.')
    if a.backend == 'deepseek' and a.endpoint:
        p.error('--endpoint is for the local backend.')
    if a.candidates < 1 or a.role_timeout <= 0:
        p.error('Candidate count and role timeout must be positive.')
    if not (a.dsh_home.expanduser() / 'profiles/headless/package.json').is_file():
        p.error('Initialize the headless profile first; see tools/HARNESS_SETUP.md.')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(key, '1')
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'genainet-mpl'))

    from RL4CRN.llm.harness_client import HarnessLLMClient
    from RL4CRN.llm.harness_runner import HarnessDeciderWriterCRNGraph
    from RL4CRN.llm.workspace_loss_protocol import PROTOCOL
    from RL4CRN.search_exploration.evaluation import build

    class SetupClient(HarnessLLMClient):
        def _profile_patch_text(self):
            patch = super()._profile_patch_text()
            if self.openai_compatible_base_url is not None and a.reasoning == 'high':
                # Same provider override used by the matched Qwen campaign.
                patch = patch.replace('reasoning: low', 'reasoning: high').replace('low: low', 'high: high')
            return patch

    output = a.output.expanduser().resolve() if a.output else Path(tempfile.mkdtemp(prefix='genainet-harness-check-'))
    if a.output:
        output.mkdir(parents=True, exist_ok=False)
    config = json.loads((REPO / 'comparisons/rpa_search/configs/rpa_100k.json').read_text())
    config['rpa']['solver'] = 'CVODE'
    config['search']['rate_constant_range'] = [.001, 100.]
    evaluator, _, _ = build('rpa', config, .001, 100.)
    client = SetupClient(
        workspace_root=output / 'workspaces', dsh_home=a.dsh_home,
        provider='deepseek-official' if a.backend == 'deepseek' else 'local-openai',
        model=a.model or 'deepseek-v4-flash', openai_compatible_base_url=a.endpoint,
        local_context_window=a.context_window, local_max_output_tokens=a.max_output_tokens,
        timeout_seconds=a.role_timeout, global_concurrency=1,
        candidate_validation_policy='independent-members', harness_protocol=PROTOCOL,
        extra_environment={'CHOKIDAR_USEPOLLING': '1', 'CHOKIDAR_INTERVAL': '1000'},
    )
    if shutil.which(client.command[0]) is None:
        raise RuntimeError(f'Harness executable not found: {client.command[0]}')
    wrapper = REPO / 'tools/dsh_bwrap_resolved_tmp.sh'
    if shutil.which('bwrap') is None or not os.access(wrapper, os.X_OK):
        raise RuntimeError('Install bubblewrap and make tools/dsh_bwrap_resolved_tmp.sh executable.')
    prompt_path = REPO / 'comparisons/search_exploration/prompts/rpa-evidence-exploitation-v1.md'
    patch_path = output / 'harness.patch.yml'
    patch_path.write_text(client._profile_patch_text())
    metadata = dict(backend=a.backend, provider=client.provider, model=client.model,
                    endpoint=a.endpoint, command=list(client.command), dsh_home=str(client.dsh_home),
                    protocol=PROTOCOL, role_timeout_seconds=a.role_timeout,
                    candidates=a.candidates, live=a.live, task_prompt=str(prompt_path),
                    local_context_window=a.context_window if a.backend == 'local' else None,
                    local_max_output_tokens=a.max_output_tokens if a.backend == 'local' else None,
                    local_reasoning=a.reasoning if a.backend == 'local' else None)
    (output / 'setup.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('Output:', output, flush=True)
    print('Offline configuration check:', shlex.join([*client.command, '--profile', 'headless',
                                                    '--patch', str(patch_path), '--dump-config']), flush=True)
    if not a.live:
        print('Offline preparation passed. No Harness process or model request was started.')
        return
    graph = HarnessDeciderWriterCRNGraph(client=client, evaluator=evaluator,
                                       writer_retry_limit=0, max_workspace_evaluations=0,
                                       max_literature_searches=0)
    started = time.monotonic()
    try:
        result = graph.run_round(task_description=prompt_path.read_text(), num_candidates=a.candidates,
                                step=0, hall_of_fame_iter=None,
                                forbidden_topologies_text='No topology exclusions supplied.',
                                sil_feedback_text='Initial request: search context withheld.',
                                jsonl_path=output / 'evaluations.jsonl')
        summary = dict(status='completed', elapsed_seconds=time.monotonic()-started,
                       workspace=str(client.last_workspace.path),
                       role_invocations=client.last_workspace.call_count,
                       returned_evaluations=len(result.evaluations),
                       valid_evaluations=sum(bool(e.valid) for e in result.evaluations),
                       losses=[e.loss for e in result.evaluations],
                       response_validation=result.response_validation)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary, indent=2))
    except Exception as exc:
        (output / 'failure.json').write_text(json.dumps(dict(
            status='failed', type=type(exc).__name__, elapsed_seconds=time.monotonic()-started,
            workspace=str(client.last_workspace.path) if client.last_workspace else None), indent=2)+'\n')
        raise


if __name__ == '__main__':
    main()
