"""Export usage/tool event fields only, without conversations or tool arguments."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
from concurrent.futures import ProcessPoolExecutor
from build_release import PAPER, dump, sha


def parse(row):
    path = Path(row['file'])
    proc = None
    if path.suffix == '.jsonl':
        stream = path.open('rb')
    else:
        proc = subprocess.Popen(['zstd', '-dc', str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stream = proc.stdout
    usages, calls, results = {}, {}, set()
    with stream:
        for line in stream:
            if not any(x in line for x in (b'"usage"', b'"tool/call"', b'"tool/result"')):
                continue
            event = json.loads(line)
            d = event.get('data', {})
            key = (d.get('turn'), d.get('step'))
            kind = event.get('type')
            usage = d.get('usage') if kind == 'assistant/message' else (
                d['chunk'].get('usage') if kind == 'assistant/chunk' and d.get('chunk', {}).get('type') == 'usage' else None)
            if usage:
                usages[key + (tuple(sorted(usage.items())),)] = dict(turn=key[0], step=key[1], usage=usage)
            if kind == 'tool/call':
                k = key + (d['callId'],)
                assert k not in calls or calls[k]['name'] == d['name']
                calls[k] = dict(turn=key[0], step=key[1], callId=d['callId'], name=d['name'])
            if kind == 'tool/result':
                cid = d.get('message', {}).get('source', {}).get('callId')
                if cid is not None:
                    results.add(key+(cid,))
    if proc:
        error = proc.stderr.read()
        assert proc.wait() == 0, error
    return dict(group=row['group'], seed=int(row['seed']), session=row['session'],
                source_sha256=sha(path), usage=list(usages.values()),
                tools=[dict(v, result_recorded=k in results) for k, v in calls.items()],
                orphan_results=len(results-set(calls)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    assert (args.output/'RELEASE_BUILD').exists()
    with (PAPER/'token_costs/sessions.csv').open() as f:
        sessions = list(csv.DictReader(f))
    target = args.output/'data/llm_usage/session_events.jsonl.gz'
    target.parent.mkdir(parents=True, exist_ok=True)
    count = usage = calls = 0
    with ProcessPoolExecutor(max_workers=4) as pool, gzip.open(target.with_suffix('.partial'), 'wt') as out:
        for record in pool.map(parse, sessions, chunksize=8):
            out.write(json.dumps(record, separators=(',', ':'))+'\n')
            count += 1
            usage += len(record['usage'])
            calls += len(record['tools'])
            if count % 500 == 0:
                print(f'{count}/{len(sessions)} sessions exported', flush=True)
    target.with_suffix('.partial').replace(target)
    dump(target.parent/'export.json', dict(sessions=count, usage_events=usage, tool_calls=calls,
         sha256=sha(target), omitted=['conversation text', 'tool arguments and contents', 'credentials']))
    print(count, 'sessions;', usage, 'usage events;', calls, 'tool calls', flush=True)


if __name__ == '__main__':
    main()
