"""Validated, offline time downsampling of completed search databases only.

Non-diagnostic rows and full-resolution protected traces are preserved exactly.
Replacement is atomic and occurs only after all validation passes.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import deque
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import time
import zlib

import orjson

SCHEMA = 'diagnostic-time-stride-v1'


def atomic_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def digest_row(h, row):
    for value in row:
        if value is None: raw = b'n'
        elif isinstance(value, bytes): raw = b'b' + value
        elif isinstance(value, str): raw = b's' + value.encode()
        elif isinstance(value, int): raw = b'i' + str(value).encode()
        elif isinstance(value, float): raw = b'f' + value.hex().encode()
        else: raise TypeError(type(value))
        h.update(len(raw).to_bytes(8, 'big')); h.update(raw)


def transformed_blob(blob, stride=10):
    info = orjson.loads(zlib.decompress(blob))
    if 'diagnostic_sampling' in info:
        raise ValueError('Already sampled diagnostic; refuse repeated downsampling')
    times = info.get('time_horizon')
    if not isinstance(times, list) or len(times) < stride * 2:
        return blob, 'preserved_short_or_unknown'
    n = len(times)
    if not all(isinstance(t, (int,float)) and math.isfinite(t) for t in times):
        raise ValueError('Invalid diagnostic times')
    if not all(a < b for a,b in zip(times, times[1:])):
        raise ValueError('Nonmonotone diagnostic times')
    indices = sorted(set(range(0, n, stride)) | {n-1})
    new = dict(info)
    changed = ['time_horizon', 'outputs', 'trajectories']
    for key in ('outputs', 'trajectories'):
        values = info.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f'Unknown {key} layout')
        for scenario in values:
            if not isinstance(scenario, list) or not scenario:
                raise ValueError(f'Unknown {key} scenario layout')
            for series in scenario:
                if not isinstance(series, list) or len(series) != n:
                    raise ValueError(f'Unaligned {key} series')
        new[key] = [[[series[i] for i in indices] for series in scenario] for scenario in values]
    new['time_horizon'] = [times[i] for i in indices]
    new['diagnostic_sampling'] = dict(schema=SCHEMA, original_samples=n,
        retained_indices=indices, stride=stride, includes_endpoint=True,
        stored_loss_resolution='original', purpose='archived_visual_diagnostics')
    raw = orjson.dumps(new, option=orjson.OPT_SORT_KEYS)
    decoded = orjson.loads(raw)
    # Validate every retained value and every untouched field, not just endpoints.
    if decoded != new:
        raise ValueError('Serialization changed diagnostic values (including nonfinite values)')
    for key, value in info.items():
        if key not in changed and decoded[key] != value:
            raise ValueError(f'Changed untouched diagnostic field {key}')
    for key in ('outputs', 'trajectories'):
        for before,after in zip(info[key], decoded[key]):
            for a,b in zip(before,after):
                if b != [a[i] for i in indices]: raise ValueError('Retained sample mismatch')
    return zlib.compress(raw), 'downsampled'


def convert_batch(rows, stride):
    converted=[]
    for rowid,key,blob,protected in rows:
        out,kind = (blob,'protected') if protected else transformed_blob(blob,stride)
        converted.append((rowid,key,out,kind,len(blob)))
    return converted


def ro(path):
    wal=Path(str(path)+'-wal')
    if wal.exists() and wal.stat().st_size:
        raise ValueError(f'Nonempty WAL; require a closed, checkpointed database: {path}')
    return sqlite3.connect(f'file:{path}?mode=ro&immutable=1',uri=True)


def assert_no_open_database(path):
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid(): continue
        try:
            for fd in (proc/'fd').iterdir():
                try: target=os.readlink(fd)
                except OSError: continue
                if target in (str(path),str(path)+'-wal',str(path)+'-shm'):
                    raise RuntimeError(f'Database open in process {proc.name}; defer migration')
        except (PermissionError, FileNotFoundError, ProcessLookupError): pass


def protected_keys(db, run, threshold):
    """Protect all parameterizations of qualifying or historically HOF topologies."""
    topologies=set(); protected=set(); unknown=set()
    for key,payload in db.execute('SELECT key,payload FROM evaluation_summaries'):
        r=orjson.loads(payload); loss=r.get('loss'); topology=r.get('topology_hash')
        if not isinstance(loss,(int,float)) or not math.isfinite(loss):
            unknown.add(key)
        elif loss < threshold:
            protected.add(key)
            if topology: topologies.add(topology)
    for (payload,) in db.execute('SELECT payload FROM designs'):
        r=orjson.loads(payload)
        if r.get('qualified') or (isinstance(r.get('loss'),(int,float)) and r['loss']<threshold):
            if r.get('topology_hash'): topologies.add(r['topology_hash'])
            if r.get('evaluation_cache_key'): protected.add(r['evaluation_cache_key'])
    results=run/'results.sqlite'
    if results.exists():
        with ro(results) as rdb:
            names={r[0] for r in rdb.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if 'hof_snapshot_entries' in names:
                topologies.update(r[0] for r in rdb.execute('SELECT DISTINCT topology_hash FROM hof_snapshot_entries'))
    known=set()
    for key,payload in db.execute('SELECT key,payload FROM evaluation_summaries'):
        known.add(key)
        if orjson.loads(payload).get('topology_hash') in topologies: protected.add(key)
    protected.update(unknown)
    # Missing scalar linkage is not permission to discard full diagnostic detail.
    for (key,) in db.execute('SELECT key FROM diagnostics'):
        if key not in known: protected.add(key)
    return protected,topologies


def migrate(run, workers=4, stride=10, min_free_gib=15):
    run=Path(run).resolve(); source=run/'exploration.sqlite'
    marker=run/'diagnostic_downsampling.json'
    if marker.exists():
        prior=json.loads(marker.read_text())
        if prior.get('status')=='complete': return prior
        raise RuntimeError('An earlier migration record exists; reconcile explicitly')
    done=json.loads((run/'completed.json').read_text())
    status=json.loads((run/'status.json').read_text())
    if done.get('status')!='complete' or status.get('status')!='complete':
        raise ValueError('Only completed runs may be migrated')
    lock=(run/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert_no_open_database(source)
    manifest=json.loads((run/'manifest.json').read_text()); threshold=manifest['config']['quality_threshold']
    initial_stat=source.stat(); temp=run/'exploration.downsample-building.sqlite'
    if temp.exists(): raise RuntimeError('Temporary database already exists')
    if shutil.disk_usage(run).free<min_free_gib*2**30: raise OSError('Insufficient temporary headroom')
    record=dict(schema=SCHEMA,status='preparing',started=time.time(),source=str(source),
        original_bytes=initial_stat.st_size,original_mtime_ns=initial_stat.st_mtime_ns,
        original_inode=initial_stat.st_ino,stride=stride,threshold=threshold,
        protection='All rate-specific records of qualifying and any historically HOF topology; unknown links retained.',
        original_identity=manifest.get('identity'),counts={},tables={},workers=workers)
    atomic_json(marker,record)
    src=ro(source); dst=None
    try:
        if src.execute('SELECT count(*) FROM evaluations').fetchone()[0]:
            raise ValueError('Legacy embedded diagnostics require a separate migration')
        if src.execute("SELECT count(*) FROM sqlite_master WHERE type IN ('view','trigger')").fetchone()[0]:
            raise ValueError('Unexpected database views/triggers')
        protected,topologies=protected_keys(src,run,threshold)
        record.update(protected_keys=len(protected),protected_topologies=len(topologies))
        # Conservatively allow all protected payloads plus 25% of other payloads.
        # Track actual free headroom on every write batch as the run progresses.
        dst=sqlite3.connect(temp); dst.execute('PRAGMA journal_mode=DELETE');dst.execute('PRAGMA synchronous=FULL')
        dst.execute('PRAGMA cache_size=-32768')
        schema=src.execute("SELECT name,sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
        for _,sql in schema:dst.execute(sql)
        for name,_ in schema:
            if name=='diagnostics':continue
            cols=[r[1] for r in src.execute(f'PRAGMA table_info("{name}")')]
            insert=f'INSERT INTO "{name}" (rowid,'+','.join('"'+c+'"' for c in cols)+') VALUES ('+','.join('?' for _ in range(len(cols)+1))+')'
            h=hashlib.sha256();n=0;cursor=src.execute(f'SELECT rowid,* FROM "{name}" ORDER BY rowid')
            while True:
                rows=cursor.fetchmany(1000)
                if not rows:break
                for row in rows:digest_row(h,row)
                dst.executemany(insert,rows);dst.commit();n+=len(rows)
            record['tables'][name]={'rows':n,'sha256':h.hexdigest()}
        record.update(status='converting',diagnostic_total=src.execute('SELECT count(*) FROM diagnostics').fetchone()[0])
        atomic_json(marker,record)
        source_hash=hashlib.sha256(); expected_hash=hashlib.sha256();protected_hash=hashlib.sha256()
        counts={'protected':0,'downsampled':0,'preserved_short_or_unknown':0};before=after=0
        cursor=src.execute('SELECT rowid,key,payload FROM diagnostics ORDER BY rowid')
        pending=deque();last_report=time.monotonic()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            finished=False
            while pending or not finished:
                while not finished and len(pending)<workers*2:
                    rows=cursor.fetchmany(24)
                    if not rows:finished=True;break
                    for row in rows:
                        digest_row(source_hash,row)
                        if row[1] in protected:digest_row(protected_hash,row)
                    pending.append(pool.submit(convert_batch,[(i,k,b,k in protected) for i,k,b in rows],stride))
                if not pending:break
                rows=pending.popleft().result()
                if shutil.disk_usage(run).free<min_free_gib*2**30:raise OSError('Temporary disk headroom exhausted; original preserved')
                for i,key,blob,kind,oldsize in rows:
                    digest_row(expected_hash,(i,key,blob));counts[kind]+=1;before+=oldsize;after+=len(blob)
                dst.executemany('INSERT INTO diagnostics(rowid,key,payload) VALUES (?,?,?)',[(i,k,b) for i,k,b,_,_ in rows]);dst.commit()
                if time.monotonic()-last_report>15:
                    record.update(counts=counts.copy(),processed=sum(counts.values()),original_payload_bytes=before,new_payload_bytes=after,updated=time.time())
                    atomic_json(marker,record);last_report=time.monotonic()
                    print(json.dumps({'run':str(run),'processed':sum(counts.values()),'total':record['diagnostic_total'],'new_GiB':temp.stat().st_size/2**30}),flush=True)
        assert sum(counts.values())==record['diagnostic_total']
        for (sql,) in src.execute("SELECT sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"):
            dst.execute(sql)
        dst.commit()
        record.update(status='validating',counts=counts,processed=sum(counts.values()),
            original_payload_bytes=before,new_payload_bytes=after,
            original_diagnostics_sha256=source_hash.hexdigest(),new_diagnostics_sha256=expected_hash.hexdigest(),
            protected_diagnostics_sha256=protected_hash.hexdigest())
        atomic_json(marker,record)
        integrity=dst.execute('PRAGMA integrity_check').fetchall()
        if integrity!=[('ok',)]:raise ValueError(f'Integrity check failed: {integrity[:3]}')
        for name,expected in record['tables'].items():
            h=hashlib.sha256();n=0
            for row in dst.execute(f'SELECT rowid,* FROM "{name}" ORDER BY rowid'):digest_row(h,row);n+=1
            if {'rows':n,'sha256':h.hexdigest()}!=expected:raise ValueError(f'Changed scalar table: {name}')
        actual=hashlib.sha256();actual_protected=hashlib.sha256();n=0
        for row in dst.execute('SELECT rowid,key,payload FROM diagnostics ORDER BY rowid'):
            digest_row(actual,row);n+=1
            if row[1] in protected:digest_row(actual_protected,row)
        if n!=record['diagnostic_total'] or actual.hexdigest()!=expected_hash.hexdigest() or actual_protected.hexdigest()!=protected_hash.hexdigest():
            raise ValueError('Diagnostic row/hash mismatch')
        src.close();dst.close();src=dst=None
        assert_no_open_database(source)
        current=source.stat()
        if (current.st_ino,current.st_size,current.st_mtime_ns)!=(initial_stat.st_ino,initial_stat.st_size,initial_stat.st_mtime_ns):
            raise RuntimeError('Source changed during migration')
        wal=Path(str(source)+'-wal')
        if wal.exists() and wal.stat().st_size:raise RuntimeError('Source acquired WAL writes')
        record.update(status='validated_ready_to_replace',new_bytes=temp.stat().st_size,validation='All scalar rows, protected blobs and transformed blobs verified; integrity_check ok')
        atomic_json(marker,record)
        os.chmod(temp,initial_stat.st_mode & 0o777)
        with temp.open('rb') as f:os.fsync(f.fileno())
        # No open reader/writer remains; stale empty WAL/SHM belong to the old inode.
        for suffix in ('-wal','-shm'):
            side=Path(str(source)+suffix)
            if side.exists():side.unlink()
        os.replace(temp,source)
        fd=os.open(run,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        record.update(status='complete',finished=time.time(),reclaimed_bytes=initial_stat.st_size-source.stat().st_size)
        atomic_json(marker,record)
        print(json.dumps({'run':str(run),'status':'complete','reclaimed_GiB':record['reclaimed_bytes']/2**30}),flush=True)
        return record
    except BaseException as error:
        record.update(status='failed_original_retained' if source.stat().st_ino==initial_stat.st_ino else 'replacement_requires_reconciliation',error=f'{type(error).__name__}: {error}',updated=time.time())
        atomic_json(marker,record);raise
    finally:
        if src is not None:src.close()
        if dst is not None:dst.close()
        lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();migrate(a.run,a.workers)
