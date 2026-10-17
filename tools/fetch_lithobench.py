"""Resume official LithoBench data download, extract training inputs, verify pairs.

Run from any directory with the project Python. See data/install_status.json and
work/lithobench-download/download.log for background progress.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import threading
import time

import requests
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
WORK = PROJECT/'work'/'lithobench-download'
DATA = PROJECT/'data'
URL = 'https://drive.usercontent.google.com/download'
PARAMS = dict(id='1MzYiRRxi8Eu2L6WHCfZ1DtRnjVNOl4vu', export='download', confirm='t')
CHUNK = 8*1024*1024
STATUS = DATA/'install_status.json'
SUBSETS = ('MetalSet', 'ViaSet', 'StdMetal', 'StdContact', 'StdContactTest')


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf8')
    os.replace(temporary,path)


def status(stage, **details):
    record = dict(stage=stage,pid=os.getpid(),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**details)
    atomic_json(STATUS, record)
    print(json.dumps(record),flush=True)


def download(workers):
    archive, partial, statefile = WORK/'lithodata.tar.gz', WORK/'lithodata.tar.gz.part', WORK/'download.json'
    if archive.exists():
        return archive
    completed = set()
    if statefile.exists():
        state = json.loads(statefile.read_text())
        if state['chunk'] != CHUNK:
            raise RuntimeError('Resume chunk size differs')
        total, completed = state['total'], set(state['done'])
    else:
        for attempt in range(8):
            try:
                with requests.get(URL,params=PARAMS,headers={'Range':'bytes=0-0'},stream=True,timeout=(20,60)) as r:
                    r.raise_for_status()
                    total = int(r.headers['Content-Range'].split('/')[-1])
                    break
            except Exception:
                if attempt == 7:
                    raise
                time.sleep(min(2**attempt,30))
    if not partial.exists():
        if completed:
            raise RuntimeError('Resume file missing')
        partial.touch()
        if os.name == 'nt':
            subprocess.run(['fsutil','sparse','setflag',str(partial)],check=True,creationflags=subprocess.CREATE_NO_WINDOW)
    count = (total+CHUNK-1)//CHUNK
    atomic_json(statefile,dict(total=total,chunk=CHUNK,done=sorted(completed)))
    lock, stop = threading.Lock(), threading.Event()
    transferred, initial = [0], sum(min(CHUNK,total-i*CHUNK) for i in completed)
    begin = time.monotonic()
    def report():
        while not stop.is_set():
            with lock:
                done_bytes = sum(min(CHUNK,total-i*CHUNK) for i in completed)
                current = transferred[0]
                done_count = len(completed)
            status('downloading',total_bytes=total,completed_bytes=done_bytes,
                   percent=round(100*done_bytes/total,3),completed_chunks=done_count,
                   total_chunks=count,workers=workers,received_this_run=current,
                   MBps=round(current/max(time.monotonic()-begin,1)/1e6,3))
            stop.wait(30)
    monitor = threading.Thread(target=report,daemon=True)
    monitor.start()
    def fetch(index):
        start, end = index*CHUNK, min((index+1)*CHUNK,total)-1
        for attempt in range(12):
            try:
                with requests.get(URL,params=PARAMS,headers={'Range':f'bytes={start}-{end}'},
                                  stream=True,timeout=(20,90)) as r:
                    r.raise_for_status()
                    if r.status_code != 206 or r.headers.get('Content-Range') != f'bytes {start}-{end}/{total}':
                        raise RuntimeError('Unexpected range response; preserving previous completed blocks')
                    data = bytearray()
                    for block in r.iter_content(256*1024):
                        data.extend(block)
                        with lock:
                            transferred[0] += len(block)
                    if len(data) != end-start+1:
                        raise RuntimeError('Incomplete range')
                    if index == 0 and data[:2] != b'\x1f\x8b':
                        raise RuntimeError('Source is not gzip')
                with partial.open('r+b') as stream:
                    stream.seek(start)
                    stream.write(data)
                return index
            except Exception as e:
                print(f'retry chunk={index} attempt={attempt+1} error={type(e).__name__}: {e}',flush=True)
                if attempt == 11:
                    raise
                time.sleep(min(2**attempt,30))
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            tasks = [executor.submit(fetch,i) for i in range(count) if i not in completed]
            for future in as_completed(tasks):
                index = future.result()
                with lock:
                    completed.add(index)
                    state = dict(total=total,chunk=CHUNK,done=sorted(completed))
                atomic_json(statefile,state)
    finally:
        stop.set()
        monitor.join()
    if partial.stat().st_size != total:
        raise RuntimeError('Final archive size differs from remote file size')
    partial.replace(archive)
    status('downloaded',total_bytes=total)
    return archive


def extract(archive):
    import gzip
    counts, skipped = Counter(), Counter()
    status('extracting',archive=str(archive))
    # Streaming extraction avoids materializing a second full unpacked archive.
    # Only regular whitelisted files are written; links and path traversal fail.
    with gzip.open(archive,'rb') as decoded:
        with tarfile.open(fileobj=decoded,mode='r|') as tar:
            for member in tar:
                if not member.isfile():
                    continue
                path = PurePosixPath(member.name)
                if path.is_absolute() or '..' in path.parts:
                    raise RuntimeError('Unsafe archive member')
                for i, part in enumerate(path.parts):
                    if part in SUBSETS:
                        subset, remainder = part, path.parts[i+1:]
                        break
                else:
                    skipped[path.parts[0]] += 1
                    continue
                if len(remainder) != 2 or remainder[0] not in ('target','pixelILT','glp'):
                    skipped[subset] += 1
                    continue
                if Path(remainder[1]).suffix.lower() not in ('.png','.glp'):
                    continue
                destination = DATA/subset/Path(*remainder)
                if not destination.resolve().is_relative_to(DATA.resolve()):
                    raise RuntimeError('Destination outside data folder')
                destination.parent.mkdir(parents=True,exist_ok=True)
                if destination.exists():
                    if destination.stat().st_size != member.size:
                        raise FileExistsError(f'Existing data differs: {destination}')
                else:
                    source = tar.extractfile(member)
                    temporary = destination.with_suffix(destination.suffix+'.partial')
                    with temporary.open('wb') as stream:
                        shutil.copyfileobj(source,stream,1024*1024)
                    temporary.replace(destination)
                counts[f'{subset}/{remainder[0]}'] += 1
                if sum(counts.values()) % 2000 == 0:
                    status('extracting',installed_files=sum(counts.values()),counts=dict(counts))
        # Consume remaining decoded bytes to check the gzip CRC and size trailer.
        while decoded.read(8*1024*1024):
            pass
    return counts, skipped


def verify(counts, skipped):
    subsets = {}
    for name in SUBSETS:
        root = DATA/name
        if not root.exists():
            continue
        targets = {p.name for p in (root/'target').glob('*.png')}
        masks = {p.name for p in (root/'pixelILT').glob('*.png')}
        paired = targets & masks
        for folder in ('target','pixelILT'):
            examples = sorted((root/folder).glob('*.png'))[:2]
            for path in examples:
                with Image.open(path) as image:
                    image.verify()
        subsets[name] = dict(targets=len(targets),masks=len(masks),paired=len(paired),
                             unpaired_targets=len(targets-masks),unpaired_masks=len(masks-targets),
                             path=str(root))
        if name in ('MetalSet','ViaSet') and (not paired or targets != masks):
            raise RuntimeError(f'Incomplete {name} target/mask pairing: {subsets[name]}')
    for name in ('MetalSet','ViaSet'):
        if name not in subsets:
            raise RuntimeError(f'Missing required training dataset {name}')
    result = dict(source='https://github.com/shelljane/lithobench#download-the-dataset',
                  google_drive_id=PARAMS['id'],subsets=subsets,extracted=dict(counts),skipped=dict(skipped),
                  note='PNG samples opened successfully; name pairing checked. No training performed.')
    atomic_json(DATA/'download_manifest.json',result)
    status('complete',**result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=12)
    args = parser.parse_args()
    WORK.mkdir(parents=True,exist_ok=True)
    DATA.mkdir(exist_ok=True)
    try:
        archive = download(args.workers)
        verify(*extract(archive))
    except Exception as e:
        status('failed',error=f'{type(e).__name__}: {e}',
               resume_command='python tools/fetch_lithobench.py')
        raise


if __name__ == '__main__':
    main()
