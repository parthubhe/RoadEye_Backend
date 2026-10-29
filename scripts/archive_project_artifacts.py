"""Audit, back up, and verify project artifacts before a separately approved cleanup.

Does not delete anything. Hugging Face authentication uses the existing local login.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import zipfile
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'cleanup_archive'
REPO='AkumaDachi/roadeeye-sewer-defects-v5'
WEIGHTS={'.pt','.pth','.ckpt','.onnx','.safetensors','.tflite','.engine'}

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def audit():
    STATE.mkdir(exist_ok=True)
    sizes=defaultdict(int);counts=defaultdict(int);weights=[]
    for directory,dirs,files in os.walk(ROOT):
        dirs[:]=[d for d in dirs if d not in {'.git','.aws','.codex','.agents','.kilo'}]
        base=Path(directory)
        for name in files:
            p=base/name;rel=p.relative_to(ROOT)
            try:size=p.stat().st_size
            except OSError:continue
            key=str(Path(*rel.parts[:2])) if rel.parts[0]=='dataset_crack' and len(rel.parts)>2 else rel.parts[0]
            sizes[key]+=size;counts[key]+=1
            if p.suffix.lower() in WEIGHTS and rel.parts[0] not in {'venv','env','cleanup_archive'}:
                weights.append({'path':str(rel).replace('\\','/'),'bytes':size,'sha256':digest(p)})
    result={'created':datetime.now().isoformat(),'root':str(ROOT),'logical_bytes_by_path':dict(sizes),
            'file_counts_by_path':dict(counts),'weights':weights,
            'note':'Logical sizes include NTFS hard-link aliases; physical space reclaimed can be smaller.'}
    (STATE/'local_inventory.json').write_text(json.dumps(result,indent=2))
    for key,size in sorted(sizes.items(),key=lambda x:-x[1])[:35]:print(f'{size/2**30:8.2f} GiB  {counts[key]:7} files  {key}',flush=True)
    unique={r['sha256']:r['bytes'] for r in weights}
    print(f'Weights: {len(weights)} files; {len(unique)} unique hashes; {sum(unique.values())/2**30:.2f} GiB unique',flush=True)

def remote_audit():
    from huggingface_hub import HfApi
    inventory=json.loads((STATE/'local_inventory.json').read_text())
    api=HfApi();info=api.repo_info(REPO,repo_type='dataset',files_metadata=True)
    hashes={}
    for f in info.siblings:
        if f.lfs:
            sha=f.lfs.sha256 if hasattr(f.lfs,'sha256') else f.lfs['sha256']
            hashes.setdefault(sha,[]).append(f.rfilename)
    for r in inventory['weights']:r['remote_paths']=hashes.get(r['sha256'],[])
    inventory['remote_revision']=info.sha
    (STATE/'remote_inventory.json').write_text(json.dumps(inventory,indent=2))
    missing={r['sha256']:r for r in inventory['weights'] if not r['remote_paths']}
    print(f'{len(missing)} unique missing checkpoints, {sum(r["bytes"] for r in missing.values())/2**30:.2f} GiB',flush=True)
    for r in missing.values():print(f'{r["bytes"]/2**20:.1f} MiB {r["path"]}',flush=True)

def backup_models():
    os.environ['HF_HUB_DISABLE_XET']='1'
    from huggingface_hub import HfApi
    api=HfApi()
    inventory=json.loads((STATE/'remote_inventory.json').read_text())
    groups={}
    for row in inventory['weights']:groups.setdefault(row['sha256'],[]).append(row)
    def priority(item):
        rows=item[1];p=rows[0]['path']
        return (0 if any('distillation' in r['path'] or 'hybrid' in r['path'] for r in rows) else
                1 if any(Path(r['path']).name.startswith('best') or 'checkpoint_best' in r['path'] for r in rows) else 2,rows[0]['bytes'])
    groups=sorted(groups.items(),key=priority)
    log=STATE/'uploaded_models.jsonl'
    start=time.monotonic()
    for index,(sha,rows) in enumerate(groups,1):
        remote=next((r['remote_paths'] for r in rows if r['remote_paths']),[])
        source=ROOT/rows[0]['path']
        if not remote:
            target=f'models/archive_20261029/weights/{sha[:16]}/{source.name}'
            print(f'[{index}/{len(groups)}] Upload {source.name}: {rows[0]["bytes"]/2**20:.1f} MiB ({rows[0]["path"]})',flush=True)
            existing=api.repo_info(REPO,repo_type='dataset',files_metadata=True)
            entry=next((f for f in existing.siblings if f.rfilename==target),None)
            entry_sha=(entry.lfs.sha256 if hasattr(entry.lfs,'sha256') else entry.lfs['sha256']) if entry and entry.lfs else None
            if entry_sha!=sha:
                api.upload_file(path_or_fileobj=str(source),path_in_repo=target,repo_id=REPO,repo_type='dataset',
                                commit_message=f'Archive local checkpoint {source.name} ({sha[:12]})')
            remote=[target]
        for row in rows:row['remote_paths']=remote
        with log.open('a',encoding='utf-8') as f:f.write(json.dumps({'sha256':sha,'remote_paths':remote,'local_paths':[r['path'] for r in rows]})+'\n')
        (STATE/'model_backup_progress.json').write_text(json.dumps(inventory,indent=2))
        print(f'[{index}/{len(groups)}] Recorded remote backup; elapsed {time.monotonic()-start:.0f}s',flush=True)
    current=api.repo_info(REPO,repo_type='dataset',files_metadata=True)
    hashes={}
    for f in current.siblings:
        if f.lfs:hashes[f.rfilename]=f.lfs.sha256 if hasattr(f.lfs,'sha256') else f.lfs['sha256']
    for row in inventory['weights']:
        if not any(hashes.get(p)==row['sha256'] for p in row['remote_paths']):raise RuntimeError(f'Unverified backup: {row["path"]}')
    inventory['verified_remote_revision']=current.sha
    (STATE/'verified_model_backups.json').write_text(json.dumps(inventory,indent=2))
    api.upload_file(path_or_fileobj=str(STATE/'verified_model_backups.json'),path_in_repo='models/archive_20261029/model_inventory.json',
                    repo_id=REPO,repo_type='dataset',commit_message='Record hash-verified local model archive and original paths')
    print(f'ALL {len(inventory["weights"])} local model aliases verified remotely.',flush=True)

def backup_evidence():
    os.environ['HF_HUB_DISABLE_XET']='1'
    from huggingface_hub import HfApi
    roots=[ROOT/'dataset_crack/runs',ROOT/'dataset_crack/notebook/runs',ROOT/'runs']
    target=STATE/'research_evidence_20261029.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for base in roots:
            if not base.exists():continue
            for p in base.rglob('*'):
                if p.is_file() and p.suffix.lower() not in WEIGHTS and p.suffix.lower() not in {'.mp4','.avi','.mov'}:
                    z.write(p,p.relative_to(ROOT))
        for base in [ROOT/'dataset_crack/hybrid_crack_masks',ROOT/'dataset_crack/sam31_preview',ROOT/'dataset_crack/distillation_outputs']:
            if not base.exists():continue
            for p in base.rglob('*'):
                if not p.is_file() or p.suffix.lower() in WEIGHTS or p.suffix=='.cache':continue
                if 'distillation_outputs' in str(base) and p.suffix.lower() in {'.jpg','.jpeg','.png','.txt'}:continue
                z.write(p,p.relative_to(ROOT))
        for p in (ROOT/'dataset_crack').rglob('*'):
            if p.is_file() and p.suffix.lower() in {'.yaml','.json','.csv'} and not any(part in {'runs','hybrid_crack_masks','sam31_preview','distillation_outputs','__pycache__'} for part in p.relative_to(ROOT/'dataset_crack').parts):
                z.write(p,p.relative_to(ROOT))
    print(f'Evidence ZIP: {target.stat().st_size/2**20:.1f} MiB',flush=True)
    sha=digest(target)
    api=HfApi();destination='archives/cleanup_20261029/research_evidence.zip'
    api.upload_file(path_or_fileobj=str(target),path_in_repo=destination,repo_id=REPO,repo_type='dataset',commit_message='Preserve research graphs, metrics, labels and curated SAM masks before local cleanup')
    info=api.repo_info(REPO,repo_type='dataset',files_metadata=True)
    f=next(f for f in info.siblings if f.rfilename==destination)
    remote_sha=f.lfs.sha256 if hasattr(f.lfs,'sha256') else f.lfs['sha256']
    if remote_sha!=sha:raise RuntimeError('Evidence checksum mismatch')
    (STATE/'verified_evidence_backup.json').write_text(json.dumps({'path':str(target.relative_to(ROOT)),
        'remote_path':destination,'sha256':sha,'remote_revision':info.sha},indent=2))
    print('Evidence archive verified on Hugging Face.',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['audit','remote-audit','backup-models','backup-evidence']);args=p.parse_args()
    {'audit':audit,'remote-audit':remote_audit,'backup-models':backup_models,'backup-evidence':backup_evidence}[args.action]()

if __name__=='__main__':main()
