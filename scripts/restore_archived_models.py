"""Restore archived weights to their original project paths from Hugging Face."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT=Path(__file__).resolve().parents[1]
REPO='AkumaDachi/roadeeye-sewer-defects-v5'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--match',default=None,help='Only restore original paths containing this text; omit to list available models')
    args=p.parse_args()
    manifest=hf_hub_download(REPO,'models/archive_20261029/model_inventory.json',repo_type='dataset')
    inventory=json.loads(Path(manifest).read_text())
    for row in inventory['weights']:
        if not args.match:
            print(row['path']);continue
        if args.match not in row['path']:continue
        destination=(ROOT/row['path']).resolve()
        if not destination.is_relative_to(ROOT):raise ValueError('Invalid archived path')
        cached=Path(hf_hub_download(REPO,row['remote_paths'][0],repo_type='dataset'))
        h=hashlib.sha256()
        with cached.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        if h.hexdigest()!=row['sha256']:raise ValueError('Remote checksum mismatch')
        if destination.exists():
            print(f'Already present, not overwritten: {destination}');continue
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(cached,destination)
        print(f'Restored: {destination}')

if __name__=='__main__':main()
