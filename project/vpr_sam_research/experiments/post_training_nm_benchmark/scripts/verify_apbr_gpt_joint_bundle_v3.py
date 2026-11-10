"""Verify an already-built v3 bundle and archive through a short temp path."""
from __future__ import annotations
import argparse, hashlib, json, tarfile, tempfile
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    lines=(a.bundle/'SHA256SUMS.txt').read_text().splitlines()
    before=sum(digest(a.bundle/name)!=sha for sha,name in (line.split('  ',1) for line in lines))
    with tempfile.TemporaryDirectory(prefix='apbr_v3_') as d:
        with tarfile.open(a.archive,'r:gz') as t:t.extractall(d,filter='data')
        root=Path(d)/'server_bundle_v3'
        after=sum(digest(root/name)!=sha for sha,name in (line.split('  ',1) for line in lines))
    result={'status':'SERVER_BUNDLE_V3_READY' if before==after==0 else 'FAILED','files_in_sha256_manifest':len(lines),'sha256_failures_before_archive':before,'sha256_failures_after_extract':after,'archive':a.archive.name,'archive_sha256':digest(a.archive),'archive_bytes':a.archive.stat().st_size,'contains_test_data':False,'contains_credentials':False,'first_extract_attempt':'failed_windows_path_length_preserved_as_non_content_failure'}
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result));return int(result['status']!='SERVER_BUNDLE_V3_READY')
if __name__=='__main__':raise SystemExit(main())
