"""Copy the exact reviewed datasets from an existing MoRF_project, with checksums."""
import argparse,hashlib,json,shutil
from pathlib import Path

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--source-project',required=True)
 ap.add_argument('--output',default='data')
 a=ap.parse_args();root=Path(a.source_project).resolve();out=Path(a.output)
 files=json.loads((Path(__file__).resolve().parents[1]/'configs/data_sources.json').read_text())['files']
 for e in files:
  src=root/e['source']
  if hashlib.sha256(src.read_bytes()).hexdigest()!=e['sha256']:raise ValueError('Dataset changed: '+e['source'])
  dst=out/e['path']
  if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest()!=e['sha256']:raise FileExistsError(dst)
 for e in files:
  dst=out/e['path'];dst.parent.mkdir(parents=True,exist_ok=True)
  if not dst.exists():shutil.copyfile(root/e['source'],dst)
 print(f'Prepared {len(files)} unchanged dataset files in {out}')

if __name__=='__main__':main()
