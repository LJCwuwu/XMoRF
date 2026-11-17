"""Assemble the 72 final checkpoints from an existing project, without retraining."""
import argparse,json,shutil
from pathlib import Path
from xmorf.predict import sha256

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--source-project',required=True)
 ap.add_argument('--output',default='weights')
 a=ap.parse_args();root=Path(a.source_project).resolve();out=Path(a.output)
 manifest=json.loads((Path(__file__).resolve().parents[1]/'configs/ensemble.json').read_text())
 for e in manifest['models']:
  src=root/e['source_run']/'model.pt';dst=out/e['path']
  if sha256(src)!=e['sha256']:raise ValueError('Wrong checkpoint: '+str(src))
  if dst.exists() and sha256(dst)!=e['sha256']:raise FileExistsError(dst)
 for e in manifest['models']:
  dst=out/e['path'];dst.parent.mkdir(parents=True,exist_ok=True)
  if not dst.exists():shutil.copyfile(root/e['source_run']/'model.pt',dst)
 raw=json.dumps(manifest,indent=2)+'\n';path=out/'manifest.json'
 if path.exists() and path.read_text()!=raw:raise FileExistsError(path)
 path.write_text(raw)
 print(f'Prepared all 72 verified checkpoints in {out}')

if __name__=='__main__':main()
