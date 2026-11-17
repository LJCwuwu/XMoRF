"""Assemble the model bundle from checkpoint files identified by SHA256 checksum."""
import argparse
import json
from pathlib import Path
from xmorf.resources import prepare_files

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', '--source-project', dest='source', required=True,
                    help='Directory containing checkpoint files; searched recursively')
    ap.add_argument('--output', default='weights')
    args = ap.parse_args()
    config = Path(__file__).resolve().parents[1] / 'configs/ensemble.json'
    manifest = json.loads(config.read_text())
    output = Path(args.output)
    path = output / 'manifest.json'
    if path.exists():
        existing = json.loads(path.read_text())
        for entry in existing.get('models', []):
            entry.pop('source_run', None)
        if existing != manifest:
            raise FileExistsError(f'Existing model manifest differs: {path}')
    count = prepare_files(manifest['models'], args.source, output,
                          suffixes={'.pt', '.pth'})
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Prepared {count} verified checkpoints in {output}')

if __name__ == '__main__':
    main()
