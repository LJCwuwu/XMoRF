"""Import benchmark files by SHA256 checksum into the dataset directory."""
import argparse
import json
from pathlib import Path
from xmorf.resources import prepare_files

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', '--source-project', dest='source', required=True,
                    help='Directory containing the benchmark files; searched recursively')
    ap.add_argument('--output', default='data')
    args = ap.parse_args()
    manifest = Path(__file__).resolve().parents[1] / 'configs/data_sources.json'
    entries = json.loads(manifest.read_text())['files']
    count = prepare_files(entries, args.source, args.output)
    print(f'Prepared {count} verified dataset files in {args.output}')

if __name__ == '__main__':
    main()
