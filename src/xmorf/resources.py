"""Import datasets and checkpoints by their recorded SHA256 checksums."""
import hashlib
from pathlib import Path, PurePosixPath
import shutil


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_files(entries, source, output, suffixes=None):
    """Find all requested files before copying; reject conflicting destinations."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    targets = []
    paths = set()
    for entry in entries:
        relative = PurePosixPath(entry['path'])
        if relative.is_absolute() or '..' in relative.parts or not relative.parts:
            raise ValueError(f'Invalid resource path: {relative}')
        if str(relative) in paths:
            raise ValueError(f'Duplicate resource path: {relative}')
        paths.add(str(relative))
        digest = entry['sha256']
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError(f'Invalid SHA256 checksum: {relative}')
        target = output.joinpath(*relative.parts).resolve()
        if not target.is_relative_to(output):
            raise ValueError(f'Resource path escapes the output directory: {relative}')
        if target.exists() and (not target.is_file() or sha256(target) != digest):
            raise FileExistsError(f'Existing file differs: {target}')
        if any(parent.exists() and not parent.is_dir() for parent in target.parents):
            raise FileExistsError(f'Output parent is not a directory: {target}')
        targets.append((entry, target))

    needed = {entry['sha256'] for entry in entries}
    sizes = {entry['bytes'] for entry in entries if 'bytes' in entry}
    filter_sizes = all('bytes' in entry for entry in entries)
    found, checked = {}, set()

    def inspect(path):
        if path in checked or path.is_symlink() or not path.is_file():
            return
        checked.add(path)
        if suffixes and path.suffix.lower() not in suffixes:
            return
        if filter_sizes and path.stat().st_size not in sizes:
            return
        digest = sha256(path)
        if digest in needed:
            found.setdefault(digest, path)

    # Standard bundle paths take priority over a recursive search.
    for entry in entries:
        inspect(source / entry['path'])
    if needed - found.keys():
        for path in source.rglob('*'):
            if '.git' not in path.relative_to(source).parts:
                inspect(path)
            if not needed - found.keys():
                break
    missing = [entry['path'] for entry in entries if entry['sha256'] not in found]
    if missing:
        raise FileNotFoundError('Missing files with matching checksums: ' + ', '.join(missing))

    for entry, target in targets:
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with found[entry['sha256']].open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst)
        if sha256(target) != entry['sha256']:
            target.unlink()
            raise ValueError(f'Source file changed during copying: {entry["path"]}')
    return len(targets)
