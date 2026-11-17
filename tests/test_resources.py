"""Resource imports preserve bytes and reject incomplete or conflicting inputs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from xmorf.resources import prepare_files


class ResourceImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'input'
        self.source.mkdir()
        self.output = self.root / 'output'

    def entry(self, path, content, sized=False):
        value = {'path': path, 'sha256': hashlib.sha256(content).hexdigest()}
        if sized:
            value['bytes'] = len(content)
        return value

    def test_renamed_nested_files_are_matched_by_content(self):
        content = b'checkpoint data'
        folder = self.source / 'arbitrary' / 'directory'
        folder.mkdir(parents=True)
        (folder / 'renamed.pth').write_bytes(content)
        entries = [self.entry('nested/f0_s0.pt', content)]
        self.assertEqual(prepare_files(entries, self.source, self.output,
                                       suffixes={'.pt', '.pth'}), 1)
        self.assertEqual((self.output / 'nested/f0_s0.pt').read_bytes(), content)
        before = (self.output / 'nested/f0_s0.pt').stat().st_mtime_ns
        prepare_files(entries, self.source, self.output, suffixes={'.pt', '.pth'})
        self.assertEqual((self.output / 'nested/f0_s0.pt').stat().st_mtime_ns, before)

    def test_missing_file_is_reported_before_copying(self):
        (self.source / 'found.txt').write_bytes(b'found')
        entries = [self.entry('first.txt', b'found'), self.entry('second.txt', b'absent')]
        with self.assertRaises(FileNotFoundError):
            prepare_files(entries, self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_same_size_corruption_is_rejected(self):
        (self.source / 'renamed.af').write_bytes(b'bad')
        with self.assertRaises(FileNotFoundError):
            prepare_files([self.entry('data.af', b'good'[:3], sized=True)],
                          self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_existing_conflict_preserves_all_output_files(self):
        (self.source / 'a').write_bytes(b'new-a')
        (self.source / 'b').write_bytes(b'new-b')
        self.output.mkdir()
        (self.output / 'b').write_bytes(b'existing')
        entries = [self.entry('a', b'new-a'), self.entry('b', b'new-b')]
        with self.assertRaises(FileExistsError):
            prepare_files(entries, self.source, self.output)
        self.assertFalse((self.output / 'a').exists())
        self.assertEqual((self.output / 'b').read_bytes(), b'existing')

    def test_duplicate_content_can_populate_multiple_paths(self):
        (self.source / 'only-copy.af').write_bytes(b'shared')
        entries = [self.entry('one.af', b'shared'), self.entry('two.af', b'shared')]
        prepare_files(entries, self.source, self.output)
        self.assertEqual((self.output / 'one.af').read_bytes(), b'shared')
        self.assertEqual((self.output / 'two.af').read_bytes(), b'shared')

    def test_parent_path_and_symlink_escape_are_rejected(self):
        (self.source / 'input').write_bytes(b'data')
        with self.assertRaises(ValueError):
            prepare_files([self.entry('../escape', b'data')], self.source, self.output)
        self.output.mkdir()
        outside = self.root / 'outside'
        outside.mkdir()
        (self.output / 'linked').symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            prepare_files([self.entry('linked/data', b'data')], self.source, self.output)
        self.assertEqual(list(outside.iterdir()), [])

    def test_suffix_filter_does_not_import_unrelated_files(self):
        (self.source / 'weights.txt').write_bytes(b'weights')
        with self.assertRaises(FileNotFoundError):
            prepare_files([self.entry('model.pt', b'weights')], self.source, self.output,
                          suffixes={'.pt', '.pth'})

    def test_preparation_commands_accept_renamed_inputs(self):
        project = Path(__file__).resolve().parents[1]
        fixture = self.root / 'project'
        (fixture / 'scripts').mkdir(parents=True)
        (fixture / 'configs').mkdir()
        env = dict(os.environ, PYTHONPATH=str(project / 'src'))
        for kind, extension in [('data', '.af'), ('weights', '.pt')]:
            with self.subTest(kind=kind):
                payload = f'{kind} fixture'.encode()
                (self.source / ('renamed' + extension)).write_bytes(payload)
                entry = self.entry('nested/file' + extension, payload, sized=kind == 'data')
                manifest = {'files' if kind == 'data' else 'models': [entry]}
                name = 'data_sources.json' if kind == 'data' else 'ensemble.json'
                (fixture / 'configs' / name).write_text(json.dumps(manifest))
                script = fixture / 'scripts' / f'prepare_{kind}.py'
                shutil.copyfile(project / 'scripts' / script.name, script)
                output = self.output / kind
                command = [sys.executable, str(script), '--source', str(self.source),
                           '--output', str(output)]
                run = subprocess.run(command, env=env, capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertEqual((output / entry['path']).read_bytes(), payload)
                if kind == 'weights':
                    saved = output / 'manifest.json'
                    self.assertEqual(json.loads(saved.read_text()), manifest)
                    compatible = json.loads(saved.read_text())
                    compatible['models'][0]['source_run'] = 'unused-path'
                    saved.write_text(json.dumps(compatible))
                command[2] = '--source-project'
                rerun = subprocess.run(command, env=env, capture_output=True, text=True)
                self.assertEqual(rerun.returncode, 0, rerun.stderr)
                if kind == 'weights':
                    self.assertEqual(json.loads(saved.read_text()), manifest)

    def test_model_manifest_conflict_stops_before_copying(self):
        project = Path(__file__).resolve().parents[1]
        fixture = self.root / 'project'
        (fixture / 'scripts').mkdir(parents=True)
        (fixture / 'configs').mkdir()
        (self.source / 'input.pt').write_bytes(b'checkpoint')
        entry = self.entry('model.pt', b'checkpoint')
        (fixture / 'configs/ensemble.json').write_text(json.dumps({'models': [entry]}))
        script = fixture / 'scripts/prepare_weights.py'
        shutil.copyfile(project / 'scripts/prepare_weights.py', script)
        self.output.mkdir()
        saved = self.output / 'manifest.json'
        saved.write_text(json.dumps({'models': []}))
        run = subprocess.run([sys.executable, str(script), '--source', str(self.source),
                              '--output', str(self.output)],
                             env=dict(os.environ, PYTHONPATH=str(project / 'src')),
                             capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertFalse((self.output / 'model.pt').exists())
        self.assertEqual(json.loads(saved.read_text()), {'models': []})


if __name__ == '__main__':
    unittest.main()
