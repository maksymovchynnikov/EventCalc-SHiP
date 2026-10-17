"""Additional production data require an explicit source and verified contents."""
import contextlib
import hashlib
import io
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from eventcalc.production.dark_photon import datasets


class ProductionDataInstallTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / 'production.tar.gz'
        self.name = 'results/idm_production/sources/provenance.json'
        self.content = b'{"test": true}\n'
        with tarfile.open(self.archive, 'w:gz') as stream:
            info = tarfile.TarInfo(self.name)
            info.size = len(self.content)
            stream.addfile(info, io.BytesIO(self.content))
        self.spec = {
            'version': 'test',
            'archive': self.archive.name,
            'sha256': hashlib.sha256(self.archive.read_bytes()).hexdigest(),
            'files': {self.name: hashlib.sha256(self.content).hexdigest()},
        }
        self.target = self.root / 'installed'
        self.manifest = patch.object(datasets, 'dataset', return_value=self.spec)
        self.manifest.start()
        self.addCleanup(self.manifest.stop)

    def test_local_install_and_verification(self):
        self.assertEqual(datasets.install_archive(self.archive, self.target), self.target)
        self.assertEqual((self.target / self.name).read_bytes(), self.content)
        self.assertEqual(datasets.verify(self.target), self.target)
        (self.target / self.name).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            datasets.verify(self.target)

    def test_fetch_requires_explicit_source_without_network_access(self):
        argv = ['datasets', 'fetch', '--destination', str(self.target)]
        with patch('sys.argv', argv), patch.object(datasets.subprocess, 'run') as run:
            with contextlib.redirect_stderr(io.StringIO()) as error:
                with self.assertRaises(SystemExit) as raised:
                    datasets.main()
            self.assertEqual(raised.exception.code, 2)
            self.assertIn('--source is required', error.getvalue())
            run.assert_not_called()

    def test_fetch_uses_supplied_source_and_verifies_archive(self):
        source = 'user@provider.example:/data/production.tar.gz'
        argv = ['datasets', 'fetch', '--source', source,
                '--destination', str(self.target), '--ssh-control-path', '/tmp/test-ssh.sock']

        def transfer(command, check):
            self.assertTrue(check)
            self.assertEqual(command[:-1], ['scp', '-o', 'ControlPath=/tmp/test-ssh.sock', '--', source])
            shutil.copyfile(self.archive, command[-1])

        with patch('sys.argv', argv), patch.object(datasets.subprocess, 'run', side_effect=transfer):
            with contextlib.redirect_stdout(io.StringIO()):
                datasets.main()
        self.assertEqual(datasets.verify(self.target), self.target)


if __name__ == '__main__':
    unittest.main()
