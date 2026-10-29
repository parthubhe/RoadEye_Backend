"""Offline regression checks: python -m unittest discover -s dataset_crack/notebook -p test_dataset_source.py"""
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run_experiment import DEFAULT_LOCAL_DATA, ROOT, arguments, check_dataset_source


class DatasetSourceTests(unittest.TestCase):
    def test_all_launchers_default_to_script_relative_v5(self):
        previous = Path.cwd()
        try:
            for cwd in (ROOT, ROOT.parent, ROOT.parent.parent):
                os.chdir(cwd)
                for experiment in ('E1', 'E2', 'E3', 'E4', 'E5', 'E6'):
                    args = arguments([], default_experiment=experiment)
                    self.assertEqual(args.local_data, DEFAULT_LOCAL_DATA.resolve())
                    self.assertFalse(args.download_data)
        finally:
            os.chdir(previous)

    def test_download_is_explicit(self):
        args = arguments(['--download-data'], default_experiment='E5')
        self.assertIsNone(args.local_data)

    def test_explicit_local_path(self):
        args = arguments(['--local-data', str(ROOT)], default_experiment='E5')
        self.assertEqual(args.local_data, ROOT)

    def test_sources_are_mutually_exclusive(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            arguments(['--local-data', str(ROOT), '--download-data'], default_experiment='E5')

    def test_missing_local_dataset_never_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(['--local-data', directory], default_experiment='E5')
            with patch('huggingface_hub.snapshot_download') as download:
                with self.assertRaisesRegex(FileNotFoundError, 'No dataset download was attempted'):
                    check_dataset_source(args)
                download.assert_not_called()


if __name__ == '__main__':
    unittest.main()
