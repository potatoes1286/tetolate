from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import release


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.files = {
            "version.py": 'VERSION = "2.0.0"\n',
            "pyproject.toml": '[project]\nname = "tetolate"\nversion = "1.0.0"\n',
            "uv.lock": '[[package]]\nname = "other"\nversion = "1.0.0"\n\n[[package]]\nname = "tetolate"\nversion = "1.0.0"\n',
            "CHANGELOG.md": "## 2.0.0 - 2026-10-30\n",
            "compose.yaml": "image: ghcr.io/potatoes1286/tetolate:latest\n",
            "README.md": "## Install\nNative and container installation.\n",
            "BUILDING.md": "Historical v1.0.0\n## Release container\ngit tag v1.0.0\ngit push origin v1.0.0\ndocker pull ghcr.io/potatoes1286/tetolate:1.0.0\n",
        }
        for name, value in self.files.items():
            (self.root / name).write_text(value)
        for name, value in (("ROOT", self.root), ("VERSION_FILE", self.root / "version.py")):
            patch = mock.patch.object(release, name, value)
            patch.start()
            self.addCleanup(patch.stop)

    def test_sync_updates_metadata_and_examples_without_changing_installation(self):
        changed = release.synchronize(release.read_version(), check_only=False)
        self.assertEqual({path.name for path in changed}, {"pyproject.toml", "uv.lock", "BUILDING.md"})
        self.assertIn('version = "2.0.0"', (self.root / "pyproject.toml").read_text())
        lock = (self.root / "uv.lock").read_text()
        self.assertIn('name = "other"\nversion = "1.0.0"', lock)
        self.assertIn('name = "tetolate"\nversion = "2.0.0"', lock)
        building = (self.root / "BUILDING.md").read_text()
        self.assertTrue(building.startswith("Historical v1.0.0\n"))
        self.assertIn("git tag v2.0.0", building)
        self.assertIn("git push origin v2.0.0", building)
        self.assertIn("ghcr.io/potatoes1286/tetolate:2.0.0", building)
        for name in ("compose.yaml", "README.md", "CHANGELOG.md"):
            self.assertEqual((self.root / name).read_text(), self.files[name])
        self.assertEqual(release.synchronize("2.0.0", check_only=True), [])

    def test_check_reports_outdated_metadata_without_writes(self):
        self.assertEqual(len(release.synchronize("2.0.0", check_only=True)), 3)
        for name, value in self.files.items():
            self.assertEqual((self.root / name).read_text(), value)

    def test_missing_release_entry_or_section_fails_before_writing(self):
        for name, content in (("CHANGELOG.md", "## Unreleased\n"), ("BUILDING.md", "## Tests\n")):
            with self.subTest(name=name):
                (self.root / name).write_text(content)
                with self.assertRaises(SystemExit):
                    release.synchronize("2.0.0", check_only=False)
                for unchanged in ("pyproject.toml", "uv.lock"):
                    self.assertEqual((self.root / unchanged).read_text(), self.files[unchanged])
                (self.root / name).write_text(self.files[name])
