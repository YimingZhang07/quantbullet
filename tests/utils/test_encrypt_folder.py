import tempfile
import unittest
from pathlib import Path

import pytest

from quantbullet.utils.encrypt import decrypt_file, encrypt_file
from scripts.encrypt_folder import decrypt_folder, encrypt_folder, read_passphrase

PLAIN = Path("docs/_treasure")
ENCRYPTED = Path("docs/_safe")


class TestEncryptFile(unittest.TestCase):
    def test_roundtrip_uses_a_temporary_directory(self):
        passphrase = "test-passphrase"
        payload = b"quantbullet encrypt roundtrip"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "plain.txt"
            encrypted = root / "plain.txt.enc"
            restored = root / "restored.txt"
            source.write_bytes(payload)

            encrypt_file(str(source), str(encrypted), passphrase)
            decrypt_file(str(encrypted), str(restored), passphrase)

            self.assertTrue(encrypted.is_file())
            self.assertNotEqual(encrypted.read_bytes(), payload)
            self.assertEqual(restored.read_bytes(), payload)

        self.assertFalse(root.exists())


@pytest.mark.manual
class TestEncryptFolder(unittest.TestCase):
    """Buttons in the Test panel. A full pytest run skips these."""

    def test_encrypt_folder(self):
        encrypt_folder(PLAIN, ENCRYPTED, read_passphrase())

    def test_decrypt_folder(self):
        decrypt_folder(ENCRYPTED, PLAIN, read_passphrase())
