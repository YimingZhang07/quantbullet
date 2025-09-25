import unittest
import yaml
import hashlib
from pathlib import Path
from quantbullet.utils.encrypt import encrypt_file, decrypt_file


def read_passphrase(path: str | Path = "credentials.yml") -> str:
    """Read the encryption passphrase from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Credentials file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        credentials = yaml.safe_load(f)

    if "passphrase" not in credentials:
        raise KeyError("'passphrase' key not found in credentials.yml")

    return credentials["passphrase"]


def fuzz_filename(relative_path: str, passphrase: str) -> str:
    """Generate a deterministic but opaque filename from path + passphrase."""
    h = hashlib.sha256()
    h.update((passphrase + relative_path).encode("utf-8"))
    return h.hexdigest()


class TestEncryptFolder(unittest.TestCase):
    def setUp(self):
        self.original_folder = "./docs/A"
        self.encrypted_folder = "./docs/B"

    def test_encrypt_folder(self):
        passphrase = read_passphrase()
        original_path = Path(self.original_folder)
        encrypted_path = Path(self.encrypted_folder)
        encrypted_path.mkdir(parents=True, exist_ok=True)

        mapping = {}
        for file in original_path.glob("**/*"):
            if file.is_file():
                relative_path = str(file.relative_to(original_path))

                # Fuzz filename
                fuzzed_name = fuzz_filename(relative_path, passphrase) + file.suffix + ".enc"
                encrypted_file_path = encrypted_path / fuzzed_name

                # Encrypt file contents
                encrypt_file(str(file), str(encrypted_file_path), passphrase)

                # Save mapping
                mapping[relative_path] = fuzzed_name

        # Save mapping to YAML and encrypt it
        mapping_file = encrypted_path / "file_mapping.yml"
        with mapping_file.open("w", encoding="utf-8") as f:
            yaml.dump(mapping, f)

        encrypt_file(str(mapping_file), str(mapping_file) + ".enc", passphrase)
        mapping_file.unlink()  # remove unencrypted mapping file

        return mapping

    def test_decrypt_folder(self):
        passphrase = read_passphrase()
        encrypted_path = Path(self.encrypted_folder)
        mapping_file = encrypted_path / "file_mapping.yml.enc"

        if not mapping_file.exists():
            raise FileNotFoundError(f"Mapping file not found: {mapping_file}")

        # Decrypt mapping file
        temp_mapping_file = encrypted_path / "file_mapping_temp.yml"
        decrypt_file(str(mapping_file), str(temp_mapping_file), passphrase)

        with temp_mapping_file.open("r", encoding="utf-8") as f:
            mapping = yaml.safe_load(f)

        temp_mapping_file.unlink()  # remove temporary plaintext mapping

        # Decrypt each file using mapping
        for original_relative_path, fuzzed_name in mapping.items():
            encrypted_file_path = encrypted_path / fuzzed_name
            original_file_path = Path(self.original_folder) / original_relative_path
            original_file_path.parent.mkdir(parents=True, exist_ok=True)
            decrypt_file(str(encrypted_file_path), str(original_file_path), passphrase)
