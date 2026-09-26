"""Encrypt or decrypt a folder.

The same actions are Test panel buttons on ``TestEncryptFolder`` in
``tests/utils/test_encrypt_folder.py``. This script is the command-line form.

    uv run python scripts/encrypt_folder.py encrypt
    uv run python scripts/encrypt_folder.py decrypt

``encrypt`` reads ``--plain`` (default ``docs/_treasure``) and writes
``--encrypted`` (default ``docs/_safe``). ``decrypt`` restores the plain
folder from that encrypted copy. The passphrase comes from ``credentials.yml``.
"""

import argparse
import hashlib
from pathlib import Path

import yaml

from quantbullet.utils.encrypt import decrypt_file, encrypt_file


def read_passphrase(path: Path) -> str:
    """Read the encryption passphrase from a YAML file."""
    if not path.exists():
        raise FileNotFoundError(f"Credentials file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        credentials = yaml.safe_load(handle)

    if not isinstance(credentials, dict) or "passphrase" not in credentials:
        raise KeyError(f"'passphrase' key not found in {path}")

    return credentials["passphrase"]


def fuzz_filename(relative_path: str, passphrase: str) -> str:
    """Generate a deterministic but opaque filename from path + passphrase."""
    digest = hashlib.sha256()
    digest.update((passphrase + relative_path).encode("utf-8"))
    return digest.hexdigest()


def encrypt_folder(plain: Path, encrypted: Path, passphrase: str) -> dict[str, str]:
    """Encrypt every file under ``plain`` into ``encrypted``."""
    if not plain.is_dir():
        raise FileNotFoundError(f"Plain folder not found: {plain}")

    encrypted.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    for file in plain.glob("**/*"):
        if not file.is_file():
            continue
        relative_path = str(file.relative_to(plain))
        fuzzed_name = fuzz_filename(relative_path, passphrase) + file.suffix + ".enc"
        encrypt_file(str(file), str(encrypted / fuzzed_name), passphrase)
        mapping[relative_path] = fuzzed_name

    mapping_file = encrypted / "file_mapping.yml"
    with mapping_file.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(mapping, handle)
    encrypt_file(str(mapping_file), str(mapping_file) + ".enc", passphrase)
    mapping_file.unlink()
    return mapping


def decrypt_folder(encrypted: Path, plain: Path, passphrase: str) -> dict[str, str]:
    """Decrypt ``encrypted`` back into ``plain`` using the stored mapping."""
    mapping_file = encrypted / "file_mapping.yml.enc"
    if not mapping_file.is_file():
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")

    temp_mapping_file = encrypted / "file_mapping_temp.yml"
    decrypt_file(str(mapping_file), str(temp_mapping_file), passphrase)
    try:
        with temp_mapping_file.open("r", encoding="utf-8") as handle:
            mapping = yaml.safe_load(handle)
    finally:
        temp_mapping_file.unlink(missing_ok=True)

    if not isinstance(mapping, dict):
        raise ValueError(f"Mapping file did not contain a dictionary: {mapping_file}")

    for original_relative_path, fuzzed_name in mapping.items():
        original_file = plain / original_relative_path
        original_file.parent.mkdir(parents=True, exist_ok=True)
        decrypt_file(str(encrypted / fuzzed_name), str(original_file), passphrase)
    return mapping


def _add_folder_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--plain",
        type=Path,
        default=Path("docs/_treasure"),
        help="plaintext folder (default: docs/_treasure)",
    )
    parser.add_argument(
        "--encrypted",
        type=Path,
        default=Path("docs/_safe"),
        help="encrypted folder (default: docs/_safe)",
    )
    parser.add_argument(
        "--credentials",
        type=Path,
        default=Path("credentials.yml"),
        help="YAML file with a passphrase key (default: credentials.yml)",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Manually encrypt or decrypt a folder. Not collected by pytest.",
        epilog=(
            "examples:\n"
            "  uv run python scripts/encrypt_folder.py encrypt\n"
            "  uv run python scripts/encrypt_folder.py decrypt"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    encrypt_parser = subparsers.add_parser("encrypt", help="encrypt --plain into --encrypted")
    _add_folder_args(encrypt_parser)
    decrypt_parser = subparsers.add_parser("decrypt", help="restore --plain from --encrypted")
    _add_folder_args(decrypt_parser)

    args = parser.parse_args(argv)
    passphrase = read_passphrase(args.credentials)
    if args.command == "encrypt":
        mapping = encrypt_folder(args.plain, args.encrypted, passphrase)
        print(f"Encrypted {len(mapping)} file(s) into {args.encrypted}")
        return
    mapping = decrypt_folder(args.encrypted, args.plain, passphrase)
    print(f"Decrypted {len(mapping)} file(s) into {args.plain}")


if __name__ == "__main__":
    main()
