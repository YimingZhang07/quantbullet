import os
import pickle
import zlib
import secrets
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.backends import default_backend

# -------------------------
# Key Derivation
# -------------------------
def _derive_key(passphrase: str, salt: bytes) -> bytes:
    """Derive a 256-bit key from passphrase + salt."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,                # AES-256
        salt=salt,
        iterations=200_000,
        backend=default_backend()
    )
    return kdf.derive(passphrase.encode())

# from argon2.low_level import hash_secret_raw, Type

# def _derive_key(passphrase: str, salt: bytes) -> bytes:
#     """Derive a 256-bit key from passphrase + salt using Argon2id."""
#     return hash_secret_raw(
#         secret=passphrase.encode(),
#         salt=salt,
#         time_cost=3,        # iterations (higher = slower = more secure)
#         memory_cost=65536,  # memory in KiB (here 64 MB)
#         parallelism=4,      # number of threads
#         hash_len=32,        # length of derived key in bytes (32 = 256-bit AES)
#         type=Type.ID        # Argon2id variant
#     )

# -------------------------
# File Encryption
# -------------------------
def encrypt_file(infile: str, outfile: str, passphrase: str):
    """Encrypt a file with AES-GCM."""
    salt = os.urandom(16)
    key = _derive_key(passphrase, salt)
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)

    with open(infile, "rb") as f:
        data = f.read()

    encrypted = aesgcm.encrypt(nonce, data, None)

    with open(outfile, "wb") as f:
        f.write(salt + nonce + encrypted)

def decrypt_file(infile: str, outfile: str, passphrase: str):
    """Decrypt an encrypted file back to original."""
    with open(infile, "rb") as f:
        raw = f.read()

    salt, nonce, ciphertext = raw[:16], raw[16:28], raw[28:]
    key = _derive_key(passphrase, salt)
    aesgcm = AESGCM(key)
    decrypted = aesgcm.decrypt(nonce, ciphertext, None)

    with open(outfile, "wb") as f:
        f.write(decrypted)

def encrypt_variable_to_file(obj, outfile: str, passphrase: str):
    """Encrypt a Python variable (pickled) and save to a file."""
    salt = os.urandom(16)
    key = _derive_key(passphrase, salt)
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)

    serialized = pickle.dumps(obj)  # serialize Python object
    encrypted = aesgcm.encrypt(nonce, serialized, None)

    with open(outfile, "wb") as f:
        f.write(salt + nonce + encrypted)

def decrypt_variable_from_file(infile: str, passphrase: str):
    """Decrypt a file back into original Python variable."""
    with open(infile, "rb") as f:
        raw = f.read()

    salt, nonce, ciphertext = raw[:16], raw[16:28], raw[28:]
    key = _derive_key(passphrase, salt)
    aesgcm = AESGCM(key)
    decrypted = aesgcm.decrypt(nonce, ciphertext, None)

    return pickle.loads(decrypted)

def split_and_shrink(input_file, out_dir="chunks", chunk_size=1024*1024):
    """Split a large file into compressed chunks with random names."""
    os.makedirs(out_dir, exist_ok=True)

    # Read + compress
    with open(input_file, "rb") as f:
        data = f.read()
    compressed = zlib.compress(data, level=9)

    # Split into chunks with numbered prefix
    for i in range(0, len(compressed), chunk_size):
        chunk = compressed[i:i+chunk_size]
        randname = secrets.token_hex(6)  # random suffix
        fname = f"{i:08d}_{randname}.bin"  # numbered prefix ensures order
        with open(os.path.join(out_dir, fname), "wb") as f:
            f.write(chunk)
    print(f"Done! Split into {len(os.listdir(out_dir))} chunks under {out_dir}/")

def reassemble_and_expand(out_dir, output_file):
    """Reassemble and decompress chunks back into original file."""
    # Sort by numeric prefix
    files = sorted(os.listdir(out_dir))
    combined = b"".join(open(os.path.join(out_dir, f), "rb").read() for f in files)

    # Decompress
    restored = zlib.decompress(combined)
    with open(output_file, "wb") as f:
        f.write(restored)
    print(f"Reassembled file written to {output_file}")