import os, pickle, base64
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

# -------------------------
# File Encryption
# -------------------------
def encrypt_file(infile: str, outfile: str, passphrase: str):
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
    with open(infile, "rb") as f:
        raw = f.read()

    salt, nonce, ciphertext = raw[:16], raw[16:28], raw[28:]
    key = _derive_key(passphrase, salt)
    aesgcm = AESGCM(key)
    decrypted = aesgcm.decrypt(nonce, ciphertext, None)

    with open(outfile, "wb") as f:
        f.write(decrypted)

# -------------------------
# Variable Encryption
# -------------------------
def encrypt_variable(obj, passphrase: str) -> bytes:
    """Encrypt a Python variable (pickled) into bytes."""
    salt = os.urandom(16)
    key = _derive_key(passphrase, salt)
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)

    serialized = pickle.dumps(obj)  # serialize Python object
    encrypted = aesgcm.encrypt(nonce, serialized, None)

    return salt + nonce + encrypted  # return as bytes

def decrypt_variable(data: bytes, passphrase: str):
    """Decrypt bytes back into original Python variable."""
    salt, nonce, ciphertext = data[:16], data[16:28], data[28:]
    key = _derive_key(passphrase, salt)
    aesgcm = AESGCM(key)
    decrypted = aesgcm.decrypt(nonce, ciphertext, None)

    return pickle.loads(decrypted)
