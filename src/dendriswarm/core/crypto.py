from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def b64e(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def b64d(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"), validate=True)



def public_key_fingerprint(public_key_b64: str) -> str:
    """Human-comparable SHA-256 fingerprint for out-of-band coordinator pinning."""
    return hashlib.sha256(b64d(public_key_b64)).hexdigest()


def node_id_from_public_key(public_key_b64: str) -> str:
    return hashlib.sha256(b64d(public_key_b64)).hexdigest()[:20]


def content_hash(value: Any, excluded_keys: set[str] | None = None) -> str:
    excluded = excluded_keys or {"sha256"}
    if isinstance(value, dict):
        value = {k: v for k, v in value.items() if k not in excluded}
    return hashlib.sha256(canonical_json(value)).hexdigest()


def nonce() -> str:
    return secrets.token_hex(24)


class Identity:
    def __init__(self, private_key: Ed25519PrivateKey):
        self.private_key = private_key
        self.public_key = private_key.public_key()

    @classmethod
    def generate(cls) -> "Identity":
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def load_or_create(cls, directory: Path) -> "Identity":
        def load_existing(path: Path) -> "Identity":
            # An O_EXCL winner owns the pathname before its PEM bytes are
            # necessarily complete. Readers therefore retry both an absent
            # file and a temporarily partial PEM instead of interpreting the
            # creation window as corruption.
            import time

            last_error: Exception | None = None
            for _ in range(200):
                try:
                    private_key = serialization.load_pem_private_key(path.read_bytes(), password=None)
                    if not isinstance(private_key, Ed25519PrivateKey):
                        raise TypeError("identity.pem is not an Ed25519 private key")
                    return cls(private_key)
                except (FileNotFoundError, ValueError) as error:
                    last_error = error
                    time.sleep(0.005)
            raise RuntimeError("identity creation race did not produce a readable key") from last_error

        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "identity.pem"
        if path.exists():
            return load_existing(path)
        private_key = Ed25519PrivateKey.generate()
        encoded = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            # Another process won the identity-creation race. Use the complete
            # winner file rather than overwriting it with a second identity.
            return load_existing(path)
        else:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            return cls(private_key)

    @property
    def public_key_b64(self) -> str:
        return b64e(
            self.public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        )

    @property
    def node_id(self) -> str:
        return node_id_from_public_key(self.public_key_b64)

    def sign(self, value: Any) -> str:
        return b64e(self.private_key.sign(canonical_json(value)))


def verify(public_key_b64: str, value: Any, signature_b64: str) -> bool:
    try:
        key = Ed25519PublicKey.from_public_bytes(b64d(public_key_b64))
        key.verify(b64d(signature_b64), canonical_json(value))
        return True
    except Exception:
        return False
