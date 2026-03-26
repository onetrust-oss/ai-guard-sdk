from __future__ import annotations

import base64
import enum
import hashlib
import secrets
import shutil
import ssl
import tempfile
from pathlib import Path
from typing import Callable

import requests
import trustme
from cryptography import x509
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
)
from requests.adapters import HTTPAdapter
from urllib3.util import connection as urllib3_connection

from .config import PROJECT_ROOT
from .ports import allocate_port
from .test_classification_service import TestClassificationService


class AuthMode(enum.Enum):
    SECRET = "secret"
    ONETRUST = "onetrust"


class ClassifierClientType(enum.Enum):
    CLIENT = "client"
    FS = "fs"


class _HostResolverAdapter(HTTPAdapter):

    def __init__(self, hostname: str):
        self._hostname = hostname
        super().__init__()

    def send(self, request, **kwargs):
        orig = urllib3_connection.create_connection
        hostname = self._hostname

        def _patched(address, *args, **kw):
            host, port = address
            if host == hostname:
                address = ("127.0.0.1", port)
            return orig(address, *args, **kw)

        urllib3_connection.create_connection = _patched
        try:
            return super().send(request, **kwargs)
        finally:
            urllib3_connection.create_connection = orig


class AIGuardTestServices:
    __test__ = False

    def __init__(
        self,
        tls: bool = True,
        auth: AuthMode = AuthMode.ONETRUST,
        classifier_client_type: ClassifierClientType = ClassifierClientType.CLIENT,
    ):
        self._tmpdir = tempfile.mkdtemp(prefix="classification-test-")
        self._tls = tls
        self._auth = auth
        self._classifier_client_type = classifier_client_type

        self._classification_token = secrets.token_hex(16)
        token_path = Path(self._tmpdir) / "classification-token"
        token_path.write_text(self._classification_token)
        self._classification_token_path = token_path

        self._hostname = f"test-{secrets.token_hex(8)}.local"
        self._tls_cert_path = Path(self._tmpdir) / "server.crt"
        self._tls_key_path = Path(self._tmpdir) / "server.key"
        self._ca_cert_path = Path(self._tmpdir) / "ca.pem"

        self._ssl_ctx: ssl.SSLContext | None = None
        if self._tls:
            ca = trustme.CA()
            server_cert = ca.issue_cert(self._hostname)

            server_cert.private_key_pem.write_to_path(str(self._tls_key_path))
            with open(self._tls_cert_path, "wb") as f:
                for blob in server_cert.cert_chain_pems:
                    f.write(blob.bytes())

            ca.cert_pem.write_to_path(str(self._ca_cert_path))

            self._ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            server_cert.configure_cert(self._ssl_ctx)
            ca.configure_trust(self._ssl_ctx)

        self._classification_service: TestClassificationService | None = None
        self._port: int | None = None

    @property
    def classification_token(self) -> str:
        return self._classification_token

    @property
    def auth(self) -> AuthMode:
        return self._auth

    @property
    def classification_service(self) -> TestClassificationService:
        assert self._classification_service is not None, "classification service not started"
        return self._classification_service

    @property
    def hostname(self) -> str:
        return self._hostname

    @property
    def server_public_key_pin(self) -> str:
        pem_data = self._tls_cert_path.read_bytes()
        cert = x509.load_pem_x509_certificate(pem_data)
        spki_der = cert.public_key().public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo)
        return base64.b64encode(hashlib.sha256(spki_der).digest()).decode("ascii")

    def set_classification_response(self, matches: list[dict]) -> None:
        self.classification_service.set_classification_response(matches)

    def set_classification_error(self, status: int, message: str) -> None:
        self.classification_service.set_classification_error(status, message)

    def set_classification_handler(
        self, handler: Callable[[dict], list[dict]],
    ) -> None:
        self.classification_service.set_classification_handler(handler)

    def start(self) -> None:
        self._port = self._allocate_port()

        self._classification_service = TestClassificationService(
            expected_token=self._classification_token,
        )
        self._classification_service.start(
            port=self._port,
            ssl_ctx=self._ssl_ctx,
        )

    def stop(self) -> None:
        if self._classification_service is not None:
            self._classification_service.stop()
            self._classification_service = None
        self._cleanup()

    @property
    def port(self) -> int:
        assert self._port is not None, "server not started"
        return self._port

    @property
    def url(self) -> str:
        if self._tls:
            return f"https://{self._hostname}:{self._port}"
        return f"http://localhost:{self._port}"

    def http_session(self) -> requests.Session:
        session = requests.Session()
        session.trust_env = False
        if self._tls:
            session.verify = str(self._ca_cert_path)
            session.mount("https://", _HostResolverAdapter(self._hostname))
        return session

    @staticmethod
    def _allocate_port() -> int:
        return allocate_port(PROJECT_ROOT / "target")

    def _cleanup(self) -> None:
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def __del__(self):
        self.stop()
