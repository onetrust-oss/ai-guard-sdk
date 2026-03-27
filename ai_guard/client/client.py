import base64
import hashlib
import logging
import ssl

import requests
import urllib3
from cryptography import x509
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from requests.adapters import HTTPAdapter

from ai_guard.api import (
    AIPlatform,
    ClassificationRequest,
    ClassificationResponse,
    MetricsEvent,
)

logger = logging.getLogger(__name__)


class _PinningAdapter(HTTPAdapter):
    def __init__(self, ssl_context: ssl.SSLContext, **kwargs):
        self._ssl_context = ssl_context
        super().__init__(**kwargs)

    def init_poolmanager(self, *args, **kwargs):
        kwargs["ssl_context"] = self._ssl_context
        return super().init_poolmanager(*args, **kwargs)


class AIGuardClient:
    """Client for the AI Guard classification and metrics API.

    Sends classification requests to an AI Guard server and returns match
    results.  Optionally records metrics events.

    Args:
        base_url: AI Guard server URL (e.g. ``"https://guard.example.com"``).
        token: Bearer token for authentication.
        agent_id: Unique identifier for the calling agent.
        platform: AI platform originating requests.
        session: Optional pre-configured :class:`requests.Session`.  When
            provided, *pin_sha256* is ignored.
        timeout: HTTP request timeout in seconds.
        pin_sha256: Base64-encoded SHA-256 of the server's SPKI for
            certificate pinning.  Mutually exclusive with *session*.
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        agent_id: str,
        platform: AIPlatform,
        session: requests.Session | None = None,
        timeout: float = 30.0,
        pin_sha256: str | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._agent_id = agent_id
        self._platform = platform
        self._timeout = timeout

        if session is not None:
            self._session = session
        else:
            self._session = requests.Session()
            if pin_sha256 is not None:
                self._apply_public_key_pin(pin_sha256)

    def _apply_public_key_pin(self, expected_key_sha256_b64: str) -> None:
        try:
            expected_digest = base64.b64decode(expected_key_sha256_b64, validate=True)
        except Exception as e:
            raise ValueError(f"pin_sha256 is not valid base64: {e}") from e

        if len(expected_digest) != 32:
            raise ValueError(
                f"pin_sha256 must be a base64-encoded SHA-256 digest (32 bytes), "
                f"got {len(expected_digest)}"
            )

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        _original_wrap_socket = ctx.wrap_socket

        def _pinning_wrap_socket(sock, *args, **kwargs):
            ssl_sock = _original_wrap_socket(sock, *args, **kwargs)
            peer_der = ssl_sock.getpeercert(binary_form=True)
            if peer_der is None:
                ssl_sock.close()
                raise ssl.SSLCertVerificationError("No peer certificate presented")
            spki_der = (
                x509.load_der_x509_certificate(peer_der)
                .public_key()
                .public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo)
            )
            if hashlib.sha256(spki_der).digest() != expected_digest:
                ssl_sock.close()
                raise ssl.SSLCertVerificationError(
                    "Server public key does not match trusted key"
                )
            return ssl_sock

        ctx.wrap_socket = _pinning_wrap_socket

        self._session.verify = False
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        self._session.mount("https://", _PinningAdapter(ssl_context=ctx))

    def classify(self, request: ClassificationRequest) -> ClassificationResponse:
        """Classify text and return matches.

        Sends a ``POST /classifications/v1`` request.  The client's
        *agent_id* and *platform* are injected into ``request.context``
        automatically.

        Raises:
            ValueError: 400 response.
            PermissionError: 401 response.
            RuntimeError: Any other non-200 response.
        """
        url = f"{self._base_url}/classifications/v1"
        if request.context is None:
            request.context = {}
        request.context["agent_id"] = self._agent_id
        request.context["platform"] = self._platform
        payload = request.to_dict()
        logger.debug("Classification request payload: %s", payload)

        resp = self._session.post(
            url,
            json=payload,
            headers={
                "accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._token}",
            },
            timeout=self._timeout,
        )

        if resp.status_code == 200:
            data = resp.json()
            return ClassificationResponse.from_dict(data)

        content_type = resp.headers.get("Content-Type", "")
        is_json = "application/json" in content_type or content_type.endswith("+json")

        message = None
        if is_json:
            body = resp.json()
            message = body.get("message")

        msg = message or f"HTTP {resp.status_code} returned by AI Guard"
        if resp.status_code == 400:
            logger.error("Classification request error: %s", msg)
            raise ValueError(msg)
        elif resp.status_code == 401:
            logger.error("Classification request authorization error: %s", msg)
            raise PermissionError(msg)
        elif resp.status_code >= 500:
            logger.error("Classification server error: %s", msg)
            raise RuntimeError(msg)
        else:
            logger.error(
                "Unexpected response from AI Guard (%s): %s",
                resp.status_code,
                msg,
            )
            raise RuntimeError(msg)

    def metric(self, event: MetricsEvent) -> None:
        """Record a metrics event.

        Sends a ``POST /metric`` request.  The client's *agent_id* and
        *platform* are injected into ``event.attributes`` automatically.

        Raises:
            ValueError: 400 response.
            PermissionError: 401 response.
            RuntimeError: Any other non-200 response.
        """
        url = f"{self._base_url}/metric"
        event.attributes["agent_id"] = self._agent_id
        event.attributes["platform"] = self._platform
        payload = event.to_dict()
        logger.debug("Metrics event payload: %s", payload)

        resp = self._session.post(
            url,
            json=payload,
            headers={
                "accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._token}",
            },
            timeout=self._timeout,
        )

        if resp.status_code == 200:
            return

        content_type = resp.headers.get("Content-Type", "")
        is_json = "application/json" in content_type or content_type.endswith("+json")

        message = None
        if is_json:
            body = resp.json()
            message = body.get("message")

        msg = message or f"HTTP {resp.status_code} returned by AI Guard"
        if resp.status_code == 400:
            logger.error("Metrics request error: %s", msg)
            raise ValueError(msg)
        elif resp.status_code == 401:
            logger.error("Metrics request authorization error: %s", msg)
            raise PermissionError(msg)
        else:
            logger.error(
                "Unexpected response from AI Guard (%s): %s",
                resp.status_code,
                msg,
            )
            raise RuntimeError(msg)
